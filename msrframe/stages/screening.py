"""Stage 1 - parameter screening with interchangeable sensitivity methods."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .. import plots
from ..data import load_stage_data
from ..metrics import PCM, cv_predict, scalar_metrics
from ..registry import create, spec_label
from ..screening import ScreeningContext
from .common import StageResult, print_table, write_json


def run(cfg):
    st = cfg["screening"]
    out = cfg.out / "stage1_screening"
    out.mkdir(parents=True, exist_ok=True)
    R = StageResult("screening", "Stage 1 - Parameter screening")
    seed = cfg["seed"]

    df = load_stage_data(cfg, "screening")          # resolves inputs: auto -> catalogue keys in the data
    keys = list(st["inputs"])
    space = cfg.space("screening", keys, df)
    U = space.to_unit(df[keys].to_numpy(float))
    yname = st.get("output", "keff")
    y = df[yname].to_numpy(float) * (PCM if yname == "keff" else 1.0)
    unit = "pcm" if yname == "keff" else ""

    # ---- dummies: use the design's dummy columns if present, otherwise append random ones
    dcols = sorted(c for c in df.columns if c.startswith("dummy"))
    nd = st.get("n_dummies", 3)
    if dcols:
        D = df[dcols].to_numpy(float)
    else:
        D = np.random.default_rng(seed).random((len(df), nd))
        dcols = [f"dummy_{i + 1}" for i in range(nd)]
    Uall = np.hstack([U, D])
    allkeys = keys + dcols
    traj = df["morris_traj"].fillna(-1).to_numpy(int) if "morris_traj" in df and df["morris_traj"].notna().any() else None
    ctx = ScreeningContext(Uall, y, allkeys, traj, seed=seed)
    print(f"[screening] {len(df)} runs x {len(keys)} inputs (+{len(dcols)} dummies); "
          f"Morris trajectories: {'yes' if traj is not None else 'no'}")

    thr_cfg = st.get("threshold", {})
    fac, min_share = thr_cfg.get("dummy_factor", 2.0), thr_cfg.get("min_share", 0.005)
    methods = st.get("methods", ["morris", "sobol_pce", "active_subspace"])
    res, info, labels = {}, {}, []
    for spec in methods:
        lab = spec_label(spec)
        labels.append(lab)
        m = create("screening", spec, seed=seed)
        t0 = time.time()
        # classic Morris uses only trajectory rows; others use every run
        r = m.run(ctx)
        dt = time.time() - t0
        r["is_dummy"] = r["key"].isin(dcols)
        real = r[~r.is_dummy]
        dmax = float(r.loc[r.is_dummy, "score"].max()) if r.is_dummy.any() else 0.0
        tot = real["score"].sum() or 1.0
        r["share"] = r["score"] / tot
        thr = fac * dmax
        r["keep"] = (~r.is_dummy) & (r["score"] > thr) & (r["share"] >= min_share)
        r["rank"] = r["score"].rank(ascending=False, method="first").astype(int)
        kept = r[r.keep]
        sep = float(kept["score"].min() / dmax) if len(kept) and dmax > 0 else np.inf
        res[lab] = r.sort_values("rank").reset_index(drop=True)
        info[lab] = dict(units=m.units, time_s=dt, threshold=thr, dummy_max=dmax, n_kept=int(r.keep.sum()),
                         dummy_separation=sep, extra=getattr(m, "info", {}), mode=getattr(m, "mode", ""))
        print(f"  {lab:<28} kept {int(r.keep.sum()):>2}  dummy-separation {sep:8.2f}x  ({dt:5.1f} s)")

    # ---- per-parameter wide table
    wide = pd.DataFrame({"key": keys, "name": [space[k].name for k in keys], "class": [space[k].cls for k in keys]})
    for lab, r in res.items():
        rr = r.set_index("key")
        wide[f"share[{lab}]"] = rr.loc[keys, "share"].to_numpy()
        wide[f"rank[{lab}]"] = rr.loc[keys, "rank"].to_numpy()
        wide[f"keep[{lab}]"] = rr.loc[keys, "keep"].to_numpy()
    wide["votes"] = wide[[f"keep[{l}]" for l in labels]].sum(1)
    wide["mean_rank"] = wide[[f"rank[{l}]" for l in labels]].mean(1)
    wide = wide.sort_values(["votes", "mean_rank"], ascending=[False, True]).reset_index(drop=True)

    # ---- agreement between methods
    agree = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            A, B = set(res[a].loc[res[a].keep, "key"]), set(res[b].loc[res[b].keep, "key"])
            jac = len(A & B) / max(len(A | B), 1)
            ra = res[a].set_index("key").loc[keys, "score"]
            rb = res[b].set_index("key").loc[keys, "score"]
            top = wide["key"].head(max(len(A | B), 5))
            agree.append(dict(pair=f"{a} vs {b}", jaccard=jac, spearman_all=spearmanr(ra, rb)[0],
                              spearman_top=spearmanr(ra[top], rb[top])[0]))
    agree = pd.DataFrame(agree)

    # ---- consensus
    cons = st.get("consensus", {})
    min_votes = cons.get("min_votes", int(np.ceil(2 * len(labels) / 3)))
    final = list(wide.loc[wide.votes >= min_votes, "key"])
    forced = cons.get("forced", {}) or {}
    for k in forced:
        if k not in final and k in keys:
            final.append(k)

    # ---- retention curves: CV error using each method's top-m inputs
    val = st.get("validation", {})
    vs = val.get("surrogate", {"name": "gp", "restarts": 0})
    ms = [m for m in val.get("top_m", [1, 2, 3, 5, 7, 10, 16]) if m <= len(keys)]
    curves = []
    if val.get("enabled", True):
        for lab in labels:
            order = [k for k in res[lab]["key"] if k in keys]
            for mm in ms:
                cols = [keys.index(k) for k in order[:mm]]
                mu, sd, _ = cv_predict(vs, U[:, cols], y / PCM if unit else y, folds=val.get("folds", 5), seed=seed)
                curves.append(dict(method=lab, m=mm, rmse=scalar_metrics(y / PCM if unit else y, mu)["rmse"]))
        mu, sd, _ = cv_predict(vs, U[:, [keys.index(k) for k in final]], y / PCM, folds=val.get("folds", 5), seed=seed)
        final_cv = scalar_metrics(y / PCM, mu)
    else:
        final_cv = {}
    curves = pd.DataFrame(curves)

    # ---- outputs
    for lab, r in res.items():
        R.table(f"importance_{lab.replace(':', '_')}", r, out)
    R.table("consensus_table", wide, out)
    R.table("method_agreement", agree, out)
    if len(curves):
        R.table("retention_curves", curves, out)
    method_tab = pd.DataFrame([dict(method=l, units=info[l]["units"], kept=info[l]["n_kept"],
                                    threshold=info[l]["threshold"], dummy_max=info[l]["dummy_max"],
                                    dummy_separation=info[l]["dummy_separation"], time_s=info[l]["time_s"])
                               for l in labels])
    R.table("methods", method_tab, out)
    print_table(method_tab, "screening methods")
    print_table(wide[["key", "votes", "mean_rank"] + [f"share[{l}]" for l in labels]].head(15), "top parameters")
    print(f"  consensus (>= {min_votes} of {len(labels)} votes + forced): {final}")

    _figures(R, out, res, info, labels, wide, keys, dcols, curves, agree, final, unit)

    best = curves.groupby("method").rmse.min().idxmin() if len(curves) else labels[0]
    R["summary"] = dict(n_runs=len(df), n_inputs=len(keys), methods=labels, retained={l: list(res[l].loc[res[l].keep, "key"]) for l in labels},
                        consensus=final, forced=forced, min_votes=min_votes, final_cv_rmse_pcm=final_cv.get("rmse"),
                        final_cv_r2=final_cv.get("r2"), info=info)
    jac = agree["jaccard"].min() if len(agree) else 1.0
    R["verdict"] = (
        f"{len(labels)} screening methods ({', '.join(labels)}) were applied to {len(df)} runs over {len(keys)} inputs. "
        f"They retain {', '.join(str(info[l]['n_kept']) for l in labels)} inputs respectively "
        f"(lowest pairwise Jaccard agreement {jac:.2f}). Consensus ({min_votes}+ votes{' plus safety-forced inputs' if forced else ''}) "
        f"keeps {len(final)} inputs: {', '.join(final)}. "
        + (f"A GP on this set reaches {final_cv['rmse']:.0f} pcm CV RMSE (R2 = {final_cv['r2']:.4f}). " if final_cv else "")
        + (f"Lowest retention-curve error: {best}." if len(curves) else ""))
    write_json(out / "summary.json", {**R["summary"], "verdict": R["verdict"]})
    return R


def _figures(R, out, res, info, labels, wide, keys, dcols, curves, agree, final, unit):
    plt = plots.plt
    # 1. importance bars per method (top 20 + dummies)
    n = len(labels)
    fig, axes = plots.grid(n, ncol=min(n, 3), w=4.6, h=6.0)
    for ax, lab in zip(axes, labels):
        r = res[lab]
        top = pd.concat([r[~r.is_dummy].head(20), r[r.is_dummy]])
        y = np.arange(len(top))[::-1]
        cols = [plots.color(lab) if k else plots.MUTED for k in top.keep] 
        cols = [plots.GRID if d else c for d, c in zip(top.is_dummy, cols)]
        ax.barh(y, top.share, color=cols, edgecolor=[plots.INK2 if d else "white" for d in top.is_dummy], height=0.7)
        ax.set_yticks(y, top.key, fontsize=7.5)
        thr = info[lab]["threshold"] / (r.loc[~r.is_dummy, "score"].sum() or 1)
        ax.axvline(thr, color=plots.INK2, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlim(max(top.share[top.share > 0].min() / 2, 1e-6), 1)
        ax.set_title(f"{lab}\n{info[lab]['n_kept']} kept, dummy sep. {info[lab]['dummy_separation']:.1f}x", fontsize=10)
        ax.set_xlabel("share of total importance (log)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Importance by method - coloured = kept, grey = dropped, outlined = dummy; dashed = dummy-calibrated threshold",
                 fontsize=9.5, color=plots.INK2, x=0.01, ha="left")
    R.fig(plots.save(fig, out / "fig_importance_by_method.png"), "Normalised importance of the top 20 inputs for each screening method.")

    # 2. votes heat-map
    top = wide.head(max(len(final) + 6, 12))
    M = np.array([[float(k) for k in top[f"keep[{l}]"]] for l in labels])
    S = np.array([top[f"share[{l}]"].to_numpy() for l in labels])
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(top) + 2), 0.55 * n + 1.8))
    ax.imshow(np.log10(np.clip(S, 1e-4, 1)), cmap="Blues", aspect="auto", vmin=-4, vmax=0)
    for i in range(n):
        for j in range(len(top)):
            ax.text(j, i, "✓" if M[i, j] else "·", ha="center", va="center", fontsize=11,
                    color="white" if S[i, j] > 0.05 else plots.INK)
    ax.set_xticks(range(len(top)), top.key, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(n), labels)
    ax.grid(False)
    for j, k in enumerate(top.key):
        if k in final:
            ax.add_patch(plt.Rectangle((j - .5, -.5), 1, n, fill=False, ec=plots.INK, lw=1.5))
    ax.set_title("Retention votes (✓ kept; shade = importance share; boxed = final consensus set)")
    R.fig(plots.save(fig, out / "fig_votes.png"), "Which method keeps which input; boxed columns are the final reduced set.")

    # 3. retention curves
    if len(curves):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for lab in labels:
            c = curves[curves.method == lab]
            ax.plot(c.m, c.rmse, color=plots.color(lab), ls=plots.dash(lab), marker=plots.marker(lab), ms=6, label=lab)
        ax.set_xlabel("number of top-ranked inputs kept")
        ax.set_ylabel("surrogate CV RMSE (pcm)")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title("Retention curves: accuracy of a GP using each method's top-m inputs")
        R.fig(plots.save(fig, out / "fig_retention_curves.png"), "Lower is better: a good ranking reaches low error with few inputs.")

    # 4. agreement + cost
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    if len(agree):
        x = np.arange(len(agree))
        a1.bar(x - 0.2, agree.jaccard, 0.38, color=plots.PALETTE[0], label="Jaccard of kept sets")
        a1.bar(x + 0.2, agree.spearman_top, 0.38, color=plots.PALETTE[1], label="Spearman rank corr. (top)")
        a1.set_xticks(x, agree.pair, rotation=20, ha="right", fontsize=8)
        a1.set_ylim(min(0, agree.spearman_top.min() - .05), 1.05)
        a1.legend(fontsize=8, loc="lower right")
        a1.set_title("Agreement between methods")
        a1.grid(axis="x", visible=False)
    plots.bars(a2, labels, [info[l]["time_s"] for l in labels], unit="seconds")
    a2.set_title("Compute time per method")
    R.fig(plots.save(fig, out / "fig_agreement_cost.png"), "Pairwise agreement of retained sets and rankings; method run-time.")
