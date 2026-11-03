"""Active learning: tell the user which OpenMC runs will improve the surrogates most.

``suggest``  scores a large candidate pool with the deployed model(s) and picks a
             spread-out batch where the models are least certain:
               std        - predictive sd of the deployed model (GP, ensembles, forests)
               committee  - disagreement between all trained models
               hybrid     - product of the two normalised scores
             The batch is written to results/suggested_runs_<stage>.csv with empty
             output columns: run them in OpenMC, fill the columns, then ``add-data``.
``loop``     closes the loop automatically with a configured simulator (your OpenMC
             driver via ``command``, or the analytic stand-in) and charts the
             accuracy gained per batch against random sampling.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from . import plots
from .data import add_data, load_stage_data
from .design import maximin_select
from .metrics import PCM, scalar_metrics
from .registry import create, get, spec_label
from .simulators import make_simulator
from .stages.common import load_model


def _acquisition(stage, cfg, U_pool, X_pool, how):
    mdir = cfg.out / "models"
    dep = load_model(mdir / f"{stage}_deployed.pkl")
    others = [load_model(p) for p in sorted(glob.glob(str(mdir / f"{stage}_*.pkl"))) if not p.endswith("deployed.pkl")]

    def pred(P):
        if stage == "keff":
            return P.predict(X_pool)
        r = P.predict(X_pool, ["keff"])["keff"]
        return r[0].mean(1), np.sqrt(np.nanmean(r[1] ** 2, 1))

    mu, sd = pred(dep)
    s_std = sd if np.isfinite(sd).all() else None
    s_com = np.std(np.array([pred(P)[0] for P in others]), 0) if len(others) > 1 else None
    if how == "std" and s_std is not None:
        return s_std, "predictive sd"
    if how == "committee" and s_com is not None:
        return s_com, "committee spread"
    if s_std is not None and s_com is not None:
        n = lambda v: (v - v.min()) / (np.ptp(v) or 1)  # noqa: E731
        return (n(s_std) + 1e-3) * (n(s_com) + 1e-3), "hybrid (sd x committee)"
    return (s_std if s_std is not None else s_com), "available score"


def suggest(cfg, stage="keff", n=20, how="hybrid", pool=20000, out_file=None):
    st = cfg[stage]
    df = load_stage_data(cfg, stage)
    if stage == "keff":
        from .stages.keff import resolve_inputs
        keys = resolve_inputs(cfg, st)
    else:
        keys = list(st["inputs"])
    space = cfg.space(stage, keys, df)
    Upool = get("sampler", "lhs")(pool, len(keys), seed=cfg["seed"] + len(df))
    Xpool = space.from_unit(Upool)
    score, what = _acquisition(stage, cfg, Upool, Xpool, how)
    Uex = space.to_unit(df.drop_duplicates("case_id")[keys].to_numpy(float) if stage == "burnup" else df[keys].to_numpy(float))
    idx = maximin_select(Upool, score, n, existing=Uex)
    sug = pd.DataFrame(Xpool[idx], columns=keys)
    sug.insert(0, "case_id", [f"AL{len(df) + i:05d}" for i in range(n)])
    sug["acquisition"] = score[idx]
    for o in ([st.get("output", "keff")] + ([st["output_std"]] if st.get("output_std") else [])) if stage == "keff" else []:
        sug[o] = np.nan
    out_file = out_file or cfg.out / f"suggested_runs_{stage}.csv"
    sug.to_csv(out_file, index=False)
    print(f"[suggest] {n} new {stage} runs chosen by {what} from {pool} candidates -> {out_file}")
    print("  run them in OpenMC, fill the output column(s), then:")
    print(f"  python -m msrframe add-data --config <cfg> --stage {stage} --file {out_file}")

    # figure: where the suggestions are, on the two most influential inputs
    fig, (a1, a2) = plots.plt.subplots(1, 2, figsize=(11, 4.2))
    a1.hist(score, bins=60, color=plots.PALETTE[0], alpha=.85)
    a1.axvline(score[idx].min(), color=plots.INK2, ls="--", lw=1)
    a1.set_xlabel(what)
    a1.set_ylabel("candidates")
    a1.set_title("Acquisition score over the candidate pool")
    i, j = 0, 1 if len(keys) > 1 else 0
    a2.scatter(Uex[:, i], Uex[:, j], s=10, color=plots.MUTED, label="existing runs")
    a2.scatter(Upool[idx, i], Upool[idx, j], s=46, color=plots.PALETTE[1], edgecolor="white", label="suggested", zorder=3)
    a2.set_xlabel(f"{keys[i]} (scaled)")
    a2.set_ylabel(f"{keys[j]} (scaled)")
    a2.set_title("Suggested OpenMC runs vs existing data")
    a2.legend(fontsize=8)
    plots.save(fig, cfg.out / f"fig_suggested_{stage}.png")
    return sug


def loop(cfg):
    """Closed active-learning loop for the k_eff stage, with a random-sampling baseline."""
    st, L = cfg["keff"], cfg["active_learning"]
    from .stages.keff import resolve_inputs
    keys = resolve_inputs(cfg, st)
    df = load_stage_data(cfg, "keff")
    space = cfg.space("keff", keys, df)
    sim = make_simulator(cfg, L.get("simulator"), keys)
    seed = cfg["seed"]
    # independent test set from the simulator (noise-free where possible) to measure true accuracy
    Ute = get("sampler", "lhs")(L.get("n_test", 400), len(keys), seed=seed + 999)
    Xte = space.from_unit(Ute)
    yte = sim.keff(Xte, noisy=False) if hasattr(sim, "keff") else sim.run(pd.DataFrame(Xte, columns=keys))["keff"].to_numpy()
    specs = L.get("models", ["gp"])
    labels = [spec_label(s) for s in specs]
    acq_model = L.get("acquisition_model", specs[0])
    batch, iters = L.get("batch", 10), L.get("iterations", 5)
    U0, y0 = space.to_unit(df[keys].to_numpy(float)), df["keff"].to_numpy(float)
    arms = {"active": (U0.copy(), y0.copy())}
    if L.get("compare_random", True):
        arms["random"] = (U0.copy(), y0.copy())
    rows = []
    rng = np.random.default_rng(seed)
    for it in range(iters + 1):
        for arm, (U, y) in arms.items():
            for spec, lab in zip(specs, labels):
                m = create("surrogate", spec, seed=seed).fit(U, y)
                rows.append(dict(iteration=it, arm=arm, model=lab, n=len(y), rmse_pcm=scalar_metrics(yte, m.predict(space.to_unit(Xte)))["rmse"]))
        r_act = [r for r in rows if r["iteration"] == it and r["arm"] == "active"]
        print(f"  iteration {it}: n = {len(arms['active'][1])}  " + "  ".join(f"{r['model']} {r['rmse_pcm']:.0f} pcm" for r in r_act))
        if it == iters:
            break
        # ---- active arm: pick a batch where the acquisition model is least certain
        U, y = arms["active"]
        m = create("surrogate", acq_model, seed=seed).fit(U, y)
        pool = rng.random((L.get("pool", 5000), len(keys)))
        _, sd = m.predict(pool, return_std=True)
        if not np.isfinite(sd).all():
            raise ValueError("acquisition_model must provide an uncertainty (gp, dnn, random_forest, ...)")
        idx = maximin_select(pool, sd, batch, existing=U)
        new = pd.DataFrame(space.from_unit(pool[idx]), columns=keys)
        new.insert(0, "case_id", [f"AL{it:02d}_{i:03d}" for i in range(batch)])
        res = sim.run(new)
        if L.get("persist", True):
            tmp = cfg.out / f"al_batch_{it:02d}.csv"
            res.to_csv(tmp, index=False)
            add_data(cfg, "keff", tmp, tag=f"active{it:02d}")
        arms["active"] = (np.vstack([U, pool[idx]]), np.r_[y, res["keff"].to_numpy(float)])
        if "random" in arms:
            U, y = arms["random"]
            ur = rng.random((batch, len(keys)))
            rr = sim.run(pd.DataFrame(space.from_unit(ur), columns=keys))
            arms["random"] = (np.vstack([U, ur]), np.r_[y, rr["keff"].to_numpy(float)])
    hist = pd.DataFrame(rows)
    out = cfg.out / "active_learning"
    out.mkdir(parents=True, exist_ok=True)
    hist.to_csv(out / "active_learning_history.csv", index=False)
    fig, ax = plots.plt.subplots(figsize=(7, 4.2))
    for lab in labels:
        for arm, ls in (("active", "-"), ("random", ":")):
            c = hist[(hist.model == lab) & (hist.arm == arm)]
            if len(c):
                ax.plot(c.n, c.rmse_pcm, color=plots.color(lab), ls=ls, marker=plots.marker(lab), ms=5, label=f"{lab} - {arm}")
    ax.set_yscale("log")
    ax.set_xlabel("training runs")
    ax.set_ylabel("RMSE on independent test set (pcm, log)")
    ax.set_title("Active learning: accuracy gained per batch of new runs")
    ax.legend(fontsize=8, ncol=2)
    p = plots.save(fig, out / "fig_active_learning.png")
    print(f"  active-learning chart -> {p}")
    return hist, p
