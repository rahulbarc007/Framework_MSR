"""Stage 3 - burnup (depletion-trajectory) surrogate with interchangeable models."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .. import plots
from ..data import append_history, histories, load_stage_data
from ..metrics import PCM
from ..registry import create, spec_label
from .common import StageResult, print_table, save_model, write_json


class BurnupPredictor:
    def __init__(self, label, model, space, times, outputs):
        self.label, self.model, self.space, self.times, self.outputs = label, model, space, times, outputs

    def predict(self, X, outputs=None):
        X = X[self.space.keys].to_numpy(float) if isinstance(X, pd.DataFrame) else np.atleast_2d(X)
        return self.model.predict(self.space.to_unit(X), outputs)


def run(cfg, quiet=False, tag=None):
    st = cfg["burnup"]
    out = cfg.out / "stage3_burnup"
    out.mkdir(parents=True, exist_ok=True)
    R = StageResult("burnup", "Stage 3 - Burnup predictor")
    seed = cfg["seed"]
    keys, outputs = list(st["inputs"]), list(st["outputs"])
    df = load_stage_data(cfg, "burnup")
    space = cfg.space("burnup", keys, df)
    X, Y, times, ids = histories(df, keys, outputs, st.get("times"))
    U = space.to_unit(X)
    specs = st.get("models", [{"name": "pca", "base": "gp"}])
    labels = [spec_label(s) for s in specs]
    folds = min(st.get("validation", {}).get("folds", 5), len(U))
    print(f"[burnup] {len(U)} histories x {len(times)} time points, {len(keys)} inputs, outputs {outputs}; models {labels}")

    oof = {l: {o: (np.zeros_like(Y[o]), np.zeros_like(Y[o])) for o in outputs} for l in labels}
    for spec, lab in zip(specs, labels):
        for tr, te in KFold(folds, shuffle=True, random_state=seed).split(U):
            m = create("trajectory", spec, seed=seed).fit(U[tr], {o: Y[o][tr] for o in outputs}, times)
            P = m.predict(U[te], noise=True)   # predictive sd for coverage
            for o in outputs:
                oof[lab][o][0][te], oof[lab][o][1][te] = P[o]

    # out-of-fold trajectories: each history predicted by models that never saw it (used by the dashboard)
    write_json(out / "cv_predictions.json", {"ids": [str(i) for i in ids], "times": list(times),
                                            "pred": {lab: {o: oof[lab][o][0] for o in outputs} for lab in labels}})

    rows, rvt = [], []
    for lab in labels:
        r = {"model": lab}
        for o in outputs:
            mu, sd = oof[lab][o]
            e = mu - Y[o]
            if o == "keff":
                ep = e * PCM
                r["keff_rmse_pcm"] = float(np.sqrt(np.mean(ep ** 2)))
                r["keff_max_pcm"] = float(np.max(np.abs(ep)))
                if np.isfinite(sd).all():
                    r["keff_coverage95"] = float(np.mean(np.abs(e) <= 1.96 * np.maximum(sd, 1e-12)))
                for t, v in zip(times, np.sqrt(np.mean(ep ** 2, 0))):
                    rvt.append(dict(model=lab, time_y=t, rmse_pcm=v))
            else:
                scale = np.maximum(np.abs(Y[o]), 1e-9)
                ok = np.abs(Y[o]) > 1e-6 * np.abs(Y[o]).max()
                r[f"{o}_rel_rmse_%"] = float(100 * np.sqrt(np.mean((e[ok] / scale[ok]) ** 2)))
        rows.append(r)
    metrics = pd.DataFrame(rows)
    sort = "keff_rmse_pcm" if "keff_rmse_pcm" in metrics else metrics.columns[1]
    metrics = metrics.sort_values(sort).reset_index(drop=True)
    ranking = list(metrics.model)
    rvt = pd.DataFrame(rvt)

    deployed = {}
    for spec, lab in zip(specs, labels):
        m = create("trajectory", spec, seed=seed).fit(U, Y, times)
        deployed[lab] = BurnupPredictor(lab, m, space, times, outputs)
        save_model(cfg.out / "models" / f"burnup_{lab.replace(':', '_')}.pkl", deployed[lab])
    best = ranking[0] if st.get("deploy", "best") == "best" else st["deploy"]
    save_model(cfg.out / "models" / "burnup_deployed.pkl", deployed[best])
    hist = append_history(cfg, "burnup", [dict(n_data=len(U), model=r["model"], keff_rmse_pcm=r.get("keff_rmse_pcm"), tag=tag or "")
                                          for r in rows])

    R.table("cv_metrics", metrics, out)
    if len(rvt):
        R.table("keff_rmse_vs_time", rvt, out)
    if not quiet:
        print_table(metrics, f"{folds}-fold cross-validation over histories")
        print(f"  deployed model: {best}")
    _figures(R, out, labels, ranking, Y, oof, times, outputs, metrics, rvt, deployed, space, ids, hist, cfg)
    R["summary"] = dict(n_histories=len(U), times=list(times), inputs=keys, outputs=outputs, ranking=ranking, deployed=best,
                        metrics=metrics.to_dict("records"))
    b = metrics.iloc[0]
    R["verdict"] = (f"Best burnup surrogate over {len(U)} histories: {b.model}"
                    + (f" with {b.keff_rmse_pcm:.0f} pcm k_eff RMSE in {folds}-fold CV" if "keff_rmse_pcm" in b else "")
                    + "; " + ", ".join(f"{o} {b[f'{o}_rel_rmse_%']:.2f}%" for o in outputs if f"{o}_rel_rmse_%" in b)
                    + f". Ranking: {' > '.join(ranking)}. Add depletion histories to data/user/ to tighten it.")
    write_json(out / "summary.json", {**R["summary"], "verdict": R["verdict"]})
    return R


def _figures(R, out, labels, ranking, Y, oof, times, outputs, metrics, rvt, deployed, space, ids, hist, cfg):
    plt = plots.plt
    if len(rvt):
        fig, ax = plt.subplots(figsize=(7, 4))
        for lab in labels:
            c = rvt[rvt.model == lab]
            ax.plot(c.time_y, c.rmse_pcm, color=plots.color(lab), ls=plots.dash(lab), marker=plots.marker(lab), ms=4, label=lab)
        ax.set_xlabel("time (years)")
        ax.set_ylabel("k_eff CV RMSE (pcm)")
        ax.set_title("Burnup surrogate k_eff error along the 30-year history")
        ax.legend()
        R.fig(plots.save(fig, out / "fig_keff_rmse_vs_time.png"), "Cross-validated k_eff error at each depletion time point.")

    rel = [c for c in metrics.columns if c.endswith("_rel_rmse_%")]
    if rel:
        fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(rel) + 2), 4))
        w = 0.8 / len(labels)
        x = np.arange(len(rel))
        for i, lab in enumerate(labels):
            v = metrics.set_index("model").loc[lab, rel].to_numpy(float)
            ax.bar(x + (i - (len(labels) - 1) / 2) * w, v, w * 0.92, color=plots.color(lab), label=lab)
        ax.set_xticks(x, [c.replace("_rel_rmse_%", "") for c in rel], rotation=20, ha="right")
        ax.set_yscale("log")
        ax.set_ylabel("relative RMSE (%)")
        ax.legend(ncol=min(4, len(labels)))
        ax.set_title("Inventory prediction error by output")
        ax.grid(axis="x", visible=False)
        R.fig(plots.save(fig, out / "fig_inventory_errors.png"), "Cross-validated relative RMSE of each inventory output.")

    # example histories: worst, median, best for the best model (k_eff)
    if "keff" in outputs:
        b = ranking[0]
        err = np.sqrt(np.mean((oof[b]["keff"][0] - Y["keff"]) ** 2, 1))
        pick = [int(np.argmin(err)), int(np.argsort(err)[len(err) // 2]), int(np.argmax(err))]
        fig, axes = plots.grid(3, ncol=3, w=4.2, h=3.3)
        for ax, i, nm in zip(axes, pick, ["best", "median", "worst"]):
            ax.plot(times, Y["keff"][i], color=plots.INK, lw=2.4, label="OpenMC")
            for lab in ranking[:3]:
                mu, sd = oof[lab]["keff"]
                ax.plot(times, mu[i], color=plots.color(lab), ls=plots.dash(lab), lw=1.6, label=lab)
                if lab == b and np.isfinite(sd[i]).all():
                    ax.fill_between(times, mu[i] - 1.96 * sd[i], mu[i] + 1.96 * sd[i], color=plots.color(lab), alpha=.15, lw=0)
            ax.set_title(f"{nm} case {ids[i]} ({err[i] * PCM:.0f} pcm)", fontsize=10)
            ax.set_xlabel("time (years)")
            ax.set_ylabel("k_eff")
        axes[0].legend(fontsize=7.5)
        R.fig(plots.save(fig, out / "fig_example_histories.png"), "Held-out histories: OpenMC vs cross-validated predictions.")

    # nominal-plant prediction of every output with the deployed model
    P = deployed[ranking[0]]
    pr = P.predict(space.nominal[None])
    fig, axes = plots.grid(len(outputs), ncol=4, w=3.5, h=2.8)
    lab3 = cfg["burnup"].get("output_labels", {})
    for ax, o in zip(axes, outputs):
        mu, sd = pr[o][0][0], pr[o][1][0]
        ax.plot(times, mu, color=plots.color(ranking[0]))
        if np.isfinite(sd).all():
            ax.fill_between(times, mu - 1.96 * sd, mu + 1.96 * sd, color=plots.color(ranking[0]), alpha=.18, lw=0)
        ax.set_title(lab3.get(o, o), fontsize=9.5)
        ax.set_xlabel("years", fontsize=8.5)
    fig.suptitle(f"Nominal-plant trajectories predicted by {ranking[0]} (band = 95%)", fontsize=10, color=plots.INK2, x=.01, ha="left")
    R.fig(plots.save(fig, out / "fig_nominal_trajectories.png"), "All outputs at nominal operating parameters.")

    h = hist[hist.model.isin(labels)]
    if h.n_data.nunique() > 1:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for lab in labels:
            c = h[h.model == lab].groupby("n_data").keff_rmse_pcm.last().reset_index()
            ax.plot(c.n_data, c.keff_rmse_pcm, color=plots.color(lab), marker=plots.marker(lab), label=lab)
        ax.set_xlabel("histories in database")
        ax.set_ylabel("k_eff CV RMSE (pcm)")
        ax.legend()
        ax.set_title("Burnup accuracy history over training sessions")
        R.fig(plots.save(fig, out / "fig_accuracy_history.png"), "From results/history/burnup_history.csv.")
