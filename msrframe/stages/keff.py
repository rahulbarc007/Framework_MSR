"""Stage 2 - static k_eff surrogate with interchangeable regressors."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .. import plots
from ..data import append_history, load_stage_data
from ..metrics import PCM, cv_predict, scalar_metrics
from ..registry import create, spec_label
from .common import StageResult, print_table, save_model, write_json


class KeffPredictor:
    """Deployed predictor in physical units: ``predict(df_or_array) -> (keff, sd)``."""

    def __init__(self, label, model, space):
        self.label, self.model, self.space = label, model, space

    def predict(self, X, return_std=True):
        X = X[self.space.keys].to_numpy(float) if isinstance(X, pd.DataFrame) else np.atleast_2d(X)
        return self.model.predict(self.space.to_unit(X), return_std=return_std)


def resolve_inputs(cfg, st):
    inp = st.get("inputs", "from_screening")
    if inp == "from_screening":
        import json
        p = cfg.out / "stage1_screening" / "summary.json"
        if not p.exists():
            raise FileNotFoundError("keff.inputs = from_screening but Stage 1 has not been run; run it or list inputs")
        st["inputs"] = json.loads(p.read_text())["consensus"]
    return list(st["inputs"])


def run(cfg, models_override=None, quiet=False, tag=None):
    st = cfg["keff"]
    out = cfg.out / "stage2_keff"
    out.mkdir(parents=True, exist_ok=True)
    R = StageResult("keff", "Stage 2 - k_eff predictor")
    seed = cfg["seed"]
    keys = resolve_inputs(cfg, st)
    df = load_stage_data(cfg, "keff")
    space = cfg.space("keff", keys, df)
    U = space.to_unit(df[keys].to_numpy(float))
    y = df[st.get("output", "keff")].to_numpy(float)
    ystd = df[st["output_std"]].to_numpy(float) if st.get("output_std") in df else None
    specs = models_override or st.get("models", ["gp", "dnn", "gbt"])
    labels = [spec_label(s) for s in specs]
    val = st.get("validation", {})
    folds = val.get("folds", 5)
    print(f"[keff] {len(df)} OpenMC runs x {len(keys)} inputs; models: {labels}; {folds}-fold CV")

    # ---- K-fold cross-validation on all data (robust as data are added)
    cv, oof = {}, {}
    for spec, lab in zip(specs, labels):
        mu, sd, ft = cv_predict(spec, U, y, folds=folds, seed=seed)
        m = scalar_metrics(y, mu, sd if np.isfinite(sd).all() else None, ystd)
        m["fit_time_s"] = ft
        cv[lab], oof[lab] = m, (mu, sd)
        if not quiet:
            print(f"  {lab:<22} CV RMSE {m['rmse']:7.1f} pcm   R2 {m['r2']:.5f}   "
                  f"coverage95 {m.get('coverage95', float('nan')):.2f}")
    metrics = pd.DataFrame([{"model": l, **cv[l]} for l in labels]).sort_values("rmse").reset_index(drop=True)
    ranking = list(metrics.model)

    # ---- fixed hold-out (split column), as in the pilot study
    hold = None
    if "split" in df and (df["split"] == "test").any() and (df["split"] == "train").any():
        tr, te = (df["split"] != "test").to_numpy(), (df["split"] == "test").to_numpy()
        rows = []
        for spec, lab in zip(specs, labels):
            m = create("surrogate", spec, seed=seed).fit(U[tr], y[tr])
            mu, sd = m.predict(U[te], return_std=True)
            sd = np.sqrt(sd ** 2 + m.noise_sd ** 2)
            rows.append({"model": lab, **scalar_metrics(y[te], mu, sd if np.isfinite(sd).all() else None,
                                                         None if ystd is None else ystd[te])})
        hold = pd.DataFrame(rows)

    # ---- learning curves on a fixed 20 % hold-out
    lc_rows = []
    fr = st.get("learning_curve", {}).get("fractions", [0.2, 0.4, 0.6, 0.8, 1.0])
    if fr:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        nte = max(int(0.2 * len(y)), 5)
        te, trall = perm[:nte], perm[nte:]
        for spec, lab in zip(specs, labels):
            for f in fr:
                n = max(int(f * len(trall)), len(keys) + 3)
                m = create("surrogate", spec, seed=seed).fit(U[trall[:n]], y[trall[:n]])
                lc_rows.append(dict(model=lab, n_train=n, rmse=scalar_metrics(y[te], m.predict(U[te]))["rmse"]))
    lc = pd.DataFrame(lc_rows)

    # ---- deploy: refit every model on all data, save
    deployed = {}
    for spec, lab in zip(specs, labels):
        deployed[lab] = KeffPredictor(lab, create("surrogate", spec, seed=seed).fit(U, y), space)
        save_model(cfg.out / "models" / f"keff_{lab.replace(':', '_')}.pkl", deployed[lab])
    choice = st.get("deploy", "best")
    best = ranking[0] if choice == "best" else choice
    save_model(cfg.out / "models" / "keff_deployed.pkl", deployed[best])

    # ---- derived quantities at nominal with each model
    der = []
    dcfg = st.get("derived", {})
    x0 = space.nominal
    for lab, P in deployed.items():
        row = {"model": lab, "keff_nominal": float(P.predict(x0[None], False)[0])}
        crit = dcfg.get("critical")
        if crit and crit in keys:
            j = keys.index(crit)
            p = space[crit]

            def g(v):
                x = x0.copy()
                x[j] = v
                return P.predict(x[None], False)[0] - 1.0
            try:
                row[f"critical_{crit}"] = brentq(g, p.low, p.high) if g(p.low) * g(p.high) < 0 else np.nan
            except Exception:
                row[f"critical_{crit}"] = np.nan
        for c in dcfg.get("coefficients", []):
            if c in keys:
                j, p = keys.index(c), space[c]
                h = 0.05 * (p.high - p.low)
                xa, xb = x0.copy(), x0.copy()
                xa[j] += h
                xb[j] -= h
                ka, kb = P.predict(np.vstack([xa, xb]), False)
                rho = lambda k: (k - 1) / k * PCM  # noqa: E731
                row[f"drho/d{c} (pcm/{p.unit or 'unit'})"] = (rho(ka) - rho(kb)) / (2 * h)
        der.append(row)
    der = pd.DataFrame(der)

    # ---- history: accuracy versus amount of data over training sessions
    hist = append_history(cfg, "keff", [dict(n_data=len(df), model=l, cv_rmse_pcm=cv[l]["rmse"], cv_r2=cv[l]["r2"],
                                             tag=tag or "") for l in labels])

    R.table("cv_metrics", metrics, out)
    if hold is not None:
        R.table("holdout_metrics", hold, out)
    if len(lc):
        R.table("learning_curves", lc, out)
    R.table("derived_at_nominal", der, out)
    pd.DataFrame({"case_id": df.get("case_id", pd.Series(range(len(df)))), "keff": y,
                  **{f"cv_pred[{l}]": oof[l][0] for l in labels}}).to_csv(out / "cv_predictions.csv", index=False)
    if not quiet:
        print_table(metrics[["model", "rmse", "mae", "max_abs", "r2"] + [c for c in ("rmse_noise_corrected", "coverage95") if c in metrics]],
                    "cross-validated accuracy (pcm)")
        if hold is not None:
            print_table(hold[["model", "rmse", "r2"] + [c for c in ("coverage95",) if c in hold]], "fixed hold-out (split=test)")
        print_table(der, "derived quantities at nominal")
        print(f"  deployed model: {best}")

    _figures(R, out, labels, y, ystd, oof, metrics, lc, hist, deployed, space, keys, ranking, df)
    R["summary"] = dict(n_runs=len(df), inputs=keys, ranking=ranking, deployed=best,
                        cv=cv, holdout=None if hold is None else hold.to_dict("records"), derived=der.to_dict("records"))
    b, w = metrics.iloc[0], metrics.iloc[-1]
    noise = b.get("mc_noise_rms", np.nan)
    R["verdict"] = (
        f"Best k_eff surrogate on {len(df)} runs: {b.model} with {b.rmse:.0f} pcm cross-validated RMSE (R2 = {b.r2:.5f}"
        + (f", 95% interval coverage {b.coverage95:.0%}" if "coverage95" in b and np.isfinite(b.coverage95) else "") + "). "
        + (f"The Monte Carlo noise of the targets is {noise:.0f} pcm rms, so part of every error is OpenMC noise. " if np.isfinite(noise) else "")
        + f"Worst: {w.model} ({w.rmse:.0f} pcm). "
        + (f"Learning curves show the error of {b.model} falling from {lc[lc.model == b.model].rmse.iloc[0]:.0f} to "
           f"{lc[lc.model == b.model].rmse.iloc[-1]:.0f} pcm as training data grow - adding OpenMC runs keeps improving it." if len(lc) else ""))
    write_json(out / "summary.json", R["summary"])
    return R


def _figures(R, out, labels, y, ystd, oof, metrics, lc, hist, deployed, space, keys, ranking, df):
    plt = plots.plt
    # parity plots
    fig, axes = plots.grid(len(labels), ncol=min(4, len(labels)), w=3.6, h=3.5)
    lo, hi = y.min(), y.max()
    for ax, lab in zip(axes, labels):
        mu, sd = oof[lab]
        if np.isfinite(sd).all():
            ax.errorbar(y, mu, yerr=1.96 * sd, fmt="none", ecolor=plots.GRID, lw=1, zorder=1)
        ax.scatter(y, mu, s=14, color=plots.color(lab), edgecolor="white", linewidth=0.5, zorder=2)
        ax.plot([lo, hi], [lo, hi], color=plots.INK2, lw=1)
        r = metrics.set_index("model").loc[lab]
        ax.set_title(f"{lab}: {r.rmse:.0f} pcm", fontsize=10)
        ax.set_xlabel("OpenMC k_eff")
        ax.set_ylabel("CV prediction")
    R.fig(plots.save(fig, out / "fig_parity.png"), "Out-of-fold predictions vs OpenMC (bars: 95% predictive interval where the model has one).")

    # RMSE bars
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(labels) + 2), 3.8))
    nf = metrics.get("mc_noise_rms")
    plots.bars(ax, list(metrics.model), list(metrics.rmse), ref=None if nf is None else float(nf.iloc[0]),
               ref_label="OpenMC noise floor", unit="CV RMSE (pcm)")
    ax.set_title("k_eff surrogate accuracy (5-fold CV) - lower is better")
    R.fig(plots.save(fig, out / "fig_rmse.png"), "Cross-validated RMSE per model; dashed line = Monte Carlo noise of the targets.")

    # learning curves
    if len(lc):
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for lab in labels:
            c = lc[lc.model == lab]
            ax.plot(c.n_train, c.rmse, color=plots.color(lab), ls=plots.dash(lab), marker=plots.marker(lab), ms=6, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training runs")
        ax.set_ylabel("hold-out RMSE (pcm)")
        ax.legend(ncol=2)
        ax.set_title("Learning curves - more OpenMC data, lower error")
        R.fig(plots.save(fig, out / "fig_learning_curves.png"), "Hold-out error against training-set size; slope shows the value of more data.")

    # accuracy history across training sessions
    h = hist[hist.model.isin(labels)]
    if h.n_data.nunique() > 1:
        fig, ax = plt.subplots(figsize=(6.4, 4))
        for lab in labels:
            c = h[h.model == lab].groupby("n_data").cv_rmse_pcm.last().reset_index()
            ax.plot(c.n_data, c.cv_rmse_pcm, color=plots.color(lab), ls=plots.dash(lab), marker=plots.marker(lab), ms=6, label=lab)
        ax.set_xlabel("runs in the database at training time")
        ax.set_ylabel("CV RMSE (pcm)")
        ax.legend(ncol=2)
        ax.set_title("Accuracy history over training sessions")
        R.fig(plots.save(fig, out / "fig_accuracy_history.png"), "Every 'train' session appends to results/history/keff_history.csv.")

    # 1-D responses of the deployed model
    P = deployed[ranking[0]]
    fig, axes = plots.grid(len(keys), ncol=3, w=3.8, h=2.9)
    x0 = space.nominal
    for ax, k in zip(axes, keys):
        p = space[k]
        v = np.geomspace(p.low, p.high, 60) if p.log else np.linspace(p.low, p.high, 60)
        X = np.tile(x0, (60, 1))
        X[:, keys.index(k)] = v
        for lab in ranking[:3]:
            mu, sd = deployed[lab].predict(X)
            ax.plot(v, mu, color=plots.color(lab), ls=plots.dash(lab), lw=1.6, label=lab)
            if lab == ranking[0] and np.isfinite(sd).all():
                ax.fill_between(v, mu - 1.96 * sd, mu + 1.96 * sd, color=plots.color(lab), alpha=0.15, lw=0)
        ax.axhline(1.0, color=plots.INK2, lw=0.8, ls=":")
        ax.set_xlabel(f"{k} ({p.unit})" if p.unit else k, fontsize=9)
        ax.set_ylabel("k_eff")
        if p.log:
            ax.set_xscale("log")
    axes[0].legend(fontsize=7.5)
    fig.suptitle("k_eff response to each input (others at nominal); band = 95% interval of the best model",
                 fontsize=9.5, color=plots.INK2, x=0.01, ha="left")
    R.fig(plots.save(fig, out / "fig_response_curves.png"), "One-at-a-time response curves of the three best models.")
