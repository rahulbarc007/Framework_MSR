"""Stage 4 - fuel-cycle controller on the burnup surrogate, with interchangeable optimisers.

Objective (same safety-first form as the EVOL studio): RMS tracking error of the
reactivity to its target over [eval_start, horizon], plus heavy penalties for leaving
the band at +- k sigma, exceeding the heavy-nuclide solubility limit or the fissile
inventory limit. The band is also limited by the temperature excursion the
feedback coefficient alpha_T would need (|alpha_T| x dT).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .. import plots
from ..metrics import PCM
from ..registry import create, spec_label
from .common import StageResult, load_model, print_table, write_json

W_BAND, W_HN, W_FIS = 20.0, 1e4, 10.0


def rho_pcm(k):
    k = np.clip(k, 0.5, 1.5)
    return (k - 1) / k * PCM


def decode(sc, space):
    free, fixed = [], {}
    ctl = sc.get("controls", {})
    for p in space.params:
        v = ctl.get(p.key, p.nominal)
        if v == "off":
            fixed[p.key] = p.high
        elif isinstance(v, (list, tuple)):
            free.append((p.key, max(v[0], p.low), min(v[1], p.high), p.log))
        else:
            fixed[p.key] = float(np.clip(v, p.low, p.high))
    return free, fixed


def params_from(u, free, fixed, keys):
    p = dict(fixed)
    for ui, (k, lo, hi, lg) in zip(u, free):
        p[k] = 10 ** (np.log10(lo) + ui * (np.log10(hi) - np.log10(lo))) if lg else lo + ui * (hi - lo)
    return np.array([p[k] for k in keys]), p


def evaluate(sc, P, x):
    T = np.asarray(P.times)
    tEnd = next((t for t in T if t >= sc["horizon_y"] - 1e-9), T[-1])
    win = (T >= sc.get("eval_start_y", 0) - 1e-9) & (T <= tEnd + 1e-9)
    pr = P.predict(x[None])
    k, ks = pr["keff"][0][0], np.nan_to_num(pr["keff"][1][0])
    r, s = rho_pcm(k), ks / np.clip(k, .5, 1.5) ** 2 * PCM
    lim = sc.get("limits", {})
    aT = abs(lim.get("alpha_T_pcm_per_K", 1e9))
    tgt, tol, kk = sc.get("rho_target_pcm", 0.0), sc.get("rho_tol_pcm", 400.0), sc.get("uncertainty_k", 1.0)
    lo, hi = max(tgt - tol, -aT * lim.get("dT_down_K", 1e9)), min(tgt + tol, aT * lim.get("dT_up_K", 1e9))
    rw, sw = r[win], s[win]
    a, b = np.maximum(0, rw + kk * sw - hi), np.maximum(0, lo - (rw - kk * sw))
    track = float(np.sqrt(np.mean((rw - tgt) ** 2)))
    pen = W_BAND * float(np.sqrt(np.mean(a ** 2 + b ** 2)))
    hn = float(pr["HN_molpct"][0][0][win].max()) if "HN_molpct" in pr else np.nan
    fis = float(pr["fissile_total_kg"][0][0][win].max()) if "fissile_total_kg" in pr else np.nan
    if np.isfinite(hn):
        pen += W_HN * max(0.0, hn - lim.get("hn_max_molpct", np.inf))
    if np.isfinite(fis):
        pen += W_FIS * max(0.0, fis - lim.get("fissile_max_kg", np.inf))
    viol = float(max(a.max(initial=0), b.max(initial=0)))
    mean_in = bool(np.all((rw >= lo) & (rw <= hi)))
    status = "feasible" if viol <= 1e-6 else (f"mean inside band, not certified at {kk} sigma" if mean_in else "infeasible")
    return dict(cost=track + pen, track=track, pen=pen, band=(lo, hi), rho=r, rho_sd=s, hn=hn, fis=fis, status=status,
                viol=viol, pred=pr, win=win)


def run(cfg):
    st = cfg["optimize"]
    out = cfg.out / "stage4_optimize"
    out.mkdir(parents=True, exist_ok=True)
    R = StageResult("optimize", "Stage 4 - Fuel-cycle optimiser")
    lab = st.get("surrogate", "deployed")
    P = load_model(cfg.out / "models" / f"burnup_{lab.replace(':', '_')}.pkl")
    keys = P.space.keys
    opt_specs = st.get("optimizers", ["differential_evolution"])
    olabels = [spec_label(s) for s in opt_specs]
    budget = st.get("budget", 4000)
    print(f"[optimize] surrogate {P.label}; optimisers {olabels}; budget {budget} evaluations each")
    rows, conv, results = [], {}, []
    for sc in st.get("scenarios", []):
        free, fixed = decode(sc, P.space)
        best = None
        conv[sc["name"]] = {}
        for spec, ol in zip(opt_specs, olabels):
            opt = create("optimizer", spec, seed=cfg["seed"], budget=budget)
            w = sc.get("weights", {}) or {}

            def cost(u, sc=sc, free=free, fixed=fixed, w=w):
                x, p = params_from(u, free, fixed, keys)
                c = evaluate(sc, P, x)["cost"]
                f = p.get("u233_feed_kgd", 0.0)   # optional reward for extracting / penalty for feeding 233U
                return c - w.get("u233_extraction", 0.0) * max(-f, 0) * 100 + w.get("u233_external", 0.0) * max(f, 0) * 100
            t0 = time.time()
            u, f, trace = opt.minimize(cost, len(free)) if free else (np.array([]), cost(np.array([])), np.array([0.0]))
            dt = time.time() - t0
            conv[sc["name"]][ol] = trace
            x, p = params_from(u, free, fixed, keys)
            ev = evaluate(sc, P, x)
            rows.append(dict(scenario=sc["name"], optimizer=ol, cost=ev["cost"], track_pcm=ev["track"], penalty=ev["pen"],
                             status=ev["status"], max_HN_molpct=ev["hn"], max_fissile_kg=ev["fis"], evals=len(trace), time_s=dt))
            if best is None or ev["cost"] < best[1]["cost"]:
                best = (ol, ev, p)
        results.append(dict(sc=sc, opt=best[0], ev=best[1], controls=best[2], free=[f[0] for f in free]))
        print(f"  {sc['name']:<34} best by {best[0]:<24} track {best[1]['track']:6.0f} pcm  [{best[1]['status']}]")
    tab = pd.DataFrame(rows)
    R.table("optimizer_comparison", tab, out)
    ctl = pd.DataFrame([dict(scenario=r["sc"]["name"], optimizer=r["opt"], **r["controls"]) for r in results])
    R.table("optimal_controls", ctl, out)
    ctl.assign(note="re-run these in OpenMC to verify").to_csv(out / "verify_in_openmc.csv", index=False)
    print_table(tab[["scenario", "optimizer", "cost", "track_pcm", "status", "time_s"]], "optimiser comparison")
    print_table(ctl, "optimal controls (also written to verify_in_openmc.csv)")
    _figures(R, out, results, conv, olabels, P)
    R["summary"] = dict(surrogate=P.label, optimizers=olabels, results=[dict(scenario=r["sc"]["name"], optimizer=r["opt"],
                        status=r["ev"]["status"], track_pcm=r["ev"]["track"], controls=r["controls"]) for r in results])
    R["verdict"] = " ".join(
        f"{r['sc'].get('title', r['sc']['name'])}: {r['ev']['status']} (RMS deviation {r['ev']['track']:.0f} pcm, best optimiser {r['opt']})."
        for r in results) + " Controls are exported to verify_in_openmc.csv for transport verification."
    write_json(out / "summary.json", R["summary"])
    return R


def _figures(R, out, results, conv, olabels, P):
    plt = plots.plt
    T = np.asarray(P.times)
    for r in results:
        sc, ev = r["sc"], r["ev"]
        has_inv = [o for o in ("HN_molpct", "fissile_total_kg") if o in ev["pred"]]
        fig, axes = plt.subplots(1, 1 + len(has_inv), figsize=(6.5 + 3.6 * len(has_inv), 3.9), squeeze=False)
        ax = axes[0, 0]
        lo, hi = ev["band"]
        ax.axhspan(lo, hi, color=plots.GRID, alpha=.8, lw=0, label="target band")
        ax.axvspan(sc.get("eval_start_y", 0), sc["horizon_y"], color=plots.PALETTE[0], alpha=.05, lw=0)
        c = plots.PALETTE[0]
        ax.plot(T, ev["rho"], color=c, marker="o", ms=3.5, label=f"predicted ({P.label})")
        kk = sc.get("uncertainty_k", 1.0)
        ax.fill_between(T, ev["rho"] - kk * ev["rho_sd"], ev["rho"] + kk * ev["rho_sd"], color=c, alpha=.2, lw=0, label=f"±{kk:g}σ")
        ax.axhline(sc.get("rho_target_pcm", 0), color=plots.INK2, lw=1, ls="--")
        ax.set_ylim(min(lo - 600, np.min(ev["rho"][ev["win"]]) - 300), max(hi + 600, np.max(ev["rho"][ev["win"]]) + 300))
        ax.set_xlabel("time (years)")
        ax.set_ylabel("reactivity (pcm)")
        ax.set_title(f"{sc.get('title', sc['name'])}\n{ev['status']}, RMS {ev['track']:.0f} pcm", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        lim = sc.get("limits", {})
        for a, o, lk, nm in zip(axes[0, 1:], has_inv, ["hn_max_molpct", "fissile_max_kg"], ["heavy-nuclide mol%", "fissile inventory (kg)"]):
            mu, sd = ev["pred"][o][0][0], ev["pred"][o][1][0]
            a.plot(T, mu, color=plots.PALETTE[1])
            if np.isfinite(sd).all():
                a.fill_between(T, mu - 1.96 * sd, mu + 1.96 * sd, color=plots.PALETTE[1], alpha=.18, lw=0)
            if lk in lim:
                a.axhline(lim[lk], color=plots.PALETTE[7], ls="--", lw=1.2)
                a.annotate("limit", (T[-1], lim[lk]), xytext=(0, 3), textcoords="offset points", ha="right", fontsize=8, color=plots.INK2)
            a.axvline(sc["horizon_y"], color=plots.MUTED, lw=.8, ls=":")
            a.set_title(nm, fontsize=10)
            a.set_xlabel("time (years)")
        R.fig(plots.save(fig, out / f"fig_{sc['name']}.png"), f"{sc['name']}: optimised reactivity trajectory and inventory limits.")

        # convergence
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        for ol in olabels:
            tr = conv[sc["name"]][ol]
            ax.plot(np.arange(1, len(tr) + 1), tr, color=plots.color(ol), ls=plots.dash(ol), label=ol)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("cost evaluations")
        ax.set_ylabel("best cost")
        ax.set_title(f"Optimiser convergence - {sc['name']}")
        ax.legend(fontsize=8)
        R.fig(plots.save(fig, out / f"fig_convergence_{sc['name']}.png"), "Best-so-far objective of each optimiser.")

    # controls table figure (normalised position in bounds)
    if results:
        fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(results) * 1 + 3))
        keys = P.space.keys
        for i, r in enumerate(results):
            u = P.space.to_unit(np.array([[r["controls"][k] for k in keys]]))[0]
            ax.scatter(u, np.arange(len(keys)) + (i - (len(results) - 1) / 2) * 0.18, s=40, color=plots.PALETTE[i % 8],
                       marker=plots.MARKERS[i % 8], label=r["sc"]["name"], zorder=3)
        ax.set_yticks(range(len(keys)), keys)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xlabel("position within the parameter range (0 = low, 1 = high; log-scaled where marked)")
        ax.set_title("Optimal controls by scenario")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        R.fig(plots.save(fig, out / "fig_controls.png"), "Where each scenario's optimum sits within the allowed range.")
