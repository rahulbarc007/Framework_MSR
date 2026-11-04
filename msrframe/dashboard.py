"""Build the interactive dashboard: one self-contained HTML file, like the EVOL MSFR studio.

The page embeds the trained models (exported by :mod:`msrframe.export`) and runs them in the
browser, so it needs no Python, server or internet connection:

* Stage 1 - screening results of whatever methods were configured, parameter catalogue
* Stage 2 - k_eff sliders with live predictions of every included model, sweeps,
  criticality search, local sensitivities, parity plots, learning curves
* Stage 3 - burnup trajectories for any operating point, compared with OpenMC histories
* Stage 4 - the fuel-cycle controller (differential evolution in the browser)
* Data    - training files, accuracy history as data were added, suggested next runs

It is rebuilt from the current models and data by every ``msrframe run`` and by
``python -m msrframe dashboard``. Which models are embedded is set by
``dashboard.keff_models`` / ``dashboard.burnup_models`` (``top3`` | ``all`` | ``topN`` | list).
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _files, histories, load_stage_data
from .export import export_scalar, export_space, export_trajectory
from .stages.common import load_model

ASSETS = Path(__file__).parent / "dashboard_assets"


# --------------------------------------------------------------------------- helpers
def _clean(o):
    """JSON-safe copy: numpy -> python, NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def _records(path):
    p = Path(path)
    return pd.read_csv(p).to_dict("records") if p.exists() else []


def _select(labels, spec):
    """'top3' | 'all' | 'topN' | 'a,b' | [a, b] -> chosen labels (in ranking order)."""
    if spec in (None, "", "top3"):
        return labels[:3]
    if spec == "all":
        return list(labels)
    if isinstance(spec, str) and spec.startswith("top") and spec[3:].isdigit():
        return labels[: int(spec[3:])]
    if isinstance(spec, str):
        spec = [s.strip() for s in spec.split(",") if s.strip()]
    missing = [s for s in spec if s not in labels]
    if missing:
        raise ValueError(f"dashboard: models {missing} were not trained; available: {labels}")
    return list(spec)


def _unit_of(o):
    if o == "keff":
        return ""
    if o.endswith("_kg"):
        return "kg"
    if "molpct" in o:
        return "mol%"
    if o.endswith("_pcm"):
        return "pcm"
    return ""


def _dev_text(info, stage):
    e = info.get("max_err")
    if e is None:
        return ""
    if stage == "keff":
        return f"{e * 1e5:.2g} pcm"
    return f"{info.get('rel_err', e):.1e} of spread" if "rel_err" in info else f"{e:.2g}"


# --------------------------------------------------------------------------- stage sections
def _screening(cfg):
    d = cfg.out / "stage1_screening"
    if not (d / "summary.json").exists():
        return None
    summ = json.loads((d / "summary.json").read_text())
    methods = summ["methods"]
    imp = {m: pd.read_csv(d / f"importance_{m.replace(':', '_')}.csv") for m in methods}
    cons = pd.read_csv(d / "consensus_table.csv")
    table = [dict(key=r["key"], name=r.get("name", r["key"]), cls=r.get("class", ""), votes=int(r["votes"]), mean_rank=r["mean_rank"],
                  share={m: r[f"share[{m}]"] for m in methods}, rank={m: int(r[f"rank[{m}]"]) for m in methods},
                  keep={m: bool(r[f"keep[{m}]"]) for m in methods}) for r in cons.to_dict("records")]
    dkeys = list(imp[methods[0]].loc[imp[methods[0]].is_dummy, "key"])
    dummies = []
    for k in dkeys:
        rr = {m: imp[m].set_index("key").loc[k] for m in methods}
        dummies.append(dict(key=k, share={m: rr[m]["share"] for m in methods}, rank={m: int(rr[m]["rank"]) for m in methods}))
    mt = pd.read_csv(d / "methods.csv")
    thr = {}
    for m in methods:
        real = imp[m].loc[~imp[m].is_dummy, "score"].sum() or 1.0
        thr[m] = float(mt.set_index("method").loc[m, "threshold"]) / real
    n_kept = {m: int(imp[m].keep.sum()) for m in methods}
    verdict = summ.get("verdict") or (
        f"{len(methods)} methods ({', '.join(methods)}) screened {summ['n_inputs']} inputs with {summ['n_runs']} runs, keeping "
        f"{', '.join(str(n_kept[m]) for m in methods)} inputs respectively; the consensus set has {len(summ['consensus'])} inputs.")
    return dict(n_runs=summ["n_runs"], n_inputs=summ["n_inputs"], methods=methods, methods_table=mt.to_dict("records"), table=table,
                dummies=dummies, thr_share=thr, consensus=summ["consensus"], forced=summ.get("forced") or {}, min_votes=summ["min_votes"],
                final_cv_rmse_pcm=summ.get("final_cv_rmse_pcm"), retention=_records(d / "retention_curves.csv"),
                agreement=_records(d / "method_agreement.csv"), verdict=verdict)


def _keff(cfg, select, infos):
    d = cfg.out / "stage2_keff"
    if not (d / "summary.json").exists():
        return None
    summ = json.loads((d / "summary.json").read_text())
    labels = [l for l in summ["ranking"] if (cfg.out / "models" / f"keff_{l.replace(':', '_')}.pkl").exists()]
    chosen = _select(labels, select)
    models, space = {}, None
    for l in chosen:
        P = load_model(cfg.out / "models" / f"keff_{l.replace(':', '_')}.pkl")
        spec, info = export_scalar(P.model, d=P.space.dim)
        models[l], space = spec, P.space
        infos.append(dict(stage="k_eff", label=l, method=info["method"], max_dev=_dev_text(info, "keff"), size_kb=info["size_kb"]))
    st = cfg["keff"]
    keys, out = space.keys, st.get("output", "keff")
    metrics = pd.read_csv(d / "cv_metrics.csv")
    data = load_stage_data(cfg, "keff")
    cvp = pd.read_csv(d / "cv_predictions.csv") if (d / "cv_predictions.csv").exists() else None
    aligned = cvp is not None and len(cvp) == len(data) and np.allclose(cvp["keff"].to_numpy(), data[out].to_numpy())
    ids = (data["case_id"].astype(str) if "case_id" in data else pd.Series([f"run {i}" for i in range(len(data))])).tolist()
    ystd = data[st["output_std"]].to_numpy(float) if st.get("output_std") in data else None
    runs = dict(ids=ids, X=data[keys].to_numpy(float), y=data[out].to_numpy(float),
                y_std=ystd, cv={l: cvp[f"cv_pred[{l}]"].to_numpy(float) for l in chosen if aligned and f"cv_pred[{l}]" in cvp})
    noise = metrics["mc_noise_rms"].dropna().iloc[0] if "mc_noise_rms" in metrics and metrics["mc_noise_rms"].notna().any() else None
    if noise is None:
        nz = [s["noise"] for s in models.values() if s.get("noise")]
        noise = float(np.mean(nz) * 1e5) if nz else None
    b = metrics.iloc[0]
    verdict = summ.get("verdict") or (f"Best model: {b.model} with {b.rmse:.0f} pcm cross-validated RMSE (R² = {b.r2:.5f}).")
    if not aligned:
        verdict += (f" Note: the data now hold {len(data)} runs but the models were trained on {summ.get('n_runs')}: "
                    "re-run 'msrframe run --stages keff' to train on the new runs.")
    return dict(space=export_space(space), order=chosen, ranking=summ["ranking"], models=models, n_runs=int(summ.get("n_runs", len(data))),
                metrics=metrics.to_dict("records"), holdout=_records(d / "holdout_metrics.csv") or None, learning=_records(d / "learning_curves.csv"),
                runs=runs, noise_pcm=noise, derived_cfg=st.get("derived", {}) or {}, verdict=verdict)


def _burnup(cfg, select, infos):
    d = cfg.out / "stage3_burnup"
    if not (d / "summary.json").exists():
        return None
    summ = json.loads((d / "summary.json").read_text())
    labels = [l for l in summ["ranking"] if (cfg.out / "models" / f"burnup_{l.replace(':', '_')}.pkl").exists()]
    chosen = _select(labels, select)
    models, P = {}, None
    for l in chosen:
        P = load_model(cfg.out / "models" / f"burnup_{l.replace(':', '_')}.pkl")
        spec, info = export_trajectory(P.model, d=P.space.dim)
        models[l] = spec
        infos.append(dict(stage="burnup", label=l, method=info["method"], size_kb=info["size_kb"],
                          max_dev=f"{info['rel_err']:.1e} of output spread" if "rel_err" in info else ""))
    st = cfg["burnup"]
    outputs, times = list(P.outputs), [float(t) for t in P.times]
    df = load_stage_data(cfg, "burnup")
    X, Y, _, ids = histories(df, P.space.keys, outputs, P.times)
    cv = None
    cvf = d / "cv_predictions.json"
    if cvf.exists():
        c = json.loads(cvf.read_text())
        if c.get("ids") == list(map(str, ids)):
            cv = {l: c["pred"][l] for l in chosen if l in c["pred"]}
    labels_o = dict(st.get("output_labels", {}) or {})
    units = {o: (st.get("output_units", {}) or {}).get(o, _unit_of(o)) for o in outputs}
    metrics = pd.read_csv(d / "cv_metrics.csv")
    b = metrics.iloc[0]
    verdict = summ.get("verdict") or f"Best burnup model: {b.model} ({b.get('keff_rmse_pcm', float('nan')):.0f} pcm k_eff RMSE in cross-validation)."
    return dict(space=export_space(P.space), order=chosen, ranking=summ["ranking"], models=models, times=times, outputs=outputs,
                labels={o: labels_o.get(o, o) for o in outputs}, units=units, n_histories=int(summ.get("n_histories", len(ids))),
                folds=int((st.get("validation") or {}).get("folds", 5)), metrics=metrics.to_dict("records"),
                rmse_vs_time=_records(d / "keff_rmse_vs_time.csv"),
                histories=dict(ids=list(map(str, ids)), X=X, Y={o: Y[o] for o in outputs}), cv=cv, verdict=verdict)


def _optimize(cfg, burnup):
    if burnup is None:
        return None
    st = cfg["optimize"]
    res, surrogate = [], None
    p = cfg.out / "stage4_optimize" / "summary.json"
    if p.exists():
        s = json.loads(p.read_text())
        surrogate = s.get("surrogate")
        res = [dict(r, surrogate=surrogate) for r in s.get("results", [])]
    return dict(scenarios=st.get("scenarios", []) or [], python_results=res, surrogate=surrogate)


def _data(cfg, infos):
    files, counts = [], {}
    for stage, label in (("screening", "screening runs"), ("keff", "k_eff runs"), ("burnup", "burnup histories")):
        st = cfg.get(stage) or {}
        if not st.get("data"):
            continue
        user_dir = str(cfg.path(st.get("user_dir", "data/user")))
        total = 0
        for f in _files(cfg, st["data"]):
            d = pd.read_csv(f)
            n = d["case_id"].nunique() if stage == "burnup" and "case_id" in d else len(d)
            total += n
            kind = ("synthetic demo" if "demo" in str(f.relative_to(cfg.root) if str(f).startswith(str(cfg.root)) else f.name)
                    else "added by you" if str(f).startswith(user_dir) else "study data")
            files.append(dict(stage=stage, file=str(f.relative_to(cfg.root)) if str(f).startswith(str(cfg.root)) else str(f),
                              rows=int(n), kind=kind))
        if total:
            counts[label] = total
    hist = {s: _records(cfg.out / "history" / f"{s}_history.csv") for s in ("keff", "burnup")}
    al = _records(cfg.out / "active_learning" / "active_learning_history.csv")
    sug = {}
    for s in ("keff", "burnup"):
        p = cfg.out / f"suggested_runs_{s}.csv"
        if p.exists():
            sug[s] = pd.read_csv(p).head(200).to_dict("records")
    cp = getattr(cfg, "config_path", None)
    try:
        cfg_rel = str(Path(cp).relative_to(cfg.root)) if cp else "config/msfr_default.yaml"
    except ValueError:
        cfg_rel = str(cp)
    return dict(files=files, counts=counts, history=hist, active_learning=al, suggested=sug, models=infos, config=cfg_rel)


# --------------------------------------------------------------------------- build
def build_dashboard(cfg, keff_models=None, burnup_models=None, out=None, quiet=False):
    dcfg = cfg.get("dashboard", {}) or {}
    infos = []
    scr = _screening(cfg)
    kef = _keff(cfg, keff_models or dcfg.get("keff_models", "top3"), infos)
    bur = _burnup(cfg, burnup_models or dcfg.get("burnup_models", "top3"), infos)
    opt = _optimize(cfg, bur) if (cfg.get("optimize") or {}).get("enabled", True) else None
    dat = _data(cfg, infos)
    cat = cfg.catalogue()
    refcard = []
    if scr:
        refcard.append(("screening runs", f"{scr['n_runs']:,}"))
    if kef:
        refcard.append(("k_eff runs", f"{kef['n_runs']:,}"))
    if bur:
        refcard.append(("burnup histories", f"{bur['n_histories']:,}"))
    refcard.append(("generated", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")))
    models_txt = []
    if kef:
        models_txt.append("k_eff: " + ", ".join(kef["order"]))
    if bur:
        models_txt.append("burnup: " + ", ".join(bur["order"]))
    meta = dict(project=cfg.get("project", "MSR design study"), refcard=refcard,
                subtitle="Surrogate models trained on your OpenMC data, running in this page. " + ("Models shown - " + "; ".join(models_txt) + "." if models_txt else ""))
    data = _clean(dict(meta=meta, catalogue=None if cat is None else cat.to_dict("records"), screening=scr, keff=kef, burnup=bur,
                       optimize=opt, data=dat))
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    page = (ASSETS / "template.html").read_text(encoding="utf-8")
    for key, val in (("CSS", (ASSETS / "style.css").read_text(encoding="utf-8")), ("RUNTIME", (ASSETS / "runtime.js").read_text(encoding="utf-8")),
                     ("CHARTS", (ASSETS / "charts.js").read_text(encoding="utf-8")), ("APP", (ASSETS / "app.js").read_text(encoding="utf-8"))):
        page = page.replace(f"/*__{key}__*/", val.replace("</script", "<\\/script"))
    page = page.replace("/*__DATA__*/", blob)
    path = Path(out) if out else cfg.out / dcfg.get("file", "dashboard.html")
    path.write_text(page, encoding="utf-8")
    mb = path.stat().st_size / 2 ** 20
    if not quiet:
        print(f"[dashboard] {path}  ({mb:.1f} MB)")
        for i in infos:
            print(f"  {i['stage']:<7} {i['label']:<24} {i['method']:<16} {i['size_kb']:8.0f} KB  {i['max_dev']}")
        if mb > dcfg.get("max_mb", 15):
            print(f"  warning: the page is {mb:.0f} MB; embed fewer or smaller models (e.g. dashboard.burnup_models: top2)")
    return path
