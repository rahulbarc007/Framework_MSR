"""Command-line interface:  python -m msrframe <command> [options]

  run          run the enabled stages (or --stages keff burnup) and write report.html
  list         list every registered plug-in (models, screening methods, optimisers, samplers)
  add-data     validate a CSV of new OpenMC runs and add it to the stage's database
  suggest      propose the next OpenMC runs that will improve a surrogate most
  loop         closed active-learning loop (suggest -> simulate -> add -> retrain)
  predict      predict k_eff (or burnup trajectories) for points in a CSV with a trained model
  design       generate an input design (lhs / sobol / morris ...) to run in OpenMC
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from .config import load_config
from .registry import available


def _sets(s):
    """--set keff.models=[gp,random_forest]  -> nested dict (value parsed as YAML)."""
    import yaml
    out = {}
    for item in s or []:
        k, v = item.split("=", 1)
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="msrframe", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "add-data", "suggest", "loop", "predict", "design", "list"):
        p = sub.add_parser(name)
        p.add_argument("--config", "-c", default="config/msfr_default.yaml")
        p.add_argument("--set", action="append", help="override a config value, e.g. screening.methods=[random_forest,sobol_pce]")
        if name == "run":
            p.add_argument("--stages", nargs="*")
            p.add_argument("--tag", default=None, help="label stored in the accuracy history")
        if name in ("add-data", "suggest", "predict", "design"):
            p.add_argument("--stage", default="keff", choices=["screening", "keff", "burnup"])
        if name in ("add-data", "predict"):
            p.add_argument("--file", required=True)
        if name == "add-data":
            p.add_argument("--retrain", action="store_true", help="re-run the stage after adding")
        if name == "suggest":
            p.add_argument("-n", type=int, default=20)
            p.add_argument("--how", default="hybrid", choices=["std", "committee", "hybrid"])
        if name == "predict":
            p.add_argument("--model", default="deployed")
            p.add_argument("--out", default=None)
        if name == "design":
            p.add_argument("--sampler", default="lhs")
            p.add_argument("-n", type=int, default=100)
            p.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.cmd == "list":
        load_config(a.config if a.config and __import__("os").path.exists(a.config) else None)
        for kind, items in available().items():
            print(f"\n{kind}:")
            for n, d in items.items():
                print(f"  {n:<24} {d}")
        return
    cfg = load_config(a.config, _sets(a.set))

    if a.cmd == "run":
        from .pipeline import run_pipeline
        run_pipeline(cfg, a.stages, tag=a.tag)
    elif a.cmd == "add-data":
        from .data import add_data
        add_data(cfg, a.stage, a.file)
        if a.retrain:
            from .pipeline import run_pipeline
            run_pipeline(cfg, [a.stage], tag="add-data")
    elif a.cmd == "suggest":
        from .active import suggest
        suggest(cfg, a.stage, a.n, a.how)
    elif a.cmd == "loop":
        from .active import loop
        from .pipeline import run_pipeline
        from .stages import keff
        hist, fig = loop(cfg)
        R = keff.run(cfg, tag="after-loop")
        from .report import write_report
        print("report ->", write_report(cfg, [R], extra_figs={"Active learning": [(fig, "Active (uncertainty-driven) vs random new runs.")]}))
    elif a.cmd == "predict":
        from .stages.common import load_model
        P = load_model(cfg.out / "models" / f"{a.stage}_{a.model.replace(':', '_')}.pkl")
        df = pd.read_csv(a.file)
        if a.stage == "keff":
            mu, sd = P.predict(df)
            df["keff_pred"], df["keff_sd"] = mu, sd
            df["rho_pred_pcm"] = (mu - 1) / mu * 1e5
        else:
            pr = P.predict(df)
            rows = []
            for i in range(len(df)):
                for it, t in enumerate(P.times):
                    rows.append({**df.iloc[i].to_dict(), "time_y": t, **{f"{o}_pred": pr[o][0][i, it] for o in pr},
                                 **{f"{o}_sd": pr[o][1][i, it] for o in pr}})
            df = pd.DataFrame(rows)
        out = a.out or str(cfg.out / f"predictions_{a.stage}.csv")
        df.to_csv(out, index=False)
        print(df.head(20).to_string(index=False))
        print(f"-> {out}  (model {P.label})")
    elif a.cmd == "design":
        from .registry import get
        st = cfg[a.stage]
        keys = st["inputs"] if isinstance(st.get("inputs"), list) else None
        if keys is None:
            raise SystemExit("design: list the stage inputs explicitly in the config")
        space = cfg.space(a.stage, keys)
        if a.sampler == "morris":
            U, tr = get("sampler", "morris")(a.n, len(keys) + 3, seed=cfg["seed"])
            df = pd.DataFrame(space.from_unit(U[:, :len(keys)]), columns=keys)
            for i in range(3):
                df[f"dummy_{i + 1}"] = U[:, len(keys) + i]
            df["morris_traj"] = tr
        else:
            df = pd.DataFrame(space.from_unit(get("sampler", a.sampler)(a.n, len(keys), seed=cfg["seed"])), columns=keys)
        df.insert(0, "case_id", [f"D{i:05d}" for i in range(len(df))])
        out = a.out or str(cfg.out / f"design_{a.stage}_{a.sampler}.csv")
        df.to_csv(out, index=False)
        print(f"{len(df)} design points -> {out}")


if __name__ == "__main__":
    main()
