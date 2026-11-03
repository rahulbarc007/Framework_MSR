"""Data store: the user's OpenMC results are just CSV files.

Each stage lists one or more CSV files / glob patterns under ``data:``. All matching
files are concatenated (later files win on duplicate inputs), so adding new runs is as
simple as dropping ``data/user/keff_batch2.csv`` next to the existing files, or using
``python -m msrframe add-data --stage keff --file my_runs.csv`` which validates first.

Required columns
  screening : every input key + the output (default ``keff``); optional ``keff_std``,
              ``morris_traj`` (trajectory id of a Morris design), ``dummy_*``
  keff      : every input key + ``keff``; optional ``keff_std``, ``split`` (train/test)
  burnup    : ``case_id``, every input key, ``time_y`` and each output column
"""
from __future__ import annotations

import datetime as _dt
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _files(cfg, patterns):
    out = []
    for p in patterns if isinstance(patterns, (list, tuple)) else [patterns]:
        hits = sorted(glob.glob(str(cfg.path(p))))
        out += [Path(h) for h in hits]
    return out


def load_stage_data(cfg, stage: str) -> pd.DataFrame:
    st = cfg[stage]
    files = _files(cfg, st.get("data", []))
    if not files:
        raise FileNotFoundError(f"[{stage}] no data files match {st.get('data')}")
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["source"] = f.name
        frames.append(d)
    df = pd.concat(frames, ignore_index=True, sort=False)
    inputs = list(st["inputs"])
    outputs = stage_outputs(st)
    missing = [c for c in inputs + outputs if c not in df]
    if missing:
        raise ValueError(f"[{stage}] data is missing columns {missing}")
    df = df.dropna(subset=inputs + outputs[:1]).reset_index(drop=True)
    if stage == "burnup":
        if "case_id" not in df:
            raise ValueError("[burnup] data needs a case_id column")
        df = df.drop_duplicates(subset=["case_id", "time_y"], keep="last")
    elif st.get("dedupe", stage != "screening"):   # never de-duplicate a Morris design
        extra = [c for c in df.columns if c.startswith("dummy") or c == "morris_traj"]
        df = df.drop_duplicates(subset=inputs + extra, keep="last").reset_index(drop=True)
    return df


def stage_outputs(st):
    o = st.get("outputs", st.get("output", "keff"))
    return [o] if isinstance(o, str) else list(o)


def histories(df: pd.DataFrame, inputs, outputs, times=None):
    """Long burnup table -> X (n_hist, d), {output: Y (n_hist, T)}, times, case ids.

    Histories on a different time grid are linearly interpolated onto ``times``
    (default: the grid of the most common history)."""
    groups = list(df.groupby("case_id", sort=False))
    if times is None:
        grids = pd.Series([tuple(g["time_y"].round(6)) for _, g in groups])
        times = np.array(grids.mode().iloc[0])
    times = np.asarray(times, float)
    X, Y, ids = [], {o: [] for o in outputs}, []
    for cid, g in groups:
        g = g.sort_values("time_y")
        if g["time_y"].max() < times[-1] - 1e-6 or g["time_y"].min() > times[0] + 1e-6:
            continue  # too short to cover the grid
        X.append(g[inputs].iloc[0].to_numpy(float))
        for o in outputs:
            Y[o].append(np.interp(times, g["time_y"].to_numpy(float), g[o].to_numpy(float)))
        ids.append(cid)
    return np.array(X), {o: np.array(v) for o, v in Y.items()}, times, ids


def add_data(cfg, stage: str, file, tag: str | None = None) -> Path:
    """Validate a CSV of new runs and copy it into the stage's user-data folder."""
    st = cfg[stage]
    new = pd.read_csv(file)
    need = list(st["inputs"]) + stage_outputs(st)[:1] + (["case_id", "time_y"] if stage == "burnup" else [])
    miss = [c for c in need if c not in new]
    if miss:
        raise ValueError(f"{file}: missing required columns {miss}")
    bad = new[need].isna().any(axis=1)
    if bad.any():
        print(f"  warning: {int(bad.sum())} rows with empty required values will be ignored")
    space = cfg.space(stage)
    out = space.outside(new[list(st["inputs"])].to_numpy(float))
    if out.any():
        cols = [k for k, o in zip(st["inputs"], out.any(0)) if o]
        print(f"  warning: {int(out.any(1).sum())} rows lie outside the declared bounds of {cols} "
              "(the surrogates will extrapolate there; widen the bounds in the config if intended)")
    dest_dir = cfg.path(st.get("user_dir", "data/user"))
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dest_dir / f"{stage}_{tag or Path(file).stem}_{stamp}.csv"
    new.to_csv(dest, index=False)
    n = new["case_id"].nunique() if stage == "burnup" else len(new)
    print(f"  added {n} {'histories' if stage == 'burnup' else 'runs'} -> {dest}")
    return dest


def append_history(cfg, stage: str, rows: list[dict]) -> pd.DataFrame:
    """Append a training-session record (n_data, model, metrics) and return the full log."""
    p = cfg.out / "history" / f"{stage}_history.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now().isoformat(timespec="seconds")
    new = pd.DataFrame([{"timestamp": now, **r} for r in rows])
    old = pd.read_csv(p) if p.exists() else None
    full = new if old is None else pd.concat([old, new], ignore_index=True)
    full.to_csv(p, index=False)
    return full
