from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


class StageResult(dict):
    """Everything a stage produced: summary dict, tables, figure paths and verdict text."""

    def __init__(self, stage, title):
        super().__init__(stage=stage, title=title, summary={}, tables={}, figures=[], verdict="", notes=[])

    def fig(self, path, caption):
        self["figures"].append((str(path), caption))

    def table(self, name, df, out_dir: Path | None = None):
        self["tables"][name] = df
        if out_dir is not None:
            df.to_csv(Path(out_dir) / f"{name}.csv", index=False)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict("records")
    return o


def write_json(path, obj):
    Path(path).write_text(json.dumps(jsonable(obj), indent=1, default=str))


def save_model(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def print_table(df, title=None, n=30):
    if title:
        print(f"\n  {title}")
    with pd.option_context("display.width", 160, "display.max_columns", 30, "display.float_format", "{:,.4g}".format):
        print("  " + df.head(n).to_string(index=False).replace("\n", "\n  "))
