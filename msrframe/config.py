"""YAML configuration loading with defaults, path resolution and plug-in import."""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import yaml

from .registry import load_plugin_file
from .space import Param, Space

DEFAULTS = {
    "project": "MSR design study",
    "output_dir": "results",
    "seed": 2026,
    "plugins": [],
    "catalogue": None,
    "screening": {"enabled": True},
    "keff": {"enabled": True},
    "burnup": {"enabled": True},
    "optimize": {"enabled": True},
    "dashboard": {"enabled": True, "keff_models": "top3", "burnup_models": "top3", "file": "dashboard.html"},
}


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


class Config(dict):
    """Dict with a ``root`` (directory of the YAML file) used to resolve relative paths."""

    root: Path
    config_path: Path | None = None

    def path(self, p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def out(self) -> Path:
        p = self.path(self["output_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    def catalogue(self) -> pd.DataFrame | None:
        if not self.get("catalogue"):
            return None
        return pd.read_csv(self.path(self["catalogue"]))

    def space(self, stage: str, keys=None, df: pd.DataFrame | None = None) -> Space:
        """Build the parameter space of a stage.

        Priority per parameter: ``<stage>.params.<key>`` in YAML  >  catalogue row  >
        min/max of the data.
        """
        st = self[stage]
        keys = keys or st.get("inputs")
        over = st.get("params", {}) or {}
        cat = self.catalogue()
        catd = {} if cat is None else {r["key"]: r for r in cat.to_dict("records")}
        ps = []
        for k in keys:
            rec = {"key": k}
            if k in catd:
                c = catd[k]
                rec.update(low=c["low"], high=c["high"], nominal=c["nominal"], name=c.get("name", k),
                           unit=str(c.get("unit", "")), cls=str(c.get("cls", "")), log=c.get("scale") == "log")
            if k in over:
                rec.update(over[k])
            if "low" not in rec:
                if df is None or k not in df:
                    raise ValueError(f"no bounds for parameter {k!r}: add it to {stage}.params or the catalogue")
                rec.update(low=float(df[k].min()), high=float(df[k].max()))
            ps.append(Param(**{a: b for a, b in rec.items() if a in Param.__dataclass_fields__}))
        return Space(ps)


def load_config(path=None, overrides: dict | None = None) -> Config:
    if path is None:
        raw, root = {}, Path.cwd()
    else:
        path = Path(path)
        raw, root = yaml.safe_load(path.read_text()) or {}, path.resolve().parent
        # configs live in config/ and refer to data relative to the project root
        if raw.get("root"):
            root = (root / raw["root"]).resolve()
    cfg = Config(deep_merge(DEFAULTS, deep_merge(raw, overrides or {})))
    cfg.root = root
    cfg.config_path = None if path is None else Path(path).resolve()
    for p in cfg.get("plugins") or []:
        load_plugin_file(cfg.path(p) if str(p).endswith(".py") else p)
    return cfg
