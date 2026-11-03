"""Simulators: what produces *new* training data for the active-learning loop.

``analytic_msfr``  - a cheap analytic stand-in for OpenMC k_eff, calibrated to the
                     signed Morris effects of the EVOL pilot (``data/screening_reference.csv``)
                     plus curvature, one interaction and Monte Carlo noise. It exists so the
                     framework can be exercised end-to-end without OpenMC; it is NOT physics.
``command``        - runs your own OpenMC driver: writes the requested points to a CSV,
                     calls e.g. ``python my_openmc_runs.py {inp} {out}``, and reads the
                     results CSV back (same columns + the output column).
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


class AnalyticMSFR:
    def __init__(self, reference_csv, keys=None, k0=0.99246, noise_pcm=280.0, seed=0, catalogue_csv=None):
        ref = pd.read_csv(reference_csv)
        ref = ref[~ref["key"].str.startswith("dummy")]
        cat = pd.read_csv(catalogue_csv).set_index("key") if catalogue_csv else None
        self.ref = ref.set_index("key")
        self.all_keys = list(self.ref.index)
        self.keys = list(keys) if keys is not None else self.all_keys
        self.k0, self.noise, self.rng = k0, noise_pcm, np.random.default_rng(seed)
        lo, hi = self.ref["low"].to_numpy(float), self.ref["high"].to_numpy(float)
        nom = cat.loc[self.all_keys, "nominal"].to_numpy(float) if cat is not None else 0.5 * (lo + hi)
        self.lo, self.hi, self.u0 = lo, hi, (nom - lo) / np.where(hi > lo, hi - lo, 1)
        self.mu = self.ref["mu"].to_numpy(float)          # pcm per unit (full-range) change
        self.curv = 0.12 * self.mu                         # mild curvature
        self.ix = [self.all_keys.index(k) for k in ("uf4_molpct", "hn_molpct") if k in self.all_keys]

    def keff(self, X_phys, noisy=True):
        """X_phys: (n, len(self.keys)) physical values; other factors stay at nominal."""
        X = np.atleast_2d(X_phys)
        U = np.tile(self.u0, (len(X), 1))
        for j, k in enumerate(self.keys):
            if k in self.all_keys:
                i = self.all_keys.index(k)
                U[:, i] = (X[:, j] - self.lo[i]) / (self.hi[i] - self.lo[i])
        d = U - self.u0
        pcm = d @ self.mu + (d ** 2) @ self.curv
        if len(self.ix) == 2:
            a, b = self.ix
            pcm += -0.08 * abs(self.mu[a]) * d[:, a] * d[:, b]
        if noisy and self.noise:
            pcm = pcm + self.rng.normal(0, self.noise, len(pcm))
        return self.k0 + pcm * 1e-5

    def run(self, df: pd.DataFrame, output="keff") -> pd.DataFrame:
        out = df.copy()
        out[output] = self.keff(df[self.keys].to_numpy(float))
        out[output + "_std"] = self.noise * 1e-5
        return out


class CommandSimulator:
    def __init__(self, command: str, workdir=None):
        self.command, self.workdir = command, workdir

    def run(self, df: pd.DataFrame, output="keff") -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as td:
            inp, out = Path(td) / "requested_runs.csv", Path(td) / "results.csv"
            df.to_csv(inp, index=False)
            cmd = self.command.format(inp=inp, out=out)
            print(f"  running: {cmd}")
            subprocess.run(cmd, shell=True, check=True, cwd=self.workdir)
            return pd.read_csv(out)


def make_simulator(cfg, spec, keys):
    spec = dict(spec or {})
    kind = spec.pop("name", "analytic_msfr")
    if kind == "analytic_msfr":
        return AnalyticMSFR(cfg.path(spec.pop("reference", "data/screening_reference.csv")), keys=keys,
                            catalogue_csv=cfg.path(cfg["catalogue"]) if cfg.get("catalogue") else None, **spec)
    if kind == "command":
        return CommandSimulator(spec["command"], spec.get("workdir"))
    raise ValueError(f"unknown simulator {kind!r} (analytic_msfr | command)")
