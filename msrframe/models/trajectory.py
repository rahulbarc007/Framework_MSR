"""Trajectory (time-series output) surrogates for the burnup stage.

Every trajectory model wraps a *base* scalar surrogate taken from the registry, so the
burnup predictor can be a PCA-GP, PCA-Random-Forest, PCA-PCE, time-as-input GBT, ...::

    burnup:
      models:
        - {name: pca, base: gp}
        - {name: pca, base: random_forest}
        - {name: time_input, base: gbt}

Interface:  ``fit(U, {out: Y(n,T)}, times)`` / ``predict(U) -> {out: (mu(n,T), sd(n,T))}``
"""
from __future__ import annotations

import numpy as np

from ..registry import create, register


class TrajectoryModel:
    def __init__(self, base="gp", seed=0, **params):
        self.base, self.seed, self.params = base, seed, params

    def _new(self, i=0):
        return create("surrogate", self.base, seed=self.seed + i)


@register("trajectory", "pca", "PCA of each output trajectory, one base surrogate per component score")
class PCATrajectory(TrajectoryModel):
    def fit(self, U, Ys, times):
        self.times, self.out = np.asarray(times), {}
        keep = self.params.get("variance", 0.99999)
        maxc = self.params.get("max_components", 8)
        for o, Y in Ys.items():
            mu = Y.mean(0)
            sc = (Y - mu).std() or 1.0
            Z = (Y - mu) / sc
            _, s, Vt = np.linalg.svd(Z, full_matrices=False)
            ev = s ** 2 / max(np.sum(s ** 2), 1e-300)
            nc = int(min(maxc, np.searchsorted(np.cumsum(ev), keep) + 1, len(s)))
            V = Vt[:nc]
            S = Z @ V.T
            trunc = float(np.mean((Z - S @ V) ** 2))
            models = [self._new(i).fit(U, S[:, i]) for i in range(nc)]
            self.out[o] = dict(mu=mu, sc=sc, V=V, models=models, trunc=trunc)
        return self

    def predict(self, U, outputs=None, noise=False):
        """noise=True adds the fitted label noise (predictive sd of a new OpenMC history)."""
        res = {}
        for o in outputs or self.out:
            O = self.out[o]
            mu = np.zeros((len(U), len(self.times)))
            var = np.full_like(mu, O["trunc"])
            for m, v in zip(O["models"], O["V"]):
                a, s = m.predict(U, return_std=True)
                mu += a[:, None] * v[None]
                var += (np.nan_to_num(s, nan=0.0)[:, None] ** 2 + (m.noise_sd ** 2 if noise else 0)) * v[None] ** 2
            has = all(m.supports_std for m in O["models"])
            res[o] = (O["mu"] + mu * O["sc"], np.sqrt(var) * O["sc"] if has else np.full_like(mu, np.nan))
        return res


@register("trajectory", "time_input", "Single base surrogate with time appended as an extra input")
class TimeInputTrajectory(TrajectoryModel):
    def fit(self, U, Ys, times):
        self.times = np.asarray(times)
        tn = (self.times - self.times[0]) / (np.ptp(self.times) or 1.0)
        self.tn, self.out = tn, {}
        n, T = len(U), len(tn)
        UU = np.hstack([np.repeat(U, T, 0), np.tile(tn, n)[:, None]])
        for i, (o, Y) in enumerate(Ys.items()):
            mu = Y.mean(0)
            sc = (Y - mu).std() or 1.0
            self.out[o] = dict(mu=mu, sc=sc, m=self._new(i).fit(UU, ((Y - mu) / sc).ravel()))
        return self

    def predict(self, U, outputs=None, noise=False):
        n, T = len(U), len(self.tn)
        UU = np.hstack([np.repeat(U, T, 0), np.tile(self.tn, n)[:, None]])
        res = {}
        for o in outputs or self.out:
            O = self.out[o]
            a, s = O["m"].predict(UU, return_std=True)
            if noise:
                s = np.sqrt(s ** 2 + O["m"].noise_sd ** 2)
            res[o] = (O["mu"] + a.reshape(n, T) * O["sc"], s.reshape(n, T) * O["sc"])
        return res


@register("trajectory", "per_time", "Independent base surrogate at every time point (no PCA)")
class PerTimeTrajectory(TrajectoryModel):
    def fit(self, U, Ys, times):
        self.times, self.out = np.asarray(times), {}
        for o, Y in Ys.items():
            self.out[o] = [self._new(t).fit(U, Y[:, t]) for t in range(Y.shape[1])]
        return self

    def predict(self, U, outputs=None, noise=False):
        res = {}
        for o in outputs or self.out:
            P = [m.predict(U, return_std=True) for m in self.out[o]]
            nz = np.array([m.noise_sd if noise else 0.0 for m in self.out[o]])
            res[o] = (np.array([p[0] for p in P]).T, np.sqrt(np.array([p[1] for p in P]).T ** 2 + nz ** 2))
        return res
