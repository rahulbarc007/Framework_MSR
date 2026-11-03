"""Global-sensitivity / importance methods for Stage 1 (parameter screening).

Each method is a ``screening`` plug-in returning one non-negative *score* per input
(higher = more important).  Methods that need a model take a ``surrogate`` option, so
the same index can be computed with different learners, e.g.::

    screening:
      methods:
        - random_forest                         # replaces Morris
        - sobol_pce
        - {name: active_subspace, surrogate: gp}
        - {name: morris, surrogate: dnn}        # surrogate-assisted Morris
        - {name: permutation, surrogate: gbt}

The stage adds independent *dummy* inputs; every method's retention threshold is
calibrated on the largest dummy score (noise floor), as in the EVOL pilot study.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .design import morris as morris_design
from .registry import create, register


@dataclass
class ScreeningContext:
    U: np.ndarray            # (n, k) unit-cube inputs, including dummy columns
    y: np.ndarray            # (n,) output (pcm for k_eff)
    keys: list               # k names
    traj: np.ndarray | None  # Morris trajectory ids (n,) or None
    simulator: object = None  # optional callable U -> y (cheap model / analytic stand-in)
    seed: int = 0


class ScreeningMethod:
    units = "score"

    def __init__(self, seed=0, **params):
        self.seed, self.params = seed, params

    def surrogate(self, ctx, default="gp"):
        return create("surrogate", self.params.get("surrogate", default), seed=self.seed).fit(ctx.U, ctx.y)

    def run(self, ctx: ScreeningContext) -> pd.DataFrame:
        raise NotImplementedError


def _df(keys, score, **extra):
    return pd.DataFrame({"key": keys, "score": np.asarray(score, float), **{k: np.asarray(v) for k, v in extra.items()}})


# --------------------------------------------------------------------------- Morris
@register("screening", "morris", "Morris elementary effects mu* (classic design, or surrogate-assisted)")
class Morris(ScreeningMethod):
    units = "mu* (pcm)"

    def run(self, ctx):
        if ctx.traj is not None and not self.params.get("force_surrogate", False):
            sel = ctx.traj >= 0               # rows that belong to a Morris trajectory
            U, y, tr = ctx.U[sel], ctx.y[sel], ctx.traj[sel]
            mode = "design"
        else:
            r = self.params.get("trajectories", 60)
            U, tr = morris_design(r, ctx.U.shape[1], seed=self.seed)
            f = ctx.simulator if (ctx.simulator is not None and self.params.get("use_simulator", False)) \
                else self.surrogate(ctx, "gp").predict
            y = f(U)
            mode = f"surrogate:{self.params.get('surrogate', 'gp')}"
        k = U.shape[1]
        EE = [[] for _ in range(k)]
        for t in np.unique(tr):
            idx = np.flatnonzero(tr == t)
            for a, b in zip(idx[:-1], idx[1:]):
                dU = U[b] - U[a]
                j = int(np.argmax(np.abs(dU)))
                if abs(dU[j]) < 1e-12:
                    continue
                # moved coordinates other than j would break OAT; dummies may move freely in data designs
                EE[j].append((y[b] - y[a]) / dU[j])
        mu = np.array([np.mean(e) if e else 0.0 for e in EE])
        mus = np.array([np.mean(np.abs(e)) if e else 0.0 for e in EE])
        sig = np.array([np.std(e, ddof=1) if len(e) > 1 else 0.0 for e in EE])
        self.mode = mode
        return _df(ctx.keys, mus, mu=mu, sigma=sig)


# --------------------------------------------------------------------------- Sobol from sparse PCE
@register("screening", "sobol_pce", "Total Sobol indices S_T from a sparse polynomial-chaos fit")
class SobolPCE(ScreeningMethod):
    units = "S_T"

    def run(self, ctx):
        m = create("surrogate", {"name": "pce", **self.params.get("pce", {})}, seed=self.seed).fit(ctx.U, ctx.y)
        S1, ST = m.sobol()
        self.info = dict(terms=m.n_terms, degree=m.p)
        return _df(ctx.keys, ST, S1=S1)


# --------------------------------------------------------------------------- Sobol (Saltelli/Jansen) on any surrogate
@register("screening", "sobol_surrogate", "Total Sobol indices (Jansen estimator) on any fitted surrogate")
class SobolSurrogate(ScreeningMethod):
    units = "S_T"

    def run(self, ctx):
        f = self.surrogate(ctx, "gp").predict
        n, k = self.params.get("n", 4096), ctx.U.shape[1]
        rng = np.random.default_rng(self.seed)
        A, B = rng.random((n, k)), rng.random((n, k))
        fA, fB = f(A), f(B)
        V = np.var(np.r_[fA, fB])
        ST, S1 = np.zeros(k), np.zeros(k)
        for j in range(k):
            ABj = A.copy()
            ABj[:, j] = B[:, j]
            fj = f(ABj)
            ST[j] = 0.5 * np.mean((fA - fj) ** 2) / V
            S1[j] = np.mean(fB * (fj - fA)) / V
        return _df(ctx.keys, np.clip(ST, 0, None), S1=S1)


# --------------------------------------------------------------------------- active subspace
@register("screening", "active_subspace", "Active-subspace activity scores from surrogate gradients")
class ActiveSubspace(ScreeningMethod):
    units = "activity"

    def run(self, ctx):
        m = self.surrogate(ctx, "gp")
        M = self.params.get("n_grad", 1000)
        Ug = np.random.default_rng(self.seed).random((M, ctx.U.shape[1]))
        G = m.gradient(Ug)
        C = G.T @ G / M
        lam, W = np.linalg.eigh(C)
        lam, W = lam[::-1], W[:, ::-1]
        lam = np.clip(lam, 0, None)
        n_act = self.params.get("n_active") or int(np.searchsorted(np.cumsum(lam) / lam.sum(), 0.95) + 1)
        act = (W[:, :n_act] ** 2) @ lam[:n_act]
        self.info = dict(n_active=n_act, eig_top5=lam[:5].tolist(),
                         eigen_gap=float(lam[0] / max(lam[1], 1e-300)) if len(lam) > 1 else np.inf)
        return _df(ctx.keys, act / act.sum(), w1=W[:, 0])


# --------------------------------------------------------------------------- derivative-based (DGSM)
@register("screening", "dgsm", "Derivative-based global sensitivity E[(df/dx)^2] on a surrogate")
class DGSM(ScreeningMethod):
    units = "nu"

    def run(self, ctx):
        m = self.surrogate(ctx, "gp")
        Ug = np.random.default_rng(self.seed).random((self.params.get("n_grad", 1000), ctx.U.shape[1]))
        nu = np.mean(m.gradient(Ug) ** 2, 0)
        return _df(ctx.keys, nu / (np.pi ** 2) / (np.var(ctx.y) or 1))  # Poincare upper bound on S_T


# --------------------------------------------------------------------------- permutation importance (any model)
@register("screening", "permutation", "Cross-validated permutation importance of any surrogate")
class Permutation(ScreeningMethod):
    units = "dMSE / Var"
    default_model = "random_forest"

    def run(self, ctx):
        from sklearn.model_selection import KFold
        folds = self.params.get("folds", 5)
        reps = self.params.get("repeats", 3)
        rng = np.random.default_rng(self.seed)
        k = ctx.U.shape[1]
        imp = np.zeros(k)
        var = np.var(ctx.y) or 1.0
        for tr, te in KFold(folds, shuffle=True, random_state=self.seed).split(ctx.U):
            m = create("surrogate", self.params.get("surrogate", self.default_model), seed=self.seed).fit(ctx.U[tr], ctx.y[tr])
            base = np.mean((m.predict(ctx.U[te]) - ctx.y[te]) ** 2)
            for j in range(k):
                d = 0.0
                for _ in range(reps):
                    Up = ctx.U[te].copy()
                    Up[:, j] = rng.permutation(Up[:, j])
                    d += np.mean((m.predict(Up) - ctx.y[te]) ** 2) - base
                imp[j] += d / reps / var / folds
        return _df(ctx.keys, np.clip(imp, 0, None))


@register("screening", "random_forest", "Random-forest permutation importance (drop-in replacement for Morris)")
class RFImportance(Permutation):
    default_model = {"name": "random_forest", "n_estimators": 200}

    def run(self, ctx):
        if self.params.get("importance", "permutation") == "impurity":
            from sklearn.ensemble import RandomForestRegressor
            m = RandomForestRegressor(self.params.get("n_estimators", 500), random_state=self.seed, n_jobs=-1).fit(ctx.U, ctx.y)
            return _df(ctx.keys, m.feature_importances_)
        return super().run(ctx)


@register("screening", "gbt_importance", "Gradient-boosted-trees permutation importance")
class GBTImportance(Permutation):
    default_model = "gbt"


# --------------------------------------------------------------------------- cheap data-only indices
@register("screening", "mutual_info", "Mutual information between each input and the output")
class MutualInfo(ScreeningMethod):
    units = "MI (nats)"

    def run(self, ctx):
        from sklearn.feature_selection import mutual_info_regression
        return _df(ctx.keys, mutual_info_regression(ctx.U, ctx.y, random_state=self.seed, n_neighbors=5))


@register("screening", "spearman", "Squared Spearman rank correlation (monotone effects)")
class Spearman(ScreeningMethod):
    units = "rho^2"

    def run(self, ctx):
        from scipy.stats import spearmanr
        r = np.array([spearmanr(ctx.U[:, j], ctx.y)[0] for j in range(ctx.U.shape[1])])
        return _df(ctx.keys, np.nan_to_num(r) ** 2, rho=r)


@register("screening", "src", "Squared standardised regression coefficients (linear variance shares)")
class SRC(ScreeningMethod):
    units = "SRC^2"

    def run(self, ctx):
        Z = (ctx.U - ctx.U.mean(0)) / (ctx.U.std(0) + 1e-12)
        b, *_ = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z], ctx.y, rcond=None)
        src = b[1:] / (ctx.y.std() or 1)
        return _df(ctx.keys, src ** 2, src=src)


@register("screening", "lasso", "|coefficient| of a cross-validated Lasso on standardised inputs")
class Lasso(ScreeningMethod):
    units = "|beta|"

    def run(self, ctx):
        from sklearn.linear_model import LassoCV
        Z = (ctx.U - ctx.U.mean(0)) / (ctx.U.std(0) + 1e-12)
        m = LassoCV(cv=5, random_state=self.seed).fit(Z, ctx.y)
        return _df(ctx.keys, np.abs(m.coef_), beta=m.coef_)
