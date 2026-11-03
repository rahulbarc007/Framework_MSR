"""Scalar surrogate models (inputs already mapped to the unit cube).

All models share one interface::

    m = create("surrogate", "gp")         # or {"name": "random_forest", "n_estimators": 800}
    m.fit(U, y)                           # U in [0, 1]^d
    mu, sd = m.predict(U, return_std=True)  # sd is NaN when the model has no uncertainty

``sd`` is the uncertainty about the *mean* (Monte Carlo label noise removed) for the
GP; for ensembles it is the member/tree spread.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np

from ..registry import register

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:  # pragma: no cover
    pass


class Surrogate:
    supports_std = False
    standardize_y = True
    noise_var = 0.0   # fitted label-noise variance (standardised units); GP sets it

    @property
    def noise_sd(self):
        """Fitted observation-noise sd in output units (0 if the model does not estimate it)."""
        return float(np.sqrt(self.noise_var) * getattr(self, "ys", 1.0))

    def __init__(self, seed: int = 0, **params):
        self.seed = seed
        self.params = params

    # ---- subclasses implement _fit / _predict on standardised y
    def fit(self, U, y, y_std=None):
        U, y = np.asarray(U, float), np.asarray(y, float).ravel()
        self.d = U.shape[1]
        self.ym, self.ys = (y.mean(), y.std() or 1.0) if self.standardize_y else (0.0, 1.0)
        self._fit(U, (y - self.ym) / self.ys)
        return self

    def predict(self, U, return_std=False):
        U = np.atleast_2d(np.asarray(U, float))
        mu, sd = self._predict(U, return_std)
        mu = mu * self.ys + self.ym
        if not return_std:
            return mu
        sd = np.full(len(U), np.nan) if sd is None else sd * self.ys
        return mu, sd

    def gradient(self, U, h=1e-3):
        """Central finite-difference gradient (in unit-cube coordinates)."""
        U = np.atleast_2d(U)
        G = np.empty_like(U)
        for j in range(U.shape[1]):
            a, b = U.copy(), U.copy()
            a[:, j] = np.clip(a[:, j] + h, 0, 1)
            b[:, j] = np.clip(b[:, j] - h, 0, 1)
            G[:, j] = (self.predict(a) - self.predict(b)) / (a[:, j] - b[:, j])
        return G


# --------------------------------------------------------------------------- Gaussian process
class _LBFGS:
    """Picklable L-BFGS-B optimiser with an iteration cap for sklearn's GP."""

    def __init__(self, max_iter):
        self.max_iter = max_iter

    def __call__(self, obj, x0, bounds):
        from scipy.optimize import minimize
        r = minimize(obj, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": self.max_iter})
        return r.x, r.fun


@register("surrogate", "gp", "Gaussian process, ARD Matern-5/2 + white noise (calibrated sd)", has_std=True)
class GP(Surrogate):
    supports_std = True

    def _fit(self, U, y):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
        nu = self.params.get("nu", 2.5)
        ard = self.params.get("ard", True)
        ls = np.ones(U.shape[1]) * 0.5 if ard else 0.5
        k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(ls, (1e-2, 1e3), nu=nu) + \
            WhiteKernel(1e-3, (1e-8, 1.0))
        hi_dim = U.shape[1] > 20      # ARD in many dimensions: cap the optimiser (cost ~ n^2 d per step)
        restarts = self.params.get("restarts", 0 if hi_dim else 2)
        max_iter = self.params.get("max_iter", 60 if hi_dim else 15000)

        self.m = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=restarts, optimizer=_LBFGS(max_iter),
                                          random_state=self.seed)
        self.m.fit(U, y)
        self.noise_var = self.m.kernel_.k2.noise_level

    def _predict(self, U, s):
        if not s:
            return self.m.predict(U), None
        mu, sd = self.m.predict(U, return_std=True)
        latent = np.sqrt(np.maximum(sd ** 2 - self.noise_var, 0.0))
        return mu, latent

    @property
    def length_scales(self):
        return np.atleast_1d(self.m.kernel_.k1.k2.length_scale)


# --------------------------------------------------------------------------- neural network ensemble
@register("surrogate", "dnn", "Ensemble of multilayer perceptrons (sd = ensemble spread)", has_std=True)
class DNN(Surrogate):
    supports_std = True

    def _fit(self, U, y):
        from sklearn.neural_network import MLPRegressor
        n = self.params.get("members", 5)
        hid = tuple(self.params.get("hidden", (64, 64)))
        self.members = []
        for i in range(n):
            m = MLPRegressor(hidden_layer_sizes=hid, activation=self.params.get("activation", "tanh"), solver="lbfgs",
                             alpha=self.params.get("alpha", 1e-3), max_iter=self.params.get("max_iter", 4000),
                             random_state=self.seed + i)
            m.fit(U, y)
            self.members.append(m)

    def _predict(self, U, s):
        P = np.array([m.predict(U) for m in self.members])
        return P.mean(0), (P.std(0) if s else None)


# --------------------------------------------------------------------------- tree models
@register("surrogate", "gbt", "Gradient-boosted regression trees (no uncertainty)")
class GBT(Surrogate):
    def _fit(self, U, y):
        from sklearn.ensemble import GradientBoostingRegressor
        p = dict(n_estimators=500, learning_rate=0.05, max_depth=3, subsample=0.8)
        p.update({k: v for k, v in self.params.items() if k in ("n_estimators", "learning_rate", "max_depth", "subsample")})
        self.m = GradientBoostingRegressor(random_state=self.seed, **p).fit(U, y)

    def _predict(self, U, s):
        return self.m.predict(U), None


class _Forest(Surrogate):
    supports_std = True
    cls = None

    def _fit(self, U, y):
        p = dict(n_estimators=500, min_samples_leaf=2, max_features=1.0)
        p.update({k: v for k, v in self.params.items() if k in ("n_estimators", "min_samples_leaf", "max_features", "max_depth")})
        self.m = self.cls(random_state=self.seed, n_jobs=-1, **p).fit(U, y)

    def _predict(self, U, s):
        if not s:
            return self.m.predict(U), None
        P = np.array([t.predict(U) for t in self.m.estimators_])
        return P.mean(0), P.std(0)


@register("surrogate", "random_forest", "Random forest (sd = tree spread)", has_std=True)
class RandomForest(_Forest):
    from sklearn.ensemble import RandomForestRegressor as cls  # noqa: N813


@register("surrogate", "extra_trees", "Extremely randomised trees (sd = tree spread)", has_std=True)
class ExtraTrees(_Forest):
    from sklearn.ensemble import ExtraTreesRegressor as cls  # noqa: N813


# --------------------------------------------------------------------------- polynomial chaos
def legendre_1d(x, p):
    """Orthonormal Legendre polynomials on [-1, 1] w.r.t. the uniform density, degree 0..p."""
    from numpy.polynomial import legendre as L
    out = np.empty((len(x), p + 1))
    for k in range(p + 1):
        c = np.zeros(k + 1)
        c[k] = 1
        out[:, k] = L.legval(x, c) * np.sqrt(2 * k + 1)
    return out


def multi_indices(d, p, q=1.0, max_inter=2):
    """Total-degree (hyperbolic q-norm) multi-index set with at most `max_inter` active dims."""
    idx = [np.zeros(d, int)]
    for deg in range(1, p + 1):
        for act in range(1, min(max_inter, deg) + 1):
            for dims in itertools.combinations(range(d), act):
                for parts in itertools.product(range(1, deg + 1), repeat=act):
                    if sum(parts) != deg:
                        continue
                    if (np.sum(np.array(parts, float) ** q)) ** (1 / q) > p + 1e-9:
                        continue
                    a = np.zeros(d, int)
                    a[list(dims)] = parts
                    idx.append(a)
    return np.array(idx)


@register("surrogate", "pce", "Sparse polynomial chaos (Legendre, LassoLars-CV); analytic Sobol indices", has_std=True)
class PCE(Surrogate):
    supports_std = True

    def _basis(self, U):
        x = 2 * U - 1
        P = [legendre_1d(x[:, j], self.p) for j in range(U.shape[1])]
        out = np.ones((len(U), len(self.A)))
        for t, a in enumerate(self.A):
            for j in np.nonzero(a)[0]:
                out[:, t] *= P[j][:, a[j]]
        return out

    def _fit(self, U, y):
        from sklearn.linear_model import LassoLarsCV, LinearRegression
        d, n = U.shape[1], len(U)
        self.p = self.params.get("degree")
        if self.p is None:  # largest degree whose basis is <= 3 n terms
            self.p = 1
            while self.p < 5 and len(multi_indices(d, self.p + 1, max_inter=self.params.get("max_inter", 2))) <= 3 * n:
                self.p += 1
        self.A = multi_indices(d, self.p, q=self.params.get("q", 1.0), max_inter=self.params.get("max_inter", 2))
        Psi = self._basis(U)
        las = LassoLarsCV(fit_intercept=False, cv=min(5, n // 4), max_n_alphas=200).fit(Psi, y)
        keep = np.flatnonzero(np.abs(las.coef_) > 1e-12)
        keep = np.union1d(keep, [0])
        ols = LinearRegression(fit_intercept=False).fit(Psi[:, keep], y)  # de-biased refit
        self.coef = np.zeros(len(self.A))
        self.coef[keep] = ols.coef_
        r = y - Psi @ self.coef
        self.res_sd = np.sqrt(np.sum(r ** 2) / max(n - len(keep), 1))
        self.A, self.coef = self.A[keep], self.coef[keep]
        self.n_terms = len(keep)

    def _predict(self, U, s):
        return self._basis(U) @ self.coef, (np.full(len(U), self.res_sd) if s else None)

    def sobol(self):
        """First-order and total Sobol indices from the PCE coefficients."""
        c2 = self.coef ** 2
        nz = np.any(self.A > 0, axis=1)
        V = c2[nz].sum() or 1.0
        S1 = np.array([c2[(self.A[:, j] > 0) & (np.count_nonzero(self.A, 1) == 1)].sum() / V for j in range(self.d)])
        ST = np.array([c2[self.A[:, j] > 0].sum() / V for j in range(self.d)])
        return S1, ST


# --------------------------------------------------------------------------- kernel & linear models
@register("surrogate", "svr", "Support-vector regression, RBF kernel (CV-tuned C, gamma)")
class SVR_(Surrogate):
    def _fit(self, U, y):
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVR
        g = GridSearchCV(SVR(kernel="rbf", epsilon=0.01),
                         {"C": [1, 10, 100, 1000], "gamma": [0.05, 0.2, 0.5, 1.0, 2.0]}, cv=min(5, len(U) // 5))
        self.m = g.fit(U, y).best_estimator_

    def _predict(self, U, s):
        return self.m.predict(U), None


@register("surrogate", "krr", "Kernel ridge regression, RBF kernel (CV-tuned)")
class KRR(Surrogate):
    def _fit(self, U, y):
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.model_selection import GridSearchCV
        g = GridSearchCV(KernelRidge(kernel="rbf"),
                         {"alpha": [1e-4, 1e-3, 1e-2, 1e-1], "gamma": [0.05, 0.2, 0.5, 1.0, 2.0]}, cv=min(5, len(U) // 5))
        self.m = g.fit(U, y).best_estimator_

    def _predict(self, U, s):
        return self.m.predict(U), None


@register("surrogate", "poly_ridge", "Quadratic (or degree-n) polynomial + Bayesian ridge", has_std=True)
class PolyRidge(Surrogate):
    supports_std = True

    def _fit(self, U, y):
        from sklearn.linear_model import BayesianRidge
        from sklearn.preprocessing import PolynomialFeatures
        self.pf = PolynomialFeatures(self.params.get("degree", 2), include_bias=False)
        self.m = BayesianRidge().fit(self.pf.fit_transform(U), y)

    def _predict(self, U, s):
        if not s:
            return self.m.predict(self.pf.transform(U)), None
        mu, sd = self.m.predict(self.pf.transform(U), return_std=True)
        return mu, np.sqrt(np.maximum(sd ** 2 - 1 / self.m.alpha_, 0))


@register("surrogate", "linear", "Bayesian linear regression", has_std=True)
class Linear(Surrogate):
    supports_std = True

    def _fit(self, U, y):
        from sklearn.linear_model import BayesianRidge
        self.m = BayesianRidge().fit(U, y)

    def _predict(self, U, s):
        if not s:
            return self.m.predict(U), None
        mu, sd = self.m.predict(U, return_std=True)
        return mu, np.sqrt(np.maximum(sd ** 2 - 1 / self.m.alpha_, 0))


@register("surrogate", "knn", "k-nearest-neighbours (distance weighted)")
class KNN(Surrogate):
    def _fit(self, U, y):
        from sklearn.neighbors import KNeighborsRegressor
        self.m = KNeighborsRegressor(self.params.get("k", 5), weights="distance").fit(U, y)

    def _predict(self, U, s):
        return self.m.predict(U), None


# --------------------------------------------------------------------------- optional third-party
try:  # pragma: no cover - only if installed
    import xgboost as _xgb

    @register("surrogate", "xgboost", "XGBoost gradient boosting (optional dependency)")
    class XGB(Surrogate):
        def _fit(self, U, y):
            self.m = _xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=4, subsample=0.8,
                                       random_state=self.seed, **self.params).fit(U, y)

        def _predict(self, U, s):
            return self.m.predict(U), None
except Exception:
    pass


# --------------------------------------------------------------------------- ensemble of others
@register("surrogate", "ensemble", "Average of several surrogates; sd combines members' sd and spread", has_std=True)
class Ensemble(Surrogate):
    supports_std = True
    standardize_y = False

    def _fit(self, U, y):
        from ..registry import create
        self.members = [create("surrogate", s, seed=self.seed).fit(U, y) for s in self.params.get("members", ["gp", "dnn"])]

    def _predict(self, U, s):
        R = [m.predict(U, return_std=True) for m in self.members]
        M = np.array([r[0] for r in R])
        mu = M.mean(0)
        if not s:
            return mu, None
        al = np.nanmean(np.array([np.nan_to_num(r[1]) ** 2 for r in R]), 0)
        return mu, np.sqrt(al + M.var(0))
