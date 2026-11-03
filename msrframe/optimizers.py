"""Box-constrained optimisers on the unit cube, registered as ``optimizer`` plug-ins.

Signature: ``opt = create("optimizer", spec); u, f, trace = opt.minimize(cost, dim)``
where ``trace`` is the best-so-far cost against the number of cost evaluations.
"""
from __future__ import annotations

import numpy as np

from .registry import create, get, register


def create_sampler(name):
    return get("sampler", name)


class Optimizer:
    def __init__(self, seed=0, budget=6000, **params):
        self.seed, self.budget, self.params = seed, budget, params

    def _wrap(self, cost):
        trace, best = [], [np.inf, None]

        def f(u):
            u = np.clip(np.asarray(u, float), 0, 1)
            c = float(cost(u))
            if c < best[0]:
                best[0], best[1] = c, u.copy()
            trace.append(best[0])
            return c
        return f, trace, best


@register("optimizer", "differential_evolution", "Differential evolution, best/1/bin with dither (scipy)")
class DE(Optimizer):
    def minimize(self, cost, dim):
        from scipy.optimize import differential_evolution
        f, trace, best = self._wrap(cost)
        pop = self.params.get("popsize", max(6, min(15, 60 // max(dim, 1))))
        maxiter = max(1, self.budget // (pop * dim) - 1)
        differential_evolution(f, [(0, 1)] * dim, popsize=pop, maxiter=maxiter, tol=1e-8, seed=self.seed,
                               mutation=(0.5, 1.0), recombination=0.7, polish=False, init="sobol")
        return best[1], best[0], np.array(trace)


@register("optimizer", "dual_annealing", "Generalised simulated annealing (scipy dual_annealing)")
class DualAnnealing(Optimizer):
    def minimize(self, cost, dim):
        from scipy.optimize import dual_annealing
        f, trace, best = self._wrap(cost)
        dual_annealing(f, [(0, 1)] * dim, maxfun=self.budget, seed=self.seed, no_local_search=False)
        return best[1], best[0], np.array(trace)


@register("optimizer", "nelder_mead", "Multi-start Nelder-Mead from the best of a Sobol sample")
class NelderMead(Optimizer):
    def minimize(self, cost, dim):
        from scipy.optimize import minimize
        f, trace, best = self._wrap(cost)
        n0 = min(self.budget // 4, 256)
        U0 = create_sampler("sobol")(n0, dim, seed=self.seed)
        c0 = np.array([f(u) for u in U0])
        starts = U0[np.argsort(c0)[: self.params.get("starts", 4)]]
        per = max(50, (self.budget - n0) // len(starts))
        for s in starts:
            minimize(f, s, method="Nelder-Mead", options=dict(maxfev=per, xatol=1e-5, fatol=1e-6))
        return best[1], best[0], np.array(trace)


@register("optimizer", "random_search", "Pure Sobol random search (baseline)")
class RandomSearch(Optimizer):
    def minimize(self, cost, dim):
        f, trace, best = self._wrap(cost)
        for u in create_sampler("sobol")(self.budget, dim, seed=self.seed):
            f(u)
        return best[1], best[0], np.array(trace)


@register("optimizer", "pso", "Particle-swarm optimisation (inertia 0.72, c1=c2=1.49)")
class PSO(Optimizer):
    def minimize(self, cost, dim):
        f, trace, best = self._wrap(cost)
        rng = np.random.default_rng(self.seed)
        n = self.params.get("particles", 30)
        X = rng.random((n, dim))
        V = 0.1 * (rng.random((n, dim)) - 0.5)
        P, Pc = X.copy(), np.array([f(x) for x in X])
        g = P[np.argmin(Pc)].copy()
        for _ in range(max(1, self.budget // n - 1)):
            r1, r2 = rng.random((n, dim)), rng.random((n, dim))
            V = 0.72 * V + 1.49 * r1 * (P - X) + 1.49 * r2 * (g - X)
            X = np.clip(X + V, 0, 1)
            c = np.array([f(x) for x in X])
            imp = c < Pc
            P[imp], Pc[imp] = X[imp], c[imp]
            g = P[np.argmin(Pc)].copy()
        return best[1], best[0], np.array(trace)


@register("optimizer", "bayesian", "Bayesian optimisation: GP + expected improvement (for expensive costs)")
class Bayesian(Optimizer):
    def minimize(self, cost, dim):
        from scipy.stats import norm
        f, trace, best = self._wrap(cost)
        budget = min(self.budget, self.params.get("max_evals", 120))
        n0 = max(2 * dim + 2, 10)
        U = list(create_sampler("lhs")(n0, dim, seed=self.seed))
        C = [f(u) for u in U]
        rng = np.random.default_rng(self.seed)
        while len(C) < budget:
            m = create("surrogate", {"name": "gp", "restarts": 0}).fit(np.array(U), np.array(C))
            cand = rng.random((2000, dim))
            mu, sd = m.predict(cand, return_std=True)
            sd = np.maximum(sd, 1e-9)
            z = (min(C) - mu) / sd
            ei = (min(C) - mu) * norm.cdf(z) + sd * norm.pdf(z)
            u = cand[int(np.argmax(ei))]
            U.append(u)
            C.append(f(u))
        return best[1], best[0], np.array(trace)
