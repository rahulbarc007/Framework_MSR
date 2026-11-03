"""Experimental designs (samplers) on the unit cube, registered as ``sampler`` plug-ins."""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", message=".*balance properties of Sobol.*")

from .registry import register


@register("sampler", "lhs", "Latin hypercube (scipy.qmc, optimised)")
def lhs(n, d, seed=0):
    from scipy.stats import qmc
    return qmc.LatinHypercube(d, seed=seed, optimization="random-cd" if d > 1 else None).random(n)


@register("sampler", "sobol", "Scrambled Sobol' low-discrepancy sequence")
def sobol(n, d, seed=0):
    from scipy.stats import qmc
    return qmc.Sobol(d, scramble=True, seed=seed).random(n)


@register("sampler", "halton", "Scrambled Halton sequence")
def halton(n, d, seed=0):
    from scipy.stats import qmc
    return qmc.Halton(d, scramble=True, seed=seed).random(n)


@register("sampler", "random", "Independent uniform random sampling")
def random(n, d, seed=0):
    return np.random.default_rng(seed).random((n, d))


@register("sampler", "morris", "Morris one-at-a-time trajectories (n = r (d+1) points)")
def morris(r, d, seed=0, levels=4):
    """Return (U, traj_id) for r trajectories of a p-level Morris design."""
    rng = np.random.default_rng(seed)
    delta = levels / (2 * (levels - 1))
    grid = np.arange(levels // 2) / (levels - 1)
    pts, ids = [], []
    for t in range(r):
        x = rng.choice(grid, d)
        order = rng.permutation(d)
        sign = rng.choice([-1, 1], d)
        traj = [x.copy()]
        for j in order:
            x = x.copy()
            step = delta * sign[j]
            if not 0 <= x[j] + step <= 1:
                step = -step
            x[j] += step
            traj.append(x)
        pts += traj
        ids += [t] * (d + 1)
    return np.array(pts), np.array(ids)


def maximin_select(cands, score, n, existing=None, alpha=1.0):
    """Greedy batch selection: high `score` but spread out (distance-penalised).

    Each pick maximises  score * (dmin / dmin_max)^alpha  where dmin is the distance to
    already chosen and existing points - avoids clustering the batch in one corner."""
    cands = np.asarray(cands)
    score = np.asarray(score, float)
    score = (score - score.min()) / (np.ptp(score) or 1.0) + 1e-6
    ref = np.empty((0, cands.shape[1])) if existing is None else np.asarray(existing)
    dmin = np.full(len(cands), np.inf) if len(ref) == 0 else \
        np.min(np.linalg.norm(cands[:, None, :] - ref[None, :, :], axis=2), axis=1)
    chosen = []
    for _ in range(n):
        dn = np.where(np.isinf(dmin), 1.0, dmin / (np.max(dmin[np.isfinite(dmin)]) if np.isfinite(dmin).any() else 1.0))
        j = int(np.argmax(score * dn ** alpha))
        chosen.append(j)
        dmin = np.minimum(dmin, np.linalg.norm(cands - cands[j], axis=1))
        dmin[j] = 0
    return np.array(chosen)
