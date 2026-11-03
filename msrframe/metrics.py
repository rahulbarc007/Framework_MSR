"""Validation metrics and cross-validation helpers (k_eff errors in pcm)."""
from __future__ import annotations

import time

import numpy as np
from sklearn.model_selection import KFold

from .registry import create

PCM = 1e5


def scalar_metrics(y, mu, sd=None, y_std=None, scale=PCM):
    e = (mu - y) * scale
    out = dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
               max_abs=float(np.max(np.abs(e))), bias=float(np.mean(e)),
               r2=float(1 - np.sum(e ** 2) / max(np.sum(((y - y.mean()) * scale) ** 2), 1e-300)))
    if y_std is not None and np.isfinite(y_std).any():
        nr = float(np.sqrt(np.nanmean((np.asarray(y_std) * scale) ** 2)))
        out["mc_noise_rms"] = nr
        out["rmse_noise_corrected"] = float(np.sqrt(max(out["rmse"] ** 2 - nr ** 2, 0)))
    if sd is not None and np.isfinite(sd).all():
        z = e / np.maximum(np.asarray(sd) * scale, 1e-300)   # sd = total predictive sd (incl. noise)
        out["coverage95"] = float(np.mean(np.abs(z) <= 1.96))
        out["z_std"] = float(np.std(z))
    return out


def cv_predict(spec, U, y, folds=5, seed=0, groups=None):
    """K-fold out-of-fold predictions (mu, total predictive sd) and the mean fit time."""
    mu, sd = np.empty(len(y)), np.empty(len(y))
    t = []
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(U):
        t0 = time.time()
        m = create("surrogate", spec, seed=seed).fit(U[tr], y[tr])
        t.append(time.time() - t0)
        mu[te], s = m.predict(U[te], return_std=True)
        sd[te] = np.sqrt(s ** 2 + m.noise_sd ** 2)   # predictive sd of a new OpenMC result
    return mu, sd, float(np.mean(t))
