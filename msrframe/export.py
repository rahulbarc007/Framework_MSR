"""Export trained msrframe models to compact JSON for in-browser (JavaScript) inference.

Every built-in surrogate is exported *exactly* (the browser reproduces ``model.predict`` to
round-off): Gaussian process, MLP ensembles, gradient-boosted trees, random / extra trees,
polynomial chaos, polynomial & linear Bayesian ridge, k-NN, SVR, kernel ridge and ensembles.
Trajectory models (``pca``, ``time_input``, ``per_time``) are exported around their base
models.

Anything else - a custom plug-in, xgboost, a new kernel - is *distilled*: the model is
sampled on a Latin hypercube and a small neural network is fitted to it (and to log sd if
the model has an uncertainty). The distillation error is measured and shown in the
dashboard. A plug-in can avoid distillation by providing ``to_json()`` returning one of the
formats below.

Large arrays are stored as base64 little-endian typed arrays::

    {"b64": "f8" | "f4" | "i4" | "i2", "shape": [...], "data": "..."}

:func:`predict_scalar` / :func:`predict_trajectory` are numpy reference implementations of
exactly what the JavaScript runtime computes; the tests check them against the models.
"""
from __future__ import annotations

import base64
import json

import numpy as np

_DT = {"f8": "<f8", "f4": "<f4", "i4": "<i4", "i2": "<i2"}


# --------------------------------------------------------------------------- encoding
def enc(a, dtype="f8"):
    a = np.ascontiguousarray(np.asarray(a), dtype=_DT[dtype])
    return {"b64": dtype, "shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode("ascii")}


def dec(o):
    """Inverse of :func:`enc`, applied recursively (used by the reference evaluator)."""
    if isinstance(o, dict):
        if "b64" in o and "data" in o:
            a = np.frombuffer(base64.b64decode(o["data"]), dtype=_DT[o["b64"]])
            return a.reshape(o["shape"]) if o["shape"] else a
        return {k: dec(v) for k, v in o.items()}
    if isinstance(o, list):
        return [dec(v) for v in o]
    return o


def f32_floor(t):
    """Largest float32 <= t. sklearn compares float32(x) <= float64(threshold); for a float32
    x that is the same as comparing with this value, so f4 storage stays exact."""
    t = np.asarray(t, float)
    t32 = t.astype(np.float32)
    up = t32.astype(float) > t
    t32[up] = np.nextafter(t32[up], np.float32(-np.inf))
    return t32


# --------------------------------------------------------------------------- scalar models
def _common(m, kind, std):
    return {"type": kind, "ym": float(getattr(m, "ym", 0.0)), "ys": float(getattr(m, "ys", 1.0)),
            "noise": float(getattr(m, "noise_sd", 0.0) or 0.0), "std": bool(std)}


def _gp(m):
    g = m.m
    k = g.kernel_
    c = float(k.k1.k1.constant_value)
    mat = k.k1.k2
    X = g.X_train_
    ls = np.broadcast_to(np.atleast_1d(mat.length_scale).astype(float), (X.shape[1],))
    n = len(X)
    L = g.L_
    packed = np.concatenate([L[i, : i + 1] for i in range(n)])
    out = _common(m, "gp", True)
    nu = float(mat.nu) if np.isfinite(mat.nu) else 1e9           # JSON has no Infinity; 1e9 = squared exponential
    out.update(X=enc(X), alpha=enc(np.ravel(g.alpha_)), L=enc(packed), c=c, ls=enc(ls), nu=nu,
               noise_var=float(k.k2.noise_level), n=n, d=int(X.shape[1]))
    return out


def _mlp_member(mlp):
    return {"act": mlp.activation,
            "layers": [{"W": enc(W, "f4"), "b": enc(b, "f4"), "nin": int(W.shape[0]), "nout": int(W.shape[1])}
                       for W, b in zip(mlp.coefs_, mlp.intercepts_)]}


def _dnn(m):
    out = _common(m, "mlp", True)
    out["members"] = [_mlp_member(mm) for mm in m.members]
    out["spread"] = True          # sd = spread of the members
    return out


def _trees(estimators):
    roots, L, R, F, T, V = [], [], [], [], [], []
    off = 0
    for est in estimators:
        t = est.tree_
        roots.append(off)
        L.append(t.children_left)
        R.append(t.children_right)
        F.append(np.where(t.children_left == -1, -1, t.feature))
        T.append(f32_floor(np.where(t.children_left == -1, 0.0, t.threshold)))
        V.append(t.value.reshape(t.node_count, -1)[:, 0])
        off += t.node_count
    return dict(roots=enc(roots, "i4"), left=enc(np.concatenate(L), "i4"), right=enc(np.concatenate(R), "i4"),
                feat=enc(np.concatenate(F), "i2"), thr=enc(np.concatenate(T), "f4"), val=enc(np.concatenate(V), "f4"),
                ntrees=len(estimators))


def _gbt(m):
    g = m.m
    out = _common(m, "trees", False)
    out.update(kind="gbt", init=float(g._raw_predict_init(np.zeros((1, g.n_features_in_)))[0, 0]),
               lr=float(g.learning_rate), **_trees([e[0] for e in g.estimators_]))
    return out


def _forest(m):
    out = _common(m, "trees", True)
    out.update(kind="forest", **_trees(m.m.estimators_))
    return out


def _pce(m):
    out = _common(m, "pce", True)
    out.update(A=enc(m.A, "i2"), coef=enc(m.coef), p=int(m.p), res_sd=float(m.res_sd), nt=int(len(m.A)), d=int(m.A.shape[1]))
    return out


def _bayes_ridge(m, powers):
    br = m.m
    out = _common(m, "poly", True)
    out.update(powers=enc(powers, "i2"), nf=int(powers.shape[0]), d=int(powers.shape[1]), coef=enc(br.coef_),
               intercept=float(br.intercept_), sigma=enc(br.sigma_), offset=enc(br.X_offset_), inv_alpha=float(1.0 / br.alpha_))
    return out


def _poly_ridge(m):
    return _bayes_ridge(m, np.asarray(m.pf.powers_))


def _linear(m):
    return _bayes_ridge(m, np.eye(m.d, dtype=int))


def _knn(m):
    k = m.m
    out = _common(m, "knn", False)
    out.update(X=enc(k._fit_X), y=enc(np.ravel(k._y)), k=int(k.n_neighbors), n=int(k._fit_X.shape[0]), d=int(k._fit_X.shape[1]))
    return out


def _svr(m):
    s = m.m
    out = _common(m, "rbf_expansion", False)
    out.update(X=enc(s.support_vectors_), w=enc(np.ravel(s.dual_coef_)), b=float(np.ravel(s.intercept_)[0]),
               gamma=float(s._gamma), n=int(len(s.support_vectors_)), d=int(s.support_vectors_.shape[1]))
    return out


def _krr(m):
    s = m.m
    g = s.gamma if s.gamma is not None else 1.0 / s.X_fit_.shape[1]
    out = _common(m, "rbf_expansion", False)
    out.update(X=enc(s.X_fit_), w=enc(np.ravel(s.dual_coef_)), b=0.0, gamma=float(g), n=int(len(s.X_fit_)),
               d=int(s.X_fit_.shape[1]))
    return out


def _ensemble(m):
    out = _common(m, "ensemble", True)
    out["members"] = [export_scalar(mm, check=False)[0] for mm in m.members]
    return out


def _exporters():
    from .models import scalar as S
    table = [(S.GP, _gp), (S.DNN, _dnn), (S.GBT, _gbt), (S.RandomForest, _forest), (S.ExtraTrees, _forest),
             (S.PCE, _pce), (S.PolyRidge, _poly_ridge), (S.Linear, _linear), (S.KNN, _knn), (S.SVR_, _svr),
             (S.KRR, _krr), (S.Ensemble, _ensemble)]
    return table


def _student(U, y, Ut, yt, seed, good=2e-3, absolute=False):
    """Fit an exportable 'student' to (U, y): sparse PCE first (smooth models, tiny, fast);
    an MLP only if the PCE misses by more than `good` (x the output spread unless `absolute`)
    on held-out points. Returns (spec, held-out rms error)."""
    from .models.scalar import DNN, PCE
    spread = 1.0 if absolute else (float(np.std(yt)) or 1.0)
    best = None
    for deg in (3, 4):
        try:
            m = PCE(seed=seed, degree=deg, max_inter=2).fit(U, y)
        except Exception:
            continue
        e = float(np.sqrt(np.mean((m.predict(Ut) - yt) ** 2)))
        if best is None or e < best[1]:
            best = (_pce(m), e)
    if best is None or best[1] > good * spread:
        net = DNN(seed=seed, members=1, hidden=(48, 48), max_iter=1000).fit(U, y)
        e = float(np.sqrt(np.mean((net.predict(Ut) - yt) ** 2)))
        if best is None or e < best[1]:
            spec = _common(net, "mlp", False)
            spec.update(members=[_mlp_member(net.members[0])], spread=False)
            best = (spec, e)
    best[0]["std"] = False
    return best


def distill_scalar(m, d, n=2500, seed=0):
    """Re-learn a model that cannot be exported exactly (custom plug-in, xgboost, ...) from
    its own predictions on a Latin hypercube; its sd, if any, is learned as log(sd)."""
    from .design import lhs
    U, Ut = lhs(n, d, seed=seed), lhs(600, d, seed=seed + 1)
    mu, sd = m.predict(U, return_std=True)
    mut, sdt = m.predict(Ut, return_std=True)
    spec, err = _student(U, mu, Ut, mut, seed)
    spec["distilled"] = {"from": type(m).__name__, "n_samples": n, "holdout_rms": err}
    if np.isfinite(sd).all() and np.isfinite(sdt).all():
        floor = 1e-9 * (float(np.std(mu)) or 1.0)
        # an error bar needs ~tens of % accuracy: accept the PCE unless log(sd) is off by > 0.5
        sspec, _ = _student(U, np.log(np.maximum(sd, floor)), Ut, np.log(np.maximum(sdt, floor)), seed + 7,
                            good=0.5, absolute=True)
        spec["sd_model"] = sspec        # predicts log(sd) in output units
        spec["std"] = True
    spec["noise"] = float(getattr(m, "noise_sd", 0.0) or 0.0)
    return spec


def export_scalar(m, d=None, check=True, tol=1e-6, seed=0):
    """Return (spec, info). info = {method: exact|plugin|distilled, max_err, size_kb}."""
    d = d or getattr(m, "d", None)
    spec, method = None, "exact"
    if hasattr(m, "to_json"):
        spec, method = m.to_json(), "plugin"
    else:
        for cls, fn in _exporters():
            if type(m) is cls:
                try:
                    spec = fn(m)
                except Exception as e:  # unexpected internals (e.g. a different kernel)
                    print(f"  export: {type(m).__name__} not exportable exactly ({e}); distilling")
                break
    info = {"method": method}
    if spec is not None and check and d:
        err, scale = _check(m, spec, d, seed)
        info["max_err"], info["rel_err"] = err, err / scale
        if not np.isfinite(err) or err > tol * 100 * scale:     # exact exports agree to ~1e-8 of the output spread
            print(f"  export: {type(m).__name__} JSON deviates by {err:.3g}; distilling instead")
            spec = None
    if spec is None:
        if not d:
            raise ValueError("cannot distil a model without knowing its input dimension")
        spec = distill_scalar(m, d, seed=seed)
        err, scale = _check(m, spec, d, seed)
        info = {"method": "distilled", "max_err": err, "rel_err": err / scale}
    info["size_kb"] = len(json.dumps(spec)) / 1024
    return spec, info


def _check(m, spec, d, seed=0):
    """Max |JSON prediction - model prediction| on random points, and the output spread."""
    U = np.random.default_rng(seed + 17).random((300, d))
    mu = m.predict(U)
    pm, _ = predict_scalar(dec(spec), U)
    scale = float(np.std(mu)) or float(abs(np.mean(mu))) or 1.0
    return float(np.max(np.abs(pm - mu))), scale


# --------------------------------------------------------------------------- trajectory models
def export_trajectory(model, d, seed=0):
    """Export a fitted trajectory model; returns (spec, info)."""
    from .models import trajectory as TR
    infos = []

    def sc(m):
        s, i = export_scalar(m, d=m.d if hasattr(m, "d") else None, seed=seed)
        infos.append(i)
        return s

    t = type(model)
    if t is TR.PCATrajectory:
        spec = {"type": "pca", "times": enc(model.times), "outputs": {
            o: {"mu": enc(O["mu"]), "sc": float(O["sc"]), "V": enc(O["V"]), "trunc": float(O["trunc"]), "nc": int(len(O["V"])),
                "models": [sc(m) for m in O["models"]]} for o, O in model.out.items()}}
    elif t is TR.TimeInputTrajectory:
        spec = {"type": "time_input", "times": enc(model.times), "tn": enc(model.tn), "outputs": {
            o: {"mu": enc(O["mu"]), "sc": float(O["sc"]), "model": sc(O["m"])} for o, O in model.out.items()}}
    elif t is TR.PerTimeTrajectory:
        spec = {"type": "per_time", "times": enc(model.times), "outputs": {o: [sc(m) for m in ms] for o, ms in model.out.items()}}
    else:  # unknown trajectory plug-in: re-learn it as PCA + MLP on sampled trajectories
        from .design import lhs
        U = lhs(1500, d, seed=seed)
        P = model.predict(U)
        Y = {o: P[o][0] for o in P}
        surrogate = TR.PCATrajectory(base={"name": "dnn", "members": 1, "hidden": (64, 64)}, seed=seed).fit(U, Y, model.times)
        spec, sub = export_trajectory(surrogate, d, seed)
        spec["distilled"] = True
        return spec, {"method": "distilled", "parts": sub.get("parts", 0), "size_kb": len(json.dumps(spec)) / 1024}
    methods = {i["method"] for i in infos}
    return spec, {"method": "exact" if methods <= {"exact", "plugin"} else "partly distilled", "parts": len(infos),
                  "rel_err": max((i.get("rel_err", 0.0) for i in infos), default=0.0), "size_kb": len(json.dumps(spec)) / 1024}


def export_space(space):
    return [dict(key=p.key, low=float(p.low), high=float(p.high), nominal=float(p.nominal), log=bool(p.log),
                 name=p.name, unit=p.unit or "", cls=p.cls or "") for p in space.params]


# --------------------------------------------------------------------------- numpy reference of the JS runtime
def _act(name, z):
    return {"tanh": np.tanh, "relu": lambda v: np.maximum(v, 0), "logistic": lambda v: 1 / (1 + np.exp(-v)),
            "identity": lambda v: v}[name](z)


def _mlp_forward(mem, U):
    h = U
    for i, ly in enumerate(mem["layers"]):
        h = h @ ly["W"].astype(float).reshape(ly["nin"], ly["nout"]) + ly["b"].astype(float)
        if i < len(mem["layers"]) - 1:
            h = _act(mem["act"], h)
    return h[:, 0]


def _matern(r, nu):
    if nu == 0.5:
        return np.exp(-r)
    if nu == 1.5:
        a = np.sqrt(3) * r
        return (1 + a) * np.exp(-a)
    if nu == 2.5:
        a = np.sqrt(5) * r
        return (1 + a + 5 * r ** 2 / 3) * np.exp(-a)
    return np.exp(-0.5 * r ** 2)


def _legendre_norm(x, p):
    P = np.zeros((len(x), p + 1))
    P[:, 0] = 1
    if p >= 1:
        P[:, 1] = x
    for k in range(1, p):
        P[:, k + 1] = ((2 * k + 1) * x * P[:, k] - k * P[:, k - 1]) / (k + 1)
    return P * np.sqrt(2 * np.arange(p + 1) + 1)


def _trees_eval(s, U):
    roots, left, right, feat, thr, val = (s[k] for k in ("roots", "left", "right", "feat", "thr", "val"))
    U32 = U.astype(np.float32)
    out = np.zeros((len(U), len(roots)))
    for ti, r in enumerate(roots):
        for i in range(len(U)):
            nd = r
            while left[nd] != -1:
                nd = r + (left[nd] if U32[i, feat[nd]] <= thr[nd] else right[nd])
            out[i, ti] = val[nd]
    return out


def predict_scalar(s, U):
    """Reference implementation of the JS ``predictScalar`` -> (mu, latent sd) in output units."""
    U = np.atleast_2d(np.asarray(U, float))
    t, n = s["type"], len(U)
    sd = None
    if t == "gp":
        X, ls = s["X"], s["ls"]
        r = np.sqrt((((U[:, None, :] - X[None, :, :]) / ls) ** 2).sum(-1))
        K = s["c"] * _matern(r, s["nu"])
        mu = K @ s["alpha"]
        L = np.zeros((s["n"], s["n"]))
        L[np.tril_indices(s["n"])] = s["L"]
        from scipy.linalg import solve_triangular
        V = solve_triangular(L, K.T, lower=True)
        tot = np.maximum(s["c"] + s["noise_var"] - (V ** 2).sum(0), 0)
        sd = np.sqrt(np.maximum(tot - s["noise_var"], 0))
    elif t == "mlp":
        P = np.array([_mlp_forward(m, U) for m in s["members"]])
        mu = P.mean(0)
        if s.get("spread"):
            sd = P.std(0)
    elif t == "trees":
        P = _trees_eval(s, U)
        if s["kind"] == "gbt":
            mu = s["init"] + s["lr"] * P.sum(1)
        else:
            mu, sd = P.mean(1), P.std(1)
    elif t == "pce":
        Ps = [_legendre_norm(2 * U[:, j] - 1, s["p"]) for j in range(s["d"])]
        B = np.ones((n, s["nt"]))
        for ti, a in enumerate(s["A"]):
            for j in np.nonzero(a)[0]:
                B[:, ti] *= Ps[j][:, a[j]]
        mu = B @ s["coef"]
        sd = np.full(n, s["res_sd"])
    elif t == "poly":
        F = np.prod(U[:, None, :] ** s["powers"][None, :, :], axis=2)
        mu = F @ s["coef"] + s["intercept"]
        Fc = F - s["offset"]
        tot = np.sqrt((Fc @ s["sigma"] * Fc).sum(1) + s["inv_alpha"])
        sd = np.sqrt(np.maximum(tot ** 2 - s["inv_alpha"], 0))
    elif t == "knn":
        D = np.sqrt(((U[:, None, :] - s["X"][None, :, :]) ** 2).sum(-1))
        mu = np.empty(n)
        for i in range(n):
            idx = np.argsort(D[i], kind="stable")[: s["k"]]
            d = D[i, idx]
            w = (d == 0).astype(float) if np.any(d == 0) else 1 / d
            mu[i] = np.sum(w * s["y"][idx]) / np.sum(w)
    elif t == "rbf_expansion":
        D2 = ((U[:, None, :] - s["X"][None, :, :]) ** 2).sum(-1)
        mu = np.exp(-s["gamma"] * D2) @ s["w"] + s["b"]
    elif t == "ensemble":
        R = [predict_scalar(m, U) for m in s["members"]]
        M = np.array([r[0] for r in R])
        al = np.nanmean(np.array([np.nan_to_num(r[1]) ** 2 for r in R]), 0)
        return M.mean(0), np.sqrt(al + M.var(0))
    else:
        raise ValueError(f"unknown exported type {t}")
    mu = mu * s["ys"] + s["ym"]
    if s.get("sd_model"):                      # distilled model: sd learned separately as log(sd)
        return mu, np.exp(predict_scalar(s["sd_model"], U)[0])
    sd = np.full(n, np.nan) if (sd is None or not s["std"]) else sd * s["ys"]
    return mu, sd


def predict_trajectory(s, U, noise=False):
    """Reference implementation of the JS ``predictTrajectory`` -> {out: (mu (n,T), sd (n,T))}."""
    U = np.atleast_2d(np.asarray(U, float))
    res = {}
    if s["type"] == "pca":
        for o, O in s["outputs"].items():
            T = len(O["mu"])
            mu = np.zeros((len(U), T))
            var = np.full_like(mu, O["trunc"])
            V = O["V"].reshape(O["nc"], T)
            for m, v in zip(O["models"], V):
                a, sd = predict_scalar(m, U)
                mu += a[:, None] * v[None]
                var += (np.nan_to_num(sd)[:, None] ** 2 + (m["noise"] ** 2 if noise else 0)) * v[None] ** 2
            has = all(m["std"] for m in O["models"])
            res[o] = (O["mu"] + mu * O["sc"], np.sqrt(var) * O["sc"] if has else np.full_like(mu, np.nan))
    elif s["type"] == "time_input":
        tn = s["tn"]
        n, T = len(U), len(tn)
        UU = np.hstack([np.repeat(U, T, 0), np.tile(tn, n)[:, None]])
        for o, O in s["outputs"].items():
            a, sd = predict_scalar(O["model"], UU)
            if noise:
                sd = np.sqrt(sd ** 2 + O["model"]["noise"] ** 2)
            res[o] = (O["mu"] + a.reshape(n, T) * O["sc"], sd.reshape(n, T) * O["sc"])
    elif s["type"] == "per_time":
        for o, ms in s["outputs"].items():
            P = [predict_scalar(m, U) for m in ms]
            nz = np.array([m["noise"] if noise else 0.0 for m in ms])
            res[o] = (np.array([p[0] for p in P]).T, np.sqrt(np.array([p[1] for p in P]).T ** 2 + nz ** 2))
    else:
        raise ValueError(s["type"])
    return res
