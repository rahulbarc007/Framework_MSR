"""Dashboard: exported models reproduce the Python models, in numpy and in the browser runtime."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import msrframe  # noqa: E402,F401
from msrframe import create  # noqa: E402
from msrframe.export import dec, export_scalar, export_trajectory, predict_scalar, predict_trajectory  # noqa: E402
from msrframe.models.scalar import Surrogate  # noqa: E402

RUNTIME = ROOT / "msrframe" / "dashboard_assets" / "runtime.js"
SMALL = {"dnn": {"name": "dnn", "members": 2, "hidden": [16, 16], "max_iter": 400},
         "gbt": {"name": "gbt", "n_estimators": 60},
         "random_forest": {"name": "random_forest", "n_estimators": 40},
         "extra_trees": {"name": "extra_trees", "n_estimators": 40}}
SCALARS = ["gp", "dnn", "gbt", "random_forest", "extra_trees", "pce", "poly_ridge", "linear", "knn", "svr", "krr",
           {"name": "ensemble", "members": ["gp", "poly_ridge"]}]


def _data(n=60, d=3, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.random((n, d))
    y = 1.0 + 0.05 * np.sin(3 * U[:, 0]) + 0.02 * U[:, 1] ** 2 - 0.01 * U[:, 2] + 5e-4 * rng.standard_normal(n)
    return U, y


def _fit(spec, U, y):
    return create("surrogate", SMALL.get(spec, spec) if isinstance(spec, str) else spec, seed=0).fit(U, y)


def _label(spec):
    return spec if isinstance(spec, str) else spec["name"]


@pytest.mark.parametrize("spec", SCALARS, ids=_label)
def test_scalar_export_is_exact(spec):
    U, y = _data()
    m = _fit(spec, U, y)
    s, info = export_scalar(m, d=3)
    assert info["method"] == "exact"
    T = np.random.default_rng(1).random((50, 3))
    mu, sd = m.predict(T, return_std=True)
    rm, rs = predict_scalar(dec(json.loads(json.dumps(s))), T)
    assert np.max(np.abs(rm - mu)) < 1e-6 * np.std(y)
    assert np.isfinite(rs).all() == np.isfinite(sd).all()
    if np.isfinite(sd).all():
        assert np.max(np.abs(rs - sd)) < 1e-5 * np.std(y)


def _traj_data(n=30, T=6, seed=2):
    rng = np.random.default_rng(seed)
    U = rng.random((n, 3))
    t = np.linspace(0, 1, T)
    Y = {"keff": 1.0 + 0.03 * U[:, :1] - 0.02 * t[None] * (1 + U[:, 1:2]) + 1e-4 * rng.standard_normal((n, T)),
         "inv_kg": 1000 + 300 * U[:, 2:3] * t[None] + 50 * U[:, :1]}
    return U, Y, t


@pytest.mark.parametrize("spec", [{"name": "pca", "base": "gp"}, {"name": "time_input", "base": SMALL["gbt"]},
                                  {"name": "per_time", "base": "poly_ridge"}], ids=lambda s: s["name"])
def test_trajectory_export_is_exact(spec):
    U, Y, t = _traj_data()
    m = create("trajectory", spec, seed=0).fit(U, Y, t)
    s, info = export_trajectory(m, d=3)
    T = np.random.default_rng(3).random((5, 3))
    pr, rr = m.predict(T), predict_trajectory(dec(json.loads(json.dumps(s))), T)
    for o in Y:
        assert np.max(np.abs(rr[o][0] - pr[o][0])) < 1e-6 * (np.std(Y[o]) or 1)
        assert np.array_equal(np.isfinite(rr[o][1]), np.isfinite(pr[o][1]))


class _Unknown(Surrogate):
    """A plug-in the exporter has never heard of: it must be distilled."""
    supports_std = True

    def _fit(self, U, y):
        self.c = np.linalg.lstsq(np.c_[np.ones(len(U)), U, U ** 2], y, rcond=None)[0]

    def _predict(self, U, s):
        mu = np.c_[np.ones(len(U)), U, U ** 2] @ self.c
        return mu, (0.01 + 0.05 * U[:, 0] if s else None)


def test_unknown_plugin_is_distilled_accurately():
    U, y = _data()
    m = _Unknown().fit(U, y)
    s, info = export_scalar(m, d=3)
    assert info["method"] == "distilled"
    assert info["rel_err"] < 1e-6                     # a quadratic is learned exactly by the PCE student
    T = np.random.default_rng(4).random((200, 3))
    _, sd = m.predict(T, return_std=True)
    _, rs = predict_scalar(dec(s), T)
    assert np.median(np.abs(rs / sd - 1)) < 0.25       # uncertainty kept to tens of %


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_runtime_matches_python(tmp_path):
    """The browser runtime (runtime.js) gives the same numbers as the Python models."""
    U, y = _data()
    cases, expect = [], []
    rng = np.random.default_rng(5)
    for spec in SCALARS:
        m = _fit(spec, U, y)
        s, _ = export_scalar(m, d=3)
        T = rng.random((20, 3))
        cases.append({"kind": "scalar", "spec": s, "U": T.tolist()})
        expect.append(m.predict(T, return_std=True))
    Ut, Y, t = _traj_data()
    for spec in ({"name": "pca", "base": "gp"}, {"name": "time_input", "base": SMALL["gbt"]}):
        m = create("trajectory", spec, seed=0).fit(Ut, Y, t)
        s, _ = export_trajectory(m, d=3)
        T = rng.random((3, 3))
        cases.append({"kind": "traj", "spec": s, "U": T.tolist()})
        expect.append(m.predict(T))
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    (tmp_path / "run.js").write_text(
        "const MSR = require(process.argv[2]); const fs = require('fs');\n"
        "const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));\n"
        "const out = cases.map(c => { const s = MSR.decode(c.spec);\n"
        "  if (c.kind === 'scalar') return c.U.map(u => MSR.predictScalar(s, u));\n"
        "  return c.U.map(u => { const r = MSR.predictTrajectory(s, u);\n"
        "    return Object.fromEntries(Object.entries(r).map(([o, v]) => [o, [Array.from(v.mu), Array.from(v.sd)]])); }); });\n"
        "process.stdout.write(JSON.stringify(out));\n")
    res = subprocess.run(["node", str(tmp_path / "run.js"), str(RUNTIME), str(tmp_path / "cases.json")],
                         capture_output=True, text=True, check=True)
    got = json.loads(res.stdout)
    for c, g, e in zip(cases, got, expect):
        if c["kind"] == "scalar":
            mu = np.array([v[0] for v in g], float)
            sd = np.array([np.nan if v[1] is None else v[1] for v in g], float)
            assert np.max(np.abs(mu - e[0])) < 1e-6 * np.std(y), c["spec"]["type"]
            if np.isfinite(e[1]).all():
                assert np.max(np.abs(sd - e[1])) < 1e-5 * np.std(y), c["spec"]["type"]
        else:
            for i, gi in enumerate(g):
                for o in Y:
                    assert np.max(np.abs(np.array(gi[o][0]) - e[o][0][i])) < 1e-6 * np.std(Y[o])


def test_build_dashboard_end_to_end(tmp_path):
    from msrframe.config import load_config
    from msrframe.dashboard import build_dashboard
    from msrframe.stages import keff
    cfg = load_config(ROOT / "config/msfr_default.yaml", {
        "output_dir": str(tmp_path), "keff": {"models": ["poly_ridge", "linear"], "learning_curve": {"fractions": [0.5, 1.0]}}})
    keff.run(cfg, quiet=True)
    path = build_dashboard(cfg, quiet=True)
    html = path.read_text()
    m = re.search(r'<script type="application/json" id="msr-data">(.*?)</script>', html, re.S)
    data = json.loads(m.group(1))
    assert data["keff"]["order"] == ["poly_ridge", "linear"]
    assert data["burnup"] is None and data["screening"] is None
    assert len(data["keff"]["runs"]["y"]) == 200 and set(data["keff"]["runs"]["cv"]) == {"poly_ridge", "linear"}
    assert "NaN" not in m.group(1) and "Infinity" not in m.group(1)
    for marker in ("MSR.predictScalar", "Charts.lineChart", "viewKeff"):
        assert marker in html
