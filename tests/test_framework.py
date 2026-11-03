"""Fast checks: registry, plug-in swapping, data recovery and a tiny end-to-end run."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import msrframe  # noqa: E402
from msrframe import available, create  # noqa: E402
from msrframe.config import load_config  # noqa: E402
from msrframe.screening import ScreeningContext  # noqa: E402


def test_registry_has_all_kinds():
    a = available()
    for name in ("gp", "dnn", "gbt", "random_forest", "pce", "svr"):
        assert name in a["surrogate"]
    for name in ("morris", "sobol_pce", "active_subspace", "random_forest", "mutual_info"):
        assert name in a["screening"]
    assert {"pca", "time_input", "per_time"} <= set(a["trajectory"])
    assert "differential_evolution" in a["optimizer"]


@pytest.mark.parametrize("name", ["gp", "random_forest", "pce", "poly_ridge", "gbt"])
def test_surrogates_learn(name):
    rng = np.random.default_rng(0)
    U = rng.random((80, 3))
    y = 2 * U[:, 0] - U[:, 1] ** 2
    m = create("surrogate", name).fit(U, y)
    T = rng.random((50, 3))
    assert np.sqrt(np.mean((m.predict(T) - (2 * T[:, 0] - T[:, 1] ** 2)) ** 2)) < 0.25


@pytest.mark.parametrize("method", ["morris", "random_forest", "sobol_pce", "spearman", "src"])
def test_screening_methods_are_interchangeable(method):
    rng = np.random.default_rng(1)
    U = rng.random((150, 5))
    y = 10 * U[:, 0] + 3 * U[:, 1] + 0.05 * rng.standard_normal(150)
    ctx = ScreeningContext(U, y, list("abcde"), None, seed=0)
    r = create("screening", {"name": method, "surrogate": "poly_ridge"} if method == "morris" else method).run(ctx)
    assert list(r.sort_values("score", ascending=False).key[:2]) == ["a", "b"]


def test_recovered_reference_data():
    k = pd.read_csv(ROOT / "data/keff_campaign.csv")
    b = pd.read_csv(ROOT / "data/burnup_histories.csv")
    assert len(k) == 200 and k.keff.between(0.6, 1.2).all()
    assert b.case_id.nunique() == 60 and b.groupby("case_id").size().eq(23).all()


def test_keff_stage_end_to_end(tmp_path):
    cfg = load_config(ROOT / "config/msfr_default.yaml", {
        "output_dir": str(tmp_path), "keff": {"models": ["poly_ridge", "linear"], "learning_curve": {"fractions": [0.5, 1.0]}}})
    from msrframe.stages import keff
    R = keff.run(cfg, quiet=True)
    assert R["summary"]["cv"]["poly_ridge"]["rmse"] < 2000
    assert (tmp_path / "models" / "keff_deployed.pkl").exists()
    assert any(Path(p).name == "fig_parity.png" for p, _ in R["figures"])
