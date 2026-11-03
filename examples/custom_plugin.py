"""Example: add your OWN surrogate model and screening method as plug-ins.

Enable it in the YAML config:

    plugins: [examples/custom_plugin.py]
    keff:
      models: [gp, my_mlp_quantile, random_forest]
    screening:
      methods: [random_forest, my_morris_rf, sobol_pce]

or try it once from the command line:

    python -m msrframe run -c config/msfr_default.yaml --stages keff \
        --set plugins=[examples/custom_plugin.py] --set keff.models=[gp,bagged_ridge]
"""
import numpy as np

from msrframe import register
from msrframe.models.scalar import Surrogate
from msrframe.screening import Morris


@register("surrogate", "bagged_ridge", "Bootstrap-bagged quadratic ridge regression (example plug-in)", has_std=True)
class BaggedRidge(Surrogate):
    """Any class with _fit(U, y) and _predict(U, want_std) -> (mu, sd|None) works.

    U is the input already scaled to [0, 1]; y is standardised by the base class."""
    supports_std = True

    def _fit(self, U, y):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import PolynomialFeatures
        self.pf = PolynomialFeatures(self.params.get("degree", 2))
        P = self.pf.fit_transform(U)
        rng = np.random.default_rng(self.seed)
        self.members = []
        for _ in range(self.params.get("bags", 20)):
            i = rng.integers(0, len(U), len(U))
            self.members.append(Ridge(alpha=self.params.get("alpha", 1e-3)).fit(P[i], y[i]))

    def _predict(self, U, want_std):
        P = self.pf.transform(U)
        M = np.array([m.predict(P) for m in self.members])
        return M.mean(0), (M.std(0) if want_std else None)


@register("screening", "my_morris_rf", "Morris mu* computed on a random-forest surrogate (example plug-in)")
class MorrisOnRF(Morris):
    """Re-use an existing method with a different learner: here Morris on a random forest."""

    def __init__(self, seed=0, **params):
        super().__init__(seed=seed, **{"surrogate": "random_forest", "force_surrogate": True, **params})
