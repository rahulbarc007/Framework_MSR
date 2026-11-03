"""msrframe - a trainable, plug-in MSR/MSFR neutronics design framework.

Stages (mirroring the EVOL MSFR design studio):
  1. screening   - which of the ~64 design/uncertain parameters matter for k_eff
  2. keff        - static k_eff surrogate on the retained parameters
  3. burnup      - 30-year depletion-trajectory surrogate (k_eff, inventories)
  4. optimize    - fuel-cycle controller built on the burnup surrogate

Every method (screening index, surrogate, trajectory model, optimiser, sampler) is a
plug-in looked up by name in :mod:`msrframe.registry`, so e.g. Morris screening can be
replaced by Random-Forest permutation importance by changing one word in the YAML config.
Feed new OpenMC runs into ``data/user/`` (or ``msrframe add-data``) and re-train: models,
validation metrics, learning curves and an accuracy-history chart are regenerated.
"""
__version__ = "1.0.0"

from . import registry  # noqa: F401
from .registry import register, create, available  # noqa: F401


def load_plugins():
    """Import all built-in plug-in modules so that they register themselves."""
    from .models import scalar, trajectory  # noqa: F401
    from . import screening, optimizers, design  # noqa: F401


load_plugins()
