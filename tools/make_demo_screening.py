"""Generate a DEMO Stage-1 screening dataset (no OpenMC needed).

The pilot's 476 raw screening runs are not stored in the original HTML, so this script
synthesises a dataset of the same size and design (5 Morris trajectories over 61
inputs + 3 dummies = 325 runs, plus 151 Latin-hypercube runs) from the analytic
stand-in calibrated to the pilot's signed Morris effects, with 280 pcm Monte-Carlo-like
noise. Replace data/screening_runs_demo.csv with your OpenMC results for real use.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import msrframe  # noqa: F401,E402
from msrframe.registry import get  # noqa: E402
from msrframe.simulators import AnalyticMSFR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ref = ROOT / "data/screening_reference.csv"
cat = pd.read_csv(ROOT / "data/catalogue.csv").set_index("key")
sim = AnalyticMSFR(ref, catalogue_csv=ROOT / "data/catalogue.csv", seed=7)
keys = sim.all_keys
lo, hi = cat.loc[keys, "low"].to_numpy(), cat.loc[keys, "high"].to_numpy()
d = len(keys) + 3
Um, tr = get("sampler", "morris")(5, d, seed=11)
Ul = get("sampler", "lhs")(151, d, seed=12)
U = np.vstack([Um, Ul])
X = lo + U[:, :len(keys)] * (hi - lo)
df = pd.DataFrame(X, columns=keys)
for i in range(3):
    df[f"dummy_{i + 1}"] = U[:, len(keys) + i]
df["morris_traj"] = np.r_[tr, np.full(len(Ul), np.nan)]
df["keff"] = sim.keff(X)
df["keff_std"] = sim.noise * 1e-5
df.insert(0, "case_id", [f"SC{i:04d}" for i in range(len(df))])
df.to_csv(ROOT / "data/screening_runs_demo.csv", index=False)
print(f"{len(df)} demo screening runs -> data/screening_runs_demo.csv")
