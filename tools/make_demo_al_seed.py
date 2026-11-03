"""Seed the active-learning demo with 30 analytic stand-in 'runs' (data/demo_al/seed_keff.csv)."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from msrframe.config import load_config  # noqa: E402
from msrframe.registry import get  # noqa: E402
from msrframe.simulators import make_simulator  # noqa: E402

cfg = load_config(ROOT / "config/demo_active_learning.yaml")
keys = cfg["keff"]["inputs"]
space = cfg.space("keff", keys)
sim = make_simulator(cfg, cfg["active_learning"]["simulator"], keys)
X = space.from_unit(get("sampler", "lhs")(30, len(keys), seed=1))
df = sim.run(pd.DataFrame(X, columns=keys))
df.insert(0, "case_id", [f"SEED{i:03d}" for i in range(len(df))])
out = ROOT / "data/demo_al"
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("keff_active*.csv"):
    f.unlink()       # restart the demo from the seed
df.to_csv(out / "seed_keff.csv", index=False)
print(f"{len(df)} seed runs -> {out / 'seed_keff.csv'}")
