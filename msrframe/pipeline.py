"""Run the enabled stages in order and write the HTML report."""
from __future__ import annotations

import time

from .report import write_report

STAGES = ("screening", "keff", "burnup", "optimize")


def run_pipeline(cfg, stages=None, tag=None):
    from .stages import burnup, keff, optimize, screening
    fns = {"screening": screening.run, "keff": lambda c: keff.run(c, tag=tag), "burnup": lambda c: burnup.run(c, tag=tag),
           "optimize": optimize.run}
    results = []
    for s in stages or STAGES:
        if not cfg[s].get("enabled", True):
            continue
        t0 = time.time()
        results.append(fns[s](cfg))
        print(f"  [{s}] done in {time.time() - t0:.1f} s\n")
    p = write_report(cfg, results)
    print(f"report -> {p}")
    maybe_dashboard(cfg)
    return results, p


def maybe_dashboard(cfg):
    """Rebuild the interactive dashboard from whatever stages have results (never fails a run)."""
    if not (cfg.get("dashboard") or {}).get("enabled", True):
        return None
    try:
        from .dashboard import build_dashboard
        return build_dashboard(cfg)
    except Exception as e:  # pragma: no cover - report, do not break the training run
        print(f"  dashboard not rebuilt: {e}")
        return None
