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
    return results, p
