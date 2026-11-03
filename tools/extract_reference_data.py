"""Extract the pilot OpenMC datasets embedded in the original MSFR design-framework HTML.

The original dashboard ships trained models, not the raw training tables, but the
Gaussian-process models store their Cholesky factor L of (K + sigma_n^2 I) and the
weights alpha = (K + sigma_n^2 I)^-1 y.  The exact (normalised) training targets are
therefore y = L L^T alpha, which this script uses to recover:

  data/keff_campaign.csv      200 OpenMC k-eigenvalue runs (166 train + 34 test), 9 inputs
  data/burnup_histories.csv   60 OpenMC depletion histories x 23 time points, 8 outputs
  data/catalogue.csv          the 100-parameter MSFR catalogue (ranges, nominal, class)
  data/screening_reference.csv  Stage-1 Morris / Sobol / AS indices of the pilot (64 factors)

Usage:  python tools/extract_reference_data.py path/to/msfr_design_framework.html
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load(html_path):
    lines = Path(html_path).read_text(encoding="utf-8").split("\n")
    line = next(l for l in lines if 'id="msfr-data"' in l)
    return json.loads(line[line.index(">") + 1: line.rindex("</script>")])


def gp_targets(g):
    n = len(g["alpha"])
    L = np.zeros((n, n))
    for i, row in enumerate(g["L"]):
        L[i, : len(row)] = row
    y = L @ (L.T @ np.asarray(g["alpha"]))
    return y * g["y_std"] + g["y_mean"]


def unscale(g, X):
    lo, hi = np.asarray(g["scaler"]["lo"]), np.asarray(g["scaler"]["hi"])
    return np.asarray(X) * (hi - lo) + lo


def main(html_path):
    d = load(html_path)
    out = ROOT / "data"
    out.mkdir(exist_ok=True)

    # ---------------------------------------------------------------- catalogue
    pd.DataFrame(d["catalogue"]).to_csv(out / "catalogue.csv", index=False)

    # ---------------------------------------------------------------- stage 1 reference indices
    pd.DataFrame(d["stage1"]["importance"]).to_csv(out / "screening_reference.csv", index=False)

    # ---------------------------------------------------------------- stage 2: k_eff campaign
    s2 = d["stage2"]
    keys = s2["keys"]
    g = s2["models"]["GP"]
    Xtr = unscale(g, g["X_train"])
    ytr = gp_targets(g)
    rows = [dict(case_id=f"TR{i:04d}", split="train", **dict(zip(keys, x)), keff=y, keff_std=np.nan)
            for i, (x, y) in enumerate(zip(Xtr, ytr))]
    for t in s2["test"]:
        rows.append(dict(case_id=t["case_id"], split="test", **dict(zip(keys, t["x"])),
                         keff=t["keff"], keff_std=t["keff_std"]))
    df2 = pd.DataFrame(rows)
    df2.to_csv(out / "keff_campaign.csv", index=False)
    # sanity: GP prediction at the training points should be close to the recovered targets
    print(f"keff_campaign.csv: {len(df2)} runs; k range {df2.keff.min():.4f}-{df2.keff.max():.4f}")

    # ---------------------------------------------------------------- stage 3: burnup histories
    s3 = d["stage3"]
    keys3, times = s3["keys"], s3["times"]
    outs = s3["models"]["GP"]["outputs"]
    X3 = None
    traj = {}
    for o, O in outs.items():
        V = np.asarray(O["V"])
        scores = np.column_stack([gp_targets(gp) for gp in O["gps"]])
        traj[o] = np.asarray(O["mu"]) + (scores @ V) * O["sc"]
        if X3 is None:
            X3 = unscale(O["gps"][0], O["gps"][0]["X_train"])
    # inputs are stored in log10 for log-scaled parameters
    raw = X3.copy()
    for j, k in enumerate(keys3):
        if s3["params"][k]["log"]:
            raw[:, j] = 10 ** raw[:, j]
    # match training rows to the test-record case ids (same 60 histories)
    ids = []
    tp = np.array([[t["params"][k] for k in keys3] for t in s3["test"]])
    for r in raw:
        j = int(np.argmin(np.abs((tp - r) / (np.abs(tp) + 1e-9)).sum(1)))
        ids.append(s3["test"][j]["case_id"])
        err = np.max(np.abs(np.asarray(s3["test"][j]["keff"]) - traj["keff"][len(ids) - 1]))
        if err > 1e-4:
            print("warning: keff reconstruction mismatch", ids[-1], err)
    rec = []
    for i, cid in enumerate(ids):
        for it, t in enumerate(times):
            r = dict(case_id=cid, **dict(zip(keys3, raw[i])), time_y=t)
            for o in outs:
                r[o] = traj[o][i, it]
            rec.append(r)
    df3 = pd.DataFrame(rec)
    df3.to_csv(out / "burnup_histories.csv", index=False)
    tv = {o: outs[o]["trunc_var"] for o in outs}
    print(f"burnup_histories.csv: {len(ids)} histories x {len(times)} times; PCA truncation variance {tv}")

    meta = dict(source=str(Path(html_path).name), stage2_keys=keys, stage3_keys=keys3, times_y=times,
                stage2_params=s2["params"], stage3_params=s3["params"], stage3_outputs=s3["outputs"],
                pilot_metrics=dict(stage2=s2["metrics"], stage3={m: v.get("keff") for m, v in s3["metrics"].items()}),
                consensus=d["stage1"]["comparison"]["consensus"], scenarios=[
                    s["request"] for s in d["stage4"]["scenarios"]],
                alpha_T_pcm_per_K=d["meta"]["alpha_T_pcm_per_K"])
    (out / "reference_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
