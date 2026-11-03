# msrframe: trainable, plug-in MSR neutronics design framework

`msrframe` is a Python framework for MSR/MSFR design studies. It follows the four stages of the
EVOL MSFR design studio (OpenMC-trained surrogates for k_eff, burnup and fission-product
control) and adds two things:

1. **You can train it on your own data.** OpenMC results are plain CSV files. Drop new runs into
   `data/user/` (or use `add-data`) and re-run. Every model, metric, learning curve and chart is
   regenerated, and an accuracy-history chart shows how the error falls as the database grows.
   The `suggest` and `loop` commands tell you *which* OpenMC runs will improve the models most.
2. **Every method is a replaceable plug-in.** Screening methods, surrogates, burnup trajectory
   models, optimisers and samplers are looked up by name, so you swap one by editing a word
   in the YAML config. For example, replacing Morris with Random Forest:

   ```yaml
   screening:
     methods: [random_forest, sobol_pce, active_subspace]   # was [morris, sobol_pce, active_subspace]
   ```

Every run prints its tables to the console and writes **charts (PNG) + CSV tables + a
self-contained HTML report** (`results/report.html`, one tab per stage).

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python -m msrframe run -c config/msfr_default.yaml          # all 4 stages + report
```

```bash
python -m msrframe run -c config/msfr_default.yaml --stages keff burnup
```

```bash
python -m msrframe list                                      # every available plug-in
```

Swap components from the command line without editing files:

```bash
python -m msrframe run -c config/msfr_default.yaml --stages screening --set "screening.methods=[random_forest,sobol_pce,mutual_info]"
```

```bash
python -m msrframe run -c config/swap_example.yaml          # a whole alternative plug-in set
```

## The four stages

| Stage | What it does | Default plug-ins | Example replacements |
|---|---|---|---|
| 1 Screening | Ranks ~61 design/uncertain inputs for k_eff. Thresholds are calibrated on dummy inputs (noise floor). Consensus vote plus safety-forced inputs. Retention curves. | `morris`, `sobol_pce`, `active_subspace` | `random_forest`, `gbt_importance`, `permutation` (any model), `sobol_surrogate` (any model), `dgsm`, `mutual_info`, `spearman`, `src`, `lasso` |
| 2 k_eff predictor | 5-fold CV, fixed hold-out, learning curves, accuracy history, response curves, critical loading and feedback coefficients | `gp`, `dnn`, `gbt`, `random_forest`, `pce` | `extra_trees`, `svr`, `krr`, `poly_ridge`, `linear`, `knn`, `ensemble`, `xgboost` (if installed), your own |
| 3 Burnup predictor | 30-year trajectories of k_eff, ²³³U, fissile, ²³³Pa, FPs, lanthanides, HN mol%, ²³²Th; CV over histories | `pca` with base `gp` / `dnn` / `random_forest`; `time_input` with base `gbt` | `{name: pca, base: <any surrogate>}`, `per_time`, `time_input` |
| 4 Fuel-cycle optimiser | Holds reactivity in a band at ±kσ under HN-solubility, fissile and temperature-feedback limits; exports the controls for OpenMC verification | `differential_evolution`, `pso`, `nelder_mead` | `dual_annealing`, `random_search`, `bayesian` |

Each surrogate-based method takes a `surrogate:` option, so the same index can be computed with
a different learner. For example, `{name: morris, surrogate: dnn}` gives surrogate-assisted
Morris, and `{name: permutation, surrogate: gp}` gives GP permutation importance.

## Training on your data (making the models more accurate)

```bash
python -m msrframe add-data -c config/msfr_default.yaml --stage keff --file my_openmc_runs.csv --retrain
```

`add-data` checks the required columns, warns about points outside the declared bounds (where
the models would extrapolate), stores the file in `data/user/`, and with `--retrain` re-runs the stage.

Required CSV columns:

| Stage | Columns |
|---|---|
| screening | every input + `keff` (optional `keff_std`, `dummy_1..3`, `morris_traj`) |
| keff | the stage inputs + `keff` (optional `keff_std`, `split` = train/test) |
| burnup | `case_id`, the inputs, `time_y`, and each output column (long format, one row per time point) |

**Active learning.** Find out where new OpenMC runs help most:

```bash
python -m msrframe suggest -c config/msfr_default.yaml --stage keff -n 20 --how hybrid
```

This writes `results/suggested_runs_keff.csv`: a spread-out batch where the GP is least certain
and/or the models disagree most, plus a chart of where those points sit. Run them in OpenMC, fill in
`keff`, and `add-data` them.

**Closed loop.** `python -m msrframe loop` repeats suggest → simulate → add → retrain. The
simulator is either your own OpenMC driver
(`active_learning.simulator: {name: command, command: "python run_openmc.py {inp} {out}"}`) or
the built-in analytic stand-in. It plots accuracy against the number of runs for active versus random
sampling.

**Design generation.** `python -m msrframe design --stage keff --sampler lhs -n 100` (or
`sobol`, `halton`, `morris` with dummy columns) writes input decks to run in OpenMC.

**Using a trained model.** This command predicts k_eff (with σ) or burnup trajectories for any CSV of design points:

```bash
python -m msrframe predict -c config/msfr_default.yaml --stage keff --file my_designs.csv
```

## Writing your own plug-in

See `examples/custom_plugin.py`:

```python
from msrframe import register
from msrframe.models.scalar import Surrogate

@register("surrogate", "my_model", "what it is")
class MyModel(Surrogate):
    supports_std = True
    def _fit(self, U, y): ...                 # U in [0,1]^d, y standardised
    def _predict(self, U, want_std): ...      # return (mu, sd or None)
```

Then add `plugins: [examples/custom_plugin.py]` and `models: [gp, my_model]` to the config.
Screening methods (`ScreeningMethod.run(ctx) -> DataFrame[key, score]`), trajectory models,
optimisers (`minimize(cost, dim) -> (u, f, trace)`) and samplers are registered the same way.

## Bundled data

| File | Content | Origin |
|---|---|---|
| `data/keff_campaign.csv` | 200 OpenMC k-eigenvalue runs over 9 inputs | **Recovered exactly** from the pilot GP (y = L Lᵀ α of the stored Cholesky factor) + the 34 test runs |
| `data/burnup_histories.csv` | 60 depletion histories × 23 time points × 8 outputs | k_eff recovered exactly from the PCA-GP; inventories PCA-reconstructed (truncation variance ≤ 8e-5) |
| `data/catalogue.csv` | 100-parameter MSFR catalogue | Pilot study |
| `data/screening_reference.csv` | Pilot Morris / Sobol / AS indices for the 64 screening factors | Pilot study |
| `data/screening_runs_demo.csv` | 476 screening runs (5 Morris trajectories + 151 LHS) | **Synthetic**: the analytic stand-in calibrated to the pilot's Morris effects, with 280 pcm noise. Replace with OpenMC runs. |

`tools/extract_reference_data.py` regenerates the first four files from the original HTML.
`tools/make_demo_screening.py` regenerates the demo screening set.

## Caveats

* The bundled data are at pilot statistics (~200–400 pcm Monte Carlo noise per k_eff). Surrogate
  errors below that noise cannot be demonstrated; the report gives noise-corrected RMSE where
  `keff_std` is known.
* The `analytic_msfr` simulator is a calibrated stand-in, not neutronics. Use it only to exercise
  the pipeline.
* High-dimensional ARD GP fits (Stage 1 with 64 inputs) cap the hyper-parameter optimiser at
  60 L-BFGS iterations. Stage 1 then takes ~4 min on 4 cores.
* Stage 4 holds controls constant over the horizon (the surrogates were trained on constant-control
  histories). Verify the exported `verify_in_openmc.csv` controls with transport.
