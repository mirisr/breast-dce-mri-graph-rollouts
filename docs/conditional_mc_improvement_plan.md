# Conditional Monte Carlo Rollout: Analysis and Improvement Plan

Generated 2026-05-01 from Cradle outputs produced 2026-04-30.

## Source artifacts

Remote Cradle paths:

- `~/3DGCNN/experiments/consistent_rollout/logs/cg-cond-mc-48326.out`
- `~/3DGCNN/experiments/consistent_rollout/logs/cg-cond-mc-48326.err`
- `~/3DGCNN/reports/conditional_mc_consistent_rollout/conditional_mc_summary.json`
- `~/3DGCNN/reports/conditional_mc_consistent_rollout/conditional_mc_calibration.json`
- `~/3DGCNN/reports/conditional_mc_consistent_rollout/conditional_mc_per_patient.parquet`
- `~/3DGCNN/reports/conditional_mc_consistent_rollout/conditional_mc_samples.parquet`

Run configuration:

- Model: `runs/consistent_forecaster_v2/v2_sched_samp`, `best.pt`
- Graphs: `datasets/ispy2/graphs_consistent`
- Patients: 758
- Rollout records: 4,548
- Monte Carlo samples: 1,164,288 (`N_MC=256`)
- Conditioning starts: `T0`, `T1`, `T2`
- Device: CUDA on Cradle
- Leakage policy in the script: held-out fold checkpoint per patient; calibration residuals exclude the target patient
- Runtime: 2026-04-30 15:06:28 to 18:56:22 CDT, about 3 h 50 min on one GPU

The first smoke job, `48323`, failed because the script called a non-existent `consistent_twin_lib.load_graph_dataset`. The later smoke (`48324`) and full run (`48326`) completed after switching to direct graph loading.

## Main results

The table below uses `abs(ftv_mc_mean_ml - obs_ftv_ml)` for `mc_mean_mae`. The report field `ftv_abs_err_ml_mc_mean` is different: it is the mean absolute error over MC draws.

| Bucket | n | obs mean | det mean | MC mean | det bias | MC bias | det MAE | MC-mean MAE | raw 90 cov | conf 90 cov | raw width | conf width | CRPS | pCR prob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0->T1 | 758 | 24.31 | 12.61 | 30.26 | -11.70 | 5.95 | 11.88 | 13.08 | 0.752 | 0.901 | 52.20 | 54.10 | 7.06 | 0.0029 |
| T0->T2 | 758 | 24.75 | 12.16 | 31.30 | -12.58 | 6.56 | 12.86 | 14.46 | 0.793 | 0.901 | 55.58 | 57.56 | 7.80 | 0.0061 |
| T0->T3 | 758 | 24.94 | 12.43 | 31.30 | -12.51 | 6.36 | 12.88 | 15.26 | 0.766 | 0.901 | 56.87 | 59.51 | 8.35 | 0.0084 |
| T1->T2 | 758 | 24.75 | 12.18 | 31.61 | -12.56 | 6.86 | 12.92 | 13.41 | 0.793 | 0.901 | 54.21 | 56.25 | 7.28 | 0.0071 |
| T1->T3 | 758 | 24.94 | 12.20 | 31.85 | -12.74 | 6.90 | 13.17 | 14.88 | 0.764 | 0.901 | 57.94 | 60.14 | 8.10 | 0.0085 |
| T2->T3 | 758 | 24.94 | 12.19 | 31.97 | -12.75 | 7.02 | 13.05 | 13.80 | 0.776 | 0.901 | 57.91 | 60.25 | 7.49 | 0.0067 |

T3-specific pattern:

| Conditioning | expected abs FTV error over MC draws | CRPS | raw width | conformal width | Dice | SWD | alive-count error |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0->T3 | 18.81 | 8.35 | 56.87 | 59.51 | 0.047 | 1.32 | 40.78 |
| T0,T1->T3 | 18.56 | 8.10 | 57.94 | 60.14 | 0.034 | 1.81 | 41.31 |
| T0,T1,T2->T3 | 17.61 | 7.49 | 57.91 | 60.25 | 0.035 | 2.97 | 41.64 |

## Interpretation

1. The deterministic rollout systematically underpredicts FTV. Across buckets, mean deterministic FTV is about 12.2-12.6 mL against observed means of 24.3-24.9 mL, giving about -12 to -13 mL bias.

2. The residual-bootstrap MC correction overcompensates. It moves the mean prediction to about 30-32 mL and creates +6 to +7 mL bias. As a point estimator, the MC mean is not better than the deterministic rollout in this run.

3. Raw 90% intervals under-cover. Raw FTV coverage is 0.752-0.793, so the MC distribution is not calibrated by itself. Split conformal expansion reaches 0.901 in every bucket, but with very wide intervals: roughly 54-60 mL.

4. Conditioning on later visits improves T3 volume scoring but not the shape metrics. T3 CRPS improves from 8.35 at T0 start to 7.49 at T2 start, but SWD worsens from 1.32 to 2.97 and Chamfer also worsens. The uncertainty sampler is capturing volume better than spatial morphology.

5. The pCR probability proxy is unusable as a response probability. `pcr_prob_mc = P(FTV_T3 < 0.1 mL)` is about 0.0075 for observed non-pCR and 0.0087 for observed pCR at T3, while the observed pCR rate is 31.5%. This is effectively a near-zero residual disease probability, not a calibrated pCR model.

6. Alive-count calibration is a major failure mode. Residual buckets show observed alive counts exceed deterministic alive mass by about 76-79 supervoxels on average. MC alive-count absolute error remains about 39-42 supervoxels across all buckets.

7. The worst cases are high-volume tumors. The top T3 errors are patients such as `ISPY2-494022`, `ISPY2-327133`, `ISPY2-357017`, and `ISPY2-601489`, where observed T3 FTV is 240-396 mL but the rollout remains far lower. Additive residual sampling is not enough for the heavy right tail.

8. Subgroup drift is visible. HR-/HER2+ has the worst T3 expected MC absolute error (21.60 mL) and CRPS (10.98). HR+/HER2+ has the largest positive MC bias (10.45 mL). Fold 2 has the weakest raw T3 coverage (0.715) and fold 3 has the weakest conformal T3 coverage (0.881).

## Improvement plan

### 1. Make the MC experiment reproducible and auditable

- Sync the successful Cradle version of `experiments/consistent_rollout/run_conditional_mc.py`, `run_conditional_mc.sbatch`, and the unit tests back into the tracked repository.
- Add a run manifest alongside each report with git commit, `git status --short`, Slurm job id, Python package versions, seed, checkpoint paths, and graph counts.
- Copy the small JSON summaries locally after each Cradle run. Leave the large sample parquet on Cradle unless it is needed for figure generation.
- Add an end-to-end smoke test that runs `PATIENT_LIMIT=3 N_MC=8` and asserts that all four report files are written with the expected row counts.
- Add a schema check for `conditional_mc_per_patient.parquet` and `conditional_mc_samples.parquet`.

### 2. Recalibrate FTV mean prediction before widening intervals

The current MC sampler treats the deterministic underprediction as a residual to add back, but it overshoots the mean. Fix the center of the distribution first.

Experiments:

- Add bucket-wise residual centering: sample residuals after subtracting the calibration-bucket mean, then add back a learned or cross-fitted bias correction only once.
- Compare additive FTV residuals against log-space residuals: `log1p(obs_ftv) - log1p(pred_ftv)`. The high-volume failures suggest an additive mL residual is too weak in the right tail.
- Normalize residuals by baseline FTV or observed context FTV, then denormalize during sampling.
- Evaluate deterministic, MC mean, MC median, and calibrated mean separately. Do not treat the MC mean as improved unless it beats deterministic MAE and bias.

Gate:

- Keep raw coverage at or above 0.85 before conformal correction.
- Reduce T3 MC-mean absolute error below deterministic T3 MAE for every start visit.
- Keep T3 CRPS no worse than the current 7.49-8.35 range.

### 3. Condition residual sampling instead of using only visit buckets

The residual library currently buckets by `(start_visit, predicted_visit)` and excludes the target patient. That preserves basic leakage control, but it mixes very different biology and tumor burden.

Add candidate strata:

- Baseline FTV quantile or observed context FTV quantile
- Subtype
- Collection
- pCR label for retrospective analysis only; do not use pCR label for deployable prediction
- Fold-specific diagnostics, but not fold-specific calibration unless the calibration sample size remains adequate

Recommended first version:

- Stratify by visit bucket plus baseline FTV quartile.
- Fall back to visit bucket when the stratum has fewer than 100 calibration patients.
- Report both conditional and fallback sample counts in the per-patient table.

Gate:

- Reduce the 95th and 99th percentile T3 FTV errors, especially high-volume patients.
- Preserve conformal coverage after target-patient exclusion.

### 4. Fix alive-count calibration as its own model

Alive mass is not calibrated, and the residual sampler uses a large additive alive-count residual. That contaminates both FTV and geometry.

Experiments:

- Calibrate `alive_prob` with temperature scaling or isotonic regression per visit bucket.
- Predict alive count with a count model conditioned on alive mass, baseline/context FTV, and visit bucket.
- Sample alive masks from calibrated Bernoulli probabilities first, then optionally enforce count only as a soft constraint.
- Report alive-count calibration curves: predicted alive mass deciles vs. observed alive counts.

Gate:

- Bring T3 alive-count absolute error below 30 supervoxels.
- Avoid degrading FTV CRPS and coverage.

### 5. Separate spatial uncertainty from volume uncertainty

The T3 volume metrics improve with later conditioning, but SWD and Chamfer worsen. The spatial residual sampler needs a better local model.

Experiments:

- Sample local displacement residuals from patients matched by supervoxel count and context FTV.
- Scale local displacement noise by observed or predicted tumor size.
- Preserve coherent spatial structure by sampling residual fields at the patient level instead of independent local residuals for every node.
- Add a diagnostic table for SWD, Chamfer, Dice, and alive-count error by supervoxel-count decile.

Gate:

- Improve T2->T3 SWD below the current 2.97 mm and Chamfer below 5.74 mm.
- Keep Dice from falling below the current 0.035-0.047 range.

### 6. Replace the pCR proxy with a real response model

`P(FTV_T3 < 0.1 mL)` should not be reported as pCR probability. It is too strict and essentially always near zero.

Plan:

- Rename the current field to something explicit, such as `prob_near_zero_ftv_mc`.
- Train a pCR calibration model on patient-level features derived from the rollout distribution:
  - MC FTV mean, median, width, and CRPS-like uncertainty summaries
  - FTV ratios relative to T0
  - alive-count summaries
  - subtype, HR/HER2 status, collection, and treatment arm when available
- Evaluate AUC, Brier score, ECE, and calibration slope under the same held-out fold protocol.
- Compare against the known FTV-only oracle and the Stage 0 encoder baseline.

Gate:

- pCR AUC must beat the FTV-only oracle or the field should be framed as residual-disease probability, not response prediction.
- Brier and ECE must be reported alongside AUC.

### 7. Run a compact ablation grid on Cradle

Start with cheap sweeps using `METRIC_DRAWS=32` and then rerun finalists with full draw metrics.

Initial variants:

| Variant | Change | Expected benefit |
|---|---|---|
| `centered_additive` | bucket residual centering | reduce MC mean overprediction |
| `log_ftv` | log-space FTV residuals | improve high-volume tail |
| `ftv_quantile_strata` | visit bucket + baseline FTV quartile | reduce subgroup mixing |
| `alive_temp` | alive-prob temperature scaling | lower alive-count error |
| `field_residual` | patient-level residual fields | improve SWD/Chamfer |

Existing rerun pattern after code changes:

```bash
cd ~/3DGCNN

N_MC=8 PATIENT_LIMIT=3 OUT_DIR=reports/conditional_mc_smoke \
  sbatch experiments/consistent_rollout/run_conditional_mc.sbatch

N_MC=256 METRIC_DRAWS=32 OUT_DIR=reports/conditional_mc_<variant> \
  sbatch experiments/consistent_rollout/run_conditional_mc.sbatch
```

Finalists should be rerun with `METRIC_DRAWS=0` so that all MC samples have cloud metrics.

### 8. Reporting package for the paper

After the improved run:

- One calibration table: raw coverage, conformal coverage, interval width, CRPS by conditioning pair.
- One T3 conditioning table: `T0->T3`, `T0,T1->T3`, `T0,T1,T2->T3`.
- One subgroup table: subtype, collection, pCR status, fold.
- One figure: observed vs. MC mean FTV with interval bars for a stratified patient subset.
- One reliability curve for pCR if a real pCR model is added.
- One failure-case panel for the high-volume tail.

## Priority order

1. Reproducibility manifest and smoke test.
2. FTV residual centering and log-space residual ablation.
3. Baseline-FTV-stratified residual sampling.
4. Alive-count calibration.
5. Patient-level spatial residual fields.
6. Real pCR calibration model.

The immediate next experiment should be a three-way comparison:

| Run | Goal |
|---|---|
| current sampler | baseline from job `48326` |
| centered additive residuals | test whether the +6 to +7 mL MC bias is removable |
| log-space FTV residuals | test whether high-volume failures improve |

Use the same 758 patients, same held-out fold checkpoints, same seed, and the same six conditioning buckets so the tables are directly comparable.
