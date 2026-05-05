# Consistent Graph Rollout Model Upgrade Plan

Generated 2026-05-01 after reviewing the Cradle conditional MC run and the current v2 rollout trainer.

## Bottom line

Yes, the model changes are feasible. They should be staged rather than launched as one giant grid. The current code already supports several of the most important changes:

- scheduled sampling
- random-start conditioning
- horizon cloud loss
- aggregate FTV loss
- alive BCE loss
- alive-mass loss
- biology-aware checkpoint selection weights

That means the first overnight wave can be mostly training/evaluation, not a major code rewrite. The larger changes, especially clinical conditioning, richer graph features, and a real pCR head, need new interfaces and should be the second wave.

## Current Cradle status

As of inspection, Cradle already has the model-level bio grid running:

- Training: Slurm array `48329`, `cg-bio-grid`, 25 tasks, throttle 5 on `gpul40q`
- Evaluation: Slurm array `48335`, `cg-bio-eval`, dependency `afterok:48329_*`
- Summary: Slurm job `48336`, `cg-bio-sum`, dependency `afterok:48335_*`

This grid trains five aggregate biology-loss variants across five folds:

| Tag | FTV loss | alive mass | alive BCE | Purpose |
|---|---:|---:|---:|---|
| `bio_ftv005_alive001` | 0.05 | 0.01 | 0.05 | low-weight aggregate correction |
| `bio_ftv010_alive002` | 0.10 | 0.02 | 0.05 | moderate correction |
| `bio_ftv020_alive005` | 0.20 | 0.05 | 0.05 | strong correction |
| `bio_ftv010_alive000` | 0.10 | 0.00 | 0.00 | isolate FTV loss |
| `bio_ftv000_alive002` | 0.00 | 0.02 | 0.05 | isolate alive loss |

This directly addresses two failures from the MC analysis: deterministic FTV underprediction and alive-count underprediction.

## Primary success gates

The current scheduled-sampling model has this failure pattern:

- deterministic FTV bias: about -12 to -13 mL
- MC mean bias after residual bootstrap: about +6 to +7 mL
- raw 90% FTV coverage: 0.752-0.793
- conformal coverage: 0.901, but with 54-60 mL intervals
- T3 alive-count absolute error in MC: about 41 supervoxels
- pCR proxy essentially near zero for everyone

The next model should be judged on:

| Metric | Current issue | Minimum acceptable next result |
|---|---|---|
| T3 deterministic FTV bias | -12 to -13 mL | within +/-5 mL |
| T3 deterministic FTV MAE | about 13 mL from sim eval | no worse than current, ideally <10 mL |
| T3 alive-count error | about 41 in MC, validation logs vary | <30 supervoxels |
| T3 SWD/Chamfer | spatial quality degrades after conditioning | no regression from `v2_sched_samp` |
| raw MC coverage | 0.75-0.79 | >=0.85 before conformal |
| conformal width | 54-60 mL | narrower at same 90% coverage |
| pCR probability | unusable threshold proxy | replace with calibrated response model |

## Overnight Wave 1: finish the active bio-loss grid

Do not launch a duplicate of `48329`. Let the active grid finish, then inspect:

```bash
ssh cradle
cd ~/3DGCNN
squeue -u $USER
tail -f experiments/stage1_forecaster/logs/cg-bio-grid-48329_*.out
```

After dependencies finish:

```bash
cat experiments/stage1_forecaster/logs/cg-bio-sum-48336.out
ls -lh reports/consistent_forecaster_v2_bio_eval/
```

Expected output:

- `reports/consistent_forecaster_v2_bio_eval/<tag>/simulation_summary.json`
- `reports/consistent_forecaster_v2_bio_eval/<tag>/simulation_per_patient.parquet`
- `reports/consistent_forecaster_v2_bio_eval/bio_grid_summary.csv`
- `reports/consistent_forecaster_v2_bio_eval/bio_grid_summary.json`

Decision rule:

1. Pick any tag with lower T3 FTV bias/MAE and no large SWD/Chamfer regression.
2. If two tags are close, prefer the one with lower alive-count error.
3. If all bio-loss variants improve validation FTV but fail simulation rollout, the current training objective is overfitting validation one-step behavior and needs the full autoregressive loss from Wave 2.

## Overnight Wave 1b: conditional MC on the best model

Once `48336` identifies a winner, rerun conditional MC on the best one or two tags. This answers whether model retraining improves uncertainty calibration, not just deterministic rollout metrics.

Example for one winner:

```bash
cd ~/3DGCNN

TAG=bio_ftv010_alive002
RUNS_DIR=runs/consistent_forecaster_v2_bio/${TAG} \
OUT_DIR=reports/conditional_mc_${TAG} \
N_MC=256 \
METRIC_DRAWS=32 \
sbatch experiments/consistent_rollout/run_conditional_mc.sbatch
```

For the final selected tag, rerun with full metric draws:

```bash
TAG=bio_ftv010_alive002
RUNS_DIR=runs/consistent_forecaster_v2_bio/${TAG} \
OUT_DIR=reports/conditional_mc_${TAG}_fullmetrics \
N_MC=256 \
METRIC_DRAWS=0 \
sbatch experiments/consistent_rollout/run_conditional_mc.sbatch
```

Gate:

- raw coverage improves toward 0.85
- conformal width decreases
- MC mean bias decreases
- high-volume failure cases improve

## Wave 2: code changes for the next training grid

These are the model changes I recommend after the current grid has a baseline result.

### 1. Full autoregressive multi-step loss

Current `curriculum_forward_patient` does scheduled sampling within the transition loop, but the loss is still mostly transition-local. Add a training mode that explicitly rolls out from each start visit and scores every future visit:

- `T0 -> T1,T2,T3`
- `T1 -> T2,T3`
- `T2 -> T3`

Loss terms per future visit:

- position SWD or Chamfer
- feature cloud loss
- aggregate log FTV
- alive mass/count
- optional final-horizon extra weight

New flags:

- `--full-rollout-loss`
- `--rollout-starts 0 1 2`
- `--lambda-rollout-pos`
- `--lambda-rollout-ftv`
- `--lambda-rollout-alive`

This targets the failure where one-step dynamics do not compose into calibrated T3 trajectories.

### 2. Log-volume and ratio targets

The model currently predicts raw `delta_feat`, and FTV is supervised through aggregate `log1p(pred_ftv) - log1p(obs_ftv)`. Add explicit node/patient volume parameterizations:

- node-level `log1p(volume)` delta loss
- patient-level `log1p(FTV_t / FTV_0)` or `log1p(FTV_t) - log1p(FTV_0)` loss
- optional high-volume reweighting by baseline FTV quantile

New flags:

- `--lambda-node-logvol`
- `--lambda-ftv-ratio`
- `--ftv-quantile-weighting`

This targets the high-volume tail failures where additive residual correction is too weak.

### 3. Alive head calibration during training

The alive head needs more than BCE. Add a differentiable calibration/count objective:

- alive mass loss, already present
- squared count loss or Huber count loss
- decile calibration diagnostic after each validation epoch
- optional temperature parameter saved with checkpoint

New flags:

- `--alive-count-loss huber`
- `--lambda-alive-count`
- `--learn-alive-temperature`

This targets the 76-79 supervoxel alive-mass residual seen in calibration buckets.

### 4. Clinical and biology conditioning

Add graph-level context to the rollout model using FiLM or context tokens. Minimum context:

- subtype
- HR/HER2 status if available separately
- collection
- treatment arm when available
- baseline FTV
- current start visit

Implementation:

- Extend graph loading to attach cohort metadata.
- Add a small context encoder.
- Inject context into node embeddings through FiLM: `h = gamma(c) * h + beta(c)`.
- Save context config in checkpoint for simulation/MC compatibility.

New flags:

- `--use-clinical-context`
- `--context-fields subtype collection baseline_ftv treatment_arm`
- `--context-dim 32`

This should be a second-wave change because it changes checkpoint loading and inference code.

### 5. Direct patient-level T3/response heads

Keep the supervoxel rollout, but add patient-level auxiliary heads:

- T3 FTV regression head
- FTV ratio regression head
- pCR/response probability head

Do not report `P(FTV_T3 < 0.1 mL)` as pCR. It should be renamed to residual-disease probability or removed from pCR reporting.

Training:

- train pCR head only where labels are present
- use fold-held-out validation
- report AUC, Brier, ECE, and calibration slope

New flags:

- `--lambda-t3-ftv-head`
- `--lambda-pcr-head`
- `--pcr-head-input rollout_latent`

This targets response prediction directly instead of expecting a hard FTV threshold to approximate pathology.

### 6. Richer graph features and biology-aware edges

This is the largest change and should run after the model-objective changes:

- UCSF PE/SER/FTV maps
- DWI/ADC for ACRIN-6698 where available
- PE/SER heterogeneity
- shape descriptors
- habitat labels
- mixed spatial/feature kNN edges
- habitat-aware edge attributes

This belongs to the v2 graph build pipeline, not just the rollout trainer.

## Wave 2 overnight grid

After implementing items 1-3, run a compact 5-fold grid. Do not include clinical context or graph-feature changes in this grid unless those interfaces have already been tested.

| Tag | Full rollout | log/ratio FTV | alive count | Purpose |
|---|---|---|---|---|
| `ar_ftv_alive_mid` | yes | no | current alive mass/BCE | isolate full autoregressive loss |
| `ar_logftv_alive_mid` | yes | yes | current alive mass/BCE | test log-volume tail handling |
| `ar_logftv_alive_count` | yes | yes | calibrated count loss | test alive-count fix |
| `ar_logftv_highvol` | yes | yes + high-volume weights | calibrated count loss | target high-volume failures |

Recommended training settings:

- `EPOCHS=160`
- `PATIENCE=45`
- `FREEZE_EPOCHS=8`
- `LR=1e-4`
- `EPS_MAX=0.5`
- `EPS_WARM=10`
- `EPS_ANNEAL=60`
- 5-fold array, throttle 4 or 5

Evaluation chain:

1. training grid
2. deterministic simulation eval for every tag
3. summary CSV/JSON
4. conditional MC for top two tags
5. final conditional MC with full cloud metrics for the winner

## Wave 3: context and response

Once the rollout objective is stable, add clinical/context conditioning and response heads.

Run these as separate ablations:

| Tag | Added component | Reason |
|---|---|---|
| `ar_logftv_context` | clinical/context FiLM | tests whether subtype/collection/treatment context improves trajectory |
| `ar_logftv_pcrhead` | patient response head | tests direct response prediction without changing graph features |
| `ar_logftv_context_pcrhead` | both | tests combined response-aware rollout |

Gate:

- pCR AUC must beat the FTV-only oracle or it should not be a headline claim.
- Brier and ECE must be reported with AUC.
- Trajectory metrics must not regress materially.

## Wave 4: richer graph/data version

After the objective and context changes are stable, move to richer graph inputs:

1. Build UCSF PE/SER/FTV-map graphs.
2. Add DWI/ADC features for ACRIN-6698 and missingness flags for ISPY2.
3. Add habitat labels.
4. Add biology-aware edge attributes.
5. Retrain the best Wave 3 model on the richer graph version.

This should be a separate experimental family because it changes the input distribution and checkpoint dimensionality.

## Recommended immediate action

For tonight:

1. Let `48329` finish.
2. Let `48335` and `48336` evaluate/summarize.
3. Run conditional MC on the best one or two `runs/consistent_forecaster_v2_bio/<tag>` candidates.
4. Do not start the full autoregressive-loss grid until the Wave 2 code changes are implemented and smoke-tested.

For the next coding session:

1. Implement full autoregressive multi-start loss.
2. Add explicit log-volume/FTV-ratio losses.
3. Add alive-count calibration loss and validation diagnostics.
4. Add a new sbatch grid for the four Wave 2 tags.
5. Smoke test with `PATIENT_LIMIT=12`, `EPOCHS=2`.
6. Launch the 20-task Wave 2 grid overnight with dependent simulation eval and summary.
