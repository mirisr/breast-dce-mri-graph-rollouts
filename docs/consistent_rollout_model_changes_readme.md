# Consistent Graph Rollout Model Changes: Plain-Language README

## What this model is trying to do

The consistent graph rollout model is a tumor trajectory model.

For each patient, the tumor is represented as a graph. Each node is a
supervoxel, meaning a small 3D region of tumor tissue. Each node has:

- a 3D position in the tumor,
- imaging features,
- a volume-like feature,
- an alive/dead status across visits.

The visits are:

- `T0`: baseline, before treatment
- `T1`: early treatment
- `T2`: mid treatment
- `T3`: later treatment / near surgery

The model sees one or more observed visits and predicts what the tumor graph
will look like at the next visit. If it predicts repeatedly from its own
outputs, it can roll forward from `T0` to `T1`, then `T2`, then `T3`.

The basic question is:

> Given what this patient's tumor looks like now, what should the tumor look
> like at the next MRI visit?

## Why this matters

This model is not just trying to draw a plausible tumor. It is trying to
predict biologically meaningful treatment response.

A good rollout model should predict:

- where tumor regions move or shrink,
- which tumor regions disappear,
- how the total active tumor burden changes,
- whether the trajectory looks like response or non-response.

The Monte Carlo analysis showed that the model is useful, but it has important
failure modes:

- It underpredicts total tumor volume.
- It underpredicts how many supervoxels remain alive.
- Its uncertainty intervals need conformal correction to reach 90% coverage.
- The current pCR proxy is not a real pCR probability.

The model changes were added to address those issues directly.

## What FTV means

FTV means Functional Tumor Volume.

In this project, FTV is the estimated volume of actively enhancing tumor tissue
from breast MRI. It is not just the geometric size of the tumor. It is meant to
capture tumor tissue that is functionally active according to contrast-enhanced
MRI signal.

In practical terms:

- high FTV means more active tumor burden,
- falling FTV during treatment usually means the tumor is responding,
- persistent or increasing FTV suggests poorer response,
- FTV is one of the most clinically meaningful imaging summaries in I-SPY-style
  breast MRI response modeling.

FTV is important because pCR, pathological complete response, is only known
after surgery. MRI-based FTV gives a way to estimate treatment response earlier
from imaging.

But FTV is not the same as pCR. A patient can have low FTV and still not have
pCR, or have residual imaging signal that does not map perfectly to pathology.
So FTV is a strong response signal, but it should not be treated as a complete
replacement for a pCR model.

## What the original rollout model did

The core model is `LSGCForecaster`.

It uses the LSGC graph neural network backbone and predicts three things for
each supervoxel:

1. `delta_pos`: how the node position changes by the next visit.
2. `delta_feat`: how the node imaging features change by the next visit.
3. `alive_logit`: whether that supervoxel is still present/alive at the next
   visit.

So if a node at `T0` represents a small active tumor region, the model predicts:

- where that region moves,
- how its imaging features change,
- whether it survives to `T1`.

Then the same logic can be repeated to roll forward to `T2` and `T3`.

## Problem with the original training setup

The original version mostly learned one-step transitions.

That means it learned:

> Given `T0`, predict `T1`.

or:

> Given `T1`, predict `T2`.

This is easier than the real rollout problem:

> Given `T0`, predict `T1`, then use the predicted `T1` to predict `T2`, then
> use the predicted `T2` to predict `T3`.

The second version is harder because errors accumulate. A small mistake at
`T1` can become a larger mistake by `T3`.

The Monte Carlo run showed exactly this kind of issue. The model's deterministic
rollout systematically underpredicted FTV by about 12 to 13 mL. That means the
model learned a tumor trajectory that was too aggressively shrinking the tumor
burden.

## Change 1: scheduled sampling

Scheduled sampling makes training look more like inference.

Without scheduled sampling, the model is usually trained with real observed
history:

```text
real T0 -> predict T1
real T1 -> predict T2
real T2 -> predict T3
```

But at inference time, the model often has to use its own predictions:

```text
real T0 -> predicted T1 -> predicted T2 -> predicted T3
```

Scheduled sampling mixes these two modes during training. Early in training,
the model mostly uses real observations. Later in training, it increasingly
uses its own predicted states.

Why this was added:

- to make the model robust to its own mistakes,
- to reduce rollout drift,
- to improve long-horizon prediction from `T0` to `T3`.

In the code, this is controlled by:

- `--scheduled-sampling`
- `--eps-max`
- `--eps-warmup-epochs`
- `--eps-anneal-epochs`

## Change 2: random-start conditioning

The model can now train from different observed start visits.

Instead of always starting from `T0`, training can start from:

- `T0`, predicting `T1`, `T2`, `T3`
- `T1`, predicting `T2`, `T3`
- `T2`, predicting `T3`

Why this was added:

- the model should work when only baseline is available,
- but it should also improve when later observed visits are available,
- the same model should support conditional forecasting from multiple clinical
  time points.

In the code, this is controlled by:

- `--random-start`

## Change 3: horizon cloud loss

The model already predicts node-level changes, but node-by-node matching is not
the whole story. A tumor is a 3D cloud of tissue regions.

The horizon cloud loss compares the predicted tumor cloud to the observed tumor
cloud at a future visit.

Why this was added:

- node-level losses can look acceptable while the whole tumor shape is wrong,
- cloud-level losses encourage the full predicted tumor geometry to match the
  observed tumor geometry,
- this is especially important for long rollouts to `T3`.

In the code, this is controlled by:

- `--lambda-cloud`
- `--lambda-cloud-feat`
- `--lambda-horizon`

## Change 4: aggregate FTV loss

This is one of the most important changes.

The model does not only need each individual node to look right. It also needs
the total tumor burden to be right.

The aggregate FTV loss computes:

```text
predicted total FTV = sum(predicted node volume * predicted alive probability)
observed total FTV  = sum(observed node volume)
```

Then it penalizes the difference in log space:

```text
abs(log1p(predicted FTV) - log1p(observed FTV))
```

Why log space?

- raw volume has a long tail,
- a few very large tumors can dominate the loss,
- log space makes small and large tumors easier to train together.

Why this was added:

- the Monte Carlo analysis showed systematic deterministic FTV underprediction,
- the model needed a direct training signal for total tumor burden,
- FTV is clinically meaningful and should be optimized directly, not only
  indirectly through node features.

In the code, this is controlled by:

- `--lambda-ftv`

## Change 5: alive BCE loss

Each supervoxel has an alive/dead target at the next visit.

Alive BCE is a standard binary classification loss. It teaches the model:

> For this node, should it still exist at the next visit?

Why this was added:

- tumor shrinkage is partly about regions disappearing,
- the model needs to know which supervoxels persist and which vanish,
- this directly affects predicted FTV because dead regions should not contribute
  to active tumor volume.

In the code, this is controlled by:

- `--lambda-alive-bce`

## Change 6: alive-mass loss

Alive BCE works node by node. Alive-mass loss works at the patient/visit level.

It compares:

```text
predicted alive mass = sum(predicted alive probabilities)
observed alive count = number of observed alive supervoxels
```

Why this was added:

- the Monte Carlo calibration showed that alive count was badly biased,
- node-level BCE can still produce the wrong total number of alive regions,
- alive mass is directly tied to predicted tumor burden and geometry.

In the code, this is controlled by:

- `--lambda-alive-mass`

## Change 7: biology-aware checkpoint selection

Previously, the best checkpoint was selected mostly using geometry-style rollout
metrics.

The new trainer can also include FTV and alive-count validation errors when
choosing the best checkpoint.

Why this was added:

- the best-looking geometric model may not be the best biological model,
- the model should not be selected only because positions look good,
- response modeling needs tumor burden and alive-count behavior to matter.

In the code, this is controlled by:

- `--selection-ftv-weight`
- `--selection-alive-weight`

## What is running on Cradle

The current Cradle grid tests five versions of these biology-aware losses.

| Tag | FTV loss | alive mass | alive BCE | Question |
|---|---:|---:|---:|---|
| `bio_ftv005_alive001` | 0.05 | 0.01 | 0.05 | Does a light biology penalty help? |
| `bio_ftv010_alive002` | 0.10 | 0.02 | 0.05 | Does a moderate penalty work better? |
| `bio_ftv020_alive005` | 0.20 | 0.05 | 0.05 | Does a strong penalty fix bias or overcorrect? |
| `bio_ftv010_alive000` | 0.10 | 0.00 | 0.00 | What does FTV supervision do by itself? |
| `bio_ftv000_alive002` | 0.00 | 0.02 | 0.05 | What does alive supervision do by itself? |

Each variant is trained across five folds. That lets the results be compared
under the same held-out-patient protocol.

## What success should look like

The goal is not just lower training loss.

A successful model should show:

- lower T3 FTV bias,
- lower T3 FTV absolute error,
- lower alive-count error,
- no major regression in spatial metrics like SWD or Chamfer,
- better Monte Carlo calibration after retraining,
- narrower conformal intervals at the same 90% coverage.

The most important immediate question is:

> Did direct FTV/alive supervision fix the systematic underprediction of tumor
> burden without damaging spatial rollout quality?

## What these changes do not solve yet

These changes improve the rollout model, but they do not fully solve response
prediction.

Still missing:

- full autoregressive multi-start loss across all future visits,
- explicit node-level log-volume targets,
- stronger alive-count calibration,
- clinical/context conditioning,
- direct pCR response head,
- richer graph features from UCSF PE/SER/FTV maps and DWI/ADC where available.

Those are the next waves in the model-upgrade plan.

## Simple summary

The old model mainly learned:

> Move each tumor region forward one visit.

The upgraded model is being trained to learn:

> Roll the tumor forward over treatment, keep the 3D shape plausible, predict
> which regions survive, and match the patient's total functional tumor volume.

FTV matters because it is the MRI-derived measure of active tumor burden. If
the model gets FTV wrong, it may still draw a plausible tumor, but it is missing
one of the most important clinical signals of treatment response.
