# Stage-1 Forecaster — cross-fold summary

## Per-method mean ± std across folds

| match_method | pos_l1 (mm) | feat MSE | position EMD (mm) | alive AUC |
|---|---:|---:|---:|---:|
| nn | 5.66 ± 0.15 | 88.0 ± 22.7 | 2.48 ± 0.09 | 0.769 ± 0.010 |
| sinkhorn | 7.28 ± 0.55 | 113.2 ± 26.7 | 2.55 ± 0.25 | 0.702 ± 0.010 |

## Per-fold detail

| match | fold | best_epoch | pos_l1 | feat_mse | emd | alive_auc |
|---|---:|---:|---:|---:|---:|---:|
| nn | 0 | 39 | 5.49 | 127.1 | 2.46 | 0.780 |
| nn | 1 | 45 | 5.70 | 78.3 | 2.49 | 0.755 |
| nn | 2 | 39 | 5.65 | 83.2 | 2.41 | 0.770 |
| nn | 3 | 33 | 5.87 | 83.1 | 2.62 | 0.775 |
| nn | 4 | 34 | 5.57 | 68.3 | 2.40 | 0.765 |
| sinkhorn | 0 | 42 | 6.89 | 157.6 | 2.37 | 0.715 |
| sinkhorn | 1 | 2 | 8.07 | 111.8 | 2.97 | 0.690 |
| sinkhorn | 2 | 28 | 7.21 | 108.0 | 2.44 | 0.698 |
| sinkhorn | 3 | 30 | 7.53 | 102.7 | 2.60 | 0.708 |
| sinkhorn | 4 | 41 | 6.67 | 85.8 | 2.36 | 0.696 |
