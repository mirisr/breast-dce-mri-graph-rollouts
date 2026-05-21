# Breast-MRI-NACT-Pilot Four-Visit Deterministic External Summary

Endpoint: source-trained T0-to-T3 deterministic rollout scored on external four-visit patients.
Fold rows score each source fold separately; ensemble rows average source-fold predictions per external patient.

## Endpoint Summary

```
              model         model_label    conditioning predicted_visit  n_patients  n_fold_rows  row_mae_ml  row_bias_ml  ensemble_mae_ml  ensemble_bias_ml  mean_obs_ftv_ml  mean_pred_ftv_ml  mean_fold_sd_pred_ftv_ml  row_swd_mm_mean  row_chamfer_mm_mean  row_dice_mean  row_alive_count_abs_err_mean
bio_ftv020_alive005     Endpoint+Active rollout_from_T0              T3          11           55   20.171914    19.707702        20.170271         19.707702        19.692993         39.400694                  1.008536         2.319501             4.283291       0.041776                      0.090909
           no_edges    No-edge endpoint rollout_from_T0              T3          11           55   19.208009    18.818389        19.208009         18.818389        19.692993         38.511382                  1.256331         2.323784             4.264116       0.046809                      0.090909
      radial_bio_k8 Radial-biologic k=8 rollout_from_T0              T3          11           55   16.707344    16.362720        16.704409         16.362720        19.692993         36.055713                  2.846769         2.260469             4.255396       0.081187                      0.090909
  hybrid_a50_bio_k8     Hybrid-Edge k=8 rollout_from_T0              T3          11           55   12.405414    12.130699        12.401936         12.130699        19.692993         31.823692                  3.589729         2.280744             4.265867       0.045946                      0.090909
```

## Exclusion Rule

One V4 candidate failed the graph support check after segmentation mapping and was excluded before scoring.
