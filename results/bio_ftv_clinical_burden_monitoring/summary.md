# Clinical Burden Monitoring Summary

This analysis uses the retained Hybrid-Edge k=8 residual MC rollout and treats pCR only as exploratory.

## Threshold Readouts

| readout | endpoint | score | n | event_rate | auc | sensitivity | specificity | ppv | npv | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Low residual MRI burden | T3 FTV < 5 mL | MC probability P(T3 FTV < 5 mL) | 758 | 0.210 | 0.957 | 0.755 | 0.970 | 0.870 | 0.937 | 120 | 18 | 581 | 39 |
| Low residual MRI burden | T3 FTV < 5 mL | Last observed FTV < 5 mL | 758 | 0.210 | 0.961 | 0.679 | 0.980 | 0.900 | 0.920 | 108 | 12 | 587 | 51 |
| High residual MRI burden | T3 FTV > 20 mL | MC probability P(T3 FTV > 20 mL) | 758 | 0.325 | 0.984 | 0.967 | 0.914 | 0.844 | 0.983 | 238 | 44 | 468 | 8 |
| High residual MRI burden | T3 FTV > 20 mL | Last observed FTV > 20 mL | 758 | 0.325 | 0.986 | 0.972 | 0.902 | 0.827 | 0.985 | 239 | 50 | 462 | 7 |

## Serial Monitoring

| conditioning | n | det_ftv_mae_ml | mc_mean_ftv_mae_ml | mc_median_ftv_mae_ml | raw_90_coverage | raw_90_width_ml | crps | auc_t3_ftv_lt5 | auc_t3_ftv_gt20 | mean_prob_t3_ftv_lt5 | mean_prob_t3_ftv_gt20 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T0 only | 758 | 4.629 | 5.564 | 4.787 | 0.877 | 19.206 | 3.998 | 0.957 | 0.984 | 0.193 | 0.397 |
| T0+T1 | 758 | 4.857 | 5.904 | 5.023 | 0.865 | 20.210 | 4.185 | 0.955 | 0.982 | 0.175 | 0.406 |
| T0+T1+T2 | 758 | 4.572 | 5.675 | 4.731 | 0.860 | 19.606 | 3.944 | 0.954 | 0.985 | 0.162 | 0.408 |
