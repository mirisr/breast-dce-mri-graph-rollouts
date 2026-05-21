# Breast-MRI-NACT-Pilot External Stress Test

This directory stores aggregate paper-facing external stress-test summaries. Raw imaging, raw segmentations, graph tensors, per-patient predictions, and MC samples are not included here.

The manuscript uses the four-model T0-to-T3 aggregate table in `paper/tables/external_nact_stress_test_t0t3.csv`.

Included aggregate support files:

- `external_4visit_paper_models_deterministic_summary.md`
- `tables_paper_models/external_4visit_t0t3_deterministic_summary.csv`
- `source_residual_mc_4visit_paper_models/external_source_residual_mc_summary.csv`
- `source_residual_mc_4visit_paper_models/external_source_residual_mc_summary.md`
- `source_residual_mc_4visit_paper_models/external_source_residual_mc_metadata.json`

These files report only the manuscript-facing model families: Endpoint+Active, No-edge endpoint, Radial-biologic k=8, and Hybrid-Edge k=8.
