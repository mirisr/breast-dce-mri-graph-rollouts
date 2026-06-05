# Release Audit

This repository has been updated toward the current TMI manuscript state, but it
should remain private until the checks below are complete.

## Current Release State

- Current manuscript source and PDF are under `paper/`.
- Current paper figures and tables are under `paper/figures/` and
  `paper/tables/`.
- Current training, edge-ablation, deterministic-evaluation, and MC scripts are
  under `experiments/`.
- Independent Breast-MRI-NACT-Pilot external stress-test helpers are under
  `experiments/breast_nact_external/` and `cradle/`.
- Current result roots used by the paper are under `results/`.
- The final manuscript model is `hybrid_a50_bio_k8` / `Hybrid-Edge k=8`.
- The older `bio_ftv020_alive005` model is a historical Endpoint+Active
  comparator.
- The TMI submission package should be a single complete manuscript PDF. Do not
  submit manuscript-style supplemental documents with expanded text, figures, or
  tables.

## Must Check Before Public GitHub Release

1. Decide whether patient-level derived graph tensors in `data/ispy2/` can be
   public, or replace them with rebuild instructions and TCIA access guidance.
2. Decide whether patient-level MC draws and per-patient result tables in
   `results/` can be public, or replace them with aggregate paper tables.
3. Mirror final Hybrid-Edge fold checkpoints into `models/hybrid_a50_bio_k8/`
   if trained weights will be included.
4. Remove any cluster logs, scratch outputs, local paths, temporary notebooks,
   credentials, or machine-specific files.
5. Re-run the manuscript build from `paper/`.
6. Run the repository tests that do not require unavailable private data.
7. Confirm the final GitHub URL and update the manuscript code-availability
   statement.
8. Add final institution-specific IRB or exemption language before submission.

## Public-Safe Default

If there is any uncertainty about patient-level derived files, make the first
public release code-only plus aggregate paper artifacts:

- include scripts, configs, aggregate tables, and figures;
- include TCIA data-access and rebuild instructions;
- exclude raw imaging, derived patient-level graph tensors, patient-level MC
  samples, and trained weights unless each has been cleared for distribution.
