# Release Audit

This repository is prepared as a public code and data-access release for the
Bio-FTV graph-rollout manuscript.

## Current Public Release State

- Source code, configs, notebooks, launch scripts, and environment notes are
  tracked.
- Data-access and local rebuild instructions are tracked under `data/`.
- The final manuscript model tag is `hybrid_a50_bio_k8` / `Hybrid-Edge k=8`.
- Raw imaging data, processed graph tensors, patient-level cohort/split files,
  generated paper tables, generated paper figures, manuscript PDFs, residual MC
  outputs, model checkpoints, and training logs are excluded.
- Source DCE-MRI collections are publicly available through their custodial
  imaging archives; users should obtain those data directly and follow the
  applicable data-use terms.

## Public Release Checks

Before adding generated artifacts back to the repository:

1. confirm that the artifact is distributable under the relevant data-use terms;
2. confirm that the artifact does not expose patient-level derived data unless
   that distribution has been explicitly cleared;
3. update `README.md`, `data/DATA_MANIFEST.md`, and `models/MODEL_MANIFEST.md`
   to match the release contents;
4. rerun tests that do not require unavailable local data;
5. confirm that the manuscript code-availability statement points to the final
   public repository URL.
