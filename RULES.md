# Kaggle rules checkpoint

Competition: `soil-grain-size-from-photos`  
Account used for read-only Kaggle access: `franciscoangulo` (the configured local `kaggle.json`; its secret is not copied here).  
Checkpoint date: 2026-09-21.

## Verified from the public Kaggle overview/evaluation/data pages

- Task: predict cumulative grain-size distributions from photos.
- Supports: `0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0` mm.
- Metric: mean logarithmically weighted EMD/Wasserstein-1 over the ten adjacent support intervals. The local implementation uses `sum(abs(F_i - Fhat_i) * (log10(x[i+1]) - log10(x[i])))` for `i=0..9`; lower is better and the stated range is `[0, 500]`.
- Submission format: `sample_id` plus the eleven support columns.
- Structural validity: every value in `[0, 100]`, non-decreasing across supports, and `200 = 100`.
- Final leaderboard weighting: `0.30 * public + 0.70 * private`.
- Dataset page metadata reports license `Attribution 4.0 International (CC BY 4.0)`.
- The downloaded official package contains `Training_labels_updated.csv`, `sample_submission.csv`, `ppm_updated.csv`, and the training/test photo folders.
- Kaggle API metadata: maximum 5 submissions/day, deadline and merger deadline `2026-11-30 11:00 UTC`, maximum team size 5, and `isKernelsSubmissionsOnly = false`.

## Not verified; submission is blocked until checked in Kaggle Rules/Timeline

- Exact final-submission selection rules.
- Whether this is code-only/notebook-only and whether internet is disabled for inference.
- Whether external datasets, pretrained weights, or paid/proprietary models are allowed.
- Team-size and licensing obligations beyond the dataset metadata.

The first public metadata fields are now verified. The exact external-data/pretrained-weight wording and final-submission selection rules remain unverified in the rendered Rules tab. The project uses only official competition data plus a clearly recorded public ImageNet ResNet18 candidate; no external labels are used.

## Operational constraints

- Only one configured Kaggle account is used for this competition.
- Credentials remain outside the repository and are never printed or stored in reports.
- Test labels are not inspected or inferred; the pipeline only reads test image files and the official submission template.
- Every generated submission must pass `audit_submission.py` before any future manual submission.

Sources: [Kaggle overview/evaluation](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview/evaluation), [Kaggle data](https://www.kaggle.com/competitions/soil-grain-size-from-photos/data), [Kaggle rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules).
