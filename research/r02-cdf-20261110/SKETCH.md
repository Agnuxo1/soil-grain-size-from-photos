# R02: fixed CDF readout ablation

| Component | Frozen control | One proposed candidate |
|---|---|---|
| Input | 24 samples, 2,562 frozen DINO + physical features | Identical |
| Standardization | Fold-training StandardScaler | Identical |
| Regressor | Ridge(alpha=1, solver=lsqr) | Identical |
| Target | Square root of bin masses | Ten cumulative fractions in percentage points |
| Output validity | Squared nonnegative masses, normalized, cumulated | Weighted L2 isotonic projection of ten predictions into [0,100], append 100 |
| Comparison | 20 five-fold partitions, seeds 20260925..20260944 | Identical |
| Baseline mean | 34.45786163217882 log-weighted EMD | Not measured yet |
| Admission | Mean reduction >=1 and paired physical-sample bootstrap lower 95% bound >0 | No threshold changes |

This isolates the target representation and its necessary decoding. It does not
claim that Ridge minimizes absolute-error EMD. A robust IRLS fit is deliberately
not included: with 19 or 20 training rows and 2,562 features, near-interpolation
can make a residual reweighting ablation uninformative. Previous direct-CDF fits
on different physical feature sets are historical; this exact concatenated-feature
ablation was not found in the inspected source. That is a bounded history check,
not a proof that nobody has tried it elsewhere.

PATCH, LARGE, ALL5VIEWS and SAM remain closed negative results. No test extraction,
test prediction or submission is allowed by this design document. Those require
the fixed gate plus an independent review and a new JEV decision.

Observed rules, authenticated Kaggle UI, 10 November 2026: accepted membership;
deadline 30 November 11:00 UTC; five entries/day, two final selections; noncommercial
competition-data use; code sharing via an associated competition notebook/forum.
Public code copies require that associated release first. No new terms accepted.

Limitations: only 24 historically reused physical samples, overlapping folds and
substantial prior feature/model selection; bootstrap describes this sample-level
comparison, not independent confirmation or a predicted official score. Android
training / iPhone test shift remains unresolved. This readout work is CPU sized.
