# Soil GSD: retained baseline and controlled comparisons

Francisco Angulo de Lafuente · [ORCID](https://orcid.org/0009-0001-1634-7063)

The task predicts an eleven-point cumulative grain-size distribution (GSD) for a soil sample
from its photographs: the percentage passing each grain diameter. Earth mover's distance
(EMD) is the competition's error metric; lower values are better. CLS means the transformer
class token, used here as an image embedding.

The retained official submission is **56566750**, public EMD **60.79507**. The authenticated
account readback on 2026-11-09T15:48:47.014629+00:00 still lists 13 submissions. This archive records
recovery of that baseline and three exploratory comparisons. It creates no competition entry.

The original training recipe was reproduced: EMD **34.45786163217882** across 20 shuffled
five-fold partitions of 24 physical samples and 127 training photographs. Its already-submitted
six-decimal CSV was reproduced byte for byte. These training and public-test metrics measure
different data; a lower training value does not predict the public leaderboard score.

| Fixed comparison | Training EMD | Reduction vs control | Exploratory 95% CI of reduction | Gate |
|---|---:|---:|---|---|
| SOIL-PATCH-STATS-20261109 | 35.304894 | -0.847032 | [-2.639749, 0.806187] | Fail |
| SOIL-LARGE-CLS-20261109 | 37.399468 | -2.941607 | [-6.494129, 0.558676] | Fail |
| SOIL-ALL5-VIEWS-20261109 | 35.300093 | -0.842232 | [-1.424222, -0.325341] | Fail |

Lower EMD is better. Positive reduction favours the candidate. The gate required at least 1.0 EMD
reduction and a lower interval endpoint above 0. All three candidates failed; none triggered new
test features or an official submission. The best existing official submission was retained.

The baseline concatenates768 DINOv2-small mean/std features with 1,794 recovered physical features,
uses fold-local scaling, `Ridge(alpha=1, solver="lsqr")`, and a square-root histogram-mass target.
Patch statistics appended 1,536 features. Large CLS replaced the small representation using pinned
DINOv2 ViT-L. The comparison with five views changed only pooling to include the first physical crop. The underlying
preprocessing produces five square physical crops; the baseline uses indices1–4.

Uncertainty uses 10,000 paired bootstrap draws over 24 physical samples, seed20261109, after averaging
each sample's losses across 20 reused partitions. This is exploratory evidence from historically
selected samples. The interval does not correct prior model selection or the Android/iPhone
acquisition shift. There are 24 physical samples, not 480 independent validation observations.

[Associated Kaggle source notebook](https://www.kaggle.com/code/franciscoangulo/soil-gsd-audit-20261109-patch-large-all-views)
· [GitHub project](https://github.com/Agnuxo1/soil-grain-size-from-photos)
· [Competition and rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules)
· [Reproduction boundaries](REPRODUCE.md)

Original project work, including Claude-assisted preparation, was recovered and preserved. Codex
performed the recovery and controlled comparisons. Fresh-context independent readiness reviews
found admission/provenance gaps; the authoring agent executed their counterexamples and corrected
them. Follow-up reviewers checked the closures. This is an internal technical review, not an
external certification. The evidence records both failed experiments and verification corrections.

Original code is MIT. Public DINOv2/timm models retain Apache-2.0 and their upstream notices;
weights are linked, not redistributed. Competition images, labels, features and prediction arrays
are absent. This is research code and evidence; laboratory soil testing remains a separate task.

**Attribution.** Cite Francisco Angulo de Lafuente and preserve the MIT copyright and permission
notice in distributed copies. Upstream materials retain their own terms.
