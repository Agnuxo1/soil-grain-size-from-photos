# Soil R02: a controlled direct-CDF readout ablation

One fixed target-space change **worsened** exploratory training cross-validation:
EMD **34.4578616322 → 37.4863959115** (lower is better). Control minus candidate
is **−3.0285342793**, with a paired physical-sample bootstrap 95% interval
**[−5.4219109307, −0.5257701840]**. Both fixed promotion gates failed. No test
prediction, new submission or official improvement resulted from this experiment.

The control uses frozen DINO/physical features and fold-local standardized Ridge
on square-root bin masses. R02 keeps exactly the same features, folds, Ridge alpha1
and lsqr solver, and predicts ten CDF percentages directly. It projects each curve
with bounded weighted L2 isotonic regression and appends100. Ridge still minimizes
squared error; this is not a claim to optimize the official absolute-error metric.
Direct CDF regression itself is established modeling practice, not a novelty claim.

There are24physical samples and20five-fold partitions (100fits/configuration),
seeds20260925..20260944. All100control fits reproduce the frozen OOF predictions.
The bootstrap first averages the paired loss difference over repeats for each
sample, then resamples24physical-sample clusters10,000times, seed20261110. The
fixed gate required at least1EMD reduction and a strictly positive lower95%bound.
No parameters or thresholds were changed after observing this result.

**Scope and limits.** These same24samples have a substantial historical feature
and model selection history. Folds overlap, and repeated splits add no new physical
samples. The interval does not correct all that prior selection or establish
performance on an independent cohort. Android/iPhone camera shift remains open.
This CPU experiment took6.22s excluding interpreter startup. It is a negative
methodological finding, not an official score, rank, prize, or prediction thereof.

`source/` contains the exact sealed evaluation/readout/control source, protocol
and input hash contract. The original control has a historical main entry point;
the R02 evaluator only imports its pure functions. Do not execute that historical
main as the R02 experiment. Readiness review executed four new synthetic checks;
a separate loop-based metric/bootstrap implementation reproduced the result.

`replay.py` recomputes the published statistical comparison using only24anonymous
per-sample **mean differences** in RESULT.json. It performs no training and has no
access to images, targets, identifiers, descriptors, predictions, model weights,
private test labels or platform credentials. These anonymous differences support
statistical audit; they are not the omitted training data or a full model rerun.

**Full reproduction prerequisites.** `source/evaluate.py` requires a sibling
`inputs/` folder with the four exact private training-only snapshots listed in
`source/INPUTS.json`, their pinned original control paths, and the pinned runtime.
Those are intentionally omitted here. The data/feature construction and rights
remain documented in the [earlier source audit](https://github.com/Agnuxo1/soil-grain-size-from-photos/tree/main/research/2026-11-09).
An end-to-end clean-room feature rebuild is not claimed by this archive.

The source is shared in the [associated competition notebook](https://www.kaggle.com/code/franciscoangulo/soil-gsd-audit-20261109-patch-large-all-views)
before additive GitHub and Hugging Face copies, as required by the
[competition rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules).
The current competition deadline was verified as30November2026,11:00UTC. Code is
MIT; this does not grant rights to redistribute the omitted competition inputs.
Visible attribution is included below. No hidden instructions or executable
watermarks are used.

<!-- author-background:start -->
## Author background

**Francisco Angulo de Lafuente**, independent AI researcher · [ORCID](https://orcid.org/0009-0001-1634-7063) · [arXiv](https://arxiv.org/a/angulodelafuente_f_1.html) · [Google Scholar](https://scholar.google.com/citations?user=6nOpJ9IAAAAJ) · [ResearchGate](https://www.researchgate.net/profile/Francisco-Angulo-Lafuente-3) · [Figshare](https://figshare.com/authors/Francisco_Angulo_de_Lafuente/22601234) · [Wikipedia](https://es.wikipedia.org/wiki/Francisco_Angulo_de_Lafuente)

Earlier research at ECOFA focused on biotechnology for clean energy and on second-generation biofuels obtained by microbiological processing of organic waste. Two Spanish patents document that work:

- **ES 2273594 B1**, *Process for obtaining fuels from organic waste*. Filed 5 September 2005, grant published 16 February 2008. [OEPM](https://www.oepm.es/pdf/ES/0000/000/02/27/35/ES-2273594_B1.pdf) · [Google Patents](https://patents.google.com/patent/ES2273594B1/en)
- **ES 2341194 B1**, *Method for obtaining paraffin by a biological process with microorganisms from organic matter, and its refining for use as fuel*. Filed 11 September 2007, grant published 26 April 2011. Also published as EP2196519 and WO2009034217. [OEPM](https://www.oepm.es/pdf/ES/0000/000/02/34/11/ES-2341194_B1.pdf) · [Google Patents](https://patents.google.com/patent/ES2341194B1/en)

These are historical publication and grant records under the inventor name Francisco Angulo Lafuente, and both have since lapsed. They are cited as evidence of a research trajectory and make no statement about current enforceability.
<!-- author-background:end -->
