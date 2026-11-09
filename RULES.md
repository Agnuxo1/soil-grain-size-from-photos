# Kaggle rules checkpoint

Competition: [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos).
Checked on 9 November 2026 against the authenticated rendered rules and evaluation pages.

## Task and evaluation

- Predict an eleven-point cumulative grain-size distribution from photographs.
- Diameter supports in mm: `0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200`.
- EMD averages the sum of absolute CDF percentage differences weighted by adjacent log10 diameter intervals. Lower is better; the stated range is 0–500.
- CSV: `sample_id` plus eleven support columns; values 0–100, nondecreasing, ending at 100.
- Final winners depend solely on the private leaderboard, subject to the rules. The earlier public/private 30/70 weighting statement was incorrect.

## Verified participation and sharing conditions

- Deadline: 30 November 2026, 11:00 UTC. Maximum five daily submissions, two final submissions and five team members.
- Data use: competition and noncommercial academic/research purposes. Data-page licence metadata does not replace these conditions.
- Publicly and equally accessible external data/models are allowed, subject to reasonable cost, rights and host restrictions.
- Private code/data sharing outside a team is prohibited. Shared code must also appear in a competition-associated Kaggle forum/notebook under an OSI-approved licence allowing commercial use.
- Manual labelling or human prediction of validation/test records is prohibited.
- Existing account registration/rules acceptance was observed. This audit accepted no new terms and changed no final-submission selection.

The new source is shared in the [associated public notebook](https://www.kaggle.com/code/franciscoangulo/soil-gsd-audit-20261109-patch-large-all-views). It publishes source and aggregates, without creating a prediction submission. Its public source was retrieved anonymously and checked byte for byte; its CPU run was queued during Kaggle's platform incident.

Credentials stay outside this repository. Generated prediction files must pass the submission audit before any official upload. The November comparisons failed their promotion gates and did not access new test features.

Sources: [official rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules), [official evaluation](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview/evaluation).
