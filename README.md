# Soil grain-size distribution from photos

![Soil GSD header](assets/header.png)

This repository contains a reproducible, CPU-friendly baseline for [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos). The task is to predict an eleven-point cumulative grain-size distribution from one or more photographs of a soil sample.

![Feature pipeline](assets/feature-pipeline.png)

## Method

`src/soil_gsd_pipeline.py` discovers official labels, the sample submission, and image folders without assuming a fixed machine path. For each sample it aggregates matching views and extracts colour, texture, gradient, and spatial-pooling features. It compares standardized Ridge with ExtraTrees and selects a blend only when cross-validation supports it.

The output is projected into the valid physical space: values remain between 0 and 100, are monotone, and end at 100. The competition metric is implemented as the log-diameter-weighted absolute error between cumulative curves.

![Uncertainty and view fusion](assets/uncertainty-and-fusion.png)

## Run

Keep Kaggle data outside version control:

```text
data/
  Training-All_Photos_updated/
  Test_All_Photos/
  Training_labels_updated.csv
  sample_submission.csv
```

Then use the commands documented by the source files. The public copy contains no competition dataset and no invented leaderboard metrics.

![Monotone cumulative curve](assets/monotone-cdf.png)

## Scope and limitations

This is an engineering baseline, not a substitute for laboratory granulometry or geotechnical validation. Photos, labels, and sponsor material remain under competition terms. Always check current Kaggle rules before submitting.

## License

MIT for the original code. See [LICENSE](LICENSE). Dependencies and competition data retain their own licenses.
