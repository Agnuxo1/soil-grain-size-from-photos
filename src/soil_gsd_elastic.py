"""Regularized multi-target linear candidate for the soil-GSD micro-dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import soil_gsd_pipeline as base


def make_estimator():
    return make_pipeline(
        StandardScaler(),
        MultiTaskElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=5000, tol=1e-3, selection="random", random_state=42),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/elastic"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = base.find_label_csv(args.data_dir)
    template = base.find_submission_csv(args.data_dir)
    if labels is None or template is None:
        raise FileNotFoundError("Both labelled CSV and sample submission are required.")
    train = base.build_bundle(args.data_dir, labels, include_targets=True, strict=True)
    test = base.build_bundle(args.data_dir, template, include_targets=False, strict=True)
    splitter = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros_like(train.targets)
    for train_idx, valid_idx in splitter.split(train.features):
        model = make_estimator()
        model.fit(train.features[train_idx], train.targets[train_idx])
        oof[valid_idx] = base.repair_cdf(model.predict(train.features[valid_idx]))
    cv = base.log_weighted_emd(train.targets, oof)
    model = make_estimator()
    model.fit(train.features, train.targets)
    pred = base.repair_cdf(model.predict(test.features))
    output = pd.DataFrame(pred, columns=base.TARGET_COLUMNS)
    output.insert(0, "sample_id", test.sample_ids)
    output.to_csv(args.output_dir / "submission.csv", index=False, float_format="%.6f")
    report = {"command": "elastic", "cv_score": cv, "output": str(args.output_dir / "submission.csv"), "alpha": 1.0, "l1_ratio": 0.5}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
