"""OOF-tuned blend of handcrafted and frozen-CNN soil-GSD models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import soil_gsd_cnn as cnn
import soil_gsd_pipeline as base


def chosen_oof(oof: dict[str, np.ndarray], report: dict) -> np.ndarray:
    selected = report.get("selected_model", "ridge")
    if selected == "blend":
        weight = float(report["ridge_weight_if_blend"])
        return base.repair_cdf(weight * oof["ridge"] + (1.0 - weight) * oof["extra_trees"])
    if selected == "mean":
        return oof["mean"]
    return oof[selected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hybrid"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/cnn_resnet18/embeddings.joblib"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels = base.find_label_csv(args.data_dir)
    template = base.find_submission_csv(args.data_dir)
    if labels is None or template is None:
        raise FileNotFoundError("Both labelled CSV and sample submission are required.")

    hand_bundle = base.build_bundle(args.data_dir, labels, include_targets=True, strict=True)
    cnn_ids, train_paths, y = cnn.labelled_or_template_rows(args.data_dir, labels, include_targets=True)
    if hand_bundle.sample_ids != cnn_ids:
        raise ValueError("Handcrafted and CNN sample ordering differs.")
    all_train_images = [path for paths in train_paths for path in paths]
    embeddings = cnn.extract_embeddings(all_train_images, args.cache)
    cnn_X = cnn.aggregate_samples(train_paths, embeddings)

    hand_report, hand_oof = base.cross_validate(hand_bundle.features, hand_bundle.targets)
    cnn_report, cnn_oof = base.cross_validate(cnn_X, y)
    hand_oof_selected = chosen_oof(hand_oof, hand_report)
    cnn_oof_selected = chosen_oof(cnn_oof, cnn_report)
    candidates = []
    for hand_weight in np.linspace(0.0, 1.0, 21):
        blended = base.repair_cdf(hand_weight * hand_oof_selected + (1.0 - hand_weight) * cnn_oof_selected)
        candidates.append((base.log_weighted_emd(y, blended), float(hand_weight)))
    local_score, hand_weight = min(candidates, key=lambda item: item[0])

    hand_model = base.fit_final(hand_bundle.features, hand_bundle.targets, hand_report)
    cnn_model = base.fit_final(cnn_X, y, cnn_report)
    test_ids, test_paths, _ = cnn.labelled_or_template_rows(args.data_dir, template, include_targets=False)
    all_test_images = [path for paths in test_paths for path in paths]
    embeddings = cnn.extract_embeddings(all_test_images, args.cache)
    test_cnn_X = cnn.aggregate_samples(test_paths, embeddings)
    test_hand = base.build_bundle(args.data_dir, template, include_targets=False, strict=True).features
    hand_pred = base.predict_final(hand_model, test_hand)
    cnn_pred = base.predict_final(cnn_model, test_cnn_X)
    predictions = base.repair_cdf(hand_weight * hand_pred + (1.0 - hand_weight) * cnn_pred)
    output = pd.DataFrame(predictions, columns=base.TARGET_COLUMNS)
    output.insert(0, "sample_id", test_ids)
    output.to_csv(args.output_dir / "submission.csv", index=False, float_format="%.6f")
    report = {
        "command": "hybrid",
        "handcrafted_cv": hand_report,
        "cnn_cv": cnn_report,
        "hybrid_oof_score": float(local_score),
        "handcrafted_weight": float(hand_weight),
        "cnn_weight": float(1.0 - hand_weight),
        "output": str(args.output_dir / "submission.csv"),
        "structural_checks": {
            "values_in_0_100": bool(np.all((predictions >= 0) & (predictions <= 100))),
            "monotonic_cdf": bool(np.all(np.diff(predictions, axis=1) >= -1e-6)),
            "last_column_is_100": bool(np.allclose(predictions[:, -1], 100.0)),
        },
    }
    (args.output_dir / "hybrid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
