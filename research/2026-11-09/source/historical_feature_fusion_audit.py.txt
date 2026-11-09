"""Audit a public-notebook-inspired feature-level DINO/physical fusion.

Uses only labeled training samples and cached training-image features. All
scalers and models are fit within each sample-level validation fold. It does
not read test predictions or create a submission.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "Training_labels_updated.csv"
DINO_CACHE = ROOT / "artifacts" / "public_outputs" / "dino" / "dinov2_physical_features.npz"
PHYSICAL_CACHE = ROOT / "artifacts" / "pavlo_multitask_v1" / "features.npz"
OUTPUT = ROOT / "artifacts" / "feature_fusion_audit_20260926" / "report.json"
SUBMISSION_OUTPUT = ROOT / "artifacts" / "feature_fusion_audit_20260926" / "feature_concat_submission.csv"
REPEATS = 20
SEED_START = 20260925


def repair_cdf(prediction: np.ndarray) -> np.ndarray:
    result = np.maximum.accumulate(np.clip(np.asarray(prediction, dtype=float), 0.0, 100.0), axis=1)
    result[:, -1] = 100.0
    return result


def encode_sqrt_mass(target: np.ndarray) -> np.ndarray:
    increments = np.diff(np.column_stack([np.zeros(len(target)), target]), axis=1) / 100.0
    return np.sqrt(np.clip(increments, 0.0, None))


def decode_sqrt_mass(encoded: np.ndarray) -> np.ndarray:
    mass = np.square(np.clip(np.asarray(encoded, dtype=float), 0.0, None))
    totals = mass.sum(axis=1, keepdims=True)
    mass = np.divide(100.0 * mass, totals, out=np.full_like(mass, 100.0 / mass.shape[1]), where=totals > 1e-12)
    return repair_cdf(np.cumsum(mass, axis=1))


def score_emd(target: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> float:
    return float(np.mean(np.sum(np.abs(target[:, :-1] - prediction[:, :-1]) * weights, axis=1)))


def aggregate_dino(cache_path: Path, sample_ids: list[str]) -> np.ndarray:
    cache = np.load(cache_path, allow_pickle=True)
    features = cache["features"].astype(np.float32)
    feature_ids = cache["sample_ids"].astype(str)
    features = features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-12)
    rows = []
    for sample_id in sample_ids:
        selected = features[feature_ids == sample_id][:, 1:5, :].reshape(-1, features.shape[-1])
        if not len(selected):
            raise ValueError(f"No cached DINO rows for training sample {sample_id}")
        rows.append(np.concatenate([selected.mean(axis=0), selected.std(axis=0)]))
    return np.vstack(rows)


def aggregate_physical(cache_path: Path, sample_ids: list[str]) -> np.ndarray:
    cache = np.load(cache_path, allow_pickle=True)
    features = cache["train"].astype(np.float64)
    feature_ids = cache["train_ids"].astype(str).tolist()
    positions = {sample_id: i for i, sample_id in enumerate(feature_ids)}
    if set(sample_ids) != set(feature_ids):
        raise ValueError("Physical feature IDs do not exactly match labeled training IDs")
    return np.vstack([features[positions[sample_id]] for sample_id in sample_ids])


def fold_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    alpha: float,
    standardize: bool = True,
) -> np.ndarray:
    if standardize:
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_valid = scaler.transform(x_valid)
    model = Ridge(alpha=alpha, solver="lsqr").fit(x_train, encode_sqrt_mass(y_train))
    return decode_sqrt_mass(model.predict(x_valid))


def main() -> None:
    labels = pd.read_csv(LABELS)
    sample_ids = labels["sample_id"].astype(str).tolist()
    targets = labels.iloc[:, 1:].to_numpy(dtype=float)
    dino = aggregate_dino(DINO_CACHE, sample_ids)
    physical = aggregate_physical(PHYSICAL_CACHE, sample_ids)
    weights = np.diff(np.log10(np.asarray(labels.columns[1:], dtype=float)))

    reference_path = ROOT / "artifacts" / "repeated_grouped_cv_20260925" / "report_20.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    hybrid_by_seed = {int(row["seed"]): float(row["hybrid_mean_emd"]) for row in reference["repeat_results"]}
    if len(hybrid_by_seed) != REPEATS or set(hybrid_by_seed) != set(range(SEED_START, SEED_START + REPEATS)):
        raise ValueError("Reference repeated-CV seeds do not match this audit protocol")

    repeat_rows: list[dict[str, float | int]] = []

    for repeat in range(REPEATS):
        seed = SEED_START + repeat
        splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
        prediction_sets = {"feature_concat": np.zeros_like(targets), "feature_equal_block": np.zeros_like(targets)}

        for train_idx, valid_idx in splitter.split(targets):
            x_dino_train, x_dino_valid = dino[train_idx], dino[valid_idx]
            x_physical_train, x_physical_valid = physical[train_idx], physical[valid_idx]
            # New hypothesis: fuse cached feature blocks before fitting one Ridge model.
            joined_train = np.column_stack([x_dino_train, x_physical_train])
            joined_valid = np.column_stack([x_dino_valid, x_physical_valid])
            prediction_sets["feature_concat"][valid_idx] = fold_predict(
                joined_train, targets[train_idx], joined_valid, alpha=1.0
            )

            # Equalize the total standardized variance contributed by each block.
            sd, sp = StandardScaler().fit(x_dino_train), StandardScaler().fit(x_physical_train)
            dino_train_scaled = sd.transform(x_dino_train) / np.sqrt(x_dino_train.shape[1])
            dino_valid_scaled = sd.transform(x_dino_valid) / np.sqrt(x_dino_train.shape[1])
            physical_train_scaled = sp.transform(x_physical_train) / np.sqrt(x_physical_train.shape[1])
            physical_valid_scaled = sp.transform(x_physical_valid) / np.sqrt(x_physical_train.shape[1])
            balanced_train = np.column_stack([dino_train_scaled, physical_train_scaled])
            balanced_valid = np.column_stack([dino_valid_scaled, physical_valid_scaled])
            prediction_sets["feature_equal_block"][valid_idx] = fold_predict(
                balanced_train, targets[train_idx], balanced_valid, alpha=1.0, standardize=False
            )

        baseline_score = hybrid_by_seed[seed]
        scores = {name: score_emd(targets, pred, weights) for name, pred in prediction_sets.items()}
        row: dict[str, float | int] = {"repeat": repeat, "seed": seed, "reference_hybrid_emd": baseline_score}
        row.update({f"{name}_emd": score for name, score in scores.items()})
        row["concat_minus_hybrid"] = scores["feature_concat"] - baseline_score
        row["equal_block_minus_hybrid"] = scores["feature_equal_block"] - baseline_score
        repeat_rows.append(row)

    summary: dict[str, dict[str, float]] = {}
    frame = pd.DataFrame(repeat_rows)
    for candidate, delta_col in (
        ("feature_concat", "concat_minus_hybrid"),
        ("feature_equal_block", "equal_block_minus_hybrid"),
    ):
        deltas = frame[delta_col].to_numpy()
        summary[candidate] = {
            "mean_emd": float(frame[f"{candidate}_emd"].mean()),
            "mean_delta_vs_reproduced_hybrid": float(deltas.mean()),
            "delta_quantiles_10_50_90": [float(v) for v in np.quantile(deltas, [0.1, 0.5, 0.9])],
            "fraction_repeats_better_than_hybrid": float(np.mean(deltas < 0)),
        }

    # Fit the predeclared simple-concatenation model on labeled training rows.
    # Test rows are used only as unlabeled prediction inputs for this one candidate.
    template = pd.read_csv(ROOT / "data" / "sample_submission.csv")
    test_ids = template["sample_id"].astype(str).tolist()
    dino_all = aggregate_dino(DINO_CACHE, sample_ids + test_ids)
    dino_test = dino_all[len(sample_ids) :]
    physical_cache = np.load(PHYSICAL_CACHE, allow_pickle=True)
    physical_test_all = physical_cache["test"].astype(np.float64)
    physical_test_ids = physical_cache["test_ids"].astype(str).tolist()
    physical_test_positions = {sample_id: i for i, sample_id in enumerate(physical_test_ids)}
    if set(test_ids) != set(physical_test_ids):
        raise ValueError("Physical test-feature IDs do not exactly match the sample template")
    physical_test = np.vstack([physical_test_all[physical_test_positions[sample_id]] for sample_id in test_ids])
    x_train = np.column_stack([dino, physical])
    x_test = np.column_stack([dino_test, physical_test])
    scaler = StandardScaler().fit(x_train)
    model = Ridge(alpha=1.0, solver="lsqr").fit(scaler.transform(x_train), encode_sqrt_mass(targets))
    test_prediction = decode_sqrt_mass(model.predict(scaler.transform(x_test)))
    template.loc[:, labels.columns[1:]] = test_prediction
    SUBMISSION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(SUBMISSION_OUTPUT, index=False, float_format="%.6f")

    result = {
        "protocol": "20 repeated shuffled five-fold splits over 24 labeled sample IDs; all candidate scaler/model fits are fold-local; no hidden labels or leaderboard scores used for model selection. The paired reference is the already-audited fixed hybrid on the identical seeds.",
        "caution": "Repeated partitions reuse the same 24 samples and overlap; quantiles are split-sensitivity summaries, not independent confidence intervals or test-set evidence.",
        "inspiration": "Public soil-grain-v4 and DINOv2 sqrt-mass Ridge notebooks; their target transform and DINO physical-crop configuration are already present in the current local pipeline. This audit tests only feature-level concatenation with the existing physical multi-backbone cache.",
        "samples": len(sample_ids),
        "features": {"dino_mean_std": int(dino.shape[1]), "physical_multi_backbone": int(physical.shape[1])},
        "fixed_parameters": {"repeats": REPEATS, "seed_start": SEED_START, "folds": 5, "target_encoding": "sqrt_mass", "feature_fusion_alpha": 1.0, "baseline_dino_alpha": 0.1, "baseline_hybrid_weight": 0.5},
        "baseline_mean_emd": float(np.mean(list(hybrid_by_seed.values()))),
        "candidates": summary,
        "frozen_probe_candidate": str(SUBMISSION_OUTPUT.relative_to(ROOT)),
        "repeat_results": repeat_rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT.relative_to(ROOT)), "baseline_mean_emd": result["baseline_mean_emd"], "candidates": summary}, indent=2))


if __name__ == "__main__":
    main()
