"""End-to-end baseline for Kaggle's soil grain-size-from-photos competition.

The competition dataset is intentionally not checked into this repository.  The
loader is schema-driven so it can run against a Kaggle download without hard
coding the exact folder names used by Kaggle.

Typical commands:
    python src/soil_gsd_pipeline.py inspect --data-dir data
    python src/soil_gsd_pipeline.py train --data-dir data --output-dir artifacts/run
    python src/soil_gsd_pipeline.py predict --data-dir data \
        --model artifacts/run/model.joblib --output artifacts/run/submission.csv
    python src/soil_gsd_pipeline.py smoke-test
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SUPPORT_MM = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0])
TARGET_COLUMNS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RANDOM_STATE = 42


def _normalise(value: object) -> str:
    """Normalise ids and stems for matching across CSVs and filenames."""

    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _matching_variants(value: object) -> set[str]:
    """Return conservative variants for common German transliterations/encoding loss."""

    base = _normalise(value)
    variants = {base}
    for source, replacements in {"ue": ["u", ""], "ae": ["a", ""], "oe": ["o", ""]}.items():
        for replacement in replacements:
            variants.add(base.replace(source, replacement))
    return {variant for variant in variants if variant}


def _target_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for col in TARGET_COLUMNS:
        aliases[_normalise(col)] = col
    return aliases


def resolve_target_columns(columns: Iterable[object]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for column in columns:
        raw = str(column).strip()
        # Exact text wins: 2.0 and 20 (and 6.3 and 63) collapse to the same
        # alphanumeric key, so using only _normalise would silently mislabel
        # columns in pandas-loaded CSV headers.
        exact = next((target for target in TARGET_COLUMNS if raw.lower() == target.lower()), None)
        if exact is not None:
            resolved[str(column)] = exact
            continue
        try:
            numeric = float(raw)
        except ValueError:
            continue
        matches = [target for target in TARGET_COLUMNS if math.isclose(float(target), numeric, rel_tol=0.0, abs_tol=1e-12)]
        if len(matches) == 1:
            resolved[str(column)] = matches[0]
    return resolved


def find_csvs(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*.csv") if p.is_file())


def read_csv_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:  # pragma: no cover - diagnostic path
        warnings.warn(f"Could not read CSV header {path}: {exc}")
        return []


def find_label_csv(data_dir: Path) -> Path | None:
    """Choose the CSV containing sample_id and the maximum target coverage."""

    candidates: list[tuple[int, int, Path]] = []
    for path in find_csvs(data_dir):
        columns = read_csv_header(path)
        target_count = len(resolve_target_columns(columns))
        has_id = any(_normalise(c) in {"sampleid", "id", "sample"} for c in columns)
        if has_id and target_count >= 3:
            name_bias = 1 if "label" in path.name.lower() or "train" in path.name.lower() else 0
            candidates.append((target_count, name_bias, path))
    return max(candidates, key=lambda item: (item[0], item[1], str(item[2])))[2] if candidates else None


def find_submission_csv(data_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in find_csvs(data_dir):
        columns = read_csv_header(path)
        target_count = len(resolve_target_columns(columns))
        has_id = any(_normalise(c) in {"sampleid", "id", "sample"} for c in columns)
        if has_id and target_count >= 3:
            bias = 1 if "submission" in path.name.lower() or "sample" in path.name.lower() else 0
            candidates.append((bias, path))
    # The sample submission is preferred over the training labels.
    preferred = [item for item in candidates if item[0] == 1]
    return sorted(preferred or candidates, key=lambda item: str(item[1]))[0][1] if (preferred or candidates) else None


def find_ppm_csv(data_dir: Path) -> Path | None:
    for path in find_csvs(data_dir):
        columns = {_normalise(column) for column in read_csv_header(path)}
        if {"ppm", "phone"}.issubset(columns) or {"ppm", "camera"}.issubset(columns):
            return path
    return None


def load_ppm_lookup(data_dir: Path) -> list[tuple[str, float]]:
    path = find_ppm_csv(data_dir)
    if path is None:
        return []
    frame = pd.read_csv(path)
    key_columns = [column for column in frame.columns if _normalise(column) in {"phone", "camera"}]
    ppm_column = next(column for column in frame.columns if _normalise(column) == "ppm")
    lookup: list[tuple[str, float]] = []
    for _, row in frame.iterrows():
        value = pd.to_numeric(row[ppm_column], errors="coerce")
        if pd.isna(value):
            continue
        for column in key_columns:
            key = _normalise(row[column])
            if key:
                lookup.append((key, float(value)))
    return lookup


def ppm_for_image(path: Path, lookup: Sequence[tuple[str, float]]) -> float | None:
    stem = _normalise(path.stem)
    matches = [(len(key), value) for key, value in lookup if key in stem]
    return max(matches, key=lambda item: item[0])[1] if matches else None


def find_id_column(columns: Iterable[object]) -> str:
    for column in columns:
        if _normalise(column) in {"sampleid", "id", "sample"}:
            return str(column)
    raise ValueError("No sample id column found. Expected sample_id, id, or sample.")


def discover_images(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def match_images(sample_id: object, image_paths: Sequence[Path]) -> list[Path]:
    """Match one sample to one or more photos using exact and safe substring rules."""

    wanted_variants = _matching_variants(sample_id)
    if not wanted_variants:
        return []
    exact = [p for p in image_paths if _matching_variants(p.stem) & wanted_variants]
    if exact:
        return exact
    if max(map(len, wanted_variants)) < 3:
        return []
    matched = [p for p in image_paths if any(wanted in candidate for wanted in wanted_variants for candidate in _matching_variants(p.stem))]
    return matched


def _resize_array(image: Image.Image, size: int = 128) -> np.ndarray:
    image = image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def extract_image_features(path: Path) -> np.ndarray:
    """Extract compact, CPU-only colour, texture and edge features.

    This intentionally avoids a pretrained network because the competition is a
    micro-dataset: fixed handcrafted descriptors are less likely to overfit and
    run in a fresh Kaggle environment without downloading model weights.
    """

    with Image.open(path) as image:
        rgb = _resize_array(image)
    gray = rgb.mean(axis=2)
    hsv = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB").convert("HSV"), dtype=np.float32) / 255.0
    dx = np.diff(gray, axis=1)
    dy = np.diff(gray, axis=0)
    grad = np.hypot(dx[:-1, :], dy[:, :-1])
    features: list[float] = []

    # Distributional colour descriptors retain soil hue and brightness cues.
    for channel in [rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2], gray]:
        features.extend(np.histogram(channel, bins=16, range=(0.0, 1.0), density=False)[0].astype(np.float32) / channel.size)
        features.extend([float(channel.mean()), float(channel.std()), float(np.quantile(channel, 0.1)), float(np.quantile(channel, 0.9))])

    # Texture and edge summaries approximate grain scale and surface roughness.
    for values in [dx, dy, grad]:
        features.extend([float(np.mean(np.abs(values))), float(np.std(values)), float(np.quantile(np.abs(values), 0.9))])
        features.extend(np.histogram(np.clip(np.abs(values), 0.0, 1.0), bins=12, range=(0.0, 1.0))[0].astype(np.float32) / values.size)

    # Spatial pooling preserves coarse arrangement while remaining resolution-safe.
    for grid in [2, 4]:
        for row in np.array_split(gray, grid, axis=0):
            for cell in np.array_split(row, grid, axis=1):
                features.extend([float(cell.mean()), float(cell.std())])
    return np.asarray(features, dtype=np.float32)


def aggregate_image_features(paths: Sequence[Path], ppm_values: Sequence[float | None] | None = None) -> np.ndarray:
    if not paths:
        raise ValueError("No image paths were supplied for a sample.")
    matrix = np.vstack([extract_image_features(path) for path in paths])
    # Keep the dimensionality identical for one-photo and multi-photo samples;
    # this is essential because train/test often contain different photo counts.
    visual = np.concatenate([matrix.mean(axis=0), matrix.std(axis=0), matrix.max(axis=0)])
    if ppm_values is None:
        return visual
    # Camera calibration is an official competition file. Keep a fixed-size
    # summary even when a row is missing from that metadata table.
    values = np.asarray([value for value in (ppm_values or []) if value is not None and np.isfinite(value)], dtype=np.float32)
    scale = np.array(
        [float(values.mean()), float(values.std()), float(values.min()), float(values.max()), float(1.0 - len(values) / len(paths))]
        if len(values)
        else [0.0, 0.0, 0.0, 0.0, 1.0],
        dtype=np.float32,
    )
    return np.concatenate([visual, scale])


@dataclass
class DatasetBundle:
    sample_ids: list[str]
    features: np.ndarray
    targets: np.ndarray | None
    image_counts: list[int]
    missing_ids: list[str]
    label_csv: str | None = None


def _load_rows(path: Path, include_targets: bool) -> tuple[list[str], np.ndarray | None]:
    frame = pd.read_csv(path)
    id_col = find_id_column(frame.columns)
    ids = frame[id_col].astype(str).tolist()
    if not include_targets:
        return ids, None
    resolved = resolve_target_columns(frame.columns)
    missing = [target for target in TARGET_COLUMNS if target not in resolved.values()]
    if missing:
        raise ValueError(f"{path} is missing target columns: {missing}")
    values = frame[[source for source, target in resolved.items() if target in TARGET_COLUMNS]].copy()
    values = values.rename(columns=resolved)
    values = values[TARGET_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError(f"Non-numeric or missing target values found in {path}")
    return ids, values.to_numpy(dtype=np.float32)


def build_bundle(data_dir: Path, csv_path: Path, include_targets: bool, strict: bool = False, use_ppm: bool = False) -> DatasetBundle:
    ids, targets = _load_rows(csv_path, include_targets=include_targets)
    images = discover_images(data_dir)
    ppm_lookup = load_ppm_lookup(data_dir)
    rows: list[np.ndarray] = []
    kept_ids: list[str] = []
    kept_targets: list[np.ndarray] = []
    counts: list[int] = []
    missing: list[str] = []
    for index, sample_id in enumerate(ids):
        matched = match_images(sample_id, images)
        if not matched:
            missing.append(sample_id)
            continue
        ppm_values = [ppm_for_image(path, ppm_lookup) for path in matched] if use_ppm else None
        rows.append(aggregate_image_features(matched, ppm_values))
        kept_ids.append(sample_id)
        counts.append(len(matched))
        if targets is not None:
            kept_targets.append(targets[index])
    if strict and missing:
        raise FileNotFoundError(f"No image matched {len(missing)} sample ids; examples: {missing[:5]}")
    if not rows:
        raise FileNotFoundError(f"No images matched ids from {csv_path}. Images found: {len(images)}")
    return DatasetBundle(
        sample_ids=kept_ids,
        features=np.vstack(rows).astype(np.float32),
        targets=np.vstack(kept_targets).astype(np.float32) if kept_targets else None,
        image_counts=counts,
        missing_ids=missing,
        label_csv=str(csv_path),
    )


def repair_cdf(predictions: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(predictions, dtype=np.float64), 0.0, 100.0)
    values = np.maximum.accumulate(values, axis=1)
    values[:, -1] = 100.0
    return values.astype(np.float32)


def log_weighted_emd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Official logarithmically weighted EMD for the eleven support points.

    The ten intervals use the cumulative value at their left support point;
    leaving the final 200 mm endpoint out also makes the documented maximum
    score 100 * log10(200 / 0.002) = 500.
    """

    truth = np.asarray(y_true, dtype=np.float64)
    pred = repair_cdf(y_pred).astype(np.float64)
    interval_weights = np.diff(np.log10(SUPPORT_MM))
    return float(np.mean(np.sum(np.abs(truth[:, :-1] - pred[:, :-1]) * interval_weights, axis=1)))


def make_models(n_samples: int) -> dict[str, object]:
    leaf = 1 if n_samples < 15 else 2
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "pls": make_pipeline(StandardScaler(), PLSRegression(n_components=min(6, max(1, n_samples - 1)))),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=leaf,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _fit_predict(model: object, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray) -> np.ndarray:
    fitted = clone(model)
    fitted.fit(x_train, y_train)
    return repair_cdf(fitted.predict(x_valid))


def cross_validate(X: np.ndarray, y: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    n_splits = min(5, len(y))
    if n_splits < 3:
        return {"status": "skipped", "reason": "at least 3 labelled samples are required"}, {}
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = {"mean": np.zeros_like(y, dtype=np.float32), **{name: np.zeros_like(y, dtype=np.float32) for name in make_models(len(y))}}
    for train_idx, valid_idx in splitter.split(X):
        oof["mean"][valid_idx] = repair_cdf(np.tile(y[train_idx].mean(axis=0), (len(valid_idx), 1)))
        for name, model in make_models(len(y)).items():
            oof[name][valid_idx] = _fit_predict(model, X[train_idx], y[train_idx], X[valid_idx])
    scores = {name: log_weighted_emd(y, pred) for name, pred in oof.items()}
    candidates = [(scores[name], name, 1.0) for name in scores]
    if set(oof) >= {"ridge", "extra_trees"}:
        for weight in np.linspace(0.0, 1.0, 21):
            blended = repair_cdf(weight * oof["ridge"] + (1.0 - weight) * oof["extra_trees"])
            candidates.append((log_weighted_emd(y, blended), "blend", float(weight)))
    best_score, best_name, best_weight = min(candidates, key=lambda item: item[0])
    return {
        "status": "ok",
        "n_samples": int(len(y)),
        "n_splits": int(n_splits),
        "model_scores": scores,
        "selected_model": best_name,
        "ridge_weight_if_blend": best_weight if best_name == "blend" else None,
        "oof_score": float(best_score),
        "metric": "log_weighted_emd_official_formula",
    }, oof


def fit_final(X: np.ndarray, y: np.ndarray, cv_report: dict) -> dict:
    models = make_models(len(y))
    selected = cv_report.get("selected_model", "ridge")
    if selected == "mean":
        return {"kind": "mean", "mean": repair_cdf(y.mean(axis=0, keepdims=True))[0]}
    if selected == "blend":
        for model in models.values():
            model.fit(X, y)
        blend_weight = float(cv_report["ridge_weight_if_blend"])
        return {"kind": "blend", "models": models, "ridge_weight": blend_weight}
    model = models.get(selected, models["ridge"])
    model.fit(X, y)
    return {"kind": "single", "model_name": selected, "model": model}


def predict_final(bundle: dict, X: np.ndarray) -> np.ndarray:
    if bundle["kind"] == "mean":
        return np.tile(bundle["mean"], (len(X), 1))
    if bundle["kind"] == "blend":
        ridge = bundle["models"]["ridge"].predict(X)
        extra = bundle["models"]["extra_trees"].predict(X)
        weight = float(bundle["ridge_weight"])
        return repair_cdf(weight * ridge + (1.0 - weight) * extra)
    return repair_cdf(bundle["model"].predict(X))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def inspect_data(data_dir: Path, output_path: Path | None = None) -> dict:
    csvs = find_csvs(data_dir) if data_dir.exists() else []
    images = discover_images(data_dir) if data_dir.exists() else []
    label_csv = find_label_csv(data_dir) if data_dir.exists() else None
    submission_csv = find_submission_csv(data_dir) if data_dir.exists() else None
    report = {
        "data_dir": str(data_dir),
        "exists": data_dir.exists(),
        "csv_files": [str(path) for path in csvs],
        "image_count": len(images),
        "label_csv": str(label_csv) if label_csv else None,
        "submission_csv_candidate": str(submission_csv) if submission_csv else None,
        "expected_targets": TARGET_COLUMNS,
        "ready_for_training": bool(label_csv and images),
    }
    if output_path:
        write_json(output_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def train(data_dir: Path, output_dir: Path, strict: bool = False, use_ppm: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    label_csv = find_label_csv(data_dir)
    if label_csv is None:
        raise FileNotFoundError("Could not find a labelled CSV with sample_id and the 11 target columns.")
    bundle = build_bundle(data_dir, label_csv, include_targets=True, strict=strict, use_ppm=use_ppm)
    cv_report, _ = cross_validate(bundle.features, bundle.targets)
    model = fit_final(bundle.features, bundle.targets, cv_report)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "support_mm": SUPPORT_MM, "target_columns": TARGET_COLUMNS}, model_path)
    pd.DataFrame(bundle.features).to_csv(output_dir / "train_features.csv", index=False)
    report = {
        "command": "train",
        "label_csv": str(label_csv),
        "n_samples_used": len(bundle.sample_ids),
        "n_images_per_sample": {"min": min(bundle.image_counts), "max": max(bundle.image_counts), "mean": float(np.mean(bundle.image_counts))},
        "missing_sample_ids": bundle.missing_ids,
        "feature_count": int(bundle.features.shape[1]),
        "use_ppm_metadata": use_ppm,
        "cv": cv_report,
        "model_path": str(model_path),
    }
    write_json(output_dir / "cv_report.json", report)
    return report


def predict(data_dir: Path, model_path: Path, output_path: Path, strict: bool = False, use_ppm: bool = False) -> dict:
    package = joblib.load(model_path)
    submission_csv = find_submission_csv(data_dir)
    if submission_csv is None:
        raise FileNotFoundError("Could not find a sample submission CSV containing sample_id and target columns.")
    bundle = build_bundle(data_dir, submission_csv, include_targets=False, strict=strict, use_ppm=use_ppm)
    predictions = predict_final(package["model"], bundle.features)
    frame = pd.DataFrame(predictions, columns=TARGET_COLUMNS)
    frame.insert(0, "sample_id", bundle.sample_ids)
    # Preserve every row in the sample submission, including unmatched ids, with a
    # conservative all-zero CDF ending at 100 so the file remains structurally valid.
    template = pd.read_csv(submission_csv)
    id_col = find_id_column(template.columns)
    output = template[[id_col]].copy().rename(columns={id_col: "sample_id"})
    output = output.merge(frame, on="sample_id", how="left", sort=False)
    output[TARGET_COLUMNS] = output[TARGET_COLUMNS].fillna(0.0)
    output[TARGET_COLUMNS] = repair_cdf(output[TARGET_COLUMNS].to_numpy())
    output.to_csv(output_path, index=False, float_format="%.6f")
    report = {
        "command": "predict",
        "submission_template": str(submission_csv),
        "output": str(output_path),
        "n_rows": int(len(output)),
        "matched_rows": int(len(bundle.sample_ids)),
        "use_ppm_metadata": use_ppm,
        "missing_sample_ids": bundle.missing_ids,
        "structural_checks": {
            "values_in_0_100": bool(np.all((output[TARGET_COLUMNS].to_numpy() >= 0) & (output[TARGET_COLUMNS].to_numpy() <= 100))),
            "monotonic_cdf": bool(np.all(np.diff(output[TARGET_COLUMNS].to_numpy(), axis=1) >= -1e-6)),
            "last_column_is_100": bool(np.allclose(output["200"].to_numpy(), 100.0)),
        },
    }
    write_json(output_path.with_suffix(".json"), report)
    return report


def create_smoke_dataset(root: Path) -> tuple[Path, Path]:
    data_dir = root / "data"
    train_dir = data_dir / "Training-All_Photos_updated"
    test_dir = data_dir / "Test_All_Photos"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)
    train_rows = []
    for index in range(8):
        sample_id = f"TRAIN_{index:02d}"
        base = np.array([0.25 + index * 0.04, 0.20 + index * 0.03, 0.12 + index * 0.02])
        image = np.clip(base + rng.normal(0, 0.08, (96, 96, 3)), 0, 1)
        Image.fromarray((image * 255).astype(np.uint8)).save(train_dir / f"{sample_id}_photo.png")
        centre = np.linspace(5 + index, 95, 11) + rng.normal(0, 1, 11)
        train_rows.append([sample_id, *repair_cdf(centre[None, :])[0]])
    labels = pd.DataFrame(train_rows, columns=["sample_id", *TARGET_COLUMNS])
    labels.to_csv(data_dir / "Training_labels_updated.csv", index=False)
    test_rows = []
    for index in range(2):
        sample_id = f"TEST_{index:02d}"
        image = np.clip(0.35 + rng.normal(0, 0.08, (96, 96, 3)), 0, 1)
        Image.fromarray((image * 255).astype(np.uint8)).save(test_dir / f"{sample_id}_photo.png")
        test_rows.append([sample_id, *([0.0] * 11)])
    submission = pd.DataFrame(test_rows, columns=["sample_id", *TARGET_COLUMNS])
    submission.to_csv(data_dir / "sample_submission.csv", index=False)
    return data_dir, root / "run"


def smoke_test(root: Path) -> dict:
    data_dir, output_dir = create_smoke_dataset(root)
    report = train(data_dir, output_dir, strict=True)
    prediction_report = predict(data_dir, output_dir / "model.joblib", output_dir / "submission.csv", strict=True)
    result = {"train": report, "predict": prediction_report, "status": "ok"}
    write_json(root / "smoke_report.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect", help="Inspect CSV/image inventory without training.")
    inspect_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    inspect_parser.add_argument("--output", type=Path)
    train_parser = sub.add_parser("train", help="Extract features, cross-validate, and fit the model.")
    train_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/soil_gsd"))
    train_parser.add_argument("--strict", action="store_true", help="Fail if any labelled sample has no matching image.")
    train_parser.add_argument("--use-ppm", action="store_true", help="Append official camera pixels-per-mm summaries as features.")
    predict_parser = sub.add_parser("predict", help="Create a competition-ready submission CSV.")
    predict_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, default=Path("artifacts/soil_gsd/submission.csv"))
    predict_parser.add_argument("--strict", action="store_true")
    predict_parser.add_argument("--use-ppm", action="store_true", help="Append official camera pixels-per-mm summaries as features.")
    smoke_parser = sub.add_parser("smoke-test", help="Run the complete pipeline on a generated fixture.")
    smoke_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke_test"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            inspect_data(args.data_dir, args.output)
        elif args.command == "train":
            print(json.dumps(train(args.data_dir, args.output_dir, args.strict, args.use_ppm), indent=2, ensure_ascii=False))
        elif args.command == "predict":
            print(json.dumps(predict(args.data_dir, args.model, args.output, args.strict, args.use_ppm), indent=2, ensure_ascii=False))
        elif args.command == "smoke-test":
            print(json.dumps(smoke_test(args.output_dir), indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
