"""Frozen ResNet18 multi-view candidate for the soil-GSD competition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import KFold
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

import soil_gsd_pipeline as base


def labelled_or_template_rows(data_dir: Path, csv_path: Path, include_targets: bool):
    frame = pd.read_csv(csv_path)
    id_col = base.find_id_column(frame.columns)
    ids = frame[id_col].astype(str).tolist()
    targets = None
    if include_targets:
        resolved = base.resolve_target_columns(frame.columns)
        values = frame[[source for source, target in resolved.items() if target in base.TARGET_COLUMNS]].copy()
        values = values.rename(columns=resolved)[base.TARGET_COLUMNS]
        targets = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    image_paths = base.discover_images(data_dir)
    matched = [base.match_images(sample_id, image_paths) for sample_id in ids]
    missing = [sample_id for sample_id, paths in zip(ids, matched) if not paths]
    if missing:
        raise FileNotFoundError(f"Missing image matches for {len(missing)} ids: {missing[:5]}")
    return ids, matched, targets


def embedding_key(path: Path, variant: str) -> str:
    return str(path) if variant == "original" else f"{path}::{variant}"


def variants_for_tta(tta: bool) -> tuple[str, ...]:
    return ("original", "hflip", "vflip", "hvflip") if tta else ("original",)


def extract_embeddings(
    image_paths: Sequence[Path],
    cache_path: Path,
    batch_size: int = 16,
    grayscale: bool = False,
    tta: bool = False,
) -> dict[str, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    if cache_path.exists():
        cache = joblib.load(cache_path)
    variants = variants_for_tta(tta)
    wanted = [
        (path, variant)
        for path in image_paths
        for variant in variants
        if embedding_key(path, variant) not in cache
    ]
    if wanted:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.hub.set_dir(str(cache_path.parent / "torch_cache"))
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval().to(device)
        # Keep the complete 4:3 soil surface. weights.transforms() uses a
        # center crop to 224x224, which discards the borders and changes the
        # apparent particle scale in these portrait photos.
        transform = transforms.Compose(
            [
                transforms.Resize((288, 384), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        with torch.inference_mode():
            for start in range(0, len(wanted), batch_size):
                batch_items = wanted[start : start + batch_size]
                tensors = []
                for path, variant in batch_items:
                    with Image.open(path) as image:
                        rgb = image.convert("L").convert("RGB") if grayscale else image.convert("RGB")
                        if variant == "hflip":
                            rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                        elif variant == "vflip":
                            rgb = rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                        elif variant == "hvflip":
                            rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                        tensors.append(transform(rgb))
                batch = torch.stack(tensors).to(device)
                values = model(batch).detach().float().cpu().numpy()
                cache.update({embedding_key(path, variant): value.astype(np.float32) for (path, variant), value in zip(batch_items, values)})
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cache, cache_path)
    return cache


def aggregate_samples(
    matched: Sequence[Sequence[Path]],
    embeddings: dict[str, np.ndarray],
    tta: bool = False,
) -> np.ndarray:
    variants = variants_for_tta(tta)
    sample_values = []
    for paths in matched:
        photo_values = []
        for path in paths:
            photo_values.append(
                np.vstack([embeddings[embedding_key(path, variant)] for variant in variants]).mean(axis=0)
            )
        sample_values.append(np.vstack(photo_values).mean(axis=0))
    return np.vstack(sample_values).astype(np.float32)


def train(data_dir: Path, output_dir: Path, cache_path: Path, grayscale: bool = False, tta: bool = False) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = base.find_label_csv(data_dir)
    if labels_path is None:
        raise FileNotFoundError("No labelled CSV found.")
    ids, matched, targets = labelled_or_template_rows(data_dir, labels_path, include_targets=True)
    all_images = [path for paths in matched for path in paths]
    embeddings = extract_embeddings(all_images, cache_path, grayscale=grayscale, tta=tta)
    X = aggregate_samples(matched, embeddings, tta=tta)
    cv_report, _ = base.cross_validate(X, targets)
    model = base.fit_final(X, targets, cv_report)
    model_path = output_dir / "cnn_model.joblib"
    variant_suffix = "_grayscale" if grayscale else ""
    tta_suffix = "_tta4" if tta else ""
    backbone = f"resnet18_imagenet_frozen_full_frame_288x384{variant_suffix}{tta_suffix}"
    joblib.dump({"model": model, "feature_dim": int(X.shape[1]), "backbone": backbone, "grayscale": grayscale, "tta": tta}, model_path)
    report = {
        "command": "cnn-train",
        "n_samples": len(ids),
        "n_images": len(all_images),
        "feature_dim": int(X.shape[1]),
        "backbone": backbone,
        "grayscale": grayscale,
        "tta": tta,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cv": cv_report,
        "model_path": str(model_path),
        "embedding_cache": str(cache_path),
    }
    (output_dir / "cnn_cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def predict(
    data_dir: Path,
    model_path: Path,
    cache_path: Path,
    output_path: Path,
    grayscale: bool = False,
    tta: bool = False,
) -> dict:
    package = joblib.load(model_path)
    template_path = base.find_submission_csv(data_dir)
    if template_path is None:
        raise FileNotFoundError("No sample submission found.")
    ids, matched, _ = labelled_or_template_rows(data_dir, template_path, include_targets=False)
    all_images = [path for paths in matched for path in paths]
    embeddings = extract_embeddings(all_images, cache_path, grayscale=grayscale, tta=tta)
    X = aggregate_samples(matched, embeddings, tta=tta)
    predictions = base.predict_final(package["model"], X)
    frame = pd.DataFrame(predictions, columns=base.TARGET_COLUMNS)
    frame.insert(0, "sample_id", ids)
    frame.to_csv(output_path, index=False, float_format="%.6f")
    return {
        "command": "cnn-predict",
        "output": str(output_path),
        "rows": len(frame),
        "feature_dim": int(X.shape[1]),
        "structural_checks": {
            "values_in_0_100": bool(np.all((predictions >= 0) & (predictions <= 100))),
            "monotonic_cdf": bool(np.all(np.diff(predictions, axis=1) >= -1e-6)),
            "last_column_is_100": bool(np.allclose(predictions[:, -1], 100.0)),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cnn_resnet18"))
    train_parser.add_argument("--cache", type=Path, default=Path("artifacts/cnn_resnet18/embeddings.joblib"))
    train_parser.add_argument("--grayscale", action="store_true", help="Convert each full-frame image to replicated grayscale channels")
    train_parser.add_argument("--tta", action="store_true", help="Average original, horizontal-flip, vertical-flip and double-flip embeddings")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--cache", type=Path, default=Path("artifacts/cnn_resnet18/embeddings.joblib"))
    predict_parser.add_argument("--output", type=Path, default=Path("artifacts/cnn_resnet18/submission.csv"))
    predict_parser.add_argument("--grayscale", action="store_true", help="Convert each full-frame image to replicated grayscale channels")
    predict_parser.add_argument("--tta", action="store_true", help="Average original, horizontal-flip, vertical-flip and double-flip embeddings")
    args = parser.parse_args(argv)
    if args.command == "train":
        print(json.dumps(train(args.data_dir, args.output_dir, args.cache, grayscale=args.grayscale, tta=args.tta), indent=2))
    else:
        print(json.dumps(predict(args.data_dir, args.model, args.cache, args.output, grayscale=args.grayscale, tta=args.tta), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
