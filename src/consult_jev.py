"""Ask Jev for a structured next-experiment decision without storing credentials."""

from __future__ import annotations

import json

from jev_credentials import post_typesafe

ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def main() -> int:
    state = {
        "competition": "Predicting Soil Grain Size Distributions from Images",
        "evidence": {
            "training_samples": 24,
            "test_samples": 10,
            "daily_submission_limit": 5,
            "best_public_score": 89.39101,
            "previous_public_scores": {
                "resnet18_center_crop": 91.97210,
                "handcrafted_ridge_extra_trees": 143.84834,
                "multitask_elastic_net": 206.89409,
            },
            "best_model": "Frozen ImageNet ResNet18, full-frame 288x384, sample-level views, ExtraTrees/blend",
            "camera_shift": "training images are Motorola/Samsung; test images are iPhone14/iPhone16",
            "local_cv": {
                "best_full_frame_cnn": 47.4706733511,
                "handcrafted_blend": 39.8532302675,
            },
            "submission_quota": "exhausted for today; next experiments can be prepared offline",
        },
        "constraints": [
            "Do not use test labels or leaderboard probing.",
            "Use only official competition data and clearly disclosed public pretrained weights.",
            "Prefer one bounded experiment with a falsifiable reason.",
        ],
    }
    question = {
        "type": "choice",
        "instructions": "Given the evidence, choose the single highest-value next offline experiment to improve the next submission. Prefer robustness to the Motorola/Samsung-to-iPhone camera shift and respect the small sample size. Do not choose an already rejected option.",
        "criteria": {
            "grayscale_full_frame_cnn": "Use the same frozen ResNet18 full-frame pipeline after converting each photo to grayscale and replicating the channel, testing whether camera colour shift is the dominant error source.",
            "tta_ensemble_cnn": "Use the full-frame ResNet18 with deterministic horizontal/vertical flip test-time augmentation and average embeddings or predictions, testing orientation robustness without new labels.",
            "physics_texture_features": "Add scale-aware autocorrelation/blob/texture descriptors from the official images and combine them with the full-frame CNN, testing whether explicit grain-scale cues improve generalisation.",
            "hold_and_validate": "Do not spend more experiments until a stronger grouped validation split or rule confirmation is available; preserve the current best submission.",
        },
    }
    response = post_typesafe({"state": state, "model": "jev-latest", "questions": {"next_experiment": question}})
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
