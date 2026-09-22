"""Ask Jev for the next bounded offline experiment after the grayscale test."""

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
            "best_model": "Frozen ImageNet ResNet18, full-frame 288x384, sample-level views, ExtraTrees/blend",
            "local_cv": {
                "handcrafted_blend": 39.8532302675,
                "colour_full_frame_cnn": 47.4706733511,
                "grayscale_full_frame_cnn": 47.4760077669,
            },
            "grayscale_result": "JEV round 1 recommendation; ExtraTrees selected, audited valid, no leaderboard submission because quota was exhausted; reject for no improvement.",
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
        "instructions": "Choose the single highest-value next offline experiment after the grayscale test. The grayscale option is rejected and must not be selected. Respect the 24-sample size and prioritize validation discipline.",
        "criteria": {
            "tta_ensemble_cnn": "Use the colour full-frame ResNet18 with deterministic horizontal and vertical flip test-time augmentation, averaging the per-view embeddings before the sample-level model; test orientation robustness without new labels.",
            "physics_texture_features": "Add scale-aware autocorrelation, blob and texture descriptors from the official images and combine them with the colour full-frame CNN; test whether explicit grain-scale cues improve generalisation.",
            "hold_and_validate": "Do not spend another experiment until a stronger grouped validation split or rule confirmation is available; preserve the current best submission.",
        },
    }
    response = post_typesafe({"state": state, "model": "jev-latest", "questions": {"next_experiment": question}})
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
