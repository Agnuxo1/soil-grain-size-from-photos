"""Ask Jev whether to test scale-aware texture features or hold the current model."""

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
            "best_local_cv": {
                "handcrafted_blend": 39.8532302675,
                "colour_full_frame_cnn": 47.4706733511,
                "grayscale_full_frame_cnn": 47.4760077669,
                "tta4_full_frame_cnn": 48.7557043270,
            },
            "completed_recommendations": [
                "grayscale_full_frame_cnn: rejected, no improvement",
                "tta_ensemble_cnn: rejected, degraded grouped CV",
            ],
            "submission_quota": "exhausted for today; no public test of a new candidate is possible",
        },
        "constraints": [
            "Do not use test labels or leaderboard probing.",
            "Use only official competition data and clearly disclosed public pretrained weights.",
            "If expected value is low, choose hold_and_validate.",
        ],
    }
    question = {
        "type": "choice",
        "instructions": "Choose one action now. There are only 24 labelled samples, two JEV-recommended CNN variants failed to improve local CV, and the public submission quota is exhausted. Prefer a falsifiable offline experiment only if its expected value is credible.",
        "criteria": {
            "physics_texture_features": "Implement a bounded set of scale-aware autocorrelation/blob/texture descriptors from the official images, aggregate per sample, and compare against the existing handcrafted and full-frame CNN features with the same grouped CV.",
            "hold_and_validate": "Stop model expansion for now, preserve the public best full-frame colour CNN submission, and focus on stronger validation/rule review before the next Kaggle quota window.",
        },
    }
    response = post_typesafe({"state": state, "model": "jev-latest", "questions": {"next_experiment": question}})
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
