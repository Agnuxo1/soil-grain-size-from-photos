"""Validate a soil-GSD submission without contacting Kaggle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMNS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]


def audit_submission(path: Path, template: Path | None = None) -> dict:
    frame = pd.read_csv(path)
    issues: list[str] = []
    expected = ["sample_id", *TARGET_COLUMNS]
    if list(frame.columns) != expected:
        issues.append(f"columns must be exactly {expected}")
    if "sample_id" not in frame.columns:
        return {"path": str(path), "rows": int(len(frame)), "expected_rows": None, "issues": issues + ["missing sample_id column"], "valid": False}
    if frame["sample_id"].duplicated().any():
        issues.append("duplicate sample_id values")
    numeric = frame.reindex(columns=TARGET_COLUMNS).apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        issues.append("target columns contain non-numeric or missing values")
    values = numeric.to_numpy(dtype=float)
    if values.size and (np.any(values < 0) or np.any(values > 100)):
        issues.append("target values must lie in [0, 100]")
    if values.size and np.any(np.diff(values, axis=1) < -1e-8):
        issues.append("CDF rows must be non-decreasing")
    if values.size and not np.allclose(values[:, -1], 100.0):
        issues.append("the 200 mm column must equal 100")
    expected_rows = None
    if template:
        expected_rows = len(pd.read_csv(template))
        if len(frame) != expected_rows:
            issues.append(f"row count {len(frame)} differs from template {expected_rows}")
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "expected_rows": expected_rows,
        "issues": issues,
        "valid": not issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--template", type=Path)
    args = parser.parse_args()
    report = audit_submission(args.submission, args.template)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
