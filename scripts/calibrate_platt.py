#!/usr/bin/env python3
"""
scripts/calibrate_platt.py

Fits a two-parameter Platt scaling logistic regression model to calibrate
the continuous anomaly score into a true posterior probability:
    P(anomalous | max_z) = sigmoid(A * (max_z - 2.5) + B)

Fulfills Phase 5 of IEEE INDICON 2026 Paper 4687 revision.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def validate_calibration_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates data integrity, checks labels, checks for duplicate IDs,
    and isolates continuous anomaly scoring records.
    """
    required_columns = [
        "event_id",
        "source_record_id",
        "max_z",
        "raw_risk",
        "ground_truth",
        "fault_type",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Calibration dataset missing required columns: {missing}")

    if len(df) == 0:
        raise ValueError("Calibration dataset is empty!")

    # Check duplicate event_ids (except known simulated duplicate injection)
    # Filter to unique event_ids to prevent artificial over-weighting
    pre_dedup_len = len(df)
    unique_df = df.drop_duplicates(subset=["event_id"]).copy()
    if len(unique_df) < pre_dedup_len:
        print(f"Info: Deduplicated {pre_dedup_len - len(unique_df)} duplicate event_id instances.")

    # Filter to eligible events: continuous anomaly scoring is defined on structurally valid events with finite max_z
    eligible = unique_df[unique_df["max_z"].notna() & np.isfinite(unique_df["max_z"])].copy()
    print(f"Eligible continuous scoring records: {len(eligible)} / {len(unique_df)} total records.")

    if len(eligible) == 0:
        raise ValueError("No eligible records with finite max_z found in calibration dataset!")

    labels = eligible["ground_truth"].unique()
    if set(labels) - {0, 1}:
        raise ValueError(f"Invalid labels found in ground_truth: {labels}. Must be binary {0, 1}.")

    if len(labels) < 2:
        raise ValueError(f"Insufficient classes: dataset contains only class {labels}. Both 0 and 1 required.")

    class_counts = eligible["ground_truth"].value_counts().to_dict()
    print(f"Class distribution: {class_counts} (positives: {class_counts.get(1, 0)}, negatives: {class_counts.get(0, 0)})")

    return eligible


def fit_platt_scaling(
    eligible_df: pd.DataFrame,
    feature_mode: str = "margin",
) -> dict[str, Any]:
    """
    Fits logistic regression to learn Platt parameters A and B.
    feature_mode='margin': uses m = max_z - 2.5
    feature_mode='max_z': uses z = max_z
    """
    y = eligible_df["ground_truth"].values.astype(int)

    if feature_mode == "margin":
        # m = max_z - 2.5 (where raw_risk = sigmoid(m))
        # Uncalibrated baseline corresponds to A=1.0, B=0.0
        X = (eligible_df["max_z"].values - 2.5).reshape(-1, 1)
        feature_name = "margin_z_minus_2_5"
        feature_formula = "max_z - 2.5"
    elif feature_mode == "max_z":
        X = eligible_df["max_z"].values.reshape(-1, 1)
        feature_name = "max_z"
        feature_formula = "max_z"
    else:
        raise ValueError(f"Unsupported feature_mode: {feature_mode}")

    # Standard Platt scaling: unregularized or very weak L2 penalty
    clf = LogisticRegression(solver="lbfgs", C=1e5, max_iter=1000)
    clf.fit(X, y)

    slope_a = float(clf.coef_[0][0])
    intercept_b = float(clf.intercept_[0])

    # Calibrated probabilities on calibration training split
    calib_probs = clf.predict_proba(X)[:, 1]
    raw_risks = eligible_df["raw_risk"].values

    train_log_loss_calib = float(log_loss(y, calib_probs))
    train_log_loss_raw = float(log_loss(y, raw_risks))
    train_brier_calib = float(brier_score_loss(y, calib_probs))
    train_brier_raw = float(brier_score_loss(y, raw_risks))

    # Also calculate equivalent parameters for the alternate parameterization
    if feature_mode == "margin":
        direct_a = slope_a
        direct_b = intercept_b - 2.5 * slope_a
    else:
        direct_a = slope_a
        direct_b = intercept_b

    model_metadata = {
        "model_type": "platt_scaling_logistic",
        "fit_timestamp": datetime.now(timezone.utc).isoformat(),
        "feature_mode": feature_mode,
        "feature_name": feature_name,
        "feature_formula": feature_formula,
        "parameters": {
            "A_slope": round(slope_a, 6),
            "B_intercept": round(intercept_b, 6),
            "equivalent_direct_z_slope": round(direct_a, 6),
            "equivalent_direct_z_intercept": round(direct_b, 6),
        },
        "formula_description": (
            f"p_calibrated = 1.0 / (1.0 + exp(-({slope_a:.6f} * ({feature_formula}) + {intercept_b:.6f})))"
        ),
        "training_diagnostics": {
            "total_samples": len(eligible_df),
            "positive_samples": int((y == 1).sum()),
            "negative_samples": int((y == 0).sum()),
            "positive_rate": round(float((y == 1).mean()), 4),
            "train_log_loss_calibrated": round(train_log_loss_calib, 6),
            "train_log_loss_uncalibrated": round(train_log_loss_raw, 6),
            "train_brier_calibrated": round(train_brier_calib, 6),
            "train_brier_uncalibrated": round(train_brier_raw, 6),
            "solver_iterations": int(clf.n_iter_[0]),
            "converged": bool(clf.n_iter_[0] < clf.max_iter),
        },
    }

    return model_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Platt scaling calibration model for SHIELD-Stream")
    parser.add_argument("--input", default="data/calibration_events.csv", help="Calibration events CSV")
    parser.add_argument("--output-model", default="results/calibration/calibration_model.json", help="Path to save model JSON")
    parser.add_argument("--feature-mode", default="margin", choices=["margin", "max_z"], help="Input feature for calibration")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading calibration dataset: {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} total rows.")

    eligible_df = validate_calibration_data(df)

    print(f"\nFitting Platt scaling model (feature_mode='{args.feature_mode}')...")
    model_metadata = fit_platt_scaling(eligible_df, feature_mode=args.feature_mode)

    out_path = Path(args.output_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model_metadata, indent=2), encoding="utf-8")

    diag = model_metadata["training_diagnostics"]
    params = model_metadata["parameters"]
    print("\n--- Platt Scaling Model Summary ---")
    print(f"Model File: {out_path}")
    print(f"Formula: {model_metadata['formula_description']}")
    print(f"Parameter A (Slope): {params['A_slope']}")
    print(f"Parameter B (Intercept): {params['B_intercept']}")
    print(f"Training Log Loss: Uncalibrated={diag['train_log_loss_uncalibrated']} -> Calibrated={diag['train_log_loss_calibrated']}")
    print(f"Training Brier Score: Uncalibrated={diag['train_brier_uncalibrated']} -> Calibrated={diag['train_brier_calibrated']}")
    print("Fitting completed successfully.")


if __name__ == "__main__":
    main()
