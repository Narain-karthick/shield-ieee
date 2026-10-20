#!/usr/bin/env python3
"""
scripts/evaluate_calibration.py

Evaluates Platt scaling probability calibration on the untouched held-out test dataset:
- Brier Score & Log Loss (Raw Risk vs Calibrated Probability)
- Expected Calibration Error (ECE) with equal-width and equal-frequency binning
- Maximum Calibration Error (MCE)
- AUROC & AUPRC
- Threshold evaluation (tau=0.5, tuned tau*, raw risk >= 0.80)
- Reliability diagram generation (PNG and PDF)
- Calibration bin sample counts

Fulfills Phase 6 of IEEE INDICON 2026 Paper 4687 revision.
"""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_ece_and_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[float, float, list[dict[str, Any]]]:
    """
    Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    strategy: 'uniform' (equal-width) or 'quantile' (equal-frequency)
    """
    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    elif strategy == "quantile":
        # Handle zero-variance quantiles gracefully
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.array([0.0, 1.0])
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    total_samples = len(y_true)
    ece = 0.0
    mce = 0.0
    bins_data: list[dict[str, Any]] = []

    actual_bins = len(bin_edges) - 1
    for b in range(actual_bins):
        mask = bin_indices == b
        bin_count = int(mask.sum())
        if bin_count == 0:
            bins_data.append({
                "bin_index": b,
                "bin_lower": round(float(bin_edges[b]), 4),
                "bin_upper": round(float(bin_edges[b + 1]), 4),
                "sample_count": 0,
                "mean_predicted_prob": float("nan"),
                "empirical_fraction_positive": float("nan"),
                "calibration_error": 0.0,
            })
            continue

        mean_prob = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        err = abs(acc - mean_prob)
        ece += (bin_count / total_samples) * err
        if err > mce:
            mce = err

        bins_data.append({
            "bin_index": b,
            "bin_lower": round(float(bin_edges[b]), 4),
            "bin_upper": round(float(bin_edges[b + 1]), 4),
            "sample_count": bin_count,
            "mean_predicted_prob": round(mean_prob, 6),
            "empirical_fraction_positive": round(acc, 6),
            "calibration_error": round(err, 6),
        })

    return round(float(ece), 6), round(float(mce), 6), bins_data


def tune_threshold_on_calibration(calib_csv: Path, model: dict[str, Any]) -> float:
    """Finds optimal decision threshold on calibration split to maximize F1."""
    if not calib_csv.exists():
        return 0.50
    df = pd.read_csv(calib_csv)
    eligible = df[df["max_z"].notna() & np.isfinite(df["max_z"])].drop_duplicates("event_id")
    y_calib = eligible["ground_truth"].values.astype(int)

    slope = model["parameters"]["A_slope"]
    intercept = model["parameters"]["B_intercept"]
    feature_mode = model["feature_mode"]

    if feature_mode == "margin":
        z_vals = eligible["max_z"].values - 2.5
    else:
        z_vals = eligible["max_z"].values

    probs = sigmoid(slope * z_vals + intercept)

    best_thresh = 0.50
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 91):
        preds = (probs >= t).astype(int)
        score = f1_score(y_calib, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)

    return round(best_thresh, 4)


def plot_reliability_diagram(
    bins_raw: list[dict[str, Any]],
    bins_calib: list[dict[str, Any]],
    output_png: Path,
    output_pdf: Path,
    ece_raw: float,
    ece_calib: float,
) -> None:
    """Generates publication-quality reliability curve and sample histogram."""
    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(6.5, 7.0), gridspec_kw={"height_ratios": [2.5, 1.2]}
    )

    # 1. Diagonal ideal reference
    ax_rel.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.7, linewidth=1.5)

    # Extract valid bins
    raw_x = [b["mean_predicted_prob"] for b in bins_raw if not math.isnan(b["mean_predicted_prob"])]
    raw_y = [b["empirical_fraction_positive"] for b in bins_raw if not math.isnan(b["empirical_fraction_positive"])]

    calib_x = [b["mean_predicted_prob"] for b in bins_calib if not math.isnan(b["mean_predicted_prob"])]
    calib_y = [b["empirical_fraction_positive"] for b in bins_calib if not math.isnan(b["empirical_fraction_positive"])]

    # Plot curves
    ax_rel.plot(
        raw_x,
        raw_y,
        marker="s",
        color="#D55E00",
        linestyle="--",
        linewidth=1.8,
        markersize=6,
        label=f"Raw Risk Score (ECE = {ece_raw:.4f})",
    )
    ax_rel.plot(
        calib_x,
        calib_y,
        marker="o",
        color="#0072B2",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        label=f"Platt Calibrated (ECE = {ece_calib:.4f})",
    )

    ax_rel.set_ylabel("Empirical Frequency of Positive Anomaly", fontsize=11)
    ax_rel.set_xlim(-0.02, 1.02)
    ax_rel.set_ylim(-0.02, 1.02)
    ax_rel.grid(True, linestyle=":", alpha=0.5)
    ax_rel.legend(loc="upper left", frameon=True, fontsize=10)
    ax_rel.set_title("Reliability Diagram (Held-Out Test Partition)", fontsize=12, fontweight="bold")

    # 2. Histogram of samples
    bin_centers = [0.5 * (b["bin_lower"] + b["bin_upper"]) for b in bins_calib]
    counts = [b["sample_count"] for b in bins_calib]
    ax_hist.bar(bin_centers, counts, width=0.08, color="#0072B2", alpha=0.6, edgecolor="black", linewidth=0.5)
    ax_hist.set_xlabel("Predicted Probability / Score", fontsize=11)
    ax_hist.set_ylabel("Sample Count", fontsize=11)
    ax_hist.set_yscale("log")
    ax_hist.set_xlim(-0.02, 1.02)
    ax_hist.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Platt scaling on held-out test data")
    parser.add_argument("--test-data", default="data/held_out_events.csv", help="Held-out events CSV")
    parser.add_argument("--calib-data", default="data/calibration_events.csv", help="Calibration events CSV for threshold tuning")
    parser.add_argument("--model", default="results/calibration/calibration_model.json", help="Trained model metadata JSON")
    parser.add_argument("--output-dir", default="results/calibration", help="Directory to save evaluation artifacts")
    args = parser.parse_args()

    test_path = Path(args.test_data)
    model_path = Path(args.model)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = json.loads(model_path.read_text(encoding="utf-8"))
    test_df = pd.read_csv(test_path)

    # Filter to unique eligible events (with finite max_z)
    unique_test = test_df.drop_duplicates(subset=["event_id"]).copy()
    eligible = unique_test[unique_test["max_z"].notna() & np.isfinite(unique_test["max_z"])].copy()
    print(f"Loaded held-out test set: {len(test_df)} records -> {len(eligible)} eligible continuous events.")

    y_test = eligible["ground_truth"].values.astype(int)
    raw_risks = np.clip(eligible["raw_risk"].values.astype(float), 1e-7, 1.0 - 1e-7)

    # Compute calibrated probabilities
    slope_a = model["parameters"]["A_slope"]
    intercept_b = model["parameters"]["B_intercept"]
    feature_mode = model["feature_mode"]

    if feature_mode == "margin":
        z_features = eligible["max_z"].values - 2.5
    else:
        z_features = eligible["max_z"].values

    logits = slope_a * z_features + intercept_b
    calib_probs = np.clip(sigmoid(logits), 1e-7, 1.0 - 1e-7)

    # 1. Brier score & Log Loss
    bs_raw = float(brier_score_loss(y_test, raw_risks))
    bs_calib = float(brier_score_loss(y_test, calib_probs))

    nll_raw = float(log_loss(y_test, raw_risks))
    nll_calib = float(log_loss(y_test, calib_probs))

    # 2. Calibration Errors (ECE and MCE)
    ece_raw, mce_raw, bins_raw = compute_ece_and_bins(y_test, raw_risks, n_bins=10, strategy="uniform")
    ece_calib, mce_calib, bins_calib = compute_ece_and_bins(y_test, calib_probs, n_bins=10, strategy="uniform")

    ece_calib_q, mce_calib_q, bins_calib_q = compute_ece_and_bins(y_test, calib_probs, n_bins=10, strategy="quantile")

    # 3. Discrimination Metrics
    auroc_raw = float(roc_auc_score(y_test, raw_risks))
    auroc_calib = float(roc_auc_score(y_test, calib_probs))
    auprc_raw = float(average_precision_score(y_test, raw_risks))
    auprc_calib = float(average_precision_score(y_test, calib_probs))

    # 4. Decision Threshold Evaluation
    # Threshold A: Standard calibrated threshold tau = 0.50
    preds_tau_50 = (calib_probs >= 0.50).astype(int)
    cm_50 = confusion_matrix(y_test, preds_tau_50)

    # Threshold B: Tuned threshold on calibration split (to avoid test leakage)
    calib_csv = Path(args.calib_data)
    tuned_tau = tune_threshold_on_calibration(calib_csv, model)
    preds_tuned = (calib_probs >= tuned_tau).astype(int)
    cm_tuned = confusion_matrix(y_test, preds_tuned)

    # Threshold C: Baseline uncalibrated risk threshold >= 0.80
    preds_raw_80 = (raw_risks >= 0.80).astype(int)
    cm_raw_80 = confusion_matrix(y_test, preds_raw_80)

    def eval_cm(cm: np.ndarray, y_t: np.ndarray, p: np.ndarray) -> dict[str, Any]:
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        prec = float(precision_score(y_t, p, zero_division=0))
        rec = float(recall_score(y_t, p, zero_division=0))
        f1 = float(f1_score(y_t, p, zero_division=0))
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    # 5. Reliability Diagram
    plot_png = out_dir / "reliability_diagram.png"
    plot_pdf = out_dir / "reliability_diagram.pdf"
    plot_reliability_diagram(bins_raw, bins_calib, plot_png, plot_pdf, ece_raw, ece_calib)

    # 6. Save Bins CSV
    bins_df = pd.DataFrame(bins_calib)
    bins_csv = out_dir / "calibration_bins.csv"
    bins_df.to_csv(bins_csv, index=False)

    # 7. Summary JSON
    summary = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "file": str(test_path),
            "total_records": len(test_df),
            "eligible_continuous_records": len(eligible),
            "positive_numerical_anomalies": int((y_test == 1).sum()),
            "negative_healthy_records": int((y_test == 0).sum()),
            "base_rate": round(float((y_test == 1).mean()), 4),
        },
        "model_parameters_used": model["parameters"],
        "calibration_metrics": {
            "brier_score": {
                "uncalibrated_raw_risk": round(bs_raw, 6),
                "platt_calibrated": round(bs_calib, 6),
                "absolute_improvement": round(bs_raw - bs_calib, 6),
                "relative_improvement_pct": round(100.0 * (bs_raw - bs_calib) / bs_raw, 2),
            },
            "log_loss_nll": {
                "uncalibrated_raw_risk": round(nll_raw, 6),
                "platt_calibrated": round(nll_calib, 6),
                "absolute_improvement": round(nll_raw - nll_calib, 6),
                "relative_improvement_pct": round(100.0 * (nll_raw - nll_calib) / nll_raw, 2),
            },
            "expected_calibration_error_ece": {
                "uncalibrated_raw_risk_uniform_10bins": ece_raw,
                "platt_calibrated_uniform_10bins": ece_calib,
                "platt_calibrated_quantile_10bins": ece_calib_q,
                "ece_improvement": round(ece_raw - ece_calib, 6),
                "relative_improvement_pct": round(100.0 * (ece_raw - ece_calib) / ece_raw, 2),
            },
            "maximum_calibration_error_mce": {
                "uncalibrated_raw_risk": mce_raw,
                "platt_calibrated": mce_calib,
            },
        },
        "discrimination_metrics": {
            "auroc": {
                "uncalibrated_raw_risk": round(auroc_raw, 6),
                "platt_calibrated": round(auroc_calib, 6),
                "note": "Monotonic transformation preserves ranking order.",
            },
            "auprc": {
                "uncalibrated_raw_risk": round(auprc_raw, 6),
                "platt_calibrated": round(auprc_calib, 6),
            },
        },
        "decision_threshold_evaluation": {
            "calibrated_standard_tau_0_50": {
                "threshold": 0.50,
                **eval_cm(cm_50, y_test, preds_tau_50),
            },
            "calibrated_tuned_tau": {
                "threshold": tuned_tau,
                "tuning_method": "Optimized on disjoint calibration split",
                **eval_cm(cm_tuned, y_test, preds_tuned),
            },
            "baseline_uncalibrated_raw_risk_0_80": {
                "threshold": 0.80,
                **eval_cm(cm_raw_80, y_test, preds_raw_80),
            },
        },
        "artifacts_generated": {
            "metrics_json": str(out_dir / "held_out_metrics.json"),
            "calibration_bins_csv": str(bins_csv),
            "reliability_diagram_png": str(plot_png),
            "reliability_diagram_pdf": str(plot_pdf),
        },
    }

    metrics_json_path = out_dir / "held_out_metrics.json"
    metrics_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n================ HELD-OUT CALIBRATION EVALUATION ================")
    print(f"Test Set: {test_path} ({len(eligible)} eligible events, {int((y_test == 1).sum())} positive anomalies)")
    print(f"Brier Score:   Uncalibrated={bs_raw:.6f} -> Calibrated={bs_calib:.6f} ({summary['calibration_metrics']['brier_score']['relative_improvement_pct']}% improvement)")
    print(f"Log Loss:      Uncalibrated={nll_raw:.6f} -> Calibrated={nll_calib:.6f} ({summary['calibration_metrics']['log_loss_nll']['relative_improvement_pct']}% improvement)")
    print(f"ECE (10-bin):  Uncalibrated={ece_raw:.6f} -> Calibrated={ece_calib:.6f} ({summary['calibration_metrics']['expected_calibration_error_ece']['relative_improvement_pct']}% improvement)")
    print(f"MCE:           Uncalibrated={mce_raw:.6f} -> Calibrated={mce_calib:.6f}")
    print(f"AUROC:         {auroc_calib:.4f} | AUPRC: {auprc_calib:.4f}")
    print("\nDecision Thresholds on Held-Out Test Data:")
    for key, item in summary["decision_threshold_evaluation"].items():
        print(f"  [{key}] Threshold={item['threshold']}: Precision={item['precision']*100:.2f}%, Recall={item['recall']*100:.2f}%, F1={item['f1']*100:.2f}% (TP={item['true_positives']}, FP={item['false_positives']}, FN={item['false_negatives']})")
    print(f"\nArtifacts saved in: {out_dir}")
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
