import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def trial_stability(frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    styles = {
        "precision": ("#0072B2", "o", "Precision"),
        "recall": ("#009E73", "s", "Recall"),
        "f1": ("#D55E00", "^", "F1-score"),
    }
    for column, (colour, marker, label) in styles.items():
        ax.plot(
            frame["trial"], frame[column] * 100,
            color=colour, marker=marker, linewidth=1.8,
            markersize=5.5, label=label,
        )
    ax.set_xlabel("Repeated trial")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(frame["trial"])
    ax.set_ylim(68, 94)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3, frameon=False, loc="lower center")
    save(fig, "trial_stability")


def fault_recall(aggregate: dict) -> None:
    preferred = [
        "SCHEMA_DRIFT", "CORRUPTION", "MISSING_DATA",
        "DUPLICATE", "CONCEPT_DRIFT",
    ]
    labels = [
        "Schema\ndrift", "Corruption", "Missing\ndata",
        "Duplicate", "Concept\ndrift",
    ]
    values = [aggregate["per_fault_recall"][fault]["mean"] * 100 for fault in preferred]
    lows = [aggregate["per_fault_recall"][fault]["ci95_low"] * 100 for fault in preferred]
    highs = [aggregate["per_fault_recall"][fault]["ci95_high"] * 100 for fault in preferred]
    errors = [
        [value - low for value, low in zip(values, lows)],
        [high - value for value, high in zip(values, highs)],
    ]

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    colours = ["#009E73", "#009E73", "#009E73", "#009E73", "#E69F00"]
    bars = ax.bar(labels, values, color=colours, edgecolor="black", linewidth=0.5)
    ax.errorbar(range(len(values)), values, yerr=errors, fmt="none", ecolor="black", capsize=3)
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%", ha="center", fontsize=9)
    save(fig, "per_fault_recall")


def phase_comparison(aggregate: dict) -> None:
    phase1_recall = 60.0
    phase2_recall = aggregate["classification"]["recall"]["mean"] * 100
    fig, ax = plt.subplots(figsize=(4.8, 3.7))
    bars = ax.bar(
        ["Phase 1\nRule baseline", "SHIELD-Stream\nPhase 2"],
        [phase1_recall, phase2_recall],
        color=["#999999", "#0072B2"], edgecolor="black", linewidth=0.5,
    )
    ax.set_ylabel("Overall fault recall (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, [phase1_recall, phase2_recall]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.2f}%", ha="center")
    save(fig, "phase1_vs_phase2_recall")


def main() -> None:
    frame = pd.read_csv(RESULTS / "phase2-trials.csv")
    aggregate = json.loads((RESULTS / "phase2-aggregate.json").read_text(encoding="utf-8"))
    trial_stability(frame)
    fault_recall(aggregate)
    phase_comparison(aggregate)
    print(f"Saved figures in: {FIGURES}")
    for path in sorted(FIGURES.iterdir()):
        print(path)


if __name__ == "__main__":
    main()