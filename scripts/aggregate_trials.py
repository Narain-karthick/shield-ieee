import csv
import json
import math
import statistics
from pathlib import Path


TRIAL_COUNT = 5
T_95_DF4 = 2.776
METRICS = ("precision", "recall", "f1")


def describe(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    margin = T_95_DF4 * sd / math.sqrt(len(values))
    return {
        "mean": round(mean, 6),
        "sample_sd": round(sd, 6),
        "ci95_low": round(mean - margin, 6),
        "ci95_high": round(mean + margin, 6),
    }


def main() -> None:
    results_dir = Path("results")
    paths = [results_dir / f"phase2-run{i}.json" for i in range(1, TRIAL_COUNT + 1)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing trial files: {', '.join(missing)}")

    trials = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    aggregate: dict[str, object] = {
        "trial_count": TRIAL_COUNT,
        "classification": {
            metric: describe([float(trial[metric]) for trial in trials])
            for metric in METRICS
        },
        "per_fault_recall": {},
        "repair_success_rate": describe([
            float(trial["repairs"]) / float(trial["per_fault"]["SCHEMA_DRIFT"]["injected"])
            for trial in trials
        ]),
        "healthy_false_positive_rate": describe([
            float(trial["false_positives"]) / 5000.0 for trial in trials
        ]),
    }

    faults = sorted(trials[0]["per_fault"])
    aggregate["per_fault_recall"] = {
        fault: describe([
            float(trial["per_fault"][fault]["recall"]) for trial in trials
        ])
        for fault in faults
    }

    json_path = results_dir / "phase2-aggregate.json"
    json_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    csv_path = results_dir / "phase2-trials.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "trial", "precision", "recall", "f1", "true_positives",
            "false_positives", "false_negatives", "repairs",
            *[f"recall_{fault.lower()}" for fault in faults],
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, trial in enumerate(trials, start=1):
            row = {
                "trial": index,
                "precision": trial["precision"],
                "recall": trial["recall"],
                "f1": trial["f1"],
                "true_positives": trial["true_positives"],
                "false_positives": trial["false_positives"],
                "false_negatives": trial["false_negatives"],
                "repairs": trial["repairs"],
            }
            row.update({
                f"recall_{fault.lower()}": trial["per_fault"][fault]["recall"]
                for fault in faults
            })
            writer.writerow(row)

    print(json.dumps(aggregate, indent=2))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()