import argparse
import json
import time
from collections import Counter
from pathlib import Path

from confluent_kafka import Consumer


def read_topic(topic: str, timeout: float) -> list[dict]:
    consumer = Consumer({
        "bootstrap.servers": "localhost:9092",
        "group.id": f"shield-evaluator-{topic}-{time.time_ns()}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })
    consumer.subscribe([topic])
    rows: list[dict] = []
    idle_since = time.monotonic()
    while time.monotonic() - idle_since < timeout:
        message = consumer.poll(0.25)
        if message is None:
            continue
        if message.error():
            continue
        idle_since = time.monotonic()
        try:
            rows.append(json.loads(message.value()))
        except (json.JSONDecodeError, TypeError):
            pass
    consumer.close()
    return rows


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 against injected ground truth")
    parser.add_argument("--idle-timeout", type=float, default=3.0)
    parser.add_argument("--output", default="results/phase2-summary.json")
    args = parser.parse_args()

    incidents = read_topic("shield-incidents-v2", args.idle_timeout)
    metrics = read_topic("shield-metrics", args.idle_timeout)
    repairs = read_topic("shield-repairs", args.idle_timeout)
    truth = {row.get("event_id"): row.get("fault", "unknown").upper() for row in incidents}
    actionable = [row for row in metrics if row.get("fault") != "DRIFT_WARNING"]
    detected = {row.get("event_id") for row in actionable}
    true_ids = set(truth)
    tp = len(detected & true_ids)
    fp = len(detected - true_ids)
    fn = len(true_ids - detected)
    truth_counts = Counter(truth.values())
    detection_counts = Counter(truth[event_id] for event_id in detected & true_ids)

    summary = {
        "ground_truth_incidents": len(true_ids),
        "detections": len(detected),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(ratio(tp, tp + fp), 4),
        "recall": round(ratio(tp, tp + fn), 4),
        "f1": round(ratio(2 * tp, 2 * tp + fp + fn), 4),
        "repairs": len(repairs),
        "actions": dict(Counter(row.get("action", "UNKNOWN") for row in metrics)),
        "per_fault": {
            fault: {
                "injected": count,
                "detected": detection_counts[fault],
                "recall": round(ratio(detection_counts[fault], count), 4),
            }
            for fault, count in sorted(truth_counts.items())
        },
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()
