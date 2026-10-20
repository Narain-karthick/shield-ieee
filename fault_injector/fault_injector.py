import argparse
import copy
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from confluent_kafka import Consumer, Producer


def mutate(event: dict[str, Any], fault: str, intensity: float) -> list[dict[str, Any]]:
    changed = copy.deepcopy(event)
    if fault == "schema_drift":
        changed["distance"] = changed.pop("trip_distance", None)
        changed["schema_version"] = 2
    elif fault == "missing_data":
        for field in ("trip_distance", "fare_amount", "PULocationID"):
            if random.random() < intensity:
                changed[field] = None
    elif fault == "corruption":
        changed["fare_amount"] = "INVALID"
        changed["passenger_count"] = -99
    elif fault == "concept_drift":
        changed["trip_distance"] = float(changed.get("trip_distance") or 0) * (1 + 5 * intensity)
        changed["fare_amount"] = float(changed.get("fare_amount") or 0) * (1 + 3 * intensity)
    elif fault == "duplicate":
        return [changed, copy.deepcopy(changed)]
    else:
        raise ValueError(f"Unsupported data fault: {fault}")
    return [changed]


def incident(
    fault: str,
    event_id: str,
    intensity: float,
    source_record_id: Any = None,
    trial_id: str = "1",
) -> dict[str, Any]:
    return {
        "fault": fault,
        "event_id": event_id,
        "source_record_id": source_record_id,
        "trial_id": trial_id,
        "intensity": intensity,
        "injected_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth": True,
        # Crucial for Phase 2: only concept_drift represents a continuous numerical anomaly target
        "is_numerical_anomaly": (fault in ("concept_drift", "numerical_anomaly")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject reproducible data faults")
    parser.add_argument("--fault", required=True, choices=["schema_drift", "missing_data", "corruption", "duplicate", "concept_drift"])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--intensity", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for deterministic reproducibility")
    parser.add_argument("--source-topic", default=os.getenv("CLEAN_TOPIC", "taxi-clean"))
    parser.add_argument("--output-topic", default=os.getenv("RAW_TOPIC", "taxi-raw"))
    parser.add_argument("--incident-topic", default=os.getenv("INCIDENT_TOPIC", "shield-incidents"))
    parser.add_argument("--manifest", default=None, help="Optional output path for incident manifest")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    source_topic = args.source_topic
    output_topic = args.output_topic
    incident_topic = args.incident_topic
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": f"shield-injector-{int(time.time())}",
            "auto.offset.reset": "earliest",
        }
    )
    producer = Producer({"bootstrap.servers": bootstrap, "enable.idempotence": True})
    consumer.subscribe([source_topic])

    injected = 0
    run_id = f"seed-{args.seed}" if args.seed is not None else str(uuid.uuid4())
    incidents_list: list[dict[str, Any]] = []

    while injected < args.count:
        message = consumer.poll(1.0)
        if message is None:
            continue
        if message.error():
            raise RuntimeError(message.error())
        event = json.loads(message.value())
        source_record_id = event.get("source_record_id")
        trial_id = str(event.get("trial_id", "1"))
        event["source_event_id"] = event.get("event_id")
        event["event_id"] = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{run_id}-{args.fault}-{injected}"))
        for changed in mutate(event, args.fault, args.intensity):
            producer.produce(output_topic, key=changed["event_id"], value=json.dumps(changed).encode())
        truth = incident(
            args.fault,
            event["event_id"],
            args.intensity,
            source_record_id=source_record_id,
            trial_id=trial_id,
        )
        producer.produce(incident_topic, key=event["event_id"], value=json.dumps(truth).encode())
        incidents_list.append(truth)
        injected += 1

    producer.flush()
    consumer.close()

    if args.manifest and incidents_list:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(incidents_list, indent=2), encoding="utf-8")
        print(f"Saved incident manifest: {manifest_path} ({len(incidents_list)} incidents)")

    print(f"Injected {injected} {args.fault} incidents (seed={args.seed})")


if __name__ == "__main__":
    main()
