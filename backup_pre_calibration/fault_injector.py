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


def incident(fault: str, event_id: str, intensity: float) -> dict[str, Any]:
    return {
        "fault": fault,
        "event_id": event_id,
        "intensity": intensity,
        "injected_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject reproducible data faults")
    parser.add_argument("--fault", required=True, choices=["schema_drift", "missing_data", "corruption", "duplicate", "concept_drift"])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--intensity", type=float, default=0.3)
    parser.add_argument("--source-topic", default=os.getenv("CLEAN_TOPIC", "taxi-clean"))
    parser.add_argument("--output-topic", default=os.getenv("RAW_TOPIC", "taxi-raw"))
    parser.add_argument("--incident-topic", default=os.getenv("INCIDENT_TOPIC", "shield-incidents"))
    args = parser.parse_args()

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
    run_id = str(uuid.uuid4())
    while injected < args.count:
        message = consumer.poll(1.0)
        if message is None:
            continue
        if message.error():
            raise RuntimeError(message.error())
        event = json.loads(message.value())
        event["source_event_id"] = event.get("event_id")
        event["event_id"] = str(uuid.uuid5(uuid.NAMESPACE_OID, f"{run_id}-{args.fault}-{injected}"))
        for changed in mutate(event, args.fault, args.intensity):
            producer.produce(output_topic, key=changed["event_id"], value=json.dumps(changed).encode())
        truth = incident(args.fault, event["event_id"], args.intensity)
        producer.produce(incident_topic, key=event["event_id"], value=json.dumps(truth).encode())
        injected += 1

    producer.flush()
    consumer.close()
    print(f"Injected {injected} {args.fault} incidents")


if __name__ == "__main__":
    main()
