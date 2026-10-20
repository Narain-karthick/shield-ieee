import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from confluent_kafka import Producer


FIELDS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
    "payment_type",
    "fare_amount",
    "total_amount",
]


def normalise(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def build_event(
    row: pd.Series,
    sequence: int,
    source_record_id: int = 0,
    trial_id: str = "1",
) -> dict[str, Any]:
    payload = {field: normalise(row.get(field)) for field in FIELDS}
    # Deterministic event_id: uses taxi-sequence for trial 1 offset 0 for backward compatibility,
    # or taxi-{trial_id}-{source_record_id} for disjoint partition experiments.
    if trial_id == "1" and source_record_id == sequence:
        event_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"taxi-{sequence}"))
    else:
        event_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"taxi-{trial_id}-{source_record_id}"))

    payload.update(
        {
            "event_id": event_uuid,
            "sequence": sequence,
            "source_record_id": source_record_id,
            "trial_id": trial_id,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
            "ground_truth": False,
            "fault_type": "healthy",
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay NYC taxi records to Kafka")
    parser.add_argument("--file", default=os.getenv("DATA_FILE", "data/yellow_tripdata_2025-01.parquet"))
    parser.add_argument("--rate", type=float, default=float(os.getenv("EVENT_RATE", "100")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0, help="Starting row offset in source dataset")
    parser.add_argument("--trial-id", default=os.getenv("TRIAL_ID", "1"), help="Trial identifier")
    parser.add_argument("--manifest", default=None, help="Optional output path for ground-truth event manifest")
    args = parser.parse_args()

    frame = pd.read_parquet(Path(args.file), columns=FIELDS)
    if args.offset > 0:
        frame = frame.iloc[args.offset:]
    if args.limit > 0:
        frame = frame.iloc[:args.limit]

    producer = Producer(
        {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "enable.idempotence": True,
            "acks": "all",
        }
    )
    topic = os.getenv("RAW_TOPIC", "taxi-raw")
    interval = 0 if args.rate <= 0 else 1.0 / args.rate

    manifest_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for sequence, (_, row) in enumerate(frame.iterrows()):
        source_record_id = args.offset + sequence
        event = build_event(row, sequence, source_record_id=source_record_id, trial_id=args.trial_id)
        producer.produce(topic, key=event["event_id"], value=json.dumps(event).encode("utf-8"))
        producer.poll(0)
        if args.manifest is not None:
            manifest_records.append({
                "event_id": event["event_id"],
                "source_record_id": source_record_id,
                "trial_id": args.trial_id,
                "ground_truth": False,
                "fault_type": "healthy",
                "is_numerical_anomaly": False,
            })
        if interval:
            time.sleep(interval)
        if sequence and sequence % 10_000 == 0:
            elapsed = time.perf_counter() - started
            print(f"sent={sequence} average_rate={sequence / elapsed:.1f} events/s")

    producer.flush()
    if args.manifest and manifest_records:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest_records, indent=2), encoding="utf-8")
        print(f"Saved manifest: {manifest_path} ({len(manifest_records)} events)")

    print(f"Completed: {len(frame)} events (offset={args.offset}, trial_id={args.trial_id})")


if __name__ == "__main__":
    main()

