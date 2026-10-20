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


def build_event(row: pd.Series, sequence: int) -> dict[str, Any]:
    payload = {field: normalise(row.get(field)) for field in FIELDS}
    payload.update(
        {
            "event_id": str(uuid.uuid5(uuid.NAMESPACE_OID, f"taxi-{sequence}")),
            "sequence": sequence,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": 1,
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay NYC taxi records to Kafka")
    parser.add_argument("--file", default=os.getenv("DATA_FILE", "data/yellow_tripdata_2025-01.parquet"))
    parser.add_argument("--rate", type=float, default=float(os.getenv("EVENT_RATE", "100")))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_parquet(Path(args.file), columns=FIELDS)
    if args.limit > 0:
        frame = frame.head(args.limit)

    producer = Producer(
        {
            "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            "enable.idempotence": True,
            "acks": "all",
        }
    )
    topic = os.getenv("RAW_TOPIC", "taxi-raw")
    interval = 0 if args.rate <= 0 else 1.0 / args.rate

    started = time.perf_counter()
    for sequence, (_, row) in enumerate(frame.iterrows()):
        event = build_event(row, sequence)
        producer.produce(topic, key=event["event_id"], value=json.dumps(event).encode("utf-8"))
        producer.poll(0)
        if interval:
            time.sleep(interval)
        if sequence and sequence % 10_000 == 0:
            elapsed = time.perf_counter() - started
            print(f"sent={sequence} average_rate={sequence / elapsed:.1f} events/s")

    producer.flush()
    print(f"Completed: {len(frame)} events")


if __name__ == "__main__":
    main()

