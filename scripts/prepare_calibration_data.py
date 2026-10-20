#!/usr/bin/env python3
"""
scripts/prepare_calibration_data.py

Generates leakage-safe, disjoint datasets for SHIELD-Stream confidence calibration:
1. Warm-up partition (rows 0-9999): initializes Welford online mean and variance.
2. Calibration partition (rows 10000-19999): non-overlapping records for Platt scaling.
3. Held-out test partition (rows 20000-29999): untouched held-out records for evaluation.

Mathematical implementation matches ai.shieldstream.ShieldStreamPipeline exactly.
"""

import argparse
import copy
import json
import math
import os
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


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


class WelfordRunningStats:
    """Matches ai.shieldstream.ShieldStreamPipeline.RunningStats exactly."""

    def __init__(self) -> None:
        self.count: int = 0
        self.mean_distance: float = 0.0
        self.m2_distance: float = 0.0
        self.mean_fare: float = 0.0
        self.m2_fare: float = 0.0

    def add(self, distance: float, fare: float) -> None:
        self.count += 1
        d_delta = distance - self.mean_distance
        self.mean_distance += d_delta / self.count
        self.m2_distance += d_delta * (distance - self.mean_distance)

        f_delta = fare - self.mean_fare
        self.mean_fare += f_delta / self.count
        self.m2_fare += f_delta * (fare - self.mean_fare)

    def z_distance(self, distance: float) -> float:
        return self._z(distance, self.mean_distance, self.m2_distance)

    def z_fare(self, fare: float) -> float:
        return self._z(fare, self.mean_fare, self.m2_fare)

    def _z(self, value: float, mean: float, m2: float) -> float:
        if self.count < 2:
            return 0.0
        variance = m2 / (self.count - 1)
        std = math.sqrt(variance)
        if std < 1e-9:
            return 0.0
        return abs(value - mean) / std


def sigmoid(x: float) -> float:
    """Matches ShieldStreamPipeline.sigmoid."""
    if x < -700:
        return 0.0
    if x > 700:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def is_structurally_valid(event: dict[str, Any]) -> bool:
    """Matches ShieldStreamPipeline.isStructurallyValid."""
    try:
        if event.get("schema_version", -1) != 1:
            return False
        dist = event.get("trip_distance")
        if dist is None or not isinstance(dist, (int, float)) or dist < 0:
            return False
        fare = event.get("fare_amount")
        if fare is None or not isinstance(fare, (int, float)):
            return False
        p_count = event.get("passenger_count", 0)
        if p_count is not None and isinstance(p_count, (int, float)) and p_count < 0:
            return False
        return True
    except Exception:
        return False


def classify_structural_fault(event: dict[str, Any]) -> str:
    """Matches ShieldStreamPipeline.classifyStructuralFault."""
    if event.get("trip_distance") is None or event.get("fare_amount") is None:
        return "MISSING_DATA"
    dist = event.get("trip_distance")
    fare = event.get("fare_amount")
    p_count = event.get("passenger_count", 0)
    if not isinstance(dist, (int, float)) or not isinstance(fare, (int, float)) or (p_count is not None and p_count < 0):
        return "CORRUPTION"
    return "SCHEMA_DRIFT"


def mutate_event(event: dict[str, Any], fault: str, intensity: float, rng: random.Random) -> list[dict[str, Any]]:
    """Mutates event matching fault_injector.py."""
    changed = copy.deepcopy(event)
    if fault == "schema_drift":
        changed["distance"] = changed.pop("trip_distance", None)
        changed["schema_version"] = 2
    elif fault == "missing_data":
        for field in ("trip_distance", "fare_amount", "PULocationID"):
            if rng.random() < intensity:
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
        raise ValueError(f"Unsupported fault: {fault}")
    return [changed]


def process_partition(
    frame: pd.DataFrame,
    start_offset: int,
    stats: WelfordRunningStats,
    trial_id: str,
    num_anomalies: int,
    num_rule_faults: int,
    seed: int,
    record_events: bool = True,
) -> tuple[list[dict[str, Any]], set[str]]:
    """
    Simulates streaming through the pipeline with stateful Welford statistics and decision rules.
    Injects faults at controlled indices to guarantee exact ground-truth accounting.
    """
    rng = random.Random(seed)
    n_rows = len(frame)
    events: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Plan fault injection positions evenly spread across the partition
    # Reserve first 500 rows for warm-up without faults
    eligible_indices = list(range(500, n_rows))
    rng.shuffle(eligible_indices)

    # 1. Continuous numerical anomalies (concept drift) - Target class for calibration
    numerical_anomaly_indices = set(eligible_indices[:num_anomalies])
    remaining = eligible_indices[num_anomalies:]

    # 2. Rule faults (schema drift, corruption, missing data, duplicate)
    per_rule_count = num_rule_faults // 4
    schema_indices = set(remaining[:per_rule_count])
    corruption_indices = set(remaining[per_rule_count : 2 * per_rule_count])
    missing_indices = set(remaining[2 * per_rule_count : 3 * per_rule_count])
    duplicate_indices = set(remaining[3 * per_rule_count : 4 * per_rule_count])

    for seq, (_, row) in enumerate(frame.iterrows()):
        source_record_id = start_offset + seq
        event_uuid = str(uuid.uuid5(uuid.NAMESPACE_OID, f"taxi-{trial_id}-{source_record_id}"))

        base_event = {
            field: (None if pd.isna(row.get(field)) else (row.get(field).isoformat() if isinstance(row.get(field), pd.Timestamp) else row.get(field)))
            for field in FIELDS
        }
        base_event.update({
            "event_id": event_uuid,
            "source_record_id": source_record_id,
            "trial_id": trial_id,
            "sequence": seq,
            "schema_version": 1,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        })

        # Determine fault to inject, if any
        fault_type = "healthy"
        is_num_anomaly = False
        intensity = 0.0

        if seq in numerical_anomaly_indices:
            fault_type = "concept_drift"
            is_num_anomaly = True
            # Vary intensity between 0.3 and 2.0 to span mild to severe drift
            intensity = rng.choice([0.3, 0.6, 1.0, 1.5, 2.0])
        elif seq in schema_indices:
            fault_type = "schema_drift"
            intensity = 1.0
        elif seq in corruption_indices:
            fault_type = "corruption"
            intensity = 1.0
        elif seq in missing_indices:
            fault_type = "missing_data"
            intensity = 0.5
        elif seq in duplicate_indices:
            fault_type = "duplicate"
            intensity = 1.0

        if fault_type != "healthy":
            mutated_events = mutate_event(base_event, fault_type, intensity, rng)
        else:
            mutated_events = [base_event]

        # Process event(s) through pipeline logic
        for ev in mutated_events:
            ev_id = ev["event_id"]
            action = "UNKNOWN"
            decision = "UNKNOWN"
            max_z = float("nan")
            raw_risk = float("nan")
            confidence = min(1.0, stats.count / 500.0)

            # 1. Duplicate check
            if ev_id in seen_ids:
                decision = "DROP"
                action = "DROP_DUPLICATE"
                raw_risk = 1.0
                max_z = float("nan")
                fault_detected = "DUPLICATE"
                if record_events:
                    events.append({
                        "event_id": ev_id,
                        "source_record_id": source_record_id,
                        "trial_id": trial_id,
                        "event_timestamp": ev["ingested_at"],
                        "max_z": max_z,
                        "raw_risk": raw_risk,
                        "confidence": confidence,
                        "ground_truth": int(is_num_anomaly),
                        "fault_type": fault_type,
                        "is_numerical_anomaly": is_num_anomaly,
                        "decision": decision,
                        "action": action,
                    })
                continue

            seen_ids.add(ev_id)

            # 2. Schema drift repair check
            if ev.get("schema_version", -1) == 2 and isinstance(ev.get("distance"), (int, float)):
                ev["trip_distance"] = ev.pop("distance")
                ev["schema_version"] = 1
                if is_structurally_valid(ev):
                    # Repaired successfully
                    decision = "REPAIR"
                    action = "AUTO_REPAIR"
                    raw_risk = 0.95
                    confidence = 0.98
                    # Updates stats after repair
                    stats.add(float(ev["trip_distance"]), float(ev["fare_amount"]))
                    if record_events:
                        events.append({
                            "event_id": ev_id,
                            "source_record_id": source_record_id,
                            "trial_id": trial_id,
                            "event_timestamp": ev["ingested_at"],
                            "max_z": max_z,
                            "raw_risk": raw_risk,
                            "confidence": confidence,
                            "ground_truth": int(is_num_anomaly),
                            "fault_type": fault_type,
                            "is_numerical_anomaly": is_num_anomaly,
                            "decision": decision,
                            "action": action,
                        })
                    continue

            # 3. Structural validation
            if not is_structurally_valid(ev):
                structural_fault = classify_structural_fault(ev)
                decision = "QUARANTINE"
                action = "ABSTAIN_QUARANTINE"
                raw_risk = 1.0
                confidence = 0.99
                if record_events:
                    events.append({
                        "event_id": ev_id,
                        "source_record_id": source_record_id,
                        "trial_id": trial_id,
                        "event_timestamp": ev["ingested_at"],
                        "max_z": max_z,
                        "raw_risk": raw_risk,
                        "confidence": confidence,
                        "ground_truth": int(is_num_anomaly),
                        "fault_type": fault_type,
                        "is_numerical_anomaly": is_num_anomaly,
                        "decision": decision,
                        "action": action,
                    })
                continue

            # 4. Continuous Anomaly Scoring (Welford statistics)
            dist_val = float(ev["trip_distance"])
            fare_val = float(ev["fare_amount"])
            z_dist = stats.z_distance(dist_val)
            z_fare = stats.z_fare(fare_val)
            max_z = max(z_dist, z_fare)
            raw_risk = sigmoid(max_z - 2.5)

            # Operational decision rule
            if stats.count >= 500 and raw_risk >= 0.80 and confidence >= 0.80:
                decision = "QUARANTINE"
                action = "ABSTAIN_QUARANTINE"
                # Quarantined events do NOT update stats
            else:
                decision = "ACCEPT"
                if stats.count >= 500 and raw_risk >= 0.55:
                    action = "SHADOW_ACCEPT"
                else:
                    action = "ACCEPT"
                # Accepted events DO update stats
                stats.add(dist_val, fare_val)

            if record_events:
                events.append({
                    "event_id": ev_id,
                    "source_record_id": source_record_id,
                    "trial_id": trial_id,
                    "event_timestamp": ev["ingested_at"],
                    "max_z": round(max_z, 6),
                    "raw_risk": round(raw_risk, 6),
                    "confidence": round(confidence, 6),
                    "ground_truth": int(is_num_anomaly),
                    "fault_type": fault_type,
                    "is_numerical_anomaly": is_num_anomaly,
                    "decision": decision,
                    "action": action,
                })

    return events, seen_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leakage-safe calibration and held-out datasets")
    parser.add_argument("--data-file", default="data/yellow_tripdata_2025-01.parquet")
    parser.add_argument("--warmup-rows", type=int, default=10000)
    parser.add_argument("--split-rows", type=int, default=10000)
    parser.add_argument("--anomalies-per-split", type=int, default=200)
    parser.add_argument("--rule-faults-per-split", type=int, default=300)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--manifest-dir", default="results/calibration")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data_file)
    if not data_path.exists():
        raise FileNotFoundError(f"Source data file not found: {data_path}")

    print(f"Reading parquet dataset: {data_path}...")
    total_needed = args.warmup_rows + 2 * args.split_rows
    frame = pd.read_parquet(data_path, columns=FIELDS).head(total_needed)
    print(f"Loaded {len(frame)} rows from dataset.")

    warmup_df = frame.iloc[0 : args.warmup_rows]
    calib_df = frame.iloc[args.warmup_rows : args.warmup_rows + args.split_rows]
    test_df = frame.iloc[args.warmup_rows + args.split_rows : args.warmup_rows + 2 * args.split_rows]

    print(f"Partition 1 (Warm-up): rows [0, {args.warmup_rows}) -> {len(warmup_df)} records")
    print(f"Partition 2 (Calibration): rows [{args.warmup_rows}, {args.warmup_rows + args.split_rows}) -> {len(calib_df)} records")
    print(f"Partition 3 (Held-Out Test): rows [{args.warmup_rows + args.split_rows}, {args.warmup_rows + 2 * args.split_rows}) -> {len(test_df)} records")

    # Step 1: Run warm-up to stabilize online Welford statistics
    stats = WelfordRunningStats()
    print("Executing warm-up phase...")
    process_partition(
        warmup_df,
        start_offset=0,
        stats=stats,
        trial_id="warmup",
        num_anomalies=0,
        num_rule_faults=0,
        seed=args.seed,
        record_events=False,
    )
    print(f"Warm-up completed. Count={stats.count}, MeanDistance={stats.mean_distance:.4f}, MeanFare={stats.mean_fare:.4f}")

    # Checkpoint stats after warm-up so calibration and test start from identical baseline stats
    stats_at_warmup = copy.deepcopy(stats)

    # Step 2: Generate Calibration Partition
    print("\nExecuting calibration partition...")
    calib_stats = copy.deepcopy(stats_at_warmup)
    calib_events, _ = process_partition(
        calib_df,
        start_offset=args.warmup_rows,
        stats=calib_stats,
        trial_id="calibration",
        num_anomalies=args.anomalies_per_split,
        num_rule_faults=args.rule_faults_per_split,
        seed=args.seed + 1,
        record_events=True,
    )

    # Step 3: Generate Held-Out Test Partition
    print("Executing held-out test partition...")
    test_stats = copy.deepcopy(stats_at_warmup)
    test_events, _ = process_partition(
        test_df,
        start_offset=args.warmup_rows + args.split_rows,
        stats=test_stats,
        trial_id="held_out_test",
        num_anomalies=args.anomalies_per_split,
        num_rule_faults=args.rule_faults_per_split,
        seed=args.seed + 2,
        record_events=True,
    )

    # Save datasets
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    calib_csv = out_dir / "calibration_events.csv"
    calib_jsonl = out_dir / "calibration_events.jsonl"
    test_csv = out_dir / "held_out_events.csv"
    test_jsonl = out_dir / "held_out_events.jsonl"

    pd.DataFrame(calib_events).to_csv(calib_csv, index=False)
    with calib_jsonl.open("w", encoding="utf-8") as f:
        for row in calib_events:
            f.write(json.dumps(row) + "\n")

    pd.DataFrame(test_events).to_csv(test_csv, index=False)
    with test_jsonl.open("w", encoding="utf-8") as f:
        for row in test_events:
            f.write(json.dumps(row) + "\n")

    # Leakage check: verify disjoint source record IDs
    calib_source_ids = {e["source_record_id"] for e in calib_events}
    test_source_ids = {e["source_record_id"] for e in test_events}
    leakage_overlap = calib_source_ids.intersection(test_source_ids)
    if leakage_overlap:
        raise RuntimeError(f"DATA LEAKAGE DETECTED: {len(leakage_overlap)} overlapping source records!")

    # Class distribution summaries
    calib_anomalies = sum(e["ground_truth"] for e in calib_events)
    test_anomalies = sum(e["ground_truth"] for e in test_events)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(data_path),
        "total_source_rows_read": total_needed,
        "partitions": {
            "warmup": {
                "start_offset": 0,
                "end_offset": args.warmup_rows,
                "row_count": args.warmup_rows,
                "purpose": "Initialize Welford online mean and variance estimators",
                "final_welford_stats": {
                    "count": stats.count,
                    "mean_distance": round(stats.mean_distance, 4),
                    "mean_fare": round(stats.mean_fare, 4),
                },
            },
            "calibration": {
                "start_offset": args.warmup_rows,
                "end_offset": args.warmup_rows + args.split_rows,
                "row_count": len(calib_events),
                "file_csv": str(calib_csv),
                "file_jsonl": str(calib_jsonl),
                "numerical_anomalies": calib_anomalies,
                "clean_or_rule_events": len(calib_events) - calib_anomalies,
                "fault_breakdown": pd.Series([e["fault_type"] for e in calib_events]).value_counts().to_dict(),
            },
            "held_out_test": {
                "start_offset": args.warmup_rows + args.split_rows,
                "end_offset": args.warmup_rows + 2 * args.split_rows,
                "row_count": len(test_events),
                "file_csv": str(test_csv),
                "file_jsonl": str(test_jsonl),
                "numerical_anomalies": test_anomalies,
                "clean_or_rule_events": len(test_events) - test_anomalies,
                "fault_breakdown": pd.Series([e["fault_type"] for e in test_events]).value_counts().to_dict(),
            },
        },
        "leakage_verification": {
            "calibration_source_id_min": min(calib_source_ids),
            "calibration_source_id_max": max(calib_source_ids),
            "test_source_id_min": min(test_source_ids),
            "test_source_id_max": max(test_source_ids),
            "source_id_intersection_size": len(leakage_overlap),
            "leakage_free": len(leakage_overlap) == 0,
        },
        "random_seeds": {
            "base_seed": args.seed,
            "calibration_seed": args.seed + 1,
            "test_seed": args.seed + 2,
        },
    }

    manifest_path = manifest_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nSaved calibration data: {calib_csv} ({len(calib_events)} records, {calib_anomalies} positive numerical anomalies)")
    print(f"Saved held-out data: {test_csv} ({len(test_events)} records, {test_anomalies} positive numerical anomalies)")
    print(f"Saved split manifest: {manifest_path}")
    print(f"Leakage verification: ZERO source_record_id overlap ({len(leakage_overlap)} overlap). Disjoint guarantee satisfied.")


if __name__ == "__main__":
    main()
