"""
tests/test_calibration.py

Automated test suite verifying confidence calibration, Platt scaling,
leakage prevention, and preservation of pipeline behavior.
Fulfills Phase 8 of IEEE INDICON 2026 Paper 4687 revision.
"""

import json
import math
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from scripts.prepare_calibration_data import (
    WelfordRunningStats,
    is_structurally_valid,
    classify_structural_fault,
    mutate_event,
    sigmoid,
)
from scripts.calibrate_platt import (
    validate_calibration_data,
    fit_platt_scaling,
)


@pytest.fixture
def initialized_welford():
    """Returns a WelfordRunningStats instance warmed up on synthetic taxi data."""
    stats = WelfordRunningStats()
    rng = np.random.RandomState(42)
    distances = rng.exponential(scale=2.5, size=2000)
    fares = 3.0 + 3.5 * distances + rng.normal(0, 1.5, size=2000)
    fares = np.maximum(fares, 2.5)

    for d, f in zip(distances, fares):
        stats.add(float(d), float(f))
    return stats


@pytest.fixture
def calibration_model_data():
    """Loads fitted calibration model metadata."""
    model_path = Path("results/calibration/calibration_model.json")
    assert model_path.exists(), "Model file must exist"
    return json.loads(model_path.read_text(encoding="utf-8"))


# --- Test 1: Healthy Low-Risk Events ---
def test_healthy_low_risk_events(initialized_welford, calibration_model_data):
    """Verify that typical healthy taxi rides produce low Z-scores and low calibrated probability."""
    stats = initialized_welford
    slope = calibration_model_data["parameters"]["A_slope"]
    intercept = calibration_model_data["parameters"]["B_intercept"]

    # Typical distance and fare near mean
    d = stats.mean_distance
    f = stats.mean_fare

    z_d = stats.z_distance(d)
    z_f = stats.z_fare(f)
    max_z = max(z_d, z_f)

    margin = max_z - 2.5
    calibrated_prob = sigmoid(slope * margin + intercept)

    assert max_z < 1.0, f"Typical event should have small Z-score, got {max_z}"
    assert calibrated_prob < 0.10, f"Typical event should have calibrated probability < 0.10, got {calibrated_prob}"


# --- Test 2: Numerical Anomalies at Different Magnitudes ---
def test_numerical_anomalies_monotonicity(initialized_welford, calibration_model_data):
    """Verify that increasing concept drift intensity strictly increases Z-score and calibrated probability."""
    stats = initialized_welford
    slope = calibration_model_data["parameters"]["A_slope"]
    intercept = calibration_model_data["parameters"]["B_intercept"]

    base_d = stats.mean_distance
    base_f = stats.mean_fare

    intensities = [0.2, 0.5, 1.0, 2.0, 4.0]
    prev_z = -1.0
    prev_prob = -1.0

    for intensity in intensities:
        mutated_d = base_d * (1.0 + 5.0 * intensity)
        mutated_f = base_f * (1.0 + 3.0 * intensity)

        z = max(stats.z_distance(mutated_d), stats.z_fare(mutated_f))
        margin = z - 2.5
        prob = sigmoid(slope * margin + intercept)

        assert z > prev_z, f"Z-score should strictly increase with intensity: {z} <= {prev_z}"
        assert prob > prev_prob, f"Calibrated prob should strictly increase with intensity: {prob} <= {prev_prob}"

        prev_z = z
        prev_prob = prob


# --- Test 3: Missing and Unmatched Ground-Truth Records ---
def test_missing_and_unmatched_records():
    """Verify data validation catches missing required fields and unparseable events."""
    bad_df = pd.DataFrame([
        {"event_id": "e1", "source_record_id": 1, "max_z": 1.2},  # missing ground_truth
    ])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_calibration_data(bad_df)


# --- Test 4: Duplicate Event IDs and Deduplication ---
def test_duplicate_event_handling():
    """Verify that duplicate occurrences are detected and do not distort continuous calibration training."""
    df_with_dups = pd.DataFrame([
        {"event_id": "dup_1", "source_record_id": 1, "max_z": 1.5, "raw_risk": 0.27, "ground_truth": 0, "fault_type": "healthy"},
        {"event_id": "dup_1", "source_record_id": 1, "max_z": 1.5, "raw_risk": 0.27, "ground_truth": 0, "fault_type": "duplicate"},
        {"event_id": "e2", "source_record_id": 2, "max_z": 6.5, "raw_risk": 0.98, "ground_truth": 1, "fault_type": "concept_drift"},
    ])

    deduped = validate_calibration_data(df_with_dups)
    assert len(deduped) == 2, "Deduplication must retain exactly 1 instance per event_id"


# --- Test 5: Leakage Check Between Partitions ---
def test_data_leakage_prevention():
    """Verify that calibration and test datasets have zero source_record_id overlap."""
    calib_csv = Path("data/calibration_events.csv")
    test_csv = Path("data/held_out_events.csv")
    manifest_path = Path("results/calibration/split_manifest.json")

    assert calib_csv.exists(), "Calibration data must exist"
    assert test_csv.exists(), "Test data must exist"
    assert manifest_path.exists(), "Manifest must exist"

    df_calib = pd.read_csv(calib_csv)
    df_test = pd.read_csv(test_csv)

    calib_ids = set(df_calib["source_record_id"].dropna().astype(int))
    test_ids = set(df_test["source_record_id"].dropna().astype(int))

    overlap = calib_ids.intersection(test_ids)
    assert len(overlap) == 0, f"DATA LEAKAGE: Overlapping source records found: {len(overlap)}"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["leakage_verification"]["leakage_free"] is True


# --- Test 6: Empty and Single-Class Dataset Error Handling ---
def test_empty_and_single_class_error_handling():
    """Verify calibrate_platt fails gracefully on empty or single-class input."""
    empty_df = pd.DataFrame(columns=["event_id", "source_record_id", "max_z", "raw_risk", "ground_truth", "fault_type"])
    with pytest.raises(ValueError, match="Calibration dataset is empty"):
        validate_calibration_data(empty_df)

    single_class_df = pd.DataFrame([
        {"event_id": f"e{i}", "source_record_id": i, "max_z": 1.0, "raw_risk": 0.18, "ground_truth": 0, "fault_type": "healthy"}
        for i in range(10)
    ])
    with pytest.raises(ValueError, match="Insufficient classes"):
        validate_calibration_data(single_class_df)


# --- Test 7: Calibration Parameter Serialization and Loading ---
def test_model_serialization_and_loading(calibration_model_data):
    """Verify calibration parameters load properly and produce matching outputs."""
    params = calibration_model_data["parameters"]
    assert "A_slope" in params
    assert "B_intercept" in params

    a = params["A_slope"]
    b = params["B_intercept"]

    # Test numerical evaluation
    z = 4.5
    margin = z - 2.5
    prob_margin = sigmoid(a * margin + b)

    a_dir = params["equivalent_direct_z_slope"]
    b_dir = params["equivalent_direct_z_intercept"]
    prob_direct = sigmoid(a_dir * z + b_dir)

    assert math.isclose(prob_margin, prob_direct, rel_tol=1e-5), "Margin and direct parametrizations must match"


# --- Test 8: Probabilities Strictly Bounded in [0, 1] ---
def test_probability_bounded_in_unit_interval(calibration_model_data):
    """Verify calibrated probabilities never escape [0, 1] for extreme inputs."""
    a = calibration_model_data["parameters"]["A_slope"]
    b = calibration_model_data["parameters"]["B_intercept"]

    extreme_zs = [-100.0, -10.0, 0.0, 2.5, 10.0, 100.0, 1000.0]
    for z in extreme_zs:
        margin = z - 2.5
        prob = sigmoid(a * margin + b)
        assert 0.0 <= prob <= 1.0, f"Calibrated probability {prob} out of bounds for z={z}"


# --- Test 9: Preservation of Existing Rule-Based Detection and Repair ---
def test_preservation_of_rule_based_behavior():
    """Verify schema drift repair and structural fault classification are preserved."""
    # 1. Schema drift event (schema_version 2, distance instead of trip_distance)
    schema_event = {
        "event_id": "test_schema_1",
        "distance": 3.4,
        "fare_amount": 14.5,
        "passenger_count": 1,
        "schema_version": 2,
    }
    assert not is_structurally_valid(schema_event), "schema_version 2 must be invalid prior to repair"

    # Simulate repair
    schema_event["trip_distance"] = schema_event.pop("distance")
    schema_event["schema_version"] = 1
    assert is_structurally_valid(schema_event), "Repaired event must be structurally valid"

    # 2. Corrupted event
    corrupt_event = {
        "event_id": "test_corrupt_1",
        "trip_distance": 2.0,
        "fare_amount": "INVALID",
        "passenger_count": -99,
        "schema_version": 1,
    }
    assert not is_structurally_valid(corrupt_event)
    assert classify_structural_fault(corrupt_event) == "CORRUPTION"

    # 3. Missing data event
    missing_event = {
        "event_id": "test_missing_1",
        "trip_distance": None,
        "fare_amount": 10.0,
        "passenger_count": 1,
        "schema_version": 1,
    }
    assert not is_structurally_valid(missing_event)
    assert classify_structural_fault(missing_event) == "MISSING_DATA"
