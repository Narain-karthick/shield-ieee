# SHIELD-Stream Phase 2 — Technical Audit for Confidence Calibration

**Research Paper:** “SHIELD-Stream: A Confidence-Aware Self-Healing Framework for Reliable Streaming Big Data Pipelines”  
**Paper ID:** 4687 (IEEE INDICON 2026)  
**Audit Purpose:** Evaluate the feasibility of implementing Option 1 (Probability Calibration using Platt Scaling) in response to Reviewer Comment:
> *“A real confidence or calibration procedure needs to be added, or the title and contributions of this paper should be revised.”*

---

## Executive Summary & Requirement Checklist

| Requirement | Status | Evidence / File Path | Action Needed |
|---|---|---|---|
| **1. Implementation Location** | **Verified** | `flink-job/.../ShieldStreamPipeline.java`, `producer/producer.py`, `fault_injector/fault_injector.py` | None; all functional components located and verified. |
| **2. Risk Formula Audit** | **Verified** | `ShieldStreamPipeline.java:335-336, 608-613` | Formula identified: $\text{risk} = \text{sigmoid}(Z_{\max} - 2.5)$. Rule faults use hardcoded constants ($0.95, 1.0$). |
| **3. Confidence Formula Audit** | **Verified** | `ShieldStreamPipeline.java:338-342, 201, 221, 241, 274, 309` | **Deficiency verified:** `confidence` is simply $\min(1.0, \text{count}/500.0)$ for drift, and hardcoded literals ($0.98, 0.99, 1.0$) for rules. It is not a statistical confidence or calibrated probability. |
| **4. Ground-Truth Labels** | **Verified** | `fault_injector/fault_injector.py:36-43, 83-84`, `scripts/evaluate_phase2.py:48` | Ground-truth faults logged to Kafka topic `shield-incidents-v2`. Healthy events are implicit (non-incidents). |
| **5. Experimental Metrics** | **Verified** | `results/phase2-aggregate.json`, `results/phase2-trials.csv`, `results/phase2-run1..5.json` | Exact paper metrics confirmed: Precision 71.82%, Recall 91.24%, F1 80.37%, exactly 179 FPs across all 5 trials. |
| **6. Platt Scaling Feasibility** | **Feasible with additional logging** | `ShieldStreamPipeline.java:364-380`, `results/` | **Not feasible from existing saved files on disk alone.** Healthy events with $risk < 0.55$ and false negatives with $risk < 0.55$ are never emitted to Kafka or saved to disk. Feasible once full-stream logging is added. |
| **7. Data Leakage Risks** | **Identified** | `producer/producer.py:57-60, 72-74`, `scripts/aggregate_trials.py:8-10` | High leakage risk if pooling trials: All 5 trials stream the exact same 5,000 parquet rows. Trials must use disjoint data slices. |
| **8. Proposed Calibration Protocol** | **Formulated** | Defined in Section 8 of this report | Use non-overlapping taxi trip slices; fit logistic calibration on continuous $Z_{\max}$ score; evaluate with ECE, Brier score, and reliability diagrams. |

---

## 1. Project Structure and Relevant File Paths

The codebase is organized as a Dockerized Kafka–Flink streaming system with Python data generation, injection, evaluation, and reporting tools.

```
c:\Users\kiran\OneDrive\Desktop\shield-stream-phase1 (2)\shield-stream-phase1\shield-stream
├── .env.example
├── docker-compose.yml
├── README.md
├── requirements.txt
├── data/
│   └── yellow_tripdata_2025-01.parquet               # NYC Yellow Taxi dataset (59.1 MB)
├── producer/
│   └── producer.py                                   # Kafka producer replaying taxi records
├── fault_injector/
│   └── fault_injector.py                             # Injects 5 synthetic faults into taxi stream
├── flink-job/
│   ├── pom.xml                                       # Maven build (Flink 1.20.1, Java 17)
│   └── src/main/java/ai/shieldstream/
│       ├── ShieldStreamPipeline.java                 # Phase 2 confidence-aware streaming pipeline
│       └── BaselinePipeline.java                     # Phase 1 rule-based baseline pipeline
├── scripts/
│   ├── download_dataset.py                           # Downloads Jan 2025 taxi dataset
│   ├── evaluate_phase2.py                            # Kafka consumer evaluating detection vs ground truth
│   ├── aggregate_trials.py                           # Computes 5-trial statistics and 95% CIs
│   └── plot_results.py                               # Generates matplotlib evaluation figures
└── results/
    ├── docker-run1.log ... docker-run5.log           # Container execution logs
    ├── phase2-run1.json ... phase2-run5.json         # Trial-level evaluation summaries
    ├── phase2-summary.json                           # Latest run summary (matches run 1)
    ├── phase2-trials.csv                             # Per-trial CSV metrics table
    ├── phase2-aggregate.json                         # Aggregated 5-trial statistics (mean, SD, 95% CI)
    └── figures/
        ├── trial_stability.pdf / .png                # Line plot across 5 trials
        ├── per_fault_recall.pdf / .png               # Bar chart of per-fault recall
        └── phase1_vs_phase2_recall.pdf / .png        # Phase 1 vs Phase 2 recall comparison
```

### Functional Component Mapping

| Subsystem | File Path | Primary Class / Function | Description |
|---|---|---|---|
| **Data Ingestion** | `producer/producer.py` | `main()`, `build_event()` | Reads NYC taxi parquet, produces events to `taxi-raw`. |
| **Fault Injection** | `fault_injector/fault_injector.py` | `mutate()`, `incident()` | Subscribes to `taxi-clean-v2`, mutates records, publishes to `taxi-raw` and `shield-incidents-v2`. |
| **Streaming Engine** | `flink-job/.../ShieldStreamPipeline.java` | `ShieldStreamPipeline.main()` | Flink job configured with Kafka source, keyBy `global`, parallelism 1, 10s checkpointing. |
| **Healing & Detection** | `flink-job/.../ShieldStreamPipeline.java` | `ConfidenceAwareHealer` | `KeyedProcessFunction` handling duplicates, schema repair, structural validation, and drift. |
| **Online Statistics** | `flink-job/.../ShieldStreamPipeline.java` | `RunningStats` | Stateful Welford accumulator for distance and fare. |
| **Baseline Pipeline** | `flink-job/.../BaselinePipeline.java` | `BaselinePipeline.main()` | Phase 1 baseline using simple non-null and positive value checks. |
| **Evaluation** | `scripts/evaluate_phase2.py` | `main()` | Matches `shield-incidents-v2` and `shield-metrics` on `event_id` to compute TP, FP, FN, P, R, F1. |
| **Aggregation** | `scripts/aggregate_trials.py` | `main()`, `describe()` | Computes mean, sample standard deviation, and Student's $t$ 95% confidence intervals across 5 runs. |
| **Visualization** | `scripts/plot_results.py` | `trial_stability()`, `fault_recall()`, `phase_comparison()` | Produces publication-ready PDF and PNG charts. |

---

## 2. Current Risk and Confidence Formulas

### 2.1 The Exact Formula for Risk Scoring

Risk scoring in `ShieldStreamPipeline.java` operates in two distinct modes:

#### A. Continuous Numerical / Concept Drift Risk (Lines 317–336, 608–613)
For structurally valid events, risk is computed from online Welford Z-scores for `trip_distance` ($d$) and `fare_amount` ($f$):

$$Z_{\text{distance}} = \frac{|d - \mu_{\text{distance}}|}{\sigma_{\text{distance}}}, \quad Z_{\text{fare}} = \frac{|f - \mu_{\text{fare}}|}{\sigma_{\text{fare}}}$$

$$Z_{\max} = \max(Z_{\text{distance}}, Z_{\text{fare}})$$

$$\text{risk} = \text{sigmoid}(Z_{\max} - 2.5) = \frac{1}{1 + e^{-(Z_{\max} - 2.5)}}$$

```java
// File: ShieldStreamPipeline.java, Lines 326-336
double distanceZ = stats.zDistance(distance);
double fareZ = stats.zFare(fare);
double maximumZ = Math.max(distanceZ, fareZ);
double risk = sigmoid(maximumZ - 2.5);
```

#### B. Discrete Rule-Based Risk (Hardcoded Constants)
For deterministic schema, data-quality, and stateful violations, `risk` is a hardcoded static float literal:
- **Malformed JSON:** $\text{risk} = 1.0$ (Line 200)
- **Missing `event_id`:** $\text{risk} = 1.0$ (Line 220)
- **Duplicate Event:** $\text{risk} = 1.0$ (Line 240)
- **Schema Drift (auto-repaired):** $\text{risk} = 0.95$ (Line 284)
- **Structural Invalidity (Missing Data / Corruption):** $\text{risk} = 1.0$ (Line 308)

---

### 2.2 The Exact Formula for Confidence

Confidence is calculated or assigned as follows:

#### A. Concept Drift Confidence (Lines 338–342)
$$\text{confidence} = \min\left(1.0, \frac{\text{stats.count}}{500.0}\right)$$

```java
// File: ShieldStreamPipeline.java, Lines 338-342
double confidence =
        Math.min(
                1.0,
                stats.count / 500.0
        );
```

#### B. Discrete Rule-Based Confidence (Hardcoded Constants)
- **Malformed JSON:** $\text{confidence} = 1.0$ (Line 201)
- **Missing `event_id`:** $\text{confidence} = 1.0$ (Line 221)
- **Duplicate Event:** $\text{confidence} = 0.999$ (Line 241)
- **Schema Drift (auto-repair):** $\text{confidence} = 0.98$ (Lines 274, 285)
- **Missing Data / Corruption:** $\text{confidence} = 0.99$ (Line 309)

---

### 2.3 Semantic Analysis of Current "Confidence"

> **Critical Finding:** The variable named `confidence` in `ShieldStreamPipeline.java` is **not a statistical confidence score, not a posterior probability, and not a model uncertainty metric**.
>
> 1. For concept drift, it is strictly an **observation counter / warm-up ramp**. As soon as 500 records have passed, `confidence` is identically $1.0$ for all subsequent events regardless of feature values or volatility.
> 2. For all structural and rule faults, `confidence` is a **hardcoded literal** ($0.98, 0.99, 0.999, 1.0$) chosen arbitrarily in code.
> 3. The reviewer's comment is **entirely justified and technically accurate**: the existing implementation lacks any calibrated confidence estimation.

---

### 2.4 Thresholds and Operational Decision Rules

The pipeline partitions events into three operating regions using `stats.count`, `risk`, and `confidence`:

```
                           Z_max scale
  0                 2.70                     3.89                +inf
--|-------------------|------------------------|-------------------|-->
    Normal Events        Drift Warning (Shadow)    Critical Drift (DLQ)
    risk < 0.55          0.55 <= risk < 0.80       risk >= 0.80 && conf >= 0.80
    Action: ACCEPT       Action: SHADOW_ACCEPT     Action: ABSTAIN_QUARANTINE
    (No metric emitted)  (Emitted to metrics)      (Emitted to metrics & DLQ)
```

1. **Normal Events ($\text{risk} < 0.55 \iff Z_{\max} < 2.70$):**
   - Condition: Structurally valid, $Z_{\max} < 2.701$.
   - Pipeline action: Collected to clean sink (`taxi-clean-v2`), online stats updated via `updateStatistics(event)`.
   - Telemetry: **No metric event is emitted to `shield-metrics`!** (Lines 364–380).
2. **Suspicious Events / Drift Warning ($0.55 \le \text{risk} < 0.80 \iff 2.70 \le Z_{\max} < 3.89$):**
   - Condition: `stats.count >= 500 && risk >= 0.55 && risk < 0.80`.
   - Pipeline action: Collected to clean sink (`output.collect(raw)`), online stats updated (`updateStatistics(event)`).
   - Telemetry: Emitted to `shield-metrics` with `fault = "DRIFT_WARNING"`, `action = "SHADOW_ACCEPT"`.
   - Evaluation impact: `evaluate_phase2.py` explicitly ignores `DRIFT_WARNING` (Line 49).
3. **Critical Concept Drift Events ($\text{risk} \ge 0.80 \land \text{confidence} \ge 0.80 \iff Z_{\max} \ge 3.886 \land \text{count} \ge 500$):**
   - Condition: `stats.count >= 500 && risk >= 0.80 && confidence >= 0.80`.
   - Pipeline action: Quarantined to DLQ (`taxi-dlq-v2`), execution returns immediately **without** updating statistics.
   - Telemetry: Emitted to `shield-metrics` with `fault = "CONCEPT_DRIFT"`, `action = "ABSTAIN_QUARANTINE"`.

---

### 2.5 Warm-Up Period and State-Update Rules

- **Warm-Up Period:**
  - Code-level gate: `stats.count >= 500`. Before 500 events, neither `CONCEPT_DRIFT` nor `DRIFT_WARNING` can be emitted.
  - Experimental protocol: `README.md` prescribes running 5,000 clean events via `producer.py --limit 5000 --rate 200` before fault injection.
- **Welford Algorithm Implementation:**
  The `RunningStats` class (Lines 411–493) implements textbook Welford online recurrence for $M_1$ (mean) and $M_2$ (sum of squared differences):
  $$M_{1, k} = M_{1, k-1} + \frac{x_k - M_{1, k-1}}{k}$$
  $$M_{2, k} = M_{2, k-1} + (x_k - M_{1, k-1})(x_k - M_{1, k})$$
  $$\sigma = \sqrt{\frac{M_{2, k}}{k - 1}}$$
- **Do Detected Faulty Events Update Online Statistics?**
  - Malformed JSON, Missing Event ID, Duplicates, and Quarantined Structural Faults: **No.** (DLQ side output, early return before `updateStatistics`).
  - Quarantined Concept Drift ($Z_{\max} \ge 3.89$): **No.** (DLQ side output, early return at Line 361).
  - Repaired Schema Drift: **Yes.** (Updated at Line 290 after field normalization).
  - Shadow-accepted Drift Warnings ($2.70 \le Z_{\max} < 3.89$): **Yes.** (Updated at Line 365).
  - False Negatives (Injected drift with $Z_{\max} < 3.89$): **Yes.** (Passes to Line 365).
- **Concept Drift vs. Individual Outliers:**
  Concept drift is evaluated strictly on an **individual event level** via $Z_{\max}$. There is no windowed distribution test (e.g., ADWIN, Page-Hinkley, Kolmogorov-Smirnov). An isolated extreme taxi fare or distance is tagged as `CONCEPT_DRIFT`.

---

## 3. Feasibility of Platt Scaling

### 3.1 Assembly of Event-Level Calibration Dataset

To fit Platt scaling, we need a dataset with columns:
`[event_id, risk_score, is_faulty, fault_type, trial_id]`

| Column Name | Origin in Codebase | Availability in Existing Saved Output | Required Action to Extract / Generate |
|---|---|---|---|
| `event_id` | Deterministic UUID in `producer.py:41` or injected UUID in `fault_injector.py:80`. | **Missing on disk.** Only aggregated counts are saved in `results/*.json`. Present temporarily in Kafka during run. | Log `event_id` for every processed event to disk or Kafka evaluation topic. |
| `risk_score` | Calculated in `ShieldStreamPipeline.java:336`. | **Incomplete in Kafka; Missing on disk.** Emitted to `shield-metrics` only when $risk \ge 0.55$ or for rule faults. Normal events ($risk < 0.55$) have no score recorded anywhere. | Modify `ShieldStreamPipeline.java` to output `risk` for all events to an evaluation sink. |
| `is_faulty` | Injected incidents have `ground_truth: True` in `shield-incidents-v2`. Healthy events have no entry. | **Missing on disk.** Stored in Kafka `shield-incidents-v2` during live runs, but healthy records are never tagged. | Generate explicit ground-truth mapping where injected = 1, healthy = 0. |
| `fault_type` | `fault_injector.py:38` (`schema_drift`, `corruption`, `missing_data`, `concept_drift`, `duplicate`). | **Missing on disk.** Stored in Kafka `shield-incidents-v2` during live runs. | Include in ground-truth log table. |
| `trial_id` | Set externally by execution sequence. | **Missing in event streams.** Inferred from separate file paths (`phase2-run1.json` ... `run5.json`). | Inject `trial_id` into event metadata. |

### 3.2 Evaluation of Ground-Truth Independence
- `shield-incidents-v2` contains **independent ground truth**: it records the exact fault type injected by `fault_injector.py` before the Flink job processes it.
- In `evaluate_phase2.py`, system predictions are taken from `shield-metrics`. True positives are computed as $\text{detected} \cap \text{true\_ids}$, and false positives as $\text{detected} - \text{true\_ids}$.
- Ground truth is strictly independent of system predictions for injected faults. However, healthy events are currently treated as "healthy" solely by virtue of not being in `true_ids`.

### 3.3 Suitability of the Risk Score for Platt Scaling

Platt scaling was originally developed to map raw support vector machine margins $f(x) \in (-\infty, \infty)$ to posterior probabilities:

$$P(Y=1 \mid s) = \frac{1}{1 + e^{A s + B}}$$

In SHIELD-Stream:
1. **Rule Faults:** For schema drift, corruption, missing data, and duplicates, the risk score is a constant ($0.95$ or $1.0$). Platt scaling is **neither mathematically meaningful nor applicable** to deterministic rule triggers. Rule-based detections do not have an underlying continuous probability density.
2. **Concept Drift Score:** The risk score is continuous:
   $$\text{risk} = \sigma(Z_{\max} - 2.5)$$
   Notice that the score is already squashed by a logistic sigmoid. Fitting Platt scaling $\sigma(A \cdot \text{risk} + B)$ directly on `risk` is equivalent to fitting a two-parameter logistic recalibration on a monotonic score.
   Even more natural is fitting Platt scaling directly on the pre-sigmoid margin $m = Z_{\max} - 2.5$ (or on $Z_{\max}$ directly):
   $$P(Y=1 \mid Z_{\max}) = \frac{1}{1 + e^{-(A \cdot Z_{\max} + B)}}$$
   This is statistically defensible, computationally stable, and resolves the reviewer's concern directly.

### 3.4 Verdict on Platt Scaling Feasibility

> **Feasibility Verdict: FEASIBLE WITH ADDITIONAL LOGGING**
> 
> Platt scaling **cannot** be implemented from the files currently saved in the repository because no event-level dataset of scores and labels was persisted to disk. Furthermore, the streaming job currently silences all telemetry for normal events ($risk < 0.55$).
> 
> However, the codebase has all necessary architectural building blocks (deterministic UUIDs, independent incident logging, Kafka streams). Adding an evaluation logging sink to record `(event_id, risk_score, is_faulty)` for all events makes Platt scaling fully feasible.

---

## 4. Inspection of Evaluation and Experimental Results

### 4.1 Verification of Reported Results

The figures reported in the user prompt were cross-referenced against `results/phase2-aggregate.json`, `results/phase2-trials.csv`, and `results/phase2-run1..5.json`.

| Metric | Paper Context / Claim | Verified Aggregate Mean | Sample SD | 95% Confidence Interval | Source File |
|---|---|---|---|---|---|
| **Precision** | $\approx 71.82\%$ | **0.71822** (71.82%) | 0.000572 | [0.71751, 0.71893] | `phase2-aggregate.json:5-8` |
| **Recall** | $\approx 91.24\%$ | **0.91240** (91.24%) | 0.002608 | [0.90916, 0.91564] | `phase2-aggregate.json:11-14` |
| **F1-Score** | $\approx 80.37\%$ | **0.80374** (80.37%) | 0.001354 | [0.80206, 0.80542] | `phase2-aggregate.json:17-20` |
| **False Positives** | 179 per trial | **179.0** | **0.000000** | [179.0, 179.0] | `phase2-trials.csv:2-6` |
| **Healthy FP Rate** | $179 / 5000 = 3.58\%$ | **0.03580** (3.58%) | 0.000000 | [0.03580, 0.03580] | `phase2-aggregate.json:62-65` |
| **Repairs** | 100 per trial | **100.0** | **0.000000** | [100.0, 100.0] | `phase2-trials.csv:2-6` |
| **Repair Success Rate** | 100% | **1.0000** (100.0%) | 0.000000 | [1.0000, 1.0000] | `phase2-aggregate.json:56-59` |

### 4.2 Per-Fault Breakdown Across Trials

From `results/phase2-trials.csv` and individual JSON runs:

| Trial | TP | FP | FN | Schema Drift Recall | Corruption Recall | Duplicate Recall | Missing Data Recall | Concept Drift Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 456 | 179 | 44 | 1.00 (100/100) | 1.00 (100/100) | 1.00 (100/100) | 1.00 (100/100) | **0.56** (56/100) |
| **2** | 455 | 179 | 45 | 1.00 (100/100) | 1.00 (100/100) | 1.00 (100/100) | 1.00 (100/100) | **0.55** (55/100) |
| **3** | 455 | 179 | 45 | 1.00 (100/100) | 1.00 (100/100) | 1.00 (100/100) | 1.00 (100/100) | **0.55** (55/100) |
| **4** | 457 | 179 | 43 | 1.00 (100/100) | 1.00 (100/100) | 1.00 (100/100) | 1.00 (100/100) | **0.57** (57/100) |
| **5** | 458 | 179 | 42 | 1.00 (100/100) | 1.00 (100/100) | 1.00 (100/100) | 1.00 (100/100) | **0.58** (58/100) |
| **Mean**| **456.2** | **179.0** | **43.8** | **1.00** | **1.00** | **1.00** | **1.00** | **0.562** |

### 4.3 Why False Positives Are Exactly 179 Across All Trials

In `results/phase2-aggregate.json`, the standard deviation of false positives is $0.000$. This occurs because:
1. `producer.py` reads `yellow_tripdata_2025-01.parquet` sequentially from row 0 with `--limit 5000`.
2. In all 5 trials, the exact same first 5,000 parquet rows are replayed.
3. Because Welford updates are deterministic, the running mean and variance evolve identically across runs.
4. Exactly the same 179 taxi trips have $Z_{\max} \ge 3.886$ and are quarantined as false-positive concept drift.
5. In a Gaussian distribution, $P(|Z| \ge 3.89) \approx 0.0001$. In real taxi data, distances and fares have heavy, right-skewed tails, causing an empirical false-positive rate of $3.58\%$ ($179 / 5000$). This proves the raw risk score is miscalibrated.

### 4.4 Missing Experimental Dimensions
- **Latency and Throughput:** Neither end-to-end event latency nor operator throughput is computed, logged, or plotted.
- **Random Seeds:** `fault_injector.py` uses `random.random() < intensity` without setting `random.seed()`.

---

## 5. Data Leakage Risks and Methodological Pitfalls

Any future calibration experiment must guard against several subtle data leakage pathways identified in the current setup:

```
                      DATA LEAKAGE RISK MAP
                      
[parquet dataset]
       │
       ▼ (first 5,000 rows replayed every trial)
 [producer.py] ──────────► [taxi-raw]
                               │
                               ▼ (accepted clean records)
                         [taxi-clean-v2]
                               │
                               ▼ (consumed by injector)
                     [fault_injector.py]
                               │
                               ▼ (mutated records with new UUID)
                          [taxi-raw]
```

1. **Trial Duplication / Cross-Trial Leakage:**
   Because all 5 trials replay the identical first 5,000 parquet records, pooling records across trials for cross-validation will cause identical underlying taxi trips to appear in both the calibration training set and the calibration test set. This will artificially deflate calibration error (ECE and Brier score).
2. **Source Record Contamination:**
   `fault_injector.py` consumes records from `taxi-clean-v2` and mutates them. If the original unmutated trip is in the training set and the mutated counterpart is in the test set, features such as pickup/dropoff locations and base passenger counts are leaked.
3. **Warm-Up Period Contamination:**
   Online statistics computed during warm-up reflect the distribution of the warm-up data. If warm-up data is also included in test evaluation, the model's test performance reflects training fit rather than generalization.

---

## 6. Proposed Scientifically Valid Calibration Experiment

To address reviewer criticism rigorously, calibration should be implemented following these guidelines:

### 6.1 Experimental Protocol

1. **Partition Strategy (Disjoint Temporal / Slice Splitting):**
   - Partition `yellow_tripdata_2025-01.parquet` into three disjoint, non-overlapping slices:
     - **Slice A (Warm-Up & Reference Stats):** Rows 0 to 9,999 (10,000 events) to stabilize Welford statistics.
     - **Slice B (Calibration Training):** Rows 10,000 to 19,999 (10,000 events), with synthetic faults injected into a subset (e.g., 500 faults). Used exclusively to fit Platt scaling parameters $(A, B)$.
     - **Slice C (Held-Out Test Set):** Rows 20,000 to 29,999 (10,000 events), with separate fault injections. Kept completely untouched until evaluation.
2. **Model Formulation (Platt Scaling on Anomaly Margin):**
   Fit a logistic regression model on the continuous anomaly feature $Z_{\max}$ (or pre-sigmoid margin $Z_{\max} - 2.5$) on Slice B:
   $$P(Y = 1 \mid Z_{\max}) = \frac{1}{1 + e^{-(A \cdot Z_{\max} + B)}}$$
   Estimate parameters $A$ and $B$ via maximum likelihood estimation (minimizing binary cross-entropy) with regularized logistic regression.
3. **Separation of Fault Detection from Repair Success:**
   Explicitly clarify in the paper that the calibrated probability $\hat{p} = P(\text{faulty} \mid Z_{\max})$ denotes the probability that an event is anomalous/faulty. It must **not** be equated with the probability of successful automated repair, which is conditional on fault class remediability (schema drift is remediable; corrupted or missing data is not).

### 6.2 Target Validation Metrics

The calibration performance should be evaluated on the held-out test set (Slice C) using:

1. **Expected Calibration Error (ECE):**
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
   Evaluate with $M = 10$ and $M = 15$ equal-frequency or equal-width bins.
2. **Maximum Calibration Error (MCE):**
   $$\text{MCE} = \max_{m=1..M} |\text{acc}(B_m) - \text{conf}(B_m)|$$
3. **Brier Score:**
   $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (\hat{p}_i - y_i)^2$$
   Compare uncalibrated risk score vs. Platt-calibrated probability.
4. **Negative Log-Likelihood (NLL / Log Loss):**
   $$\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln(1 - \hat{p}_i) \right]$$
5. **Reliability Diagram:** Plot observed empirical frequency against mean predicted probability across bins, including confidence intervals and histogram of bin counts.
6. **Discrimination Metrics:** Report Area Under the ROC Curve (AUROC), Area Under the Precision-Recall Curve (AUPRC), and Precision/Recall/F1 at decision threshold $\tau = 0.5$ (and tuned optimal threshold).

---

## 7. Implementation Gaps & Minimal Required Changes

The following table details the exact engineering gaps and smallest practical modifications required to execute the calibration experiment:

| # | Gap Component | Current Code Limitation | Smallest Practical Change Needed | Target File |
|---|---|---|---|---|
| **1** | **Full-Stream Score Logging** | `ShieldStreamPipeline.java:367` only emits metric if $risk \ge 0.55$. Normal events and low-risk false negatives produce no telemetry. | Add an evaluation mode or side-output logging `(event_id, z_dist, z_fare, max_z, risk, action)` for every processed event. | `flink-job/.../ShieldStreamPipeline.java` |
| **2** | **Explicit Ground-Truth Logging** | Only injected faults are published to `shield-incidents-v2`. Healthy events have no ground-truth records. | Producer logs an evaluation manifest or emits healthy records to an evaluation topic with `ground_truth: false`. | `producer/producer.py` |
| **3** | **Disjoint Dataset Partitioning** | `producer.py` always reads `--limit 5000` from row 0; all trials reuse identical records. | Add `--offset` and `--trial-id` CLI parameters to `producer.py` so trials stream non-overlapping parquet slices. | `producer/producer.py` |
| **4** | **Calibration Training Script** | No script exists to train calibrators or optimize $(A, B)$. | Create `scripts/calibrate_platt.py` using `scikit-learn`'s `LogisticRegression(solver='lbfgs')` on calibration split. | New file: `scripts/calibrate_platt.py` |
| **5** | **Held-Out Calibration Evaluator** | `evaluate_phase2.py` only computes binary classification counts (TP, FP, FN). | Extend evaluation to compute ECE, MCE, Brier score, and plot reliability diagrams. | New file or `scripts/evaluate_calibration.py` |
| **6** | **Pipeline Decision Policy** | Pipeline gates decisions on static thresholds `risk >= 0.80 && confidence >= 0.80`. | Configure pipeline with calibrated parameters $(A, B)$ or calibrated probability threshold $\tau_{\text{quarantine}}$. | `flink-job/.../ShieldStreamPipeline.java` |

---

## 8. Exact Files Requiring Future Modification

When the user is ready to implement calibration, only the following files will need to be touched:

1. **`flink-job/src/main/java/ai/shieldstream/ShieldStreamPipeline.java`**
   - Emit telemetry for all events in evaluation mode to capture full distribution (including $risk < 0.55$).
   - (Optional) Incorporate Platt scaling parameters $A$ and $B$ into `ConfidenceAwareHealer` so `confidence` reflects true posterior probability $\hat{p} = \sigma(A \cdot Z_{\max} + B)$.
2. **`producer/producer.py`**
   - Add `--offset` and `--trial-id` flags to stream disjoint parquet slices across trials.
   - Output ground-truth manifest of emitted healthy event IDs.
3. **`fault_injector/fault_injector.py`**
   - Add random seed parameter (`--seed`) for reproducibility of `missing_data` and sampling.
4. **`scripts/evaluate_phase2.py` / New Calibration Scripts**
   - Create `scripts/calibrate_platt.py` to fit logistic regression on calibration split.
   - Create `scripts/evaluate_calibration.py` to calculate Brier score, ECE, NLL, and generate reliability diagrams.
5. **`scripts/plot_results.py`**
   - Add plotting function for reliability curves and calibration histograms.

---

## 9. Missing Information & Unresolved Questions

1. **Scope of Calibration Claim:** Does the paper intend to claim calibration only for numerical concept drift, or for the entire hybrid framework? (Calibration is mathematically valid for the continuous Z-score detector, but not for deterministic boolean rules like schema renaming or JSON parsing).
2. **Real-Time vs. Offline Calibration:** Will Platt scaling parameters $(A, B)$ be fitted offline on a calibration batch and loaded into Flink via environment variables, or does the pipeline require an online adaptive calibrator? (Offline batch-fitted Platt scaling is standard and significantly easier to validate).
3. **Operating Decision Threshold:** Once calibrated probabilities $\hat{p} \in [0, 1]$ are obtained, what probability threshold should trigger `ABSTAIN_QUARANTINE`? (e.g., standard Bayes decision $\tau = 0.5$, or a cost-sensitive threshold reflecting false positive penalty in streaming pipelines).

---

## 10. Key Code Excerpts for Reviewer Presentation

These concise code snippets directly demonstrate the existing mechanism to an external auditor or reviewer:

### Excerpt 1: Existing Continuous Risk & Confidence Formula
```java
// File: flink-job/src/main/java/ai/shieldstream/ShieldStreamPipeline.java, Lines 326-342
double distanceZ = stats.zDistance(distance);
double fareZ = stats.zFare(fare);
double maximumZ = Math.max(distanceZ, fareZ);

double risk = sigmoid(maximumZ - 2.5);

double confidence =
        Math.min(
                1.0,
                stats.count / 500.0
        );
```

### Excerpt 2: Threshold Gating and Silenced Normal Telemetry
```java
// File: flink-job/src/main/java/ai/shieldstream/ShieldStreamPipeline.java, Lines 344-380
if (stats.count >= 500
        && risk >= 0.80
        && confidence >= 0.80) {

    context.output(DLQ, raw);
    context.output(METRICS, metric(eventId, "CONCEPT_DRIFT", risk, confidence, "ABSTAIN_QUARANTINE"));
    return;
}

output.collect(raw);
updateStatistics(event);

// NOTE: Events with risk < 0.55 do NOT emit any metric record to Kafka!
if (stats.count >= 500
        && risk >= 0.55) {

    context.output(METRICS, metric(eventId, "DRIFT_WARNING", risk, confidence, "SHADOW_ACCEPT"));
}
```

### Excerpt 3: Welford Online Variance Accumulator
```java
// File: flink-job/src/main/java/ai/shieldstream/ShieldStreamPipeline.java, Lines 425-451
public void add(double distance, double fare) {
    count++;

    double distanceDelta = distance - meanDistance;
    meanDistance += distanceDelta / count;
    m2Distance += distanceDelta * (distance - meanDistance);

    double fareDelta = fare - meanFare;
    meanFare += fareDelta / count;
    m2Fare += fareDelta * (fare - meanFare);
}
```

### Excerpt 4: Independent Ground-Truth Incident Generation
```python
# File: fault_injector/fault_injector.py, Lines 36-43, 83-84
def incident(fault: str, event_id: str, intensity: float) -> dict[str, Any]:
    return {
        "fault": fault,
        "event_id": event_id,
        "intensity": intensity,
        "injected_at": datetime.now(timezone.utc).isoformat(),
        "ground_truth": True,
    }

# Injected truth published to Kafka topic 'shield-incidents-v2'
truth = incident(args.fault, event["event_id"], args.intensity)
producer.produce(incident_topic, key=event["event_id"], value=json.dumps(truth).encode())
```

---
*Report completed on 2026-10-20. All factual claims derived directly from repository source code and output artifacts.*
