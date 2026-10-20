# SHIELD-Stream Phase 2

Confidence-aware self-healing research prototype for Kafka--Flink streaming pipelines.

## Capabilities

- schema and data-quality validation
- stateful duplicate detection by `event_id`
- online Welford statistics and risk scoring for concept drift
- confidence-gated automatic schema repair
- abstention and quarantine when repair is unsafe
- clean, DLQ, repair-audit, metric, and ground-truth Kafka topics
- evaluation of precision, recall, F1, repair count, actions, and per-fault recall

## Windows PowerShell

```powershell
docker compose up -d
docker run --rm -v shield-maven-cache:/root/.m2 -v "${PWD}\flink-job:/app" -w /app maven:3.9.9-eclipse-temurin-17 mvn clean package
docker compose up -d --force-recreate topic-init jobmanager taskmanager
docker exec shield-flink-jobmanager flink run -d /opt/flink/usrlib/shield-stream-pipeline-0.2.0.jar
```

Phase 2 begins at the latest `taxi-raw` offset with group `shield-phase2-v1`, so historical Phase 1 data cannot contaminate the new experiment. It writes to:

- `taxi-clean-v2`: accepted or repaired records
- `taxi-dlq-v2`: quarantined and duplicate records
- `shield-repairs`: repair audit records
- `shield-metrics`: fault, risk, confidence, and action decisions

## Evaluation

After the job becomes RUNNING, first stream 5,000 healthy events to warm up the online statistics. Then inject each fault into `taxi-raw` using clean Phase 2 records as the source:

```powershell
python .\producer\producer.py --limit 5000 --rate 200
python .\fault_injector\fault_injector.py --source-topic taxi-clean-v2 --incident-topic shield-incidents-v2 --fault schema_drift --count 100 --intensity 1.0
python .\fault_injector\fault_injector.py --source-topic taxi-clean-v2 --incident-topic shield-incidents-v2 --fault corruption --count 100 --intensity 1.0
python .\fault_injector\fault_injector.py --source-topic taxi-clean-v2 --incident-topic shield-incidents-v2 --fault missing_data --count 100 --intensity 1.0
python .\fault_injector\fault_injector.py --source-topic taxi-clean-v2 --incident-topic shield-incidents-v2 --fault concept_drift --count 100 --intensity 1.0
python .\fault_injector\fault_injector.py --source-topic taxi-clean-v2 --incident-topic shield-incidents-v2 --fault duplicate --count 100 --intensity 1.0
python .\scripts\evaluate_phase2.py
```

The new `-v2` output topics keep Phase 2 results separate from the Phase 1 baseline.
