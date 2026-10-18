# Applied AI prediction backend (prototype)

A simple but scalable backend that takes a tabular dataset, learns its pattern and
predicts an outcome, for **any** use case with numeric, binary or categorical columns.
A new dataset is added with configuration only: the core is never changed.

The prototype implements the design from the research (sub-questions 1 to 4):

| Research | In this prototype |
|---|---|
| SQ1 modular, model-agnostic architecture | Modular monolith with ports and adapters; use cases are `manifest.json` packages; architecture tests keep the core clean |
| SQ2 data formats | JSON for requests/responses, JSONL for batches (streamed in bounded chunks), JSON Schema for validation |
| SQ3 local vs. cloud | One Docker image that runs fully offline (`--network none`) and can be deployed to any cloud unchanged |
| SQ4 model choice | Linear baseline vs. LightGBM per use case, Isolation Forest without labels, all exported to ONNX |

Included use cases:

| Use case | Data | Task | Chosen model |
|---|---|---|---|
| `greenhouse-climate` | synthetic greenhouse climate, 10-minute intervals | regression, forecast 1 h ahead | LightGBM |
| `absence-forecast` | synthetic sickness absence per department/month | regression, strict privacy policy | linear regression (won on validation) |
| `machine-failure` | real [AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i) machine data (CC BY 4.0) | classification | LightGBM |
| `machine-anomaly` | same, without using the labels | anomaly detection | Isolation Forest |

## Quick start

### Docker (recommended, runs offline)

```bash
cd prototype
docker build -t applied-ai-predictor .          # generates data and trains all use cases
docker run --rm --network none applied-ai-predictor   # proof: works without any network
docker run --rm -p 8000:8000 applied-ai-predictor     # reachable on http://localhost:8000
./scripts/demo.sh                                     # stakeholder demo (9 steps)
```

The image is built in two stages. Training happens in the first stage; the serving image
contains **no pandas, scikit-learn or LightGBM**, only the ONNX runtime.

### Local development

```bash
cd prototype
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[train,dev]"
python scripts/make_datasets.py   # writes data/*.csv (downloads AI4I 2020 once)
predictor train                   # trains every use case in usecases/
predictor serve                   # http://127.0.0.1:8000/docs for the interactive API
pytest                            # 52 tests, ~30 s, fully offline
```

## Adding a new dataset (no code)

```bash
predictor init my-use-case --data path/to/dataset.csv --target column_to_predict
# prints a profile of the data, then writes usecases/my-use-case/manifest.json
# review it (policy, units, dropped columns), then:
predictor train my-use-case
predictor serve
```

`init` profiles the data, detects the task (regression, classification, or anomaly
detection when there is no target), proposes dropping identifier columns, and flags
columns that look like personal data. `train` then:

1. applies the manifest (rename/drop columns, policy: forbidden columns stop training);
2. infers each column's type (number, integer, binary 0/1, categorical) and writes a JSON Schema;
3. splits train/validation/test (chronologically when there is a `time_column`, to prevent leakage);
4. trains the baseline and LightGBM, keeps the better one on the validation split;
5. exports the whole pipeline (imputation, scaling and model) to ONNX;
6. **checks parity**: the ONNX model, run through the real serving runner, must match the original;
   it also warns about probable **target leakage** (a feature that on its own determines the target,
   such as AI4I's failure-mode columns, which `init` cannot know are derived from the target);
7. publishes a new version in `models/<use_case>/<version>/` with `descriptor.json`,
   `feature_schema.json`, `model.onnx` and a human-readable `report.md`.

Retraining creates version 2, 3, and so on. A manifest can pin `"model": {"version": "2"}`.

## Architecture

```
predictor/
├── core/            identical for every use case; imports no ML library (enforced by tests)
│   ├── api.py           FastAPI routes
│   ├── service.py       policy check -> validation -> typed matrix -> runner -> response
│   ├── validation.py    JSON Schema validation, finite numbers, out-of-range warnings
│   ├── policy.py        per-use-case data-protection rules
│   ├── manifest.py      strict manifest model, validated at startup
│   ├── ports.py         ModelRunner and ModelRegistry interfaces
│   └── monitoring.py    counters + structured logs (feature values only if allowed)
├── adapters/        technology behind the ports
│   ├── onnx_runner.py   ONNX Runtime (CPU)
│   ├── fs_registry.py   versioned models on the filesystem, atomic publish
│   └── sources.py       csv-file, jsonl-file, parquet-file readers
├── trainer/         profiling, model candidates, ONNX export and parity check
├── bootstrap.py     composition root: the only place that wires core to adapters
└── cli.py           predictor init | train | list | serve
usecases/<id>/manifest.json   configuration only (enforced by tests)
```

Adding a different model technology means writing one `ModelRunner` adapter and
registering it in `bootstrap.RUNNERS`. The core does not change.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + number of ready use cases |
| GET | `/use-cases` | all use cases with status (`ready` / `not_trained`) |
| GET | `/models/{id}` | metadata: inputs, kinds, units, training ranges, metrics, policy, JSON Schema |
| POST | `/models/{id}/predict` | JSON, up to 10,000 records, all-or-nothing |
| POST | `/models/{id}/predict/batch` | JSONL in, JSONL out, unlimited size, errors per line |
| GET | `/metrics` | per use case: requests, predicted/rejected records, warnings, latency |

```bash
curl -X POST localhost:8000/models/greenhouse-climate/predict -H 'content-type: application/json' -d '{
  "records": [{"record_id": "obs-1", "features": {
    "hour_of_day": 13, "outside_temp_c": 9.5, "solar_radiation_wm2": 420, "inside_humidity_pct": 80,
    "co2_ppm": 520, "ventilation_on": 0, "heating_on": 0, "inside_temp_c": 21.4}}]}'
```
```json
{"schema_version": "1.0", "use_case_id": "greenhouse-climate", "model_version": "1",
 "task_type": "regression", "target_name": "inside_temp_c (+6 steps)", "target_unit": "°C",
 "predictions": [{"record_id": "obs-1", "prediction": 21.488338, "warnings": []}]}
```

Classification responses add `probabilities`; anomaly responses return `"anomaly"`/`"normal"`
plus an `anomaly_score` (lower = more unusual). Inputs outside the range seen in training
produce a warning, not an error.

Invalid input gets a `422` that names the field: a missing feature, an unknown field,
the wrong type, a binary value other than 0/1 (booleans are deliberately not accepted),
an unknown category, or NaN/Infinity.

## Manifest reference

```jsonc
{
  "manifest_version": "1.0",
  "use_case_id": "absence-forecast",            // must equal the directory name
  "description": "...",
  "data_source": { "adapter": "csv-file", "path": "data/absence.csv" },
  "training": {
    "target": "absence_pct",                    // omit for anomaly detection
    "task": "auto",                             // auto | regression | classification | anomaly
    "time_column": "period",                    // sort + chronological split; not a feature
    "horizon_steps": null,                      // forecast the target N rows ahead
    "drop_columns": ["department_id"],
    "rename_columns": {},
    "evaluation_label": null,                   // anomaly only: a label used just for scoring
    "anomaly_share": 0.05,
    "test_fraction": 0.2,
    "units": { "absence_pct": "%" }
  },
  "model": { "runner": "onnx", "version": "latest" },
  "policy": {
    "contains_personal_data": true,
    "special_category_data": true,
    "reject_unknown_fields": true,
    "log_feature_values": false,                // must be false for personal data
    "forbidden_fields": ["diagnosis", "illness_cause", "treatment", "employee_id", "bsn"],
    "min_group_size": { "field": "headcount", "min": 5 }
  }
}
```

Unknown keys, typos and inconsistent combinations are rejected at startup, which limits
configuration debt. The policy is enforced twice: training refuses a dataset that contains
a forbidden column, and the API refuses requests that contain one or describe a group
smaller than `min_group_size`.

## Measured results (validation for the portfolio)

Measured on 2026-10-18, AMD Ryzen 7 9700X, Docker image, single uvicorn worker.
Reproduce with `python scripts/benchmark.py`.

**Model quality (test split, from `models/*/1/report.md`)**

| Use case | Baseline | LightGBM / chosen | Reference |
|---|---|---|---|
| greenhouse-climate (MAE °C, 1 h ahead) | 1.077 | **0.762** | persistence forecast 1.040 |
| absence-forecast (MAE %-points) | **0.967** | 0.998 | (synthetic data is nearly linear, so the baseline wins) |
| machine-failure (F1 / ROC AUC) | 0.220 / 0.901 | **0.741 / 0.970** | 3.4% failures |
| machine-anomaly (recall of real failures at 5% flagged) | | 0.262 | without using any labels |

**ONNX parity** (max absolute difference vs. the original model): 1.1e-07 to 1.2e-05 for all
use cases. While building this, the parity check caught a real problem: LightGBM stores
split thresholds as float64 and ONNX as float32, which flipped decisions for some inputs
(differences up to 0.037). The trainer now snaps thresholds to float32
(`make_thresholds_float32_safe`), which makes the two exactly equivalent for the float32
inputs the core sends.

**Performance** (`machine-failure`, LightGBM, 6 features)

| Measurement | Result |
|---|---|
| single record, p50 / p95 / p99 latency | 1.64 / 1.94 / 2.19 ms (HTTP round trip included) |
| one JSON request with 10,000 records (1.8 MB) | 0.44 s, ~22,700 records/s |
| JSONL batch with 20,000 records (3.7 MB) | 1.08 s, ~18,600 records/s |
| training all four use cases | ~2 s |
| container start to first prediction | ~5 s |

**Modularity**: `tests/test_training.py::test_new_dataset_end_to_end_without_code_changes`
takes a dataset the system has never seen, runs `init` and `train`, and serves predictions
with configuration only. All four use cases share the same core, and `usecases/` contains
nothing but `manifest.json` files (enforced by `tests/test_architecture.py`, which also
fails if the core imports an ML library, an adapter or a use case).

## Limitations and next steps

- Tabular data only (no images, free text or audio).
- Time series: forecasting uses the current row's values; lag and rolling features are a
  next step (they need history at prediction time).
- No authentication, rate limiting or multi-tenant isolation: needed before a real deployment.
- Models are loaded at startup; a reload endpoint or file watcher would allow hot updates.
- LightGBM uses fixed, sensible hyperparameters; FLAML tuning could be added in the trainer
  without touching the core.
- Parquet input needs `pip install pyarrow`.
