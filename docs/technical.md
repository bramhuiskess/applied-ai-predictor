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
| SQ4 model choice | Linear baseline vs. LightGBM vs. their average (blend) per use case, Isolation Forest without labels, all exported to ONNX |

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
docker build -t applied-ai-predictor .          # generates data and trains all use cases
docker run --rm --network none applied-ai-predictor   # proof: works without any network
docker run --rm -p 8000:8000 applied-ai-predictor     # dashboard on http://localhost:8000
./scripts/demo.sh                                     # stakeholder demo (9 steps)
```

The image is built in two stages. Training happens in the first stage; the serving image
contains **no pandas, scikit-learn or LightGBM**, only the ONNX runtime.

### Local development

Python 3.11 or newer is required (macOS's built-in 3.9 does not work; `brew install python@3.12`).

```bash
./scripts/setup.sh                # picks Python >= 3.11, creates .venv, installs the project
. .venv/bin/activate              # starts the virtual environment
python scripts/make_datasets.py   # writes data/*.csv (downloads AI4I 2020 once)
python scripts/make_nursery.py && python scripts/make_arable.py   # scenario datasets (KNMI/CBS)
python scripts/make_greenhouse_challenge.py                       # real greenhouse data (WUR, 8 MB)
predictor train                   # trains every use case in usecases/
predictor serve                   # dashboard on http://127.0.0.1:8000, API docs on /docs
pytest                            # 321 tests, ~95 s, fully offline
```

### Technical notebook

[`notebooks/model_internals.ipynb`](../notebooks/model_internals.ipynb) walks through the model side
using the trainer's own code: profiling and the JSON Schema, chronological vs. random splits,
baseline vs. LightGBM (and why F1), the forecast horizon and persistence baseline, Isolation
Forest, the ONNX export with the float32 threshold bug the parity check catches, and leakage
detection. It is saved with its outputs, so it can be read without running it. To run it:

```bash
./scripts/setup.sh train,notebook
python scripts/make_datasets.py
jupyter lab notebooks/model_internals.ipynb
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
   `feature_schema.json`, `model.onnx` and a human-readable `report.md` (and, with joins, `joins.json` and
   `lookups/`).

With `joins` in the manifest, step 0 connects the lookup tables to the base table and checks the keys, see
[Joining datasets](#joining-datasets).

Retraining creates version 2, 3, and so on. A manifest can pin `"model": {"version": "2"}`.

## Joining datasets

A use case can have one base table (`data_source`, one row per thing you predict) and any number of lookup
tables that are connected to it **in the manifest**. This replaces joining in a data script: the manifest
says where each column comes from, and the running service can fill those columns itself.

```jsonc
"data_source": { "adapter": "csv-file", "path": "data/nursery_base.csv" },
"joins": [
  { "name": "compartment",                      // key join: static attributes per key
    "source": { "adapter": "csv-file", "path": "data/compartments.csv" },
    "keys": { "compartment": "compartment" },     // base column -> lookup column
    "aggregate": { "area_m2": { "column": "area_m2", "op": "first" },
                   "lamp_type": { "column": "lamp_type", "op": "first" } } },
  { "name": "weather",                          // time join: aggregate a window after a date
    "source": { "adapter": "csv-file", "path": "data/weather_rotterdam.csv" },
    "time": { "base": "start_date", "lookup": "date" },
    "window": { "start": { "offset": 0 }, "end": { "offset": 10, "unit": "weeks" } },   // half-open [start, end)
    "aggregate": { "outside_temp_avg_c": { "column": "temp_mean_c", "op": "mean", "round": 2 } },
    "fallback": "climatology" }                 // serving only, see below
]
```

- **`keys`** connect by equality (a number and its text form match: `344` = `"344"`). A key-only join needs one lookup
  row per key (op `first`). **`time`** + **`window`** aggregate the lookup rows of the same key (if any) between two dates.
  Use both for "the weather of this province during the growing season".
- **Window edges** are a date plus an offset in `days` or `weeks`. They start from `time.base`, or from any base column:
  `"start": {"column": "season_start"}, "end": {"column": "season_end", "offset": 1}` (an inclusive end date).
  Nothing in the window is tied to the generator that made the data (it need not start on a Monday).
- **`aggregate`** maps each *new* column to a lookup column and an op: `mean sum min max count first last`, with an
  optional `scale` and `round`. Empty values are skipped, but a day only counts as covered when its value is there:
  a window needs `min_coverage` (default 90%) of its days.
- Joins run **before** `rename_columns`, on the raw base column names. Dates that only place a window are not model
  features (they are dropped automatically); key columns stay features unless listed in `drop_columns`.
- Rules checked at startup: unique join names, no new column that is the target, time column, a dropped column or a
  forbidden field, no window that ends before it starts, no join input or new column in `rename_columns` (joins run before
  renaming, the service uses the renamed names), no `horizon_steps` (rows without a match are dropped, which would shift
  the forecast), and **a joined column cannot be a scenario lever** (it is derived, not chosen). Training also refuses a
  new column that already exists in the base table (it would silently replace it).
- Dates must look like `YYYY-MM-DD` (a time part is ignored) and lie in 1900-2200: numbers such as `20240304`, bare years
  and lists are errors, not guesses. A join may be keyed on a column an earlier join created (joins run in order).

### The key check

`predictor train` checks every join first, and `predictor check-joins [use-case ...]` does the same without training.
It fails with a readable message on: missing columns, dates that do not parse, duplicate lookup keys or `(key, date)` rows
(they would multiply rows), and keys that do not match (below `min_match`, default 95% of the rows). Typos come with a
suggestion (`'Friesland' is not in the lookup; did you mean 'Fryslân'?`, or *differs only in case, accents or spaces*).
It warns about windows that are only partly or not at all in the lookup, gaps inside windows, and new columns that are
constant. `predictor check-joins --suggest base.csv lookup.csv` proposes key and date columns when you write a new join.

### At prediction time

The join runs in the same code (`predictor/core/joins.py`, numpy only) when the model is trained and when it is used.
At training the lookup tables are read, joined, and **a snapshot is stored with the model version**
(`joins.json`, `lookups/<name>.npz`), so the service needs no data files and an old version keeps behaving the same.
A caller can then send the *join inputs* instead of the joined columns:

```jsonc
{"features": {"compartment": "K07", "start_date": "2026-10-26", "temp_setpoint_c": 19, "light_hours": 7.5, ...}}
```

- Columns the caller sends win over the lookup (also for "what if the winter is 5 degrees colder"). Join inputs that are
  not model features are removed before the schema check, so `reject_unknown_fields` still holds, and the policy sees the
  request before anything is filled in.
- A window that is not (fully) in the lookup, e.g. a start date next year, is an error unless the join says
  `"fallback": "climatology"`: then the same calendar window is averaged over every year of the lookup in which it is fully
  present, and the response carries a warning (*"outside_temp_avg_c are not observed for ...; filled with the average of the
  same period in 10 other year(s)"*). Training never uses the fallback.
- Scenarios fill the base once; every scenario starts from that row, so a lever is never overwritten by a join.
- `GET /models/{id}` shows per input `derived_by`/`required: false`, and a `joins` list with what each join `accepts`.

Not yet: conditional counts (days above 30 °C), running totals (rain deficit) and calendar-month windows, so the
arable use case still has five weather columns in its base table; a derived `start_week` from `start_date`.

## Architecture

```
predictor/
├── core/            identical for every use case; imports no ML library (enforced by tests)
│   ├── api.py           FastAPI routes
│   ├── dashboard.html   self-contained stakeholder dashboard (served at /)
│   ├── service.py       policy check -> validation -> typed matrix -> runner -> response
│   ├── validation.py    JSON Schema validation, finite numbers, out-of-range warnings
│   ├── joins.py         joining lookup tables (numpy only): the engine used by training and serving
│   ├── join_checks.py   automatic key check and join suggestions
│   ├── join_values.py   strict dates, key text, "did you mean"
│   ├── policy.py        per-use-case data-protection rules
│   ├── manifest.py      strict manifest model, validated at startup
│   ├── ports.py         ModelRunner and ModelRegistry interfaces
│   └── monitoring.py    counters + structured logs (feature values only if allowed)
├── adapters/        technology behind the ports
│   ├── onnx_runner.py   ONNX Runtime (CPU)
│   ├── fs_registry.py   versioned models on the filesystem, atomic publish
│   └── sources.py       csv-file, jsonl-file, parquet-file readers
├── trainer/         profiling, joins (join.py), model candidates, ONNX export and parity check
├── bootstrap.py     composition root: the only place that wires core to adapters
└── cli.py           predictor init | train | check-joins | list | serve
usecases/<id>/manifest.json   configuration only (enforced by tests)
```

Adding a different model technology means writing one `ModelRunner` adapter and
registering it in `bootstrap.RUNNERS`. The core does not change.

## Dashboard

`http://localhost:8000/` opens a dashboard for stakeholders: plain language, few numbers, and
technical terms only as small print. It is one self-contained HTML file in the core (no CDN), so it
also works in the offline container, and it adapts to every use case from the API alone:

- a one-sentence summary per use case (e.g. "of every 10 real machine failures, the model spots about 6")
  and three headline figures;
- predictions vs. what really happened (forecasts), caught / missed / false alarms (yes-no outcomes),
  or "random checks vs. the model" (anomaly detection);
- why this model was chosen (simple vs. advanced model), what matters most, and the privacy rules;
- **Try it yourself**: sliders for every input and an instant answer. The raw request and response
  JSON are under "Technical details", including the refusal when private data is sent;
- **only for use cases with `joins`**: a *Connected data* card (what each table adds, how well it matched, how much of
  the prediction it explains), a *Look up / Enter myself* switch per join in the form (a date picker and dropdowns for the
  join inputs instead of the joined sliders), and a "Looked up from connected data" list with the values the service
  filled in. All of it is built from `GET /models/{id}` (`joins`, `inputs[].derived_by`) and the `filled_in` field of a
  prediction, so a new dataset with joins needs no dashboard code. Use cases without joins look exactly as before.

For the in-depth version of the same story, see the technical notebook below.

The charts use a small sample that `predictor train` stores in `descriptor.json`. For use cases
with personal data only aggregates are stored, unless the manifest sets
`policy.publish_evaluation_rows: true` (e.g. for synthetic or public data). Models trained before this
existed still work; retrain them to see the charts.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + number of ready use cases |
| GET | `/use-cases` | all use cases with status (`ready` / `not_trained`) |
| GET | `/models/{id}` | metadata: inputs, kinds, units, training ranges, metrics, policy, JSON Schema |
| GET | `/models/{id}/evaluation` | model comparison, feature importance and a test-set sample for charts |
| POST | `/models/{id}/scenario/check` | check a scenario request (base + up to 20 scenarios) and return what would be calculated, see [scenarios.md](scenarios.md) |
| POST | `/models/{id}/scenario` | calculate the base and up to 20 scenarios, with the difference in units, %, euros and against the margin of error |
| POST | `/models/{id}/scenario/sweep` | slide one lever over a range and calculate every point |
| POST | `/models/{id}/predict` | JSON, up to 10,000 records, all-or-nothing |
| POST | `/models/{id}/predict/batch` | JSONL in, JSONL out, unlimited size, errors per line |
| GET | `/` | dashboard (redirects to `/dashboard`) |
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
  "joins": [],                                  // optional: connect other tables, see "Joining datasets" below
  "training": {
    "target": "absence_pct",                    // omit for anomaly detection
    "task": "auto",                             // auto | regression | classification | anomaly
    "time_column": "period",                    // sort + chronological split; not a feature
    "horizon_steps": null,                      // forecast the target N rows ahead
    "test_from": null,                          // backtest: rows from this date on are the test set ("2021-01-01")
    "test_group": null,                         // or hold out whole groups as the test set, e.g.
                                                // {"column": "greenhouse", "values": ["Reference 2020"]};
                                                // the model is then chosen by leaving each other group out
                                                // in turn, and the simplest one within 1 standard error wins
    "target_min": null, "target_max": null,     // physical limits of the output (gas >= 0); predictions stay within
    "baseline": null,                           // a column that is a simple guess for the target, e.g.
                                                // {"column": "yield_5y_avg", "label": "same as the last 5 years"}
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
  },
  "scenario": {                                 // optional: inputs a user may change, see docs/scenarios.md
    "levers": {
      "light_hours": { "label": "Lamp hours", "unit": "h/day", "min": 0, "max": 18, "step": 0.5,
                       "effects": { "gas_m3": "decreasing" } }   // only where it always holds
    },
    "outputs": { "gas_m3": { "label": "Gas use", "price_from": "gas_price_eur_m3", "is_cost": true } },
    "case_label": "batches"                     // "based on 14 comparable batches in the history"
                                                // or a fixed "value_eur" per unit (negative for a cost)
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
