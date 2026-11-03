import json
import logging

import pytest
from conftest import ABSENCE, GREENHOUSE, PUMP


def predict(client, use_case, *features, path="predict"):
    records = [{"record_id": f"r{i}", "features": f} for i, f in enumerate(features)]
    return client.post(f"/models/{use_case}/{path}", json={"records": records})


def test_lists_all_use_cases_with_status(client):
    status = {uc["use_case_id"]: uc["status"] for uc in client.get("/use-cases").json()}
    assert status == {
        "absence-forecast": "ready",
        "greenhouse-climate": "ready",
        "pump-anomaly": "ready",
        "pump-failure": "ready",
        "untrained": "not_trained",
    }


def test_metadata_describes_inputs(client):
    body = client.get("/models/greenhouse-climate").json()
    assert body["task_type"] == "regression"
    assert body["target_unit"] == "°C"
    names = [i["name"] for i in body["inputs"]]
    assert names == list(GREENHOUSE)
    assert next(i for i in body["inputs"] if i["name"] == "ventilation_on")["kind"] == "binary"


def test_regression_prediction(client):
    response = predict(client, "greenhouse-climate", GREENHOUSE)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_version"] == "1"
    assert body["predictions"][0]["record_id"] == "r0"
    assert 10 < body["predictions"][0]["prediction"] < 35
    assert body["predictions"][0]["warnings"] == []


def test_out_of_range_input_gives_warning_not_error(client):
    response = predict(client, "greenhouse-climate", {**GREENHOUSE, "outside_temp_c": 45.0})
    assert response.status_code == 200
    assert "outside the training range" in response.json()["predictions"][0]["warnings"][0]


def test_classification_returns_probabilities(client):
    body = predict(client, "pump-failure", PUMP, {**PUMP, "vibration_mm_s": 1.0}).json()
    first, second = body["predictions"]
    assert set(first["probabilities"]) == {"0", "1"}
    assert sum(first["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert first["probabilities"]["1"] > second["probabilities"]["1"]  # more vibration, more risk


def test_anomaly_detection(client):
    body = predict(client, "pump-anomaly", PUMP, {**PUMP, "vibration_mm_s": 12.0, "pressure_bar": 2.0}).json()
    normal, extreme = body["predictions"]
    assert body["task_type"] == "anomaly"
    assert extreme["prediction"] == "anomaly"
    assert extreme["anomaly_score"] < normal["anomaly_score"]


@pytest.mark.parametrize(
    "features, field",
    [
        ({k: v for k, v in GREENHOUSE.items() if k != "co2_ppm"}, "co2_ppm"),  # missing
        ({**GREENHOUSE, "colour": "red"}, "features"),  # unknown field
        ({**GREENHOUSE, "ventilation_on": 2}, "ventilation_on"),  # binary outside {0, 1}
        ({**GREENHOUSE, "ventilation_on": True}, "ventilation_on"),  # booleans are not 0/1
        ({**GREENHOUSE, "outside_temp_c": "warm"}, "outside_temp_c"),  # wrong type
        ({**GREENHOUSE, "outside_temp_c": None}, "outside_temp_c"),  # null
    ],
)
def test_invalid_records_are_rejected_with_the_field(client, features, field):
    response = predict(client, "greenhouse-climate", features)
    assert response.status_code == 422
    assert field in [e["field"] for e in response.json()["errors"]]


def test_nan_is_rejected(client):
    body = json.dumps({"records": [{"features": GREENHOUSE}]}).replace("21.4", "NaN")
    response = client.post("/models/greenhouse-climate/predict", content=body,
                           headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "finite" in response.json()["errors"][0]["message"]


def test_unknown_category_is_rejected(client):
    response = predict(client, "pump-failure", {**PUMP, "pump_type": "Z"})
    assert response.status_code == 422


def test_one_bad_record_rejects_the_whole_json_request(client):
    response = predict(client, "greenhouse-climate", GREENHOUSE, {**GREENHOUSE, "heating_on": 5})
    assert response.status_code == 422
    assert [e["record_id"] for e in response.json()["errors"]] == ["r1"]


def test_policy_rejects_forbidden_fields(client):
    response = predict(client, "absence-forecast", {**ABSENCE, "diagnosis": "flu"})
    assert response.status_code == 422
    error = response.json()["errors"][0]
    assert error["field"] == "diagnosis" and "data-protection policy" in error["message"]


def test_policy_rejects_small_groups(client):
    response = predict(client, "absence-forecast", {**ABSENCE, "headcount": 3})
    assert response.status_code == 422
    assert "re-identification" in response.json()["errors"][0]["message"]


def test_personal_feature_values_are_never_logged(client, caplog):
    caplog.set_level(logging.DEBUG, logger="predictor")
    assert predict(client, "absence-forecast", ABSENCE).status_code == 200
    assert "care" not in caplog.text
    caplog.clear()
    assert predict(client, "greenhouse-climate", GREENHOUSE).status_code == 200
    assert "inside_temp_c" in caplog.text  # allowed by the greenhouse policy


def test_unknown_and_untrained_use_cases(client):
    assert predict(client, "does-not-exist", GREENHOUSE).status_code == 404
    assert predict(client, "untrained", PUMP).status_code == 409
    assert client.get("/models/untrained").status_code == 409


def test_transport_contract_is_enforced(client):
    assert client.post("/models/pump-failure/predict", json={"records": []}).status_code == 422
    assert client.post("/models/pump-failure/predict", json={"rows": [PUMP]}).status_code == 422
    bad_version = {"schema_version": "2.0", "records": [{"features": PUMP}]}
    assert client.post("/models/pump-failure/predict", json=bad_version).status_code == 422


def test_batch_jsonl_streams_results_and_errors_per_line(client):
    lines = [
        json.dumps({"record_id": "a", "features": ABSENCE}),
        "this is not json",
        json.dumps({"record_id": "c", "features": {**ABSENCE, "employee_id": "E12"}}),
        "",
        json.dumps({"features": ABSENCE}),
    ]
    response = client.post("/models/absence-forecast/predict/batch", content="\n".join(lines),
                           headers={"content-type": "application/x-ndjson"})
    assert response.status_code == 200
    out = [json.loads(line) for line in response.text.splitlines()]
    assert [o["record_id"] for o in out] == ["a", "1", "c", "3"]
    assert "prediction" in out[0] and "prediction" in out[3]
    assert "invalid JSON" in out[1]["errors"][0]["message"]
    assert out[2]["errors"][0]["field"] == "employee_id"


def test_batch_handles_more_records_than_one_chunk(client):
    body = "\n".join(json.dumps({"record_id": str(i), "features": PUMP}) for i in range(1234))
    response = client.post("/models/pump-failure/predict/batch", content=body)
    out = [json.loads(line) for line in response.text.splitlines()]
    assert len(out) == 1234 and out[-1]["record_id"] == "1233"
    assert all("prediction" in o for o in out)


def test_dashboard_is_served_without_external_resources(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Prediction Dashboard" in response.text
    assert "https://" not in response.text  # must work in the offline container


def test_evaluation_for_charts(client):
    regression = client.get("/models/greenhouse-climate/evaluation").json()
    sample = regression["sample"]["regression"]
    assert len(sample["actual"]) == len(sample["predicted"]) == len(sample["persistence"]) == len(sample["time"])
    assert set(regression["sample"]["example_input"]) == set(GREENHOUSE)

    matrix = client.get("/models/pump-failure/evaluation").json()["sample"]["confusion_matrix"]
    assert matrix["classes"] == ["0", "1"] and len(matrix["counts"]) == 2

    anomaly = client.get("/models/pump-anomaly/evaluation").json()["sample"]["anomaly"]
    assert len(anomaly["scores"]) == len(anomaly["label"])


def test_evaluation_withholds_rows_for_personal_data(client):
    body = client.get("/models/absence-forecast/evaluation").json()
    assert body["sample"]["samples_withheld"]
    assert "regression" not in body["sample"]
    assert body["metrics"]["mae"] > 0  # aggregates are still shown


def test_metrics_and_health(client):
    predict(client, "pump-failure", PUMP)
    metrics = client.get("/metrics").json()
    assert metrics["pump-failure"]["records_predicted"] >= 1
    assert client.get("/health").json() == {"status": "ok", "use_cases_ready": 4}
