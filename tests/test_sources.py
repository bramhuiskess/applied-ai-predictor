"""Data-source adapters: every supported input format must give the same table as the CSV file."""
 
import io
import json
 
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from predictor import bootstrap
from predictor.adapters.sources import SOURCES, adapter_for, read_jsonl_stream, read_table
from predictor.trainer.run import init_use_case, train_use_case

DF = pd.DataFrame({
    "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:10:00", "2026-01-01 00:20:00"],
    "temp_c": [21.5, 22.0, 22.4],
    "vent_on": [0, 1, 1],
    "zone": ["a", "b", "a"],
})
 
 
def write(df, path):
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True)
    elif path.suffix == ".json":
        path.write_text(df.to_json(orient="records"), encoding="utf-8")
    elif path.suffix == ".parquet":
        df.to_parquet(path)
    return path
 
 
@pytest.fixture(params=[".csv", ".json", ".jsonl", ".parquet"])
def file(request, tmp_path):
    if request.param == ".parquet":
        pytest.importorskip("pyarrow")
    return write(DF, tmp_path / f"data{request.param}")
 
 
def test_every_format_gives_the_same_table_as_csv(file):
    table = read_table(adapter_for(file), file)
    pd.testing.assert_frame_equal(table, DF, check_dtype=False)
    assert table["timestamp"].astype(str).iloc[0] == "2026-01-01 00:00:00"  # not silently converted to a datetime
 
 
def test_extension_decides_the_adapter():
    assert adapter_for("a.CSV") == "csv-file"
    assert adapter_for("a.json") == "json-file"
    assert adapter_for("a.ndjson") == "jsonl-file"
    assert set(SOURCES) >= {"csv-file", "json-file", "jsonl-file", "parquet-file"}
 
 
def test_unknown_extension_is_refused_with_a_clear_message():
    with pytest.raises(ValueError, match="cannot guess the format"):
        adapter_for("data.xml")
 
 
def test_json_with_a_records_object_is_accepted(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(json.dumps({"records": DF.to_dict("records")}))
    pd.testing.assert_frame_equal(read_table("json-file", path), DF, check_dtype=False)
 
 
@pytest.mark.parametrize("content", ['{"a": 1}', '[1, 2]', '"text"', ""])
def test_json_with_the_wrong_shape_is_refused(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError):  # JSONDecodeError is a ValueError too
        read_table("json-file", path)
 
 
def test_jsonl_ignores_blank_lines(tmp_path):
    path = tmp_path / "blank.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert read_table("jsonl-file", path)["a"].tolist() == [1, 2]
 
 
@pytest.mark.parametrize("name", ["empty.csv", "empty.json", "empty.jsonl"])
def test_empty_file_is_refused(tmp_path, name):
    path = tmp_path / name
    path.write_text("[]" if name.endswith(".json") else "")
    with pytest.raises(ValueError):
        read_table(adapter_for(path), path)
 
 
def test_corrupt_jsonl_is_refused(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\n{"a": \n')
    with pytest.raises(ValueError):
        read_table("jsonl-file", path)


NESTED = [
    {"timestamp": "2026-01-01 00:00:00", "climate": {"temp_c": 21.5, "hum_pct": 60.0}, "vent_on": 1},
    {"timestamp": "2026-01-01 00:10:00", "climate": {"temp_c": 22.0, "hum_pct": 61.0}, "vent_on": 0},
]
FLAT = pd.DataFrame({
    "timestamp": ["2026-01-01 00:00:00", "2026-01-01 00:10:00"],
    "climate_temp_c": [21.5, 22.0],
    "climate_hum_pct": [60.0, 61.0],
    "vent_on": [1, 0],
})


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_nested_sensor_readings_are_flattened_to_the_flat_table(tmp_path, suffix):
    path = tmp_path / f"sensors{suffix}"
    path.write_text("\n".join(json.dumps(r) for r in NESTED) if suffix == ".jsonl" else json.dumps(NESTED))
    table = read_table(adapter_for(path), path)
    pd.testing.assert_frame_equal(table[FLAT.columns], FLAT, check_dtype=False)


def test_jsonl_can_be_read_from_a_stream_that_is_not_a_file():
    stream = io.StringIO('{"a": 1}\r\n\r\n{"a": 2}\r\n')
    assert read_jsonl_stream(stream)["a"].tolist() == [1, 2]


def test_jsonl_stream_is_read_lazily_line_by_line():
    def feed():
        yield '{"a": 1}\n'
        yield '{"a": 2}\n'

    assert read_jsonl_stream(feed())["a"].tolist() == [1, 2]


def test_jsonl_with_a_byte_order_mark_is_accepted(tmp_path):
    path = tmp_path / "bom.jsonl"
    path.write_bytes(b'\xef\xbb\xbf{"a": 1}\n{"a": 2}\n')
    assert read_table("jsonl-file", path)["a"].tolist() == [1, 2]


def test_jsonl_does_not_convert_text_to_numbers(tmp_path):
    path = tmp_path / "text.jsonl"
    path.write_text('{"a": "1"}\n{"a": "2"}\n')
    assert read_table("jsonl-file", path)["a"].tolist() == ["1", "2"] 


def test_a_bad_jsonl_line_is_reported_with_its_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\nnot json\n{"a": 3}\n')
    with pytest.raises(ValueError, match="line 2"):
        read_table("jsonl-file", path)


@pytest.mark.parametrize("line", ["[1, 2]", '"text"', "42"])
def test_a_jsonl_line_that_is_not_an_object_is_refused(line):
    with pytest.raises(ValueError, match="line 2 is not a JSON object"):
        read_jsonl_stream(io.StringIO('{"a": 1}\n' + line + "\n"))


FEATURES = {"pump_type": "A", "vibration_mm_s": 3.0, "pressure_bar": 6.0, "filter_ok": 1}
 
 
def convert(scratch, suffix):
    """Write the pumps dataset in another format next to the CSV."""
    df = pd.read_csv(scratch / "data" / "pumps.csv")
    path = scratch / "data" / f"pumps{suffix}"
    if suffix == ".json":
        df.to_json(path, orient="records")
    elif suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True)
    elif suffix == ".parquet":
        pytest.importorskip("pyarrow")
        df.to_parquet(path)
    return path
 
 
def train(scratch, use_case_id, data_path):
    path, _ = init_use_case(use_case_id, data_path, scratch / "usecases", target="failure")
    registry = bootstrap.build_registry(scratch / "models")
    return path, train_use_case(path, registry)  # no --data override: the manifest names the adapter
 
 
@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".parquet"])
def test_model_trained_from_another_format_matches_the_csv_model(scratch, suffix):
    _, csv_result = train(scratch, "pumps-csv", scratch / "data" / "pumps.csv")
    manifest_path, other = train(scratch, "pumps-other", convert(scratch, suffix))
 
    adapter = json.loads(manifest_path.read_text())["data_source"]["adapter"]
    assert adapter != "csv-file"  # the manifest really used the new adapter
    assert other["parity"]["passed"] is True
    assert other["metrics"] == pytest.approx(csv_result["metrics"], rel=1e-6)
 
 
@pytest.mark.parametrize("suffix", [".json", ".jsonl", ".parquet"])
def test_model_trained_from_another_format_serves_predictions(scratch, suffix):
    train(scratch, "pumps-other", convert(scratch, suffix))
    client = TestClient(bootstrap.build_app(scratch / "usecases", scratch / "models"))
 
    response = client.post("/models/pumps-other/predict", json={"records": [{"features": FEATURES}]})
    assert response.status_code == 200, response.text