import numpy as np
import pandas as pd
import pytest
 
from predictor import bootstrap
from predictor.trainer.errors import TrainingError
from predictor.trainer.run import init_use_case, train_use_case
 
 
@pytest.fixture
def pumps(scratch):
    manifest, _ = init_use_case("bad-data", scratch / "data" / "pumps.csv", scratch / "usecases",
                                target="failure")
    registry = bootstrap.build_registry(scratch / "models")
    return manifest, registry, scratch / "data"
 
 
def save(df, folder, name):
    path = folder / name
    df.to_csv(path, index=False)
    return path
 
 
def test_unrelated_dataset_is_refused(pumps):
    manifest, registry, data = pumps
    with pytest.raises(TrainingError, match="not in the dataset"):
        train_use_case(manifest, registry, data / "greenhouse.csv")
 
 
def test_a_file_that_is_not_a_dataset_is_refused(pumps):
    manifest, registry, data = pumps
    path = data / "notes.csv"
    path.write_text("this is not data\njust some text\n", encoding="utf-8")
    with pytest.raises(TrainingError, match="not in the dataset"):
        train_use_case(manifest, registry, path)
 
 
def test_dataset_without_the_target_column_is_refused(pumps):
    manifest, registry, data = pumps
    df = pd.read_csv(data / "pumps.csv").rename(columns={"failure": "outcome"})
    with pytest.raises(TrainingError, match="not in the dataset"):
        train_use_case(manifest, registry, save(df, data, "no_target.csv"))
 
 
def test_too_few_rows_is_refused(pumps):
    manifest, registry, data = pumps
    df = pd.read_csv(data / "pumps.csv").groupby("failure").head(15)
    with pytest.raises(TrainingError, match="at least 50"):
        train_use_case(manifest, registry, save(df, data, "tiny.csv"))
 
 
def test_target_with_a_single_value_is_refused(pumps):
    manifest, registry, data = pumps
    df = pd.read_csv(data / "pumps.csv")
    df["failure"] = 0
    with pytest.raises(TrainingError, match="one class"):
        train_use_case(manifest, registry, save(df, data, "one_class.csv"))
 
 
def test_noise_labels_are_accepted_but_the_metrics_show_the_model_is_useless(pumps):
    manifest, registry, data = pumps
    df = pd.read_csv(data / "pumps.csv")
    df["failure"] = np.random.default_rng(1).integers(0, 2, len(df))
    result = train_use_case(manifest, registry, save(df, data, "noise.csv"))
    assert result["metrics"]["roc_auc"] < 0.65
 
 
def test_text_in_a_numeric_column_silently_drops_that_feature(pumps):
    manifest, registry, data = pumps
    df = pd.read_csv(data / "pumps.csv")
    df["vibration_mm_s"] = df["vibration_mm_s"].astype(object)
    df.loc[0, "vibration_mm_s"] = "sensor error"
    result = train_use_case(manifest, registry, save(df, data, "dirty.csv"))
    assert "vibration_mm_s" not in result["features"]
    assert any("vibration_mm_s" in note and "dropped" in note for note in result["notes"])