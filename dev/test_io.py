from windwhisper.windturbines import WindTurbines
from pathlib import Path

FIXTURE_DIR = str(Path(__file__).parent / 'fixtures')


def test_default_model_loading():
    wt = WindTurbines()
    assert wt.model is not None
    assert wt.noise_cols is not None


def test_custom_model_loading():
    wt = WindTurbines(model_file=f"{FIXTURE_DIR}/some_other_model.skops")
    assert wt.model is not None
    assert wt.noise_cols is not None


def retrain_model():
    wt = WindTurbines(retrain_model=True)
    assert wt.model is not None
    assert wt.noise_cols is not None


def retrain_model_with_new_dataset():
    wt = WindTurbines(dataset_file=f"{FIXTURE_DIR}/some_other_dataset.csv")
    assert wt.model is not None
    assert wt.noise_cols is not None
