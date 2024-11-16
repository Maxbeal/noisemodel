import pytest
from unittest.mock import patch, MagicMock
from windwhisper.windturbines import WindTurbines, check_wind_turbine_specs, train_wind_turbine_model
import pandas as pd

def test_check_wind_turbine_specs_valid():
    wind_turbines = {
        "Turbine1": {
            "power": 3000,
            "diameter": 100,
            "hub height": 120,
            "position": (47.5, 8.2)
        }
    }
    assert check_wind_turbine_specs(wind_turbines) == wind_turbines


def test_check_wind_turbine_specs_invalid():
    wind_turbines = {
        "Turbine1": {
            "power": -3000,
            "diameter": 100,
            "hub height": 120,
            "position": (47.5, 8.2)
        }
    }
    with pytest.raises(ValueError):
        check_wind_turbine_specs(wind_turbines)


@patch('windwhisper.windturbines.sio.dump')
@patch('windwhisper.windturbines.pd.read_csv')
def test_train_wind_turbine_model(mock_read_csv, mock_dump):
    # Mocking realistic data
    mock_data = {
        "Power": [100] * 100,
        "Diameter": [80] * 100,
        "hub height [m]": [100] * 100,
        "Noise_3": [60] * 100,
        "Noise_4": [62] * 100,
    }
    mock_df = pd.DataFrame(mock_data)
    mock_read_csv.return_value = mock_df

    model, noise_cols = train_wind_turbine_model()

    assert model is not None
    assert noise_cols == ["Noise_3", "Noise_4"]

