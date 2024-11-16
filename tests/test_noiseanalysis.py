from unittest.mock import MagicMock
from windwhisper.noiseanalysis import NoiseAnalysis
import xarray as xr

def test_noise_analysis():
    # Mock the NoiseMap class
    NoiseMap = MagicMock()

    NoiseMap.individual_noise = [
        {
            "turbine_name": "Turbine1",
            "turbine_position": (47.5, 8.2),
            "listener_name": "Listener1",
            "listener_position": (47.5, 8.25),
            "distance": 10.0,
            "intensity_level_dB": xr.DataArray(
                [[65.0, 70.0, 75.0]],
                dims=["wind_speed", "hour"],
                coords={"wind_speed": [5], "hour": [8, 14, 20]},
            ),  # Simulated intensity level data
        }
    ]

    wind_turbines = {
        "Turbine1": {
            "position": (47.5, 8.2),
            "mean_wind_speed": MagicMock(sel=MagicMock(return_value=xr.DataArray([5], dims=["wind_speed"]))),
        }
    }

    listeners = {"Listener1": {"position": (47.5, 8.25)}}

    analysis = NoiseAnalysis(NoiseMap, wind_turbines, listeners)
    assert analysis is not None

