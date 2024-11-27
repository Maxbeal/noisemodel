import pytest
from windwhisper.noisemap import NoiseMap
from unittest.mock import patch
import numpy as np
import xarray as xr


def test_noise_map_initialization():
    wind_turbines = {
        "Turbine1": {
            "position": (47.5, 8.2),
            "noise_vs_wind_speed": xr.DataArray(
                [60, 62], dims=["wind_speed"], coords={"wind_speed": [3, 4]}
            ),  # Mock noise data as xarray
            "mean_wind_speed": xr.DataArray(
                [3.5], dims=["time"]  # Mean wind speed between 3 and 4 m/s
            ),
        }
    }
    listeners = {"Listener1": {"position": (47.5, 8.25)}}

    # Mock sound level and noise map to avoid detailed calculations
    with patch.object(NoiseMap, 'noise_map_at_wind_speeds', return_value=np.array([[55.0]])):
        with patch.object(NoiseMap, 'calculate_sound_level_at_distance', return_value=55.0):
            nm = NoiseMap(wind_turbines, listeners)
            nm.superimpose_wind_turbines_noise()
            assert nm.individual_noise is not None