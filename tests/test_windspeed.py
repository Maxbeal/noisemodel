from unittest.mock import MagicMock
from windwhisper.windspeed import WindSpeed


def test_wind_speed_initialization():
    # Mock the wind_turbines input
    wind_turbines = {
        "Turbine1": {
            "hub height": 100,
            "position": (47.5, 8.2),  # Mock position (latitude, longitude)
        }
    }

    # Mock wind speed data with appropriate structure
    mock_wind_speed = MagicMock()
    mock_wind_speed.height.max.return_value = 120  # Mock max height
    mock_wind_speed.height.min.return_value = 50  # Mock min height

    # Mock the `sel` and `interp` methods for wind speed interpolation
    mock_wind_speed.sel.return_value = MagicMock(interp=MagicMock(return_value=10.0))

    # Create an actual WindSpeed instance, injecting the mock wind speed
    instance = WindSpeed(wind_turbines, mock_wind_speed)

    # Allow the real `calculate_mean_speed` to run
    instance.calculate_mean_speed()

    # Ensure `mean_wind_speed` is calculated for Turbine1
    assert "mean_wind_speed" in instance.wind_turbines["Turbine1"]
    assert instance.wind_turbines["Turbine1"]["mean_wind_speed"] == 10.0
