"""
This module contains the WindTurbines class which models a wind turbine and predicts noise levels based on wind speed.
"""

from sys import exit
from typing import List, Tuple, Dict
from pathlib import Path
import re
import uuid

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
import skops.io as sio
import xarray as xr
from xarray import DataArray
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from scipy.stats import pearsonr
import osmnx as ox
from osmnx._errors import InsufficientResponseError


from . import DATA_DIR
from .windspeed import WindSpeed
from .noisemap import NoiseMap
from .noiseanalysis import NoiseAnalysis


def train_wind_turbine_model(file_path: str = None) -> Tuple[RegressorMixin, List[str]]:
    """Trains the wind turbine model using the given data file.

    :param file_path: Path to the CSV file containing the training data.
    :return: A tuple containing the trained model and the noise columns.
    """

    if file_path is None:
        file_path = Path(DATA_DIR / "training_data" / "noise_wind_turbines.csv")

    # Check that the file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    # File extension must be .csv
    if Path(file_path).suffix != ".csv":
        raise ValueError(f"The file extension for '{file_path}' must be '.csv'.")

    # Read the CSV file, skipping metadata rows
    df = pd.read_csv(file_path, skiprows=[1, 2])

    # List of noise columns
    noise_cols = [col for col in df.columns if "Noise" in col]

    # Select only the columns of interest
    cols_to_select = ["Power", "Diameter", "hub height [m]"] + noise_cols
    df = df[cols_to_select]

    # Remove rows where all values in noise columns are NaN
    df = df.dropna(subset=noise_cols, how="all")

    # Separate input and output data
    X = df[["Power", "Diameter", "hub height [m]"]]
    Y = df[noise_cols]
    print("Number of observations in whole set:", Y.shape[0])

    # Convert non-numeric values in 'X' to NaN
    X = X.apply(pd.to_numeric, errors="coerce")

    # Convert non-numeric values in 'Y' to NaN
    Y = Y.apply(pd.to_numeric, errors="coerce")

    # Use mean imputation for missing input values
    imputer_X = SimpleImputer()
    X = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns)

    # Use kNN imputation for missing output values
    imputer_Y = KNNImputer(n_neighbors=5)
    Y = pd.DataFrame(imputer_Y.fit_transform(Y), columns=Y.columns)

    # Split the data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )
    print("Number of observations in test set:", X_test.shape[0])

    # Create and train the multi-output model
    model = MultiOutputRegressor(HistGradientBoostingRegressor())
    model.fit(X_train.values, Y_train.values)

    # Predict and evaluate the model
    Y_pred = model.predict(X_test.values)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(Y_test, Y_pred, multioutput="raw_values")
    print("Mean Squared Error (MSE):", mse)

    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print("Root Mean Squared Error (RMSE):", rmse)

    # Save the trained model for future use
    sio.dump(obj=(model, noise_cols), file=f"{Path(DATA_DIR / 'default_model')}.skops")

    # Print the location of the saved model
    print(f"Trained model saved to {Path(DATA_DIR / 'default_model')}.skops")

    return model, noise_cols


def load_model(filepath=None) -> Tuple[RegressorMixin, List[str]]:
    """
    Loads the trained model from the 'default_model.joblib' file.
    :return: A tuple containing the trained model and the noise columns.
    """

    if filepath is None:
        filepath = Path(DATA_DIR / "default_model.skops")

    # check that the model exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"The trained model file {filepath} was not found.")

    model, noise_cols = sio.load(
        filepath, trusted=True
    )  # Load both model and noise columns

    return model, noise_cols


def check_wind_turbine_specs(wind_turbines: dict) -> dict:
    """
    Check that the dictionary contain all the needed keys.
    :param wind_turbines: list of dictionaries with turbines specifics
    :return: None or Exception
    """

    mandatory_fields = ["power", "diameter", "hub height", "position"]

    for turbine, specs in wind_turbines.items():
        if not all(
            x in specs for x in mandatory_fields
        ):  # check that all mandatory fields are present
            raise KeyError(f"Missing mandatory field(s) in turbine {turbine}.")

    # check that `power`, `diameter` and `hub height` are numeric
    # and positive

    for turbine, specs in wind_turbines.items():
        for field in ["power", "diameter", "hub height"]:
            try:
                specs[field] = float(specs[field])
            except ValueError:
                raise ValueError(
                    f"The field '{field}' in turbine {turbine} must be numeric."
                ) from None

            if specs[field] <= 0:
                raise ValueError(
                    f"The field '{field}' in turbine {turbine} must be positive."
                )

    # check that the radius is inferior to the hub height
    for turbine, specs in wind_turbines.items():
        if specs["diameter"] / 2 > specs["hub height"]:
            raise ValueError(
                f"The radius of turbine {turbine} must be inferior to its hub height."
            )

    # check that `hub height` is inferior to 300 m and that `power`
    # is inferior to 20 MW
    for turbine, specs in wind_turbines.items():
        if specs["hub height"] > 300:
            raise ValueError(
                f"The hub height of turbine {turbine} must be inferior to 300 m."
            )
        if specs["power"] > 20000:
            raise ValueError(
                f"The power of turbine {turbine} must be inferior to 20 MW."
            )

    # check that the value for `position`is a tuple of two floats
    for turbine, specs in wind_turbines.items():
        if not isinstance(specs["position"], tuple):
            raise ValueError(f"The position of turbine {turbine} must be a tuple.")
        if len(specs["position"]) != 2:
            raise ValueError(
                f"The position of turbine {turbine} must contain two values."
            )
        if not all(isinstance(x, float) for x in specs["position"]):
            raise ValueError(
                f"The position of turbine {turbine} must contain two floats."
            )

    return wind_turbines


def check_listeners(listeners):
    """
    Check that the list of dictionaries contain all the needed keys.
    :param listeners: list of dictionaries with listeners specifics
    :return: None or Exception
    """

    mandatory_fields = [
        "position",
    ]

    for listener, specs in listeners.items():
        if not all(x in specs for x in mandatory_fields):
            raise KeyError(f"Missing mandatory field(s) in listener {listener}.")

        # check that the value for `position`is a tuple of two floats
        if not isinstance(specs["position"], tuple):
            raise ValueError(f"The position of listener {listener} must be a tuple.")
        if len(specs["position"]) != 2:
            raise ValueError(
                f"The position of listener {listener} must contain two values."
            )
        if not all(isinstance(x, float) for x in specs["position"]):
            raise ValueError(
                f"The position of listener {listener} must contain two floats."
            )

    return listeners


class WindTurbines:
    """
    This class models a wind turbine and predicts noise levels based on wind speed.
    """

    def __init__(
        self,
        wind_turbines: dict,
        listeners: dict = None,
        model_file: str = None,
        retrain_model: bool = False,
        dataset_file: str = None,
        wind_speed_data: xr.DataArray | str = None,
        radius_threshold: float = 300.0,
    ):
        """
        Initializes the WindTurbines object.
        :param model_file: if specified, another model than the default one is used.
        :type model_file: str
        :param dataset_file: if specified, the model is retrained using the given dataset.
        :type dataset_file: str
        """

        self.noise_map = None
        self.ws = None
        self.noise_analysis = None
        self.wind_turbines = check_wind_turbine_specs(wind_turbines)
        if listeners is not None:
            self.listeners = check_listeners(listeners)
        else:
            self.listeners = self.find_affected_buildings_from_radius(
                radius=radius_threshold
            )
        self.fetch_wind_speeds(wind_speed_data)

        if retrain_model:
            print("Retraining the model...")
            self.model, self.noise_cols = train_wind_turbine_model(dataset_file)
        else:
            try:
                self.model, self.noise_cols = load_model(model_file)
            except FileNotFoundError:
                self.model, self.noise_cols = train_wind_turbine_model(dataset_file)

        self.fetch_noise_level_vs_wind_speed()
        self.fetch_noise_map()

    def fetch_noise_level_vs_wind_speed(self):
        """
        Predicts noise levels based on turbine specifications for
        multiple turbines.

        :return: A DataFrame containing the noise predictions
        for each turbine.
        """

        # create xarray that stores the parameters for the list
        # of wind turbines passed by the user
        # plus the noise values predicted by the model

        pattern = re.compile(r"[-+]?\d*\.\d+|\d+")
        arr = xr.DataArray(
            np.zeros((len(self.wind_turbines), len(self.noise_cols))),
            dims=("turbine", "wind_speed"),
            coords={
                "turbine": list(self.wind_turbines.keys()),
                "wind_speed": [
                    float(re.findall(pattern, s)[0]) for s in self.noise_cols
                ],
            },
        )

        arr.coords["wind_speed"].attrs["units"] = "m/s"
        arr.coords["turbine"].attrs["units"] = None

        # convert self.wind_turbines into a numpy array
        # to be able to pass it to the model
        arr_input = np.array(
            [
                [specs["power"], specs["diameter"], specs["hub height"]]
                for turbine, specs in self.wind_turbines.items()
            ]
        )

        # predict the noise values
        arr.values = self.model.predict(arr_input)
        arr.loc[
            dict(wind_speed=arr.wind_speed < 3)
        ] = 0  # set noise to 0 for wind speeds < 3 m/s

        for turbine, specs in self.wind_turbines.items():
            specs["noise_vs_wind_speed"] = arr.loc[dict(turbine=turbine)]

    def plot_noise_curve(self):
        """
        Plots noise levels for all wind speeds between 3 and 12 m/s.
        """

        # Different line styles and markers
        line_styles = ["-", "--", "-.", ":"]
        markers = ["o", "^", "s", "p", "*", "+", "x", "D"]

        fig, ax = plt.subplots(figsize=(10, 6))

        i = 0
        for turbine, specs in self.wind_turbines.items():
            style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            ax.plot(
                specs["noise_vs_wind_speed"],
                linestyle=style,
                marker=marker,
                label=turbine,
            )
            i += 1

        plt.title("Noise vs Wind Speed")
        plt.xlabel("Wind Speed (m/s)")
        plt.ylabel("Noise (dBa)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def fetch_wind_speeds(self, wind_speed_data: xr.DataArray | str = None):
        """
        Fetches the wind speed data. Either the wind speed data is provided
        as an argument to the constructor, or it is loaded from
        tests/fixtures/era5_mean_2013-2022_month_by_hour.nc
        or, as a last resort, it is downloaded from the internet.

        :return: Updated wind_turbines with wind speed data.
        """

        self.wind_turbines = WindSpeed(
            wind_turbines=self.wind_turbines,
            wind_speed_data=wind_speed_data,
        ).wind_turbines

    def fetch_noise_map(self):
        self.noise_map = NoiseMap(
            wind_turbines=self.wind_turbines,
            listeners=self.listeners,
        )

    def analyze_noise(self):
        self.noise_analysis = NoiseAnalysis(
            noise_map=self.noise_map,
            wind_turbines=self.wind_turbines,
            listeners=self.listeners,
        )
        self.wind_turbines = self.noise_analysis.wind_turbines
        self.listeners = self.noise_analysis.listeners

    def _generate_power_curves(self):
        """
        Generates the power curves for each wind turbine.
        """
        power_curves = {}
        ws = np.arange(0, 25, 0.01)  # Wind speeds from 0 to 25 m/s

        for turbine in self.wind_turbines:
            gen_wt = GenericWindTurbine(
                name=turbine["name"],
                diameter=turbine["diameter"],
                hub_height=turbine["hub height"],
                power_norm=turbine["power"],
            )
            power_curves[turbine["name"]] = gen_wt.power(ws) * 1e-3

        return power_curves

    def plot_power_curves(self):
        """
        Plots the power curves for each wind turbine in small figures.
        """
        n_turbines = len(self.power_curves)
        ws = np.arange(0, 25, 0.01)  # Wind speeds from 0 to 25 m/s

        fig, axs = plt.subplots(1, n_turbines, figsize=(n_turbines * 5, 4))

        if n_turbines == 1:
            axs = [axs]

        for ax, (name, power_curve) in zip(axs, self.power_curves.items()):
            ax.plot(ws, power_curve, label=name)
            ax.set_title(name)
            ax.set_xlabel("Wind Speed [m/s]")
            ax.set_ylabel("Power [kW]")
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def find_affected_buildings_from_radius(self, radius: float) -> dict:
        """
        Finds the buildings located within the given radius of the wind turbines.
        :param radius: The radius in meters.
        :return: A dictionary (self.listeners) containing the affected buildings,
        with their locations (using osmnx).
        """

        building_locations = {}

        for turbine, specs in self.wind_turbines.items():
            location = specs["position"]

            try:
                a = ox.features.features_from_point(
                    location, tags={"building": True}, dist=float(radius)
                )

                # iterate over "geometry"
                # if it's a polygon, find its centroid
                # if it's a point, use directly

                for b, building in enumerate(a["geometry"]):
                    building_locations[f"Building {str(uuid.uuid4())}"] = {
                        "position": (
                            building.centroid.y,
                            building.centroid.x,
                        )
                    }

            except InsufficientResponseError:
                pass

        return building_locations
