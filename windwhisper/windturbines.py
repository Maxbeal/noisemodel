"""
This module contains the WindTurbines class which models a wind turbine and predicts noise levels based on wind speed.
"""

from sys import exit
from typing import List, Tuple, Dict
from pathlib import Path
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from tqdm import tqdm
import skops.io as sio
import xarray as xr
from xarray import DataArray

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
        file_path = Path(DATA_DIR / 'training_data' / 'noise_wind_turbines.csv')

    # check that the file exists
    if not Path(file_path).exists():
        raise FileNotFoundError(f"The file '{file_path}' was not found.")

    # file extension must be .csv
    if Path(file_path).suffix != '.csv':
        raise ValueError(f"The file extension for '{file_path}' must be '.csv'.")

    # Read the CSV file, skipping metadata rows
    df = pd.read_csv(file_path, skiprows=[1, 2])

    # List of noise columns
    noise_cols = [col for col in df.columns if 'Noise' in col]

    # Select only the columns of interest
    cols_to_select = ['Power', 'Diameter', 'hub height [m]'] + noise_cols
    df = df[cols_to_select]

    # Remove rows where all values in noise columns are NaN
    df = df.dropna(subset=noise_cols, how='all')

    # Separate input and output data
    X = df[['Power', 'Diameter', 'hub height [m]']]
    Y = df[noise_cols]

    # Convert non-numeric values in 'X' to NaN
    X = X.apply(pd.to_numeric, errors='coerce')

    # Convert non-numeric values in 'Y' to NaN
    Y = Y.apply(pd.to_numeric, errors='coerce')

    # Use mean imputation for missing input values
    imputer_X = SimpleImputer()
    X = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns)

    # Use kNN imputation for missing output values
    imputer_Y = KNNImputer(n_neighbors=5)
    Y = pd.DataFrame(imputer_Y.fit_transform(Y), columns=Y.columns)

    # Split the data into training and tests sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Create and train the multi-output model
    model = MultiOutputRegressor(HistGradientBoostingRegressor())
    model.fit(X_train.values, Y_train.values)

    # Predict and evaluate the model
    Y_pred = model.predict(X_test.values)
    mse = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')

    for i, column in enumerate(Y.columns):
        print(f"Mean Squared Error for {column}: {mse[i]}")

    # Save the trained model for future use
    sio.dump(obj=(model, noise_cols), file=f"{Path(DATA_DIR / 'default_model')}.skops")

    # print the location of the saved model
    print(f"Trained model saved to {Path(DATA_DIR / 'default_model')}.skops")

    return model, noise_cols


def load_model(filepath=None) -> Tuple[RegressorMixin, List[str]]:
    """
    Loads the trained model from the 'default_model.joblib' file.
    :return: A tuple containing the trained model and the noise columns.
    """

    if filepath is None:
        filepath = Path(DATA_DIR / 'default_model.skops')

    # check that the model exists
    if not Path(filepath).exists():
        raise FileNotFoundError(f"The trained model file {filepath} was not found.")

    model, noise_cols = sio.load(filepath, trusted=True)  # Load both model and noise columns

    return model, noise_cols


def check_wind_turbine_specs(wind_turbines: List[Dict]) -> List[Dict]:
    """
    Check that teh list of dictionaries contain all the needed keys.
    :param wind_turbines: list of dictionaries with turbines specifics
    :return: None or Exception
    """

    mandatory_fields = [
        "name",
        "power",
        "diameter",
        "hub height",
        "position"
    ]

    for turbine in wind_turbines:
        for field in mandatory_fields:
            try:
                turbine[field]
            except KeyError:
                raise KeyError(f"Missing field '{field}' in turbine {turbine['name']}") from None

    # check that `power`, `diameter` and `hub height` are numeric
    # and positive

    for turbine in wind_turbines:
        for field in ["power", "diameter", "hub height"]:
            try:
                turbine[field] = float(turbine[field])
            except ValueError:
                raise ValueError(f"The field '{field}' in turbine {turbine['name']} must be numeric.") from None

            if turbine[field] <= 0:
                raise ValueError(f"The field '{field}' in turbine {turbine['name']} must be positive.")

    # check that the radius is inferior to the hub height
    for turbine in wind_turbines:
        if turbine["diameter"] / 2 > turbine["hub height"]:
            raise ValueError(f"The radius of turbine {turbine['name']} must be inferior to its hub height.")

    # check that `hub height` is inferior to 300 m and that `power`
    # is inferior to 20 MW
    for turbine in wind_turbines:
        if turbine["hub height"] > 300:
            raise ValueError(f"The hub height of turbine {turbine['name']} must be inferior to 300 m.")
        if turbine["power"] > 20000:
            raise ValueError(f"The power of turbine {turbine['name']} must be inferior to 20 MW.")

    # check that the value for `position`is a tuple of two floats
    for turbine in wind_turbines:
        if not isinstance(turbine["position"], tuple):
            raise ValueError(f"The position of turbine {turbine['name']} must be a tuple.")
        if len(turbine["position"]) != 2:
            raise ValueError(f"The position of turbine {turbine['name']} must contain two values.")
        if not all(isinstance(x, float) for x in turbine["position"]):
            raise ValueError(f"The position of turbine {turbine['name']} must contain two floats.")

    return wind_turbines

def check_listeners(listeners):
    """
    Check that the list of dictionaries contain all the needed keys.
    :param listeners: list of dictionaries with listeners specifics
    :return: None or Exception
    """

    mandatory_fields = [
        "name",
        "position"
    ]

    for listener in listeners:
        for field in mandatory_fields:
            try:
                listener[field]
            except KeyError:
                raise KeyError(f"Missing field '{field}' in listener {listener['name']}") from None

    # check that the value for `position`is a tuple of two floats
    for listener in listeners:
        if not isinstance(listener["position"], tuple):
            raise ValueError(f"The position of listener {listener['name']} must be a tuple.")
        if len(listener["position"]) != 2:
            raise ValueError(f"The position of listener {listener['name']} must contain two values.")
        if not all(isinstance(x, float) for x in listener["position"]):
            raise ValueError(f"The position of listener {listener['name']} must contain two floats.")

    return listeners


class WindTurbines:
    """
    This class models a wind turbine and predicts noise levels based on wind speed.
    """

    def __init__(self,
                 wind_turbines: List[Dict],
                 listeners: List[Dict] = None,
                 model_file: str = None,
                 retrain_model: bool = False,
                 dataset_file: str = None
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
        self.wind_turbines = check_wind_turbine_specs(wind_turbines)
        self.listeners = check_listeners(listeners)
        self.na = None

        if retrain_model:
            self.model, self.noise_cols = train_wind_turbine_model(dataset_file)
        else:
            try:
                self.model, self.noise_cols = load_model(model_file)
            except:
                self.model, self.noise_cols = train_wind_turbine_model(dataset_file)

        self.noise = self.predict_noise()

    def predict_noise(self) -> DataArray:
        """Predicts noise levels based on turbine specifications for multiple turbines.

        :return: A DataFrame containing the noise predictions for each turbine.
        """

        # create xarray that stores the parameters for the list
        # of wind turbines passed by the user
        # plus the noise values predicted by the model

        pattern = re.compile(r"[-+]?\d*\.\d+|\d+")
        arr = xr.DataArray(
            np.zeros((len(self.wind_turbines), len(self.noise_cols))),
            dims=("turbine", "wind_speed"),
            coords={
                "turbine": [turbine["name"] for turbine in self.wind_turbines],
                "wind_speed": [float(re.findall(pattern, s)[0]) for s in self.noise_cols],
            },
        )

        arr.coords["wind_speed"].attrs["units"] = "m/s"
        arr.coords["turbine"].attrs["units"] = None

        # convert self.wind_turbines into a numpy array
        # to be able to pass it to the model
        arr_input = np.array(
            [
                [turbine["power"], turbine["diameter"], turbine["hub height"]]
                for turbine in self.wind_turbines
            ]
        )

        # predict the noise values
        arr.values = self.model.predict(arr_input)

        return arr

    def predict_noise_at_wind_speed(self, wind_speed: [float, list, np.array]) -> xr.DataArray:
        """
        Predicts noise levels at a specific wind speed
        for multiple turbines.
        :param wind_speed: Wind speed in m/s.
        :return: A xr.DataArray containing the noise predictions for each turbine.
        """

        if not isinstance(wind_speed, np.ndarray):
            wind_speed = np.array(wind_speed)

        wind_speed = np.clip(wind_speed, 0, 25)

        results = self.noise.interp(
            wind_speed=wind_speed,
            kwargs={"fill_value": "extrapolate"},
        ).clip(0, None)

        results.loc[{"wind_speed": wind_speed < 3}] = 0

        return results




    def plot_noise_curve(self):
        """
        Plots noise levels for all wind speeds between 3 and 12 m/s.
        """

        predictions = self.predict_noise_at_wind_speed(np.arange(3, 12, 0.5))
        df = predictions.to_dataframe("val").unstack()["val"].T

        # Different line styles and markers
        line_styles = ['-', '--', '-.', ':']
        markers = ['o', '^', 's', 'p', '*', '+', 'x', 'D']

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, col in enumerate(df.columns):
            style = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            ax.plot(df.index, df[col], linestyle=style, marker=marker, label=col)

        plt.title('Noise vs Wind Speed')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Noise (dBa)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def fetch_wind_speeds(self, start_year=None, end_year=None, debug=False):
        """
            Fetches wind speeds for a given range of years and stores the data in the 
            WindSpeed instance.

            Args:
                start_year (int, optional): The starting year for data fetching. If not 
                                             provided, defaults to 2016.
                end_year (int, optional): The ending year for data fetching. If not 
                                           provided, defaults to 2017.
                debug (bool, optional): A flag used to control the data loading mechanism in 
                                         the WindSpeed instance. When True, data is loaded 
                                         from a local file, otherwise data is downloaded from 
                                         an external source. Default is False.

            Returns:
                None
            """
        start_year = start_year or 2016
        end_year = end_year or 2017

        self.ws = WindSpeed(
            wind_turbines=self.wind_turbines,
            start_year=start_year,
            end_year=end_year,
            debug=debug,
        )

    def plot_wind_rose(self):
        self.ws.create_wind_roses()

    def fetch_noise_map(self):

        self.noise_map = NoiseMap(
            wind_turbines=self.wind_turbines,
            noise=self.noise,
            listeners=self.listeners,
        )

    def analyze_noise(self):
        self.na = NoiseAnalysis(
            wind_speed=self.ws.wind_speed,
            noise=self.noise,
            noise_map=self.noise_map,
            wind_turbines=self.wind_turbines,
            listeners=self.listeners
        )
