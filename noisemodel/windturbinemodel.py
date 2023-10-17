from sys import exit
from typing import List, Tuple, Dict

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


class WindTurbineModel:
    """
    This class models a wind turbine and predicts noise levels based on wind speed.
    """

    def __init__(self, file_path: str = None, retrain: bool = False, dataset_name: str = None):
        try:
            if retrain or file_path:
                self.model, self.noise_cols = self.train_wind_turbine_model(file_path)
            else:
                self.model, self.noise_cols = self.load_model()
            self.results = None
            self.prediction_cache = {}  # Cache for detailed_noise_prediction
        except FileNotFoundError:
            print("The trained model 'trained_model.joblib' was not found.")
            if dataset_name:
                self.model, self.noise_cols = self.train_wind_turbine_model(dataset_name)
            else:
                print("Exiting the program. Please ensure the trained model is available next time.")
                exit()

    def train_wind_turbine_model(self, file_path: str) -> Tuple[RegressorMixin, List[str]]:
        """Trains the wind turbine model using the given data file.

        :param file_path: Path to the CSV file containing the training data.
        :return: A tuple containing the trained model and the noise columns.
        """
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
        imputer_X = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer_X.fit_transform(X), columns=X.columns)

        # Use kNN imputation for missing output values
        imputer_Y = KNNImputer(n_neighbors=5)
        Y = pd.DataFrame(imputer_Y.fit_transform(Y), columns=Y.columns)

        # Split the data into training and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Create and train the multi-output model
        model = MultiOutputRegressor(HistGradientBoostingRegressor())
        model.fit(X_train, Y_train)

        # Predict and evaluate the model
        Y_pred = model.predict(X_test)
        for i, column in enumerate(Y.columns):
            mse = mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i])
            print(f"Mean Squared Error for {column}: {mse}")

        # Save the trained model for future use
        dump((model, noise_cols), 'trained_model.joblib')  # This line saves the model to a file

        return model, noise_cols

    def load_model(self):
        model, noise_cols = load('trained_model.joblib')  # Load both model and noise columns
        return model, noise_cols

    def predict_noise(self, wind_turbines: List[Dict]) -> pd.DataFrame:
        """Predicts noise levels based on turbine specifications for multiple turbines.

        :param wind_turbines: List of dictionaries containing turbine specifications.
        :return: A DataFrame containing the noise predictions for each turbine.
        """
        
        # Create a list to store individual result DataFrames
        results = []

        for turbine in wind_turbines:
            power = turbine['power']
            diameter = turbine['diameter']
            hub_height = turbine['hub_height']

            # Create a DataFrame with user inputs
            user_input = pd.DataFrame({'Power': [power], 'Diameter': [diameter], 'hub height [m]': [hub_height]})

            # Use the trained model to predict noise values
            noise_prediction = self.model.predict(user_input)

            # Create a result DataFrame for the current turbine
            result = pd.DataFrame(noise_prediction, columns=self.noise_cols)
            result['Turbine Name'] = turbine.get('name', 'Unknown')  # Add turbine name to the result

            # Append the result DataFrame to the results list
            results.append(result)

        # Concatenate all the result DataFrames into a single DataFrame
        results_df = pd.concat(results, ignore_index=True)

        return results_df


    def predict_noise_at_wind_speed(self, wind_turbines: List[Dict], wind_speed: float) -> Dict[str, float]:
        """Predicts noise levels at a specific wind speed for multiple turbines."""

        # Ensure wind_speed is within the valid range
        if wind_speed < 3 or wind_speed > 25:
            raise ValueError("wind_speed must be between 3 and 25 (inclusive).")

        # If wind_speed is greater than 12 m/s, set it to 12 m/s since noise is constant from 12 to 25 m/s
        if wind_speed > 12:
            wind_speed_for_prediction = 12
        else:
            wind_speed_for_prediction = wind_speed

        # Use the predict_noise method to get the noise prediction for the turbines
        noise_predictions_df = self.predict_noise(wind_turbines)

        results = {}
        for index, turbine in enumerate(wind_turbines):
            turbine_name = turbine.get('name', f'Turbine {index + 1}')

            # Extract noise values and corresponding wind speeds
            wind_speeds = np.arange(3, 13)
            noise_values = noise_predictions_df.iloc[index, :-1].values  # Exclude the 'Turbine Name' column

            # Convert noise_values to numeric type
            noise_values_numeric = pd.to_numeric(noise_values, errors='coerce')

            # Interpolate to get the noise value at the desired wind speed
            interpolated_noise = np.interp(wind_speed_for_prediction, wind_speeds, noise_values_numeric)

            results[turbine_name] = {f"{wind_speed:.1f}m/s": interpolated_noise}

        return results



    def detailed_noise_prediction(self, wind_turbines: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Predicts the noise level at various wind speeds for given wind turbines.
        """
        cache_key = tuple((turbine.get('name', 'Unknown'), turbine.get('power'), turbine.get('diameter'), turbine.get('hub_height')) for turbine in wind_turbines)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        results = {}
        for turbine in tqdm(wind_turbines, desc="Predicting noise for turbines", total=len(wind_turbines)):
            turbine_name = turbine.get('name', 'Unknown')
            turbine_noise_values = {}

            for speed in np.arange(3, 12.1, 0.1):
                noise_prediction = self.predict_noise_at_wind_speed([turbine], speed)
                turbine_noise_values[f"{speed:.1f}m/s"] = noise_prediction[turbine_name][f"{speed:.1f}m/s"]

            noise_below_3 = {f"{speed:.1f}m/s": 0.0 for speed in np.arange(0, 3, 0.1)}
            noise_above_12 = {f"{speed:.1f}m/s": turbine_noise_values["12.0m/s"] for speed in np.arange(12.1, 25.1, 0.1)}

            turbine_noise_values = {**noise_below_3, **turbine_noise_values, **noise_above_12}
            results[turbine_name] = turbine_noise_values

        self.prediction_cache[cache_key] = results
        return results

    def plot_noise_vs_wind_speed(self, wind_turbines: List[Dict]):
        """Plots noise levels for all wind speeds between 3 and 12 m/s."""

        predictions = self.detailed_noise_prediction(wind_turbines)

        plt.figure(figsize=(10, 6))
        for turbine_name, noise_values in predictions.items():
            wind_speeds = [float(speed[:-3]) for speed in noise_values.keys() if 3 <= float(speed[:-3]) <= 12]
            noise_levels = [noise_values[f"{speed:.1f}m/s"] for speed in wind_speeds]

            plt.plot(wind_speeds, noise_levels, label=turbine_name)

        plt.title('Noise vs Wind Speed')
        plt.xlabel('Wind Speed (m/s)')
        plt.ylabel('Noise (dB)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

