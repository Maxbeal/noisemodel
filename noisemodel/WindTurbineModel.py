from sys import exit
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import pandas as pd
from joblib import dump, load  # Import these libraries to save and load the machine learning model
from sklearn.base import RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


class WindTurbineModel:
    """Handles the training and prediction of a wind turbine noise model.

    Attributes:
        model: The trained model.
        noise_cols: The columns used in the noise prediction.
    """

    def __init__(self, file_path: str = None, retrain: bool = False, dataset_name: str = None):
        try:
            if retrain or file_path:
                self.model, self.noise_cols = self.train_wind_turbine_model(file_path)
            else:
                self.model, self.noise_cols = self.load_model()
            self.results = None
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

    def predict_noise(self, power: float, diameter: float, hub_height: float) -> pd.DataFrame:
        """Predicts noise levels based on turbine specifications.

        :param power: Turbine power in watts.
        :param diameter: Rotor diameter in meters.
        :param hub_height: Hub height in meters.
        :return: A DataFrame containing the noise predictions.
        """

        # Create a DataFrame with user inputs
        user_input = pd.DataFrame({'Power': [power], 'Diameter': [diameter], 'hub height [m]': [hub_height]})

        # Use the trained model to predict noise values
        noise_prediction = self.model.predict(user_input)

        # Create a result DataFrame
        result = pd.DataFrame(noise_prediction, columns=self.noise_cols)

        return result

    def plot_noise_prediction(self, power: float, diameter: float, hub_height: float):
        """Plots the noise prediction for the given turbine specifications.

        :param power: Turbine power in watts.
        :param diameter: Rotor diameter in meters.
        :param hub_height: Hub height in meters.
        """
        # Predicting the noise for the given inputs
        noise_prediction = self.predict_noise(power, diameter, hub_height)

        # Plotting the noise prediction
        plt.figure(figsize=(10, 6))
        plt.plot(range(3, 13), noise_prediction.iloc[0], marker='o')
        plt.title('Noise Prediction')
        plt.xlabel('Wind Speed')
        plt.ylabel('Noise [dB]')
        plt.xticks(range(3, 13), [f"{i}m/s" for i in range(3, 13)])
        plt.grid(True)
        plt.show()

    # self.results = np.array()
    # self.predictive_model

    def predict_noise_at_wind_speed(self, power: float, diameter: float, hub_height: float) -> Dict[str, float]:
        """Predicts noise levels at different wind speeds.

        :param power: Turbine power in watts.
        :param diameter: Rotor diameter in meters.
        :param hub_height: Hub height in meters.
        :return: A dictionary containing noise levels at different wind speeds.
        """
        # Ask the user for the average wind speed
        avg_wind_speed = int(input("Please enter the average wind speed at the location (3m/s to 12m/s): "))

        # Ensure the input is within the expected range
        if avg_wind_speed < 3 or avg_wind_speed > 12:
            print("Invalid wind speed. Please enter a value between 3 and 12 m/s.")
            return

        # Predict the noise for the given inputs
        noise_prediction = self.predict_noise(power, diameter, hub_height)

        # Get the noise at the specific wind speed
        noise_at_wind_speed = noise_prediction.iloc[0, avg_wind_speed - 3]
        print(f"The predicted noise at {avg_wind_speed}m/s wind speed is {noise_at_wind_speed} dB.")

        return noise_at_wind_speed
