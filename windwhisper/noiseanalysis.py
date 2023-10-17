import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .windspeed import WindSpeed
from .windturbines import WindTurbines
from .noisemap import NoiseMap
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict
from haversine import haversine, Unit


class NoiseAnalysis:
    MAX_WIND_SPEED = 30  # Assuming max wind speed is 30 m/s

    def __init__(self, wind_module: WindSpeed, turbine_model: WindTurbines, noise_map: NoiseMap, wind_turbines: List[Dict]):
        self.wind_data = wind_module.dataframes
        self.turbine_model = turbine_model
        self.noise_map = noise_map
        self.turbines = wind_turbines
        self.detailed_prediction_cache = None  # Cache for detailed_no
        self.noise_dataframes_cache = None  # Cache for get_noise_dataframes
        self.wind_module = wind_module

    def calculate_annual_hours_from_dataframe(self) -> Dict[str, Dict[str, float]]:
        results = {}
        bins = [0, 3, 6, 9, 12, 16, self.MAX_WIND_SPEED]
        labels = ['0-3 m/s', '3-6 m/s', '6-9 m/s', '9-12 m/s', '12-16 m/s', '>16 m/s']

        for turbine_name, df in self.wind_data.items():
            df['SpeedRange'] = pd.cut(df['WS10'], bins=bins, labels=labels, right=False)
            hours = df['SpeedRange'].value_counts(normalize=True) * (df.shape[0] * 0.5)
            results[turbine_name] = hours.to_dict()

        return results

    def calculate_perceived_noise(self, turbine: Dict, wind_speed: float) -> float:
        if 3 <= wind_speed <= 25:
            noise_predictions = self.turbine_model.predict_noise_at_wind_speed([turbine], wind_speed)
            return noise_predictions.get(turbine.get('name'), {}).get(f"{wind_speed:.1f}m/s")
        return None


    def _categorize_time_period(self, time: np.datetime64) -> str:
        timestamp = pd.Timestamp(time)
        if 6 <= timestamp.hour < 18:
            return "day"
        elif 18 <= timestamp.hour < 22:
            return "evening"
        else:
            return "night"


    def separate_time_periods(self, df: pd.DataFrame) -> dict:
        categorize_func = np.vectorize(self._categorize_time_period)
        df['Period'] = categorize_func(df['Time'])
        return {period: df[df['Period'] == period] for period in ['day', 'evening', 'night']}

    def calculate_lden(self, day_df, evening_df, night_df) -> float:
        Lday = 10 * np.log10(np.mean(10 ** (day_df['Noise Level (dB)'] / 10)))
        Levening = 10 * np.log10(np.mean(10 ** ((evening_df['Noise Level (dB)'] + 5) / 10)))
        Lnight = 10 * np.log10(np.mean(10 ** ((night_df['Noise Level (dB)'] + 10) / 10)))
        return (12 * Lday + 4 * Levening + 8 * Lnight) / 24

    def calculate_noise_hours_per_period(self) -> Dict[str, Dict[str, Dict[int, float]]]:
        results = {}
        bins = np.arange(0, 30, 0.1)

        for turbine in self.turbines:
            turbine_name = turbine['name']
            df = self.wind_data[turbine_name].copy()
            df['SpeedBin'] = pd.cut(df['WS10'], bins=bins, labels=bins[:-1])
            avg_wind_speeds = df.groupby('SpeedBin')['WS10'].mean()

            period_hours = {'day': {}, 'evening': {}, 'night': {}}
            for speed_bin, avg_wind_speed in avg_wind_speeds.items():
                noise = self.calculate_perceived_noise(turbine, avg_wind_speed)
                if noise is not None:
                    rounded_noise = round(noise)
                    bin_data = df[df['SpeedBin'] == speed_bin]
                    for period, hours_range in {
                        'day': range(7, 19),
                        'evening': range(19, 23),
                        'night': list(range(0, 7)) + [23]
                    }.items():
                        period_count = bin_data.index.hour.isin(hours_range).sum()
                        period_hours[period][rounded_noise] = period_hours[period].get(rounded_noise, 0) + period_count

            results[turbine_name] = period_hours

        return results
            
    def convert_dB_to_W(self, noise_dict: Dict[str, Dict[str, Dict[int, float]]]) -> Dict[str, Dict[str, Dict[int, float]]]:
        """
        Converts the noise levels from dB to W using the calculate_sound_intensity_level function.

        :param noise_dict: A dictionary with noise levels in dB.
        :return: A dictionary with noise levels in W.
        """
        results_W = {}
        for turbine_name, periods in noise_dict.items():
            results_W[turbine_name] = {}
            for period, noise_levels in periods.items():
                results_W[turbine_name][period] = {}
                for dB, hours in noise_levels.items():
                    # Utilisez l'instance noise_map pour appeler la fonction calculate_sound_intensity_level
                    intensity_level = self.noise_map.calculate_sound_intensity_level(dB, 100)  # Assuming a distance of 100 meters for conversion
                    results_W[turbine_name][period][intensity_level] = hours
        return results_W


    def calculate_noise_hours_in_W(self) -> Dict[str, Dict[str, Dict[float, float]]]:
        """
        Calculates the number of hours per year that a certain noise level (in W) is reached for each turbine, 
        separated by day, evening, and night periods.

        :return: A dictionary with turbine names as keys and another dictionary as values. 
                 The inner dictionary has periods (day, evening, night) as keys and another dictionary as values.
                 The innermost dictionary has noise levels (in W, rounded) as keys and the number of hours as values.
        """
        results_dB = {}
        results_W = {}

        for turbine in self.turbines:
            turbine_name = turbine['name']
            df = self.wind_data[turbine_name]

            # Calculate the number of years in the data
            num_years = df.index.year.nunique()

            # Group wind speeds into bins and calculate the average wind speed in each bin
            bins = np.arange(0, 30, 0.1)  # Assuming max wind speed is 30 m/s
            df['SpeedBin'] = pd.cut(df['WS10'], bins=bins, labels=bins[:-1])
            avg_wind_speeds = df.groupby('SpeedBin')['WS10'].mean()

            period_hours_dB = {
                'day': {},
                'evening': {},
                'night': {}
            }

            for speed_bin, avg_wind_speed in avg_wind_speeds.items():
                noise_dB = self.calculate_perceived_noise(turbine, avg_wind_speed)
                if noise_dB is not None:
                    bin_data = df[df['SpeedBin'] == speed_bin]

                    for period, hours_range in {
                        'day': range(7, 19),
                        'evening': range(19, 23),
                        'night': list(range(0, 7)) + [23]
                    }.items():
                        period_count = len(bin_data[bin_data.index.hour.isin(hours_range)])
                        period_hours_dB[period][noise_dB] = period_hours_dB[period].get(noise_dB, 0) + (period_count * 0.5 / num_years)

            results_dB[turbine_name] = period_hours_dB

            # Convert dB to W
            results_W[turbine_name] = {}
            for period, noise_levels in period_hours_dB.items():
                results_W[turbine_name][period] = {}
                for dB, hours in noise_levels.items():
                    intensity_level = self.noise_map.calculate_sound_intensity_level(dB, 100)  # Assuming a distance of 100 meters for conversion
                    rounded_intensity = round(intensity_level, 12)  # Adjust the precision as needed
                    results_W[turbine_name][period][rounded_intensity] = hours

        return results_W

    

    def _process_single_turbine(self, df, turbine, detailed_predictions):
        turbine_name = turbine.get('name', 'Unknown')
        turbine_predictions = detailed_predictions[turbine_name]

        # Extract wind speeds from dataframe
        wind_speeds = df['WS10']

        # Convert wind speeds to string format for dictionary lookup
        wind_speeds_str = wind_speeds.apply(lambda x: f"{x:.1f}m/s")

        # Map wind speeds to their corresponding noise levels using the turbine_predictions dictionary
        noise_levels = wind_speeds_str.map(turbine_predictions)

        # Create a new dataframe
        noise_df = pd.DataFrame({
            'Wind Speed (m/s)': wind_speeds,
            'Noise Level (dB)': noise_levels
        })

        return turbine_name, noise_df

    def get_noise_dataframes(self, wind_turbines: List[Dict]) -> List[Tuple[str, pd.DataFrame]]:
        """
        Generate a list of tuples with turbine name and its corresponding dataframe with wind speeds and noise levels.
        """
        # Check if get_noise_dataframes has already been computed
        if self.noise_dataframes_cache is not None:
            return self.noise_dataframes_cache

        noise_dataframes = []

        # Check if detailed_noise_prediction has already been computed
        if self.detailed_prediction_cache is None:
            self.detailed_prediction_cache = self.turbine_model.detailed_noise_prediction(wind_turbines)

        detailed_predictions = self.detailed_prediction_cache

        # Use ThreadPoolExecutor to process data concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
            results = list(executor.map(self._process_single_turbine, self.wind_data.values(), wind_turbines, [detailed_predictions]*len(wind_turbines)))

        # Wrap the results with tqdm for progress bar
        for result in tqdm(results, total=len(wind_turbines), desc="Processing turbines", ncols=100):
            noise_dataframes.append(result)

        # Cache the results of get_noise_dataframes
        self.noise_dataframes_cache = noise_dataframes

        return noise_dataframes



    def calculate_sound_at_observation_points(self, observation_points: List[Tuple[float, float]]) -> Dict[Tuple[float, float], pd.DataFrame]:
        """
        Calculate the noise levels at specific observation points.
        """
        # Check if get_noise_dataframes has already been computed
        if self.noise_dataframes_cache is None:
            self.noise_dataframes_cache = self.get_noise_dataframes(self.turbines)

        noise_dataframes = self.noise_dataframes_cache

        # Create a dictionary for faster lookup of turbines by name
        turbine_dict = {t['name']: t for t in self.turbines}

        # Pre-process noise_dataframes for faster access
        noise_dict = {turbine: df['Noise Level (dB)'] for turbine, df in noise_dataframes}

        # Use the first dataframe's 'time' index as the reference times
        times = noise_dataframes[0][1].index.tolist()

        results = {}

        # Wrap the observation_points with tqdm for progress bar
        for observation_point in tqdm(observation_points, desc="Processing observation points", ncols=100):
            noise_levels = []

            for time in times:
                total_noise = 0
                for turbine, _ in noise_dataframes:
                    if turbine in noise_dict:
                        turbine_dict[turbine]['noise_level'] = noise_dict[turbine].loc[time]

                total_noise += self.noise_map.superpose_several_wind_turbine_sounds([observation_point])[observation_point]

                # If the total noise is less than 0 dB, set it to 0 dB
                if total_noise < 0:
                    total_noise = 0

                noise_levels.append(total_noise)

            df_result = pd.DataFrame({'Time': times, 'Noise Level (dB)': noise_levels})
            results[observation_point] = df_result

        return results
    
    def calculate_sound_at_observation_points2(self, observation_points: List[Tuple[float, float]]) -> Dict[Tuple[float, float], pd.DataFrame]:
        """
        Calculate the noise levels at specific observation points.
        """
        noise_dataframes = self.get_noise_dataframes(self.turbines)

        # Create a dictionary for faster lookup of turbines by name
        turbine_dict = {t['name']: t for t in self.turbines}

        # Pre-process noise_dataframes for faster access
        noise_dict = {turbine: df['Noise Level (dB)'] for turbine, df in noise_dataframes}

        # Use the first dataframe's 'time' index as the reference times
        times = noise_dataframes[0][1].index.tolist()

        results = {}

        # Wrap the observation_points with tqdm for progress bar
        for observation_point in tqdm(observation_points, desc="Processing observation points", ncols=100):
            noise_levels = []

            for time in times:
                total_noise = 0
                for turbine, _ in noise_dataframes:
                    if turbine in noise_dict:
                        turbine_dict[turbine]['noise_level'] = noise_dict[turbine].loc[time]

                total_noise += self.noise_map.superpose_several_wind_turbine_sounds2([observation_point])[observation_point]

                # If the total noise is less than 0 dB, set it to 0 dB
                if total_noise < 0:
                    total_noise = 0

                noise_levels.append(total_noise)

            df_result = pd.DataFrame({'Time': times, 'Noise Level (dB)': noise_levels})
            results[observation_point] = df_result

        return results
    
    

    def simulate_sound_at_observer(self, observation_points: List[Tuple[float, float]]) -> Dict[float, Dict[Tuple[float, float], float]]:
        """Superposes sounds from multiple wind turbines at specific observation points for various wind speeds."""

        wind_speed_results = {}

        for wind_speed in range(3, 13):  # Iterate over wind speeds from 3 to 12 m/s
            results = {}
            noise_predictions = self.turbine_model.predict_noise_at_wind_speed(self.turbines, wind_speed)

            for observation_point in observation_points:
                total_intensity_level = 0
                for turbine in self.turbines:
                    turbine_name = turbine['name']
                    dBsource = noise_predictions[turbine_name][f'{wind_speed}.0m/s']
                    distance = haversine(observation_point, turbine['position'], unit=Unit.METERS)
                    intensity_level = self.noise_map.calculate_sound_intensity_level(dBsource, distance)
                    total_intensity_level += intensity_level
                dB_total = self.noise_map.convert_intensity_level_into_dB(total_intensity_level)
                results[observation_point] = dB_total

            wind_speed_results[wind_speed] = results

        return wind_speed_results
