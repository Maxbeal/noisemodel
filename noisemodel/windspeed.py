import requests
import tempfile
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes


class WindModule:
    """
    This class handles the basic functionalities related to wind data and operations.
    """

    def __init__(self, wind_turbines, start_year=2016, end_year=2018):
        print("Initializing WindModule...")
        self.wind_turbines = wind_turbines
        self.start_year = start_year
        self.end_year = end_year
        print(f"Targeting wind turbines from {start_year} to {end_year}.")
        self.dataframes = self.download_data()
        print("Initialization complete.")

    def _download_single_turbine_data(self, turbine):
        print(f"Downloading data for turbine: {turbine['name']}...")
        latitude = turbine["position"][0]
        longitude = turbine["position"][1]
        url = f"https://wps.neweuropeanwindatlas.eu/api/mesoscale-ts/v1/get-data-point?latitude={latitude}&longitude={longitude}&variable=WD10&variable=WS10&dt_start={self.start_year}-01-01T00:00:00&dt_stop={self.end_year}-12-31T23:30:00"
        response = requests.get(url)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            ds = xr.open_dataset(tmp_file_path)
            print(f"Data for turbine {turbine['name']} downloaded successfully.")
            return turbine["name"], ds.to_dataframe()
        else:
            raise Exception(f"Error downloading data for {turbine['name']}. Please check the latitude and longitude.")

    def download_data(self):
        print("Starting concurrent data download for all turbines...")
        dataframes = {}

        # Use ThreadPoolExecutor to download data concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
            futures = [executor.submit(self._download_single_turbine_data, turbine) for turbine in self.wind_turbines]

            # Wait for all futures to complete
            results = [future.result() for future in futures]

            # Process the results in the order they were submitted
            for turbine_name, df in results:
                dataframes[turbine_name] = df

        print("All turbine data downloaded.")
        return dataframes

    def calculate_mean_speed_in_direction(self, direction):
        print(f"Calculating mean wind speed in direction {direction}°...")
        results = {}
        for turbine_name, df in self.dataframes.items():
            mask = (df['WD10'] >= direction) & (df['WD10'] < direction + 1)
            mean_speed_in_direction = df['WS10'][mask].mean()
            results[turbine_name] = mean_speed_in_direction
        print("Calculation complete.")
        return results

    def create_wind_roses(self):
        print("Creating wind roses for each turbine...")
        for turbine_name, df in self.dataframes.items():
            fig = plt.figure(figsize=(8, 8))
            ax = WindroseAxes.from_ax(fig=fig)
            ax.bar(df['WD10'], df['WS10'], normed=True, opening=0.8, edgecolor='white')
            ax.set_legend(title=f"Wind Speed (m/s) for {turbine_name}")
            plt.show()
        print("Wind roses created.")

    def calculate_statistics(self):
        print("Calculating statistics for each turbine...")
        results = {}
        num_years = self.end_year - self.start_year + 1
        speed_ranges = [(i, i + 1) for i in range(3, 21)]
        orientation_ranges = [(i * 30, (i + 1) * 30) for i in range(12)]

        for turbine_name, df in self.dataframes.items():
            table = pd.DataFrame(index=[f"{s[0]}-{s[1]} m/s" for s in speed_ranges],
                                 columns=[f"{o[0]}-{o[1]}°" for o in orientation_ranges])
            for speed_range in speed_ranges:
                for orientation_range in orientation_ranges:
                    mask = (df['WD10'] >= orientation_range[0]) & (df['WD10'] < orientation_range[1]) & (df['WS10'] >= speed_range[0]) & (df['WS10'] < speed_range[1])
                    hours = mask.sum() / 2
                    annual_hours = hours / num_years
                    table.loc[f"{speed_range[0]}-{speed_range[1]} m/s", f"{orientation_range[0]}-{orientation_range[1]}°"] = annual_hours
            mean_wind_speed = df['WS10'].mean()
            results[turbine_name] = {"table": table, "mean_wind_speed": mean_wind_speed}

        print("Statistics calculated.")
        return results
