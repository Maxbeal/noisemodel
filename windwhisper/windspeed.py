import requests
from requests.exceptions import Timeout
import tempfile
import xarray as xr
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import numpy as np
from typing import Dict
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

# path to tests/fixtures folder, with test being in the root folder
FIXTURE_DIR = str(Path(__file__).parent.parent / 'tests' / 'fixtures')

class WindSpeed:
    """
    This class handles the basic functionalities related to wind data and operations.
    """

    def __init__(self, wind_turbines, start_year=2016, end_year=2018, debug=False):
        self.wind_turbines = wind_turbines
        self.start_year = start_year
        self.end_year = end_year

        if debug is True:
            # we unpickle the data to save time
            self.wind_speed = xr.open_dataset(f"{FIXTURE_DIR}/wind_speed_dummy_data.nc")

        else:
            self.wind_speed = self.download_data()
        self.mean_directional_wind_speed = self.calculate_mean_speed()
        self.hours_of_operation = self.calculate_operation_time()

    def _download_single_turbine_data(self, turbine):

        latitude = turbine["position"][0]
        longitude = turbine["position"][1]
        url = (f"https://wps.neweuropeanwindatlas.eu/api/mesoscale-ts/"
               f"v1/get-data-point?latitude={latitude}&longitude={longitude}"
               f"&variable=WD10&variable=WS10&dt_start={self.start_year}-"
               f"01-01T00:00:00&dt_stop={self.end_year}-12-31T23:30:00")

        try:
            response = requests.get(url, timeout=20)
        except Timeout as e:
            raise Timeout(f"The request to wps.neweuropeanwindatlas.eu has timed out.") from None

        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            ds = xr.open_dataset(tmp_file_path)
            return ds[["WD10", "WS10"]]
        else:
            raise Exception(f"Error downloading data for {turbine['name']}. Please check the latitude and longitude.")

    def download_data(self) -> xr.DataArray:
        print("Starting concurrent data download for all turbines...")
        dataframes = {}

        # Use ThreadPoolExecutor to download data concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:  # Adjust max_workers as needed
            futures = [executor.submit(self._download_single_turbine_data, turbine) for turbine in self.wind_turbines]

            # Wait for all futures to complete
            results = [future.result() for future in futures]

            # store the results in a xr.DataArray
            # with dimensions turbine and time
            arr = xr.concat(results, dim="turbine")
            arr = arr.assign_coords(turbine=[turbine["name"] for turbine in self.wind_turbines])

        # remove all coordinates except "turbine" and "time"
        arr = arr.reset_coords(drop=True)
        print("Done.")

        return arr

    def calculate_mean_speed(self) -> xr.DataArray:
        """
        Calculate the mean wind speed for each turbine for each direction.
        :return: a xr.DataArray with dimensions turbine and direction
        """

        # wind_speed_data is a xr.DataArray with dimensions turbine and time
        # and variables WD10 (wind direction) and WS10 (wind speed)
        # We want to calculate the mean wind speed for each turbine for each direction
        # We can do this by grouping the data by turbine and direction
        # with 12 categories (one for each 30°)

        result = self.wind_speed.groupby_bins(
            "WD10",
            bins=range(0, 361, 30),
        ).mean()
        # drop "WD10" coordinate
        result = result.reset_coords(drop=True)
        # rename "WS10" coordinate to "mean_wind_speed"
        result = result.rename({"WS10": "mean_wind_speed"})

        return result.to_array()

    def calculate_operation_time(self) -> xr.DataArray:
        """
        Shows hte numbers of hours of operation per direction and wind speed
        intervals.
        :return: array with `turbine`, `WS10` and `WD10` dimensions.
        """

        # Convert to DataFrame
        df = self.wind_speed.to_array().to_dataframe("val").unstack()["val"].T

        wd_bins = np.arange(0, 361, 30)  # Wind direction bins of 30 degrees
        ws_bins = np.arange(0, df.loc[:, ('WS10', slice(None))].max().max() + 1, 1)  # Wind speed bins of 1 km/h

        # Separate the DataFrame into two DataFrames for WD10 and WS10
        df_wd = df['WD10']
        df_ws = df['WS10']

        # Initialize an empty dictionary to hold the results
        result_tables = {}

        # Get the unique turbine names
        turbines = df.columns.get_level_values('turbine').unique()

        # Iterate over each turbine
        for turbine in turbines:
            # Create binned data for wind direction and wind speed
            wd_binned = pd.cut(df_wd[turbine], bins=wd_bins,
                               labels=[f'{left}-{right}' for left, right in zip(wd_bins[:-1], wd_bins[1:])],
                               right=False)
            ws_binned = pd.cut(df_ws[turbine], bins=ws_bins,
                               labels=[f'{left}-{right}' for left, right in zip(ws_bins[:-1], ws_bins[1:])],
                               right=False)

            # Merge the binned data into a single DataFrame
            binned_df = pd.DataFrame({'WD_bin': wd_binned, 'WS_bin': ws_binned})

            # Create a crosstab table to count the number of hours per bin
            crosstab = pd.crosstab(binned_df['WD_bin'], binned_df['WS_bin'])

            # Store the crosstab table in the result dictionary
            result_tables[turbine] = crosstab.T

        data_arrays = [
            xr.DataArray(table, dims=['wind_direction_bin', 'wind_speed_bin'], name=turbine)
            for turbine, table in result_tables.items()
        ]

        # Concatenate these DataArrays along a new 'turbine' dimension
        combined_da = xr.concat(data_arrays, dim=pd.Index(result_tables.keys(), name='turbine'))

        return combined_da.fillna(0)

    def _create_wind_rose(self, turbine, ax):
        data = self.wind_speed.sel(turbine=turbine["name"])

        divider = make_axes_locatable(ax)
        ax_windrose = divider.append_axes("right", size="100%", pad=0.0, axes_class=WindroseAxes)

        ax_windrose.bar(data["WD10"], data["WS10"], normed=True, opening=0.8, edgecolor='white',
                        bins=np.arange(0, 25, 5))
        ax_windrose.set_legend(title=turbine["name"], loc="upper left", bbox_to_anchor=(1.2, 1.0))
        return ax_windrose

    def create_wind_roses(self):
        """
        Create a matplotlib.pyplot figure,
        with as many subplots as there are turbines.
        Then, create a wind rose for each subplot.
        :return: None
        """

        # We want two plots per row
        # and as many rows as there are turbines divide by 2

        n_rows = int(np.ceil(len(self.wind_turbines) / 2))
        # Create the figure and axes
        fig, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(10, 5 * n_rows))

        for i, turbine in enumerate(self.wind_turbines):
            # Convert 1D index to 2D index
            row, col = np.unravel_index(i, (n_rows, 2))
            # Hide the original axes
            axs[row, col].axis('off')
            # Create a wind rose for this turbine
            self._create_wind_rose(turbine, axs[row, col])

        # remove empty subplots
        for i in range(len(self.wind_turbines), n_rows * 2):
            row, col = np.unravel_index(i, (n_rows, 2))
            fig.delaxes(axs[row, col])

        plt.show()