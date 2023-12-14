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
import requests
import json
import netCDF4
from scipy.stats import weibull_min
import seaborn as sns
import py_wake
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.site.xrsite import XRSite
from py_wake.deficit_models import *
from py_wake.deficit_models.deficit_model import *
from py_wake.superposition_models import *
from py_wake.rotor_avg_models import *


# path to tests/fixtures folder, with test being in the root folder
FIXTURE_DIR = str(Path(__file__).parent.parent / "tests" / "fixtures")


class WindSpeed:
    """
    This class handles the basic functionalities related to wind data and operations.
    """

    def __init__(self, wind_turbines, start_year=2016, end_year=2018, debug=False):
        self.wind_turbines = wind_turbines
        self.start_year = start_year
        self.end_year = end_year
        self.weibull_params = {}

        if debug is True:
            # we unpickle the data to save time
            self.wind_speed = xr.open_dataset(f"{FIXTURE_DIR}/wind_speed_dummy_data.nc")

        else:
            self.wind_speed = self.download_data()
        self.mean_directional_wind_speed = self.calculate_mean_speed()
        self.hours_of_operation = self.calculate_operation_time()

    def _download_single_turbine_data(self, turbine):
        """
        Downloads data for a single wind turbine, with retry logic in case of failure.

        Args:
            turbine (dict): Dictionary containing the wind turbine's data.

        Returns:
            xarray.Dataset: Dataset containing the downloaded data.
        """
        latitude = turbine["position"][0]
        longitude = turbine["position"][1]
        url = (
            f"https://wps.neweuropeanwindatlas.eu/api/mesoscale-ts/"
            f"v1/get-data-point?latitude={latitude}&longitude={longitude}"
            f"&variable=WD10&variable=WS10&dt_start={self.start_year}-"
            f"01-01T00:00:00&dt_stop={self.end_year}-12-31T23:30:00"
        )

        attempts = 0
        max_attempts = 10
        total_size_kb = 0  # Initialize a variable to store the total size

        while attempts < max_attempts:
            try:
                response = requests.get(url, timeout=25)
                if response.status_code == 200:
                    size_kb = len(response.content) / 1024  # Calculate the size of the response in kilobytes
                    total_size_kb += size_kb  # Add the size of this response to the total
                    print(
                        f"Downloaded {size_kb:.2f} kB for {turbine['name']}")  # Optional: Print the size for this turbine

                    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp_file:
                        tmp_file.write(response.content)
                        tmp_file_path = tmp_file.name
                    ds = xr.open_dataset(tmp_file_path)

                    # Simplify the dataset
                    ds_simplified = ds[["WD10", "WS10"]].copy()
                    ds_simplified.attrs.clear()

                    return ds_simplified, size_kb # Return the simplified dataset
                else:
                    raise Exception(
                        f"Error downloading data for {turbine['name']}. Status code: {response.status_code}"
                    )
            except (Timeout, Exception) as e:
                attempts += 1
                if attempts >= max_attempts:
                    raise Exception(
                        f"Failed to download data for {turbine['name']} after {max_attempts} attempts."
                    )
                print(
                    f"Retrying download for {turbine['name']} (Attempt {attempts}/{max_attempts})"
                )

    def download_data(self) -> xr.DataArray:
        print("Starting concurrent data download for all turbines...")
        total_download_size_kb = 0  # Initialize total download size

        # Create a dictionary to store the dataframes for each turbine
        turbine_data = {}

        # Use ThreadPoolExecutor to download data concurrently
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Create a future for each turbine
            futures = {executor.submit(self._download_single_turbine_data, turbine): turbine for turbine in
                       self.wind_turbines}

            for future in as_completed(futures):
                # Get the turbine information
                turbine = futures[future]

                try:
                    # Get the result from the future
                    ds_simplified, size_kb = future.result()
                    total_download_size_kb += size_kb  # Add the size of this download to the total

                    # Store the data in the dictionary with the turbine's name as the key
                    turbine_data[turbine["name"]] = ds_simplified
                except Exception as exc:
                    print(f'{turbine["name"]} generated an exception: {exc}')

        # Concatenate the dataframes along a new 'turbine' dimension
        all_data = xr.concat(turbine_data.values(), dim=pd.Index(turbine_data.keys(), name="turbine"))

        # Reset all coordinates except "turbine" and "time"
        all_data = all_data.reset_coords(drop=True)

        print(f"Total download size: {total_download_size_kb:.2f} kB")  # Print the total download size
        print("Done.")

        return all_data

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
        ws_bins = np.arange(
            0, df.loc[:, ("WS10", slice(None))].max().max() + 1, 1
        )  # Wind speed bins of 1 km/h

        # Separate the DataFrame into two DataFrames for WD10 and WS10
        df_wd = df["WD10"]
        df_ws = df["WS10"]

        # Initialize an empty dictionary to hold the results
        result_tables = {}

        # Get the unique turbine names
        turbines = df.columns.get_level_values("turbine").unique()

        # Iterate over each turbine
        for turbine in turbines:
            # Create binned data for wind direction and wind speed
            wd_binned = pd.cut(
                df_wd[turbine],
                bins=wd_bins,
                labels=[
                    f"{left}-{right}" for left, right in zip(wd_bins[:-1], wd_bins[1:])
                ],
                right=False,
            )
            ws_binned = pd.cut(
                df_ws[turbine],
                bins=ws_bins,
                labels=[
                    f"{left}-{right}" for left, right in zip(ws_bins[:-1], ws_bins[1:])
                ],
                right=False,
            )

            # Merge the binned data into a single DataFrame
            binned_df = pd.DataFrame({"WD_bin": wd_binned, "WS_bin": ws_binned})

            # Create a crosstab table to count the number of hours per bin
            crosstab = pd.crosstab(binned_df["WD_bin"], binned_df["WS_bin"])

            # Store the crosstab table in the result dictionary
            result_tables[turbine] = crosstab.T

        data_arrays = [
            xr.DataArray(
                table, dims=["wind_direction_bin", "wind_speed_bin"], name=turbine
            )
            for turbine, table in result_tables.items()
        ]

        # Concatenate these DataArrays along a new 'turbine' dimension
        combined_da = xr.concat(
            data_arrays, dim=pd.Index(result_tables.keys(), name="turbine")
        )

        return combined_da.fillna(0)

    def _create_wind_rose(self, turbine, ax):
        data = self.wind_speed.sel(turbine=turbine["name"])

        divider = make_axes_locatable(ax)
        ax_windrose = divider.append_axes(
            "right", size="100%", pad=0.0, axes_class=WindroseAxes
        )

        ax_windrose.bar(
            data["WD10"],
            data["WS10"],
            normed=True,
            opening=0.8,
            edgecolor="white",
            bins=np.arange(0, 25, 5),
        )
        ax_windrose.set_legend(
            title=turbine["name"], loc="upper left", bbox_to_anchor=(1.2, 1.0)
        )
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
            axs[row, col].axis("off")
            # Create a wind rose for this turbine
            self._create_wind_rose(turbine, axs[row, col])

        # remove empty subplots
        for i in range(len(self.wind_turbines), n_rows * 2):
            row, col = np.unravel_index(i, (n_rows, 2))
            fig.delaxes(axs[row, col])

        plt.show()

    def download_weibull_coefficients(self):
        weibull_data = {}

        for turbine in self.wind_turbines:
            latitude, longitude = turbine["position"]
            url = (
                f"https://wps.neweuropeanwindatlas.eu/api/microscale-atlas/v1/get-data-point"
                f"?latitude={latitude}&longitude={longitude}&height=100"
                f"&variable=weib_A_combined&variable=weib_k_combined"
            )

            try:
                response = requests.get(url, timeout=20)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"Request failed: {e}")
                continue

            with netCDF4.Dataset("inmemory.nc", memory=response.content) as ds:
                # Assuming 'weib_A_combined' and 'weib_k_combined' are variable names in the dataset
                A = ds["weib_A_combined"][:]
                k = ds["weib_k_combined"][:]

                weibull_data[turbine["name"]] = {"A": A, "k": k}

        # Set the weibull_params attribute
        self.weibull_params = weibull_data
        return weibull_data

    def plot_weibull_wind_speed_distribution(self):
        wind_speeds = np.linspace(0, 25, 1000)  # Array of wind speeds from 0 to 25 m/s

        # Set the aesthetic style of the plots
        sns.set_style("whitegrid")

        for turbine, params in self.weibull_params.items():
            A = params["A"].data[0]
            k = params["k"].data[0]

            # Calculate the PDF values for each wind speed
            pdf_values = weibull_min.pdf(wind_speeds, c=k, scale=A)

            # Calculate the number of hours per year for each wind speed
            hours_per_year = pdf_values * (wind_speeds[1] - wind_speeds[0]) * 8760

            # Create the plot
            plt.figure(figsize=(10, 5))
            plt.plot(
                wind_speeds,
                hours_per_year,
                label=f"Turbine: {turbine}\nA={A:.2f}, k={k:.2f}",
            )
            plt.xlabel("Wind Speed (m/s)")
            plt.ylabel("Hours per Year")
            plt.title("Wind Speed Distribution")
            plt.legend()
            plt.show()

    def run_simulation(self):
        # Download Weibull coefficients for each turbine
        weibull_data = self.download_weibull_coefficients()

        total_aep = 0  # Initialize total AEP

        for turbine in self.wind_turbines:
            # Extract Weibull A and k values for the current turbine
            A = float(weibull_data[turbine["name"]]["A"])
            k = float(weibull_data[turbine["name"]]["k"])

            # Creating a single sector with frequency 1
            f = [1]
            wd = [0]  # Single wind direction sector

            # Create XRSite with Weibull distributed wind speed for the current turbine
            site_ds = xr.Dataset({
                'Sector_frequency': ('wd', f),
                'Weibull_A': ('wd', [A]),
                'Weibull_k': ('wd', [k])
            }, coords={'wd': wd})

            site = XRSite(ds=site_ds)

            # Create and run wind farm model for the current turbine
            # Note: Adjust the model as per the specific turbine's characteristics
            # For instance, using GenericWindTurbine for each turbine
            gen_wt = GenericWindTurbine(
                name=turbine["name"],
                diameter=turbine["diameter"],
                hub_height=turbine["hub height"],
                power_norm=turbine["power"],
                turbulence_intensity=0.0
            )

            # Assuming x, y coordinates for the turbine are set
            x = [turbine["position"][0]]  # Replace with actual coordinate
            y = [turbine["position"][1]]  # Replace with actual coordinate

            wfm = PropagateDownwind(site, [gen_wt], ...)
            sim_res = wfm(x, y)

            # Aggregate AEP
            total_aep += sim_res.aep().sum()

        return total_aep
