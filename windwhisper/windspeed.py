from requests.exceptions import Timeout
import tempfile
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path
import requests

# path to dev/fixtures folder, with test being in the root folder
FIXTURE_DIR = Path(__file__).parent.parent / "dev" / "fixtures"


def load_wind_speed_data(filepath_wind_speed: Path, filepath_correction: Path = None, array: xr.DataArray = None) -> xr.DataArray:
    """
    Load the wind speed data from a file, or from a xarray.DataArray if provided.
    :param filepath_wind_speed: Filepath to the wind speed data.
    :param filepath_correction: Filepath to the correction data.
    :param array: A xarray.DataArray containing the wind speed data. If provided, the filepaths are ignored.
    :return: A xarray.DataArray containing the wind speed data.
    """
    if array is not None:
        return array
    else:
        wind_speed = xr.open_dataset(filepath_wind_speed).to_array().mean(dim="month")
        if filepath_correction is None:
            return wind_speed
        correction = xr.open_dataset(filepath_correction).to_array()
        correction = correction.sel(variable='ratio_gwa2_era5_mean_WS').interp(latitude=wind_speed.latitude, longitude=wind_speed.longitude, method="linear")

        return wind_speed * correction


def _download_single_turbine_data(turbine):
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
        f"&variable=WD10&variable=WS10&dt_start=2016-"
        f"01-01T00:00:00&dt_stop=2018-12-31T23:30:00"
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

                return ds_simplified, size_kb  # Return the simplified dataset
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


def download_data(wind_turbines) -> xr.DataArray:
    print("Starting concurrent data download for all turbines...")
    total_download_size_kb = 0  # Initialize total download size

    # Create a dictionary to store the dataframes for each turbine
    turbine_data = {}

    # Use ThreadPoolExecutor to download data concurrently
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Create a future for each turbine
        futures = {executor.submit(_download_single_turbine_data, turbine): turbine for turbine in
                   wind_turbines}

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


class WindSpeed:
    """
    This class handles the basic functionalities related to wind data loading.

    :ivar wind_turbines: A list of dictionaries containing the wind turbine data.
    :ivar wind_speed: A xarray.DataArray containing the wind speed data.

    """

    def __init__(
            self,
            wind_turbines: dict,
            wind_speed_data: xr.DataArray | None = None
    ):
        self.wind_turbines = wind_turbines
        try:
            self.wind_speed = load_wind_speed_data(
                filepath_wind_speed=FIXTURE_DIR / "era5_mean_2013-2022_month_by_hour.nc",
                filepath_correction=FIXTURE_DIR / "ratio_gwa2_era5.nc",
                array=wind_speed_data
            )
        except FileNotFoundError:
            self.wind_speed = download_data(self.wind_turbines)

        self.calculate_mean_speed()

    def calculate_mean_speed(self) -> None:
        """
        Calculate the mean wind speed for each turbine, for each hour of the day.
        :return: Nothing. Populates the `mean_wind_speed` key of self.wind_turbines.
        """

        for turbine, specs in self.wind_turbines.items():
            specs["mean_wind_speed"] = self.wind_speed.sel(
                latitude=specs["position"][0],
                longitude=specs["position"][1],
                method="nearest",
            ).interp(height=specs["hub height"])

