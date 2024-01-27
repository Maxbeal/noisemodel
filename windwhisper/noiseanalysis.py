import numpy as np
import xarray as xr
import folium
from pathlib import Path
from datetime import datetime
from typing import Dict

from . import DATA_DIR


def compute_lden(data: dict) -> Dict[str, float]:
    """
    Compute L_den values for each listener.
    :param data: a dictionary with listener names as keys and xarray DataArrays as values
    :return: a dictionary with listener names as keys and L_den values as values
    """
    lden_values = {}

    for listener, vals in data.items():
        # Convert dB values to power, average them, and convert back to dB for each period

        # Convert dB to power, apply weightings, and sum

        l_den_power = (
            (
                12
                * 10
                ** (
                    np.nanmean(
                        np.where(
                            (vals.hour.values > 7) & (vals.hour.values <= 19),
                            vals,
                            np.nan,
                        )
                        / 10
                    )
                )
            )
            + (
                4
                * 10
                ** (
                    np.nanmean(
                        np.where(
                            (vals.hour.values > 19) & (vals.hour.values <= 23),
                            vals,
                            np.nan,
                        )
                        + 5
                    )
                    / 10
                )
            )
            + (
                8
                * 10
                ** (
                    np.nanmean(
                        np.where(
                            (vals.hour.values > 0) & (vals.hour.values <= 7),
                            vals,
                            np.nan,
                        )
                        + 10
                    )
                    / 10
                )
            )
        )

        # Convert back to dB to get L_den
        l_den = 10 * np.log10(l_den_power / 24)

        # Store L_den value for this listener
        lden_values[listener] = np.round(l_den, 1)

    return lden_values


class NoiseAnalysis:
    """
    This class handles the basic functionalities related to noise data analysis.

    :ivar wind_turbines: A list of dictionaries containing the wind turbine data.
    :ivar noise_map: A NoiseMap object containing the noise data.
    :ivar listeners: A list of dictionaries containing the observation points data.
    :ivar alpha: Air absorption coefficient.

    """

    def __init__(self, noise_map, wind_turbines, listeners):
        self.noise_map = noise_map
        self.wind_turbines = wind_turbines
        self.listeners = listeners
        self.distance_data = {
            (p["turbine_name"], p["listener_name"]): p
            for p in self.noise_map.individual_noise
        }
        self.alpha = 1 / 1000  # atmospheric absorption in dB/m
        self.analyze_and_calculate_lden()

    def analyze_and_calculate_lden(self):
        """
        Analyze the noise data and calculate L_den values for each listener.
        :return: updates the wt.listeners dictionary with L_den values
        """
        # Step 1: Interpolate noise based on wind speed data
        interpolated_noise = self.interpolate_noise()
        # Step 2: Calculate the cumulative dB
        cumulative_dB = self.calculate_cumulative_dB(interpolated_noise)
        # Step 3: Separate sound emissions into day, evening, and night
        # Step 4: Calculate L_den for each listener
        l_den = compute_lden(cumulative_dB)
        # Step 5: Update wt.listeners with L_den values
        self.update_listeners_with_lden(l_den)

    def interpolate_noise(self):
        # Creating a dictionary from noise_pairs with keys as (turbine_name, listener_name) tuples
        d = {
            (p["turbine_name"], p["listener_name"]): p
            for p in self.noise_map.individual_noise
        }

        # Initialize a dictionary to store the results
        interpolated_results = {}

        # Loop over each listener
        for listener in self.listeners:
            # Loop over each turbine
            for turbine in self.wind_turbines:
                # Retrieve the distance data for the current listener-turbine pair.
                distance = d[(turbine, listener)]["distance"]

                # Select the wind speed data for the current turbine.
                wind_speed_data = self.wind_turbines[turbine]["mean_wind_speed"].sel(
                    variable="wind_speed"
                )

                # Retrieve the intensity level data for the current listener-turbine pair.
                intensity_data = d[(turbine, listener)]["intensity_level_dB"]

                # Interpolate the intensity data at the wind speeds of the current turbine.
                interpolated_value = intensity_data.interp(wind_speed=wind_speed_data)

                # Replace NaN values in the interpolated data with 0.
                interpolated_value = xr.where(
                    np.isnan(interpolated_value), 0, interpolated_value
                )

                # Store the interpolated values and the distance in the dictionary.
                interpolated_results[(listener, turbine)] = {
                    "interpolated_intensity": interpolated_value,
                    "distance": distance,
                }

        return interpolated_results

    def calculate_cumulative_dB(self, interpolated_noise):
        """
        Calculate the cumulative dB for each listener at each time point.

        Parameters:
            interpolated_noise (dict): A dictionary with interpolated noise data for each turbine-listener pair.

        Returns:
            dict: A dictionary with cumulative dB values at each time point for each listener.
        """

        # Initialize a dictionary to store the cumulative dB for each listener.
        all_listeners_time_series = {}

        # Loop over each listener in the class's listeners list.
        for listener in self.listeners:
            # This dictionary comprehension filters the interpolated noise data for each listener.
            # It iterates over all items in the interpolated_noise dictionary, where each item's
            # key is a tuple (lstnr, turbine). The comprehension includes only those items
            # where the listener part of the key matches the current listener_name.
            # The resulting dictionary, listener_data, maps each turbine associated with
            # the current listener to its respective noise data.

            listener_data = {
                turbine: data
                for (lstnr, turbine), data in interpolated_noise.items()
                if lstnr == listener
            }

            # Convert dB to power for each turbine's data at each time point and sum them.
            # dB values are converted to power using the formula: power = 10^(dB/10).
            # The sum is computed using a list comprehension that iterates over each turbine's data.
            power_sums = sum(
                [
                    10 ** (turbine_data["interpolated_intensity"] / 10)
                    for turbine_data in listener_data.values()
                ]
            )

            # Convert the total power sum back to dB for each time point.
            # The dB value is calculated using the formula: dB = 10 * log10(power).
            total_dB_time_series = 10 * np.log10(power_sums)

            # Store the resulting time series of cumulative dB values for the listener.
            # The key is the listener's name, and the value is the cumulative dB time series.
            all_listeners_time_series[listener] = total_dB_time_series

        # Return the dictionary containing cumulative dB time series for each listener.
        return all_listeners_time_series

    def find_closest_wind_turbine(self):
        closest_turbine_distance = {}

        # Loop over each listener
        for listener in self.listeners:
            # Find the distance to each turbine and get the minimum
            min_distance = min(
                self.distance_data[(turbine, listener)]["distance"]
                for turbine in self.wind_turbines
            )
            closest_turbine_distance[listener] = min_distance

        return closest_turbine_distance

    def update_listeners_with_lden(self, l_den):
        for listener, specs in self.listeners.items():
            if listener in l_den:
                # Extract the numerical value from the xarray DataArray
                specs["l_den"] = l_den[listener]

    def display_listeners_on_map_with_Lden(self):
        """
        Display a map with listeners and their L_den values.
        :return: print statement with the map saved location.
        """
        icon_file_path = str(DATA_DIR / "pictures" / "icon_turbine.png")
        house_green_file_path = str(DATA_DIR / "pictures" / "house_green.png")
        house_orange_file_path = str(DATA_DIR / "pictures" / "house_orange.png")
        house_red_file_path = str(DATA_DIR / "pictures" / "house_red.png")
        house_dark_red_file_path = str(DATA_DIR / "pictures" / "house_dark_red.png")

        # Create a folium map
        # find firs the center of the map
        center_lat = np.mean(
            [specs["position"][0] for specs in self.listeners.values()]
        )
        center_lon = np.mean(
            [specs["position"][1] for specs in self.listeners.values()]
        )
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=12, tiles="CartoDB positron"
        )

        # Get the closest turbine distance for each listener
        closest_turbine_distance = self.find_closest_wind_turbine()

        # Define color map based on L_den value
        def get_file_path(lden_value):
            if lden_value < 43:
                return house_green_file_path
            elif lden_value < 45:
                return house_orange_file_path
            elif lden_value < 47:
                return house_red_file_path
            else:
                return house_dark_red_file_path

        # Add markers for each listener with L_den value
        # and distance to the closest turbine
        for listener, specs in self.listeners.items():
            icon_image = folium.features.CustomIcon(
                get_file_path(specs["l_den"]), icon_size=(50, 50)
            )
            folium.Marker(
                location=specs["position"],
                popup=f"{listener}: L_den {specs['l_den']} dB, "
                f"Closest turbine distance: {closest_turbine_distance[listener]} m",
                icon=icon_image,
            ).add_to(m)

        # Add markers for wind turbines
        for turbine, specs in self.wind_turbines.items():
            icon_image = folium.features.CustomIcon(icon_file_path, icon_size=(50, 50))
            folium.Marker(
                location=specs["position"],
                popup=f"{turbine}",
                icon=icon_image,
            ).add_to(m)

        # Adding a legend
        legend_html = """
         <div style="position: fixed; 
         bottom: 50px; left: 50px; width: 150px; height: 150px; 
         border:2px solid grey; z-index:9999; font-size:14px;
         ">&nbsp; L_den Legend <br>
         &nbsp; <i class="fa fa-map-marker fa-2x" style="color:green"></i>&nbsp; < 43 dB <br>
         &nbsp; <i class="fa fa-map-marker fa-2x" style="color:orange"></i>&nbsp; 43-46 dB <br>
         &nbsp; <i class="fa fa-map-marker fa-2x" style="color:red"></i>&nbsp; 46-48 dB <br>
         &nbsp; <i class="fa fa-map-marker fa-2x" style="color:darkred"></i>&nbsp; > 48 dB
          </div>
         """
        m.get_root().html.add_child(folium.Element(legend_html))

        # save the with today's date
        m.save(str(Path.cwd() / f"lden_map_{datetime.now().date()}.html"))

        print(f"Map saved to {Path.cwd() / f'lden_map_{datetime.now()}.html'}")
