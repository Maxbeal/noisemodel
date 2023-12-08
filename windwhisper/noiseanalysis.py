import pandas as pd
import numpy as np
import xarray as xr
import folium
from pathlib import Path
from datetime import datetime

from . import DATA_DIR

MAX_WIND_SPEED = 30  # Assuming max wind speed is 30 m/s


class NoiseAnalysis:
    def __init__(self, wind_speed, noise, noise_map, wind_turbines, listeners):
        self.wind_speed = wind_speed
        self.noise = noise
        self.noise_map = noise_map

        self.turbines = wind_turbines
        self.listeners = listeners
        self.distance_data = {
            (p["turbine_name"], p["listener_name"]): p
            for p in self.noise_map.individual_noise
        }
        self.alpha = 1 / 1000  # atmospheric absorption in dB/m

    def analyze_and_calculate_lden(self):
        # Step 1: Interpolate noise based on wind speed data
        interpolated_noise = self.interpolate_noise(
            self.wind_speed, self.noise_map.individual_noise
        )
        # Step 2: Calculate the cumulative dB
        cumulative_dB = self.calculate_cumulative_dB(interpolated_noise)
        # Step 3: Separate sound emissions into day, evening, and night
        noise_separated = self.separate_noise_emissions(cumulative_dB)
        # Step 4: Calculate L_den for each listener
        Lden = self.compute_lden(noise_separated)
        # Step 5: Update wt.listeners with L_den values
        self.update_listeners_with_lden(Lden)
        # Return values if needed, otherwise they can be accessed directly from the instance
        return Lden, self.listeners, noise_separated

    def interpolate_noise(self, wind_speed, noise_pairs):
        # Creating a dictionary from noise_pairs with keys as (turbine_name, listener_name) tuples
        d = {
            (p["turbine_name"], p["listener_name"]): p
            for p in self.noise_map.individual_noise
        }

        # Initialize a dictionary to store the results
        interpolated_results = {}

        # Loop over each listener
        for listener in self.listeners:
            listener_name = listener[
                "name"
            ]  # Extract the name of the current listener.

            # Loop over each turbine
            for turbine in self.turbines:
                turbine_name = turbine[
                    "name"
                ]  # Extract the name of the current turbine.

                # Retrieve the distance data for the current listener-turbine pair.
                distance = d[(turbine_name, listener_name)]["distance"]

                # Select the wind speed data for the current turbine.
                wind_speed_data = self.wind_speed["WS10"].sel(turbine=turbine_name)

                # Retrieve the intensity level data for the current listener-turbine pair.
                intensity_data = d[(turbine_name, listener_name)]["intensity_level_dB"]

                # Interpolate the intensity data at the wind speeds of the current turbine.
                interpolated_value = intensity_data.interp(wind_speed=wind_speed_data)

                # Replace NaN values in the interpolated data with 0.
                interpolated_value = xr.where(
                    np.isnan(interpolated_value), 0, interpolated_value
                )

                # Store the interpolated values and the distance in the dictionary.
                interpolated_results[(listener_name, turbine_name)] = {
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
            listener_name = listener[
                "name"
            ]  # Extract the name of the current listener.

            # This dictionary comprehension filters the interpolated noise data for each listener.
            # It iterates over all items in the interpolated_noise dictionary, where each item's key is a tuple (lstnr, turbine).
            # The comprehension includes only those items where the listener part of the key matches the current listener_name.
            # The resulting dictionary, listener_data, maps each turbine associated with the current listener to its respective noise data.

            listener_data = {
                turbine: data
                for (lstnr, turbine), data in interpolated_noise.items()
                if lstnr == listener_name
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
            all_listeners_time_series[listener_name] = total_dB_time_series

        # Return the dictionary containing cumulative dB time series for each listener.
        return all_listeners_time_series

    def separate_noise_emissions(
        self,
        all_listeners_time_series,
        day_start="07:00",
        day_end="19:00",
        evening_start="19:00",
        evening_end="23:00",
    ):
        # Initialize a dictionary to store separated data for each listener
        separated_data = {}

        # Loop through each listener's data in the provided time series data
        for listener, data in all_listeners_time_series.items():
            # Convert the 'time' coordinate of the data to a pandas datetime index
            # This allows for easier manipulation and filtering based on time
            time_index = pd.to_datetime(data["time"].values)

            # Create a boolean mask for the day period
            # The mask is True for times within the defined day start and end times
            day_mask = xr.DataArray(
                (time_index.time >= pd.to_datetime(day_start).time())
                & (time_index.time < pd.to_datetime(day_end).time()),
                coords={"time": data["time"]},
                dims=["time"],
            )

            # Create a boolean mask for the evening period
            # The mask is True for times within the defined evening start and end times
            evening_mask = xr.DataArray(
                (time_index.time >= pd.to_datetime(evening_start).time())
                & (time_index.time < pd.to_datetime(evening_end).time()),
                coords={"time": data["time"]},
                dims=["time"],
            )

            # Create a boolean mask for the night period
            # The mask is True for times outside the defined day period, i.e., from evening end to day start
            night_mask = xr.DataArray(
                (time_index.time >= pd.to_datetime(evening_end).time())
                | (time_index.time < pd.to_datetime(day_start).time()),
                coords={"time": data["time"]},
                dims=["time"],
            )

            # Filter the data for the day period using the day_mask
            # 'drop=True' removes the time points where the condition is False (outside the day period)
            day_data = data.where(day_mask, drop=True)

            # Filter the data for the evening period using the evening_mask
            evening_data = data.where(evening_mask, drop=True)

            # Filter the data for the night period using the night_mask
            night_data = data.where(night_mask, drop=True)

            # Store the separated day, evening, and night data for the current listener
            # The data for each period is stored in a dictionary under keys 'day', 'evening', and 'night'
            separated_data[listener] = {
                "day": day_data,
                "evening": evening_data,
                "night": night_data,
            }

        # Return the dictionary containing separated data for each listener
        return separated_data

    def find_closest_wind_turbine(self):
        closest_turbine_distance = {}

        # Loop over each listener
        for listener in self.listeners:
            listener_name = listener["name"]
            # Find the distance to each turbine and get the minimum
            min_distance = min(
                self.distance_data[(turbine["name"], listener_name)]["distance"]
                for turbine in self.turbines
            )
            closest_turbine_distance[listener_name] = min_distance

        return closest_turbine_distance

    def compute_lden(self, separated_data):
        lden_values = {}

        for listener, periods in separated_data.items():
            # Convert dB values to power, average them, and convert back to dB for each period
            L_day_avg = 10 * np.log10((10 ** (periods["day"] / 10)).mean())
            L_evening_avg = 10 * np.log10((10 ** (periods["evening"] / 10)).mean())
            L_night_avg = 10 * np.log10((10 ** (periods["night"] / 10)).mean())

            # Convert dB to power, apply weightings, and sum
            L_den_power = (
                (12 * 10 ** (L_day_avg / 10))
                + (4 * 10 ** ((L_evening_avg + 5) / 10))
                + (8 * 10 ** ((L_night_avg + 10) / 10))
            )

            # Convert back to dB to get L_den
            L_den = 10 * np.log10(L_den_power / 24)

            # Store L_den value for this listener
            lden_values[listener] = np.round(L_den.values, 1)

        return lden_values

    def update_listeners_with_lden(self, Lden):
        for listener in self.listeners:
            listener_name = listener["name"]
            if listener_name in Lden:
                # Extract the numerical value from the xarray DataArray
                lden_value = Lden[listener_name]
                listener["L_den"] = lden_value

        # Optionally, you can return the updated listeners list
        return self.listeners

    def display_listeners_on_map_with_Lden(self):
        icon_file_path = str(DATA_DIR / "pictures" / "icon_turbine.png")
        house_green_file_path = str(DATA_DIR / "pictures" / "house_green.png")
        house_orange_file_path = str(DATA_DIR / "pictures" / "house_orange.png")
        house_red_file_path = str(DATA_DIR / "pictures" / "house_red.png")
        house_dark_red_file_path = str(DATA_DIR / "pictures" / "house_dark_red.png")

        # Create a folium map
        m = folium.Map(
            location=[47.3769, 8.5417], zoom_start=12, tiles="CartoDB positron"
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

        # Add markers for each listener with L_den value and distance to closest turbine
        for listener in self.listeners:
            listener_name = listener["name"]
            icon_image = folium.features.CustomIcon(
                get_file_path(listener["L_den"]), icon_size=(50, 50)
            )
            folium.Marker(
                location=listener["position"],
                popup=f"{listener_name}: L_den {listener['L_den']} dB, Closest turbine distance: {closest_turbine_distance[listener_name]} m",
                icon=icon_image,
            ).add_to(m)

        # Add markers for wind turbines
        for turbine in self.turbines:
            icon_image = folium.features.CustomIcon(icon_file_path, icon_size=(50, 50))
            folium.Marker(
                location=turbine["position"],
                popup=f"{turbine['name']}",
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

        m.save(Path.cwd() / f"lden_map_{datetime.now()}.html")
