import numpy as np
from typing import List, Dict, Tuple, Any
from haversine import haversine, Unit
import folium
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
import xarray as xr
from . import SECRETS

class NoiseMap:
    """
    The NoiseMap class is responsible for generating and displaying noise maps based on sound intensity levels.

    :ivar wind_turbines: A list of dictionaries containing the wind turbine data.
    :ivar noise: A xarray.DataArray containing the noise data vs wind speed.
    :ivar listeners: A list of dictionaries containing the observation points data.
    :ivar alpha: Air absorption coefficient.
    """

    def __init__(
        self,
        wind_turbines: dict,
        listeners: dict,
        alpha: float = 2.0,
    ):
        """
        Initialize the NoiseMap class.

        """
        self.alpha = alpha / 1000  # Convert alpha from dB/km to dB/m
        self.wind_turbines = wind_turbines
        self.listeners = listeners
        self.individual_noise = (
            self.superimpose_wind_turbines_noise()
        )  # pairs table with for each turbine, each listener

        self.generate_noise_map()

    def calculate_sound_level_at_distance(
        self, dBsource: float, distance: float
    ) -> float:
        """
        Calculate the sound level at a given distance from the source considering
        attenuation due to distance and atmospheric absorption.

        Parameters:
            dBsource (float): Sound level in decibels at the source.
            distance (float): Distance from the source in meters.

        Returns:
            float: Sound level at the given distance in decibels.
        """
        if distance == 0:
            return dBsource

        geometric_spreading_loss = 10 * np.log10(4 * np.pi * distance**2) + 11
        atmospheric_absorption_loss = self.alpha * distance

        total_attenuation = geometric_spreading_loss + atmospheric_absorption_loss
        resulting_sound_level = dBsource - total_attenuation

        return resulting_sound_level

    def superimpose_wind_turbines_noise(self):
        """
        Superimposes the sound levels of several wind turbines
        :return: a list of dictionaries, with each dictionary representing a pair of turbine and listener
        and the distance between them and the sound level at that distance for
        each wind speed level
        """
        pairs = [
            {
                "turbine_name": turbine,
                "turbine_position": turbine_specs["position"],
                "listener_name": listener,
                "listener_position": listener_specs["position"],
                "distance": round(
                    haversine(
                        turbine_specs["position"], listener_specs["position"], unit=Unit.METERS
                    )
                ),
            }
            for turbine, turbine_specs in self.wind_turbines.items()
            for listener, listener_specs in self.listeners.items()
        ]

        # add dB level for each turbine
        for p, pair in enumerate(pairs):
            noise = self.wind_turbines[pair["turbine_name"]]["noise_vs_wind_speed"]
            dB_level = self.calculate_sound_level_at_distance(noise, pair["distance"])
            pair["intensity_level_dB"] = dB_level
        return pairs

    def generate_noise_map(self):
        """
        Generates a noise map for the wind turbines
        and observation points based on the given wind speed.
        """

        # Determine the bounding box for the map
        lat_min = min(
            turbine["turbine_position"][0] for turbine in self.individual_noise
        )
        lat_max = max(
            turbine["turbine_position"][0] for turbine in self.individual_noise
        )
        lon_min = min(
            turbine["turbine_position"][1] for turbine in self.individual_noise
        )
        lon_max = max(
            turbine["turbine_position"][1] for turbine in self.individual_noise
        )
        margin = (1/125)

        # Adjust the map size to include observation points
        for point in self.individual_noise:
            lat_min = min(lat_min, point["listener_position"][0]) - margin
            lat_max = max(lat_max, point["listener_position"][0]) + margin
            lon_min = min(lon_min, point["listener_position"][1]) - margin
            lon_max = max(lon_max, point["listener_position"][1]) + margin

        lon_array = np.linspace(lon_min, lon_max, 100)
        lat_array = np.linspace(lat_min, lat_max, 100)
        LON, LAT = np.meshgrid(lon_array, lat_array)

        # Calculate the noise level at each point
        positions = [point["position"] for point in self.wind_turbines.values()]

        distances = np.array(
            [
                haversine(point1=(lat, lon), point2=position, unit=Unit.METERS)
                for lat, lon in zip(LAT.flatten(), LON.flatten())
                for position in positions
            ]
        ).reshape(LAT.shape[0], LAT.shape[1], len(positions))

        distance_noise = 20 * np.log10(distances)  # dB

        noise = np.vstack(
            [
                specs["noise_vs_wind_speed"].values
                for specs in self.wind_turbines.values()
            ]
        )

        intensity_distance = noise - distance_noise[..., None]
        # dB at distance
        Z = 10 * np.log10((10 ** (intensity_distance / 10)).sum(axis=2))
        # create xarray to store Z
        Z = xr.DataArray(
            data=Z,
            dims=("lat", "lon", "wind_speed"),
            coords={"lat": LAT[0], "lon": LON[0], "wind_speed": range(3, 13)},
        )

        # Store the values for later use
        self.LAT, self.LON, self.Z = LAT, LON, Z

    def plot_noise_map(self):
        """
        Plots the noise map with wind turbines and observation points.
        """

        # Create a wind speed slider for user interaction
        wind_speed_slider = widgets.FloatSlider(
            value=7.0,
            min=3.0,
            max=12.0,
            step=1.0,
            description="Wind Speed (m/s):",
            continuous_update=True,
        )

        @widgets.interact(wind_speed=wind_speed_slider)
        def interactive_plot(wind_speed):
            plt.figure(figsize=(10, 6))

            # Define contour levels starting from 35 dB
            contour_levels = [35, 40, 45, 50, 55, 60]

            # add bounding box
            plt.xlim(self.LON.min(), self.LON.max())
            plt.ylim(self.LAT.min(), self.LAT.max())

            plt.contourf(
                self.LON,  # x-axis, longitude
                self.LAT,  # y-axis, latitude
                self.Z.interp(
                    wind_speed=wind_speed, kwargs={"fill_value": "extrapolate"}
                ),
                levels=contour_levels,
                cmap="RdYlBu_r",
            )
            plt.colorbar(label="Noise Level (dB)")
            plt.title("Wind Turbine Noise Contours")
            plt.xlabel("Longitude")  # Correct label for x-axis
            plt.ylabel("Latitude")  # Correct label for y-axis

            # Plot wind turbines
            for turbine, specs in self.wind_turbines.items():
                plt.plot(
                    *specs["position"][::-1], "ko"
                )  # Make sure the position is in (Longitude, Latitude) order
                # add label next to it, add a small offset to avoid overlapping
                plt.text(
                    specs["position"][1] + 0.003,
                    specs["position"][0] + 0.002,
                    turbine,
                )

            # Plot observation points
            for point, specs in self.listeners.items():
                plt.plot(
                    *specs["position"][::-1], "ro"
                )  # Make sure the position is in (Longitude, Latitude) order
                # add label next to it
                plt.text(
                    specs["position"][1] + 0.002,
                    specs["position"][0] + 0.002,
                    point,
                )

            plt.grid(True)
            plt.show()

    def display_turbines_on_map(self):
        """
        Displays the wind turbines and observation points
        on a real map using their latitude and longitude.
        """

        # Get the average latitude and longitude to center the map
        avg_lat = sum(turbine["position"][0] for turbine in self.wind_turbines.values()) / len(
            self.wind_turbines
        )
        avg_lon = sum(turbine["position"][1] for turbine in self.wind_turbines.values()) / len(
            self.wind_turbines
        )

        # Create a folium map centered at the average latitude and longitude
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

        # Add markers for each wind turbine with a noise level of 105 dB
        for turbine, specs in self.wind_turbines.items():
            folium.Marker(
                location=specs["position"],
                tooltip=f"Noise Level: 105 dB",  # Set noise level to 105 dB
                icon=folium.Icon(icon="cloud"),
            ).add_to(m)

        # Add markers for the observation points
        for observation_point, specs in self.listeners.items():
            folium.Marker(
                location=specs["position"],
                tooltip="Observation Point",
                icon=folium.Icon(icon="star", color="red"),
            ).add_to(m)

        # Display the map
        display(m)

