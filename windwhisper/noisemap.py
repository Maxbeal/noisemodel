import numpy as np
from typing import List, Dict, Tuple, Any
from haversine import haversine, Unit
import folium
import ipywidgets as widgets
from IPython.display import display
import osmnx as ox
import geopandas as gpd
import requests
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import xarray as xr
import math

from . import SECRETS

# API Google key
GOOGLE_ELEVATION_API_KEY = SECRETS["GOOGLE_ELEVATION_API_KEY"]


class NoiseMap:
    """
    The NoiseMap class is responsible for generating and displaying noise maps based on sound intensity levels.

    Attributes:
        alpha (float): Air absorption coefficient.
        wind_turbines (list): List of wind turbines with their specifications and noise levels.
    """

    def __init__(
        self,
        wind_turbines: List[Dict],
        noise: xr.DataArray,
        listeners: List[Dict],
        alpha: float = 2.0,
    ):
        """
        Initialize the NoiseMap class.

        Parameters:
            wind_turbines (List[Dict]): List of wind turbines with their specifications.
            wind_speed (WindTurbines): Noise levels asa function of wind speed.
            alpha (float, optional): Air absorption coefficient. Defaults to 2.0.
        """
        self.alpha = alpha / 1000  # Convert alpha from dB/km to dB/m

        self.wind_turbines = wind_turbines
        self.noise = noise
        self.listeners = listeners
        self.individual_noise = (
            self.superpose_several_wind_turbine_sounds_in_dB()
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

    def superpose_several_wind_turbine_sounds_in_dB(self):
        pairs = [
            {
                "turbine_name": turbine["name"],
                "turbine_position": turbine["position"],
                "listener_name": listener["name"],
                "listener_position": listener["position"],
                "distance": round(
                    haversine(
                        turbine["position"], listener["position"], unit=Unit.METERS
                    )
                ),
            }
            for turbine in self.wind_turbines
            for listener in self.listeners
        ]

        # add dB level for each turbine
        for pair in pairs:
            noise = self.noise.sel(turbine=pair["turbine_name"])
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
        margin = (1/250)

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
        positions = [point["position"] for point in self.wind_turbines]

        distances = np.array(
            [
                haversine(point1=(lat, lon), point2=position, unit=Unit.METERS)
                for lat, lon in zip(LAT.flatten(), LON.flatten())
                for position in positions
            ]
        ).reshape(LAT.shape[0], LAT.shape[1], len(positions))

        distance_noise = 20 * np.log10(distances)  # dB
        intensity_distance = self.noise.values - distance_noise[..., None]
        # dB at distance
        Z = 10 * np.log10((10 ** (intensity_distance / 10)).sum(axis=2))
        # create xarray to store Z
        Z = xr.DataArray(
            data=Z,
            dims=("lat", "lon", "wind_speed"),
            coords={"lat": LAT[0], "lon": LON[0], "wind_speed": self.noise.wind_speed},
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
            for turbine in self.wind_turbines:
                plt.plot(
                    *turbine["position"][::-1], "ko"
                )  # Make sure the position is in (Longitude, Latitude) order
                # add label next to it, add a small offset to avoid overlapping
                plt.text(
                    turbine["position"][1] + 0.002,
                    turbine["position"][0] + 0.002,
                    turbine["name"],
                )

            # Plot observation points
            for point in self.listeners:
                plt.plot(
                    *point["position"][::-1], "ro"
                )  # Make sure the position is in (Longitude, Latitude) order
                # add label next to it
                plt.text(
                    point["position"][1] + 0.002,
                    point["position"][0] + 0.002,
                    point["name"],
                )

            plt.grid(True)
            plt.show()

    def display_turbines_on_map(self):
        """
        Displays the wind turbines and observation points
        on a real map using their latitude and longitude.
        """

        # Get the average latitude and longitude to center the map
        avg_lat = sum(turbine["position"][0] for turbine in self.wind_turbines) / len(
            self.wind_turbines
        )
        avg_lon = sum(turbine["position"][1] for turbine in self.wind_turbines) / len(
            self.wind_turbines
        )

        # Create a folium map centered at the average latitude and longitude
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

        # Add markers for each wind turbine with a noise level of 105 dB
        for turbine in self.wind_turbines:
            folium.Marker(
                location=turbine["position"],
                tooltip=f"Noise Level: 105 dB",  # Set noise level to 105 dB
                icon=folium.Icon(icon="cloud"),
            ).add_to(m)

        # Add markers for the observation points
        for observation_point in self.listeners:
            folium.Marker(
                location=observation_point["position"],
                tooltip="Observation Point",
                icon=folium.Icon(icon="star", color="red"),
            ).add_to(m)

        # Display the map
        display(m)

    def get_landuse_between_points(self, turbine_name, observation_point):
        # Find the position of the turbine by its name
        turbine_position = next(
            (
                turbine["position"]
                for turbine in self.wind_turbines
                if turbine["name"] == turbine_name
            ),
            None,
        )
        if not turbine_position:
            raise ValueError(f"No turbine found with the name {turbine_name}")

        # Get the bounding box coordinates
        north, south = max(turbine_position[0], observation_point[0]), min(
            turbine_position[0], observation_point[0]
        )
        east, west = max(turbine_position[1], observation_point[1]), min(
            turbine_position[1], observation_point[1]
        )

        # Fetch land use data
        landuse = ox.features_from_bbox(
            north, south, east, west, tags={"landuse": True}
        )
        return landuse

    def plot_landuse_for_turbines_and_observation_points(
        self, wind_turbines, observation_points
    ):
        # Extract turbine positions and combine with observation points
        turbine_positions = [turbine["position"] for turbine in wind_turbines]
        all_points = turbine_positions + observation_points

        # Create GeoDataFrame for all points
        gdf_points = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lat, lon in all_points], crs="EPSG:4326"
        )

        # Extract latitude and longitude for bounding box calculation
        all_lats, all_lons = zip(*all_points)
        margin = 0.01
        bbox = dict(
            north=max(all_lats) + margin,
            south=min(all_lats) - margin,
            east=max(all_lons) + margin,
            west=min(all_lons) - margin,
        )

        # Fetch landuse features and create a color map for unique landuse types
        landuse = ox.features_from_bbox(**bbox, tags={"landuse": True})
        landuse_types = landuse["landuse"].unique()
        cmap = plt.get_cmap("tab20", len(landuse_types))
        color_map = {
            landuse_type: cmap(i) for i, landuse_type in enumerate(landuse_types)
        }

        # Plot landuse, turbines, and observation points
        ax = landuse.to_crs(epsg=3857).plot(
            color=landuse["landuse"].map(color_map), figsize=(12, 12)
        )
        gdf_points.iloc[: len(turbine_positions)].to_crs(epsg=3857).plot(
            ax=ax, color="blue", markersize=100, label="Éoliennes"
        )
        gdf_points.iloc[len(turbine_positions) :].to_crs(epsg=3857).plot(
            ax=ax, color="red", markersize=100, label="Points d'observation"
        )

        # Plot lines between turbine positions and observation points
        lines = [
            LineString([turbine[::-1], observation[::-1]])
            for turbine in turbine_positions
            for observation in observation_points
        ]
        gpd.GeoSeries(lines, crs="EPSG:4326").to_crs(epsg=3857).plot(
            ax=ax, color="black", linewidth=1
        )

        # Create legend
        patches = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=landuse_type,
                markersize=10,
                markerfacecolor=color_map[landuse_type],
            )
            for landuse_type in landuse_types
        ]
        ax.legend(
            handles=patches
            + [
                plt.Line2D(
                    [0], [0], marker="o", color="blue", label="Éoliennes", markersize=10
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="red",
                    label="Points d'observation",
                    markersize=10,
                ),
            ],
            title="Légende",
        )
        plt.show()

    def get_altitudes_between_points(self, point1, point2, num_samples=100):
        latitudes = np.linspace(point1[0], point2[0], num_samples)
        longitudes = np.linspace(point1[1], point2[1], num_samples)

        locations = "|".join(
            [f"{lat},{lon}" for lat, lon in zip(latitudes, longitudes)]
        )

        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={GOOGLE_ELEVATION_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if data["status"] != "OK":
            raise Exception(f"Erreur lors de l'appel à l'API: {data['status']}")

        altitudes = [result["elevation"] for result in data["results"]]

        return altitudes

    def plot_relief_between_points(self, wind_turbines_names, observation_points_names):
        for turbine_name in wind_turbines_names:
            turbine_position = next(
                (
                    turbine["position"]
                    for turbine in self.wind_turbines
                    if turbine["name"] == turbine_name
                ),
                None,
            )
            if turbine_position is None:
                continue  # Skip if turbine name not found

            for listener_name in observation_points_names:
                listener = next(
                    (
                        listener
                        for listener in self.listeners
                        if listener["name"] == listener_name
                    ),
                    None,
                )
                if listener is None:
                    continue  # Skip if listener name not found

                altitudes = self.get_altitudes_between_points(
                    turbine_position, listener["position"]
                )
                plt.figure(figsize=(5, 3))
                plt.plot(altitudes)
                plt.title(f"Relief between {turbine_name} and {listener_name}")
                plt.xlabel("Intermediate Points")
                plt.ylabel("Altitude (m)")
                plt.show()
