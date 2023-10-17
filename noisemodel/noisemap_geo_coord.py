import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from windturbinemodel import WindTurbineModel
from haversine import haversine, Unit
import folium
from IPython.core.display import HTML
from windspeed import WindModule
import ipywidgets as widgets
from IPython.display import display
import osmnx as ox
import geopandas as gpd
import requests
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import folium.plugins as plugins
import pandas as pd

# Votre clé API Google
GOOGLE_ELEVATION_API_KEY = "AIzaSyDdmjBS36KrwL-pDYBANiPM1mky_Jd15OQ"



class NoiseMap:
    """
    The NoiseMap class is responsible for generating and displaying noise maps based on sound intensity levels.
   
    Attributes:
        alpha (float): Air absorption coefficient.
        W0 (float): Reference power.
        I0 (float): Reference intensity.
        wind_turbines (list): List of wind turbines with their specifications and noise levels.
    """

    def __init__(self, wind_turbines: List[Dict], wind_turbine_model: WindTurbineModel, alpha: float = 2.0, wind_speed: float = 8.0):
        """
        Initialize the NoiseMap class.

        Parameters:
            wind_turbines (List[Dict]): List of wind turbines with their specifications.
            wind_turbine_model (WindTurbineModel): Model to predict noise levels.
            alpha (float, optional): Air absorption coefficient. Defaults to 2.0.
        """
        self.alpha = alpha / 1000  # Convert alpha from dB/km to dB/m
        self.W0 = 1e-12
        self.I0 = 1e-12
        
        self.wind_turbines = wind_turbines
        self.wind_turbine_model = wind_turbine_model

    def calculate_sound_intensity_level(self, dBsource: float, distance: float) -> float:
        """
        Calculates sound intensity level at a given distance from the source.

        Parameters:
            dBsource (float): Sound level in decibels at the source.
            distance (float): Distance from the source in meters.

        Returns:
            float: Sound intensity level at the given distance.
        """
        if distance == 0:
            return float('inf')  # return infinity if distance is zero
        wattsource = (10 ** (dBsource / 10)) * self.W0
        intensity_level = wattsource / (4 * math.pi * distance ** 2)
        # apply air absorption
        intensity_level = 10 ** (-self.alpha * distance / 10) * intensity_level
        return intensity_level #given in W 

    def convert_intensity_level_into_dB(self, intensity_level: float) -> float:
        """Converts a sound intensity level into decibels."""
        return 10 * math.log10(intensity_level / self.I0)

    def superpose_several_wind_turbine_sounds(self, observation_points: List[Tuple[float, float]], wind_speed: float) -> Dict[Tuple[float, float], float]:
        """Superposes sounds from multiple wind turbines at specific observation points based on the given wind speed."""

        noise_predictions = self.wind_turbine_model.predict_noise_at_wind_speed(self.wind_turbines, wind_speed)
        results = {}

        for observation_point in observation_points:
            total_intensity_level = 0
            for turbine in self.wind_turbines:
                turbine_name = turbine.get('name')
                dBsource = noise_predictions[turbine_name][f"{wind_speed:.1f}m/s"]
                distance = haversine(observation_point, turbine['position'], unit=Unit.METERS)
                intensity_level = self.calculate_sound_intensity_level(dBsource, distance)
                total_intensity_level += intensity_level
            dB_total = self.convert_intensity_level_into_dB(total_intensity_level)
            results[observation_point] = dB_total

        return results

    
    def generate_noise_map(self, observation_points: List[Tuple[float, float]], wind_speed: float):
        """Generates a noise map for the wind turbines and observation points based on the given wind speed."""

        # Get the noise predictions for the given wind speed
        noise_predictions = self.wind_turbine_model.predict_noise_at_wind_speed(self.wind_turbines, wind_speed)

        # Determine the bounding box for the map
        lat_min = min(turbine['position'][0] for turbine in self.wind_turbines)
        lat_max = max(turbine['position'][0] for turbine in self.wind_turbines)
        lon_min = min(turbine['position'][1] for turbine in self.wind_turbines)
        lon_max = max(turbine['position'][1] for turbine in self.wind_turbines)
        delay = 1/250

        # Adjust the map size to include observation points
        for point in observation_points:
            lat_min = min(lat_min, point[0]) - delay
            lat_max = max(lat_max, point[0]) + delay
            lon_min = min(lon_min, point[1]) - delay
            lon_max = max(lon_max, point[1]) + delay

        LAT, LON = np.meshgrid(np.linspace(lat_min, lat_max, 100), np.linspace(lon_min, lon_max, 100))
        Z = np.zeros_like(LAT)

        for turbine in self.wind_turbines:
            distances = np.vectorize(lambda lat, lon: haversine((lat, lon), turbine['position'], unit=Unit.METERS))(LAT, LON)

            # Use the predicted noise level for the turbine at the given wind speed
            turbine_name = turbine.get('name')
            dBsource = noise_predictions[turbine_name][f"{wind_speed:.1f}m/s"]

            intensity_source = 10 ** (dBsource / 10) * 1e-12
            intensity = intensity_source / (4 * np.pi * distances ** 2)
            Z += intensity

        Z = 10 * np.log10(Z / 1e-12)

        self.LAT, self.LON, self.Z = LAT, LON, Z  # Store the values in the object's attributes


    def plot_noise_map(self, observation_points: List[Tuple[float, float]]):
        """Plots the noise map with wind turbines and observation points."""
        plt.figure(figsize=(10, 6))
        
        # Define contour levels starting from 35 dB
        contour_levels = [35, 40, 45, 50, 55, 60]
        
        plt.contourf(self.LON, self.LAT, self.Z, levels=contour_levels, cmap='RdYlBu_r')
        plt.colorbar(label='Noise Level (dB)')
        plt.title('Wind Turbine Noise Contours')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        # Plot wind turbines
        for turbine in self.wind_turbines:
            plt.plot(*turbine['position'][::-1], 'ko')  # Plot longitude before latitude

        # Plot observation points
        for point in observation_points:
            plt.plot(*point[::-1], 'ro')

        plt.grid(True)
        plt.show()

    def generate_and_plot_noise_map(self, observation_points: List[Tuple[float, float]]):
        """
        Generates and plots a noise map for the wind turbines and observation points based on the desired noise level.
        Allows the user to choose the wind speed using a cursor.
        """

        # Create a wind speed slider for user interaction
        wind_speed_slider = widgets.FloatSlider(
            value=7.0,
            min=3.0,
            max=12.0,
            step=1.0,
            description='Wind Speed (m/s):',
            continuous_update=False
        )

        @widgets.interact(wind_speed=wind_speed_slider)
        def interactive_plot(wind_speed):
            # Get the noise predictions for the given wind speed
            noise_predictions = self.wind_turbine_model.predict_noise_at_wind_speed(self.wind_turbines, wind_speed)

            # Calculate the combined noise level at each observation point
            noise_levels = self.superpose_several_wind_turbine_sounds(observation_points, wind_speed)

            # Generate the noise map data
            lat_min = min(turbine['position'][0] for turbine in self.wind_turbines)
            lat_max = max(turbine['position'][0] for turbine in self.wind_turbines)
            lon_min = min(turbine['position'][1] for turbine in self.wind_turbines)
            lon_max = max(turbine['position'][1] for turbine in self.wind_turbines)
            delay = 1/250

            # Adjust the map size to include observation points
            for point in observation_points:
                lat_min = min(lat_min, point[0]) - delay
                lat_max = max(lat_max, point[0]) + delay
                lon_min = min(lon_min, point[1]) - delay
                lon_max = max(lon_max, point[1]) + delay

            LAT, LON = np.meshgrid(np.linspace(lat_min, lat_max, 100), np.linspace(lon_min, lon_max, 100))
            Z = np.zeros_like(LAT)

            for turbine in self.wind_turbines:
                distances = np.vectorize(lambda lat, lon: haversine((lat, lon), turbine['position'], unit=Unit.METERS))(LAT, LON)

                # Use the predicted noise level for the turbine at the given wind speed
                turbine_name = turbine.get('name')
                dBsource = noise_predictions[turbine_name][f"{wind_speed:.1f}m/s"]

                intensity_source = 10 ** (dBsource / 10) * 1e-12
                intensity = intensity_source / (4 * np.pi * distances ** 2)
                Z += intensity

            Z = 10 * np.log10(Z / 1e-12)

            # Plot the noise map
            plt.figure(figsize=(10, 6))
            contour_levels = np.arange(35, 60, 5)  # Define contour levels starting from 35 dB
            plt.contourf(LON, LAT, Z, levels=contour_levels, cmap='RdYlBu_r')
            plt.colorbar(label='Noise Level (dB)')
            plt.title('Wind Turbine Noise Contours')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')

            # Plot wind turbines
            for turbine in self.wind_turbines:
                plt.plot(*turbine['position'][::-1], 'ko')  # Plot longitude before latitude

            # Plot observation points
            for point in observation_points:
                plt.plot(*point[::-1], 'ro')

            plt.grid(True)
            plt.show()

        return interactive_plot

        
    
    def display_turbines_on_map(self, observation_points: List[Tuple[float, float]] = []):
        """Displays the wind turbines and observation points on a real map using their latitude and longitude."""

        # Get the average latitude and longitude to center the map
        avg_lat = sum(turbine['position'][0] for turbine in self.wind_turbines) / len(self.wind_turbines)
        avg_lon = sum(turbine['position'][1] for turbine in self.wind_turbines) / len(self.wind_turbines)

        # Create a folium map centered at the average latitude and longitude
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

        # Add markers for each wind turbine with a noise level of 105 dB
        for turbine in self.wind_turbines:
            folium.Marker(
                location=turbine['position'],
                tooltip=f"Noise Level: 105 dB",  # Set noise level to 105 dB
                icon=folium.Icon(icon="cloud")
            ).add_to(m)

        # Add markers for the observation points
        for observation_point in observation_points:
            folium.Marker(
                location=observation_point,
                tooltip="Observation Point",
                icon=folium.Icon(icon="star", color="red")
            ).add_to(m)

        # Display the map
        display(m)



    def calculate_noise_at_position(self, observation_point: Tuple[float, float]) -> float:
        """Calculates the noise level at a given observation point."""
        return self.superpose_several_wind_turbine_sounds(observation_point)




    def get_landuse_between_points(self, turbine_name, observation_point):
        # Find the position of the turbine by its name
        turbine_position = next((turbine["position"] for turbine in self.wind_turbines if turbine["name"] == turbine_name), None)
        if not turbine_position:
            raise ValueError(f"No turbine found with the name {turbine_name}")

        # Get the bounding box coordinates
        north, south = max(turbine_position[0], observation_point[0]), min(turbine_position[0], observation_point[0])
        east, west = max(turbine_position[1], observation_point[1]), min(turbine_position[1], observation_point[1])

        # Fetch land use data
        landuse = ox.features_from_bbox(north, south, east, west, tags={'landuse': True})
        return landuse


    def plot_landuse_for_turbines_and_observation_points(self, wind_turbines, observation_points):
        # Extract turbine positions and combine with observation points
        turbine_positions = [turbine["position"] for turbine in wind_turbines]
        all_points = turbine_positions + observation_points

        # Create GeoDataFrame for all points
        gdf_points = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lat, lon in all_points], crs="EPSG:4326")

        # Extract latitude and longitude for bounding box calculation
        all_lats, all_lons = zip(*all_points)
        margin = 0.01
        bbox = dict(north=max(all_lats) + margin, south=min(all_lats) - margin, east=max(all_lons) + margin, west=min(all_lons) - margin)

        # Fetch landuse features and create a color map for unique landuse types
        landuse = ox.features_from_bbox(**bbox, tags={'landuse': True})
        landuse_types = landuse['landuse'].unique()
        cmap = plt.get_cmap('tab20', len(landuse_types))
        color_map = {landuse_type: cmap(i) for i, landuse_type in enumerate(landuse_types)}

        # Plot landuse, turbines, and observation points
        ax = landuse.to_crs(epsg=3857).plot(color=landuse['landuse'].map(color_map), figsize=(12, 12))
        gdf_points.iloc[:len(turbine_positions)].to_crs(epsg=3857).plot(ax=ax, color='blue', markersize=100, label='Éoliennes')
        gdf_points.iloc[len(turbine_positions):].to_crs(epsg=3857).plot(ax=ax, color='red', markersize=100, label='Points d\'observation')

        # Plot lines between turbine positions and observation points
        lines = [LineString([turbine[::-1], observation[::-1]]) for turbine in turbine_positions for observation in observation_points]
        gpd.GeoSeries(lines, crs="EPSG:4326").to_crs(epsg=3857).plot(ax=ax, color='black', linewidth=1)

        # Create legend
        patches = [plt.Line2D([0], [0], marker='o', color='w', label=landuse_type, markersize=10, markerfacecolor=color_map[landuse_type]) for landuse_type in landuse_types]
        ax.legend(handles=patches + [plt.Line2D([0], [0], marker='o', color='blue', label='Éoliennes', markersize=10),
                                    plt.Line2D([0], [0], marker='o', color='red', label='Points d\'observation', markersize=10)],
                  title="Légende")
        plt.show()


    def get_altitudes_between_points(self, point1, point2, num_samples=100):
        latitudes = np.linspace(point1[0], point2[0], num_samples)
        longitudes = np.linspace(point1[1], point2[1], num_samples)

        locations = "|".join([f"{lat},{lon}" for lat, lon in zip(latitudes, longitudes)])

        url = f"https://maps.googleapis.com/maps/api/elevation/json?locations={locations}&key={GOOGLE_ELEVATION_API_KEY}"
        response = requests.get(url)
        data = response.json()

        if data['status'] != 'OK':
            raise Exception(f"Erreur lors de l'appel à l'API: {data['status']}")

        altitudes = [result['elevation'] for result in data['results']]

        return altitudes


    def plot_relief_between_points(self, wind_turbines, observation_points):
        for turbine in wind_turbines:
            turbine_position = turbine["position"]
            for idx, observation_point in enumerate(observation_points):
                altitudes = self.get_altitudes_between_points(turbine_position, observation_point)
                plt.figure(figsize=(5, 3))
                plt.plot(altitudes)
                plt.title(f"Relief entre {turbine['name']} et Point d'observation {idx + 1}")
                plt.xlabel("Points intermédiaires")
                plt.ylabel("Altitude (m)")
                plt.show()
                
                

    def superpose_several_wind_turbine_sounds2(self, observation_points: List[Tuple[float, float]]) -> Dict[Tuple[float, float], float]:
            """Superposes sounds from multiple wind turbines at specific observation points."""

            results = {}

            for observation_point in observation_points:
                total_intensity_level = 0
                for turbine in self.wind_turbines:
                    dBsource = turbine['noise_level']
                    distance = haversine(observation_point, turbine['position'], unit=Unit.METERS)
                    intensity_level = self.calculate_sound_intensity_level(dBsource, distance)
                    total_intensity_level += intensity_level
                dB_total = self.convert_intensity_level_into_dB(total_intensity_level)
                results[observation_point] = dB_total

            return results