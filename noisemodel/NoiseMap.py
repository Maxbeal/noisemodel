import numpy as np
import math
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import WindTurbineModel


class NoiseMap:
    """Handles the creation and manipulation of a noise map for multiple wind turbines.

    Attributes:
        alpha: Air absorption coefficient in dB/m.
        wind_turbines: A list of wind turbines.
        W0: Reference value for sound power.
        I0: Reference value for sound intensity.
    """

    def __init__(self, wind_turbines: List[Dict], wind_turbine_model: WindTurbineModel, alpha: float = 2.0):
        self.alpha = alpha / 1000  # Convert alpha from dB/km to dB/m
        self.W0 = 1e-12
        self.I0 = 1e-12
        new_wind_turbines = []  # Nouvelle liste pour stocker les turbines avec les niveaux de bruit
        for turbine in wind_turbines:
            x, y = turbine['position']
            power = turbine['power']
            diameter = turbine['diameter']
            hub_height = turbine['hub_height']
            wind_speed = turbine['wind_speed']

            if wind_speed < 3 or wind_speed > 12:
                print("Invalid wind speed. Please enter an integer value between 3 and 12 m/s.")
                return

            noise_prediction = wind_turbine_model.predict_noise(power, diameter, hub_height)
            noise_level = noise_prediction.iloc[0, wind_speed - 3]

            new_wind_turbines.append({'position': (x, y), 'noise_level': noise_level})  # Ajouter à la nouvelle liste

        self.wind_turbines = new_wind_turbines  # Affecter la nouvelle liste à self.wind_turbines

    def calculate_sound_intensity_level(self, dBsource: float, distance: float) -> float:
        """Calculates sound intensity level at a given distance from the source.

        :param dBsource: Sound intensity level at the source in dB.
        :param distance: Distance from the source in meters.
        :return: Sound intensity level at the given distance.
        """
        if distance == 0:
            return float('inf')  # return infinity if distance is zero
        wattsource = (10 ** (dBsource / 10)) * self.W0
        intensity_level = wattsource / (4 * math.pi * distance ** 2)
        # apply air absorption
        intensity_level = 10 ** (-self.alpha * distance / 10) * intensity_level
        return intensity_level

    def convert_intensity_level_into_dB(self, intensity_level: float) -> float:
        """Converts a sound intensity level into decibels.

        :param intensity_level: Sound intensity level.
        :return: Equivalent value in decibels.
        """
        return 10 * math.log10(intensity_level / self.I0)

    def superpose_several_wind_turbine_sounds(self) -> float:
        """Superposes sounds from multiple wind turbines.

        :return: Total sound level in decibels.
        """
        total_intensity_level = 0
        for turbine in self.wind_turbines:
            dBsource = turbine['noise_level']
            distance = np.linalg.norm(np.array(turbine['position']))
            intensity_level = self.calculate_sound_intensity_level(dBsource, distance)
            print(
                f"Wind turbine at {turbine['position']} has intensity_level {intensity_level} W/m2 at a distance {distance} m")
            total_intensity_level += intensity_level
        dB_total = self.convert_intensity_level_into_dB(total_intensity_level)
        print(f"The total_intensity_level is {total_intensity_level} W/m2 or {dB_total} dB")
        return dB_total

    def generate_noise_map(self, listening_point: Optional[Tuple[float, float]] = None):
        """Generates a noise map for the wind turbines."""

        x_min = min(turbine['position'][0] for turbine in self.wind_turbines)
        x_max = max(turbine['position'][0] for turbine in self.wind_turbines)
        y_min = min(turbine['position'][1] for turbine in self.wind_turbines)
        y_max = max(turbine['position'][1] for turbine in self.wind_turbines)

        # adapt size of the map to the listening point
        if listening_point:
            x_min = min(x_min, listening_point[0]) - 100
            x_max = max(x_max, listening_point[0]) + 100
            y_min = min(y_min, listening_point[1]) - 100
            y_max = max(y_max, listening_point[1]) + 100
        else:
            x_min -= 100
            x_max += 100
            y_min -= 100
            y_max += 100

        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        for turbine in self.wind_turbines:
            dx = X - turbine['position'][0]  # Correction ici
            dy = Y - turbine['position'][1]  # Correction ici
            distance = np.sqrt(dx ** 2 + dy ** 2)
            mask = distance >= 1
            intensity_source = 10 ** (turbine['noise_level'] / 10) * 1e-12
            intensity = np.where(mask, intensity_source / (4 * np.pi * distance ** 2), 0)
            intensity = 10 ** (-self.alpha * distance / 10) * intensity
            Z += intensity

        Z = 10 * np.log10(Z / 1e-12)

        self.X, self.Y, self.Z = X, Y, Z  # Store the values in the object's attributes

    def plot_noise_map(self, listening_point: Optional[Tuple[float, float]] = None):
        """Plots the noise map.

        :param listening_point: Optional listening point coordinates (x, y).
        """
        plt.figure(figsize=(10, 6))
        plt.contourf(self.X, self.Y, self.Z, levels=20, cmap='RdYlBu_r')
        plt.colorbar(label='Noise Level (dB)')
        plt.title('Wind Turbine Noise Contours')
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')

        for turbine in self.wind_turbines:
            plt.plot(*turbine['position'], 'ko')

        if listening_point:  # Plot listening point if provided
            plt.plot(*listening_point, 'ro')

        plt.grid(True)
        plt.show()

    def get_noise_level_at_point(self, x: float, y: float) -> float:
        """Retrieves the noise level at a specific point on the map.

        :param x: X coordinate of the point.
        :param y: Y coordinate of the point.
        :return: Noise level at the given point in decibels.
        """
        total_intensity = 0
        for turbine in self.wind_turbines:
            dx = x - turbine['position'][0]
            dy = y - turbine['position'][1]
            distance = math.sqrt(dx ** 2 + dy ** 2)
            intensity_source = 10 ** (turbine['noise_level'] / 10) * 1e-12
            intensity = intensity_source / (4 * math.pi * distance ** 2) if distance >= 1 else 0
            total_intensity += intensity
        total_dB = 10 * math.log10(total_intensity / self.I0)
        return total_dB