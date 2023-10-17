#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
from typing import Dict,List
from windspeed import WindModule
from windturbinemodel import WindTurbineModel
from noisemap_geo_coord import NoiseMap


class WindFarmAnalysis:
    """
    Provides methods to analyze a wind farm, including noise predictions and wind data retrieval.

    Attributes:
        wind_turbines (list): List of wind turbines with their specifications.
        wind_turbine_model (WindTurbineModel): Model to predict noise levels.
        display_outputs (bool): Whether to display outputs.
    """
    
    def __init__(self, wind_turbines, retrain_model=False, display_outputs=True):
        """
        Initialize the WindFarmAnalysis class.

        Parameters:
            wind_turbines (list): List of wind turbines with their specifications.
            retrain_model (bool, optional): Whether to retrain the noise prediction model. Defaults to False.
            display_outputs (bool, optional): Whether to display outputs. Defaults to True.
        """
        
        self.wind_turbines = wind_turbines
        self.wind_turbine_model = WindTurbineModel(retrain=retrain_model)
        self.display_outputs = display_outputs

    def process_turbines(self):
        """
        Processes each turbine in the wind farm, predicting noise levels and retrieving wind data.
        """
        for turbine in self.wind_turbines:
            self._predict_noise(turbine)
            self._retrieve_wind_data(turbine)

    def generate_noise_map(self):
        noise_map = NoiseMap(self.wind_turbines, self.wind_turbine_model)
        noise_map.generate_noise_map()

        if self.display_outputs:
            noise_map.plot_noise_map()
            noise_map.display_turbines_on_map()

    def noise_at_location(self, position):
        noise_map = NoiseMap(self.wind_turbines, self.wind_turbine_model)
        return noise_map.calculate_noise_at_position(position)

    def _predict_noise(self, turbine):
        print(f"Processing {turbine['name']}...")
        noise_prediction = self.wind_turbine_model.predict_noise(turbine['power'], turbine['diameter'], turbine['hub_height'])
        
        if self.display_outputs:
            print(f"Noise prediction for {turbine['name']}:\n")
            print(noise_prediction)
            print("------")
            self.wind_turbine_model.plot_noise_prediction(turbine['power'], turbine['diameter'], turbine['hub_height'])

    def _retrieve_wind_data(self, turbine):
        latitude, longitude = turbine['position']
        wind_module = WindModule(latitude=latitude, longitude=longitude)
        table, mean_wind_speed = wind_module.calculate_statistics()
        
        if self.display_outputs:
            print(f"Mean wind speed for {turbine['name']}: {mean_wind_speed:.2f} m/s")
            print("Statistics table:")
            print(table)
            print("------")
            wind_module.create_wind_rose()

    def analyze_wind_farm(self, position):
        self.process_turbines()
        self.generate_noise_map()
        noise_level = self.noise_at_location(position)
        return noise_level