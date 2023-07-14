import math
import numpy as np
import matplotlib.pyplot as plt

class NoisePoint:
    W0 = 10 ** (-12)  # reference value for sound power [W]
    I0 = 10 ** (-12)  # reference value for sound intensity [W/m2]

    def __init__(self):
        self.alpha = float(input("Enter the air absorption coefficient alpha in [dB/km]: "))
        self.alpha /= 1000  # Convert alpha from dB/km to dB/m
    #...


    def calculate_sound_intensity_level(self, dBsource, distance):
        wattsource = (10 ** (dBsource / 10)) * self.W0
        intensity_level = wattsource / (4 * math.pi * distance ** 2)
        # apply air absorption
        intensity_level = 10 ** (-self.alpha * distance / 10) * intensity_level
        return intensity_level

    def convert_intensity_level_into_dB(self, intensity_level):
        return 10 * math.log10(intensity_level / self.I0)

    def superpose_several_wind_turbine_sounds(self, num_sources):
        total_intensity_level = 0
        for i in range(num_sources):
            dBsource = float(input(f"Enter the power of source {i + 1} in dB: "))
            distance = float(input(f"Enter the distance from source {i + 1} in meters: "))
            intensity_level = self.calculate_sound_intensity_level(dBsource, distance)
            print(f"Wind turbine {i + 1} has intensity_level {intensity_level} W/m2 at a distance {distance} m")
            total_intensity_level += intensity_level
        dB_total = self.convert_intensity_level_into_dB(total_intensity_level)
        print(f"The total_intensity_level is {total_intensity_level} W/m2 or {dB_total} dB")
        return dB_total


class NoiseMap:
    W0 = 10 ** (-12)  # reference value for sound power [W]
    I0 = 10 ** (-12)  # reference value for sound intensity [W/m2]

    def __init__(self, num_turbines):
        self.alpha = float(input("Enter the air absorption coefficient alpha [dB/km]: "))
        self.alpha /= 1000  # Convert alpha from dB/km to dB/m
        self.wind_turbines = []
        for i in range(num_turbines):
            x = float(input(f"Enter the x-coordinate of wind turbine {i+1} (in meters): "))
            y = float(input(f"Enter the y-coordinate of wind turbine {i+1} (in meters): "))
            noise_level = float(input(f"Enter the noise level at source of wind turbine {i+1} in dB: "))
            self.wind_turbines.append({'position': (x, y), 'noise_level': noise_level})


    def calculate_sound_intensity_level(self, dBsource, distance):
        if distance == 0:
            return float('inf')  # return infinity if distance is zero
        wattsource = (10 ** (dBsource / 10)) * self.W0
        intensity_level = wattsource / (4 * math.pi * distance ** 2)
        # apply air absorption
        intensity_level = 10 ** (-self.alpha * distance / 10) * intensity_level
        return intensity_level

    def convert_intensity_level_into_dB(self, intensity_level):
        return 10 * math.log10(intensity_level / self.I0)

    def superpose_several_wind_turbine_sounds(self):
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

    def generate_noise_map(self, listening_point=None): #generate mao
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
            dx = X - turbine['position'][0]
            dy = Y - turbine['position'][1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            mask = distance >= 1 #mask = distance >= 1 is used to create a boolean mask that identifies locations at least 1 meter away from the sound source, avoiding issues with high sound intensity for points too close to the source.
            #apply geometrical absorption
            intensity_source = 10 ** (turbine['noise_level'] / 10) * 1e-12
            intensity = np.where(mask & (distance != 0), intensity_source / (4 * np.pi * distance ** 2), 0) #mask variable is used to selectively apply the sound intensity calculation to elements where the distance is both greater than or equal to 1 and not zero, while setting the intensity to zero for all other elements.
            # apply air absorption
            intensity = 10 ** (-self.alpha * distance / 10) * intensity
            Z += intensity

        Z = 10 * np.log10(Z / 1e-12)

        self.X, self.Y, self.Z = X, Y, Z

    def plot_noise_map(self):
        # matplotlib library
        plt.figure(figsize=(10, 6))
        plt.contourf(self.X, self.Y, self.Z, levels=20, cmap='RdYlBu_r')
        plt.colorbar(label='Noise Level (dB)')
        plt.title('Wind Turbine Noise Contours')
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')

        for turbine in self.wind_turbines:
            plt.plot(*turbine['position'], 'ko')

        plt.grid(True)
        plt.show()

    def get_noise_level_at_point(self, x, y):
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

    def plot_noise_map(self, listening_point=None): #listening_point parameter is optional
        plt.figure(figsize=(10, 6))
        plt.contourf(self.X, self.Y, self.Z, levels=20, cmap='RdYlBu_r') #creates contour lines at 20 differents levels
        plt.colorbar(label='Noise Level (dB)')
        plt.title('Wind Turbine Noise Contours')
        plt.xlabel('x (meters)')
        plt.ylabel('y (meters)')

        for turbine in self.wind_turbines:
            plt.plot(*turbine['position'], 'ko')

        if listening_point:
            plt.plot(*listening_point, 'ro')

        plt.grid(True)
        plt.show()


# Ask the user which class they want to use
choice = input("Would you like to use 'NoisePoint' or 'NoiseMap'? ")

if choice.lower() == 'noisepoint':
    noise_point = NoisePoint()
    num_sources = int(input("Enter the number of sound sources: "))
    noise_point.superpose_several_wind_turbine_sounds(num_sources)

elif choice.lower() == 'noisemap':
    num_turbines = int(input("Enter the number of wind turbines: "))
    noise_map = NoiseMap(num_turbines)
    noise_map.superpose_several_wind_turbine_sounds()
    x = float(input("Enter the x-coordinate of the listening point (in meters): "))
    y = float(input("Enter the y-coordinate of the listening point (in meters): "))
    noise_map.generate_noise_map(listening_point=(x, y))
    dB = noise_map.get_noise_level_at_point(x, y)
    print(f"The noise level at the listening point ({x}, {y}) is {dB} dB")
    noise_map.plot_noise_map(listening_point=(x, y))

else:
    print("Error: Invalid choice. Please choose 'NoisePoint' or 'NoiseMap'.")
