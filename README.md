# windwhisper

``windwhisper`` is a Python package for estimating wind turbine 
noise propagation and its impacts on surrounding populations.

## Installation

As ``windwhisper`` is being actively developed, 
it is best to install from Github using ``pip``:

```bash
  pip install git+https://github.com/Maxbeal/noisemodel.git
```

## Usage

``windwhisper`` can be used to estimate the noise propagation from wind turbines. 
The following example shows how to estimate the noise propagation from a series of
wind turbines, and the exposure of a number of listeners to noise, expressed using
the L_den indicator. Results can be exported on a map.


### Initializing wind turbines and listeners:

```python

    from windturbines import WindTurbines

    wind_turbines_data = {
        "Turbine1": {
            "power": 3000,
            "diameter": 100,
            "hub height": 120,
            "position": (47.5, 8.2)  # Latitude and Longitude
        }
    }
    
    listeners = {
        "Listener1": {
            "position": (47.5, 8.25)
        }
    }
    
    wt = WindTurbines(
        wind_turbines=wind_turbines_data,
        listeners=listeners,
        retrain_model=False  # Set to True to retrain the noise model
    )

```

### Fetching and Analyzing Noise Levels

```python
    wt.fetch_noise_level_vs_wind_speed()
    wt.plot_noise_curve()  # Visualize noise levels for different wind speeds

```
### Noise Map Generation

```python
    wt.fetch_noise_map()
```

### Noise Impact Analysis

```python
    wt.analyze_noise()
```




## License

``windwhisper`` is distributed under the terms of the BSD 3-Clause license (see LICENSE).

## Authors

* Maxime Balandret, Paul Scherrer Institut (PSI)
* Romain Sacchi (romain.sacchi@psi.ch), Paul Scherrer Institut (PSI)

## Acknowledgements
The development of `windwhisper` is supported by the European project
[WIMBY](https://cordis.europa.eu/project/id/101083460) (Wind In My BackYard, grant agreement No 101083460).
