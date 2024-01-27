# windwhisper

``windwhisper`` is a Python package for estimating wind turbine 
noise propagation and its impacts on surrounding populations.

## Installation

As ``windwhisper`` is being actively developed, 
it is best to install from Github using ``pip``:

```bash
  
  pip git+https://github.com/Maxbeal/noisemodel.git
```

## Usage

``windwhisper`` can be used to estimate the noise propagation from wind turbines. 
The following example shows how to estimate the noise propagation from a series of
wind turbines, and the exposure of a number of listeners to noise, expressed using
the L_den indicator. Results cna be exported on a map.

```python

    from windwhisper import windturbines
    wind_turbines = {   
        "Turbine 1": {"power": 2.5e3, "diameter": 100, "hub height": 80, "position": (47.5, 8.55)},
        "Turbine 2": {"power": 3.0e3, "diameter": 105, "hub height": 85, "position": (47.3869, 8.5517)},
        "Turbine 3": {"power": 3.5e3, "diameter": 110, "hub height": 90, "position": (47.3969, 8.5617)},
        "Turbine 4": {"power": 4.0e3, "diameter": 115, "hub height": 95, "position": (47.3869, 8.5317)},
        "Turbine 5": {"power": 4.0e3, "diameter": 120, "hub height": 95, "position": (47.34955801547433, 8.491580864126439)},
    }
    
    listening_points = {
        "Listener 1": {"position": (47.3769, 8.5517)},
        "Listener 2": {"position": (47.3869, 8.552)},
        "Listener 3": {"position": (47.3461, 8.5175)},
    }
    
    
    wt = windturbines.WindTurbines(
        wind_turbines=wind_turbines,
        listeners=listening_points,
    )
    
    wt.analyze_noise()
    wt.noise_analysis.display_listeners_on_map_with_Lden()
```

## License

``windwhisper`` is distributed under the terms of the BSD 3-Clause license.

## Authors

* Maxime Balandret
* Romain Sacchi (romain.sacchi@psi.ch)