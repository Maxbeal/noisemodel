from pathlib import Path
import os
import json

__version__ = (0, 0, 1)
__all__ = (
    "__version__",
    "DATA_DIR",
    "WindTurbines",
    "WindSpeed",
    "NoiseMap",
    "NoiseAnalysis",
)

DATA_DIR = Path(__file__).resolve().parent / "data"


from .windturbines import WindTurbines
from .windspeed import WindSpeed
from .noisemap import NoiseMap
from .noiseanalysis import NoiseAnalysis
