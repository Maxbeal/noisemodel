from pathlib import Path

__version__ = (0, 0, 1)
__all__ = ("__version__", "DATA_DIR", "WindTurbineModel", "WindModule", "NoiseMap", "WindNoiseAnalysis")

DATA_DIR = Path(__file__).resolve().parent / "data"

from .windturbinemodel import WindTurbineModel
from .windspeed import WindModule
from .noisemap_geo_coord import NoiseMap
from .windnoiseanalysis import WindNoiseAnalysis
