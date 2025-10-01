"""
SpATS: Spatial Analysis of Field Trials with Splines

A Python implementation for analyzing field trial experiments using 
two-dimensional Penalised splines (P-splines).
"""

from .core import SpATS
from .control import SpATSControl
from .plotting import plot_spats, plot_variogram
from .variogram import variogram
from .utils import SAP, PSANOVA, get_heritability

__version__ = "0.1"
__author__ = "Python SpATS Implementation"

__all__ = [
    "SpATS",
    "SpATSControl",
    "plot_spats",
    "plot_variogram",
    "variogram",
    "SAP",
    "PSANOVA",
    "get_heritability",
]