"""
SpATS: Spatial Analysis of Field Trials with Splines

A Python implementation for analyzing field trial experiments using
two-dimensional Penalised splines (P-splines).
"""

from .core import SpATS, fit_trial
from .model_basis import SpatialSpec
from .engine import ConvergenceWarning
from .control import SpATSControl
from .batch import fit_trials, TrialFit
from .plotting import plot_spats, plot_variogram
from .variogram import variogram
from .utils import SAP, PSANOVA, get_heritability

__version__ = "0.3.0"
__author__ = "Python SpATS Implementation"

__all__ = [
    "SpATS",
    "fit_trial",
    "fit_trials",
    "TrialFit",
    "SpatialSpec",
    "ConvergenceWarning",
    "SpATSControl",
    "plot_spats",
    "plot_variogram",
    "variogram",
    "SAP",
    "PSANOVA",
    "get_heritability",
]
