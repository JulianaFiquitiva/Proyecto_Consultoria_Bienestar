"""Módulo de métodos estadísticos."""

from agent.statistics.descriptive import DescriptiveStats
from agent.statistics.inference import InferenceEngine
from agent.statistics.regression import RegressionEngine

__all__ = ["DescriptiveStats", "InferenceEngine", "RegressionEngine"]
