"""
STDP Neural Plasticity Simulation

A comprehensive Python library for simulating Spike-Timing-Dependent Plasticity (STDP) 
in neural networks.
"""

from .core.neuron import LeakyIntegrateFireNeuron
from .core.stdp import STDPRule, ExponentialSTDP, SimpleSTDP
from .visualization.plotter import STDPPlotter

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "LeakyIntegrateFireNeuron",
    "STDPRule", 
    "ExponentialSTDP",
    "SimpleSTDP",
    "STDPPlotter",
]
