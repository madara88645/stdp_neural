"""
Configuration management and utility functions for STDP simulations.

This module provides configuration classes and utility functions to help
manage simulation parameters and settings.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import numpy as np


@dataclass
class NeuronConfig:
    """Configuration for neuron parameters."""
    v_rest: float = 0.0
    v_thresh: float = 1.0
    v_reset: float = 0.0
    tau_m: float = 20.0
    refractory_period: int = 5
    dt: float = 1.0


@dataclass  
class STDPConfig:
    """Configuration for STDP parameters."""
    initial_weight: float = 0.5
    A_plus: float = 0.02
    A_minus: float = -0.025
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    w_min: float = 0.0
    w_max: float = 1.0
    time_window: float = 1.0  # For simple STDP


@dataclass
class StimulusConfig:
    """Configuration for stimulus/input parameters."""
    rate: float = 10.0  # Hz
    duration: float = 200.0  # ms
    dt: float = 1.0  # ms
    seed: Optional[int] = None


@dataclass
class SimulationConfig:
    """Master configuration for entire simulation."""
    neuron: NeuronConfig
    stdp: STDPConfig  
    stimulus: StimulusConfig
    verbose: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SimulationConfig':
        """Create configuration from dictionary."""
        return cls(
            neuron=NeuronConfig(**config_dict.get('neuron', {})),
            stdp=STDPConfig(**config_dict.get('stdp', {})),
            stimulus=StimulusConfig(**config_dict.get('stimulus', {})),
            verbose=config_dict.get('verbose', False)
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'SimulationConfig':
        """Load configuration from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'neuron': {
                'v_rest': self.neuron.v_rest,
                'v_thresh': self.neuron.v_thresh,
                'v_reset': self.neuron.v_reset,
                'tau_m': self.neuron.tau_m,
                'refractory_period': self.neuron.refractory_period,
                'dt': self.neuron.dt
            },
            'stdp': {
                'initial_weight': self.stdp.initial_weight,
                'A_plus': self.stdp.A_plus,
                'A_minus': self.stdp.A_minus,
                'tau_plus': self.stdp.tau_plus,
                'tau_minus': self.stdp.tau_minus,
                'w_min': self.stdp.w_min,
                'w_max': self.stdp.w_max,
                'time_window': self.stdp.time_window
            },
            'stimulus': {
                'rate': self.stimulus.rate,
                'duration': self.stimulus.duration,
                'dt': self.stimulus.dt,
                'seed': self.stimulus.seed
            },
            'verbose': self.verbose
        }
    
    def save_json(self, json_path: str) -> None:
        """Save configuration to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def calculate_firing_rate(spike_times: np.ndarray, duration: float) -> float:
    """
    Calculate average firing rate from spike times.
    
    Args:
        spike_times: Array of spike times (ms)
        duration: Total duration (ms)
        
    Returns:
        Firing rate in Hz
    """
    return len(spike_times) / (duration / 1000.0)


def calculate_isi_stats(spike_times: np.ndarray) -> Dict[str, float]:
    """
    Calculate inter-spike interval statistics.
    
    Args:
        spike_times: Array of spike times (ms)
        
    Returns:
        Dictionary with ISI statistics
    """
    if len(spike_times) < 2:
        return {'mean_isi': np.nan, 'std_isi': np.nan, 'cv_isi': np.nan}
    
    isis = np.diff(spike_times)
    mean_isi = np.mean(isis)
    std_isi = np.std(isis)
    cv_isi = std_isi / mean_isi if mean_isi > 0 else np.nan
    
    return {
        'mean_isi': mean_isi,
        'std_isi': std_isi,
        'cv_isi': cv_isi
    }


def analyze_weight_changes(
    weight_history: np.ndarray, 
    threshold: float = 0.001
) -> Dict[str, int]:
    """
    Analyze synaptic weight changes during simulation.
    
    Args:
        weight_history: Array of synaptic weights over time
        threshold: Minimum change magnitude to count
        
    Returns:
        Dictionary with weight change statistics
    """
    weight_changes = np.diff(weight_history)
    
    significant_changes = np.abs(weight_changes) > threshold
    ltp_events = np.sum(weight_changes > threshold)
    ltd_events = np.sum(weight_changes < -threshold)
    
    return {
        'total_changes': np.sum(significant_changes),
        'ltp_events': ltp_events,
        'ltd_events': ltd_events,
        'net_change': weight_history[-1] - weight_history[0]
    }


def create_default_configs() -> Dict[str, SimulationConfig]:
    """
    Create a set of default simulation configurations for common scenarios.
    
    Returns:
        Dictionary of named configuration presets
    """
    configs = {}
    
    # Basic STDP demonstration
    configs['basic'] = SimulationConfig(
        neuron=NeuronConfig(),
        stdp=STDPConfig(),
        stimulus=StimulusConfig(rate=10.0, duration=200.0)
    )
    
    # High-frequency stimulation
    configs['high_freq'] = SimulationConfig(
        neuron=NeuronConfig(),
        stdp=STDPConfig(A_plus=0.01, A_minus=-0.01),
        stimulus=StimulusConfig(rate=50.0, duration=100.0)
    )
    
    # Long-term plasticity study
    configs['long_term'] = SimulationConfig(
        neuron=NeuronConfig(),
        stdp=STDPConfig(A_plus=0.005, A_minus=-0.005),
        stimulus=StimulusConfig(rate=5.0, duration=1000.0)
    )
    
    # Weak plasticity (hard to detect changes)
    configs['weak_plasticity'] = SimulationConfig(
        neuron=NeuronConfig(),
        stdp=STDPConfig(A_plus=0.001, A_minus=-0.001),
        stimulus=StimulusConfig(rate=20.0, duration=500.0)
    )
    
    return configs


def print_simulation_summary(results: Dict[str, Any]) -> None:
    """
    Print a summary of simulation results.
    
    Args:
        results: Dictionary containing simulation results
    """
    duration = results['duration']
    pre_spikes = results['pre_spikes']
    post_spikes = results['post_spikes']
    weights = results['weights']
    
    # Calculate statistics
    pre_rate = calculate_firing_rate(results['time'][pre_spikes], duration)
    post_rate = calculate_firing_rate(results['time'][post_spikes], duration)
    weight_stats = analyze_weight_changes(weights)
    
    print("=" * 50)
    print("STDP SIMULATION SUMMARY")
    print("=" * 50)
    print(f"Duration: {duration:.1f} ms")
    print(f"Time step: {results['dt']:.1f} ms")
    print()
    print("Spike Statistics:")
    print(f"  Pre-synaptic spikes: {np.sum(pre_spikes)}")
    print(f"  Post-synaptic spikes: {np.sum(post_spikes)}")
    print(f"  Pre-synaptic rate: {pre_rate:.2f} Hz")
    print(f"  Post-synaptic rate: {post_rate:.2f} Hz")
    print()
    print("Weight Changes:")
    print(f"  Initial weight: {weights[0]:.3f}")
    print(f"  Final weight: {weights[-1]:.3f}")
    print(f"  Net change: {weight_stats['net_change']:.3f}")
    print(f"  LTP events: {weight_stats['ltp_events']}")
    print(f"  LTD events: {weight_stats['ltd_events']}")
    print("=" * 50)
