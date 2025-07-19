"""
Neuron models for STDP simulation.

This module contains implementations of various neuron models used in 
spike-timing-dependent plasticity simulations.
"""

from typing import Optional, Tuple
import numpy as np


class LeakyIntegrateFireNeuron:
    """
    Leaky Integrate-and-Fire neuron model with STDP capabilities.
    
    This model implements a simple but biologically plausible neuron that:
    1. Integrates incoming synaptic currents
    2. Exhibits exponential membrane potential decay (leak)
    3. Fires spikes when threshold is reached
    4. Implements refractory period after spiking
    
    Attributes:
        v_rest (float): Resting membrane potential (mV)
        v_thresh (float): Spike threshold (mV)
        v_reset (float): Reset potential after spike (mV)
        tau_m (float): Membrane time constant (ms)
        refractory_period (int): Refractory period duration (time steps)
        dt (float): Time step size (ms)
    """
    
    def __init__(
        self,
        v_rest: float = 0.0,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
        tau_m: float = 20.0,
        refractory_period: int = 5,
        dt: float = 1.0
    ):
        """
        Initialize the Leaky Integrate-and-Fire neuron.
        
        Args:
            v_rest: Resting membrane potential (mV)
            v_thresh: Spike threshold (mV)  
            v_reset: Reset potential after spike (mV)
            tau_m: Membrane time constant (ms)
            refractory_period: Refractory period duration (time steps)
            dt: Time step size (ms)
        """
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau_m = tau_m
        self.refractory_period = refractory_period
        self.dt = dt
        
        # State variables
        self.v = v_rest  # Current membrane potential
        self.refractory_counter = 0  # Remaining refractory time
        self.last_spike_time = -np.inf  # Time of last spike
        
        # History tracking
        self.spike_times = []  # List of spike times
        self.voltage_history = []  # Membrane potential history
        
    def reset(self) -> None:
        """Reset neuron to initial state."""
        self.v = self.v_rest
        self.refractory_counter = 0
        self.last_spike_time = -np.inf
        self.spike_times.clear()
        self.voltage_history.clear()
        
    def update(self, input_current: float, current_time: float) -> bool:
        """
        Update neuron state for one time step.
        
        Args:
            input_current: Synaptic input current
            current_time: Current simulation time (ms)
            
        Returns:
            bool: True if neuron spiked, False otherwise
        """
        spike_occurred = False
        
        # Handle refractory period
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.v = self.v_reset
        else:
            # Update membrane potential using Euler integration
            # dv/dt = -(v - v_rest)/tau_m + I/C
            # Assuming C=1 for simplicity
            dv_dt = -(self.v - self.v_rest) / self.tau_m + input_current
            self.v += dv_dt * self.dt
            
            # Check for spike
            if self.v >= self.v_thresh:
                spike_occurred = True
                self.v = self.v_reset
                self.last_spike_time = current_time
                self.spike_times.append(current_time)
                self.refractory_counter = self.refractory_period
        
        # Record voltage history
        self.voltage_history.append(self.v)
        
        return spike_occurred
    
    def get_spike_train(self, duration: float) -> np.ndarray:
        """
        Get binary spike train for given duration.
        
        Args:
            duration: Total simulation duration (ms)
            
        Returns:
            Binary array indicating spike times
        """
        n_steps = int(duration / self.dt)
        spike_train = np.zeros(n_steps, dtype=bool)
        
        for spike_time in self.spike_times:
            step = int(spike_time / self.dt)
            if 0 <= step < n_steps:
                spike_train[step] = True
                
        return spike_train
    
    def get_voltage_trace(self) -> np.ndarray:
        """Get membrane potential history as numpy array."""
        return np.array(self.voltage_history)
    
    def is_in_refractory_period(self) -> bool:
        """Check if neuron is currently in refractory period."""
        return self.refractory_counter > 0
    
    def time_since_last_spike(self, current_time: float) -> float:
        """
        Get time elapsed since last spike.
        
        Args:
            current_time: Current simulation time (ms)
            
        Returns:
            Time since last spike (ms)
        """
        return current_time - self.last_spike_time


class PoissonSpikeGenerator:
    """
    Generates Poisson-distributed spike trains for presynaptic input.
    
    This class generates random spike trains following a Poisson process,
    commonly used to model background neural activity or sensory input.
    """
    
    def __init__(self, rate: float, dt: float = 1.0, seed: Optional[int] = None):
        """
        Initialize Poisson spike generator.
        
        Args:
            rate: Average firing rate (Hz)
            dt: Time step size (ms)
            seed: Random seed for reproducibility
        """
        self.rate = rate
        self.dt = dt
        
        if seed is not None:
            np.random.seed(seed)
            
        # Convert rate from Hz to probability per time step
        self.spike_probability = rate * dt / 1000.0
        
    def generate_spike(self) -> bool:
        """
        Generate a spike for current time step.
        
        Returns:
            bool: True if spike occurs, False otherwise
        """
        return np.random.random() < self.spike_probability
    
    def generate_spike_train(self, duration: float) -> np.ndarray:
        """
        Generate complete spike train for given duration.
        
        Args:
            duration: Total duration (ms)
            
        Returns:
            Binary array indicating spike times
        """
        n_steps = int(duration / self.dt)
        return np.random.random(n_steps) < self.spike_probability
