"""
STDP (Spike-Timing-Dependent Plasticity) learning rules.

This module implements various STDP learning rules that modify synaptic
strengths based on the relative timing of pre- and post-synaptic spikes.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class STDPRule(ABC):
    """
    Abstract base class for STDP learning rules.
    
    All STDP implementations should inherit from this class and implement
    the update_weight method.
    """
    
    def __init__(self, initial_weight: float = 0.5):
        """
        Initialize STDP rule.
        
        Args:
            initial_weight: Initial synaptic weight
        """
        self.weight = initial_weight
        self.weight_history = [initial_weight]
        
    @abstractmethod
    def update_weight(
        self, 
        pre_spike: bool, 
        post_spike: bool, 
        current_time: float,
        last_pre_time: float,
        last_post_time: float
    ) -> float:
        """
        Update synaptic weight based on spike timing.
        
        Args:
            pre_spike: Whether presynaptic spike occurred
            post_spike: Whether postsynaptic spike occurred  
            current_time: Current simulation time
            last_pre_time: Time of last presynaptic spike
            last_post_time: Time of last postsynaptic spike
            
        Returns:
            Weight change (delta_w)
        """
        pass
    
    def get_weight_history(self) -> np.ndarray:
        """Get history of synaptic weights."""
        return np.array(self.weight_history)
    
    def reset(self, initial_weight: Optional[float] = None) -> None:
        """Reset weight to initial value."""
        if initial_weight is not None:
            self.weight = initial_weight
        self.weight_history = [self.weight]


class ExponentialSTDP(STDPRule):
    """
    Exponential STDP rule as described in Bi & Poo (1998).
    
    This implements the classic exponential STDP learning rule where:
    - LTP: Δw = A+ * exp(-Δt/τ+) for pre-before-post (Δt > 0)
    - LTD: Δw = A- * exp(Δt/τ-) for post-before-pre (Δt < 0)
    
    Where Δt = t_post - t_pre
    """
    
    def __init__(
        self,
        initial_weight: float = 0.5,
        A_plus: float = 0.02,
        A_minus: float = -0.025,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        """
        Initialize exponential STDP rule.
        
        Args:
            initial_weight: Initial synaptic weight
            A_plus: Amplitude of potentiation (LTP)
            A_minus: Amplitude of depression (LTD)
            tau_plus: Time constant for potentiation (ms)
            tau_minus: Time constant for depression (ms)
            w_min: Minimum allowed weight
            w_max: Maximum allowed weight
        """
        super().__init__(initial_weight)
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max
        
    def update_weight(
        self,
        pre_spike: bool,
        post_spike: bool,
        current_time: float,
        last_pre_time: float,
        last_post_time: float
    ) -> float:
        """Update weight using exponential STDP rule."""
        delta_w = 0.0
        
        if post_spike and last_pre_time > -np.inf:
            # Post-synaptic spike: check for recent pre-synaptic activity
            dt = current_time - last_pre_time
            if dt > 0:  # Pre before post (LTP)
                delta_w += self.A_plus * np.exp(-dt / self.tau_plus)
                
        if pre_spike and last_post_time > -np.inf:
            # Pre-synaptic spike: check for recent post-synaptic activity  
            dt = current_time - last_post_time
            if dt > 0:  # Post before pre (LTD)
                delta_w += self.A_minus * np.exp(-dt / self.tau_minus)
        
        # Update weight with bounds
        self.weight = np.clip(self.weight + delta_w, self.w_min, self.w_max)
        self.weight_history.append(self.weight)
        
        return delta_w


class SimpleSTDP(STDPRule):
    """
    Simplified STDP rule with fixed time window.
    
    This implements a simplified version where weight changes are constant
    within a fixed time window, making it computationally efficient and
    easier to understand.
    """
    
    def __init__(
        self,
        initial_weight: float = 0.5,
        A_plus: float = 0.01,
        A_minus: float = -0.01,
        time_window: float = 1.0,
        w_min: float = 0.0,
        w_max: float = 1.0
    ):
        """
        Initialize simple STDP rule.
        
        Args:
            initial_weight: Initial synaptic weight
            A_plus: Weight change for potentiation
            A_minus: Weight change for depression
            time_window: Time window for STDP (ms)
            w_min: Minimum allowed weight
            w_max: Maximum allowed weight
        """
        super().__init__(initial_weight)
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.time_window = time_window
        self.w_min = w_min
        self.w_max = w_max
        
    def update_weight(
        self,
        pre_spike: bool,
        post_spike: bool,
        current_time: float,
        last_pre_time: float,
        last_post_time: float
    ) -> float:
        """Update weight using simple STDP rule."""
        delta_w = 0.0
        
        if post_spike and (current_time - last_pre_time) <= self.time_window:
            # Post-synaptic spike after recent pre-synaptic spike (LTP)
            delta_w += self.A_plus
            
        if pre_spike and (current_time - last_post_time) <= self.time_window:
            # Pre-synaptic spike after recent post-synaptic spike (LTD)
            delta_w += self.A_minus
        
        # Update weight with bounds
        self.weight = np.clip(self.weight + delta_w, self.w_min, self.w_max)
        self.weight_history.append(self.weight)
        
        return delta_w


class STDPSimulator:
    """
    Main simulator class that coordinates neuron and STDP updates.
    
    This class manages the simulation loop, coordinating spike generation,
    neuron updates, and STDP weight modifications.
    """
    
    def __init__(
        self,
        neuron,
        stdp_rule: STDPRule,
        presynaptic_generator,
        dt: float = 1.0
    ):
        """
        Initialize STDP simulator.
        
        Args:
            neuron: Post-synaptic neuron model
            stdp_rule: STDP learning rule
            presynaptic_generator: Pre-synaptic spike generator
            dt: Time step size (ms)
        """
        self.neuron = neuron
        self.stdp_rule = stdp_rule
        self.presynaptic_generator = presynaptic_generator
        self.dt = dt
        
        # Simulation state
        self.current_time = 0.0
        self.last_pre_time = -np.inf
        self.last_post_time = -np.inf
        
        # History tracking
        self.time_history = []
        self.pre_spikes = []
        self.post_spikes = []
        self.weight_changes = []
        
    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.neuron.reset()
        self.stdp_rule.reset()
        self.current_time = 0.0
        self.last_pre_time = -np.inf
        self.last_post_time = -np.inf
        
        # Clear history
        self.time_history.clear()
        self.pre_spikes.clear()
        self.post_spikes.clear()
        self.weight_changes.clear()
        
    def step(self) -> dict:
        """
        Execute one simulation time step.
        
        Returns:
            Dictionary containing step results
        """
        # Generate pre-synaptic spike
        pre_spike = self.presynaptic_generator.generate_spike()
        if pre_spike:
            self.last_pre_time = self.current_time
            
        # Calculate synaptic input current
        synaptic_current = self.stdp_rule.weight if pre_spike else 0.0
        
        # Update neuron
        post_spike = self.neuron.update(synaptic_current, self.current_time)
        if post_spike:
            self.last_post_time = self.current_time
            
        # Update STDP weight
        weight_change = self.stdp_rule.update_weight(
            pre_spike, post_spike, self.current_time,
            self.last_pre_time, self.last_post_time
        )
        
        # Record history
        self.time_history.append(self.current_time)
        self.pre_spikes.append(pre_spike)
        self.post_spikes.append(post_spike)
        self.weight_changes.append(weight_change)
        
        # Advance time
        self.current_time += self.dt
        
        return {
            'time': self.current_time - self.dt,
            'pre_spike': pre_spike,
            'post_spike': post_spike,
            'weight': self.stdp_rule.weight,
            'weight_change': weight_change,
            'voltage': self.neuron.v
        }
    
    def run(self, duration: float, verbose: bool = False) -> dict:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation duration (ms)
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing simulation results
        """
        n_steps = int(duration / self.dt)
        
        if verbose:
            print(f"Running STDP simulation for {duration} ms ({n_steps} steps)")
            
        for step in range(n_steps):
            step_result = self.step()
            
            if verbose and step % (n_steps // 10) == 0:
                progress = 100 * step / n_steps
                print(f"Progress: {progress:.1f}% - Weight: {step_result['weight']:.3f}")
        
        results = {
            'time': np.array(self.time_history),
            'pre_spikes': np.array(self.pre_spikes),
            'post_spikes': np.array(self.post_spikes),
            'weights': self.stdp_rule.get_weight_history()[:-1],  # Exclude final append
            'weight_changes': np.array(self.weight_changes),
            'voltage': self.neuron.get_voltage_trace(),
            'dt': self.dt,
            'duration': duration
        }
        
        if verbose:
            n_pre = np.sum(results['pre_spikes'])
            n_post = np.sum(results['post_spikes'])
            final_weight = results['weights'][-1]
            print(f"Simulation complete:")
            print(f"  Pre-synaptic spikes: {n_pre}")
            print(f"  Post-synaptic spikes: {n_post}")
            print(f"  Final weight: {final_weight:.3f}")
            
        return results
