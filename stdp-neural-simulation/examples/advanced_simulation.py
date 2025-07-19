"""
Advanced STDP Simulation Example

This example demonstrates more advanced features including:
- Multiple synapses with different properties
- Parameter sweeps and optimization
- Custom stimulation patterns
- Advanced analysis and visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from stdp_simulation.core.neuron import LeakyIntegrateFireNeuron, PoissonSpikeGenerator
from stdp_simulation.core.stdp import ExponentialSTDP, STDPSimulator
from stdp_simulation.visualization.plotter import STDPPlotter
from stdp_simulation.utils.config import (
    SimulationConfig, create_default_configs, print_simulation_summary
)


class MultiSynapseNetwork:
    """
    A network with multiple synapses converging on a single neuron.
    
    This demonstrates how multiple inputs with different STDP properties
    can interact and compete for control of the postsynaptic neuron.
    """
    
    def __init__(self, n_synapses: int = 3, dt: float = 1.0):
        """
        Initialize multi-synapse network.
        
        Args:
            n_synapses: Number of presynaptic inputs
            dt: Time step size (ms)
        """
        self.n_synapses = n_synapses
        self.dt = dt
        
        # Create single postsynaptic neuron
        self.neuron = LeakyIntegrateFireNeuron(dt=dt)
        
        # Create multiple STDP synapses with different properties
        self.synapses = []
        self.generators = []
        
        for i in range(n_synapses):
            # Vary STDP parameters across synapses
            A_plus = 0.01 + i * 0.005  # Increasing potentiation strength
            A_minus = -0.01 - i * 0.002  # Increasing depression strength
            
            stdp = ExponentialSTDP(
                initial_weight=0.3 + i * 0.1,
                A_plus=A_plus,
                A_minus=A_minus,
                tau_plus=20.0,
                tau_minus=20.0
            )
            
            # Different firing rates for each input
            rate = 5.0 + i * 5.0  # 5, 10, 15 Hz etc.
            generator = PoissonSpikeGenerator(rate=rate, dt=dt, seed=42+i)
            
            self.synapses.append(stdp)
            self.generators.append(generator)
        
        # Track network state
        self.current_time = 0.0
        self.last_post_time = -np.inf
        self.results = {
            'time': [],
            'weights': [[] for _ in range(n_synapses)],
            'pre_spikes': [[] for _ in range(n_synapses)],
            'post_spikes': [],
            'voltage': []
        }
    
    def step(self) -> Dict[str, Any]:
        """Execute one network time step."""
        # Generate presynaptic spikes and calculate total input
        total_input = 0.0
        pre_spikes = []
        
        for i, (generator, synapse) in enumerate(zip(self.generators, self.synapses)):
            spike = generator.generate_spike()
            pre_spikes.append(spike)
            
            if spike:
                total_input += synapse.weight
        
        # Update neuron
        post_spike = self.neuron.update(total_input, self.current_time)
        if post_spike:
            self.last_post_time = self.current_time
        
        # Update all synapses
        for i, (synapse, spike) in enumerate(zip(self.synapses, pre_spikes)):
            last_pre_time = self.current_time if spike else -np.inf
            synapse.update_weight(
                spike, post_spike, self.current_time,
                last_pre_time, self.last_post_time
            )
        
        # Record results
        self.results['time'].append(self.current_time)
        for i, synapse in enumerate(self.synapses):
            self.results['weights'][i].append(synapse.weight)
        for i, spike in enumerate(pre_spikes):
            self.results['pre_spikes'][i].append(spike)
        self.results['post_spikes'].append(post_spike)
        self.results['voltage'].append(self.neuron.v)
        
        self.current_time += self.dt
        
        return {
            'pre_spikes': pre_spikes,
            'post_spike': post_spike,
            'weights': [s.weight for s in self.synapses],
            'total_input': total_input
        }
    
    def run(self, duration: float) -> Dict[str, Any]:
        """Run network simulation."""
        n_steps = int(duration / self.dt)
        
        for _ in range(n_steps):
            self.step()
        
        # Convert to numpy arrays
        results = {
            'time': np.array(self.results['time']),
            'weights': [np.array(w) for w in self.results['weights']],
            'pre_spikes': [np.array(s) for s in self.results['pre_spikes']],
            'post_spikes': np.array(self.results['post_spikes']),
            'voltage': np.array(self.results['voltage']),
            'duration': duration,
            'dt': self.dt
        }
        
        return results


def run_multi_synapse_simulation():
    """Demonstrate multi-synapse network with competing inputs."""
    
    print("Running Multi-Synapse Network Simulation...")
    print("=" * 50)
    
    # Create and run network
    network = MultiSynapseNetwork(n_synapses=3, dt=1.0)
    results = network.run(duration=300.0)
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    time = results['time']
    
    # Plot voltage
    axes[0].plot(time, results['voltage'], color='red', linewidth=1)
    axes[0].set_ylabel('Membrane\nPotential (mV)')
    axes[0].set_title('Multi-Synapse Network Competition')
    axes[0].grid(True, alpha=0.3)
    
    # Plot presynaptic spikes (all on one plot)
    colors = ['blue', 'green', 'orange']
    for i, (spikes, color) in enumerate(zip(results['pre_spikes'], colors)):
        spike_times = time[spikes]
        if len(spike_times) > 0:
            axes[1].eventplot([spike_times], lineoffsets=i, colors=[color], 
                            linewidths=2, label=f'Input {i+1}')
    axes[1].set_ylabel('Pre-synaptic\nInputs')
    axes[1].set_ylim(-0.5, 2.5)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot postsynaptic spikes
    post_spike_times = time[results['post_spikes']]
    if len(post_spike_times) > 0:
        axes[2].eventplot([post_spike_times], lineoffsets=0, colors=['red'], linewidths=2)
    axes[2].set_ylabel('Post-synaptic\nSpikes')
    axes[2].set_ylim(-0.5, 0.5)
    axes[2].grid(True, alpha=0.3)
    
    # Plot weight evolution for all synapses
    for i, (weights, color) in enumerate(zip(results['weights'], colors)):
        axes[3].plot(time, weights, color=color, linewidth=2, label=f'Synapse {i+1}')
    axes[3].set_ylabel('Synaptic\nWeights')
    axes[3].set_xlabel('Time (ms)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("multi_synapse_network.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print competition results
    final_weights = [w[-1] for w in results['weights']]
    winner = np.argmax(final_weights)
    
    print(f"\nSynaptic Competition Results:")
    for i, weight in enumerate(final_weights):
        status = " (WINNER)" if i == winner else ""
        print(f"  Synapse {i+1}: {weight:.3f}{status}")
    
    return results


def parameter_sweep_analysis():
    """Perform parameter sweep to analyze STDP sensitivity."""
    
    print("\nRunning Parameter Sweep Analysis...")
    print("=" * 50)
    
    # Parameter ranges to sweep
    A_plus_values = np.logspace(-3, -1, 10)  # 0.001 to 0.1
    durations = [100, 200, 500]  # Different simulation lengths
    
    results_matrix = np.zeros((len(A_plus_values), len(durations)))
    
    for i, A_plus in enumerate(A_plus_values):
        for j, duration in enumerate(durations):
            # Create simulation components
            neuron = LeakyIntegrateFireNeuron()
            stdp_rule = ExponentialSTDP(
                initial_weight=0.5,
                A_plus=A_plus,
                A_minus=-A_plus * 1.25  # Keep ratio constant
            )
            pre_generator = PoissonSpikeGenerator(rate=10.0, seed=42)
            
            # Run simulation
            simulator = STDPSimulator(neuron, stdp_rule, pre_generator)
            results = simulator.run(duration)
            
            # Calculate final weight change
            weight_change = results['weights'][-1] - results['weights'][0]
            results_matrix[i, j] = weight_change
    
    # Plot parameter sweep results
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for j, duration in enumerate(durations):
        ax.semilogx(A_plus_values, results_matrix[:, j], 
                   marker='o', linewidth=2, label=f'{duration} ms')
    
    ax.set_xlabel('LTP Amplitude (A+)')
    ax.set_ylabel('Final Weight Change')
    ax.set_title('STDP Parameter Sensitivity Analysis')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("parameter_sweep.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Parameter sweep analysis complete!")
    
    return results_matrix


def custom_stimulation_patterns():
    """Demonstrate different stimulation patterns and their effects."""
    
    print("\nTesting Custom Stimulation Patterns...")
    print("=" * 50)
    
    patterns = {
        'regular': {'rate': 10.0, 'description': 'Regular Poisson'},
        'burst': {'rate': 50.0, 'description': 'High-frequency bursts'},
        'sparse': {'rate': 2.0, 'description': 'Sparse activation'}
    }
    
    fig, axes = plt.subplots(len(patterns), 1, figsize=(12, 10), sharex=True)
    
    for idx, (pattern_name, params) in enumerate(patterns.items()):
        # Create simulation
        neuron = LeakyIntegrateFireNeuron()
        stdp_rule = ExponentialSTDP(initial_weight=0.5)
        pre_generator = PoissonSpikeGenerator(
            rate=params['rate'], 
            seed=42+idx
        )
        
        simulator = STDPSimulator(neuron, stdp_rule, pre_generator)
        results = simulator.run(duration=200.0)
        
        # Plot weight evolution
        axes[idx].plot(results['time'], results['weights'], 
                      linewidth=2, color=f'C{idx}')
        axes[idx].set_ylabel(f'Weight\n({params["description"]})')
        axes[idx].grid(True, alpha=0.3)
        
        # Add spike indicators
        pre_times = results['time'][results['pre_spikes']]
        post_times = results['time'][results['post_spikes']]
        
        if len(pre_times) > 0:
            axes[idx].scatter(pre_times, [axes[idx].get_ylim()[1]] * len(pre_times),
                            s=10, color='blue', alpha=0.6, label='Pre')
        if len(post_times) > 0:
            axes[idx].scatter(post_times, [axes[idx].get_ylim()[1]] * len(post_times),
                            s=20, color='red', alpha=0.8, marker='^', label='Post')
        
        if idx == 0:
            axes[idx].legend()
    
    axes[-1].set_xlabel('Time (ms)')
    plt.suptitle('Effects of Different Stimulation Patterns on STDP')
    plt.tight_layout()
    plt.savefig("stimulation_patterns.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Stimulation pattern analysis complete!")


if __name__ == "__main__":
    # Run all advanced examples
    print("Advanced STDP Simulation Examples")
    print("=" * 60)
    
    # 1. Multi-synapse competition
    multi_results = run_multi_synapse_simulation()
    
    # 2. Parameter sensitivity analysis
    sweep_results = parameter_sweep_analysis()
    
    # 3. Custom stimulation patterns
    custom_stimulation_patterns()
    
    print("\nAll advanced examples complete!")
    print("Check the generated plots for detailed analysis.")
