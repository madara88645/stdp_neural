"""
Quick test of STDP simulation functionality.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stdp_simulation.core.neuron import LeakyIntegrateFireNeuron, PoissonSpikeGenerator
from stdp_simulation.core.stdp import ExponentialSTDP, STDPSimulator

def quick_test():
    """Run a quick STDP simulation test."""
    print("Running Quick STDP Test...")
    print("=" * 40)
    
    # Create components
    neuron = LeakyIntegrateFireNeuron(dt=1.0)
    stdp_rule = ExponentialSTDP(initial_weight=0.5)
    pre_generator = PoissonSpikeGenerator(rate=10.0, dt=1.0, seed=42)
    
    # Create simulator
    simulator = STDPSimulator(neuron, stdp_rule, pre_generator, dt=1.0)
    
    # Run short simulation
    duration = 100.0
    results = simulator.run(duration, verbose=True)
    
    # Print results
    n_pre = np.sum(results['pre_spikes'])
    n_post = np.sum(results['post_spikes'])
    initial_weight = results['weights'][0]
    final_weight = results['weights'][-1]
    weight_change = final_weight - initial_weight
    
    print(f"\nResults:")
    print(f"  Duration: {duration} ms")
    print(f"  Pre-synaptic spikes: {n_pre}")
    print(f"  Post-synaptic spikes: {n_post}")
    print(f"  Initial weight: {initial_weight:.3f}")
    print(f"  Final weight: {final_weight:.3f}")
    print(f"  Weight change: {weight_change:+.3f}")
    
    # Simple plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    time = results['time']
    
    # Plot voltage
    axes[0].plot(time, results['voltage'], 'r-', linewidth=1)
    axes[0].set_ylabel('Membrane Potential (mV)')
    axes[0].set_title('Quick STDP Test Results')
    axes[0].grid(True, alpha=0.3)
    
    # Plot weight
    axes[1].plot(time, results['weights'], 'g-', linewidth=2)
    axes[1].set_ylabel('Synaptic Weight')
    axes[1].set_xlabel('Time (ms)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: quick_test_results.png")
    plt.show()
    
    print("\nQuick test completed successfully! ✅")

if __name__ == "__main__":
    quick_test()
