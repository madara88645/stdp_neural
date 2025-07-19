"""
Basic STDP Simulation Example

This example demonstrates the basic usage of the STDP simulation library
with a simple leaky integrate-and-fire neuron and exponential STDP rule.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import our STDP simulation modules
from stdp_simulation.core.neuron import LeakyIntegrateFireNeuron, PoissonSpikeGenerator
from stdp_simulation.core.stdp import ExponentialSTDP, STDPSimulator
from stdp_simulation.visualization.plotter import STDPPlotter
from stdp_simulation.utils.config import SimulationConfig, print_simulation_summary


def run_basic_stdp_simulation():
    """Run a basic STDP simulation and display results."""
    
    print("Running Basic STDP Simulation...")
    print("=" * 50)
    
    # Simulation parameters
    duration = 200.0  # ms
    dt = 1.0  # ms
    
    # Create neuron
    neuron = LeakyIntegrateFireNeuron(
        v_rest=0.0,
        v_thresh=1.0,
        v_reset=0.0,
        tau_m=20.0,
        refractory_period=5,
        dt=dt
    )
    
    # Create STDP rule
    stdp_rule = ExponentialSTDP(
        initial_weight=0.5,
        A_plus=0.02,
        A_minus=-0.025,
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=0.0,
        w_max=1.0
    )
    
    # Create presynaptic spike generator
    pre_generator = PoissonSpikeGenerator(
        rate=10.0,  # 10 Hz
        dt=dt,
        seed=42  # For reproducibility
    )
    
    # Create simulator
    simulator = STDPSimulator(neuron, stdp_rule, pre_generator, dt)
    
    # Run simulation
    results = simulator.run(duration, verbose=True)
    
    # Print summary
    print_simulation_summary(results)
    
    # Create plotter and visualize results
    plotter = STDPPlotter()
    
    # Plot main simulation results
    plotter.plot_simulation_results(results, save_path="basic_stdp_results.png")
    
    # Plot detailed weight evolution
    plotter.plot_weight_evolution(results, save_path="weight_evolution.png")
    
    # Plot spike timing analysis
    plotter.plot_spike_timing_analysis(results, save_path="spike_analysis.png")
    
    # Plot theoretical STDP curve
    plotter.plot_stdp_curve(stdp_rule, save_path="stdp_curve.png")
    
    print("\nSimulation complete! Check the generated plots.")
    
    return results


def compare_stdp_rules():
    """Compare different STDP learning rules."""
    
    print("\nComparing STDP Rules...")
    print("=" * 50)
    
    # Import simple STDP for comparison
    from stdp_simulation.core.stdp import SimpleSTDP
    
    duration = 200.0
    dt = 1.0
    
    # Create base components
    neuron1 = LeakyIntegrateFireNeuron(dt=dt)
    neuron2 = LeakyIntegrateFireNeuron(dt=dt)
    
    # Create different STDP rules
    exp_stdp = ExponentialSTDP(initial_weight=0.5)
    simple_stdp = SimpleSTDP(initial_weight=0.5)
    
    # Use same presynaptic input for fair comparison
    pre_generator = PoissonSpikeGenerator(rate=15.0, dt=dt, seed=42)
    
    # Run simulations
    sim1 = STDPSimulator(neuron1, exp_stdp, pre_generator, dt)
    results1 = sim1.run(duration)
    
    # Reset generator for second simulation
    pre_generator = PoissonSpikeGenerator(rate=15.0, dt=dt, seed=42)
    sim2 = STDPSimulator(neuron2, simple_stdp, pre_generator, dt)
    results2 = sim2.run(duration)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    time1 = results1['time']
    time2 = results2['time']
    
    axes[0].plot(time1, results1['weights'], label='Exponential STDP', linewidth=2)
    axes[0].set_ylabel('Synaptic Weight')
    axes[0].set_title('Comparison of STDP Learning Rules')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(time2, results2['weights'], label='Simple STDP', 
                color='orange', linewidth=2)
    axes[1].set_ylabel('Synaptic Weight')
    axes[1].set_xlabel('Time (ms)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig("stdp_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("STDP rule comparison complete!")


if __name__ == "__main__":
    # Run basic simulation
    results = run_basic_stdp_simulation()
    
    # Compare different STDP rules
    compare_stdp_rules()
    
    print("\nExample complete! Check the generated plots and output files.")
