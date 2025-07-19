# STDP Neural Plasticity Simulation

A comprehensive Python library for simulating Spike-Timing-Dependent Plasticity (STDP) in neural networks.

## Overview

This project implements a biologically-inspired neural simulation that demonstrates how synaptic connections strengthen or weaken based on the timing of pre- and post-synaptic spikes. STDP is a fundamental mechanism of synaptic plasticity that underlies learning and memory in biological neural networks.

## Features

- 🧠 **Biologically Accurate Modeling**: Implements leaky integrate-and-fire neurons with realistic membrane dynamics
- ⚡ **STDP Learning Rules**: Multiple STDP variants including exponential and simplified models
- 📊 **Rich Visualization**: Real-time plotting of membrane potential, spike trains, and synaptic weight evolution
- 🔧 **Configurable Parameters**: Easy-to-adjust simulation parameters for different experimental conditions
- 🧪 **Comprehensive Testing**: Unit tests ensuring simulation accuracy
- 📚 **Educational Examples**: Step-by-step tutorials and advanced simulation scenarios

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/stdp-neural-simulation.git
cd stdp-neural-simulation
pip install -r requirements.txt
```

### Basic Usage

```python
from stdp_simulation.core.neuron import LeakyIntegrateFireNeuron, PoissonSpikeGenerator
from stdp_simulation.core.stdp import ExponentialSTDP, STDPSimulator
from stdp_simulation.visualization.plotter import STDPPlotter

# Create components
neuron = LeakyIntegrateFireNeuron()
stdp_rule = ExponentialSTDP()
pre_generator = PoissonSpikeGenerator(rate=10.0)

# Create simulator
simulator = STDPSimulator(neuron, stdp_rule, pre_generator)
results = simulator.run(duration=200.0)

# Visualize results
plotter = STDPPlotter()
plotter.plot_simulation_results(results)
```

### Quick Test

To quickly test the installation:

```bash
python quick_test.py
```

## Project Structure

```
stdp-neural-simulation/
├── stdp_simulation/          # Main package
│   ├── core/                 # Core simulation components
│   │   ├── neuron.py        # Neuron models
│   │   └── stdp.py          # STDP learning rules
│   ├── visualization/        # Plotting utilities
│   │   └── plotter.py       # Visualization functions
│   └── utils/               # Utility functions
│       └── config.py        # Configuration management
├── examples/                 # Usage examples
├── tests/                   # Unit tests
├── docs/                    # Documentation
└── requirements.txt         # Dependencies
```

## Scientific Background

STDP is a biological learning rule that modifies synaptic strength based on the relative timing of pre- and post-synaptic spikes:

- **Long-Term Potentiation (LTP)**: If a pre-synaptic spike precedes a post-synaptic spike within a critical time window, the synapse strengthens
- **Long-Term Depression (LTD)**: If the post-synaptic spike precedes the pre-synaptic spike, the synapse weakens

This timing-dependent plasticity is crucial for:
- Associative learning
- Memory formation
- Neural circuit development
- Temporal sequence learning

## Examples

### Basic STDP Simulation
See `examples/basic_simulation.py` for a simple STDP demonstration.

### Advanced Multi-Synapse Network
See `examples/advanced_simulation.py` for complex network simulations.

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this simulation in your research, please cite:

```bibtex
@software{stdp_simulation,
  title={STDP Neural Plasticity Simulation},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/stdp-neural-simulation}
}
```

## References

1. Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type. Journal of neuroscience, 18(24), 10464-10472.

2. Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. Nature neuroscience, 3(9), 919-926.
