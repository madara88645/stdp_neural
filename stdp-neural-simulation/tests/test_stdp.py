"""
Unit tests for STDP simulation components.

This module contains comprehensive tests for all major components
of the STDP simulation library.
"""

import unittest
import numpy as np

from stdp_simulation.core.neuron import LeakyIntegrateFireNeuron, PoissonSpikeGenerator
from stdp_simulation.core.stdp import ExponentialSTDP, SimpleSTDP, STDPSimulator
from stdp_simulation.utils.config import SimulationConfig, NeuronConfig, STDPConfig


class TestLeakyIntegrateFireNeuron(unittest.TestCase):
    """Test cases for LeakyIntegrateFireNeuron."""
    
    def setUp(self):
        """Set up test neuron."""
        self.neuron = LeakyIntegrateFireNeuron()
    
    def test_initialization(self):
        """Test neuron initialization."""
        self.assertEqual(self.neuron.v, 0.0)
        self.assertEqual(self.neuron.refractory_counter, 0)
        self.assertEqual(len(self.neuron.spike_times), 0)
    
    def test_subthreshold_input(self):
        """Test neuron response to subthreshold input."""
        initial_v = self.neuron.v
        spike = self.neuron.update(input_current=0.1, current_time=1.0)
        
        self.assertFalse(spike)
        self.assertGreater(self.neuron.v, initial_v)
        self.assertEqual(len(self.neuron.spike_times), 0)
    
    def test_suprathreshold_input(self):
        """Test neuron spiking with suprathreshold input."""
        # Apply large input to trigger spike
        spike = self.neuron.update(input_current=10.0, current_time=1.0)
        
        self.assertTrue(spike)
        self.assertEqual(self.neuron.v, self.neuron.v_reset)
        self.assertEqual(len(self.neuron.spike_times), 1)
        self.assertEqual(self.neuron.spike_times[0], 1.0)
    
    def test_refractory_period(self):
        """Test refractory period functionality."""
        # Trigger spike
        self.neuron.update(input_current=10.0, current_time=1.0)
        
        # During refractory period, neuron should not spike
        for t in range(2, 7):  # 5-step refractory period
            spike = self.neuron.update(input_current=10.0, current_time=float(t))
            self.assertFalse(spike)
            self.assertEqual(self.neuron.v, self.neuron.v_reset)
        
        # After refractory period, neuron should be able to spike again
        spike = self.neuron.update(input_current=10.0, current_time=7.0)
        self.assertTrue(spike)
    
    def test_reset_functionality(self):
        """Test neuron reset."""
        # Modify neuron state
        self.neuron.update(input_current=0.5, current_time=1.0)
        self.neuron.update(input_current=10.0, current_time=2.0)  # Trigger spike
        
        # Reset neuron
        self.neuron.reset()
        
        self.assertEqual(self.neuron.v, self.neuron.v_rest)
        self.assertEqual(self.neuron.refractory_counter, 0)
        self.assertEqual(len(self.neuron.spike_times), 0)
        self.assertEqual(len(self.neuron.voltage_history), 0)


class TestPoissonSpikeGenerator(unittest.TestCase):
    """Test cases for PoissonSpikeGenerator."""
    
    def test_initialization(self):
        """Test generator initialization."""
        generator = PoissonSpikeGenerator(rate=10.0, dt=1.0, seed=42)
        self.assertEqual(generator.rate, 10.0)
        self.assertEqual(generator.dt, 1.0)
    
    def test_spike_probability(self):
        """Test that spike probability is correctly calculated."""
        rate = 100.0  # Hz
        dt = 1.0  # ms
        generator = PoissonSpikeGenerator(rate=rate, dt=dt)
        
        expected_prob = rate * dt / 1000.0  # Convert to probability per time step
        self.assertAlmostEqual(generator.spike_probability, expected_prob)
    
    def test_spike_train_statistics(self):
        """Test that generated spike trains have correct statistics."""
        rate = 50.0  # Hz
        duration = 1000.0  # ms
        dt = 1.0
        
        generator = PoissonSpikeGenerator(rate=rate, dt=dt, seed=42)
        spike_train = generator.generate_spike_train(duration)
        
        # Check spike count is approximately correct
        expected_spikes = rate * duration / 1000.0
        actual_spikes = np.sum(spike_train)
        
        # Allow 20% tolerance due to randomness
        self.assertGreater(actual_spikes, expected_spikes * 0.8)
        self.assertLess(actual_spikes, expected_spikes * 1.2)


class TestExponentialSTDP(unittest.TestCase):
    """Test cases for ExponentialSTDP."""
    
    def setUp(self):
        """Set up test STDP rule."""
        self.stdp = ExponentialSTDP(
            initial_weight=0.5,
            A_plus=0.01,
            A_minus=-0.01,
            tau_plus=20.0,
            tau_minus=20.0
        )
    
    def test_initialization(self):
        """Test STDP initialization."""
        self.assertEqual(self.stdp.weight, 0.5)
        self.assertEqual(len(self.stdp.weight_history), 1)
    
    def test_ltp_potentiation(self):
        """Test long-term potentiation (pre before post)."""
        initial_weight = self.stdp.weight
        
        # Pre-spike followed by post-spike
        dw = self.stdp.update_weight(
            pre_spike=False, post_spike=True, current_time=10.0,
            last_pre_time=5.0, last_post_time=-np.inf
        )
        
        self.assertGreater(dw, 0)  # Should be potentiation
        self.assertGreater(self.stdp.weight, initial_weight)
    
    def test_ltd_depression(self):
        """Test long-term depression (post before pre)."""
        initial_weight = self.stdp.weight
        
        # Post-spike followed by pre-spike
        dw = self.stdp.update_weight(
            pre_spike=True, post_spike=False, current_time=10.0,
            last_pre_time=-np.inf, last_post_time=5.0
        )
        
        self.assertLess(dw, 0)  # Should be depression
        self.assertLess(self.stdp.weight, initial_weight)
    
    def test_weight_bounds(self):
        """Test that weights stay within bounds."""
        # Test upper bound
        self.stdp.weight = 0.99
        self.stdp.update_weight(
            pre_spike=False, post_spike=True, current_time=10.0,
            last_pre_time=9.0, last_post_time=-np.inf
        )
        self.assertLessEqual(self.stdp.weight, self.stdp.w_max)
        
        # Test lower bound
        self.stdp.weight = 0.01
        self.stdp.update_weight(
            pre_spike=True, post_spike=False, current_time=10.0,
            last_pre_time=-np.inf, last_post_time=9.0
        )
        self.assertGreaterEqual(self.stdp.weight, self.stdp.w_min)
    
    def test_no_change_without_pairing(self):
        """Test that weight doesn't change without spike pairing."""
        initial_weight = self.stdp.weight
        
        # No spikes
        dw = self.stdp.update_weight(
            pre_spike=False, post_spike=False, current_time=10.0,
            last_pre_time=-np.inf, last_post_time=-np.inf
        )
        
        self.assertEqual(dw, 0.0)
        self.assertEqual(self.stdp.weight, initial_weight)


class TestSimpleSTDP(unittest.TestCase):
    """Test cases for SimpleSTDP."""
    
    def setUp(self):
        """Set up test STDP rule."""
        self.stdp = SimpleSTDP(
            initial_weight=0.5,
            A_plus=0.01,
            A_minus=-0.01,
            time_window=5.0
        )
    
    def test_ltp_within_window(self):
        """Test LTP when spikes are within time window."""
        initial_weight = self.stdp.weight
        
        dw = self.stdp.update_weight(
            pre_spike=False, post_spike=True, current_time=10.0,
            last_pre_time=7.0, last_post_time=-np.inf
        )
        
        self.assertAlmostEqual(dw, self.stdp.A_plus)
        self.assertGreater(self.stdp.weight, initial_weight)
    
    def test_no_ltp_outside_window(self):
        """Test no LTP when spikes are outside time window."""
        initial_weight = self.stdp.weight
        
        dw = self.stdp.update_weight(
            pre_spike=False, post_spike=True, current_time=10.0,
            last_pre_time=1.0, last_post_time=-np.inf  # 9ms gap > 5ms window
        )
        
        self.assertEqual(dw, 0.0)
        self.assertEqual(self.stdp.weight, initial_weight)


class TestSTDPSimulator(unittest.TestCase):
    """Test cases for STDPSimulator."""
    
    def setUp(self):
        """Set up test simulator."""
        self.neuron = LeakyIntegrateFireNeuron()
        self.stdp_rule = ExponentialSTDP()
        self.pre_generator = PoissonSpikeGenerator(rate=10.0, seed=42)
        self.simulator = STDPSimulator(
            self.neuron, self.stdp_rule, self.pre_generator
        )
    
    def test_initialization(self):
        """Test simulator initialization."""
        self.assertEqual(self.simulator.current_time, 0.0)
        self.assertEqual(len(self.simulator.time_history), 0)
    
    def test_single_step(self):
        """Test single simulation step."""
        result = self.simulator.step()
        
        self.assertIn('time', result)
        self.assertIn('pre_spike', result)
        self.assertIn('post_spike', result)
        self.assertIn('weight', result)
        self.assertIn('voltage', result)
        
        self.assertEqual(self.simulator.current_time, 1.0)  # dt=1.0
    
    def test_run_simulation(self):
        """Test full simulation run."""
        duration = 50.0
        results = self.simulator.run(duration)
        
        self.assertIn('time', results)
        self.assertIn('pre_spikes', results)
        self.assertIn('post_spikes', results)
        self.assertIn('weights', results)
        self.assertIn('voltage', results)
        
        # Check array lengths
        expected_length = int(duration / self.simulator.dt)
        self.assertEqual(len(results['time']), expected_length)
        self.assertEqual(len(results['pre_spikes']), expected_length)
        self.assertEqual(len(results['post_spikes']), expected_length)
    
    def test_reset_functionality(self):
        """Test simulator reset."""
        # Run simulation briefly
        self.simulator.run(10.0)
        
        # Reset simulator
        self.simulator.reset()
        
        self.assertEqual(self.simulator.current_time, 0.0)
        self.assertEqual(len(self.simulator.time_history), 0)
        self.assertEqual(self.neuron.v, self.neuron.v_rest)


class TestConfiguration(unittest.TestCase):
    """Test cases for configuration management."""
    
    def test_neuron_config(self):
        """Test neuron configuration."""
        config = NeuronConfig(v_thresh=1.5, tau_m=30.0)
        self.assertEqual(config.v_thresh, 1.5)
        self.assertEqual(config.tau_m, 30.0)
        self.assertEqual(config.v_rest, 0.0)  # Default value
    
    def test_stdp_config(self):
        """Test STDP configuration."""
        config = STDPConfig(A_plus=0.05, tau_plus=15.0)
        self.assertEqual(config.A_plus, 0.05)
        self.assertEqual(config.tau_plus, 15.0)
        self.assertEqual(config.initial_weight, 0.5)  # Default value
    
    def test_simulation_config_creation(self):
        """Test complete simulation configuration."""
        config = SimulationConfig(
            neuron=NeuronConfig(),
            stdp=STDPConfig(),
            stimulus=None,  # This might need to be fixed based on actual implementation
            verbose=True
        )
        self.assertTrue(config.verbose)
        self.assertIsInstance(config.neuron, NeuronConfig)
        self.assertIsInstance(config.stdp, STDPConfig)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
