"""
Visualization and plotting utilities for STDP simulations.

This module provides comprehensive plotting capabilities for visualizing
STDP simulation results including spike trains, membrane potential,
and synaptic weight evolution.
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class STDPPlotter:
    """
    Comprehensive plotting class for STDP simulation visualization.
    
    This class provides various plotting methods to visualize different
    aspects of STDP simulations including:
    - Spike raster plots
    - Membrane potential traces
    - Synaptic weight evolution
    - STDP learning curves
    """
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize STDP plotter.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
        """
        self.style = style
        self.figsize = figsize
        
        # Color scheme
        self.colors = {
            'pre_spike': '#1f77b4',    # Blue
            'post_spike': '#ff7f0e',   # Orange  
            'weight': '#2ca02c',       # Green
            'voltage': '#d62728',      # Red
            'background': '#f7f7f7'    # Light gray
        }
        
    def plot_simulation_results(
        self, 
        results: Dict[str, Any], 
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        Create comprehensive plot of simulation results.
        
        Args:
            results: Dictionary containing simulation results
            save_path: Optional path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib Figure object
        """
        plt.style.use(self.style)
        fig, axes = plt.subplots(4, 1, figsize=self.figsize, sharex=True)
        
        time = results['time']
        
        # Plot 1: Membrane potential
        axes[0].plot(time, results['voltage'], 
                    color=self.colors['voltage'], linewidth=1.5)
        axes[0].set_ylabel('Membrane\nPotential (mV)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_title('STDP Simulation Results', fontsize=14, fontweight='bold')
        
        # Plot 2: Pre-synaptic spikes
        pre_spike_times = time[results['pre_spikes']]
        if len(pre_spike_times) > 0:
            axes[1].eventplot([pre_spike_times], lineoffsets=0.5, 
                            colors=[self.colors['pre_spike']], linewidths=2)
        axes[1].set_ylabel('Pre-synaptic\nSpikes')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Post-synaptic spikes  
        post_spike_times = time[results['post_spikes']]
        if len(post_spike_times) > 0:
            axes[2].eventplot([post_spike_times], lineoffsets=0.5,
                            colors=[self.colors['post_spike']], linewidths=2)
        axes[2].set_ylabel('Post-synaptic\nSpikes')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        
        # Plot 4: Synaptic weight
        axes[3].plot(time, results['weights'], 
                    color=self.colors['weight'], linewidth=2)
        axes[3].set_ylabel('Synaptic\nWeight')
        axes[3].set_xlabel('Time (ms)')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
    
    def plot_weight_evolution(
        self,
        results: Dict[str, Any],
        highlight_changes: bool = True,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        Plot detailed synaptic weight evolution.
        
        Args:
            results: Simulation results dictionary
            highlight_changes: Whether to highlight weight change events
            save_path: Optional path to save figure  
            show: Whether to display the plot
            
        Returns:
            Matplotlib Figure object
        """
        plt.style.use(self.style)
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        time = results['time']
        weights = results['weights']
        
        # Plot weight evolution
        ax.plot(time, weights, color=self.colors['weight'], 
               linewidth=2, label='Synaptic Weight')
        
        if highlight_changes:
            # Highlight significant weight changes
            weight_changes = results['weight_changes']
            significant_changes = np.abs(weight_changes) > 0.001
            
            if np.any(significant_changes):
                change_times = time[significant_changes]
                change_weights = weights[significant_changes]
                
                # Potentiation (positive changes)
                ltp_mask = weight_changes[significant_changes] > 0
                if np.any(ltp_mask):
                    ax.scatter(change_times[ltp_mask], change_weights[ltp_mask],
                             color='red', s=30, alpha=0.7, label='LTP Events')
                
                # Depression (negative changes)  
                ltd_mask = weight_changes[significant_changes] < 0
                if np.any(ltd_mask):
                    ax.scatter(change_times[ltd_mask], change_weights[ltd_mask],
                             color='blue', s=30, alpha=0.7, label='LTD Events')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Synaptic Weight')
        ax.set_title('Synaptic Weight Evolution During STDP', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
    
    def plot_spike_timing_analysis(
        self,
        results: Dict[str, Any],
        time_window: float = 50.0,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        Analyze and plot spike timing relationships.
        
        Args:
            results: Simulation results dictionary
            time_window: Time window for analysis (ms)
            save_path: Optional path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib Figure object
        """
        plt.style.use(self.style)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        time = results['time']
        pre_spikes = results['pre_spikes']
        post_spikes = results['post_spikes']
        
        # Get spike times
        pre_times = time[pre_spikes]
        post_times = time[post_spikes]
        
        # Plot 1: Spike raster
        axes[0, 0].eventplot([pre_times, post_times], 
                           colors=[self.colors['pre_spike'], self.colors['post_spike']],
                           linewidths=2)
        axes[0, 0].set_ylabel('Neuron')
        axes[0, 0].set_title('Spike Raster Plot')
        axes[0, 0].set_yticks([0, 1])
        axes[0, 0].set_yticklabels(['Pre-synaptic', 'Post-synaptic'])
        
        # Plot 2: Inter-spike intervals
        if len(pre_times) > 1:
            pre_isi = np.diff(pre_times)
            axes[0, 1].hist(pre_isi, bins=20, alpha=0.7, 
                          color=self.colors['pre_spike'], label='Pre-synaptic')
        
        if len(post_times) > 1:
            post_isi = np.diff(post_times)
            axes[0, 1].hist(post_isi, bins=20, alpha=0.7,
                          color=self.colors['post_spike'], label='Post-synaptic')
        
        axes[0, 1].set_xlabel('Inter-spike Interval (ms)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Inter-spike Interval Distribution')
        axes[0, 1].legend()
        
        # Plot 3: Spike timing differences
        if len(pre_times) > 0 and len(post_times) > 0:
            # Calculate all pairwise timing differences
            timing_diffs = []
            for post_time in post_times:
                for pre_time in pre_times:
                    diff = post_time - pre_time
                    if abs(diff) <= time_window:
                        timing_diffs.append(diff)
            
            if timing_diffs:
                axes[1, 0].hist(timing_diffs, bins=30, alpha=0.7,
                              color='purple', edgecolor='black')
                axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
                axes[1, 0].set_xlabel('Δt = t_post - t_pre (ms)')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('Spike Timing Differences')
        
        # Plot 4: Firing rates over time
        bin_size = 10.0  # ms
        n_bins = int(results['duration'] / bin_size)
        bin_edges = np.linspace(0, results['duration'], n_bins + 1)
        
        pre_rates, _ = np.histogram(pre_times, bins=bin_edges)
        post_rates, _ = np.histogram(post_times, bins=bin_edges)
        
        # Convert to rates (Hz)
        pre_rates = pre_rates / (bin_size / 1000.0)
        post_rates = post_rates / (bin_size / 1000.0)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        axes[1, 1].plot(bin_centers, pre_rates, 
                       color=self.colors['pre_spike'], label='Pre-synaptic')
        axes[1, 1].plot(bin_centers, post_rates,
                       color=self.colors['post_spike'], label='Post-synaptic')
        axes[1, 1].set_xlabel('Time (ms)')
        axes[1, 1].set_ylabel('Firing Rate (Hz)')
        axes[1, 1].set_title('Firing Rates Over Time')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
    
    def plot_stdp_curve(
        self,
        stdp_rule,
        time_range: Tuple[float, float] = (-50, 50),
        n_points: int = 1000,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Figure:
        """
        Plot theoretical STDP learning curve.
        
        Args:
            stdp_rule: STDP rule object
            time_range: Range of timing differences to plot (ms)
            n_points: Number of points to compute
            save_path: Optional path to save figure
            show: Whether to display the plot
            
        Returns:
            Matplotlib Figure object
        """
        plt.style.use(self.style)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Generate timing differences
        dt_values = np.linspace(time_range[0], time_range[1], n_points)
        weight_changes = np.zeros_like(dt_values)
        
        # Calculate weight changes for each timing difference
        for i, dt in enumerate(dt_values):
            if hasattr(stdp_rule, 'A_plus') and hasattr(stdp_rule, 'tau_plus'):
                # Exponential STDP
                if dt > 0:  # Pre before post (LTP)
                    weight_changes[i] = stdp_rule.A_plus * np.exp(-dt / stdp_rule.tau_plus)
                else:  # Post before pre (LTD)
                    weight_changes[i] = stdp_rule.A_minus * np.exp(dt / stdp_rule.tau_minus)
            else:
                # Simple STDP or other rules
                if hasattr(stdp_rule, 'time_window'):
                    if 0 < dt <= stdp_rule.time_window:
                        weight_changes[i] = stdp_rule.A_plus
                    elif -stdp_rule.time_window <= dt < 0:
                        weight_changes[i] = stdp_rule.A_minus
        
        # Plot STDP curve
        ax.plot(dt_values, weight_changes, linewidth=3, color='purple')
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Add annotations
        ax.fill_between(dt_values[dt_values > 0], 0, 
                       weight_changes[dt_values > 0], 
                       alpha=0.3, color='red', label='LTP (Potentiation)')
        ax.fill_between(dt_values[dt_values < 0], 0,
                       weight_changes[dt_values < 0],
                       alpha=0.3, color='blue', label='LTD (Depression)')
        
        ax.set_xlabel('Δt = t_post - t_pre (ms)')
        ax.set_ylabel('Weight Change (Δw)')
        ax.set_title('STDP Learning Curve', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if show:
            plt.show()
            
        return fig
