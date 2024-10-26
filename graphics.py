import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from dataclasses import dataclass
from collections import deque
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

@dataclass
class VisualizationData:
    """Simplified data structure for visualization"""
    entropies: List[float]
    logits_varentropy: List[float]
    attn_entropy: List[float]
    attn_varentropy: List[float]
    tokens: List[str]
    time_steps: List[int]

class TokenVisualizerWindow:
    def __init__(self, parent=None):
        self.parent = parent
        self.window = tk.Toplevel(self.parent)
        self.window.title("Generation Statistics Visualization")
        self.window.geometry("1200x800")
        
        # Initialize data buffers
        self.buffer_size = 1000
        self.entropy_buffer = deque(maxlen=self.buffer_size)
        self.logits_varentropy_buffer = deque(maxlen=self.buffer_size)
        self.attn_entropy_buffer = deque(maxlen=self.buffer_size)
        self.attn_varentropy_buffer = deque(maxlen=self.buffer_size)
        self.tokens_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        
        self.setup_plots()
        
        # Update interval in milliseconds
        self.update_interval = 100
        
    def setup_plots(self):
        """Create the plot layout"""
        # Create main figure with subplots
        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        gs = self.fig.add_gridspec(2, 2)
        
        # Create subplots
        self.entropy_ax = self.fig.add_subplot(gs[0, 0])
        self.varentropy_ax = self.fig.add_subplot(gs[0, 1])
        self.attn_ax = self.fig.add_subplot(gs[1, 0])
        self.combined_ax = self.fig.add_subplot(gs[1, 1])
        
        # Setup canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots
        self.entropy_line, = self.entropy_ax.plot([], [], 'b-', label='Entropy')
        self.varentropy_line, = self.varentropy_ax.plot([], [], 'r-', label='Varentropy')
        self.attn_entropy_line, = self.attn_ax.plot([], [], 'g-', label='Attention Entropy')
        
        # Setup combined plot with multiple lines
        self.combined_entropy_line, = self.combined_ax.plot([], [], 'b-', label='Entropy', alpha=0.5)
        self.combined_varentropy_line, = self.combined_ax.plot([], [], 'r-', label='Varentropy', alpha=0.5)
        self.combined_attn_line, = self.combined_ax.plot([], [], 'g-', label='Attn Entropy', alpha=0.5)
        
        # Configure plot layouts
        self.setup_plot_layouts()
        
    def setup_plot_layouts(self):
        """Configure the appearance of all plots"""
        plots = [
            (self.entropy_ax, "Entropy Over Time", "Time", "Entropy"),
            (self.varentropy_ax, "Variance Entropy Over Time", "Time", "Varentropy"),
            (self.attn_ax, "Attention Entropy Over Time", "Time", "Attention Entropy"),
            (self.combined_ax, "Combined Metrics", "Time", "Value")
        ]
        
        for ax, title, xlabel, ylabel in plots:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.legend()
            
    def update_visualization(self, viz_data: VisualizationData):
        """Update the visualization with new data"""
        try:
            # Update data buffers
            self.entropy_buffer.extend(viz_data.entropies)
            self.logits_varentropy_buffer.extend(viz_data.logits_varentropy)
            self.attn_entropy_buffer.extend(viz_data.attn_entropy)
            self.attn_varentropy_buffer.extend(viz_data.attn_varentropy)
            self.time_buffer.extend(viz_data.time_steps)
            
            # Convert deques to lists for plotting
            time_points = list(self.time_buffer)
            entropy_data = list(self.entropy_buffer)
            varentropy_data = list(self.logits_varentropy_buffer)
            attn_entropy_data = list(self.attn_entropy_buffer)
            
            # Update individual plots
            self.update_line(self.entropy_line, time_points, entropy_data)
            self.update_line(self.varentropy_line, time_points, varentropy_data)
            self.update_line(self.attn_entropy_line, time_points, attn_entropy_data)
            
            # Update combined plot
            self.update_line(self.combined_entropy_line, time_points, entropy_data)
            self.update_line(self.combined_varentropy_line, time_points, varentropy_data)
            self.update_line(self.combined_attn_line, time_points, attn_entropy_data)
            
            # Adjust plot limits
            self.adjust_plot_limits()
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating visualization: {str(e)}")
            
    def update_line(self, line, x_data, y_data):
        """Update a single line plot"""
        if len(x_data) == len(y_data) and len(x_data) > 0:
            line.set_data(x_data, y_data)
            
    def adjust_plot_limits(self):
        """Adjust the limits of all plots"""
        for ax in [self.entropy_ax, self.varentropy_ax, self.attn_ax, self.combined_ax]:
            ax.relim()
            ax.autoscale_view()
            
    def on_closing(self):
        """Handle window closing"""
        self.window.destroy()

class VisualizationManager:
    def __init__(self, parent):
        self.parent = parent
        self.visualizer = None
        
    def initialize(self):
        if not self.visualizer:
            self.visualizer = TokenVisualizerWindow(self.parent)
            
    def update(self, stats: Dict):
        """Update visualization with current statistics"""
        if not self.visualizer:
            self.initialize()
            
        # Extract relevant data from stats dictionary
        viz_data = VisualizationData(
            entropies=[stats.get('logits_entropy', 0.0)],
            logits_varentropy=[stats.get('logits_varentropy', 0.0)],
            attn_entropy=[stats.get('attn_entropy', 0.0)],
            attn_varentropy=[stats.get('attn_varentropy', 0.0)],
            tokens=[stats.get('current_token', '')],
            time_steps=[len(self.visualizer.time_buffer)]
        )
        
        self.visualizer.update_visualization(viz_data)

def integrate_visualization(gui_instance):
    """Integrate visualization with GUI instance"""
    viz_manager = VisualizationManager(gui_instance.root)
    viz_manager.initialize()
    gui_instance.viz_manager = viz_manager
    
    # Store original update_stats method
    original_update_stats = gui_instance.update_stats
    
    def update_stats_with_viz(stats):
        """Wrapper to add visualization to stats updates"""
        original_update_stats(stats)
        viz_manager.update(stats)
        
    # Replace update_stats method
    gui_instance.update_stats = update_stats_with_viz
    
    return viz_manager