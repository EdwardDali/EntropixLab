import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from dataclasses import dataclass
from collections import deque
import logging
from typing import List, Dict
from datetime import datetime
import os

logger = logging.getLogger(__name__)

PLOT_STYLES = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.color': '#e9ecef',
    'axes.edgecolor': '#dee2e6',
    'axes.labelcolor': '#495057',
    'xtick.color': '#495057',
    'ytick.color': '#495057',
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
}

@dataclass
class VisualizationData:
    """Simplified data structure for visualization"""
    entropies: List[float]
    logits_varentropy: List[float]
    attn_entropy: List[float]
    attn_varentropy: List[float]
    tokens: List[str]
    time_steps: List[int]
    rolling_entropy: List[float]
    rolling_varentropy: List[float]
    current_strategy: str = ""

class TokenVisualizerWindow:
    def __init__(self, parent=None):
        self.parent = parent
        self.window = tk.Toplevel(self.parent)
        self.window.title("Generation Statistics Visualization")
        self.window.geometry("1400x1000")
        
        # Apply plot styling
        for key, value in PLOT_STYLES.items():
            plt.rcParams[key] = value
        
        # Initialize data buffers
        self.buffer_size = 1000
        self.entropy_buffer = deque(maxlen=self.buffer_size)
        self.logits_varentropy_buffer = deque(maxlen=self.buffer_size)
        self.attn_entropy_buffer = deque(maxlen=self.buffer_size)
        self.attn_varentropy_buffer = deque(maxlen=self.buffer_size)
        self.tokens_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.rolling_entropy_buffer = deque(maxlen=self.buffer_size)
        self.rolling_varentropy_buffer = deque(maxlen=self.buffer_size)
        self.strategy_buffer = deque(maxlen=self.buffer_size)
        
        self.create_gui()
        
        # Update interval in milliseconds
        self.update_interval = 100
        
    def create_gui(self):
        """Create the complete GUI with controls and plots"""
        # Main container
        main_container = ttk.Frame(self.window)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_panel = self.create_control_panel(main_container)
        control_panel.pack(fill=tk.X, pady=(0, 10))
        
        # Plots
        plot_container = ttk.Frame(main_container)
        plot_container.pack(fill=tk.BOTH, expand=True)
        
        self.setup_plots(plot_container)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_container, textvariable=self.status_var)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        
    def create_control_panel(self, parent):
        """Create control panel with buttons and options"""
        panel = ttk.LabelFrame(parent, text="Controls", padding="5")
        
        # Buttons
        btn_frame = ttk.Frame(panel)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            btn_frame,
            text="Save Plot",
            command=self.save_plot
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            btn_frame,
            text="Clear Data",
            command=self.clear_data
        ).pack(side=tk.LEFT, padx=5)
        
        # Plot options
        options_frame = ttk.Frame(panel)
        options_frame.pack(fill=tk.X, pady=5)
        
        # Window size control
        ttk.Label(options_frame, text="Window Size:").pack(side=tk.LEFT, padx=5)
        self.window_size_var = tk.StringVar(value="100")
        window_size_entry = ttk.Entry(options_frame, textvariable=self.window_size_var, width=10)
        window_size_entry.pack(side=tk.LEFT, padx=5)
        
        # Auto-scaling checkbox
        self.autoscale_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Auto-scale",
            variable=self.autoscale_var
        ).pack(side=tk.LEFT, padx=20)
        
        return panel
        
    def setup_plots(self, parent):
        """Create the plot layout with enhanced styling"""
        # Create main figure with subplots
        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Create subplots
        self.entropy_ax = self.fig.add_subplot(gs[0, 0])
        self.varentropy_ax = self.fig.add_subplot(gs[0, 1])
        self.attn_ax = self.fig.add_subplot(gs[1, 0])
        self.combined_ax = self.fig.add_subplot(gs[1, 1])
        
        # Setup canvas with toolbar
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.draw()
        
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()
        
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize plots with improved styling
        self.entropy_line, = self.entropy_ax.plot([], [], 'b-', label='Entropy', linewidth=2)
        self.rolling_entropy_line, = self.entropy_ax.plot([], [], 'b--', label='Rolling Entropy', alpha=0.5)
        
        self.varentropy_line, = self.varentropy_ax.plot([], [], 'r-', label='Varentropy', linewidth=2)
        self.rolling_varentropy_line, = self.varentropy_ax.plot([], [], 'r--', label='Rolling Varentropy', alpha=0.5)
        
        self.attn_entropy_line, = self.attn_ax.plot([], [], 'g-', label='Attention Entropy', linewidth=2)
        
        # Combined plot with multiple lines
        self.combined_entropy_line, = self.combined_ax.plot([], [], 'b-', label='Entropy', linewidth=2, alpha=0.7)
        self.combined_varentropy_line, = self.combined_ax.plot([], [], 'r-', label='Varentropy', linewidth=2, alpha=0.7)
        self.combined_attn_line, = self.combined_ax.plot([], [], 'g-', label='Attn Entropy', linewidth=2, alpha=0.7)
        
        self.setup_plot_layouts()
        
    def setup_plot_layouts(self):
        """Configure the appearance of all plots with enhanced styling"""
        plots = [
            (self.entropy_ax, "Entropy Over Time", "Time", "Entropy"),
            (self.varentropy_ax, "Variance Entropy Over Time", "Time", "Varentropy"),
            (self.attn_ax, "Attention Entropy Over Time", "Time", "Attention Entropy"),
            (self.combined_ax, "Combined Metrics", "Time", "Value")
        ]
        
        for ax, title, xlabel, ylabel in plots:
            ax.set_title(title, pad=10, fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True)
            ax.legend(loc='upper right', framealpha=0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
    def update_visualization(self, viz_data: VisualizationData):
        """Update the visualization with new data"""
        try:
            # Update data buffers
            self.entropy_buffer.extend(viz_data.entropies)
            self.logits_varentropy_buffer.extend(viz_data.logits_varentropy)
            self.attn_entropy_buffer.extend(viz_data.attn_entropy)
            self.attn_varentropy_buffer.extend(viz_data.attn_varentropy)
            self.time_buffer.extend(viz_data.time_steps)
            self.rolling_entropy_buffer.extend(viz_data.rolling_entropy)
            self.rolling_varentropy_buffer.extend(viz_data.rolling_varentropy)
            self.strategy_buffer.append(viz_data.current_strategy)
            
            # Convert deques to lists for plotting
            time_points = list(self.time_buffer)
            
            # Update individual plots
            self.update_line(self.entropy_line, time_points, list(self.entropy_buffer))
            self.update_line(self.rolling_entropy_line, time_points, list(self.rolling_entropy_buffer))
            
            self.update_line(self.varentropy_line, time_points, list(self.logits_varentropy_buffer))
            self.update_line(self.rolling_varentropy_line, time_points, list(self.rolling_varentropy_buffer))
            
            self.update_line(self.attn_entropy_line, time_points, list(self.attn_entropy_buffer))
            
            # Update combined plot
            self.update_line(self.combined_entropy_line, time_points, list(self.entropy_buffer))
            self.update_line(self.combined_varentropy_line, time_points, list(self.logits_varentropy_buffer))
            self.update_line(self.combined_attn_line, time_points, list(self.attn_entropy_buffer))
            
            # Update plot limits if auto-scale is enabled
            if self.autoscale_var.get():
                self.adjust_plot_limits()
            
            # Update status with current strategy
            if viz_data.current_strategy:
                self.status_var.set(f"Current Strategy: {viz_data.current_strategy}")
            
            # Redraw canvas
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating visualization: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            
    def update_line(self, line, x_data, y_data):
        """Update a single line plot"""
        if len(x_data) == len(y_data) and len(x_data) > 0:
            line.set_data(x_data, y_data)
            
    def adjust_plot_limits(self):
        """Adjust the limits of all plots"""
        for ax in [self.entropy_ax, self.varentropy_ax, self.attn_ax, self.combined_ax]:
            ax.relim()
            ax.autoscale_view()
            
    def save_plot(self):
        """Save the current plot to a file"""
        try:
            # Create output directory if it doesn't exist
            output_dir = "visualization_outputs"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"visualization_{timestamp}.png")
            
            # Save figure
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            self.status_var.set(f"Plot saved to {filename}")
            logger.info(f"Plot saved to {filename}")
        except Exception as e:
            self.status_var.set(f"Error saving plot: {str(e)}")
            logger.error(f"Error saving plot: {str(e)}")
            
    def clear_data(self):
        """Clear all data buffers and reset plots"""
        # Clear all data buffers
        for buffer in [self.entropy_buffer, self.logits_varentropy_buffer,
                      self.attn_entropy_buffer, self.attn_varentropy_buffer,
                      self.time_buffer, self.rolling_entropy_buffer,
                      self.rolling_varentropy_buffer, self.strategy_buffer]:
            buffer.clear()
        
        # Reset all lines
        for line in [self.entropy_line, self.rolling_entropy_line,
                    self.varentropy_line, self.rolling_varentropy_line,
                    self.attn_entropy_line, self.combined_entropy_line,
                    self.combined_varentropy_line, self.combined_attn_line]:
            line.set_data([], [])
        
        # Redraw canvas
        self.canvas.draw()
        self.status_var.set("Data cleared")
            
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
            time_steps=[len(self.visualizer.time_buffer)],
            rolling_entropy=[stats.get('rolling_entropy', 0.0)],
            rolling_varentropy=[stats.get('rolling_varentropy', 0.0)],
            current_strategy=stats.get('current_strategy', '')
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
        try:
            # Update GUI stats
            original_update_stats(stats)
            
            # Add rolling statistics if not present
            if 'rolling_entropy' not in stats and hasattr(gui_instance, 'entropy_window'):
                stats['rolling_entropy'] = sum(gui_instance.entropy_window) / len(gui_instance.entropy_window) \
                    if gui_instance.entropy_window else 0.0
                    
            if 'rolling_varentropy' not in stats and hasattr(gui_instance, 'varentropy_window'):
                stats['rolling_varentropy'] = sum(gui_instance.varentropy_window) / len(gui_instance.varentropy_window) \
                    if gui_instance.varentropy_window else 0.0
            
            # Add strategy information
            if hasattr(gui_instance, 'current_strategy'):
                stats['current_strategy'] = gui_instance.current_strategy
            
            # Update visualization
            viz_manager.update(stats)
            
        except Exception as e:
            logger.error(f"Error in visualization update: {str(e)}")
    
    # Replace update_stats method
    gui_instance.update_stats = update_stats_with_viz
    
    # Add keyboard shortcuts
    def add_keyboard_shortcuts():
        def save_plot(event):
            if viz_manager.visualizer:
                viz_manager.visualizer.save_plot()
                
        def clear_data(event):
            if viz_manager.visualizer:
                viz_manager.visualizer.clear_data()
                
        def toggle_autoscale(event):
            if viz_manager.visualizer:
                viz_manager.visualizer.autoscale_var.set(
                    not viz_manager.visualizer.autoscale_var.get()
                )
        
        gui_instance.root.bind('<Control-s>', save_plot)
        gui_instance.root.bind('<Control-l>', clear_data)
        gui_instance.root.bind('<Control-a>', toggle_autoscale)
    
    add_keyboard_shortcuts()
    
    return viz_manager

class StatisticsTracker:
    """Helper class to track and calculate statistics"""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.entropy_window = deque(maxlen=window_size)
        self.varentropy_window = deque(maxlen=window_size)
        self.attn_entropy_window = deque(maxlen=window_size)
        self.strategy_counts = Counter()
        self.total_tokens = 0
        
    def update(self, stats: Dict):
        """Update statistics with new data"""
        # Update windows
        if 'logits_entropy' in stats:
            self.entropy_window.append(stats['logits_entropy'])
            
        if 'logits_varentropy' in stats:
            self.varentropy_window.append(stats['logits_varentropy'])
            
        if 'attn_entropy' in stats:
            self.attn_entropy_window.append(stats['attn_entropy'])
            
        # Update strategy counts
        if 'current_strategy' in stats:
            self.strategy_counts[stats['current_strategy']] += 1
            self.total_tokens += 1
            
    def get_rolling_stats(self) -> Dict:
        """Calculate rolling statistics"""
        return {
            'rolling_entropy': np.mean(list(self.entropy_window)) if self.entropy_window else 0.0,
            'rolling_varentropy': np.mean(list(self.varentropy_window)) if self.varentropy_window else 0.0,
            'rolling_attn_entropy': np.mean(list(self.attn_entropy_window)) if self.attn_entropy_window else 0.0,
            'strategy_distribution': {
                strategy: count/self.total_tokens 
                for strategy, count in self.strategy_counts.items()
            } if self.total_tokens > 0 else {}
        }
        
    def reset(self):
        """Reset all statistics"""
        self.entropy_window.clear()
        self.varentropy_window.clear()
        self.attn_entropy_window.clear()
        self.strategy_counts.clear()
        self.total_tokens = 0

# Add mouse interaction handlers
def add_mouse_interactions(visualizer):
    def on_mouse_move(event):
        if event.inaxes:
            # Get data coordinates
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                # Update status bar with coordinates
                visualizer.status_var.set(f"Time: {x:.1f}, Value: {y:.4f}")
                
    def on_mouse_click(event):
        if event.inaxes and event.dblclick:
            # On double click, create annotation
            ax = event.inaxes
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                ann = ax.annotate(
                    f"({x:.1f}, {y:.4f})",
                    xy=(x, y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->')
                )
                visualizer.canvas.draw()
                
    visualizer.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    visualizer.canvas.mpl_connect('button_press_event', on_mouse_click)

# Add this to the TokenVisualizerWindow.__init__ method:
"""
Add this line after setting up the plots:
add_mouse_interactions(self)
"""

# Add export functionality
def export_data(visualizer, format='csv'):
    """Export visualization data to file"""
    try:
        # Create output directory if it doesn't exist
        output_dir = "visualization_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'csv':
            filename = os.path.join(output_dir, f"visualization_data_{timestamp}.csv")
            data = {
                'time': list(visualizer.time_buffer),
                'entropy': list(visualizer.entropy_buffer),
                'varentropy': list(visualizer.logits_varentropy_buffer),
                'attn_entropy': list(visualizer.attn_entropy_buffer),
                'strategy': list(visualizer.strategy_buffer)
            }
            
            # Convert to pandas DataFrame and save
            import pandas as pd
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            
        elif format == 'json':
            filename = os.path.join(output_dir, f"visualization_data_{timestamp}.json")
            data = {
                'time': list(visualizer.time_buffer),
                'entropy': list(visualizer.entropy_buffer),
                'varentropy': list(visualizer.logits_varentropy_buffer),
                'attn_entropy': list(visualizer.attn_entropy_buffer),
                'strategy': list(visualizer.strategy_buffer),
                'metadata': {
                    'timestamp': timestamp,
                    'window_size': visualizer.buffer_size,
                    'total_samples': len(visualizer.time_buffer)
                }
            }
            
            import json
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
                
        visualizer.status_var.set(f"Data exported to {filename}")
        logger.info(f"Data exported to {filename}")
        
    except Exception as e:
        visualizer.status_var.set(f"Error exporting data: {str(e)}")
        logger.error(f"Error exporting data: {str(e)}")

# Add export button to control panel
"""
Add this to create_control_panel method:
ttk.Button(
    btn_frame,
    text="Export Data",
    command=lambda: export_data(self)
).pack(side=tk.LEFT, padx=5)
"""