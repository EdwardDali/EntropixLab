import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
import threading
from queue import Queue
import logging
from typing import Dict, Union, Tuple, List, Optional
import torch
from collections import Counter
import os
import json
from threading import Lock
import platformdirs
import re
from pathlib import Path
from datetime import datetime
import json
import hashlib

# Import from main_t
from main_t import (
    generate_response,
    AutoModelForCausalLM,
    AutoTokenizer,
    device,
    SamplerConfig,
    SamplerState,
    logger,
    EntropixSampler
)



class ToolTip:
    """Tooltip widget implementation"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(
            self.tooltip,
            text=self.text,
            justify='left',
            background="#ffffe0",
            relief='solid',
            borderwidth=1,
            wraplength=300
        )
        label.pack()

    def hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

# Parameter tooltips
PARAMETER_TOOLTIPS = {
     "basic_sampling": {
        "temp": "Temperature for logits. Higher values increase randomness.",
        "top_p": "Nucleus sampling threshold. Cumulative probability cutoff.",
        "top_k": "Top-k sampling. Number of highest probability tokens to consider.",
        "min_p": "Minimum probability threshold for token selection.",
        "repetition_penalty": "Penalty applied to repeated tokens.",
        "strategy_change_batch_size": "Number of tokens to generate before allowing strategy changes"
    },
    "strategy_thresholds": {
        "argmax_entropy_thresh": "Maximum entropy threshold for ARGMAX (bottom-left quadrant: low entropy, low variance)",
        "sample_entropy_thresh": "Entropy threshold for SAMPLE (bottom-right quadrant: low entropy, high variance)",
        "sample_varentropy_thresh": "Minimum variance threshold for SAMPLE strategy",
        "cot_min_entropy_thresh": "Minimum entropy for INSERT_COT (top-left quadrant: high entropy, low variance)",
        "cot_max_entropy_thresh": "Maximum entropy for INSERT_COT",
        "cot_varentropy_thresh": "Maximum variance threshold for INSERT_COT",
        "resample_min_entropy_thresh": "Minimum entropy for RESAMPLE (top-right quadrant: high entropy, high variance)",
        "resample_varentropy_thresh": "Minimum variance threshold for RESAMPLE",
        "adaptive_radius": "Radius around center point for ADAPTIVE strategy (central region)",
    },
    "adaptive_sampling": {
        "ada_temp_logits": "Temperature adjustment based on logits uncertainty.",
        "ada_temp_attn": "Temperature adjustment based on attention uncertainty.",
        "ada_temp_agree": "Temperature adjustment based on head agreement.",
        "n_adaptive_samples": "Number of samples to generate in adaptive mode."
    },
    "attention_coefficients": {
        "helv_attn_ent_offset": "High Entropy Low Variance attention entropy offset.",
        "helv_attn_ent_coef": "High Entropy Low Variance attention entropy coefficient.",
        "lehv_interaction_strength_offset": "Low Entropy High Variance interaction strength offset.",
        "lehv_interaction_strength_coef": "Low Entropy High Variance interaction strength coefficient.",
        "hehv_attn_ent_coef": "High Entropy High Variance attention entropy coefficient.",
        "hehv_attn_vent_offset": "High Entropy High Variance attention variance offset.",
        "hehv_attn_vent_coef": "High Entropy High Variance attention variance coefficient."
    },
    "rope_parameters": {
        "rope_theta": "Base value for RoPE (Rotary Position Embedding).",
        "rope_scaling": "Scaling factor for RoPE computations.",
        "rope_scale_base": "Base value for RoPE scaling.",
        "rope_scale_factor": "Factor for RoPE scaling calculations."
    },
    "memory_context": {
        "max_ngram_size": "Maximum size of n-grams to track for repetition.",
        "max_ngram_repeat": "Maximum allowed repetitions of any n-gram.",
        "window_size": "Size of the rolling window for statistics.",
        "long_window_size": "Size of the long-term rolling window.",
        "decay_factor": "Decay rate for rolling statistics.",
        "long_decay_factor": "Decay rate for long-term statistics."
    }
}


class ModelDiscovery:
    def __init__(self):
        # Get default HF cache directory
        self.default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        self.custom_model_dir = None
        self.available_models = {}
        self.scan_models()

    def is_valid_model_dir(self, path: Path) -> bool:
        """Check if directory contains a valid model"""
        # For HF cache, look inside the 'snapshots' subdirectory
        if path.name.startswith("models--"):
            # Get the latest snapshot
            snapshots_dir = path / "snapshots"
            if not snapshots_dir.exists():
                return False
            
            # Find the latest snapshot (highest hash)
            snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
            if not snapshot_dirs:
                return False
            
            # Use the latest snapshot
            model_dir = sorted(snapshot_dirs)[-1]
        else:
            model_dir = path

        # Check for required files
        required_files = ['config.json']
        optional_files = ['pytorch_model.bin', 'model.safetensors']
        
        if not any((model_dir / file).exists() for file in required_files):
            return False
            
        # Check for model files (including sharded ones)
        has_model_files = (
            any((model_dir / file).exists() for file in optional_files) or
            list(model_dir.glob("pytorch_model-*.bin")) or
            list(model_dir.glob("model-*.safetensors"))
        )
        
        return has_model_files

    def get_model_path(self, path: Path) -> Path:
        """Get the actual model path from HF cache directory structure"""
        if path.name.startswith("models--"):
            snapshots_dir = path / "snapshots"
            if snapshots_dir.exists():
                snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                if snapshot_dirs:
                    return sorted(snapshot_dirs)[-1]
        return path

    def get_model_info(self, path: Path) -> Optional[Dict[str, str]]:
        """Extract model information from config.json"""
        try:
            # Get the actual model directory
            model_dir = self.get_model_path(path)
            config_path = model_dir / 'config.json'
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Convert models--org--name format to org/name
                model_name = path.name
                if model_name.startswith("models--"):
                    parts = model_name.split("--")
                    if len(parts) >= 3:
                        model_name = f"{parts[1]}/{'/'.join(parts[2:])}"
                
                # Try to get model size
                model_size = None
                if 'n_params' in config:
                    model_size = f"{config['n_params'] / 1e9:.1f}B"
                elif 'num_parameters' in config:
                    model_size = f"{config['num_parameters'] / 1e9:.1f}B"
                elif 'model_type' in config:
                    model_size = config['model_type']
                
                return {
                    'name': model_name,
                    'path': str(model_dir),  # Use the snapshot directory path
                    'architecture': config.get('architectures', ['Unknown'])[0],
                    'size': model_size if model_size else 'Unknown'
                }
        except Exception as e:
            logger.warning(f"Error reading model config at {path}: {e}")
        return None

    def scan_directory(self, base_path: Path) -> Dict[str, Dict[str, str]]:
        """Recursively scan directory for models"""
        models = {}
        
        try:
            # For HF cache, look for directories starting with "models--"
            model_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("models--")]
            
            for model_dir in model_dirs:
                if self.is_valid_model_dir(model_dir):
                    model_info = self.get_model_info(model_dir)
                    if model_info:
                        display_name = f"{model_info['name']} ({model_info['size']})"
                        models[display_name] = model_info
                        logger.info(f"Found model: {display_name} at {model_info['path']}")
                
        except Exception as e:
            logger.error(f"Error scanning directory {base_path}: {e}")
        
        return models

    def set_custom_dir(self, directory: str):
        """Set custom directory for model scanning"""
        if directory and os.path.exists(directory):
            self.custom_model_dir = directory
            self.scan_models()
            logger.info(f"Added custom model directory: {directory}")
        else:
            logger.warning(f"Invalid custom directory: {directory}")

    def scan_models(self):
        """Scan both default and custom directories for models"""
        self.available_models.clear()
        
        # Scan default HF cache directory
        if os.path.exists(self.default_cache_dir):
            logger.info(f"Scanning default cache directory: {self.default_cache_dir}")
            cache_models = self.scan_directory(Path(self.default_cache_dir))
            self.available_models.update(cache_models)
            logger.info(f"Found {len(cache_models)} models in default cache")
        
        # Scan custom directory if set
        if self.custom_model_dir and os.path.exists(self.custom_model_dir):
            logger.info(f"Scanning custom directory: {self.custom_model_dir}")
            custom_models = self.scan_directory(Path(self.custom_model_dir))
            self.available_models.update(custom_models)
            logger.info(f"Found {len(custom_models)} models in custom directory")

        if not self.available_models:
            logger.warning("No models found in any directory")

class EntropixTGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Entropix Text Generator")
        
        # Initialize model discovery
        self.model_discovery = ModelDiscovery()
        
        # Initialize variables
        self.model = None
        self.tokenizer = None
        self.sampler_config = None
        self.generation_thread = None
        self.stop_generation = False
        self.response_queue = Queue()
        self.strategy_counter = Counter()
        self.stats_lock = Lock()
        
        # Initialize all parameter variables before GUI setup
        self.initialize_parameter_vars()
        
        # Create main notebook for primary tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create widgets and setup GUI
        self.create_widgets()

        # Create directories if they don't exist
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        self.config_dir = Path("configs")
        self.config_dir.mkdir(exist_ok=True)
        
        # Track current model info
        self.current_model_hash = None
        self.current_model_name = None

    def create_widgets(self):
        """Create and setup all GUI widgets"""
        # Create main tab
        self.main_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main")
        
        # Create controls tab
        self.controls_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.controls_tab, text="Controls")
        
        # Setup main tab components
        self.create_main_tab()
        
        # Setup controls tab components
        self.create_controls_tab()

    

    def create_main_tab(self):
        """Create main tab with model selection, input, and output areas"""
        # Model frame
        model_frame = ttk.LabelFrame(self.main_tab, text="Model", padding="5")
        model_frame.pack(fill="x", padx=5, pady=5)
        self.create_model_selector()
        
        # Input frame
        input_frame = ttk.LabelFrame(self.main_tab, text="Input", padding="5")
        input_frame.pack(fill="x", padx=5, pady=5)
        self.create_input_area(input_frame)
        
        # Output frame
        output_frame = ttk.LabelFrame(self.main_tab, text="Output", padding="5")
        output_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.create_output_areas(output_frame)

    def create_controls_tab(self):
        """Create organized parameter controls in the Controls tab"""
        # Create notebook for tabbed organization within Controls tab
        control_notebook = ttk.Notebook(self.controls_tab)
        control_notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Basic Sampling tab
        basic_frame = ttk.Frame(control_notebook)
        control_notebook.add(basic_frame, text="Basic Sampling")
        self.create_basic_controls(basic_frame)

        # Entropy & Strategy tab
        entropy_frame = ttk.Frame(control_notebook)
        control_notebook.add(entropy_frame, text="Entropy & Strategy")
        self.create_entropy_controls(entropy_frame)

        # Adaptive Sampling tab
        adaptive_frame = ttk.Frame(control_notebook)
        control_notebook.add(adaptive_frame, text="Adaptive")
        self.create_adaptive_controls(adaptive_frame)

        # Attention Coefficients tab
        attention_frame = ttk.Frame(control_notebook)
        control_notebook.add(attention_frame, text="Attention")
        self.create_attention_controls(attention_frame)

        # RoPE & Position tab
        rope_frame = ttk.Frame(control_notebook)
        control_notebook.add(rope_frame, text="RoPE")
        self.create_rope_controls(rope_frame)

        # Memory & Context tab
        memory_frame = ttk.Frame(control_notebook)
        control_notebook.add(memory_frame, text="Memory")
        self.create_memory_controls(memory_frame)

    def create_save_buttons(self):
        """Add save buttons to the interface"""
        save_frame = ttk.Frame(self.main_tab)
        save_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(
            save_frame,
            text="Save Configuration",
            command=self.save_current_config
        ).pack(side="left", padx=5)
        
        ttk.Button(
            save_frame,
            text="Save Output",
            command=self.save_output
        ).pack(side="left", padx=5)

    def save_current_config(self):
        """Save current configuration to a timestamped JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config = {
            "basic_sampling": {
                "temp": self.temp_var.get(),
                "top_p": self.top_p_var.get(),
                "top_k": self.top_k_var.get(),
                "min_p": self.min_p_var.get(),
                "repetition_penalty": self.repetition_penalty_var.get()
            },
            "entropy_thresholds": {
                "low_ent_thresh": self.low_ent_thresh_var.get(),
                "med_ent_thresh": self.med_ent_thresh_var.get(),
                "high_ent_thresh": self.high_ent_thresh_var.get(),
                "high_vent_thresh": self.high_vent_thresh_var.get(),
                "varentropy_threshold": self.varentropy_threshold_var.get()
            },
            "adaptive_sampling": {
                "ada_temp_logits": self.ada_temp_logits_var.get(),
                "ada_temp_attn": self.ada_temp_attn_var.get(),
                "ada_temp_agree": self.ada_temp_agree_var.get(),
                "n_adaptive_samples": self.n_adaptive_samples_var.get()
            },
            "rope_parameters": {
                "rope_theta": self.rope_theta_var.get(),
                "rope_scaling": self.rope_scaling_var.get(),
                "rope_scale_base": self.rope_scale_base_var.get(),
                "rope_scale_factor": self.rope_scale_factor_var.get()
            },
            "attention_coefficients": {
                "helv_attn_ent_offset": self.helv_attn_ent_offset_var.get(),
                "helv_attn_ent_coef": self.helv_attn_ent_coef_var.get(),
                "lehv_interaction_strength_offset": self.lehv_interaction_strength_offset_var.get(),
                "lehv_interaction_strength_coef": self.lehv_interaction_strength_coef_var.get(),
                "hehv_attn_ent_coef": self.hehv_attn_ent_coef_var.get(),
                "hehv_attn_vent_offset": self.hehv_attn_vent_offset_var.get(),
                "hehv_attn_vent_coef": self.hehv_attn_vent_coef_var.get()
            },
            "memory_context": {
                "max_ngram_size": self.max_ngram_size_var.get(),
                "max_ngram_repeat": self.max_ngram_repeat_var.get(),
                "window_size": self.window_size_var.get(),
                "long_window_size": self.long_window_size_var.get(),
                "decay_factor": self.decay_factor_var.get(),
                "long_decay_factor": self.long_decay_factor_var.get()
            }
        }
        
        # Save as both current config and timestamped config
        try:
            # Save as current config (for loading on startup)
            current_config_path = self.config_dir / "current_config.json"
            with open(current_config_path, "w") as f:
                json.dump(config, f, indent=4)
            
            # Save timestamped version
            config_path = self.config_dir / f"config_{timestamp}.json"
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            
            self.output_text.insert("end", f"\nConfiguration saved to {config_path}\n")
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            self.output_text.insert("end", f"\nError saving configuration: {str(e)}\n")
            logger.error(f"Error saving configuration: {str(e)}")

    def save_output(self):
        """Save current output to a timestamped text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            # Get the entire output text
            output_text = self.output_text.get("1.0", "end-1c")
            
            # Add generation statistics
            output_text += "\n\n=== Generation Statistics ===\n"
            total_tokens = sum(self.strategy_counter.values())
            if total_tokens > 0:
                for strategy, count in self.strategy_counter.most_common():
                    percentage = (count / total_tokens) * 100
                    output_text += f"{strategy}: {count} ({percentage:.1f}%)\n"
            
            # Add configuration used
            output_text += "\n=== Configuration Used ===\n"
            current_config = {
                "basic_sampling": {
                    "temp": self.temp_var.get(),
                    "top_p": self.top_p_var.get(),
                    "top_k": self.top_k_var.get(),
                    "min_p": self.min_p_var.get(),
                    "repetition_penalty": self.repetition_penalty_var.get()
                },
                # ... (add other configuration sections)
            }
            output_text += json.dumps(current_config, indent=4)
            
            # Save the file
            output_path = self.output_dir / f"output_{timestamp}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_text)
            
            self.output_text.insert("end", f"\nOutput saved to {output_path}\n")
            logger.info(f"Output saved to {output_path}")
        except Exception as e:
            self.output_text.insert("end", f"\nError saving output: {str(e)}\n")
            logger.error(f"Error saving output: {str(e)}")

    def load_config(self):
        """Load configuration from file"""
        try:
            # Try to load the current configuration
            current_config_path = self.config_dir / "current_config.json"
            if not current_config_path.exists():
                logger.info("No saved configuration found, using defaults")
                return
                
            with open(current_config_path, "r") as f:
                config = json.load(f)
                
            # Update all GUI variables from loaded config
            for section, params in config.items():
                for param, value in params.items():
                    var = getattr(self, f"{param}_var", None)
                    if var:
                        var.set(value)
                        
            logger.info("Configuration loaded successfully")
            self.update_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def setup_gui(self):
        """Setup main GUI with tabs for main interface and controls"""
        # Create main notebook for primary tabs
        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create main interface tab
        main_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(main_tab, text="Main")
        
        # Create controls tab
        controls_tab = ttk.Frame(self.main_notebook)
        self.main_notebook.add(controls_tab, text="Controls")
        
        # Setup main interface components
        self.setup_main_interface(main_tab)
        
        # Setup controls interface
        self.setup_controls_interface(controls_tab)

    def setup_main_interface(self, parent):
        """Setup the main interface tab with model selection, input, and output"""
        # Model frame
        model_frame = ttk.LabelFrame(parent, text="Model", padding="5")
        model_frame.pack(fill="x", padx=5, pady=5)
        
        # Model selection frame
        model_select_frame = ttk.Frame(model_frame)
        model_select_frame.pack(fill="x", pady=5)
        
        ttk.Label(model_select_frame, text="Model:").pack(side="left", padx=5)
        
        # Create model dropdown
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_select_frame, 
            textvariable=self.model_var,
            width=50
        )
        self.model_combo.pack(side="left", padx=5, fill="x", expand=True)
        
        # Model buttons frame
        btn_frame = ttk.Frame(model_frame)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(
            btn_frame,
            text="Scan Default Directory",
            command=self.rescan_models
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="Select Custom Directory",
            command=self.select_custom_dir
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="Load Model",
            command=self.load_selected_model
        ).pack(side="left", padx=5)
        
        # Model info display
        self.model_info = ttk.Label(model_frame, text="No model loaded")
        self.model_info.pack(fill="x", padx=5)
        
        # Input frame
        input_frame = ttk.LabelFrame(parent, text="Input", padding="5")
        input_frame.pack(fill="x", padx=5, pady=5)
        
        # Input text area
        self.prompt_input = scrolledtext.ScrolledText(input_frame, height=5)
        self.prompt_input.pack(fill="x", expand=True, padx=5, pady=5)
        
        # Input buttons
        input_btn_frame = ttk.Frame(input_frame)
        input_btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(input_btn_frame, text="Generate", command=self.start_generation).pack(side="left", padx=5)
        ttk.Button(input_btn_frame, text="Stop", command=self.stop_generation_request).pack(side="left", padx=5)
        ttk.Button(input_btn_frame, text="Clear", command=self.clear_output).pack(side="left", padx=5)
        
        # Output frame with increased height
        output_frame = ttk.LabelFrame(parent, text="Output", padding="5")
        output_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(output_frame, height=30)
        self.output_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Statistics area
        stats_frame = ttk.LabelFrame(output_frame, text="Statistics", padding="5")
        stats_frame.pack(fill="x", padx=5, pady=5)
        
        # Create three columns for statistics
        left_stats = ttk.Frame(stats_frame)
        left_stats.pack(side="left", fill="both", expand=True)
        
        middle_stats = ttk.Frame(stats_frame)
        middle_stats.pack(side="left", fill="both", expand=True)
        
        right_stats = ttk.Frame(stats_frame)
        right_stats.pack(side="left", fill="both", expand=True)
        
        # Initialize stat_labels dictionary
        self.stat_labels = {}
        
        # Strategy counter in left column
        self.strategy_var = tk.StringVar(value="Strategy Usage: N/A")
        ttk.Label(left_stats, textvariable=self.strategy_var).pack(anchor="w")
        
        # Entropy metrics in middle column
        entropy_metrics = ["Entropy", "Varentropy", "Attn Entropy"]
        for metric in entropy_metrics:
            var = tk.StringVar(value=f"{metric}: N/A")
            self.stat_labels[metric] = var
            ttk.Label(middle_stats, textvariable=var).pack(anchor="w")
        
        # Additional metrics in right column
        additional_metrics = ["Rolling Entropy", "Rolling Varentropy", "Current Strategy"]
        for metric in additional_metrics:
            var = tk.StringVar(value=f"{metric}: N/A")
            self.stat_labels[metric] = var
            ttk.Label(right_stats, textvariable=var).pack(anchor="w")

    def setup_controls_interface(self, parent):
        """Setup the controls interface tab with all parameter controls"""
        # Create notebook for control categories
        control_notebook = ttk.Notebook(parent)
        control_notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create frames for each control category
        basic_frame = ttk.Frame(control_notebook)
        entropy_frame = ttk.Frame(control_notebook)
        adaptive_frame = ttk.Frame(control_notebook)
        attention_frame = ttk.Frame(control_notebook)
        rope_frame = ttk.Frame(control_notebook)
        memory_frame = ttk.Frame(control_notebook)
        
        # Add frames to notebook
        control_notebook.add(basic_frame, text="Basic Sampling")
        control_notebook.add(entropy_frame, text="Entropy & Strategy")
        control_notebook.add(adaptive_frame, text="Adaptive")
        control_notebook.add(attention_frame, text="Attention")
        control_notebook.add(rope_frame, text="RoPE")
        control_notebook.add(memory_frame, text="Memory")
        
        # Create controls in each frame
        self.create_basic_controls(basic_frame)
        self.create_entropy_controls(entropy_frame)
        self.create_adaptive_controls(adaptive_frame)
        self.create_attention_controls(attention_frame)
        self.create_rope_controls(rope_frame)
        self.create_memory_controls(memory_frame)

    def initialize_parameter_vars(self):
        """Initialize all parameter variables with defaults from main_t"""
        # Basic sampling parameters
        self.temp_var = tk.DoubleVar(value=0.666)
        self.top_p_var = tk.DoubleVar(value=0.90)
        self.top_k_var = tk.IntVar(value=27)
        self.min_p_var = tk.DoubleVar(value=0.05)
        self.repetition_penalty_var = tk.DoubleVar(value=1.2)

        # Entropy thresholds
        self.low_ent_thresh_var = tk.DoubleVar(value=0.1)
        self.med_ent_thresh_var = tk.DoubleVar(value=1.8)
        self.high_ent_thresh_var = tk.DoubleVar(value=2.5)
        self.high_vent_thresh_var = tk.DoubleVar(value=3.0)
        self.varentropy_threshold_var = tk.DoubleVar(value=0.1)

        # Adaptive sampling coefficients
        self.ada_temp_logits_var = tk.DoubleVar(value=0.2)
        self.ada_temp_attn_var = tk.DoubleVar(value=0.3)
        self.ada_temp_agree_var = tk.DoubleVar(value=0.1)
        self.n_adaptive_samples_var = tk.IntVar(value=3)

        # Add new adaptive scoring parameters
        self.ada_score_logits_ent_var = tk.DoubleVar(value=0.1)
        self.ada_score_attn_ent_var = tk.DoubleVar(value=0.2)
        self.ada_score_logits_vent_var = tk.DoubleVar(value=0.3)
        self.ada_score_attn_vent_var = tk.DoubleVar(value=0.4)
        self.ada_score_agree_var = tk.DoubleVar(value=0.5)
        self.ada_score_int_var = tk.DoubleVar(value=0.6)

        # RoPE parameters
        self.rope_theta_var = tk.DoubleVar(value=10000.0)
        self.rope_scaling_var = tk.DoubleVar(value=1.0)
        self.rope_scale_base_var = tk.DoubleVar(value=8.0)
        self.rope_scale_factor_var = tk.DoubleVar(value=0.25)

        # Attention coefficients
        self.helv_attn_ent_offset_var = tk.DoubleVar(value=1.3)
        self.helv_attn_ent_coef_var = tk.DoubleVar(value=0.2)
        self.lehv_interaction_strength_offset_var = tk.DoubleVar(value=1.2)
        self.lehv_interaction_strength_coef_var = tk.DoubleVar(value=0.3)
        self.hehv_attn_ent_coef_var = tk.DoubleVar(value=0.2)
        self.hehv_attn_vent_offset_var = tk.DoubleVar(value=2.0)
        self.hehv_attn_vent_coef_var = tk.DoubleVar(value=0.5)

        # Memory and window parameters
        self.max_ngram_size_var = tk.IntVar(value=5)
        self.max_ngram_repeat_var = tk.IntVar(value=3)
        self.window_size_var = tk.IntVar(value=50)
        self.long_window_size_var = tk.IntVar(value=500)
        self.decay_factor_var = tk.DoubleVar(value=0.95)
        self.long_decay_factor_var = tk.DoubleVar(value=0.95)

        # Add to initialize_parameter_vars
        self.strategy_change_batch_size_var = tk.IntVar(value=1)

         # Strategy-specific thresholds adjusted for observed entropy ranges
        self.argmax_entropy_thresh_var = tk.DoubleVar(value=0.5)
        
        self.sample_min_entropy_thresh_var = tk.DoubleVar(value=0.5)
        self.sample_max_entropy_thresh_var = tk.DoubleVar(value=1.5)
        self.sample_varentropy_thresh_var = tk.DoubleVar(value=2.0)  # Increased for observed values
        
        self.cot_min_entropy_thresh_var = tk.DoubleVar(value=1.5)
        self.cot_max_entropy_thresh_var = tk.DoubleVar(value=2.5)
        self.cot_varentropy_thresh_var = tk.DoubleVar(value=2.0)  # Increased for observed values
        
        self.resample_min_entropy_thresh_var = tk.DoubleVar(value=1.0)
        self.resample_max_entropy_thresh_var = tk.DoubleVar(value=2.5)
        self.resample_varentropy_thresh_var = tk.DoubleVar(value=4.0)  # Adjusted for high varentropy
        
        self.adaptive_entropy_thresh_var = tk.DoubleVar(value=2.0)
        self.adaptive_varentropy_thresh_var = tk.DoubleVar(value=4.0)  # Adjusted for high varentropy

            # Adaptive center point parameters
        self.adaptive_entropy_center_var = tk.DoubleVar(value=2.0)
        self.adaptive_varentropy_center_var = tk.DoubleVar(value=2.0)
        self.adaptive_radius_var = tk.DoubleVar(value=0.5)

        # Strategy positioning parameters
        self.quadrant_separation_var = tk.DoubleVar(value=1.0)
        self.boundary_softness_var = tk.DoubleVar(value=0.3)

        # Strategy-specific parameters
        self.argmax_range_var = tk.DoubleVar(value=0.3)
        self.sample_variance_threshold_var = tk.DoubleVar(value=0.5)
        self.cot_entropy_range_var = tk.DoubleVar(value=0.8)
        self.resample_min_variance_var = tk.DoubleVar(value=1.0)

    def create_parameter_controls(self, parent):
        """Create organized parameter controls with additional tab"""
        # Create notebook for tabbed organization
        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # Basic Sampling tab
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Sampling")
        self.create_basic_controls(basic_frame)

        # Entropy & Strategy tab
        entropy_frame = ttk.Frame(notebook)
        notebook.add(entropy_frame, text="Entropy & Strategy")
        self.create_entropy_controls(entropy_frame)

        # Adaptive Sampling tab
        adaptive_frame = ttk.Frame(notebook)
        notebook.add(adaptive_frame, text="Adaptive")
        self.create_adaptive_controls(adaptive_frame)

        # Attention Coefficients tab
        attention_frame = ttk.Frame(notebook)
        notebook.add(attention_frame, text="Attention")
        self.create_attention_controls(attention_frame)

        # RoPE & Position tab
        rope_frame = ttk.Frame(notebook)
        notebook.add(rope_frame, text="RoPE")
        self.create_rope_controls(rope_frame)

        # Memory & Context tab
        memory_frame = ttk.Frame(notebook)
        notebook.add(memory_frame, text="Memory")
        self.create_memory_controls(memory_frame)

        # Advanced Settings tab (new)
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Advanced")
        self.create_advanced_controls(advanced_frame)

    def create_advanced_controls(self, parent):
        """Create advanced settings controls"""
        # You can move some of the less frequently used controls here
        pass

    def create_model_selector(self):
        """Create model selection interface"""
        # Create frame in main tab
        frame = ttk.Frame(self.main_tab)
        frame.pack(fill="x", padx=5, pady=5)
        
        # Model selection frame
        model_select_frame = ttk.Frame(frame)
        model_select_frame.pack(fill="x", pady=5)
        
        ttk.Label(model_select_frame, text="Model:").pack(side="left", padx=5)
        
        # Create model dropdown
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            model_select_frame, 
            textvariable=self.model_var,
            width=50
        )
        self.model_combo.pack(side="left", padx=5, fill="x", expand=True)
        
        # Buttons frame
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(
            btn_frame,
            text="Scan Default Directory",
            command=self.rescan_models
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="Select Custom Directory",
            command=self.select_custom_dir
        ).pack(side="left", padx=5)
        
        ttk.Button(
            btn_frame,
            text="Load Model",
            command=self.load_selected_model
        ).pack(side="left", padx=5)
        
        # Model info display
        self.model_info = ttk.Label(frame, text="No model loaded")
        self.model_info.pack(fill="x", padx=5)
        
        # Initial model scan
        self.update_model_list()

    def create_input_area(self, parent):
        """Create the input area with prompt textbox"""
        # Input text area with reduced height
        self.prompt_input = scrolledtext.ScrolledText(parent, height=3)  # Reduced height
        self.prompt_input.pack(fill="both", expand=True)

        # Generation controls
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="Generate", command=self.start_generation).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Stop", command=self.stop_generation_request).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_output).pack(side="left", padx=5)

    def create_output_areas(self, parent):
        """Create output and statistics areas with right-side stats column"""
        # Main container for output and stats
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)
        container.grid_columnconfigure(0, weight=1)  # Output column expands
        container.grid_columnconfigure(1, weight=0)  # Stats column fixed width
        
        # Left side: Output text area
        output_frame = ttk.Frame(container)
        output_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(0, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(output_frame, height=35)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        
        # Right side: Statistics panel with fixed width
        stats_frame = ttk.LabelFrame(container, text="Generation Statistics", padding=5, width=200)
        stats_frame.grid(row=0, column=1, sticky="nsew")
        stats_frame.grid_propagate(False)  # Prevent frame from shrinking
        
        # Strategy Usage Section
        strategy_frame = ttk.LabelFrame(stats_frame, text="Strategy Usage", padding=5)
        strategy_frame.pack(fill="x", pady=(0, 10))
        
        # Use tk.Text instead of ttk.Text
        self.strategy_stats = tk.Text(strategy_frame, height=8, width=25, wrap="word")
        self.strategy_stats.pack(fill="x")
        self.strategy_stats.configure(state="disabled")  # Make it read-only by default
        
        # CoT Statistics Section
        cot_frame = ttk.LabelFrame(stats_frame, text="Chain of Thought", padding=5)
        cot_frame.pack(fill="x", pady=(0, 10))
        
        self.cot_count_var = tk.StringVar(value="Insertions: 0")
        self.cot_ratio_var = tk.StringVar(value="Ratio: 0%")
        
        ttk.Label(cot_frame, textvariable=self.cot_count_var).pack(fill="x")
        ttk.Label(cot_frame, textvariable=self.cot_ratio_var).pack(fill="x")
        
        #  Add score display to State frame
        state_frame = ttk.LabelFrame(stats_frame, text="Current State", padding=5)
        state_frame.pack(fill="x", pady=(0, 10))
        
        stats_vars = {
            "Entropy": "N/A",
            "Varentropy": "N/A",
            "Attn Entropy": "N/A",
            "Rolling Entropy": "N/A",
            "Rolling Varentropy": "N/A",
            "Current Strategy": "N/A",
            "Sample Score": "N/A",  # New field for adaptive sampling score
            "Agreement": "N/A",     # New field for head agreement
            "Interaction": "N/A"    # New field for interaction strength
        }

        
        self.stat_labels = {}
        for name, initial_value in stats_vars.items():
            var = tk.StringVar(value=f"{name}: {initial_value}")
            self.stat_labels[name] = var
            ttk.Label(state_frame, textvariable=var).pack(fill="x")

    def update_model_list(self):
        """Update the model dropdown with available models"""
        model_list = sorted(self.model_discovery.available_models.keys())
        self.model_combo['values'] = model_list
        
        if model_list:
            self.model_var.set(model_list[0])
            
        # Update status
        total_models = len(model_list)
        self.model_info.config(
            text=f"Found {total_models} model{'s' if total_models != 1 else ''}"
        )

    def rescan_models(self):
        """Rescan for available models"""
        self.model_discovery.scan_models()
        self.update_model_list()

    def select_custom_dir(self):
        """Open directory selector and scan for models"""
        directory = filedialog.askdirectory(
            title="Select Model Directory",
            mustexist=True
        )
        if directory:
            self.model_discovery.set_custom_dir(directory)
            self.update_model_list()

    def load_selected_model(self):
        """Load the selected model and its associated configuration"""
        selected = self.model_var.get()
        if not selected:
            self.output_text.insert("end", "Please select a model first.\n")
            return
            
        model_info = self.model_discovery.available_models.get(selected)
        if not model_info:
            self.output_text.insert("end", "Invalid model selection.\n")
            return
            
        self.output_text.insert("end", f"Loading model from {model_info['path']}...\n")
        self.root.update_idletasks()
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_info['path'],
                local_files_only=True,
                attn_implementation="eager"
            ).to(device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_info['path'],
                local_files_only=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Set current model information
            self.current_model_name = model_info['name']
            self.current_model_hash = self.get_model_hash(model_info['path'])
            
            self.sampler_config = SamplerConfig(self.tokenizer)
            
            # Update model info display
            model_size = sum(p.numel() for p in self.model.parameters()) / 1e6
            self.model_info.config(
                text=f"Model loaded: {model_info['name']} ({model_size:.1f}M parameters)"
            )
            
            self.output_text.insert("end", "Model loaded successfully!\n")
            
            # Load model-specific configuration
            self.load_model_config()
            
        except Exception as e:
            self.output_text.insert("end", f"Error loading model: {str(e)}\n")
            logger.error(f"Error loading model: {str(e)}")

    def get_model_hash(self, model_path: str) -> str:
        """Generate a unique hash for the model based on its path"""
        return hashlib.md5(str(model_path).encode()).hexdigest()[:8]

    def get_model_config_path(self) -> Path:
        """Get the configuration file path for the current model"""
        if not self.current_model_hash:
            return None
        return self.config_dir / f"config_{self.current_model_hash}.json"
    
    def save_model_config(self):
        """Save configuration for the current model"""
        if not self.current_model_hash:
            return
        
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(exist_ok=True)
            
            config = {
                "model_name": self.current_model_name,
                "model_hash": self.current_model_hash,
                "last_updated": datetime.now().isoformat(),
                "parameters": {
                    "basic_sampling": {
                        "temp": self.temp_var.get(),
                        "top_p": self.top_p_var.get(),
                        "top_k": self.top_k_var.get(),
                        "min_p": self.min_p_var.get(),
                        "repetition_penalty": self.repetition_penalty_var.get()
                    },
                    "entropy_thresholds": {
                        "low_ent_thresh": self.low_ent_thresh_var.get(),
                        "med_ent_thresh": self.med_ent_thresh_var.get(),
                        "high_ent_thresh": self.high_ent_thresh_var.get(),
                        "high_vent_thresh": self.high_vent_thresh_var.get(),
                        "varentropy_threshold": self.varentropy_threshold_var.get()
                    },
                    "adaptive_sampling": {
                        "ada_temp_logits": self.ada_temp_logits_var.get(),
                        "ada_temp_attn": self.ada_temp_attn_var.get(),
                        "ada_temp_agree": self.ada_temp_agree_var.get(),
                        "n_adaptive_samples": self.n_adaptive_samples_var.get()
                    },
                    "rope_parameters": {
                        "rope_theta": self.rope_theta_var.get(),
                        "rope_scaling": self.rope_scaling_var.get(),
                        "rope_scale_base": self.rope_scale_base_var.get(),
                        "rope_scale_factor": self.rope_scale_factor_var.get()
                    },
                    "attention_coefficients": {
                        "helv_attn_ent_offset": self.helv_attn_ent_offset_var.get(),
                        "helv_attn_ent_coef": self.helv_attn_ent_coef_var.get(),
                        "lehv_interaction_strength_offset": self.lehv_interaction_strength_offset_var.get(),
                        "lehv_interaction_strength_coef": self.lehv_interaction_strength_coef_var.get(),
                        "hehv_attn_ent_coef": self.hehv_attn_ent_coef_var.get(),
                        "hehv_attn_vent_offset": self.hehv_attn_vent_offset_var.get(),
                        "hehv_attn_vent_coef": self.hehv_attn_vent_coef_var.get()
                    },
                    "memory_context": {
                        "max_ngram_size": self.max_ngram_size_var.get(),
                        "max_ngram_repeat": self.max_ngram_repeat_var.get(),
                        "window_size": self.window_size_var.get(),
                        "long_window_size": self.long_window_size_var.get(),
                        "decay_factor": self.decay_factor_var.get(),
                        "long_decay_factor": self.long_decay_factor_var.get()

                    },

                    "strategy_thresholds": {
                        "argmax_entropy_thresh": self.argmax_entropy_thresh_var.get(),
                        
                        "sample_min_entropy_thresh": self.sample_min_entropy_thresh_var.get(),
                        "sample_max_entropy_thresh": self.sample_max_entropy_thresh_var.get(),
                        "sample_varentropy_thresh": self.sample_varentropy_thresh_var.get(),
                        
                        "cot_min_entropy_thresh": self.cot_min_entropy_thresh_var.get(),
                        "cot_max_entropy_thresh": self.cot_max_entropy_thresh_var.get(),
                        "cot_varentropy_thresh": self.cot_varentropy_thresh_var.get(),
                        
                        "resample_min_entropy_thresh": self.resample_min_entropy_thresh_var.get(),
                        "resample_max_entropy_thresh": self.resample_max_entropy_thresh_var.get(),
                        "resample_varentropy_thresh": self.resample_varentropy_thresh_var.get(),
                        
                        "adaptive_entropy_thresh": self.adaptive_entropy_thresh_var.get(),
                        "adaptive_varentropy_thresh": self.adaptive_varentropy_thresh_var.get()
                    },

                    "core_entropy": {
                        "low_ent_thresh": self.low_ent_thresh_var.get(),
                        "med_ent_thresh": self.med_ent_thresh_var.get(),
                        "high_ent_thresh": self.high_ent_thresh_var.get(),
                        "high_vent_thresh": self.high_vent_thresh_var.get(),
                        "varentropy_threshold": self.varentropy_threshold_var.get()
                    }                   
                }
            }
            
            config_path = self.get_model_config_path()
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            logger.info(f"Configuration saved for model {self.current_model_name}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")

    def check_saved_configs(self):
        """Check for existing configurations in the config directory"""
        try:
            # Ensure config directory exists
            self.config_dir.mkdir(exist_ok=True)
            
            # List all config files
            config_files = list(self.config_dir.glob("config_*.json"))
            
            if config_files:
                logger.info(f"Found {len(config_files)} existing configurations:")
                for config_file in config_files:
                    try:
                        with open(config_file, "r") as f:
                            config = json.load(f)
                        logger.info(f"  - {config['model_name']} (Last updated: {config['last_updated']})")
                    except Exception as e:
                        logger.warning(f"Error reading config file {config_file}: {str(e)}")
            else:
                logger.info("No existing configurations found")
                
            return config_files
        except Exception as e:
            logger.error(f"Error checking configurations: {str(e)}")
            return []

    def load_model_config(self):
        """Load configuration for the current model"""
        try:
            config_path = self.get_model_config_path()
            if not config_path or not config_path.exists():
                logger.info(f"No saved configuration found for model {self.current_model_name}")
                return
                
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Update all GUI variables from loaded config
            params = config.get("parameters", {})
            for section, section_params in params.items():
                for param, value in section_params.items():
                    var = getattr(self, f"{param}_var", None)
                    if var:
                        var.set(value)
            
            logger.info(f"Configuration loaded for model {self.current_model_name}")
            self.update_config()
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def update_config(self):
        """Update sampler config from GUI values and save"""
        if self.sampler_config:
            # Update sampler config with current GUI values
            # ... (existing update code) ...
            
            # Automatically save the configuration
            self.save_model_config()


    def update_config(self):
        """Update sampler config from GUI values"""
        if self.sampler_config:
            # Basic sampling parameters
            self.sampler_config.temp = self.temp_var.get()
            self.sampler_config.top_p = self.top_p_var.get()
            self.sampler_config.top_k = self.top_k_var.get()
            self.sampler_config.min_p = self.min_p_var.get()
            self.sampler_config.repetition_penalty = self.repetition_penalty_var.get()
            self.sampler_config.strategy_change_batch_size = self.strategy_change_batch_size_var.get()
            
            # Entropy thresholds
            self.sampler_config.low_ent_thresh = self.low_ent_thresh_var.get()
            self.sampler_config.med_ent_thresh = self.med_ent_thresh_var.get()
            self.sampler_config.high_ent_thresh = self.high_ent_thresh_var.get()
            self.sampler_config.high_vent_thresh = self.high_vent_thresh_var.get()
            self.sampler_config.varentropy_threshold = self.varentropy_threshold_var.get()
            
            # Adaptive sampling parameters
            self.sampler_config.ada_temp_logits = self.ada_temp_logits_var.get()
            self.sampler_config.ada_temp_attn = self.ada_temp_attn_var.get()
            self.sampler_config.ada_temp_agree = self.ada_temp_agree_var.get()
            self.sampler_config.n_adaptive_samples = self.n_adaptive_samples_var.get()

            # RoPE parameters
            self.sampler_config.rope_theta = self.rope_theta_var.get()
            self.sampler_config.rope_scaling = self.rope_scaling_var.get()
            self.sampler_config.rope_scale_base = self.rope_scale_base_var.get()
            self.sampler_config.rope_scale_factor = self.rope_scale_factor_var.get()
            
            # Attention coefficients
            self.sampler_config.helv_attn_ent_offset = self.helv_attn_ent_offset_var.get()
            self.sampler_config.helv_attn_ent_coef = self.helv_attn_ent_coef_var.get()
            self.sampler_config.lehv_interaction_strength_offset = self.lehv_interaction_strength_offset_var.get()
            self.sampler_config.lehv_interaction_strength_coef = self.lehv_interaction_strength_coef_var.get()
            self.sampler_config.hehv_attn_ent_coef = self.hehv_attn_ent_coef_var.get()
            self.sampler_config.hehv_attn_vent_offset = self.hehv_attn_vent_offset_var.get()
            self.sampler_config.hehv_attn_vent_coef = self.hehv_attn_vent_coef_var.get()
            
            # Memory and window parameters
            self.sampler_config.max_ngram_size = self.max_ngram_size_var.get()
            self.sampler_config.max_ngram_repeat = self.max_ngram_repeat_var.get()
            self.sampler_config.window_size = self.window_size_var.get()
            self.sampler_config.long_window_size = self.long_window_size_var.get()
            self.sampler_config.decay_factor = self.decay_factor_var.get()
            self.sampler_config.long_decay_factor = self.long_decay_factor_var.get()

             # Strategy thresholds
            self.sampler_config.argmax_entropy_thresh = self.argmax_entropy_thresh_var.get()
            self.sampler_config.sample_min_entropy_thresh = self.sample_min_entropy_thresh_var.get()
            self.sampler_config.sample_max_entropy_thresh = self.sample_max_entropy_thresh_var.get()
            self.sampler_config.sample_varentropy_thresh = self.sample_varentropy_thresh_var.get()
            self.sampler_config.cot_min_entropy_thresh = self.cot_min_entropy_thresh_var.get()
            self.sampler_config.cot_max_entropy_thresh = self.cot_max_entropy_thresh_var.get()
            self.sampler_config.cot_varentropy_thresh = self.cot_varentropy_thresh_var.get()
            self.sampler_config.resample_min_entropy_thresh = self.resample_min_entropy_thresh_var.get()
            self.sampler_config.resample_max_entropy_thresh = self.resample_max_entropy_thresh_var.get()
            self.sampler_config.resample_varentropy_thresh = self.resample_varentropy_thresh_var.get()
            self.sampler_config.adaptive_entropy_thresh = self.adaptive_entropy_thresh_var.get()
            self.sampler_config.adaptive_varentropy_thresh = self.adaptive_varentropy_thresh_var.get()

                # Get center point and radius
            center_entropy = self.adaptive_entropy_center_var.get()
            center_varentropy = self.adaptive_varentropy_center_var.get()
            radius = self.adaptive_radius_var.get()
            separation = self.quadrant_separation_var.get()
            softness = self.boundary_softness_var.get()

            # Calculate strategy thresholds relative to center
            self.sampler_config.argmax_entropy_thresh = max(0.1, 
                center_entropy - (separation + radius))
            
            self.sampler_config.sample_min_entropy_thresh = max(0.1,
                center_entropy - separation)
            self.sampler_config.sample_max_entropy_thresh = center_entropy
            self.sampler_config.sample_varentropy_thresh = center_varentropy + self.sample_variance_threshold_var.get()
            
            self.sampler_config.cot_min_entropy_thresh = center_entropy + radius
            self.sampler_config.cot_max_entropy_thresh = center_entropy + radius + self.cot_entropy_range_var.get()
            self.sampler_config.cot_varentropy_thresh = center_varentropy - separation
            
            self.sampler_config.resample_min_entropy_thresh = center_entropy + radius
            self.sampler_config.resample_varentropy_thresh = center_varentropy + self.resample_min_variance_var.get()
            
            # Adaptive thresholds are directly based on center point
            self.sampler_config.adaptive_entropy_thresh = center_entropy
            self.sampler_config.adaptive_varentropy_thresh = center_varentropy
           
            # Save configuration
            self.save_config()

    def save_config(self):
        """Save current configuration to file"""
        config = {
            "basic_sampling": {
                "temp": self.temp_var.get(),
                "top_p": self.top_p_var.get(),
                "top_k": self.top_k_var.get(),
                "min_p": self.min_p_var.get(),
                "repetition_penalty": self.repetition_penalty_var.get()
            },
            "entropy_thresholds": {
                "low_ent_thresh": self.low_ent_thresh_var.get(),
                "med_ent_thresh": self.med_ent_thresh_var.get(),
                "high_ent_thresh": self.high_ent_thresh_var.get(),
                "high_vent_thresh": self.high_vent_thresh_var.get(),
                "varentropy_threshold": self.varentropy_threshold_var.get()
            },
            "adaptive_sampling": {
                "ada_temp_logits": self.ada_temp_logits_var.get(),
                "ada_temp_attn": self.ada_temp_attn_var.get(),
                "ada_temp_agree": self.ada_temp_agree_var.get(),
                "n_adaptive_samples": self.n_adaptive_samples_var.get()
            },
            "rope_parameters": {
                "rope_theta": self.rope_theta_var.get(),
                "rope_scaling": self.rope_scaling_var.get(),
                "rope_scale_base": self.rope_scale_base_var.get(),
                "rope_scale_factor": self.rope_scale_factor_var.get()
            },
            "attention_coefficients": {
                "helv_attn_ent_offset": self.helv_attn_ent_offset_var.get(),
                "helv_attn_ent_coef": self.helv_attn_ent_coef_var.get(),
                "lehv_interaction_strength_offset": self.lehv_interaction_strength_offset_var.get(),
                "lehv_interaction_strength_coef": self.lehv_interaction_strength_coef_var.get(),
                "hehv_attn_ent_coef": self.hehv_attn_ent_coef_var.get(),
                "hehv_attn_vent_offset": self.hehv_attn_vent_offset_var.get(),
                "hehv_attn_vent_coef": self.hehv_attn_vent_coef_var.get()
            },
            "memory_context": {
                "max_ngram_size": self.max_ngram_size_var.get(),
                "max_ngram_repeat": self.max_ngram_repeat_var.get(),
                "window_size": self.window_size_var.get(),
                "long_window_size": self.long_window_size_var.get(),
                "decay_factor": self.decay_factor_var.get(),
                "long_decay_factor": self.long_decay_factor_var.get()
            }
        }
        
        try:
            with open("entropix_config.json", "w") as f:
                json.dump(config, f, indent=4)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")

    def load_config(self):
        """Load configuration from file"""
        try:
            with open("entropix_config.json", "r") as f:
                config = json.load(f)
                
            # Update all GUI variables from loaded config
            for section, params in config.items():
                for param, value in params.items():
                    var = getattr(self, f"{param}_var", None)
                    if var:
                        var.set(value)
                        
            logger.info("Configuration loaded successfully")
            self.update_config()
        except FileNotFoundError:
            logger.info("No saved configuration found, using defaults")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def create_basic_controls(self, parent):
        """Create basic sampling controls"""
        group = ttk.LabelFrame(parent, text="Basic Sampling Parameters", padding=5)
        group.pack(fill="x", padx=5, pady=5)
        
        self.create_slider_with_tooltip(
            group, "Temperature", self.temp_var, 0.1, 2.0,
            PARAMETER_TOOLTIPS["basic_sampling"]["temp"]
        )
        self.create_slider_with_tooltip(
            group, "Top P", self.top_p_var, 0.0, 1.0,
            PARAMETER_TOOLTIPS["basic_sampling"]["top_p"]
        )
        self.create_slider_with_tooltip(
            group, "Top K", self.top_k_var, 1, 100,
            PARAMETER_TOOLTIPS["basic_sampling"]["top_k"]
        )
        self.create_slider_with_tooltip(
            group, "Min P", self.min_p_var, 0.0, 0.5,
            PARAMETER_TOOLTIPS["basic_sampling"]["min_p"]
        )
        self.create_slider_with_tooltip(
            group, "Repetition Penalty", self.repetition_penalty_var, 1.0, 2.0,
            PARAMETER_TOOLTIPS["basic_sampling"]["repetition_penalty"]
        )

        self.create_slider_with_tooltip(
            group, "Strategy Change Batch Size", self.strategy_change_batch_size_var, 1, 10,
            "Number of tokens to generate before allowing strategy changes"
        )

    def create_slider(self, parent, label: str, variable: tk.Variable, min_val: float, max_val: float, integer: bool = False):
        """Create a labeled slider with entry box"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(frame, text=label).pack(side="left")
        
        slider = ttk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            variable=variable,
            orient="horizontal"
        )
        slider.pack(side="left", fill="x", expand=True, padx=5)
        
        entry = ttk.Entry(frame, width=8, textvariable=variable)
        entry.pack(side="left")

    def create_slider_with_tooltip(self, parent, label: str, variable: tk.Variable, 
                                min_val: float, max_val: float, tooltip_text: str,
                                integer: bool = False):
        """Create a labeled slider with entry box and tooltip"""
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=5, pady=2)
        
        # Configure frame grid weights
        frame.grid_columnconfigure(1, weight=1)  # Make the slider column expand
        
        label_widget = ttk.Label(frame, text=label)
        label_widget.grid(row=0, column=0, padx=(0, 10))  # Add some padding between label and slider
        
        # Create tooltip
        ToolTip(label_widget, text=tooltip_text)
        
        slider = ttk.Scale(
            frame,
            from_=min_val,
            to=max_val,
            variable=variable,
            orient="horizontal"
        )
        slider.grid(row=0, column=1, sticky="ew", padx=5)  # Make slider expand horizontally
        
        entry = ttk.Entry(frame, width=8, textvariable=variable)
        entry.grid(row=0, column=2, padx=(5, 0))
        
        return frame

    def create_entropy_controls(self, parent):
        """Create entropy threshold controls based on adaptive center point"""
        # Create main container frame
        main_container = ttk.Frame(parent)
        main_container.pack(fill="both", expand=True, padx=5, pady=5)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Add quadrant explanation label
        explanation = ttk.Label(main_container, text="""
        Strategy Selection:
        Adaptive strategy forms the center point, with other strategies
        positioned relative to it in the entropy-varentropy space.
        
        Quadrant Positions (relative to center):
        - Top-Left: INSERT_COT (Higher entropy, Lower variance)
        - Top-Right: RESAMPLE (Higher entropy, Higher variance)
        - Bottom-Left: ARGMAX (Lower entropy, Lower variance)
        - Bottom-Right: SAMPLE (Lower entropy, Higher variance)
        """, justify="left")
        explanation.pack(fill="x", pady=10)

        # Adaptive Center Point Configuration
        adaptive_group = ttk.LabelFrame(main_container, text="Adaptive Strategy (Center Point)", padding=5)
        adaptive_group.pack(fill="x", pady=5)
        
        self.create_slider_with_tooltip(
            adaptive_group, "Center Entropy", self.adaptive_entropy_center_var, 0.0, 4.0,
            "Base entropy value for the center point of the adaptive strategy"
        )
        self.create_slider_with_tooltip(
            adaptive_group, "Center Varentropy", self.adaptive_varentropy_center_var, 0.0, 4.0,
            "Base variance entropy value for the center point of the adaptive strategy"
        )
        self.create_slider_with_tooltip(
            adaptive_group, "Radius", self.adaptive_radius_var, 0.1, 2.0,
            "Radius of the adaptive strategy region"
        )

        # Strategy Offset Configuration
        offset_group = ttk.LabelFrame(main_container, text="Strategy Offsets", padding=5)
        offset_group.pack(fill="x", pady=5)
        
        self.create_slider_with_tooltip(
            offset_group, "Quadrant Separation", self.quadrant_separation_var, 0.5, 2.0,
            "Distance between quadrants relative to center point"
        )
        self.create_slider_with_tooltip(
            offset_group, "Strategy Boundary Softness", self.boundary_softness_var, 0.1, 1.0,
            "Softness of boundaries between strategy regions"
        )

        # Strategy-specific adjustments
        strategies_group = ttk.LabelFrame(main_container, text="Strategy-Specific Adjustments", padding=5)
        strategies_group.pack(fill="x", pady=5)
        
        self.create_slider_with_tooltip(
            strategies_group, "ARGMAX Range", self.argmax_range_var, 0.1, 1.0,
            "Range of influence for ARGMAX strategy"
        )
        self.create_slider_with_tooltip(
            strategies_group, "SAMPLE Variance Threshold", self.sample_variance_threshold_var, 0.1, 2.0,
            "Minimum variance required for SAMPLE strategy"
        )
        self.create_slider_with_tooltip(
            strategies_group, "COT Entropy Range", self.cot_entropy_range_var, 0.1, 2.0,
            "Entropy range for INSERT_COT strategy"
        )
        self.create_slider_with_tooltip(
            strategies_group, "RESAMPLE Minimum Variance", self.resample_min_variance_var, 0.5, 2.0,
            "Minimum variance required for RESAMPLE strategy"
        )
            
    def create_adaptive_controls(self, parent):
        adaptive_group = ttk.LabelFrame(parent, text="Adaptive Parameters", padding=5)
        adaptive_group.pack(fill="x", padx=5, pady=5)
        
        # Add new scoring coefficient controls
        scoring_group = ttk.LabelFrame(adaptive_group, text="Scoring Coefficients", padding=5)
        scoring_group.pack(fill="x", pady=5)
        
        self.create_slider_with_tooltip(
            scoring_group, "Score Logits Entropy", self.ada_score_logits_ent_var, 0.0, 1.0,
            "Weight for logits entropy in sample scoring"
        )
        self.create_slider_with_tooltip(
            scoring_group, "Score Attn Entropy", self.ada_score_attn_ent_var, 0.0, 1.0,
            "Weight for attention entropy in sample scoring"
        )
        self.create_slider_with_tooltip(
            scoring_group, "Score Logits Varentropy", self.ada_score_logits_vent_var, 0.0, 1.0,
            "Weight for logits variance entropy in sample scoring"
        )
        self.create_slider_with_tooltip(
            scoring_group, "Score Attn Varentropy", self.ada_score_attn_vent_var, 0.0, 1.0,
            "Weight for attention variance entropy in sample scoring"
        )
        self.create_slider_with_tooltip(
            scoring_group, "Score Agreement", self.ada_score_agree_var, 0.0, 1.0,
            "Weight for head agreement in sample scoring"
        )
        self.create_slider_with_tooltip(
            scoring_group, "Score Interaction", self.ada_score_int_var, 0.0, 1.0,
            "Weight for head interaction in sample scoring"
        )

    def create_attention_controls(self, parent):
        """Create attention coefficient controls"""
        helv_group = ttk.LabelFrame(parent, text="High Entropy Low Variance", padding=5)
        helv_group.pack(fill="x", padx=5, pady=5)
        
        self.create_slider_with_tooltip(
            helv_group, "Entropy Offset", self.helv_attn_ent_offset_var, 0.0, 3.0,
            PARAMETER_TOOLTIPS["attention_coefficients"]["helv_attn_ent_offset"]
        )
        self.create_slider_with_tooltip(
            helv_group, "Entropy Coefficient", self.helv_attn_ent_coef_var, 0.0, 1.0,
            PARAMETER_TOOLTIPS["attention_coefficients"]["helv_attn_ent_coef"]
        )

        lehv_group = ttk.LabelFrame(parent, text="Low Entropy High Variance", padding=5)
        lehv_group.pack(fill="x", padx=5, pady=5)
        
        self.create_slider_with_tooltip(
            lehv_group, "Interaction Offset", self.lehv_interaction_strength_offset_var, 0.0, 3.0,
            PARAMETER_TOOLTIPS["attention_coefficients"]["lehv_interaction_strength_offset"]
        )
        self.create_slider_with_tooltip(
            lehv_group, "Interaction Coefficient", self.lehv_interaction_strength_coef_var, 0.0, 1.0,
            PARAMETER_TOOLTIPS["attention_coefficients"]["lehv_interaction_strength_coef"]
        )

        hehv_group = ttk.LabelFrame(parent, text="High Entropy High Variance", padding=5)
        hehv_group.pack(fill="x", padx=5, pady=5)
        
        self.create_slider_with_tooltip(
            hehv_group, "Entropy Coefficient", self.hehv_attn_ent_coef_var, 0.0, 1.0,
            PARAMETER_TOOLTIPS["attention_coefficients"]["hehv_attn_ent_coef"]
        )
        self.create_slider_with_tooltip(
            hehv_group, "Variance Offset", self.hehv_attn_vent_offset_var, 0.0, 5.0,
            PARAMETER_TOOLTIPS["attention_coefficients"]["hehv_attn_vent_offset"]
        )
        self.create_slider_with_tooltip(
            hehv_group, "Variance Coefficient", self.hehv_attn_vent_coef_var, 0.0, 1.0,
            PARAMETER_TOOLTIPS["attention_coefficients"]["hehv_attn_vent_coef"]
        )

    def create_rope_controls(self, parent):
        """Create RoPE parameter controls"""
        rope_group = ttk.LabelFrame(parent, text="RoPE Parameters", padding=5)
        rope_group.pack(fill="x", padx=5, pady=5)
        
        self.create_slider_with_tooltip(
            rope_group, "Theta", self.rope_theta_var, 1000.0, 100000.0,
            PARAMETER_TOOLTIPS["rope_parameters"]["rope_theta"]
        )
        self.create_slider_with_tooltip(
            rope_group, "Scaling", self.rope_scaling_var, 0.1, 2.0,
            PARAMETER_TOOLTIPS["rope_parameters"]["rope_scaling"]
        )
        self.create_slider_with_tooltip(
            rope_group, "Scale Base", self.rope_scale_base_var, 1.0, 16.0,
            PARAMETER_TOOLTIPS["rope_parameters"]["rope_scale_base"]
        )
        self.create_slider_with_tooltip(
            rope_group, "Scale Factor", self.rope_scale_factor_var, 0.0, 1.0,
            PARAMETER_TOOLTIPS["rope_parameters"]["rope_scale_factor"]
        )

    def create_memory_controls(self, parent):
        """Create memory and context controls"""
        ngram_group = ttk.LabelFrame(parent, text="N-gram Controls", padding=5)
        ngram_group.pack(fill="x", padx=5, pady=5)
        
        self.create_slider_with_tooltip(
            ngram_group, "Max N-gram Size", self.max_ngram_size_var, 1, 10,
            PARAMETER_TOOLTIPS["memory_context"]["max_ngram_size"]
        )
        self.create_slider_with_tooltip(
            ngram_group, "Max N-gram Repeats", self.max_ngram_repeat_var, 1, 10,
            PARAMETER_TOOLTIPS["memory_context"]["max_ngram_repeat"]
        )

        window_group = ttk.LabelFrame(parent, text="Window Parameters", padding=5)
        window_group.pack(fill="x", padx=5, pady=5)
        
        self.create_slider_with_tooltip(
            window_group, "Window Size", self.window_size_var, 10, 200,
            PARAMETER_TOOLTIPS["memory_context"]["window_size"]
        )
        self.create_slider_with_tooltip(
            window_group, "Long Window Size", self.long_window_size_var, 100, 1000,
            PARAMETER_TOOLTIPS["memory_context"]["long_window_size"]
        )
        self.create_slider_with_tooltip(
            window_group, "Decay Factor", self.decay_factor_var, 0.1, 1.0,
            PARAMETER_TOOLTIPS["memory_context"]["decay_factor"]
        )
        self.create_slider_with_tooltip(
            window_group, "Long Decay Factor", self.long_decay_factor_var, 0.1, 1.0,
            PARAMETER_TOOLTIPS["memory_context"]["long_decay_factor"]
        )

    def start_generation(self):
        if self.generation_thread and self.generation_thread.is_alive():
            return
            
        if not self.model or not self.tokenizer:
            self.output_text.insert("end", "Please load a model first.\n")
            return
            
        self.stop_generation = False
        self.strategy_counter.clear()
        prompt = self.prompt_input.get("1.0", "end-1c")
        
        if not prompt.strip():
            self.output_text.insert("end", "Please enter a prompt.\n")
            return
            
        self.update_config()
        self.output_text.delete("1.0", "end")
        self.generation_thread = threading.Thread(target=self.generate_text, args=(prompt,))
        self.generation_thread.start()
        self.root.after(100, self.check_response_queue)

    def stop_generation_request(self):
        self.stop_generation = True
        self.output_text.insert("end", "\n[Generation stopped by user]\n")

    def clear_output(self):
        self.output_text.delete("1.0", "end")
        self.strategy_counter.clear()
        self.update_stats({})  # Clear statistics

    def check_response_queue(self):
        """Process responses from the generation thread"""
        try:
            while not self.response_queue.empty():
                response = self.response_queue.get_nowait()
                if isinstance(response, tuple):
                    response_type, content = response
                    
                    if response_type == "token":
                        self.output_text.insert("end", content)
                        self.output_text.see("end")
                    elif response_type == "stats":
                        if isinstance(content, dict):  # Ensure content is a dictionary
                            self.update_stats(content)
                    elif response_type == "strategy":
                        self.strategy_counter[content] += 1
                        self.update_strategy_display()
                    elif response_type == "error":
                        self.output_text.insert("end", f"\nError: {content}\n")
                        
            if self.generation_thread and self.generation_thread.is_alive():
                self.root.after(100, self.check_response_queue)
        except Exception as e:
            logger.error(f"Error in check_response_queue: {str(e)}")
            self.root.after(100, self.check_response_queue)

    def update_stats(self, stats: Dict[str, Union[float, str]]):
        with self.stats_lock:
            # Update entropy metrics with correct keys
            if 'logits_entropy' in stats:
                self.stat_labels["Entropy"].set(f"Entropy: {stats['logits_entropy']:.4f}")
            if 'logits_varentropy' in stats:
                self.stat_labels["Varentropy"].set(f"Varentropy: {stats['logits_varentropy']:.4f}")
            if 'attn_entropy' in stats:
                self.stat_labels["Attn Entropy"].set(f"Attn Entropy: {stats['attn_entropy']:.4f}")
            
            # Update rolling statistics
            if 'rolling_entropy' in stats:
                self.stat_labels["Rolling Entropy"].set(f"Rolling Entropy: {stats['rolling_entropy']:.4f}")
            if 'rolling_varentropy' in stats:
                self.stat_labels["Rolling Varentropy"].set(f"Rolling Varentropy: {stats['rolling_varentropy']:.4f}")

            # Add new adaptive sampling metrics
            if 'sample_score' in stats:
                self.stat_labels["Sample Score"].set(f"Sample Score: {stats['sample_score']:.4f}")
            if 'agreement' in stats:
                self.stat_labels["Agreement"].set(f"Agreement: {stats['agreement']:.4f}")
            if 'interaction_strength' in stats:
                self.stat_labels["Interaction"].set(f"Interaction: {stats['interaction_strength']:.4f}")
            
            # Update current strategy
            if 'current_strategy' in stats:
                self.stat_labels["Current Strategy"].set(f"Current: {stats['current_strategy']}")
                
            # Log the stats for debugging
            logger.debug(f"Updating stats: {stats}")

    def update_strategy_display(self):
        """Update the strategy usage display with percentages"""
        total_tokens = sum(self.strategy_counter.values())
        if total_tokens > 0:
            # Enable text widget for updating
            self.strategy_stats.configure(state="normal")
            self.strategy_stats.delete("1.0", "end")
            
            for strategy, count in self.strategy_counter.most_common():
                percentage = (count / total_tokens) * 100
                self.strategy_stats.insert("end", f"{strategy}:\n{count} ({percentage:.1f}%)\n\n")
            
            # Disable text widget after updating
            self.strategy_stats.configure(state="disabled")
        
        # Update CoT statistics
        cot_count = self.strategy_counter.get(SamplerState.INSERT_COT, 0)
        cot_ratio = (cot_count / total_tokens * 100) if total_tokens > 0 else 0
        self.cot_count_var.set(f"Insertions: {cot_count}")
        self.cot_ratio_var.set(f"Ratio: {cot_ratio:.1f}%")

    def generate_text(self, prompt: str):
        """Generate text with proper error handling and output saving"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_tokens = []  # Store tokens
        current_stats = {}  # Store current statistics
        
        try:
            # Encode input with proper truncation
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.config.max_position_embeddings - 1000
            ).to(device)
            
            attention_mask = torch.ones_like(input_ids)
            
            # Add proper newlines in the prompt display
            formatted_prompt = prompt.replace('\\n', '\n')
            self.response_queue.put(("token", f"Prompt: {formatted_prompt}\n\nGenerated response:\n"))
            
            sampler = EntropixSampler(self.sampler_config)
            
            with torch.inference_mode():
                for _ in range(1000):  # Max tokens
                    if self.stop_generation:
                        break
                    
                    outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        output_attentions=True
                    )
                    
                    logits = outputs.logits
                    attention = outputs.attentions[-1]  # Last layer's attention
                    
                    try:
                        sampled_token, state = sampler.sample(logits, attention)
                        current_stats = sampler.calculate_metrics(logits, attention)
                        current_stats['current_strategy'] = state.name
                        self.response_queue.put(("stats", current_stats))
                        self.response_queue.put(("strategy", state.name))
                        
                    except Exception as sampling_error:
                        self.response_queue.put(("error", f"Sampling error: {str(sampling_error)}"))
                        break
                    
                    if state == SamplerState.EOT or sampled_token[0] == self.tokenizer.eos_token_id:
                        self.response_queue.put(("token", "\n[End of Text]\n"))
                        generated_tokens.append("[End of Text]")
                        break
                    
                    if state == SamplerState.INSERT_COT and sampled_token[0] == self.sampler_config.cot_token:
                        next_token_text = "[...]"
                        self.response_queue.put(("token", f"\n{next_token_text}\n"))
                        generated_tokens.append(next_token_text)
                    else:
                        next_token_text = self.tokenizer.decode(sampled_token[0])
                        next_token_text = next_token_text.replace('\\n', '\n')
                        self.response_queue.put(("token", next_token_text))
                        generated_tokens.append(next_token_text)
                    
                    input_ids = torch.cat([input_ids, sampled_token], dim=-1)
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((1, 1), dtype=torch.long, device=device)
                    ], dim=1)
                    
                    if input_ids.shape[1] >= self.model.config.max_position_embeddings:
                        self.response_queue.put(("token", "\n[Reached maximum sequence length]\n"))
                        generated_tokens.append("[Reached maximum sequence length]")
                        break

                # Save generation output after successful generation
                self.save_generation_output(prompt, generated_tokens, current_stats)
                
        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            self.response_queue.put(("error", error_msg))
            logger.error(error_msg)
            
            # Try to save partial output if available
            if generated_tokens:
                self.save_generation_output(prompt, generated_tokens, current_stats)


    def get_current_config(self):
        """Get current configuration as a dictionary with all parameters"""
        return {
            "model_info": {
                "name": self.current_model_name,
                "hash": self.current_model_hash,
                "timestamp": datetime.now().isoformat()
            },
            "parameters": {
                "basic_sampling": {
                    "temp": self.temp_var.get(),
                    "top_p": self.top_p_var.get(),
                    "top_k": self.top_k_var.get(),
                    "min_p": self.min_p_var.get(),
                    "repetition_penalty": self.repetition_penalty_var.get(),
                    "strategy_change_batch_size": self.strategy_change_batch_size_var.get()
                },
                "entropy_thresholds": {
                    "low_ent_thresh": self.low_ent_thresh_var.get(),
                    "med_ent_thresh": self.med_ent_thresh_var.get(),
                    "high_ent_thresh": self.high_ent_thresh_var.get(),
                    "high_vent_thresh": self.high_vent_thresh_var.get(),
                    "varentropy_threshold": self.varentropy_threshold_var.get()
                },
                "adaptive_sampling": {
                    "ada_temp_logits": self.ada_temp_logits_var.get(),
                    "ada_temp_attn": self.ada_temp_attn_var.get(),
                    "ada_temp_agree": self.ada_temp_agree_var.get(),
                    "n_adaptive_samples": self.n_adaptive_samples_var.get()
                },
                "rope_parameters": {
                    "rope_theta": self.rope_theta_var.get(),
                    "rope_scaling": self.rope_scaling_var.get(),
                    "rope_scale_base": self.rope_scale_base_var.get(),
                    "rope_scale_factor": self.rope_scale_factor_var.get()
                },
                "attention_coefficients": {
                    "helv_attn_ent_offset": self.helv_attn_ent_offset_var.get(),
                    "helv_attn_ent_coef": self.helv_attn_ent_coef_var.get(),
                    "lehv_interaction_strength_offset": self.lehv_interaction_strength_offset_var.get(),
                    "lehv_interaction_strength_coef": self.lehv_interaction_strength_coef_var.get(),
                    "hehv_attn_ent_coef": self.hehv_attn_ent_coef_var.get(),
                    "hehv_attn_vent_offset": self.hehv_attn_vent_offset_var.get(),
                    "hehv_attn_vent_coef": self.hehv_attn_vent_coef_var.get()
                },
                "memory_context": {
                    "max_ngram_size": self.max_ngram_size_var.get(),
                    "max_ngram_repeat": self.max_ngram_repeat_var.get(),
                    "window_size": self.window_size_var.get(),
                    "long_window_size": self.long_window_size_var.get(),
                    "decay_factor": self.decay_factor_var.get(),
                    "long_decay_factor": self.long_decay_factor_var.get()
                },
                "strategy_thresholds": {
                    "argmax_entropy_thresh": self.argmax_entropy_thresh_var.get(),
                    "sample_min_entropy_thresh": self.sample_min_entropy_thresh_var.get(),
                    "sample_max_entropy_thresh": self.sample_max_entropy_thresh_var.get(),
                    "sample_varentropy_thresh": self.sample_varentropy_thresh_var.get(),
                    "cot_min_entropy_thresh": self.cot_min_entropy_thresh_var.get(),
                    "cot_max_entropy_thresh": self.cot_max_entropy_thresh_var.get(),
                    "cot_varentropy_thresh": self.cot_varentropy_thresh_var.get(),
                    "resample_min_entropy_thresh": self.resample_min_entropy_thresh_var.get(),
                    "resample_max_entropy_thresh": self.resample_max_entropy_thresh_var.get(),
                    "resample_varentropy_thresh": self.resample_varentropy_thresh_var.get(),
                    "adaptive_entropy_thresh": self.adaptive_entropy_thresh_var.get(),
                    "adaptive_varentropy_thresh": self.adaptive_varentropy_thresh_var.get()
                }
            },
            "generation_settings": {
                "max_tokens": 1000,
                "device": str(device),
                "model_max_length": self.model.config.max_position_embeddings if self.model else None,
                "truncation_length": self.model.config.max_position_embeddings - 1000 if self.model else None
            }
        }

    def save_generation_output(self, prompt: str, generated_tokens: List[str], final_stats: dict) -> None:
        """Save the generation output automatically with all context"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            # Ensure output directory exists
            self.output_dir.mkdir(exist_ok=True)
            
            # Create the output content
            generated_text = ''.join(str(token) for token in generated_tokens)
            
            # Prepare the complete output content
            output_content = [
                "=== Generation Information ===",
                f"Timestamp: {datetime.now().isoformat()}",
                f"Model: {self.current_model_name}",
                f"Model Hash: {self.current_model_hash}",
                "",
                "=== Prompt ===",
                prompt,
                "",
                "=== Generated Output ===",
                generated_text,
                "",
                "=== Generation Statistics ===",
                f"Total Tokens: {sum(self.strategy_counter.values())}",
                "",
                "Strategy Usage:",
            ]
            
            # Add strategy statistics
            total_tokens = sum(self.strategy_counter.values())
            if total_tokens > 0:
                for strategy, count in self.strategy_counter.most_common():
                    percentage = (count / total_tokens) * 100
                    output_content.append(f"  {strategy}: {count} ({percentage:.1f}%)")
            
            # Add final statistics
            output_content.extend([
                "",
                "=== Final Statistics ===",
                json.dumps(final_stats, indent=2)
            ])
            
            # Save to file
            output_filename = f"{self.current_model_hash}_{timestamp}.txt"
            output_path = self.output_dir / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(output_content))
            
            logger.info(f"Generation output saved to {output_path}")
                    
        except Exception as e:
            logger.error(f"Error saving output: {str(e)}")
            # Try to save partial output
            try:
                if generated_tokens:
                    error_output_path = self.output_dir / f"error_{self.current_model_hash}_{timestamp}.txt"
                    with open(error_output_path, "w", encoding="utf-8") as f:
                        f.write(f"Error during generation: {str(e)}\n\nPartial output:\n{''.join(str(token) for token in generated_tokens)}")
                    logger.info(f"Partial output saved to {error_output_path}")
            except Exception as save_error:
                logger.error(f"Error saving partial output: {str(save_error)}")

def main():
    """Main function to run the Entropix GUI"""
    app = EntropixTGUI()
    app.root.mainloop()

if __name__ == "__main__":
    main()