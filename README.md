# EntropixLab

![GUI 1](https://github.com/EdwardDali/EntropixLab/blob/main_torch/docs/GUI1.PNG)
![GUI 2](https://github.com/EdwardDali/EntropixLab/blob/main_torch/docs/GUI2.PNG)

# Entropix Text Generator

A sophisticated text generation tool with dynamic sampling strategies based on entropy and attention metrics. This project implements an advanced text generation interface that adapts its sampling approach based on real-time analysis of model outputs.

## Features

- **Dynamic Sampling Strategies**
  - Adaptive sampling based on entropy and attention metrics
  - Multiple sampling strategies (ARGMAX, SAMPLE, INSERT_COT, RESAMPLE, ADAPTIVE)
  - Real-time strategy selection based on model uncertainty

- **Advanced Attention Analysis**
  - Real-time attention pattern monitoring
  - Head agreement and interaction strength analysis
  - Variance-based entropy calculations

- **Comprehensive GUI Interface**
  - Interactive parameter adjustment
  - Real-time generation statistics
  - Strategy usage visualization
  - Model selection and management

- **Extended RoPE Implementation**
  - Configurable rotary positional embeddings
  - Scaled RoPE for improved position handling
  - Customizable scaling parameters

- **Memory Management**
  - N-gram repetition control
  - Adaptive window sizes
  - Long-term and short-term context tracking

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- tkinter
- numpy

```bash
pip install torch transformers numpy
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/entropix-text-generator.git
cd entropix-text-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### GUI Interface

Run the GUI application:
```bash
python gui.py
```

The interface provides:
- Model selection and loading
- Parameter adjustment through interactive sliders
- Real-time generation statistics
- Output saving and configuration management

### Command Line Interface

Run the core generation script:
```bash
python main_t.py
```

## Configuration

The tool provides extensive configuration options:

### Basic Sampling Parameters
- Temperature
- Top-p (nucleus sampling)
- Top-k
- Minimum-p threshold
- Repetition penalty

### Entropy & Strategy Parameters
- Adaptive center point configuration
- Strategy boundary controls
- Quadrant separation settings

### Attention Coefficients
- High Entropy Low Variance (HELV) settings
- Low Entropy High Variance (LEHV) settings
- High Entropy High Variance (HEHV) settings

### RoPE Parameters
- Base theta value
- Scaling factors
- Position embedding controls

## Architecture

The project consists of two main components:

1. **Core Generation Engine (main_t.py)**
   - Implementation of sampling strategies
   - Attention and entropy calculations
   - Model interaction handling

2. **GUI Interface (gui.py)**
   - Interactive parameter control
   - Real-time visualization
   - Model and configuration management

## Strategy Selection

The tool employs a quadrant-based strategy selection system:

```
Strategy Positioning:
┌─────────────────┬─────────────────┐
│   INSERT_COT    │    RESAMPLE     │
│  High Entropy   │  High Entropy   │
│  Low Variance   │  High Variance  │
├─────────────────┼─────────────────┤
│    ARGMAX       │     SAMPLE      │
│  Low Entropy    │  Low Entropy    │
│  Low Variance   │  High Variance  │
└─────────────────┴─────────────────┘
        Center: ADAPTIVE
```


## Citation

If you use this tool in your research, please cite:

```bibtex
@software{entropix_text_generator,
  title = {Entropix Text Generator},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/entropix-text-generator}
}
```


