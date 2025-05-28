# Hierarchical Optimization with Large Language Models (HOLLM)

This repository contains a minimal implementation of Hierarchical Optimization with Large Language Models (HOLLM)

## Getting Started

### 1. Setup Environment

```bash
# Create the conda environment
conda create -n hollm python=3.11

# Activate the environment
conda activate hollm

# Install the dependencies
python -m pip install -r requirements.txt
```

### 2. LLM API Keys
In our experiments we used ```gemini-1.5-flash```. If using the same, set up the environment variables first:
```
echo "export GEMINI_API_KEY={api_key}" >> ~/.zshrc
source ~/.zshrc
```

### 3. Generate Configuration Files

```bash
# Generate YAML config files for all test functions
python generate_configs.py

```

### 4. Submit Jobs

```bash
# Submit jobs with specified settings in the config files
python run_benchmarks.py --config configs/levy.yaml

```

## Directory Structure

```
bayesian_optimization_ackley/
├── README.md
├── requirements.txt
├── run_benchmarks.py          # Main script (now uses YAML configs)
├── configs/                       # YAML configuration files
│   ├── ackley.yaml
│   ├── hartmann.yaml
│   └── ...
├── src/                           # Core implementation
│   ├── benchmark_functions.py
│   ├── acquisition_strategies.py
│   └── ...
├── slurm/                         # SLURM scripts
│   ├── run_optimizer.slurm
│   ├── run_optimizer_llm.slurm
│   ├── aggregate_results.slurm
│   └── ...
└── generate_configs.py            # Helper to generate YAML configs
```

## Optimizers Implemented

- TuRBO-1 (Trust Region Bayesian Optimization)
- Expected Improvement (EI)
- Log Expected Improvement (LogEI)
- Thompson Sampling (TS)
- Random Search
- Sobol Sequence
- LLM-based Global Optimization
- HOLLM
