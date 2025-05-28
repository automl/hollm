#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate YAML configuration files for all test functions.
"""

import os
import yaml
import argparse

# Define base directory for configs
CONFIG_DIR = "configs"

# Define configurations for different test functions
TEST_FUNCTIONS = {
    "ackley": {
        "function": "ackley",
        "dim": 20,
        "M": 5,
        "k": 5,
        "lam": 0.0,
        "batch_size": 4,
        "n_init": 5,
        "max_evals": 100,
        "alpha_min": 0.01,
        "alpha_max": 1.0,
    },
    "levy": {
        "function": "levy",
        "dim": 10,
        "M": 5,
        "k": 5,
        "lam": 0.0,
        "batch_size": 4,
        "n_init": 5,
        "max_evals": 100,
        "alpha_min": 0.01,
        "alpha_max": 1.0,
    },
    "rosenbrock": {
        "function": "rosenbrock",
        "dim": 8,
        "M": 5,
        "k": 5,
        "lam": 0.0,
        "batch_size": 4,
        "n_init": 5,
        "max_evals": 100,
        "alpha_min": 0.01,
        "alpha_max": 1.0,
    },
    "rastrigin": {
        "function": "rastrigin",
        "dim": 10,
        "M": 5,
        "k": 5,
        "lam": 0.0,
        "batch_size": 4,
        "n_init": 30,
        "max_evals": 150,
        "alpha_min": 0.01,
        "alpha_max": 1.0,
    },
    "hartmann3": {
        "function": "hartmann",
        "dim": 3,
        "M": 5,
        "k": 5,
        "lam": 0.0,
        "batch_size": 4,
        "n_init": 5,
        "max_evals": 100,
        "alpha_min": 0.01,
        "alpha_max": 1.0,
    },
    "hartmann6": {
        "function": "hartmann",
        "dim": 6,
        "M": 5,
        "k": 5,
        "lam": 0.0,
        "batch_size": 4,
        "n_init": 5,
        "max_evals": 100,
        "alpha_min": 0.01,
        "alpha_max": 1.0,
    },
    "nb201-c10": {
        "function": "nb201-c10",
        "dim": 6,
        "M": 5,
        "k": 5,
        "lam": 0.0,
        "batch_size": 4,
        "n_init": 15,
        "max_evals": 100,
        "dataset": "cifar10",
        "device_metric": "1080ti_32_latency",
        "alpha_min": 0.01,
        "alpha_max": 1.0,
    },
    "nb201-c100": {
        "function": "nb201-c100",
        "dim": 6,
        "M": 5,
        "k": 5,
        "lam": 0.0,
        "batch_size": 4,
        "n_init": 15,
        "max_evals": 100,
        "dataset": "cifar100",
        "device_metric": "1080ti_32_latency",
        "alpha_min": 0.01,
        "alpha_max": 1.0,
    },
    "nb201-imgnet": {
        "function": "nb201-imgnet",
        "dim": 6,
        "M": 5,
        "k": 5,
        "lam": 0.0,
        "batch_size": 4,
        "n_init": 15,
        "max_evals": 100,
        "dataset": "imagenet16",
        "device_metric": "1080ti_32_latency",
        "alpha_min": 0.01,
        "alpha_max": 1.0,
    },
}


def generate_config(function_name, config, output_dir, api_key="gemini"):
    """
    Generate a YAML configuration file for a test function.
    
    Args:
        function_name: Name of the test function
        config: Configuration dictionary
        output_dir: Directory to save the config file
        api_key: API key for LLM-based optimizers
        optimizers: List of optimizers to include (defaults to DEFAULT_OPTIMIZERS)
    
    Returns:
        Path to the generated config file
    """
    # Create a copy of the config to avoid modifying the original
    config_copy = config.copy()
    
    # Add common parameters
    config_copy["api_key"] = api_key
    config_copy["optimizer"] = 'hollm'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    file_path = os.path.join(output_dir, f"{function_name}.yaml")
    
    # Write configuration to file
    with open(file_path, "w") as f:
        yaml.dump(config_copy, f, default_flow_style=False)
    
    return file_path


def main(args):
    """
    Generate configuration files for all test functions.
    
    Args:
        args: Command-line arguments
    """
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate configurations for all test functions
    for function_name, config in TEST_FUNCTIONS.items():
        # Skip functions that were excluded
        if args.exclude and function_name in args.exclude:
            print(f"Skipping excluded function: {function_name}")
            continue
        
        # Generate config file
        file_path = generate_config(
            function_name,
            config,
            args.output_dir,
            args.api_key,
        )
        
        print(f"Generated config file for {function_name}: {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YAML configuration files for test functions")
    parser.add_argument("--output-dir", type=str, default=CONFIG_DIR, help="Directory to save config files")
    parser.add_argument("--api-key", type=str, default="gemini", help="API key for LLM-based optimizers")
    parser.add_argument("--exclude", type=str, nargs="+", help="Functions to exclude")
    
    args = parser.parse_args()
    main(args)
