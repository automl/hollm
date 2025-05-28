# -*- coding: utf-8 -*-
"""
Run benchmark experiments.
"""

import os
import argparse
import torch
import time
import yaml

from src.benchmark_functions import get_benchmark_function, eval_objective
from src.utils import save_results
from src.models.llm_model import LLMOptimizer

from src.optimizers import run_turbo, run_random_search, run_sobol, run_gpbo, run_llm, run_hollm


def setup_objective(fun_name, dim, device, dtype):
    """
    Setup the objective function.

    Args:
        fun_name: Name of the benchmark function
        dim: Dimension of the function
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        Objective function that takes normalized inputs
    """
    fun = get_benchmark_function(fun_name, dim, True, device, dtype)
    return lambda x: eval_objective(x, fun)


def load_config(config_file):
    """
    Load configuration from a YAML file.

    Args:
        config_file: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Set device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.float64
    print(f"Using device: {device}")

    if args.api_key == 'gemini':
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup objective function
    objective = setup_objective(args.function, args.dim, device, dtype)

    # Run selected optimizers
    results = {}

    optimizer = args.optimizer
    print(f"\nRunning {optimizer}...")
    print(f"\nArguments: {args}")
    start_time = time.time()

    if optimizer == 'random':
        X, Y = run_random_search(objective, args.dim, args.max_evals, device, dtype, args.seed)
    elif optimizer == 'sobol':
        X, Y = run_sobol(objective, args.dim, args.max_evals, device, dtype, args.seed)
    elif optimizer == 'turbo':
        X, Y = run_turbo(objective, args.dim, args.batch_size, args.n_init, args.max_evals,
                         device, dtype, "ts", args.seed)
    elif optimizer == 'ei':
        X, Y = run_gpbo(objective, args.dim, args.batch_size, args.n_init,
                        'ei', args.max_evals, device, dtype, args.seed)
    elif optimizer == 'logei':
        X, Y = run_gpbo(objective, args.dim, args.batch_size, args.n_init,
                        'logei', args.max_evals, device, dtype, args.seed)
    elif optimizer == 'llm':
        if not args.api_key:
            print("API key is required for LLM-based optimization.")
        llm_optimizer = LLMOptimizer(api_key=api_key,
                                     model_family="models/gemini-1.5-flash")
        X, Y = run_llm(objective, args.dim, args.batch_size, args.n_init,
                       args.max_evals, llm_optimizer, device,
                       args.k,
                       args.M,
                       dtype, api_key, args.seed)
    elif optimizer == 'hollm':
        if not args.api_key:
            print("API key is required for LLM-based optimization.")
        llm_optimizer = LLMOptimizer(api_key=api_key,
                                     model_family="models/gemini-1.5-flash")
        print(f"alpha_min={args.alpha_min}, alpha_max={args.alpha_max}")
        X, Y = run_hollm(objective, args.dim, args.batch_size, args.n_init, args.max_evals,
                         llm_optimizer, device, dtype, args.seed,
                         lam=args.lam,
                         k=args.k,
                         M=args.M,
                         alpha_min=args.alpha_min,
                         alpha_max=args.alpha_max)
    else:
        print(f"Unknown optimizer: {optimizer}")

    elapsed_time = time.time() - start_time
    print(f"Finished {optimizer} in {elapsed_time:.2f} seconds")

    # Store results
    results[f'X_{optimizer}'] = X
    results[f'Y_{optimizer}'] = Y

    # Save results
    results_file = os.path.join(args.output_dir, f'{optimizer}_results.json')
    save_results(results, results_file)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Bayesian optimization benchmarks')
    parser.add_argument('--config', type=str, help='Path to configuration YAML file')
    parser.add_argument('--function', type=str, default='ackley', help='Benchmark function to optimize')
    parser.add_argument('--dim', type=int, default=20, help='Dimension of the function')
    parser.add_argument('--k', type=int, default=5, help='Number of sample per partition in HOLLM')
    parser.add_argument('--M', type=int, default=5, help='Number of selected partitions in HOLLM')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for batch optimization methods')
    parser.add_argument('--n_init', type=int, default=5, help='Number of initial points (default: 2*dim)')
    parser.add_argument('--max_evals', type=int, default=100, help='Maximum number of function evaluations')
    parser.add_argument('--optimizer', type=str, default='random', help='Optimizer to run')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--cpu', action='store_true', help='Force using CPU even if CUDA is available')
    parser.add_argument('--api_key', type=str, default='gemini', help='API key for LLM-based optimization')
    parser.add_argument('--alpha_min', type=float, default=0.01, help='Minimum exploration factor for cosine annealing')
    parser.add_argument('--alpha_max', type=float, default=1.0, help='Maximum exploration factor for cosine annealing')
    parser.add_argument('--lam', type=float, default=0.0, help='Adaptive leaf size factor')

    args = parser.parse_args()

    # Load configuration file if provided
    if args.config:
        config = load_config(args.config)
        for k, v in config.items():
            setattr(args, k, v)

    main(args)
