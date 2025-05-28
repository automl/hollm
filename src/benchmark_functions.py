# -*- coding: utf-8 -*-
"""
Benchmark functions for global optimization.
"""

import numpy as np
import torch
from botorch.test_functions import (
    Ackley,
    Levy,
    Rosenbrock,
    Rastrigin,
    Hartmann
)


def get_benchmark_function(name, dim=20, negate=True, device=None, dtype=None):
    """
    Get a benchmark function by name.

    Args:
        name: Name of the benchmark function
        dim: Dimension of the function
        negate: Whether to negate the function (for maximization)
        device: PyTorch device
        dtype: PyTorch dtype

    Returns:
        The benchmark function
    """
    if name.lower() == "ackley":
        fun = Ackley(dim=dim, negate=negate).to(dtype=dtype, device=device)
        # Modify the default bounds to match our experiment
        fun.bounds[0, :].fill_(-32.768)
        fun.bounds[1, :].fill_(32.768)
        return fun

    elif name.lower() == "levy":
        fun = Levy(dim=dim, negate=negate).to(dtype=dtype, device=device)
        # Default bounds: [0, 10]^d
        return fun

    elif name.lower() == "rosenbrock":
        fun = Rosenbrock(dim=dim, negate=negate).to(dtype=dtype, device=device)
        # Default bounds: [0, 2]^d (for standard Rosenbrock)
        return fun

    elif name.lower() == "rastrigin":
        fun = Rastrigin(dim=dim, negate=negate).to(dtype=dtype, device=device)
        # Default bounds: [-5.12, 5.12]^d
        return fun

    elif name.lower() == "hartmann":
        # Hartmann is only defined for dims 3 and 6
        if dim not in [3, 6]:
            raise ValueError("Hartmann function is only defined for dimensions 3 and 6")
        fun = Hartmann(dim=dim, negate=negate).to(dtype=dtype, device=device)
        # Default bounds: [0, 1]^d
        return fun

    elif name.lower() == "nb201-c10":
        # Import the NB201Benchmark class
        from src.test_functions.nb201 import NB201Benchmark

        # Create the benchmark
        fun = NB201Benchmark(
            path="src/test_functions/nb201.pkl",
            dataset="cifar10",
        )

        return fun
    elif name.lower() == "nb201-c100":
        # Import the NB201Benchmark class
        from src.test_functions.nb201 import NB201Benchmark

        # Create the benchmark
        fun = NB201Benchmark(
            path="src/test_functions/nb201.pkl",
            dataset="cifar100", #imagenet16, cifar10
        )

        return fun
    elif name.lower() == "nb201-imgnet":
        # Import the NB201Benchmark class
        from src.test_functions.nb201 import NB201Benchmark

        # Create the benchmark
        fun = NB201Benchmark(
            path="src/test_functions/nb201.pkl",
            dataset="imagenet16", #imagenet16, cifar10
        )

        return fun

    else:
        raise ValueError(f"Unknown benchmark function: {name}")


def eval_objective(x, fun):
    """
    Helper function to unnormalize and evaluate a point on the objective function.

    Args:
        x: Input point (normalized to [0, 1])
        fun: Benchmark function

    Returns:
        Function value at the unnormalized point
    """
    if hasattr(fun, 'objective_function'):
        # Handle NB201 benchmark differently
        config = {}
        operation_types = ["none", "skip_connect", "avg_pool_3x3", "nor_conv_1x1", "nor_conv_3x3"]

        # Convert continuous values to categorical values
        x_numpy = x.cpu().numpy()
        for i in range(len(x_numpy)):
            # Map the continuous value to a discrete choice
            idx = min(int(x_numpy[i] * len(operation_types)), len(operation_types) - 1)
            config[f"op_{i}_to_{i+1 if i < 2 else 3}"] = operation_types[idx]

        # Call the objective function
        error, latency = fun.objective_function(config)
        # Return negative error to make it a maximization problem (consistent with other functions)
        return -error
    else:
        from botorch.utils.transforms import unnormalize
        return fun(unnormalize(x, fun.bounds))

