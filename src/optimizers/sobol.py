# -*- coding: utf-8 -*-
"""
Sobol sequence based optimization.
"""

import torch


def run_sobol(
    fun,
    dim,
    n_evals,
    device=None,
    dtype=None,
    seed=0,
):
    """
    Run optimization using a Sobol sequence.
    
    Args:
        fun: Objective function (with eval_objective interface)
        dim: Dimension of the search space
        n_evals: Number of function evaluations
        device: PyTorch device
        dtype: PyTorch dtype
        seed: Random seed
        
    Returns:
        X: Evaluated points
        Y: Function values
    """
    # Set random seed
    torch.manual_seed(seed)
    
    # Generate Sobol sequence
    sobol = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
    X = sobol.draw(n_evals).to(dtype=dtype, device=device)
    
    # Evaluate points
    Y = torch.tensor(
        [fun(x) for x in X], dtype=dtype, device=device
    ).unsqueeze(-1)
    
    # Print progress periodically
    for i in range(0, n_evals, 10):
        best_val = Y[:i+1].max().item() if i < n_evals else Y.max().item()
        print(f"{i+1}) Best value: {best_val:.2e}")
    
    return X, Y
