# -*- coding: utf-8 -*-
"""
Utility functions for Bayesian optimization.
"""

import torch
import json
import numpy as np


def get_initial_points(dim, n_pts, seed=0, dtype=torch.float64, random=False):
    """
    Generate initial points using a Sobol sequence.
    
    Args:
        dim: Dimension of the search space
        n_pts: Number of points to generate
        seed: Random seed
        
    Returns:
        X_init: Tensor of initial points
    """
    if random:
        X_init = torch.rand(n_pts, dim, dtype=dtype)
    else:
        sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts, dtype=dtype)
    return X_init

def cosine_annealing_schedule(iteration, total_iterations, alpha_max=1.0,
                              alpha_min=0.0):
    """
    Cosine annealing schedule that returns alpha values starting at 1 and ending at 0.

    Args:
        iteration: Current iteration (0-indexed)
        total_iterations: Total number of iterations

    Returns:
        alpha: Value between 0 and 1
    """
    alpha = alpha_min + (alpha_max - alpha_min) * 0.5 * (1 + np.cos(np.pi * iteration/total_iterations))
    return alpha

def save_results(results, filename):
    """
    Save optimization results to a JSON file.
    
    Args:
        results: Dictionary of results
        filename: Name of the file to save
    """
    # Convert tensors to lists
    for key, val in results.items():
        if isinstance(val, torch.Tensor):
            results[key] = val.tolist()
        elif isinstance(val, np.ndarray):
            results[key] = val.tolist()
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(results, f)


def load_results(filename):
    """
    Load optimization results from a JSON file.
    
    Args:
        filename: Name of the file to load
        
    Returns:
        Dictionary of results
    """
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results
