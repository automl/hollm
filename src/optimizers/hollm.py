# -*- coding: utf-8 -*-
"""
HOLLM: LLM-based optimization with KD-tree partitioning.
"""

import numpy as np
import torch

from src.optimizers.kd_utils import compute_kd_partitions
from src.utils import cosine_annealing_schedule


def run_hollm(
    fun,
    dim,
    batch_size,
    n_init,
    max_evals,
    llm_optimizer,
    device=None,
    dtype=None,
    seed=0,
    lam=0.0,
    k=5,
    M=5,
    alpha_min=0.1,
    alpha_max=1.0,
):
    """
    Run LLM-based optimization with KD-tree partitioning and adaptive leaf size.

    Args:
        fun: Objective function (with eval_objective interface)
        dim: Dimension of the search space
        batch_size: Number of points to evaluate in each batch
        n_init: Number of initial points
        max_evals: Maximum number of function evaluations
        llm_optimizer: LLM-based optimizer instance
        device: PyTorch device
        dtype: PyTorch dtype
        seed: Random seed

    Returns:
        X: Evaluated points
        Y: Function values
    """
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate initial points
    from src.utils import get_initial_points
    X = get_initial_points(dim, n_init, seed, random=True).cpu().numpy()
    Y = np.array([fun(torch.tensor(x, dtype=dtype, device=device)) for x in X])

    # Define bounds
    bounds = [(0, 1)] * dim

    # Run optimization
    while len(Y) < max_evals:
        iteration = len(Y)
        alpha =  cosine_annealing_schedule(iteration,
                                           max_evals,
                                           alpha_max,
                                           alpha_min)

        # Compute KD partitions with adaptive leaf size
        cells, probabilities, indices = compute_kd_partitions(
            X, Y, bounds, iteration, max_evals, None,
            m0=0.5*dim, lam=lam, alpha=alpha
        )

        if len(cells) < M:
            M_temp = len(cells)
        else:
            M_temp = M

        # Sample M cells based on probabilities
        # add a tiny chance to probabilities to handle edge cases
        p_smooth = probabilities + 1e-8
        p_smooth /= p_smooth.sum()
        selected_cell_indices = np.random.choice(len(cells), M_temp,
                                                 replace=False,
                                                 p=p_smooth)

        # For HOLLM, use the LLM to generate points in each cell
        all_candidates = []
        all_candidates_pred_f = []

        for idx in selected_cell_indices:
            mins, maxs = cells[idx]
            bounds_cell = list(zip(mins, maxs))

            # Get suggestions from the LLM within this partition
            candidates, candidates_pred_f = llm_optimizer.generate_new_candidates(
                X, Y, bounds_cell, n_candidates=k
            )

            all_candidates.append(candidates)
            all_candidates_pred_f.append(candidates_pred_f)

        # Combine all candidates
        all_candidates = np.vstack(all_candidates)
        all_candidates_pred_f = np.concatenate(all_candidates_pred_f)

        # Sort candidates by predicted function value
        sorted_indices = np.argsort(all_candidates_pred_f)[::-1]  # Reverse for maximization
        sorted_candidates = all_candidates[sorted_indices]

        # Select the best candidates
        if len(Y) + batch_size > max_evals:
            batch_size = max_evals - len(Y)
        new_points = sorted_candidates[:batch_size]

        # Evaluate new points
        Y_new = np.array([fun(torch.tensor(x, dtype=dtype, device=device)) for x in new_points])

        # Append data
        X = np.vstack((X, new_points))
        Y = np.concatenate((Y, Y_new))

        # Print current status
        print(f"{len(X)}) Best value: {Y.max():.2e}")

    return X, Y
