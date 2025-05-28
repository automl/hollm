# -*- coding: utf-8 -*-
"""
Acquisition strategies for Bayesian optimization.
"""

import torch
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf


def generate_batch_ts(
    model, 
    x_center, 
    tr_lb, 
    tr_ub, 
    n_candidates, 
    batch_size, 
    dim,
    dtype,
    device
):
    """
    Generate a batch of points using Thompson sampling.
    
    Args:
        model: GP model
        x_center: Center of the trust region
        tr_lb: Lower bound of the trust region
        tr_ub: Upper bound of the trust region
        n_candidates: Number of candidate points
        batch_size: Number of points to select
        dim: Dimension of the search space
        dtype: PyTorch dtype
        device: PyTorch device
        
    Returns:
        A batch of selected points
    """
    # Create a set of candidate points by perturbing the best point
    sobol = torch.quasirandom.SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points
    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():  # Don't need gradients for TS
        X_next = thompson_sampling(X_cand, num_samples=batch_size)
        
    return X_next


def generate_batch_ei(
    model, 
    best_f, 
    bounds,
    batch_size, 
    log,
    num_restarts, 
    raw_samples
):
    """
    Generate a batch of points using Expected Improvement.

    Args:
        model: GP model
        best_f: Best function value observed so far
        tr_lb: Lower bound of the trust region
        tr_ub: Upper bound of the trust region
        batch_size: Number of points to select
        num_restarts: Number of restarts for optimization
        raw_samples: Number of raw samples for optimization

    Returns:
        A batch of selected points and their acquisition values
    """

    acq = qLogExpectedImprovement if log else qExpectedImprovement

    ei =acq(model, best_f)
    X_next, acq_value = optimize_acqf(
        ei,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return X_next, acq_value

