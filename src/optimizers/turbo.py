# -*- coding: utf-8 -*-
"""
Implementation of Trust Region Bayesian Optimization (TuRBO).
"""

import math
import torch
import gpytorch
from dataclasses import dataclass

from src.models.gp_model import create_and_fit_gp_model
from src.acquisition_strategies import generate_batch_ts, generate_batch_ei


@dataclass
class TurboState:
    """
    Class for maintaining the state of TuRBO.
    """
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    """
    Update the TuRBO state based on new function evaluations.
    
    Args:
        state: Current TuRBO state
        Y_next: New function evaluations
        
    Returns:
        Updated TuRBO state
    """
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    
    return state


def generate_batch(
    state,
    model,
    X,
    Y,
    batch_size,
    n_candidates=None,
    num_restarts=10,
    raw_samples=512,
    acqf="ts",
):
    """
    Generate a batch of points for TuRBO.

    Args:
        state: Current TuRBO state
        model: GP model
        X: Evaluated points (normalized to [0, 1])
        Y: Function values
        batch_size: Number of points to select
        n_candidates: Number of candidates for Thompson sampling
        num_restarts: Number of restarts for optimization
        raw_samples: Number of raw samples for optimization
        acqf: Acquisition function ("ts" or "ei")

    Returns:
        A batch of selected points
    """
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        X_next = generate_batch_ts(
            model, 
            x_center, 
            tr_lb, 
            tr_ub, 
            n_candidates, 
            batch_size, 
            X.shape[-1],
            X.dtype,
            X.device
        )
    elif acqf == "ei":
        X_next, _ = generate_batch_ei(
            model, 
            Y.max(), 
            tr_lb, 
            tr_ub, 
            batch_size, 
            num_restarts, 
            raw_samples
        )

    return X_next


def run_turbo(
    fun,
    dim,
    batch_size,
    n_init,
    max_evals,
    device=None,
    dtype=None,
    acqf="ts",
    seed=0,
):
    """
    Run TuRBO until convergence or max evaluations.
    
    Args:
        fun: Objective function (with eval_objective interface)
        dim: Dimension of the search space
        batch_size: Number of points to evaluate in each batch
        n_init: Number of initial points
        max_evals: Maximum number of function evaluations
        device: PyTorch device
        dtype: PyTorch dtype
        acqf: Acquisition function ("ts" or "ei")
        seed: Random seed
        
    Returns:
        X: Evaluated points
        Y: Function values
    """
    # Set random seed
    torch.manual_seed(seed)
    
    # Generate initial points
    from src.utils import get_initial_points
    X = get_initial_points(dim, n_init, seed, dtype, random=True)
    Y = torch.tensor(
        [fun(x) for x in X], dtype=dtype, device=device
    ).unsqueeze(-1)
    
    # Initialize TuRBO state
    state = TurboState(dim, batch_size=batch_size, best_value=max(Y).item())
    
    # Constants for optimization
    num_restarts = 10
    raw_samples = 512
    n_candidates = min(5000, max(2000, 200 * dim))
    max_cholesky_size = float("inf")
    
    # Run TuRBO until convergence or max evaluations
    while len(X) < max_evals:# and not state.restart_triggered:

        if len(X) + batch_size > max_evals:
            batch_size = max_evals - len(Y)

        # Fit a GP model
        model = create_and_fit_gp_model(X, Y, dim, max_cholesky_size=max_cholesky_size)

        # Generate batch of new points
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            X_next = generate_batch(
                state=state,
                model=model,
                X=X,
                Y=Y,
                batch_size=batch_size,
                n_candidates=n_candidates,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                acqf=acqf,
            )
        
        # Evaluate new points
        Y_next = torch.tensor(
            [fun(x) for x in X_next], dtype=dtype, device=device
        ).unsqueeze(-1)
        
        # Update state
        state = update_state(state=state, Y_next=Y_next)
        
        # Append data
        X = torch.cat((X, X_next), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)
        
        # Print current status
        print(
            f"{len(X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
        )
    
    return X, Y
