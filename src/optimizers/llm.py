import numpy as np
import torch

from src.utils import get_initial_points


def run_llm(fun, dim, batch_size, n_init, max_evals,
            llm_optimizer, device, k, M, dtype, api_key, seed=0):
    """
    Run optimization using LLM-based suggestions.

    Args:
        fun: Objective function
        dim: Dimension of the search space
        batch_size: Number of points to evaluate in each batch
        n_init: Number of initial points
        max_evals: Maximum number of function evaluations
        device: PyTorch device
        dtype: PyTorch dtype
        api_key: API key for LLM
        seed: Random seed

    Returns:
        X: Evaluated points
        Y: Function values
    """

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate initial points
    X = get_initial_points(dim, n_init, seed, dtype, random=True).cpu().numpy()
    Y = np.array([fun(torch.tensor(x, dtype=dtype, device=device)) for x in X])

    if len(Y) + batch_size > max_evals:
        batch_size = max_evals - len(Y)

    # Define bounds for normalized space
    bounds = [(0, 1)] * dim

    # Run optimization
    while len(X) < max_evals:

        if len(X) + batch_size > max_evals:
            batch_size = max_evals - len(Y)

        # Generate candidates using LLM
        candidates, candidates_pred_f = llm_optimizer.generate_new_candidates(
            X, Y, bounds, n_candidates=k*M
        )

        # Sort candidates by predicted function value
        sorted_indices = np.argsort(candidates_pred_f)[::-1]  # Reverse for maximization
        sorted_candidates = candidates[sorted_indices]

        # Select top candidates
        new_points = sorted_candidates[:batch_size]

        # Evaluate new points
        Y_new = np.array([fun(torch.tensor(x, dtype=dtype, device=device)) for x in new_points])

        # Append data
        X = np.vstack((X, new_points))
        Y = np.concatenate((Y, Y_new))

        # Print current status
        print(f"{len(X)}) Best value: {Y.max():.2e}")

    return X, Y

