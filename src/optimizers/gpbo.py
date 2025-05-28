from src.utils import get_initial_points
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from src.acquisition_strategies import generate_batch_ei


def run_gpbo(fun, dim, batch_size, n_init, acq, max_evals, device, dtype,
             seed=0):
    """
    Run optimization using Log Expected Improvement.

    Args:
        fun: Objective function
        dim: Dimension of the search space
        batch_size: Number of points to evaluate in each batch
        n_init: Number of initial points
        max_evals: Maximum number of function evaluations
        device: PyTorch device
        dtype: PyTorch dtype
        seed: Random seed

    Returns:
        X: Evaluated points
        Y: Function values
    """

    # Set random seed
    torch.manual_seed(seed)

    # Generate initial points
    X = get_initial_points(dim, n_init, seed, random=True).to(dtype=dtype, device=device)
    Y = torch.tensor(
        [fun(x) for x in X], dtype=dtype, device=device
    ).unsqueeze(-1)

    # Run until max evaluations
    while len(X) < max_evals:
        if len(X) + batch_size > max_evals:
            batch_size = max_evals - len(Y)

        # Normalize Y
        train_Y = (Y - Y.mean()) / Y.std()

        # Create and fit model
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        model = SingleTaskGP(X, train_Y, likelihood=likelihood)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Create batch
        bounds = torch.stack([
            torch.zeros(dim, dtype=dtype, device=device),
            torch.ones(dim, dtype=dtype, device=device)
        ])

        candidate, _ = generate_batch_ei(
            model,
            train_Y.max(),
            bounds,
            batch_size,
            log=True if acq=='logei' else False,
            num_restarts=10,
            raw_samples=512
        )

        # Evaluate new points
        Y_next = torch.tensor(
            [fun(x) for x in candidate], dtype=dtype, device=device
        ).unsqueeze(-1)

        # Append data
        X = torch.cat((X, candidate), dim=0)
        Y = torch.cat((Y, Y_next), dim=0)

        # Print current status
        print(f"{len(X)}) Best value: {Y.max().item():.2e}")

    return X, Y


