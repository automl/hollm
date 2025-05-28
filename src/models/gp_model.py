# -*- coding: utf-8 -*-
"""
GP model implementations for Bayesian optimization.
"""

import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood


def create_and_fit_gp_model(X, Y, dim, standardize=True, max_cholesky_size=float("inf")):
    """
    Create and fit a GP model.

    Args:
        X: Input points (n x d)
        Y: Function values (n x 1)
        dim: Dimension of the input space
        standardize: Whether to standardize Y
        max_cholesky_size: Maximum size for using Cholesky decomposition

    Returns:
        Fitted GP model
    """

    # Ensure consistent data type between X and Y
    dtype = X.dtype
    X = X.to(dtype=dtype)
    Y = Y.to(dtype=dtype)

    # Standardize Y if requested
    if standardize:
        train_Y = (Y - Y.mean()) / Y.std()
    else:
        train_Y = Y

    # Create likelihood and kernel
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
        )
    )

    # Create model
    model = SingleTaskGP(
        X, train_Y, covar_module=covar_module, likelihood=likelihood
    )

    # Create marginal log likelihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Fit the model
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        fit_gpytorch_mll(mll)

    return model
