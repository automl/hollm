# -*- coding: utf-8 -*-
"""
Utility functions for KD-tree based partitioning.
"""

import numpy as np
from scipy.spatial import KDTree


def min_max_norm(array):
    min_val = np.min(array)
    max_val = np.max(array)
    diff = max_val - min_val
    if diff == 0:
        # handle case when all values identical
        normalized_array = np.array([0.0 for _ in array])
    else:
        normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

def adaptive_leafsize(t, d, m0=None, lam=1.0, lam2=0.2):
    """
    Calculate adaptive leaf size for KD-tree based on iteration.

    Args:
        t: Current iteration
        d: Dimension of the search space
        m0: Initial leaf size
        lam: Growth rate parameter

    Returns:
        Leaf size
    """
    if m0 is None:
        m0 = 0.5 * d
    return m0 + int(np.ceil(lam * np.log1p(t)))

def compute_kd_partitions(X, Y, bounds, iteration, total_iterations,
                          f_min=None, m0=None, lam=0.0,
                          alpha=0.5, acq_function='ucbv'):
    """
    KD-tree partitioning and scoring function.

    Args:
        X: Current evaluated points
        Y: Function values at X
        bounds: Search space bounds
        iteration: Current iteration
        total_iterations: Total iterations planned
        alpha: Exploration-exploitation trade-off parameter
        f_min: Minimum function value for exploitation calculation
        m0: Initial leaf size for adaptive sizing
        lam: Growth rate parameter for adaptive sizing
        adaptive: Whether to use adaptive leaf size
        acq_function: Acquisition function to use ('moss' or 'ucb')

    Returns:
        cells: List of cell boundaries
        probabilities: Sampling probabilities for each cell
        indices: List of indices of points in each cell
    """
    dim = len(bounds)

    # Calculate adaptive leaf size
    m_leaf = adaptive_leafsize(iteration, dim, m0, lam)
    print(f'>>>> \t Leaf size: {m_leaf}')

    # Compute current best objective if not provided
    if f_min is None:
        f_min = np.min(Y)
    positive_Y = Y - f_min + 1e-6

    # Construct KD-tree with adaptive leaf size
    kdtree = KDTree(X, leafsize=m_leaf, balanced_tree=False)

    # Extract leaf cells
    leaf_cells = get_kd_tree_leaf_cells(kdtree, bounds)

    # Total number of cells/arms
    K = len(leaf_cells)

    # Initialize result containers
    cells = []
    cell_idx = []
    exploitation = []
    ucb = []
    volume = []

    # Process each leaf cell
    for (cell_mins, cell_maxs), indices in leaf_cells:
        idx = np.array(indices, dtype=int)
        cell_idx.append(idx)
        n_ell = len(idx)

        # Handle empty cells
        if n_ell == 0:
            cells.append((cell_mins, cell_maxs))
            exploitation.append(0.0)
            volume.append(1.0)  # Default high exploration for empty cells
            ucb.append(1.0)
            continue

        # Get cell points and values
        cell_points = X[idx]
        cell_values = positive_Y[idx]
        cells.append((cell_mins, cell_maxs))

        # Compute exploitation term (best observed value)
        mu = cell_values.max()
        exploitation.append(mu)

        # Use volume-based sparsity
        sparsity = np.prod(cell_maxs - cell_mins)**(1/dim)
        volume.append(sparsity)

        # Compute variance for confidence bounds
        var = np.var(cell_values, ddof=1) if n_ell > 1 else 0.01

        # Different acquisition functions
        if acq_function == 'ucbv':
            # UCB-V exploration term
            log_term = max(0, np.log(total_iterations / (K * n_ell)))
            ucbv_term = np.sqrt(2 * var * log_term / n_ell) + log_term / n_ell
            ucb.append(ucbv_term)
        else:
            # UCB1 exploration term
            delta = 1.0 / iteration  # Confidence parameter
            beta = np.sqrt(2 * np.log(1/delta))
            ucb1 = beta / np.sqrt(n_ell)
            ucb.append(ucb1)

    # Normalize values
    if len(cells) > 1:
        norm_exploitation = min_max_norm(exploitation)
        norm_volume = min_max_norm(volume)
        norm_ucb = min_max_norm(ucb)
    else:
        norm_exploitation, norm_volume, norm_ucb = np.array([1]), np.array([0]), np.array([0])

    # Combine scores with alpha for exploration-exploitation balance
    combined_scores = norm_exploitation + alpha*(0.5*norm_volume + 0.5*norm_ucb)
    probabilities = combined_scores / np.sum(combined_scores)

    return cells, probabilities, cell_idx


def get_kd_tree_leaf_cells(kdtree, bounds):
    """
    Extract exact hyperrectangles (leaf cells) from a KD-tree.

    Args:
        kdtree: scipy.spatial.KDTree object
        bounds: Original search space bounds

    Returns:
        List of tuples (cell_bounds, cell_indices) where:
        - cell_bounds is a tuple (mins, maxs) defining the cell boundaries
        - cell_indices is a list of point indices belonging to this cell
    """
    from numpy import inf

    def recursive_extract_cells(node, cell_bounds, depth=0):
        """
        Recursively traverse the KD-tree and extract leaf cell boundaries.

        Args:
            node: Current node (can be a leafnode or innernode)
            cell_bounds: Current cell boundaries [[min_x1, max_x1], ..., [min_xd, max_xd]]
            depth: Current depth in the tree (used to determine splitting dimension)

        Returns:
            List of (cell_bounds, cell_indices) for all leaf nodes under this node
        """
        dim = len(bounds)

        # Check if this is a leaf node
        if hasattr(node, 'idx'):  # Leaf node
            # Convert cell_bounds to min/max format
            mins = np.array([b[0] for b in cell_bounds])
            maxs = np.array([b[1] for b in cell_bounds])

            # Return the leaf cell and its point indices
            return [((mins, maxs), node.idx)]

        # Otherwise, it's an inner node
        split_dim = node.split_dim
        split_val = node.split

        # Create bounds for left and right children
        left_bounds = [b.copy() for b in cell_bounds]
        right_bounds = [b.copy() for b in cell_bounds]

        left_bounds[split_dim][1] = split_val  # Left child: upper bound becomes split value
        right_bounds[split_dim][0] = split_val  # Right child: lower bound becomes split value

        # Recursively process left and right children
        cells = []
        if node.less is not None:
            cells.extend(recursive_extract_cells(node.less, left_bounds, depth + 1))
        if node.greater is not None:
            cells.extend(recursive_extract_cells(node.greater, right_bounds, depth + 1))

        return cells

    # Initialize cell bounds to cover the entire search space
    initial_bounds = [[bounds[i][0], bounds[i][1]] for i in range(len(bounds))]

    # Extract all leaf cells from the tree
    leaf_cells = recursive_extract_cells(kdtree.tree, initial_bounds)

    return leaf_cells

