import torch
import numpy as np
import warnings

from qpsolvers import solve_qp as solve_qp_base


def solve_qp(C, b, method="cvxopt", eps=1e-6, maxiters=100,
             positive=False):
    """Minimize 0.5 x^T C x + b^T x.

    Args:
      positive: Whether to add x > 0 constraint or not.
    """
    dtype = C.dtype
    device = C.device
    n = len(C)
    C = C.cpu().double().numpy()
    b = b.cpu().double().numpy()

    scale = max(C.std(), eps)
    C /= scale
    b /= scale

    # Regularize.
    if np.linalg.det(C) < eps:
        C = C + np.eye(n) * eps

    zeros = np.zeros(n, dtype=np.double)
    weights = solve_qp_base(C, b,
                            lb=np.zeros(n) if positive else None,
                            initvals=zeros, solver=method, maxiters=maxiters)
    if weights is None:
        warnings.warn(f"QP solution was not found for C={C} and b={b}.")
        weights = zeros
    if positive:
        weights = np.clip(weights, a_min=0, a_max=None)
        if 0.5 * weights.T @ C @ weights + weights @ b > 0:
            weights[:] = 0
    return torch.from_numpy(weights).to(dtype=dtype, device=device)
