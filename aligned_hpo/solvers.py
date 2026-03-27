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


def solve_qcqp(C, b, positive=False, tol=1e-12, rcond=1e-8, max_iter=100):
    """Solve:

        maximize    b^T x
        subject to  x^T C x = 1
                    x >= 0

    using an active-set method.
    """
    dtype = C.dtype
    device = C.device
    n = len(C)
    C = C.cpu().double().numpy()
    b = b.cpu().double().numpy()

    if not positive:
        # Closed-form solution.
        prod = np.linalg.lstsq(C, b, rcond=rcond)[0]
        norm2 = b @ prod
        if norm2 < 0:
            x = np.zeros_like(b)
        else:
            x = prod / (norm2 ** 0.5 + 1e-12)
        return torch.from_numpy(x).to(dtype=dtype, device=device)

    # initial support
    S = set(np.where(b > 0)[0])
    if not S:
        j = np.argmax(b / np.sqrt(np.diag(C)))
        S = {j}

    for _ in range(max_iter):
        S_list = sorted(S)
        CS = C[np.ix_(S_list, S_list)]
        bS = b[S_list]

        # reduced solve
        y = np.linalg.lstsq(CS, bS, rcond=rcond)[0]

        # remove nonpositive components
        keep = y > tol
        if not np.all(keep):
            S = {S_list[i] for i in range(len(S_list)) if keep[i]}
            continue

        norm2 = bS @ y
        if norm2 < 0:
            x = np.zeros_like(b)
            break
        denom = np.sqrt(norm2) + 1e-12
        x = np.zeros(n)
        x[S_list] = y / denom

        lam2 = denom   # equals 2*lambda

        # KKT check on inactive indices
        r = lam2 * (C @ x) - b
        violated = [i for i in range(n) if i not in S and r[i] < -tol]

        if not violated:
            break

        # add most violated index
        j = min(violated, key=lambda i: r[i])
        S.add(j)

    return torch.from_numpy(x).to(dtype=dtype, device=device)
