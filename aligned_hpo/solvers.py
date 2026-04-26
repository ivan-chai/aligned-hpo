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

    # Regularize to ensure positive-definiteness. Use the smallest eigenvalue
    # as a robust check instead of the determinant, which can be misleadingly
    # small for higher-dimensional matrices (product of many eigenvalues).
    if np.linalg.eigvalsh(C)[0] < eps:
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


def solve_qcqp(C, b, positive=False, tol=1e-12, rcond=1e-8, max_iter=100, eps=1e-12):
    """Solve:

        maximize    b^T x
        subject to  x^T C x = 1
                    x >= 0 (if positive=True)

    using the closed-form solution (positive=False) or an active-set method (positive=True).

    C and b are scaled internally for numerical stability with small magnitudes or
    small singular values of C.
    """
    dtype = C.dtype
    device = C.device
    n = len(C)
    C = C.cpu().double().numpy()
    b = b.cpu().double().numpy()

    # Scale inputs for numerical stability. All computation is done in the scaled
    # space where C_s = C / scale_C and b_s = b / scale_b are O(1).
    scale_C = np.abs(C).max()
    scale_b = np.abs(b).max()

    if scale_C < eps or scale_b < eps:
        # Degenerate problem: C is singular to machine precision, or there is no
        # gradient signal in b.  Return the zero vector as a sentinel.
        return torch.zeros(n, dtype=dtype, device=device)

    C_s = C / scale_C
    b_s = b / scale_b

    if not positive:
        # Closed-form solution via KKT: x* = C^{-1} b / sqrt(b^T C^{-1} b).
        # All arithmetic is on scaled quantities so intermediate values are O(1).
        prod = np.linalg.lstsq(C_s, b_s, rcond=rcond)[0]
        norm2 = b_s @ prod  # b_s^T C_s^{-1} b_s, expected O(1) after scaling
        if norm2 <= eps:
            x = np.zeros(n)
        else:
            # x_s is on the unit ellipsoid of C_s.  Convert to unit ellipsoid of C:
            # x_s^T C_s x_s = 1  =>  (x_s/sqrt(scale_C))^T C (x_s/sqrt(scale_C)) = 1
            x_s = prod / np.sqrt(norm2)
            x = x_s / np.sqrt(scale_C)
        return torch.from_numpy(x).to(dtype=dtype, device=device)

    # Positive case: active-set method on scaled problem.
    S = set(np.where(b_s > 0)[0])
    if not S:
        # All b_s[i] <= 0; seed with the index most "aligned" with C_s^{-1/2} b_s.
        diag = np.maximum(np.diag(C_s), eps)
        j = int(np.argmax(b_s / np.sqrt(diag)))
        S = {j}

    x = np.zeros(n)
    max_weights = 1 / np.sqrt(np.diag(C_s)).clip(min=eps)
    best_x = np.argmax(b_s * max_weights)
    x[best_x] = max_weights[best_x]
    for _ in range(max_iter):
        S_list = sorted(S)
        CS = C_s[np.ix_(S_list, S_list)]
        bS = b_s[S_list]

        # Reduced solve on the active set.
        try:
            y = np.linalg.lstsq(CS, bS, rcond=rcond)[0]
        except np.linalg.LinAlgError:
            break

        # Remove nonpositive components from the active set.
        keep = y > tol
        if not np.all(keep):
            S = {S_list[i] for i in range(len(S_list)) if keep[i]}
            if not S:
                break
            continue

        norm2 = bS @ y  # O(1) in scaled space
        if norm2 <= eps:
            break

        denom = np.sqrt(norm2)
        x_s = np.zeros(n)
        x_s[S_list] = y / denom
        lam2 = denom  # equals 2*lambda at the KKT point

        # KKT check on inactive indices.
        r = lam2 * (C_s @ x_s) - b_s
        violated = [i for i in range(n) if i not in S and r[i] < -tol]

        if not violated:
            # Convert x_s (unit ellipsoid of C_s) to unit ellipsoid of C.
            x = x_s / np.sqrt(scale_C)
            break

        # Add the most violated index.
        j = min(violated, key=lambda i: r[i])
        S.add(j)

    return torch.from_numpy(x).to(dtype=dtype, device=device)
