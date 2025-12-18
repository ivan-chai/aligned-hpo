import torch
import numpy as np

from scipy.optimize import minimize


def solve_qp(C, b, steps=100, method="SLSQP", eps=1e-6,
             positive=False, norm=False):
    """Minimize 0.5 x^T C x + b^T x.

    Args:
      positive: Whether to add x > 0 constraint or not.
      norm: Whether to add |x| = 1 constraint or not. Boolean or a number of features to normalize.
    """
    dtype = C.dtype
    device = C.device
    n = len(C)
    C = C.cpu().double().numpy()
    b = b.cpu().double().numpy()

    scale = max(C.mean(), eps ** 2)
    C /= scale
    b /= scale

    func = lambda x: 0.5 * x.T @ C @ x + b @ x
    jac = lambda x: C @ x + b
    cons = []
    if positive:
        eye = np.eye(n)
        cons.append({
            "type": "ineq",
            "fun": lambda x: x,
            "jac": lambda x: eye
        })
    if norm:
        if isinstance(norm, bool):
            norm_size = n
        else:
            norm_size = int(norm)
        cons.append({
            "type": "eq",
            "fun": lambda x: (x[:norm_size] ** 2).sum() - 1,
            "jac": lambda x: 2 * np.concatenate([x[:norm_size], np.zeros_like(x[norm_size:])])
        })
    opt = {
        "disp": False,
        "maxiter": steps
    }
    x0 = np.ones(n) / n
    weights = minimize(func, x0, jac=jac, constraints=cons,
                       method=method, options=opt)["x"]
    if positive:
        weights = np.clip(weights, a_min=0, a_max=None)
    if norm:
        scale = np.linalg.norm(weights[:norm_size])
        if scale < eps:
            weights = np.ones_like(weights)
            weights = np.concatenate([np.ones_like(weights[:norm_size]), weights[norm_size:]])
            scale = np.linalg.norm(weights[:norm_size])
        weights = np.concatenate([weights[:norm_size] / (scale + eps), weights[norm_size:]])
    return torch.from_numpy(weights).to(dtype=dtype, device=device)
