import torch


@torch.no_grad()
def procrustes(grads, unit_scale=False):
    """Orthogonalize gradients.

    See: Senushkin, Dmitry, et al. "Independent component alignment for multi-task learning."
         Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

    Args:
        grads: Tensor with shape (K, D).
    """
    assert len(grads.shape) == 2, \
        f"Invalid shape of 'grads': {grads.shape}. Only 2D tensors are applicable"
    
    cov_grad_matrix_e = grads @ grads.T

    singulars, basis = torch.linalg.eigh(cov_grad_matrix_e, UPLO="U")
    tol = torch.max(singulars) * max(cov_grad_matrix_e.shape[-2:]) * torch.finfo().eps
    rank = sum(singulars > tol)

    order = torch.argsort(singulars, dim=-1, descending=True)
    singulars, basis = singulars[order][:rank], basis[:, order][:, :rank]

    if unit_scale: 
        weights = basis
    else:
        weights = basis * torch.sqrt(singulars[-1]).view(1, -1) 
    weights = weights / torch.sqrt(singulars).view(1, -1)
    weights = weights @ basis.T  # (K, K).
    merged = (weights.sum(1).T @ grads).squeeze(0)
    assert merged.ndim == 1
    return merged
