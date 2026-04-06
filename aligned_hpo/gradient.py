import math
import torch


class GradientNormalizer(torch.nn.Module):
    """Normalize gradient value using running gradient norm mean.

    Outputs:
        Normalized gradient norm.
    """

    def __init__(self, clip=1e-2, momentum=0.9):
        super().__init__()
        self._momentum = momentum
        self._clip = clip
        self.register_buffer("is_first", torch.ones(1, dtype=torch.bool))
        self.register_buffer("moving_norm", torch.zeros([]))

    def forward(self, parameters):
        parameters = [p for p in parameters if p.requires_grad and (p.grad is not None)]
        if not parameters:
            return 0
        device = parameters[0].device
        if self.moving_norm.device != device:
            self.to(device)
        with torch.no_grad():
            norm = self._compute_grad_norm(parameters)
            momentum = 0 if self.is_first else self._momentum
            self.moving_norm.fill_(self.moving_norm * momentum + (1 - momentum) * norm)
            mean = self.moving_norm.clip(min=self._clip).item()
            for p in parameters:
                p.grad /= mean
            self.is_first.fill_(False)
        return norm / mean

    @staticmethod
    def _compute_grad_norm(parameters):
        norms = torch.zeros(len(parameters), device=parameters[0].device)
        for i, p in enumerate(parameters):
            norms[i] = p.grad.data.norm(2)
        return norms.square().sum().item() ** 0.5
