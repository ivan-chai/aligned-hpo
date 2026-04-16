import math
import torch


class GradientNormalizer(torch.nn.Module):
    """Normalize gradient value using running gradient norm mean.

    Outputs:
        Normalized gradient norm.
    """

    def __init__(self, clip=1e-2, momentum=0.9, check_shape=True):
        super().__init__()
        self._check_shape = check_shape
        self._momentum = momentum
        self._clip = clip
        self._shape = None
        self._is_first = True
        self.register_buffer("moving_norm", torch.zeros([]))

    @property
    def is_first(self):
        return self._is_first

    def forward(self, parameters):
        if isinstance(parameters, torch.Tensor):
            shape = list(parameters.shape)
            device = parameters.device
            with torch.no_grad():
                norm = torch.linalg.norm(parameters.flatten())
        else:
            parameters = [p for p in parameters if p.requires_grad and (p.grad is not None)]
            if not parameters:
                return 0
            shape = [sum(p.numel() for p in parameters)]
            device = parameters[0].device
            with torch.no_grad():
                norm = self._compute_grad_norm(parameters)
        if self._shape is None:
            self._shape = shape
        elif self._check_shape and (self._shape != shape):
            raise ValueError("Gradient shape mismatch")
        if self.moving_norm.device != device:
            self.to(device)
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized() and (torch.distributed.get_world_size() > 1)
        if is_distributed:
            # We use triangle inequality and approximate norm of the mean with mean of the norms.
            torch.distributed.all_reduce(norm, op=torch.distributed.ReduceOp.SUM)
            norm /= torch.distributed.get_world_size()
        with torch.no_grad():
            momentum = 0 if self._is_first else self._momentum
            self.moving_norm.fill_(self.moving_norm * momentum + (1 - momentum) * norm)
            mean = self.moving_norm.clip(min=self._clip).item()
            if isinstance(parameters, torch.Tensor):
                parameters /= mean
            else:
                for p in parameters:
                    p.grad /= mean
            self._is_first = False
        return norm / mean

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["is_first"] = self._is_first
        state_dict["shape"] = self._shape
        return state_dict

    def load_state_dict(self, state_dict):
        self._is_first = state_dict["is_first"]
        self._shape = state_dict["shape"]
        state_dict = dict(state_dict)
        del state_dict["is_first"]
        del state_dict["shape"]
        super().load_state_dict(state_dict)

    @staticmethod
    def _compute_grad_norm(parameters):
        norms = torch.zeros(len(parameters), device=parameters[0].device)
        for i, p in enumerate(parameters):
            norms[i] = p.grad.data.norm(2)
        return norms.square().sum().item() ** 0.5
