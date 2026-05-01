import torch

from .aligned_hpo import HPO_STAGE_DOWNSTREAM
from .stats import StatsTracker


class DWAOptimizer(torch.optim.Optimizer):
    """Dynamic Weight Averaging (Liu et al., 2019).

    Computes task weights closed-form from the ratio of consecutive per-task losses:

        r_i = L_i(t-1) / L_i(t-2),  w_i = n_tasks * softmax(r / T)

    where T is the temperature hyperparameter. On the first two steps (insufficient
    history) uniform weights are used. No gradient update is performed on the weights.

    Args:
        params: Parameter groups in the same structure as GradNormOptimizer:
            Group 0: Task weights — a single 1D tensor (used for storage only).
            Group 1+: Task-specific head parameters (indices given by heads_groups).
            Group 2+: Shared encoder parameters (all remaining groups).
        base_optimizer_cls: Optimizer class for model parameters.
        base_optimizer_params: Keyword arguments for the base optimizer.
        heads_groups: Indices of task-head parameter groups. Default: (1,).
        weights_names: Optional names for task weights (for logging).
        temperature: Softmax temperature T. Paper default: 2.0.
        ema: EMA coefficient for statistics tracking. Default: 0.9.
        encoder_decoder: Use encoder-decoder decomposition. The closure must return
            (z, losses) where z is the encoder output with z.grad populated after
            backward. A closure_encoder must be passed to hpo_step. See hpo_step.

    The closure passed to hpo_step must have the signature:
        closure(down_weight, weights, retain_graph=False, stage=None) -> Tensor(n_tasks,)
    It must call zero_grad(), compute
        (down_weight * downstream_loss + (weights * individual_losses).sum()).backward(...),
    and return the individual task losses. The optimizer detaches them internally.
    down_weight is 0 or 1; stage is HPO_STAGE_DOWNSTREAM or None.

    In encoder_decoder mode the closure returns (z, Tensor(n_tasks,)) and a separate
    closure_encoder(z_grad) must be provided to hpo_step.
    """

    def __init__(self, params, base_optimizer_cls, base_optimizer_params=None,
                 heads_groups=(1,), weights_names=None,
                 temperature=2.0, ema=0.9, encoder_decoder=False):
        params = list(params)
        if len(params) < 3 or not isinstance(params[0], dict) or not isinstance(params[1], dict):
            raise ValueError(
                "Expected at least 3 parameter groups: group 0 for task weights, "
                "group 1+ for heads, and group 2+ for the shared encoder."
            )
        if len(params[0]["params"]) != 1 or params[0]["params"][0].ndim != 1:
            raise ValueError("Task weights must be a single flat 1D tensor in group 0.")

        defaults = dict(base_optimizer_params or {})
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_cls(
            self.param_groups[1:], **(base_optimizer_params or {})
        )
        self.param_groups = [self.param_groups[0]] + self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        if 0 in heads_groups:
            raise ValueError("Group 0 is reserved for task weights.")
        self.heads_groups = list(sorted(set(heads_groups)))
        encoder_groups = set(range(len(self.param_groups))) - {0} - set(heads_groups)
        self.encoder_groups = list(sorted(encoder_groups))
        if not self.encoder_groups:
            raise ValueError(
                "DWAOptimizer requires at least one shared encoder group (group index 2+)."
            )

        self.n_weights = self.weights.numel()
        if self.n_weights == 0:
            raise ValueError("Task weights tensor is empty.")
        if weights_names is None:
            weights_names = [str(i) for i in range(self.n_weights)]
        elif len(weights_names) != self.n_weights:
            raise ValueError(
                f"weights_names has {len(weights_names)} entries, expected {self.n_weights}."
            )
        self.weights_names = weights_names

        self.temperature = temperature
        self.encoder_decoder = encoder_decoder
        self._loss_history = []
        self._n_updates = 0

        self._weights_tracker = StatsTracker("weights", ema, track_median=False)
        self._losses_tracker = StatsTracker("losses", ema, track_median=False)

    @property
    def need_losses(self):
        return True

    @property
    def use_validation(self):
        return False

    @property
    def weights(self):
        return self.param_groups[0]["params"][0]

    @property
    def metrics(self):
        result = {}
        if self._weights_tracker.last_value is not None:
            for key, val in self._weights_tracker.get().items():
                for name, c in zip(self.weights_names, val):
                    result[f"{key}_{name}"] = c
        if self._losses_tracker.last_value is not None:
            for key, val in self._losses_tracker.get().items():
                for name, c in zip(self.weights_names, val):
                    result[f"{key}_{name}"] = c
        return result

    def step(self, closure=None, *, inner=False):
        if not inner:
            raise ValueError("Use hpo_step() instead of step().")
        self.base_optimizer.step(closure=closure)

    @torch.no_grad()
    def _gather_grads(self, part):
        """Return a flat gradient vector for the named model part ('heads' or 'encoder')."""
        if part == "heads":
            groups = [self.param_groups[i] for i in self.heads_groups]
        else:
            assert part == "encoder"
            groups = [self.param_groups[i] for i in self.encoder_groups]
        grads = []
        for group in groups:
            for p in group["params"]:
                grads.append(p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten())
        if not grads:
            return self.weights.new_zeros(0)
        return torch.cat(grads)

    def _compute_weights(self):
        """DWA closed-form weights from loss history. Uniform when history < 2."""
        n = self.n_weights
        if len(self._loss_history) < 2:
            return self.weights.new_ones(n)
        L_prev = self._loss_history[-1]
        L_prev2 = self._loss_history[-2]
        r = L_prev / L_prev2.clamp(min=1e-8)
        return torch.softmax(r / self.temperature, dim=0).mul_(n)

    def hpo_step(self, closure, closure_encoder=None, embed_fn=None, after_backward_hook=None):
        """Make one DWA optimization step.

        Args:
            closure: callable(down_weight, weights, retain_graph=False, stage=None) -> losses or (z, losses).
                Must zero gradients, compute
                (down_weight * downstream_loss + (weights * individual_losses).sum()).backward(...),
                and return individual task losses. In encoder_decoder mode must return
                (z, losses) where z.grad holds the gradient w.r.t. the encoder output.
                down_weight is 0 or 1; stage is HPO_STAGE_DOWNSTREAM or None.
            closure_encoder: callable(z_grad) -> None. Required in encoder_decoder mode.
                Must zero gradients and call embeddings.backward(z_grad.reshape_as(embeddings)).
            embed_fn: Unused.
            after_backward_hook: A function to call after gradients are estimated (gradient clipping etc.).

        Returns:
            Task weights used for this step, shape (n_tasks,).
        """
        if self.encoder_decoder and closure_encoder is None:
            raise ValueError("Need closure_encoder in encoder_decoder mode.")

        closure = torch.enable_grad()(closure)
        if closure_encoder is not None:
            closure_encoder = torch.enable_grad()(closure_encoder)

        output_weights = torch.empty_like(self.weights)

        @torch.no_grad()
        def inner_closure():
            device = self.weights.device
            computed_weights = self._compute_weights().to(device)

            # Phase 1: downstream backward — collect head grads for downstream loss.
            weights_zeros = self.weights.new_zeros(self.n_weights)
            closure(1, weights_zeros, retain_graph=True, stage=HPO_STAGE_DOWNSTREAM)
            heads_down_grads = self._gather_grads("heads").clone()

            # Phase 2: pretrain backward with DWA-computed weights.
            result = closure(0, computed_weights, retain_graph=False, stage=None)

            if self.encoder_decoder:
                z, losses = result
                if z is None or z.grad is None:
                    raise TypeError(
                        "In encoder_decoder mode, closure must return (z, losses) "
                        "with z.grad set after backward."
                    )
            else:
                losses = result

            if not isinstance(losses, torch.Tensor):
                raise TypeError(
                    f"Closure must return a Tensor (got {type(losses).__name__})."
                )
            if losses.shape != (self.n_weights,):
                raise ValueError(
                    f"Closure returned losses of shape {tuple(losses.shape)}, "
                    f"expected ({self.n_weights},)."
                )
            losses_detached = losses.detach().to(device=device)

            is_distributed = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
            if is_distributed:
                world_size = torch.distributed.get_world_size()
                torch.distributed.all_reduce(losses_detached, op=torch.distributed.ReduceOp.SUM)
                losses_detached /= world_size

            self._loss_history.append(losses_detached.clone())
            if len(self._loss_history) > 2:
                self._loss_history.pop(0)

            self.weights.data.copy_(computed_weights)
            output_weights.copy_(computed_weights)

            if self.encoder_decoder:
                # Save pretrain head grads before closure_encoder calls zero_grad().
                heads_pretrain_grads = self._gather_grads("heads").clone()
                closure_encoder(z.grad)
                # Restore combined head grads (closure_encoder zeroes all grads).
                heads_grad = heads_pretrain_grads + heads_down_grads
                offset = 0
                for i in self.heads_groups:
                    for p in self.param_groups[i]["params"]:
                        numel = p.numel()
                        p.grad = heads_grad[offset:offset + numel].reshape(p.shape)
                        offset += numel
            else:
                # Combine pretrain head grads with downstream head grads.
                pretrain_heads_grad = self._gather_grads("heads").clone()
                heads_grad = pretrain_heads_grad + heads_down_grads
                offset = 0
                for i in self.heads_groups:
                    for p in self.param_groups[i]["params"]:
                        numel = p.numel()
                        p.grad = heads_grad[offset:offset + numel].reshape(p.shape)
                        offset += numel

            self._weights_tracker.update(computed_weights.clone())
            self._losses_tracker.update(losses_detached.clone())
            self._n_updates += 1

            if after_backward_hook is not None:
                after_backward_hook()

        self.step(inner_closure, inner=True)
        return output_weights

    def state_dict(self):
        state = self.base_optimizer.state_dict()
        state["weights_data"] = self.weights.data.clone()
        state["loss_history"] = list(self._loss_history)
        state["n_updates"] = self._n_updates
        state["weights_tracker"] = self._weights_tracker.state_dict()
        state["losses_tracker"] = self._losses_tracker.state_dict()
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        weights_data = state_dict.pop("weights_data", None)
        loss_history = state_dict.pop("loss_history", [])
        n_updates = state_dict.pop("n_updates", 0)
        weights_tracker_state = state_dict.pop("weights_tracker", None)
        losses_tracker_state = state_dict.pop("losses_tracker", None)
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = [self.param_groups[0]] + self.base_optimizer.param_groups
        if weights_data is not None:
            self.weights.data.copy_(weights_data)
        self._loss_history = list(loss_history)
        self._n_updates = n_updates
        if weights_tracker_state is not None:
            self._weights_tracker.load_state_dict(weights_tracker_state)
        if losses_tracker_state is not None:
            self._losses_tracker.load_state_dict(losses_tracker_state)
