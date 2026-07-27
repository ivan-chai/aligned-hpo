import torch

from .aligned_hpo import HPO_STAGE_DOWNSTREAM
from .stats import StatsTracker


def _pcgrad_surgery(grads):
    """Apply PCGrad gradient surgery to per-task gradient vectors.

    For each task i, projects out the component of g_i along g_j whenever
    g_i · g_j < 0 (conflicting gradients). Surgery uses the original g_j
    vectors as projection directions (not the iteratively modified ones).

    Args:
        grads: list of n tensors, shape (dim,) each.
    Returns:
        list of n modified tensors, shape (dim,) each.
    """
    n = len(grads)
    result = [g.clone() for g in grads]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dot = result[i] @ grads[j]
            if dot < 0:
                norm_sq = grads[j] @ grads[j]
                if norm_sq > 1e-12:
                    result[i] = result[i] - (dot / norm_sq) * grads[j]
    return result


class PCGradOptimizer(torch.optim.Optimizer):
    """PCGrad: Gradient Surgery for Multi-Task Learning (Yu et al., 2020).

    For each task pair (i, j) where g_i · g_j < 0, projects the conflicting
    component of g_i away from g_j. The encoder (and head) update is the sum
    of all surgically modified per-task gradients plus the downstream head
    gradient. No learned task weights are maintained; the weights tensor in
    group 0 stores uniform values 1/n_tasks.

    In encoder_decoder mode the surgery is applied in the embedding space
    (to z.grad vectors before passing the aggregated gradient to the encoder).

    Args:
        params: Parameter groups:
            Group 0: Task weights — a single 1D tensor (storage for 1/n weights).
            Group 1: Task-specific head parameters (indices given by heads_groups).
            Group 2+: Shared encoder parameters (all remaining groups).
        base_optimizer_cls: Optimizer class for model parameters.
        base_optimizer_params: Keyword arguments for the base optimizer.
        heads_groups: Indices of task-head parameter groups. Default: (1,).
        shared_groups: Indices of parameter groups, related to shared loss heads parameters.
        weights_names: Optional names for task weights (for logging).
        ema: EMA coefficient for statistics tracking. Default: 0.9.
        encoder_decoder: Use encoder-decoder decomposition. Surgery is applied
            in the embedding space to z.grad vectors. A closure_encoder must
            be passed to hpo_step.

    The closure passed to hpo_step must have the signature:
        closure(down_weight, weights, retain_graph=False, stage=None) -> Tensor(n_tasks,)
    It must call zero_grad(), compute
        (down_weight * downstream_loss + (weights * individual_losses).sum()).backward(...),
    and return the individual task losses. The optimizer detaches them internally.
    down_weight is 0 or 1; stage is HPO_STAGE_DOWNSTREAM or integer task index.

    In encoder_decoder mode the closure returns (z, Tensor(n_tasks,)) and a separate
    closure_encoder(z_grad) must be provided to hpo_step.

    DDP note: surgery is applied locally per-rank (same approximation as the other
    HPO optimizers — full gradient vectors are not all-reduced).
    """

    def __init__(self, params, base_optimizer_cls, base_optimizer_params=None,
                 heads_groups=(1,), shared_groups=(), weights_names=None,
                 ema=0.9, encoder_decoder=False):
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

        heads_groups = set(heads_groups) | set(shared_groups)
        if 0 in heads_groups:
            raise ValueError("Group 0 is reserved for task weights.")
        self.heads_groups = list(sorted(set(heads_groups)))
        encoder_groups = set(range(len(self.param_groups))) - {0} - set(heads_groups)
        self.encoder_groups = list(sorted(encoder_groups))
        if not self.encoder_groups:
            raise ValueError(
                "PCGradOptimizer requires at least one shared encoder group (group index 2+)."
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

        self.encoder_decoder = encoder_decoder
        self._n_updates = 0

        self._weights_tracker = StatsTracker("weights", ema, track_median=False)
        self._g_norms_tracker = StatsTracker("grad_norms", ema, track_median=False)

    @property
    def need_losses(self):
        return False

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
        if self._g_norms_tracker.last_value is not None:
            for key, val in self._g_norms_tracker.get().items():
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

    def hpo_step(self, closure, closure_encoder=None, embed_fn=None, after_backward_hook=None):
        """Make one PCGrad optimization step.

        Args:
            closure: callable(down_weight, weights, retain_graph=False, stage=None) -> losses or (z, losses).
                Must zero gradients, compute
                (down_weight * downstream_loss + (weights * individual_losses).sum()).backward(...),
                and return individual task losses. In encoder_decoder mode must also zero z.grad
                and return (z, losses) where z.grad is set after backward.
                down_weight is 0 or 1; stage is HPO_STAGE_DOWNSTREAM or integer task index.
            closure_encoder: callable(z_grad) -> None. Required in encoder_decoder mode.
                Must zero gradients and call embeddings.backward(z_grad.reshape_as(embeddings)).
            embed_fn: Unused.
            after_backward_hook: A function to call after gradients are estimated (gradient clipping etc.).

        Returns:
            Uniform task weights 1/n_tasks, shape (n_tasks,).
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

            # Phase 1: downstream backward — collect head grads for downstream loss.
            weights_zeros = self.weights.new_zeros(self.n_weights)
            closure(1, weights_zeros, retain_graph=True, stage=HPO_STAGE_DOWNSTREAM)
            heads_down_grads = self._gather_grads("heads").clone()

            # Phase 2: per-task backwards — collect per-task encoder and head grads.
            all_encoder_grads = []
            all_heads_grads = []

            weights_i = self.weights.new_zeros(self.n_weights)
            for i in range(self.n_weights):
                weights_i[i] = 1.0
                result = closure(0, weights_i, retain_graph=(i < self.n_weights - 1), stage=i)
                weights_i[i] = 0.0

                if self.encoder_decoder:
                    z = result
                    if z is None or z.grad is None:
                        raise TypeError(
                            "In encoder_decoder mode, closure must return (z, losses) "
                            "with z.grad set after backward."
                        )
                    all_encoder_grads.append(z.grad.flatten().clone())
                else:
                    all_encoder_grads.append(self._gather_grads("encoder").clone())

                all_heads_grads.append(self._gather_grads("heads").clone())

            # PCGrad surgery on encoder (or embedding) gradients.
            surgered_encoder = _pcgrad_surgery(all_encoder_grads)
            encoder_grad = torch.stack(surgered_encoder).sum(0)  # (dim,)

            # PCGrad surgery on head gradients + downstream head gradients.
            surgered_heads = _pcgrad_surgery(all_heads_grads)
            heads_grad = torch.stack(surgered_heads).sum(0) + heads_down_grads

            # Per-task gradient norms (before surgery) for logging.
            g_norms = torch.stack([g.norm() for g in all_encoder_grads]).detach()

            if self.encoder_decoder:
                # closure_encoder calls zero_grad() internally; apply head grads after.
                closure_encoder(encoder_grad)
            else:
                offset = 0
                for i in self.encoder_groups:
                    for p in self.param_groups[i]["params"]:
                        numel = p.numel()
                        p.grad = encoder_grad[offset:offset + numel].reshape(p.shape)
                        offset += numel

            offset = 0
            for i in self.heads_groups:
                for p in self.param_groups[i]["params"]:
                    numel = p.numel()
                    p.grad = heads_grad[offset:offset + numel].reshape(p.shape)
                    offset += numel

            uniform_weights = self.weights.new_ones(self.n_weights) / self.n_weights
            self.weights.data.copy_(uniform_weights)
            output_weights.copy_(uniform_weights)

            self._weights_tracker.update(uniform_weights.clone())
            self._g_norms_tracker.update(g_norms.clone())
            self._n_updates += 1

            if after_backward_hook is not None:
                after_backward_hook()

        self.step(inner_closure, inner=True)
        return output_weights

    def hpo_state_dict(self, add_names=False):
        state = {}
        state["weights_data"] = self.weights.data.clone()
        state["n_updates"] = self._n_updates
        state["weights_tracker"] = self._weights_tracker.state_dict()
        state["g_norms_tracker"] = self._g_norms_tracker.state_dict()
        return state

    def state_dict(self):
        state = self.base_optimizer.state_dict()
        state.update(self.hpo_state_dict())
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        weights_data = state_dict.pop("weights_data", None)
        n_updates = state_dict.pop("n_updates", 0)
        weights_tracker_state = state_dict.pop("weights_tracker", None)
        g_norms_tracker_state = state_dict.pop("g_norms_tracker", None)
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = [self.param_groups[0]] + self.base_optimizer.param_groups
        if weights_data is not None:
            self.weights.data.copy_(weights_data)
        self._n_updates = n_updates
        if weights_tracker_state is not None:
            self._weights_tracker.load_state_dict(weights_tracker_state)
        if g_norms_tracker_state is not None:
            self._g_norms_tracker.load_state_dict(g_norms_tracker_state)
