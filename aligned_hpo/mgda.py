import torch

from .aligned_hpo import HPO_STAGE_DOWNSTREAM
from .stats import StatsTracker


def _solve_min_norm(G, max_iter=200):
    """Frank-Wolfe min-norm solver: min_{α: Σα_i=1, α_i≥0} α^T G α.

    G: (n, n) Gram matrix G_ij = g_i · g_j of per-task encoder gradients.
    Returns α of shape (n,) summing to 1 with non-negative entries.
    """
    n = G.shape[0]
    alpha = G.new_ones(n) / n
    for _ in range(max_iter):
        grad = G @ alpha  # gradient of α^T G α (up to factor 2)
        j = int(grad.argmin())
        s = torch.zeros_like(alpha)
        s[j] = 1.0
        d = s - alpha
        num = -(d @ grad)
        den = d @ (G @ d)
        if den < 1e-12:
            break
        gamma = (num / den).clamp(0.0, 1.0)
        alpha = alpha + gamma * d
    return alpha


class MGDAOptimizer(torch.optim.Optimizer):
    """MGDA: Multi-Task Learning as Multi-Objective Optimization (Sener & Koltun, 2018).

    Finds per-task weights α by solving the min-norm point problem in the convex hull
    of per-task encoder gradients:

        min_{α: Σα_i=1, α_i≥0}  ||Σ α_i g_i||²

    solved via the Frank-Wolfe algorithm on the Gram matrix G_ij = g_i · g_j.

    Args:
        params: Parameter groups:
            Group 0: Task weights — a single 1D tensor (storage for computed α).
            Group 1+: Task-specific head parameters (indices given by heads_groups).
            Group 2+: Shared encoder parameters (all remaining groups).
        base_optimizer_cls: Optimizer class for model parameters.
        base_optimizer_params: Keyword arguments for the base optimizer.
        heads_groups: Indices of task-head parameter groups. Default: (1,).
        weights_names: Optional names for task weights (for logging).
        max_iter: Frank-Wolfe iteration limit. Default: 200.
        ema: EMA coefficient for statistics tracking. Default: 0.9.
        encoder_decoder: Use encoder-decoder decomposition. The closure must return
            (z, losses) where z is the encoder output with z.grad populated after
            backward. A closure_encoder must be passed to hpo_step.

    The closure passed to hpo_step must have the signature:
        closure(down_weight, weights, retain_graph=False, stage=None) -> Tensor(n_tasks,)
    It must call zero_grad(), compute
        (down_weight * downstream_loss + (weights * individual_losses).sum()).backward(...),
    and return the individual task losses. The optimizer detaches them internally.
    down_weight is 0 or 1; stage is HPO_STAGE_DOWNSTREAM or integer task index.

    In encoder_decoder mode the closure returns (z, Tensor(n_tasks,)) and a separate
    closure_encoder(z_grad) must be provided to hpo_step.
    """

    def __init__(self, params, base_optimizer_cls, base_optimizer_params=None,
                 heads_groups=(1,), weights_names=None,
                 max_iter=200, ema=0.9, encoder_decoder=False):
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
                "MGDAOptimizer requires at least one shared encoder group (group index 2+)."
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

        self.max_iter = max_iter
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
        """Make one MGDA optimization step.

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
            Task weights α (summing to 1, non-negative) used for this step, shape (n_tasks,).
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
                z = closure(0, weights_i, retain_graph=(i < self.n_weights - 1), stage=i)
                weights_i[i] = 0.0

                if self.encoder_decoder:
                    if z is None or z.grad is None:
                        raise TypeError(
                            "In encoder_decoder mode, closure must return (z, losses) "
                            "with z.grad set after backward."
                        )
                    all_encoder_grads.append(z.grad.flatten().clone())
                else:
                    all_encoder_grads.append(self._gather_grads("encoder").clone())

                all_heads_grads.append(self._gather_grads("heads").clone())

            # Build Gram matrix and solve min-norm QP via Frank-Wolfe.
            grads_stack = torch.stack(all_encoder_grads)  # (n_tasks, dim)
            G = grads_stack @ grads_stack.T  # (n_tasks, n_tasks)

            is_distributed = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
            if is_distributed:
                world_size = torch.distributed.get_world_size()
                torch.distributed.all_reduce(G, op=torch.distributed.ReduceOp.SUM)
                G /= world_size

            alpha = _solve_min_norm(G, max_iter=self.max_iter).to(device=device)

            # Per-task gradient norms for logging (before applying alpha).
            g_norms = torch.stack([v.norm() for v in all_encoder_grads]).detach()

            # Weighted encoder gradient: Σ α_i g_i.
            encoder_grad = alpha @ grads_stack  # (dim,)

            # Weighted head gradient + downstream head gradient.
            all_heads_grads_t = torch.stack(all_heads_grads)  # (n_tasks, dim)
            heads_grad = alpha @ all_heads_grads_t + heads_down_grads  # computed before zero_grad

            if self.encoder_decoder:
                # closure_encoder calls zero_grad() internally; apply head grads after.
                closure_encoder(encoder_grad)
                offset = 0
                for i in self.heads_groups:
                    for p in self.param_groups[i]["params"]:
                        numel = p.numel()
                        p.grad = heads_grad[offset:offset + numel].reshape(p.shape)
                        offset += numel
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

            self.weights.data.copy_(alpha)
            output_weights.copy_(alpha)

            self._weights_tracker.update(alpha.clone())
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
