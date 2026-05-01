import torch

from .aligned_hpo import HPO_STAGE_DOWNSTREAM
from .stats import StatsTracker


class GradNormOptimizer(torch.optim.Optimizer):
    """GradNorm: Gradient Normalization for Adaptive Loss Balancing (Chen et al., 2018).

    Automatically balances task weights so that each task's gradient norm at the shared
    encoder matches a target that accounts for its relative training rate.

    Args:
        params: Parameter groups in the same structure as AlignedHPOptimizer:
            Group 0: Task weights — a single 1D tensor, one element per task.
            Group 1+: Task-specific head parameters (indices given by heads_groups).
            Group 2+: Shared encoder parameters (all remaining groups).
        base_optimizer_cls: Optimizer class for model parameters.
        base_optimizer_params: Keyword arguments for the base optimizer.
        weights_optimizer_cls: Optimizer for task weights. Defaults to base_optimizer_cls.
        weights_optimizer_params: Keyword arguments for the weights optimizer.
        heads_groups: Indices of task-head parameter groups. Default: (1,).
        weights_names: Optional names for task weights (for logging).
        alpha: Restoring force strength. Paper default: 1.5.
        ema: EMA coefficient for statistics tracking. Default: 0.9.
        encoder_decoder: Use encoder-decoder decomposition. The closure must return
            (z, losses) where z is the encoder output with z.grad populated after
            backward. A closure_encoder must be passed to hpo_step. See hpo_step.

    The closure passed to hpo_step must have the signature:
        closure(down_weight, weights, retain_graph=False, stage=None) -> Tensor(n_tasks,)
    It must call zero_grad(), compute
        (down_weight * downstream_loss + (weights * individual_losses).sum()).backward(...),
    and return the individual task losses. GradNorm detaches them internally. The closure
    is called once with down_weight=1 to obtain downstream head gradients, then once per
    task with a one-hot weight vector to obtain per-task encoder and head gradients.
    down_weight is 0 or 1; stage is HPO_STAGE_DOWNSTREAM or integer task index.

    In encoder_decoder mode the closure returns (z, Tensor(n_tasks,)) and a separate
    closure_encoder(z_grad) must be provided to hpo_step.

    Example (full mode):
        task_weights = torch.ones(2, requires_grad=True)

        optimizer = GradNormOptimizer(
            [{"params": [task_weights]},
             {"params": list(head1.parameters()) + list(head2.parameters()) + list(head_down.parameters())},
             {"params": encoder.parameters()}],
            torch.optim.Adam, {"lr": 1e-3},
        )

        z = encoder(x)
        l_down = criterion_down(head_down(z), y_down)
        loss1 = criterion1(head1(z), y1)
        loss2 = criterion2(head2(z), y2)

        def closure(down_weight, weights, retain_graph=False, stage=None):
            optimizer.zero_grad()
            losses = torch.stack([loss1, loss2])
            loss = down_weight * l_down + (weights * losses).sum()
            loss.backward(retain_graph=retain_graph)
            return losses

        optimizer.hpo_step(closure)

    Example (encoder_decoder mode):
        optimizer = GradNormOptimizer(..., encoder_decoder=True)

        embeddings = encoder(x)
        z = embeddings.detach().clone()
        z.requires_grad = True
        l_down = criterion_down(head_down(z), y_down)
        loss1 = criterion1(head1(z), y1)
        loss2 = criterion2(head2(z), y2)

        def closure(down_weight, weights, retain_graph=False, stage=None):
            optimizer.zero_grad()
            z.grad = None
            losses = torch.stack([loss1, loss2])
            loss = down_weight * l_down + (weights * losses).sum()
            loss.backward(retain_graph=retain_graph)
            return z, losses

        def closure_encoder(z_grad):
            optimizer.zero_grad()
            embeddings.backward(z_grad.reshape_as(embeddings))

        optimizer.hpo_step(closure, closure_encoder)
    """

    def __init__(self, params, base_optimizer_cls, base_optimizer_params=None,
                 weights_optimizer_cls=None, weights_optimizer_params=None,
                 heads_groups=(1,), weights_names=None,
                 alpha=1.5, ema=0.9, encoder_decoder=False):
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

        if weights_optimizer_cls is None:
            weights_optimizer_cls = base_optimizer_cls
            weights_optimizer_params = base_optimizer_params
        self.weights_optimizer = weights_optimizer_cls(
            [self.param_groups[0]], **(weights_optimizer_params or {})
        )
        self.base_optimizer = base_optimizer_cls(
            self.param_groups[1:], **(base_optimizer_params or {})
        )
        self.param_groups = self.weights_optimizer.param_groups + self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        if 0 in heads_groups:
            raise ValueError("Group 0 is reserved for task weights.")
        self.heads_groups = list(sorted(set(heads_groups)))
        encoder_groups = set(range(len(self.param_groups))) - {0} - set(heads_groups)
        self.encoder_groups = list(sorted(encoder_groups))
        if not self.encoder_groups:
            raise ValueError(
                "GradNorm requires at least one shared encoder group (group index 2+)."
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

        self.alpha = alpha
        self.stats_momentum = ema
        self.encoder_decoder = encoder_decoder

        self._initial_losses = None
        self._n_updates = 0

        self._weights_tracker = StatsTracker("weights", ema, track_median=False)
        self._g_norms_tracker = StatsTracker("grad_norms", ema, track_median=False)

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
        if self._g_norms_tracker.last_value is not None:
            for key, val in self._g_norms_tracker.get().items():
                for name, c in zip(self.weights_names, val):
                    result[f"{key}_{name}"] = c
        return result

    def step(self, closure=None, *, inner=False):
        if not inner:
            raise ValueError("Use hpo_step() instead of step().")
        self.base_optimizer.step(closure=closure)
        self.weights_optimizer.step()

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
        """Make one GradNorm optimization step.

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
            Task weights (absolute values) used for this step, shape (n_tasks,).
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
            all_encoder_grads = []  # z.grad (encoder_decoder) or encoder param grads
            all_heads_grads = []
            losses_tensor = None

            weights_i = self.weights.new_zeros(self.n_weights)
            for i in range(self.n_weights):
                weights_i[i] = 1.0
                result = closure(0, weights_i, retain_graph=(i < self.n_weights - 1), stage=i)
                weights_i[i] = 0.0

                if self.encoder_decoder:
                    z, task_losses = result
                    if z is None or z.grad is None:
                        raise TypeError(
                            "In encoder_decoder mode, closure must return (z, losses) "
                            "with z.grad set after backward."
                        )
                    all_encoder_grads.append(z.grad.flatten().clone())
                else:
                    task_losses = result
                    all_encoder_grads.append(self._gather_grads("encoder").clone())

                all_heads_grads.append(self._gather_grads("heads").clone())
                losses_tensor = task_losses

            if not isinstance(losses_tensor, torch.Tensor):
                raise TypeError(
                    f"Closure must return a Tensor (got {type(losses_tensor).__name__})."
                )
            if losses_tensor.shape != (self.n_weights,):
                raise ValueError(
                    f"Closure returned losses of shape {tuple(losses_tensor.shape)}, "
                    f"expected ({self.n_weights},)."
                )
            losses_tensor = losses_tensor.detach().to(device=device)

            # Bootstrap initial losses on first step.
            if self._initial_losses is None:
                self._initial_losses = losses_tensor.abs().clamp(min=1e-8)
            if self._initial_losses.device != device:
                self._initial_losses = self._initial_losses.to(device)

            # Per-task gradient norms at encoder (or z proxy).
            g_norms = torch.stack([v.norm() for v in all_encoder_grads]).detach()

            # DDP: average scalar norms and losses across ranks so the GradNorm
            # weight update (w.grad) is identical everywhere. Full gradient vectors
            # are not all-reduced — same approximation as AlignedHPO.
            is_distributed = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
            if is_distributed:
                world_size = torch.distributed.get_world_size()
                flat = torch.cat([g_norms, losses_tensor])
                torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
                flat /= world_size
                g_norms = flat[:self.n_weights]
                losses_tensor = flat[self.n_weights:]

            w = self.weights
            w_abs = w.abs().detach()

            # Training rate ratio r̃_i = (L_i/L_i0) / mean_j(L_j/L_j0).
            L_hat = losses_tensor.abs() / self._initial_losses
            r_tilde = (L_hat / L_hat.mean().clamp(min=1e-8)).detach()

            # GradNorm weight gradient — only w in the autograd graph, cheap.
            with torch.enable_grad():
                G_W = w.abs() * g_norms
                G_tilde = (G_W.detach().mean() * r_tilde.pow(self.alpha)).detach()
                L_gradnorm = (G_W - G_tilde).abs().sum()
                w.grad = None
                L_gradnorm.backward()

            # Weighted encoder gradient: Σ w_i * g_i.
            all_encoder_grads_t = torch.stack(all_encoder_grads)  # (n_tasks, dim)
            encoder_grad = w_abs @ all_encoder_grads_t

            if self.encoder_decoder:
                # Pass aggregated z gradient through the encoder; protect w.grad.
                w_grad = w.grad
                w.grad = None
                closure_encoder(encoder_grad)
                w.grad = w_grad
            else:
                offset = 0
                for i in self.encoder_groups:
                    for p in self.param_groups[i]["params"]:
                        numel = p.numel()
                        p.grad = encoder_grad[offset:offset + numel].reshape(p.shape)
                        offset += numel

            # Weighted head gradient + downstream head gradient.
            all_heads_grads_t = torch.stack(all_heads_grads)  # (n_tasks, dim)
            heads_grad = w_abs @ all_heads_grads_t + heads_down_grads
            offset = 0
            for i in self.heads_groups:
                for p in self.param_groups[i]["params"]:
                    numel = p.numel()
                    p.grad = heads_grad[offset:offset + numel].reshape(p.shape)
                    offset += numel

            output_weights.copy_(w_abs)
            self._weights_tracker.update(w_abs.clone())
            self._g_norms_tracker.update(g_norms.clone())
            self._n_updates += 1

            if after_backward_hook is not None:
                after_backward_hook()

        self.step(inner_closure, inner=True)

        # Renormalize task weights so Σ|w_i| = n_tasks.
        with torch.no_grad():
            w_abs_sum = self.weights.abs().sum().clamp(min=1e-8)
            self.weights.data = self.weights.data.abs().mul_(self.n_weights / w_abs_sum)

        return output_weights

    def hpo_state_dict(self, add_names=False):
        state = {}
        state["weights_data"] = self.weights.data.clone()
        state["initial_losses"] = self._initial_losses
        state["n_updates"] = self._n_updates
        state["weights_tracker"] = self._weights_tracker.state_dict()
        state["g_norms_tracker"] = self._g_norms_tracker.state_dict()
        return state

    def state_dict(self):
        state = self.base_optimizer.state_dict()
        state["weights_optimizer"] = self.weights_optimizer.state_dict()
        state.update(self.hpo_state_dict())
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        weights_opt_state = state_dict.pop("weights_optimizer", None)
        weights_data = state_dict.pop("weights_data", None)
        self.base_optimizer.load_state_dict(state_dict)
        if weights_opt_state is not None:
            self.weights_optimizer.load_state_dict(weights_opt_state)
        self.param_groups = self.weights_optimizer.param_groups + self.base_optimizer.param_groups
        if weights_data is not None:
            self.weights.data.copy_(weights_data)
        self._initial_losses = state_dict.get("initial_losses")
        self._n_updates = state_dict.get("n_updates", 0)
        if "weights_tracker" in state_dict:
            self._weights_tracker.load_state_dict(state_dict["weights_tracker"])
        if "g_norms_tracker" in state_dict:
            self._g_norms_tracker.load_state_dict(state_dict["g_norms_tracker"])
