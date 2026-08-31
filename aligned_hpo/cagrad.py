import torch

from .aligned_hpo import HPO_STAGE_DOWNSTREAM
from .stats import StatsTracker


def _project_simplex(v):
    """Euclidean projection of v onto the probability simplex {w: Σw_i = 1, w_i ≥ 0}.

    Implements the sort-based algorithm from Duchi et al., "Efficient projections
    onto the l1-ball for learning in high dimensions" (2008).

    Args:
        v: Tensor with shape (n,).

    Returns:
        Projected tensor with shape (n,).
    """
    n = v.numel()
    u = torch.sort(v, descending=True).values
    cssv = torch.cumsum(u, 0) - 1.0
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = (u - cssv / ind) > 0
    # cond[0] is always True since u_0 - (u_0 - 1) = 1 > 0.
    rho = int(cond.nonzero()[-1]) if bool(cond.any()) else 0
    theta = cssv[rho] / (rho + 1)
    return (v - theta).clamp(min=0)


def _solve_cagrad_dual(G, phi_sqrt, max_iter=200, eps=1e-8, tol=1e-10):
    """Minimize the CAGrad dual objective over the probability simplex.

        min_{w: Σw_i = 1, w_i ≥ 0}  F(w) = g_w · g_0 + sqrt(φ) ||g_w||
                                         = w^T G u + sqrt(φ) sqrt(w^T G w)

    where u = 1/n (so g_0 = Σ u_i g_i is the average gradient) and
    G_ij = g_i · g_j. F is convex, so this is solved by projected gradient
    descent with a monotone backtracking line search.

    Args:
        G: (n, n) Gram matrix G_ij = g_i · g_j of per-task gradients.
        phi_sqrt: sqrt(φ) = c ||g_0||, the constraint radius.
        max_iter: Projected gradient iteration limit.
        eps: Numerical stabilizer inside square roots.
        tol: Convergence threshold on the projected step length.

    Returns:
        w of shape (n,) summing to 1 with non-negative entries.
    """
    n = G.shape[0]
    u = G.new_full((n,), 1.0 / n)
    Gu = G @ u  # (n,) — dot products of per-task gradients with g_0.

    def objective(w):
        return w @ Gu + phi_sqrt * ((w @ (G @ w)).clamp(min=0) + eps).sqrt()

    w = u.clone()
    # Scale the initial step by the curvature of the linear part.
    step = 1.0 / float(G.diagonal().max().clamp(min=eps))
    value = objective(w)
    for _ in range(max_iter):
        Gw = G @ w
        gw_norm = ((w @ Gw).clamp(min=0) + eps).sqrt()
        grad = Gu + phi_sqrt * Gw / gw_norm
        for _ in range(30):
            w_new = _project_simplex(w - step * grad)
            value_new = objective(w_new)
            if value_new <= value:
                break
            step *= 0.5
        else:
            # No decrease found even for a tiny step: w is (numerically) optimal.
            break
        delta = (w_new - w).norm()
        w, value = w_new, value_new
        if delta < tol:
            break
        step *= 1.5  # Line search shrinks it back if this overshoots.
    return w


def _solve_cagrad(G, c=0.5, rescale=1, max_iter=200, eps=1e-8):
    """Compute CAGrad task coefficients from the Gram matrix of per-task gradients.

    Solves the dual problem for w and returns the coefficients of the update
    direction expressed in the per-task gradient basis:

        d = g_0 + (sqrt(φ) / ||g_w||) g_w = Σ_i (1/n + λ w_i) g_i,
        λ = sqrt(φ) / ||g_w||,  φ = c² ||g_0||².

    Args:
        G: (n, n) Gram matrix G_ij = g_i · g_j of per-task gradients.
        c: CAGrad hyperparameter (called alpha in the reference implementation),
            the constraint radius as a fraction of ||g_0||.
        rescale: Output scaling, following the reference implementation:
            0 — none, 1 — divide by (1 + c²), 2 — divide by (1 + c).
        max_iter: Projected gradient iteration limit for the dual problem.
        eps: Numerical stabilizer inside square roots.

    Returns:
        Non-negative coefficients of shape (n,). Unlike MGDA weights, they do
        not sum to one (they sum to (1 + λ) before rescaling).
    """
    n = G.shape[0]
    g0_norm = (G.mean().clamp(min=0) + eps).sqrt()  # ||g_0||, since ||g_0||² = mean(G).
    phi_sqrt = c * g0_norm
    w = _solve_cagrad_dual(G, phi_sqrt, max_iter=max_iter, eps=eps)
    gw_norm = ((w @ (G @ w)).clamp(min=0) + eps).sqrt()
    lmbda = phi_sqrt / gw_norm
    coef = 1.0 / n + lmbda * w
    if rescale == 1:
        coef = coef / (1 + c ** 2)
    elif rescale == 2:
        coef = coef / (1 + c)
    return coef


class CAGradOptimizer(torch.optim.Optimizer):
    """CAGrad: Conflict-Averse Gradient Descent (Liu et al., 2021).

    Maximizes the worst-case per-task improvement in a ball around the average
    gradient g_0 = (1/n) Σ g_i:

        max_d min_i  g_i · d    s.t.  ||d - g_0|| ≤ c ||g_0||

    The dual problem is convex on the probability simplex,

        min_{w: Σw_i = 1, w_i ≥ 0}  g_w · g_0 + sqrt(φ) ||g_w||,   φ = c² ||g_0||²,

    and is solved by projected gradient descent on the Gram matrix
    G_ij = g_i · g_j. The resulting update direction

        d = g_0 + (sqrt(φ) / ||g_w||) g_w = Σ_i coef_i g_i

    is applied to the shared encoder. The coefficients are also applied to the
    per-task head gradients (as in MGDAOptimizer) and stored in group 0. They
    are non-negative but do not sum to one.

    Args:
        params: Parameter groups:
            Group 0: Task weights — a single 1D tensor (storage for computed coefficients).
            Group 1: Task-specific head parameters (indices given by heads_groups).
            Group 2+: Shared encoder parameters (all remaining groups).
        base_optimizer_cls: Optimizer class for model parameters.
        base_optimizer_params: Keyword arguments for the base optimizer.
        heads_groups: Indices of task-head parameter groups. Default: (1,).
        shared_groups: Indices of parameter groups, related to shared loss heads parameters.
            In this method, shared groups are merged into heads groups.
        weights_names: Optional names for task weights (for logging).
        c: CAGrad hyperparameter (called alpha in the reference implementation),
            the constraint radius as a fraction of ||g_0||. Paper default: 0.5.
        rescale: Output scaling, following the reference implementation:
            0 — none, 1 — divide by (1 + c²), 2 — divide by (1 + c). Default: 1.
        max_iter: Projected gradient iteration limit for the dual problem. Default: 200.
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

    DDP note: the Gram matrix is averaged across ranks, so all ranks solve the same
    dual problem, but the aggregated gradient itself is local (same approximation
    as the other HPO optimizers — full gradient vectors are not all-reduced).
    """

    def __init__(self, params, base_optimizer_cls, base_optimizer_params=None,
                 heads_groups=(1,), shared_groups=(), weights_names=None,
                 c=0.5, rescale=1, max_iter=200, ema=0.9, encoder_decoder=False):
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
                "CAGradOptimizer requires at least one shared encoder group (group index 2+)."
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

        if c < 0:
            raise ValueError(f"CAGrad c must be non-negative, got {c}.")
        if rescale not in {0, 1, 2}:
            raise ValueError(f"CAGrad rescale must be 0, 1 or 2, got {rescale}.")
        self.c = c
        self.rescale = rescale
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
        """Make one CAGrad optimization step.

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
            Task coefficients (non-negative) used for this step, shape (n_tasks,).
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

            # Build Gram matrix and solve the CAGrad dual problem.
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

            coef = _solve_cagrad(
                G, c=self.c, rescale=self.rescale, max_iter=self.max_iter
            ).to(device=device)

            # Per-task gradient norms for logging (before applying coefficients).
            g_norms = torch.stack([v.norm() for v in all_encoder_grads]).detach()

            # Conflict-averse encoder gradient: Σ coef_i g_i.
            encoder_grad = coef @ grads_stack  # (dim,)

            # Weighted head gradient + downstream head gradient.
            all_heads_grads_t = torch.stack(all_heads_grads)  # (n_tasks, dim)
            heads_grad = coef @ all_heads_grads_t + heads_down_grads  # computed before zero_grad

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

            self.weights.data.copy_(coef)
            output_weights.copy_(coef)

            self._weights_tracker.update(coef.clone())
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
