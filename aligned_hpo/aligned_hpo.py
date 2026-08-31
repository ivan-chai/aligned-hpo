import itertools
import torch
import warnings
from contextlib import contextmanager
from numbers import Number

from .amtl import procrustes
from .gradient import GradientNormalizer
from .solvers import solve_qcqp
from .stats import StatsTracker


HPO_STAGE_DOWNSTREAM = "downstream"


class ZeroWeightsException(Exception):
    pass


class AlignedHPOptimizer(torch.optim.Optimizer):
    """Aligned Hyperparameter Optimizer.

    Args:
        params: Model parameters with 3 or more groups. See parameter groups note below.
        base_optimizer_cls: The optimizer to use.
        base_optimizer_params: Parameters of the base optimizer.
        weights_optimizer_cls: The optimizer to use for weights. By default equal to base_optimizer_cls.
        weights_optimizer_params: Parameters of the weights optimizer.
        heads_groups: Indices of parameter groups, related to individual loss heads.
        shared_groups: Indices of parameter groups, related to shared loss heads parameters.
        weights_names: An optional list of names for hyperparameters (for logging).
        weights_parametrization: Either "linear" or "abs".
        weights_normalization: Either "gradnorm" (default), "sum", or "none".
        encoder_decoder: Whether to use encoder-decoder decomposition and upper bound optimization for fast hyperparameter tuning.
            This flag affects an intereface of the provided closure. See notes below.
        ema: Exponential smoothing factor for logging statistics.
        algorithm: Either "sgd" or "none" to disable HPO.
        apply_optimizer_correction: Try to approximate an actual optimizer step rather than simple SGD.
        scale_gradients: A fixed scale for gradients.
        eps: Roughly the square root of the minimum gradients correlation value.

    NOTE. Algorithm.
    1. Compute downstream gradients, apply runnig normalization.
    2. Compute per-loss gradients, apply running normalization.
    3. Compute weights update (without actual update).
    4. Apply gradients, normalized at step 2 with current weights.
    5. Update the weights.

    NOTE. Parameter groups.
    There are 4 or more parameter groups:
    0. Loss weights.
    1. Individual loss heads.
    ?. Optional shared head.
    ... Encoder parts.

    NOTE. Encoder-Decoder vs full gradients.
    In a simple "full" approach, closure must compute gradients for all model parameters, leading to multiple backward passes at each step.
    A more effective method decomposes the model into encoder and decoder part. A closure must be able to compute gradients w.r.t. to the
    embedding output of the encoder. A separate step is performed to pass aggregated gradient to the encoder part of the model.
    See examples below.

    Example usage (full gradients):
    ```
    optimizer = AlignedHPOptimizer([{"params": [weights]},  # Weights for tuning.
                                    {"params": heads.parameters()},  # Loss heads parameters.
                                    {"params": model.parameters()}],  # Other model parameters.
                                   torch.optim.Adam,
                                   {"lr": 0.01})  # Optimizer parameters.

    output = model(x)
    down_loss, loss1, loss2 = criterion(output, y)

    def closure(down_weight, weights, retain_graph=False, stage=None):
        optimizer.zero_grad()
        loss = down_weight * down_loss + weights[0] * loss1 + weights[1] * loss2
        loss.backward(retain_graph=retain_graph)

    optimizer.hpo_step(closure)
    ```

    Example usage (encoder-decoder):
    ```
    optimizer = AlignedHPOptimizer([{"params": [weights]},  # Weights for tuning.
                                    {"params": heads.parameters()},  # Loss heads parameters.
                                    {"params": model.encoder.parameters()}],  # Encoder.
                                   torch.optim.Adam,
                                   {"lr": 0.01})  # Optimizer parameters.

    embeddings = model.encode(x)  # Apply shared encoder.
    z = embeddings.detach().clone()
    z.requires_grad = True
    output = model.decode(z)  # Appry individual heads.
    down_loss, loss1, loss2 = criterion(output, y)

    def closure(down_weight, weights, retain_graph=False, stage=None):
        optimizer.zero_grad()
        z.grad = None  # New.
        loss = down_weight * down_loss + weights[0] * loss1 + weights[1] * loss2
        loss.backward(retain_graph=retain_graph)
        return z  # New, return embeddings with gradients.

    # New, backward aggregated encoder gradients.
    def closure_encoder(z_grad):
        optimizer.zero_grad()
        embeddings.backward(z_grad)

    optimizer.hpo_step(closure, closure_encoder)
    ```
    """
    def __init__(self, params, base_optimizer_cls, base_optimizer_params=None,
                 weights_optimizer_cls=None, weights_optimizer_params=None,
                 heads_groups=(1,), shared_groups=(), weights_names=None,
                 weights_parametrization="abs", weights_normalization="gradnorm",
                 encoder_decoder=False, ema=0.9, algorithm="sgd",
                 apply_optimizer_correction=False, scale_gradients=1, eps=1e-8):
        params = list(params)
        if len(params) < 3 or not isinstance(params[0], dict) or not isinstance(params[1], dict):
            raise ValueError("Expected at least three param groups with the first group being hyperparameters weights, the second group being projection heads weights, and the third group being encoder weights.")
        if (len(params[0]["params"]) != 1) or (params[0]["params"][0].ndim != 1):
            raise ValueError("Weights must be flat.")
        if algorithm not in {"sgd", "none"}:
            raise ValueError(f"Unexpected algorithm: {algorithm}")
        if weights_parametrization not in ["linear", "abs"]:
            raise ValueError(f"Unknown weights parametrization method: {weights_parametrization}")
        if weights_normalization not in ["gradnorm", "sum", "none"]:
            raise ValueError(f"Unknown weights normalization method: {weights_normalization}")
        # Sub-optimizers fill their own defaults into the groups they own, so the outer
        # optimizer must not pre-populate them with the base optimizer hyperparameters.
        super().__init__(params, {})
        if weights_optimizer_cls is None:
            weights_optimizer_cls = base_optimizer_cls
            if weights_optimizer_params is None:
                weights_optimizer_params = base_optimizer_params

        if 0 in heads_groups:
            raise ValueError("The first group (index 0) is reserved for weights and can not relate to individual heads.")
        if 0 in shared_groups:
            raise ValueError("The first group (index 0) is reserved for weights and can not relate to shared heads.")
        if set(heads_groups) & set(shared_groups):
            raise ValueError("Groups intersection.")
        n_groups = len(self.param_groups)
        for i in itertools.chain(heads_groups, shared_groups):
            if not (0 <= i < n_groups):
                raise ValueError(f"Group index {i} is out of range for {n_groups} parameter groups.")
        self.heads_groups = list(sorted(set(heads_groups)))
        self.shared_groups = list(sorted(set(shared_groups)))
        encoder_groups = set(range(n_groups)) - {0} - set(heads_groups) - set(shared_groups)
        self.encoder_groups = list(sorted(encoder_groups))

        groups = list(self.param_groups)
        base_optimizer_groups = list(range(1, n_groups))
        self.weights_optimizer = weights_optimizer_cls([groups[0]], **(weights_optimizer_params or {}))
        self.base_optimizer = base_optimizer_cls([groups[i] for i in base_optimizer_groups],
                                                 **(base_optimizer_params or {}))
        # Group indices (heads_groups, shared_groups, encoder_groups) address self.param_groups,
        # so the original group order must be restored after the split.
        self._group_owners = [None] * n_groups
        self._group_owners[0] = ("weights", 0)
        for position, i in enumerate(base_optimizer_groups):
            self._group_owners[i] = ("base", position)
        self._sync_param_groups()
        self.defaults.update(self.base_optimizer.defaults)

        self.n_weights = len(self.logits)
        if weights_names is None:
            weights_names = [str(i) for i in range(self.n_weights)]
        elif len(weights_names) != self.n_weights:
            raise ValueError("Names and weights lengths mismatch")
        self.weights_names = weights_names
        if self.n_weights == 0:
            raise ValueError("Empty hyperparameters list.")
        self.weights_parametrization = weights_parametrization
        self.weights_normalization = weights_normalization

        self.encoder_decoder = encoder_decoder
        self.algorithm = algorithm
        self.eps = eps

        self.ema = ema
        self._normalizers_trackers = {name: StatsTracker(f"grad_norm_{name}", self.ema, track_median=True)
                                      for name in self.weights_names}

        self.apply_optimizer_correction = apply_optimizer_correction
        self.scale_gradients = scale_gradients

        # TODO: use optimizer state for gradient caches.
        self._running_stats = {"covs": None, "products": None, "weights": None}
        if encoder_decoder:
            self._running_stats["encoder_transmission"] = None
        self._buffers = {
            "n_updates": 0
        }

        self._weights_tracker = StatsTracker("weights", self.ema)
        self._effective_weights_tracker = StatsTracker("effective_weights", self.ema)  # Weights scaled by grad norm.
        self._heads_grad_norm_tracker = StatsTracker("heads_grad_norm", self.ema, track_median=False)
        self._shared_grad_norm_tracker = StatsTracker("shared_grad_norm", self.ema, track_median=False)
        self._encoder_grad_norm_tracker = StatsTracker("encoder_grad_norm", self.ema, track_median=False)
        self._correlations_tracker = StatsTracker("grad_correlations", self.ema, track_median=False)

    def _sync_param_groups(self):
        """Rebuild self.param_groups from the sub-optimizers, keeping the original group order."""
        sources = {
            "weights": self.weights_optimizer.param_groups,
            "base": self.base_optimizer.param_groups,
        }
        self.param_groups = [sources[owner][position] for owner, position in self._group_owners]

    def _group_optimizer(self, index):
        """Get the sub-optimizer which fits the given parameter group."""
        owner, _ = self._group_owners[index]
        if owner == "weights":
            return self.weights_optimizer
        assert owner == "base"
        return self.base_optimizer

    def _group_indices(self, part):
        """Get indices of the parameter groups for the given model part.

        Args:
            part: Part of the model (`all`, `heads`, `shared`, or `encoder`).
        """
        if part == "all":
            return list(range(1, len(self.param_groups)))
        elif part == "heads":
            return list(self.heads_groups)
        elif part == "shared":
            return list(self.shared_groups)
        else:
            assert part == "encoder"
            # All except hyperparameters, individual heads, and shared heads.
            return list(self.encoder_groups)

    @property
    def need_losses(self):
        return False

    @property
    def use_validation(self):
        return False

    @property
    def logits(self):
        return self.param_groups[0]["params"][0]

    @property
    def metrics(self):
        result = {}
        for name, c in zip(self.weights_names, self.logits):
            result[f"logits_{name}"] = c
        if self._correlations_tracker.last_value is not None:
            for key, val in self._correlations_tracker.get().items():
                for wname, c in zip(self.weights_names, val):
                    result[f"{key}_{wname}"] = c
        if self._weights_tracker.last_value is not None:
            for key, val in self._weights_tracker.get().items():
                for wname, c in zip(self.weights_names, val):
                    result[f"{key}_{wname}"] = c
        if self._effective_weights_tracker.last_value is not None:
            for key, val in self._effective_weights_tracker.get().items():
                for wname, c in zip(self.weights_names, val):
                    result[f"{key}_{wname}"] = c
        if self._heads_grad_norm_tracker.last_value is not None:
            result.update(self._heads_grad_norm_tracker.get())
        if self._shared_grad_norm_tracker.last_value is not None:
            result.update(self._shared_grad_norm_tracker.get())
        if self._encoder_grad_norm_tracker.last_value is not None:
            result.update(self._encoder_grad_norm_tracker.get())
        for name, normalizer in self._normalizers_trackers.items():
            if normalizer.n_updates > 0:
                result.update(normalizer.get())
        if self.encoder_decoder and (self._running_stats["encoder_transmission"] is not None):
            result["encoder_transmission"] = self._running_stats["encoder_transmission"]
        return result

    def step(self, closure=None, *, inner=False):
        if not inner:
            raise ValueError("Please, use 'hpo_step' function.")
        self.base_optimizer.step(closure=closure)
        self.weights_optimizer.step()

    def _unnormalized_weights(self):
        if self.weights_parametrization == "abs":
            weights = torch.where(self.logits >= 0, self.logits, -self.logits)  # Abs with positive grad at zero.
        elif self.weights_parametrization == "linear":
            weights = self.logits
        else:
            raise RuntimeError(f"Unknown parametrization: {self.weights_parametrization}")
        return weights

    def _normalize_weights(self, weights, pretrain_covariances):
        if torch.linalg.norm(weights) < self.eps ** 2:
            return weights
        if self.weights_normalization == "gradnorm":
            pretrain_covariances = pretrain_covariances.detach()
            weights_grad_norm = (weights[None] @ pretrain_covariances @ weights).sqrt()
            weights = weights / weights_grad_norm.clamp(min=self.eps ** 2)
        elif self.weights_normalization == "sum":
            weights = weights / weights.sum()
        elif self.weights_normalization != "none":
            raise ValueError(f"Unknown weights normalization: {self.weights_normalization}")
        return weights

    def _update_running_stats(self, value, stage=None):
        if stage not in self._running_stats:
            raise ValueError(f"Unknown stage: {stage}")
        if stage == "weights":
            momentum = 0
        elif stage in {"covs", "products", "encoder_grads"}:
            momentum = 0
        elif stage in {"encoder_transmission"}:
            momentum = self.ema
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if self._running_stats[stage] is None:
            self._running_stats[stage] = value
        else:
            if momentum > 0:
                value = self._running_stats[stage] * momentum + value * (1 - momentum)
            self._running_stats[stage] = value
        return self._running_stats[stage]

    @torch.no_grad()
    def _get_loss_grads(self, closure):
        """Update weights (gradient or value) and return cached gradients."""
        loss_weights = torch.zeros_like(self.logits)

        # Below:
        # - heads grads: individual heads weights gradients.
        # - z grads: gradient w.r.t. encoder output.
        # - grads: HPO grads after EMA smoothing used for weights tuning.

        # Compute downstream grads.
        downstream_weight = 1
        z_down = closure(downstream_weight, loss_weights, retain_graph=True, stage=HPO_STAGE_DOWNSTREAM)
        if self.encoder_decoder and (z_down is None or z_down.grad is None):
            raise TypeError("In the encoder-decoder mode, closure must return embedding with gradient.")
        if self.encoder_decoder:
            z_down_grads = z_down.grad.flatten()
            z_down = z_down.clone()
        else:
            encoder_down_grads = self._gather_grads("encoder")
        heads_down_grads = self._gather_grads("heads")

        # Caches for normalization differentiation.
        all_z_grads = []
        all_heads_grads = []
        all_shared_grads = []
        all_encoder_grads = []

        # Compute main losses grads.
        downstream_weight = 0
        for i, name in enumerate(self.weights_names):
            loss_weights[i] = 1
            z = closure(downstream_weight, loss_weights, retain_graph=(i < self.n_weights - 1), stage=i)
            loss_weights[i] = 0
            if self.encoder_decoder and (z is None or z.grad is None):
                raise TypeError("In the encoder-decoder mode, closure must return gradient w.r.t. encoder output.")
            heads_grads = self._gather_grads("heads")
            shared_grads = self._gather_grads("shared")
            if self.encoder_decoder:
                z_grad = z.grad.flatten().clone()
                all_z_grads.append(z_grad)
                grad_norm = torch.linalg.norm(z_grad)
            else:
                encoder_grads = self._gather_grads("encoder")
                all_encoder_grads.append(encoder_grads)
                grad_norm = torch.linalg.norm(encoder_grads)
            self._normalizers_trackers[name].update(grad_norm)
            all_heads_grads.append(heads_grads)
            all_shared_grads.append(shared_grads)

        return {
            "heads_down_grads": heads_down_grads,
            "encoder_down_grads": encoder_down_grads if not self.encoder_decoder else None,
            "z_down": z_down if self.encoder_decoder else None,
            "z_down_grads": z_down_grads if self.encoder_decoder else None,
            "all_heads_grads": all_heads_grads,
            "all_shared_grads": all_shared_grads,
            "all_encoder_grads": all_encoder_grads if not self.encoder_decoder else None,
            "all_z_grads": all_z_grads if self.encoder_decoder else None
        }

    @torch.no_grad()
    def _compute_weights_and_gradients(self, all_grads_covs, products):
        algorithm = self.algorithm

        if "sgd" in algorithm:
            with torch.enable_grad():
                weights = self._normalize_weights(self._unnormalized_weights(), pretrain_covariances=all_grads_covs)
            # Normalize products before backward to avoid tiny gradient magnitudes
            # propagating through the normalization graph when gradients are small.
            self.logits.grad = None
            weights.backward(-products)
            return weights, self.logits.grad.clone()
        else:
            assert "none" in algorithm
            weights = self._unnormalized_weights()
            return weights, None

    @torch.no_grad()
    def _tune_weights(self, closure, embed_fn=None):
        """Update weights (gradient or value) and return cached gradients."""
        grads = self._get_loss_grads(closure)

        if self.encoder_decoder:
            down_grads = grads["z_down_grads"]
        else:
            down_grads = grads["encoder_down_grads"]
            if self.apply_optimizer_correction:
                down_grads = down_grads.clone()
                self.apply_optimizer_correction_("encoder", down_grads)

        # Caches for normalization differentiation.
        all_grads = []

        # Compute main losses grads.
        for i in range(self.n_weights):
            if self.encoder_decoder:
                loss_grads = grads["all_z_grads"][i]
            else:
                loss_grads = grads["all_encoder_grads"][i]
                if self.apply_optimizer_correction:
                    loss_grads = loss_grads.clone()
                    self.apply_optimizer_correction_("encoder", loss_grads)
            all_grads.append(loss_grads)

        all_grads = torch.stack(all_grads, 0)  # (W, P).
        all_grads_covs = all_grads @ all_grads.T
        products = all_grads @ down_grads  # (W).
        del all_grads
        del down_grads

        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized() and (torch.distributed.get_world_size() > 1)
        if is_distributed:
            # Merge two all_reduces into one by concatenating tensors.
            world_size = torch.distributed.get_world_size()
            flat = torch.cat([all_grads_covs.flatten(), products])
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
            covs_numel = all_grads_covs.numel()
            all_grads_covs.copy_(flat[:covs_numel].view_as(all_grads_covs) / world_size)
            products.copy_(flat[covs_numel:] / world_size)

        all_grads_covs = self._update_running_stats(all_grads_covs, "covs")
        products = self._update_running_stats(products, "products")
        grads["all_grads_covs"] = all_grads_covs

        weights, logits_grads = self._compute_weights_and_gradients(all_grads_covs, products)

        weights = self._update_running_stats(weights, stage="weights")

        self.logits.grad = logits_grads
        if (logits_grads is None) and ("none" not in self.algorithm):
            self.logits.copy_(weights)

        self._correlations_tracker.update(products.detach().clone())
        self._weights_tracker.update(weights.detach().clone())

        moving_norms = []
        for name in self.weights_names:
            normalizer = self._normalizers_trackers[name]
            if not normalizer.n_updates:
                break
            moving_norms.append(normalizer.ema_value)
        else:
            moving_norms = torch.stack(moving_norms).clamp(min=self.eps)
            effective_weights = weights.detach() * moving_norms
            self._effective_weights_tracker.update(effective_weights.clone())

        self._buffers["n_updates"] += 1
        return grads

    @torch.no_grad()
    def val_step(self, closure, closure_encoder=None, embed_fn=None, after_backward_hook=None):
        """Make a single step on a validation set to cache downstream grads and (optionally) update downstream head.

        Args:
            closure: A closure to compute inidivdual gradients.
            after_backward_hook: A function to call after gradients are estimated (gradient clipping etc.).

        Returns:
            Weights used in current step.

        The closure is used like this: closure(target_loss_weight, *loss_weights, retain_graph=False, stage=None).
        The closure_encoder is used like this: closure(encoder_output_grad).

        Each closure must zero grads and compute gradients.
        """
        if not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0):
            warnings.warn("Calling val_step, when align is `train`.")

    @torch.no_grad()
    def hpo_step(self, closure, closure_encoder=None, embed_fn=None, after_backward_hook=None):
        """Make a single step.

        Args:
            closure: A closure to compute inidivdual gradients.
            closure_encoder: A closure to pass embedding gradients to the encoder.
            after_backward_hook: A function to call after gradients are estimated (gradient clipping etc.).

        Returns:
            Weights used in current step.

        The closure is used like this: closure(target_loss_weight, *loss_weights, retain_graph=False, stage=None).
        The closure_encoder is used like this: closure(encoder_output_grad).

        Each closure must zero grads and compute gradients.
        """
        closure = torch.enable_grad()(closure)  # The closure should do a full forward-backward pass.
        if self.encoder_decoder:
            if closure_encoder is None:
                raise ValueError("Need encoder closure.")
            closure_encoder = torch.enable_grad()(closure_encoder)  # The closure should do a full forward-backward pass.

        output_weights = torch.empty_like(self.logits)

        @torch.no_grad()
        def inner_closure():
            grads = self._tune_weights(closure, embed_fn=embed_fn)
            weights = self._running_stats["weights"]

            if weights is None:
                raise RuntimeError("Validation batch must be consumed first, when align is val")

            # Cache returned value.
            output_weights.copy_(weights)

            # Set gradients for the encoder model weights.
            encoder_grads_scale = 1
            if self.encoder_decoder:
                # Backprop with z grads. Keep logits grad intact.
                encoder_grads_scale *= (self._running_stats["encoder_transmission"] or 1)
                scale = encoder_grads_scale * self.scale_gradients
                z_grad = (scale * weights) @ torch.stack(grads["all_z_grads"])
                z_grad_norm = torch.linalg.norm(z_grad)
                logits_grad = self.logits.grad
                self.logits.grad = None
                closure_encoder(z_grad)
                self.logits.grad = logits_grad
                del z_grad
                encoder_grad_norm = torch.linalg.norm(self._gather_grads("encoder"))
                self._update_running_stats(z_grad_norm / encoder_grad_norm.clamp(min=self.eps ** 2), stage="encoder_transmission")
                self._encoder_grad_norm_tracker.update(encoder_grad_norm)
            else:
                encoder_grad = weights @ torch.stack(grads["all_encoder_grads"])
                encoder_grad *= self.scale_gradients * encoder_grads_scale
                self._encoder_grad_norm_tracker.update(torch.linalg.norm(encoder_grad))
                param_groups = [self.param_groups[i] for i in self.encoder_groups]
                downstream_weight = self.encoder_downstream_weight * self.scale_gradients * encoder_grads_scale
                offset = 0
                for i, group in enumerate(param_groups):
                    for p in group["params"]:
                        numel = p.numel()
                        p.grad = encoder_grad[offset:offset + numel].reshape(p.shape)
                        if downstream_weight > 0:
                            p.grad += downstream_weight * encoder_down_grads[offset:offset + numel].reshape(p.shape)
                        offset += numel
                assert offset == len(encoder_grad)
                del encoder_grad

            # Set grads for individual heads model.
            heads_grad = torch.stack(grads["all_heads_grads"]).sum(0)
            heads_grad.add_(grads["heads_down_grads"])
            if self.scale_gradients != 1:
                heads_grad *= self.scale_gradients
            self._heads_grad_norm_tracker.update(torch.linalg.norm(heads_grad))
            offset = 0
            for i in self.heads_groups:
                for p in self.param_groups[i]["params"]:
                    numel = p.numel()
                    p.grad = heads_grad[offset:offset + numel].reshape(p.shape)
                    offset += numel
            assert offset == len(heads_grad)
            del heads_grad

            # Set grads for shared heads.
            shared_grad = torch.stack(grads["all_shared_grads"]).sum(0)
            if self.scale_gradients != 1:
                shared_grad *= self.scale_gradients
            self._shared_grad_norm_tracker.update(torch.linalg.norm(shared_grad))
            offset = 0
            for i in self.shared_groups:
                for p in self.param_groups[i]["params"]:
                    numel = p.numel()
                    p.grad = shared_grad[offset:offset + numel].reshape(p.shape)
                    offset += numel
            assert offset == len(shared_grad)
            del shared_grad

            if after_backward_hook is not None:
                after_backward_hook()
        try:
            self.step(inner_closure, inner=True)
        except ZeroWeightsException:
            pass
        return output_weights

    def hpo_state_dict(self, add_names=False):
        state = {}
        if add_names:
            state["weights_names"] = self.weights_names
        state["running_stats"] = dict(self._running_stats)
        state["buffers"] = dict(self._buffers)
        state["normalizers_trackers"] = {k: v.state_dict() for k, v in self._normalizers_trackers.items()}
        state["weights_tracker"] = self._weights_tracker.state_dict()
        state["effective_weights_tracker"] = self._effective_weights_tracker.state_dict()
        state["heads_grad_norm_tracker"] = self._heads_grad_norm_tracker.state_dict()
        state["shared_grad_norm_tracker"] = self._shared_grad_norm_tracker.state_dict()
        state["encoder_grad_norm_tracker"] = self._encoder_grad_norm_tracker.state_dict()
        state["correlations_tracker"] = self._correlations_tracker.state_dict()
        return state

    def state_dict(self):
        state = self.base_optimizer.state_dict()
        state["weights_optimizer"] = self.weights_optimizer.state_dict()
        state.update(self.hpo_state_dict())
        return state

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)  # shallow copy — don't mutate caller's dict
        weights_opt_state = state_dict.pop("weights_optimizer", None)
        self.base_optimizer.load_state_dict(state_dict)
        if weights_opt_state is not None:
            self.weights_optimizer.load_state_dict(weights_opt_state)
        self._sync_param_groups()
        p = self.logits
        self._running_stats.update({k: (v.to(device=p.device, dtype=p.dtype) if v is not None else None)
                                    for k, v in state_dict.get("running_stats", {}).items()})
        self._buffers.update({k: (v.to(device=p.device, dtype=p.dtype) if isinstance(v, torch.Tensor) else v)
                              for k, v in state_dict.get("buffers", {}).items()})
        for k, v in state_dict["normalizers_trackers"].items():
            self._normalizers_trackers[k].load_state_dict(v)
        if "weights_tracker" in state_dict:
            self._weights_tracker.load_state_dict(state_dict["weights_tracker"])
        if "effective_weights_tracker" in state_dict:
            self._effective_weights_tracker.load_state_dict(state_dict["effective_weights_tracker"])
        if "heads_grad_norm_tracker" in state_dict:
            self._heads_grad_norm_tracker.load_state_dict(state_dict["heads_grad_norm_tracker"])
        if "shared_grad_norm_tracker" in state_dict:
            self._shared_grad_norm_tracker.load_state_dict(state_dict["shared_grad_norm_tracker"])
        if "encoder_grad_norm_tracker" in state_dict:
            self._encoder_grad_norm_tracker.load_state_dict(state_dict["encoder_grad_norm_tracker"])
        self._correlations_tracker.load_state_dict(state_dict["correlations_tracker"])

    @torch.no_grad()
    def _gather_grads(self, part):
        """Get gradients vector.

        Model parts:
            heads: Individual losses heads.
            encoder: Encoder part (before embedding) in the encoder-decoder model.

        Args:
            part: Part of the model to extract gradients for (`all`, `heads`, `shared`, or `encoder`).
        """
        grads = []
        for group in [self.param_groups[i] for i in self._group_indices(part)]:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p).flatten())
                else:
                    grads.append(p.grad.flatten())
        if not grads:
            return torch.zeros_like(self.logits[:0])
        return torch.cat(grads)

    def apply_optimizer_correction_(self, part, grads):
        offset = 0
        for i in self._group_indices(part):
            group = self.param_groups[i]
            optimizer = self._group_optimizer(i)  # Groups can be fit by different optimizers.
            if isinstance(optimizer, torch.optim.Adam):
                _, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    state = optimizer.state[p]
                    exp_avg_sq = state.get("exp_avg_sq", None)
                    if exp_avg_sq is None:
                        offset += p.numel()
                        continue
                    exp_avg_sq = exp_avg_sq.flatten() * beta2 + grads[offset:offset + p.numel()].square() * (1 - beta2)
                    step = state["step"]
                    bias_correction2_sqrt = (1 - beta2 ** step) ** 0.5
                    grads[offset:offset + p.numel()] /= exp_avg_sq.sqrt() / bias_correction2_sqrt + eps
                    offset += p.numel()
            elif isinstance(optimizer, torch.optim.SGD):
                offset += sum(p.numel() for p in group["params"])  # No need for correction.
            else:
                raise NotImplementedError(f"Can't apply correction to {type(optimizer).__name__}")
        assert offset == len(grads)
