import itertools
import math
import numpy as np
import re
import scipy.optimize
import torch
import warnings
from contextlib import contextmanager
from copy import deepcopy
from numbers import Number

from .gradient import GradientNormalizer
from .solvers import solve_qp, solve_qcqp
from .stats import StatsTracker


HPO_STAGE_DOWNSTREAM = "downstream"


DEFAULT_EMA_COVS = 0.9
DEFAULT_EMA_WEIGHTS = 0
DEFAULT_EMA_STATS = 0.9
DEFAULT_WARMUP = 10


class ZeroWeightsException(Exception):
    pass


class AlignedHPOptimizer(torch.optim.Optimizer):
    """Aligned Hyperparameter Optimizer.

    Args:
        params: Model parameters with 3 or more groups. See parameter groups note below.
        base_optimizer_cls: The optimizer to use.
        base_optimizer_params: Parameters of the base optimizer.
        heads_groups: Indices of parameter groups, related to individual loss heads.
        weights_names: An optional list of names for hyperparameters (for logging).
        weights_parametrization: Either "linear" or "abs".
        encoder_downstream_weight: The weight of the downstream gradient in encoder optimization. Default is 0 (disable).
        downstream_merge: Fill zero values in encoder gradient with downsrtream gradient.
        encoder_decoder: Whether to use encoder-decoder decomposition and upper bound optimization for fast hyperparameter tuning.
            This flag affects an intereface of the provided closure. See notes below.
        algorithm: Either "sgd", "warmup-sgd", "mse", or "none" to disable HPO.
        ema: Use momentum for smoothing statistics. Can be a dictionary with "covs", "weights", and "stats" keys. See notes below.
        warmup: Average the specified number of initial observation instead of EMA.
        align: Either `train` to tune weights on the train set only, `val` to tune weights on validation, or `train-val` to align training gradients with validation downstream grad.
        train_downstream_head: Which dataset use to train the downstream head. Either "train", "val", or "train-val".
        regularization: Regularization weight for the SGD algorithm. Logits are kept close to 1.
        apply_optimizer_correction: Try to approximate an actual optimizer step rather than simple SGD.
        scale_gradients: A fixed scale for gradients.
        skip_step_zero_weights_limit: Skip optimizer step, when weights are zero. When the limit reached, continue training with equal weights.
        z_grad_lr: The encoder step size, when encoder-decoder is used with `train-val` alignment.
        maxiters: The maxmum number of iterations in the closed-form solver.
        eps: Roughly the square root of the minimum gradients correlation value.

    NOTE. Algorithm.
    1. Compute downstream gradients, normalize them.
    2. Compute per-loss gradients, apply running normalization.
    3. Compute weights as if gradient norms were equal to ones.
    4. Apply gradients, normalized at step 2 with compute weights.

    NOTE. Parameter groups.

    There are 4 or more parameter groups:
    0. Loss weights.
    1. Individual loss heads.
    2+. Encoder parts.

    NOTE. Exponential Moving Average (EMA)

    There are multiple smoothing techinques that help to reduce overfitting on the val set.
    Available parameters: "covs", "stats", and "weights", that control smoothing of the covariance matrices, logging statistics, and mixing weights.

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
    def __init__(self, params, base_optimizer_cls,
                 base_optimizer_params=None, heads_groups=(1,), weights_names=None,
                 weights_parametrization="abs", encoder_downstream_weight=0, downstream_merge=False,
                 encoder_decoder=False, algorithm="sgd", ema=None, warmup=DEFAULT_WARMUP,
                 align="train", train_downstream_head="train",
                 regularization=0, apply_optimizer_correction=False, scale_gradients=1,
                 skip_step_zero_weights_limit=5, z_grad_lr=0.001, maxiters=100, eps=1e-8):
        params = list(params)
        if len(params) < 3 or not isinstance(params[0], dict) or not isinstance(params[1], dict):
            raise ValueError("Expected at least three param groups with the first group being hyperparameters weights, the second group being projection heads weights, and the third group being encoder weights.")
        if (len(params[0]["params"]) != 1) or (params[0]["params"][0].ndim != 1):
            raise ValueError("Weights must be flat.")
        if algorithm not in {"sgd", "warmup-sgd", "mse", "none"}:
            raise ValueError(f"Unexpected algorithm: {algorithm}")
        if weights_parametrization not in ["linear", "abs"]:
            raise ValueError(f"Unknown weights parametrization method: {weights_parametrization}")
        if align not in ["train", "val", "train-val"]:
            raise ValueError(f"Unknown align mode: {align}")
        if train_downstream_head not in ["train", "val", "train-val"]:
            raise ValueError(f"Unknown train downstream head mode: {train_downstream_head}")
        defaults = dict(base_optimizer_params or {})
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **(base_optimizer_params or {}))
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        if 0 in heads_groups:
            raise ValueError("The first group (index 0) is reserved for weights and can not relate to individual heads.")
        self.heads_groups = list(sorted(set(heads_groups)))
        encoder_groups = set(range(len(self.param_groups))) - {0} - set(heads_groups)
        self.encoder_groups = list(sorted(encoder_groups))
        self.n_weights = len(self.logits)
        if weights_names is None:
            weights_names = [str(i) for i in range(self.n_weights)]
        elif len(weights_names) != self.n_weights:
            raise ValueError("Names and weights lengths mismatch")
        self.weights_names = weights_names
        if self.n_weights == 0:
            raise ValueError("Empty hyperparameters list.")
        self.weights_parametrization = weights_parametrization
        self.encoder_downstream_weight = encoder_downstream_weight
        self.downstream_merge = downstream_merge
        self.encoder_decoder = encoder_decoder
        self.algorithm = algorithm
        self.align = align
        self.train_downstream_head = train_downstream_head
        self.regularization = regularization
        self.skip_step_zero_weights_limit = skip_step_zero_weights_limit
        self.eps = eps

        if ema is None:
            ema = {}
        elif isinstance(ema, Number):
            ema = {"covs": ema}
        ema_defaults = {
            "covs": DEFAULT_EMA_COVS,
            "weights": DEFAULT_EMA_WEIGHTS,
            "stats": DEFAULT_EMA_STATS
        }
        ema = dict(ema_defaults, **ema)
        unknown_keys = set(ema) - {"covs", "weights", "stats"}
        if unknown_keys:
            raise ValueError(f"Unknown EMA keys: {unknown_keys}")
        self.covs_momentum = ema["covs"]
        self.weights_momentum = ema["weights"]
        self.stats_momentum = ema["stats"]
        self.warmup = warmup

        self._normalizers = {name: GradientNormalizer(clip=self.eps ** 2,
                                                      momentum=self.stats_momentum,
                                                      check_shape=not encoder_decoder)
                             for name in self.weights_names}

        self.apply_optimizer_correction = apply_optimizer_correction
        self.scale_gradients = scale_gradients
        self.z_grad_lr = z_grad_lr
        self.maxiters = maxiters

        # TODO: use optimizer state for gradient caches.
        self._running_stats = {"covs": None, "products": None, "weights": None}
        if encoder_decoder:
            self._running_stats["encoder_transmission"] = None
            if (self.align == "train-val") and self.encoder_decoder:
                self._running_stats["encoder_grads"] = None
        self._buffers = {
            "n_skipped_steps": 0,
            "n_updates": 0
        }

        self._n_skipped_steps_tracker = StatsTracker("n_skipped_steps", self.stats_momentum, track_median=False)
        self._weights_tracker = StatsTracker("weights", self.stats_momentum)
        self._effective_weights_tracker = StatsTracker("effective_weights", self.stats_momentum)
        self._heads_grad_norm_tracker = StatsTracker("heads_grad_norm", self.stats_momentum, track_median=False)
        self._encoder_grad_norm_tracker = StatsTracker("encoder_grad_norm", self.stats_momentum, track_median=False)
        self._correlations_tracker = StatsTracker("grad_correlations", self.stats_momentum, track_median=False)

    @property
    def use_validation(self):
        return self.align != "train"

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
        if self._n_skipped_steps_tracker.last_value is not None:
            result.update(self._n_skipped_steps_tracker.get())
        if self._heads_grad_norm_tracker.last_value is not None:
            result.update(self._heads_grad_norm_tracker.get())
        if self._encoder_grad_norm_tracker.last_value is not None:
            result.update(self._encoder_grad_norm_tracker.get())
        for name, normalizer in self._normalizers.items():
            if not normalizer.is_first:
                result[f"moving_norm_{name}"] = normalizer.moving_norm
        if self.encoder_decoder and (self._running_stats["encoder_transmission"] is not None):
            result["encoder_transmission"] = self._running_stats["encoder_transmission"]
        return result

    def step(self, closure=None, *, inner=False):
        if not inner:
            raise ValueError("Please, use 'hpo_step' function.")
        self.base_optimizer.step(closure=closure)

    @property
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
        pretrain_covariances = pretrain_covariances.detach()
        norm = (weights[None] @ pretrain_covariances @ weights).sqrt()
        weights = weights / norm.clamp(min=self.eps ** 2)
        return weights

    def _update_running_stats(self, value, stage=None):
        warmup = False
        if stage not in self._running_stats:
            raise ValueError(f"Unknown stage: {stage}")
        if stage == "weights":
            momentum = self.weights_momentum
        elif stage in {"covs", "products", "encoder_grads"}:
            n_updates = self._buffers["n_updates"]
            warmup = n_updates < self.warmup
            momentum = self.covs_momentum
        elif stage in {"encoder_transmission"}:
            momentum = self.stats_momentum
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if self._running_stats[stage] is None:
            self._running_stats[stage] = value
        elif warmup:
            self._running_stats[stage] = (self._running_stats[stage] * n_updates + value) / (n_updates + 1)
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
            z_down_grads = torch.nn.functional.normalize(z_down.grad.flatten(), dim=0)
            z_down = z_down.clone()
        else:
            encoder_down_grads = torch.nn.functional.normalize(self._gather_grads("encoder"), dim=0)
        heads_down_grads = self._gather_grads("heads")

        # Caches for normalization differentiation.
        all_z_grads = []
        all_heads_grads = []
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
            if self.encoder_decoder:
                z_grad = z.grad.flatten().clone()
                self._normalizers[name](z_grad)
                all_z_grads.append(z_grad)
            else:
                encoder_grads = self._gather_grads("encoder")
                self._normalizers[name](encoder_grads)
                all_encoder_grads.append(encoder_grads)
            all_heads_grads.append(heads_grads)

        return {
            "heads_down_grads": heads_down_grads,
            "encoder_down_grads": encoder_down_grads if not self.encoder_decoder else None,
            "z_down": z_down if self.encoder_decoder else None,
            "z_down_grads": z_down_grads if self.encoder_decoder else None,
            "all_heads_grads": all_heads_grads,
            "all_encoder_grads": all_encoder_grads if not self.encoder_decoder else None,
            "all_z_grads": all_z_grads if self.encoder_decoder else None
        }

    @torch.no_grad()
    def _compute_weights_and_gradients(self, all_grads_covs, products):
        algorithm = self.algorithm
        if algorithm == "warmup-sgd":
            warmup = self._buffers["n_updates"] < self.warmup
            algorithm = "mse" if warmup else "sgd"

        if algorithm == "sgd":
            with torch.enable_grad():
                weights = self._normalize_weights(self._unnormalized_weights, pretrain_covariances=all_grads_covs)
            # Normalize products before backward to avoid tiny gradient magnitudes
            # propagating through the normalization graph when gradients are small.
            self.logits.grad = None
            weights.backward(-products)
            if self.regularization != 0:
                with torch.enable_grad():
                    regularization = (torch.linalg.norm(self.logits) - 1).square()
                    (regularization * self.regularization).backward()
            return weights, self.logits.grad.clone()
        elif algorithm == "mse":
            if self.weights_parametrization == "abs":
                positive = True
            else:
                assert self.weights_parametrization == "linear"
                positive = False

            if (products <= 0).all():
                # There is no positive step.
                weights = torch.zeros_like(self.logits)
            else:
                weights = solve_qcqp(all_grads_covs, products, positive=positive)
            return weights, None
        else:
            assert algorithm == "none"
            weights = self._normalize_weights(self._unnormalized_weights, pretrain_covariances=all_grads_covs)
            return weights, None

    @torch.no_grad()
    def _tune_weights(self, closure, embed_fn=None):
        """Update weights (gradient or value) and return cached gradients."""
        grads = self._get_loss_grads(closure)

        if self.align in {"train", "val"}:
            # Downstream and pretrain gradients are computed on the same batch.
            if self.encoder_decoder:
                down_grads = grads["z_down_grads"]
            else:
                down_grads = grads["encoder_down_grads"]
                if self.apply_optimizer_correction:
                    down_grads = down_grads.clone()
                    self.apply_optimizer_correction_("encoder", down_grads)
        else:
            assert self.align == "train-val"
            # Downstream and pretrain gradients are computed on different batches.
            # Now we process a training batch.
            down_grads = self._running_stats["encoder_grads"]
            if self.apply_optimizer_correction:
                down_grads = down_grads.clone()
                self.apply_optimizer_correction_("encoder", down_grads)
            if self.encoder_decoder:
                # Estimate z gradient on the current batch.
                if embed_fn is None:
                    raise ValueError("Need embed_fn for train-val alignment in encoder_decoder mode")
                with self._tmp_encoder_update(down_grads, lr=self.z_grad_lr):
                    z_down_after = embed_fn()
                z_grad = (grads["z_down"] - z_down_after) / self.z_grad_lr
                down_grads = z_grad.flatten()

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
            flat = torch.cat([all_grads_covs.flatten(), products])
            torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
            covs_numel = all_grads_covs.numel()
            all_grads_covs.copy_(flat[:covs_numel].view_as(all_grads_covs))
            products.copy_(flat[covs_numel:])
        # Equalize gradient norms.
        reduced_norms = all_grads_covs.diag().sqrt()  # (W).
        all_grads_covs /= reduced_norms[:, None] * reduced_norms[None, :]
        products /= reduced_norms

        all_grads_covs = self._update_running_stats(all_grads_covs, "covs")
        products = self._update_running_stats(products, "products")

        weights, logits_grads = self._compute_weights_and_gradients(all_grads_covs, products)

        weights = self._update_running_stats(weights, stage="weights")

        self.logits.grad = logits_grads
        if (logits_grads is None) and (self.algorithm != "none"):
            self.logits.copy_(weights)

        self._correlations_tracker.update(products.detach().clone())
        self._weights_tracker.update(weights.detach().clone())

        moving_norms = []
        for name in self.weights_names:
            normalizer = self._normalizers[name]
            if normalizer.is_first:
                break
            moving_norms.append(normalizer.moving_norm)
        else:
            moving_norms = torch.stack(moving_norms).clamp(min=self.eps)
            effective_weights = weights / moving_norms * (self._running_stats.get("encoder_transmission", None) or 1)
            self._effective_weights_tracker.update(effective_weights.detach().clone())

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
        if self.align == "train":
            if not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0):
                warnings.warn("Calling val_step, when align is `train`.")
            return
        assert closure is not None, "need closure"
        closure = torch.enable_grad()(closure)  # The closure should do a full forward-backward pass.
        if self.encoder_decoder:
            if closure_encoder is None:
                raise ValueError("Need encoder closure.")
            closure_encoder = torch.enable_grad()(closure_encoder)  # The closure should do a full forward-backward pass.

        if self.align == "train-val":
            # Cache encoder grads.
            downstream_weight = 1
            loss_weights = torch.zeros_like(self.logits)
            z_down = torch.enable_grad()(closure)(downstream_weight, loss_weights, retain_graph=False, stage=HPO_STAGE_DOWNSTREAM)
            if self.encoder_decoder:
                closure_encoder(z_down.grad.flatten())
            encoder_grads = self._gather_grads("encoder")
            if torch.linalg.norm(encoder_grads) < self.eps:
                raise RuntimeError("Zero-norm encoder gradient")
            self._update_running_stats(encoder_grads, stage="encoder_grads")
            return

        # Tune weights.
        assert self.align == "val"
        output_weights = torch.empty_like(self.logits)
        grads_ref = [None]

        @torch.no_grad()
        def inner_closure():
            grads_ref[0] = self._tune_weights(closure, embed_fn=embed_fn)
            output_weights.copy_(self._running_stats["weights"])

            if after_backward_hook is not None:
                after_backward_hook()

            if self.train_downstream_head in {"val", "train-val"}:
                # Apply downstream head gradients; zero encoder grads only.
                heads_down_grads = grads_ref[0]["heads_down_grads"]
                if self.scale_gradients != 1:
                    heads_down_grads = heads_down_grads * self.scale_gradients
                offset = 0
                for i in self.heads_groups:
                    for p in self.param_groups[i]["params"]:
                        numel = p.numel()
                        p.grad = heads_down_grads[offset:offset + numel].reshape(p.shape)
                        offset += numel
                encoder_group_indices = set(range(len(self.param_groups))) - {0} - set(self.heads_groups)
                for i in encoder_group_indices:
                    for p in self.param_groups[i]["params"]:
                        p.grad = None
            else:
                for group in self.param_groups[1:]:
                    for p in group["params"]:
                        p.grad = None

        if self.algorithm in {"sgd", "warmup-sgd"}:
            # Use optimizer — also applies head grads when train_downstream_head is "val" or "train-val".
            self.step(inner_closure, inner=True)
        else:
            # Closed-form computation.
            assert self.algorithm in {"mse", "none"}
            inner_closure()
            if self.train_downstream_head in {"val", "train-val"}:
                self.step(inner=True)
        return output_weights

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
            if self.align == "val":
                grads = self._get_loss_grads(closure)
            else:
                grads = self._tune_weights(closure, embed_fn=embed_fn)
            weights = self._running_stats["weights"]

            if weights is None:
                raise RuntimeError("Validation batch must be consumed first, when align is val")

            if (self.algorithm not in {"sgd", "warmup-sgd"}) and (self.skip_step_zero_weights_limit is not None) and (torch.linalg.norm(weights) < self.eps):
                skip_step = self._buffers["n_skipped_steps"] < self.skip_step_zero_weights_limit
                self._buffers["n_skipped_steps"] += 1
                if not skip_step:
                    weights = torch.ones_like(weights)
            else:
                skip_step = False
                self._buffers["n_skipped_steps"] = 0

            self._n_skipped_steps_tracker.update(self._buffers["n_skipped_steps"])

            # Cache returned value.
            output_weights.copy_(weights)

            # Set gradients for the encoder model weights.
            if self.encoder_decoder:
                # Backprop with z grads. Keep logits grad intact.
                scale = self.scale_gradients * (self._running_stats["encoder_transmission"] or 1)
                z_grad = (scale * weights) @ torch.stack(grads["all_z_grads"])
                z_grad.add_((scale * self.encoder_downstream_weight) * grads["z_down_grads"])
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
                if self.scale_gradients != 1:
                    encoder_grad *= self.scale_gradients
                self._encoder_grad_norm_tracker.update(torch.linalg.norm(encoder_grad))
                if self.downstream_merge:
                    encoder_down_grads = grads["encoder_down_grads"]
                    mask = encoder_grad == 0
                    encoder_grad = torch.where(mask, encoder_down_grads, encoder_grad)
                    encoder_down_grads.masked_fill_(mask, 0)
                    del mask
                param_groups = [self.param_groups[i] for i in self.encoder_groups]
                downstream_weight = self.encoder_downstream_weight
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
            if self.train_downstream_head in {"train", "train-val"}:
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

            if after_backward_hook is not None:
                after_backward_hook()

            if skip_step:
                raise ZeroWeightsException()
        try:
            self.step(inner_closure, inner=True)
        except ZeroWeightsException:
            pass
        return output_weights

    @contextmanager
    def _tmp_encoder_update(self, grads, lr):
        """Save encoder parameters on entry and restore them on exit."""
        encoder_params = [p for i in self.encoder_groups for p in self.param_groups[i]["params"]]
        saved = [p.data.clone() for p in encoder_params]
        try:
            offset = 0
            for p in encoder_params:
                numel = p.numel()
                p.data.copy_(p.data - lr * grads[offset:offset + numel].reshape(p.shape))
                offset += numel
            assert offset == len(grads)
            yield
        finally:
            for p, saved_data in zip(encoder_params, saved):
                p.data.copy_(saved_data)

    def state_dict(self):
        state = self.base_optimizer.state_dict()
        state["running_stats"] = dict(self._running_stats)
        state["buffers"] = dict(self._buffers)
        state["normalizers"] = {k: v.state_dict() for k, v in self._normalizers.items()}
        state["n_skipped_steps_tracker"] = self._n_skipped_steps_tracker.state_dict()
        state["weights_tracker"] = self._weights_tracker.state_dict()
        state["effective_weights_tracker"] = self._effective_weights_tracker.state_dict()
        state["heads_grad_norm_tracker"] = self._heads_grad_norm_tracker.state_dict()
        state["encoder_grad_norm_tracker"] = self._encoder_grad_norm_tracker.state_dict()
        state["correlations_tracker"] = self._correlations_tracker.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
        p = self.logits
        self._running_stats.update({k: (v.to(device=p.device, dtype=p.dtype) if v is not None else None)
                                    for k, v in state_dict.get("running_stats", {}).items()})
        self._buffers.update({k: (v.to(device=p.device, dtype=p.dtype) if isinstance(v, torch.Tensor) else v)
                              for k, v in state_dict.get("buffers", {}).items()})
        for k, v in state_dict["normalizers"].items():
            self._normalizers[k].load_state_dict(v)
        if "n_skipped_steps_tracker" in state_dict:
            self._n_skipped_steps_tracker.load_state_dict(state_dict["n_skipped_steps_tracker"])
        if "weights_tracker" in state_dict:
            self._weights_tracker.load_state_dict(state_dict["weights_tracker"])
        if "effective_weights_tracker" in state_dict:
            self._effective_weights_tracker.load_state_dict(state_dict["effective_weights_tracker"])
        if "heads_grad_norm_tracker" in state_dict:
            self._heads_grad_norm_tracker.load_state_dict(state_dict["heads_grad_norm_tracker"])
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
            part: Part of the model to extract gradients for (`all`, `heads`, or `encoder`).
        """
        if part == "all":
            param_groups = self.param_groups[1:]
        elif part == "heads":
            param_groups = [self.param_groups[i] for i in self.heads_groups]
        else:
            assert part == "encoder"
            # All except hyperparameters and individual heads.
            param_groups = [self.param_groups[i] for i in self.encoder_groups]
        grads = []
        for group in param_groups:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p).flatten())
                else:
                    grads.append(p.grad.flatten())
        if not grads:
            return torch.zeros_like(self.logits[:0])
        return torch.cat(grads)

    def apply_optimizer_correction_(self, part, grads):
        if part == "all":
            param_groups = self.param_groups[1:]
        elif part == "heads":
            param_groups = [self.param_groups[i] for i in self.heads_groups]
        else:
            assert part == "encoder"
            # All except hyperparameters and individual heads.
            param_groups = [self.param_groups[i] for i in self.encoder_groups]
        if isinstance(self.base_optimizer, torch.optim.Adam):
            offset = 0
            for group in param_groups:
                _, beta2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    state = self.base_optimizer.state[p]
                    exp_avg_sq = state.get("exp_avg_sq", None)
                    if exp_avg_sq is None:
                        offset += p.numel()
                        continue
                    exp_avg_sq = exp_avg_sq.flatten() * beta2 + grads[offset:offset + p.numel()].square() * (1 - beta2)
                    step = state["step"]
                    bias_correction2_sqrt = (1 - beta2 ** step) ** 0.5
                    grads[offset:offset + p.numel()] /= exp_avg_sq.sqrt() / bias_correction2_sqrt + eps
                    offset += p.numel()
            assert offset == len(grads)
        elif isinstance(self.base_optimizer, torch.optim.SGD):
            pass  # No need for correction.
        else:
            raise NotImplementedError(f"Can't apply correction to {type(self.base_optimizer).__name__}")
