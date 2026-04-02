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

from .gradient import GradientNormalizer, RescaleWeights
from .solvers import solve_qp, solve_qcqp


HPO_STAGE_DOWNSTREAM = "downstream"


DEFAULT_EMA_STATS = 0.9


class ZeroWeightsException(Exception):
    pass


class AlignedHPOptimizer(torch.optim.Optimizer):
    """Aligned Hyperparameter Optimizer.

    Args:
        params: Model parameters with 3 or more groups for encoder-decoder mode and 2 or more groups otherwise.
             The first group is for loss weights. In the encoder-decoder mode, second group is responsible for the
             decoder part of the model.
        base_optimizer_cls: The optimizer to use.
        base_optimizer_params: Parameters of the base optimizer.
        weights_names: An optional list of names for hyperparameters (for logging).
        weights_parametrization: Either "linear" or "abs".
        weights_normalization: Weights normalization type ("sum", "norm", "grad-norm", "grad-norm-scaled", or "none"), or a number to divide weights by.
        downstream_weight: The weight of the downstream gradient in encoder optimization. Default is 0 (disable).
        downstream_merge: Fill zero values in encoder gradient with downsrtream gradient.
        encoder_decoder: Whether to use encoder-decoder decomposition and upper bound optimization for fast hyperparameter tuning.
            This flag affects an intereface of the provided closure. See notes below.
        algorithm: Either "sgd", "mse", or "none" to disable HPO.
        ema: Use momentum for gradient smoothing. Can be a dictionary with "downstream", "main", "covs", "weights", and "stats" keys. See notes below.
        align: Either `train` to tune weights on the train set only, `val`, or `train-val` to align training gradients with validation downstream grad.
        apply_gradient_normalizer: Normalize gradients using running statistics.
        apply_optimizer_correction: Try to approximate an actual optimizer step rather than simple SGD.
        hp_simple_gd: Use simple gradient descent for the weights.
        clip_hp_grad: Clipping value for hyperparameters gradients when "sgd" algorithm is used.
        maxiters: The maximum number of iterations in the QP solver, used for the "mse" algorithm.

    NOTE. Exponential Moving Average (EMA)

    There are multiple smoothing techinques that help to reduce overfitting on the val set.
    Three main parameters: "downstream", "main", and "weights", that control smoothing of the pretraining and downstream gradients respectivelly.
    There are some additional smoothing parameters:
    - "covs" can be used to control smoothing of gradient covariances.
    - "stats" can be used to control covariances estimation smoothing. By default, "stats" smoothing is equal to DEFAULT_EMA_STATS otherwise.

    NOTE. Encoder-Decoder vs full gradients.

    In a simple "full" approach, closure must compute gradients for all model parameters, leading to multiple backward passes at each step.
    A more effective method decomposes the model into encoder and decoder part. A closure must be able to compute gradients w.r.t. to the
    embedding output of the encoder. A separate step is performed to pass aggregated gradient to the encoder part of the model.
    See examples below.

    Example usage (full gradients):
    ```
    optimizer = AlignedHPOptimizer([{"params": [weights]},  # Weights for tuning.
                                    {"params": heads.parameters()},  # Loss heads parameters.
                                    {"params": shared_head.parameters()},  # Shared losses head parameters, optional.
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
                                    {"params": model.decoder.parameters()},   # Shared decoder parameters (except individual heads), optional.
                                    {"params": model.encoder.parameters()}],  # Encoder.
                                   torch.optim.Adam,
                                   {"lr": 0.01})  # Optimizer parameters.

    embeddings = model.encode(x)
    z = embeddings.detach().clone()
    z.requires_grad = True
    output = model.decode(z)
    down_loss, loss1, loss2 = criterion(output, y)

    def closure(down_weight, weights, retain_graph=False, stage=None):
        optimizer.zero_grad()
        z.grad = None  # New.
        loss = down_weight * down_loss + weights[0] * loss1 + weights[1] * loss2
        loss.backward(retain_graph=retain_graph)
        return z  # New.

    def closure_encoder(z_grad):
        optimizer.zero_grad()
        embeddings.backward(z_grad)

    optimizer.hpo_step(closure, closure_encoder)
    ```
    """
    def __init__(self, params, base_optimizer_cls, base_optimizer_params=None, weights_names=None,
                 weights_parametrization="abs", weights_normalization="grad-norm-scaled",
                 encoder_downstream_weight=0, shared_downstream_weight=0, downstream_merge=False,
                 encoder_decoder=False, algorithm="sgd", ema=0, align="train",
                 apply_optimizer_correction=False, apply_gradient_normalizer=False,
                 skip_step_zero_weights_limit=5, hp_simple_gd=True, clip_hp_grad=None,
                 warmup_steps=0, z_grad_lr=0.001, maxiters=100, eps=1e-6):
        if (weights_parametrization == "linear") and (weights_normalization == "sum"):
            raise ValueError("A 'sum' normalization can be applied to positive weights only.")
        params = list(params)
        if len(params) < 3 or not isinstance(params[0], dict) or not isinstance(params[1], dict):
            raise ValueError("Expected at least three param groups with the first group being hyperparameters weights, second group being projectin heads weights, and third group being encoder weights.")
        if (len(params[0]["params"]) != 1) or (params[0]["params"][0].ndim != 1):
            raise ValueError("Weights must be flat.")
        if algorithm not in {"sgd", "mse", "none"}:
            raise ValueError(f"Unexpected algorithm: {algorithm}")
        if weights_parametrization not in ["linear", "abs"]:
            raise ValueError(f"Unknown weights parametrization method: {weights_parametrization}")
        if weights_normalization not in ["sum", "norm", "grad-norm", "grad-norm-scaled", "none"] and not isinstance(weights_normalization, Number):
            raise ValueError(f"Unknown weights normalization method: {weights_normalization}")
        if align not in ["train", "val", "train-val"]:
            raise ValueError(f"Unknown align mode: {align}")
        defaults = dict(base_optimizer_params or {})
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **(base_optimizer_params or {}))
        self.param_groups = self.base_optimizer.param_groups
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
        self.shared_downstream_weight = shared_downstream_weight
        self.encoder_downstream_weight = encoder_downstream_weight
        self.downstream_merge = downstream_merge
        self.encoder_decoder = encoder_decoder
        self.algorithm = algorithm
        self.align = align
        self.skip_step_zero_weights_limit = skip_step_zero_weights_limit

        if isinstance(ema, Number):
            ema = {k: ema for k in ["downstream", "main"]}
        ema_defaults = {
            "downstream": 0,
            "main": 0,
            "covs": 0,
            "weights": 0,
            "stats": DEFAULT_EMA_STATS
        }
        ema = dict(ema_defaults, **ema)
        unknown_keys = set(ema) - {"downstream", "main", "covs", "weights", "stats"}
        if unknown_keys:
            raise ValueError(f"Unknown EMA keys: {unknown_keys}")
        self.downstream_momentum = ema["downstream"]
        self.main_momentum = ema["main"]
        self.covs_momentum = ema["covs"]
        self.weights_momentum = ema["weights"]
        self.stats_momentum = ema["stats"]

        self.apply_optimizer_correction = apply_optimizer_correction
        self.apply_gradient_normalizer = apply_gradient_normalizer
        if apply_gradient_normalizer:
            self.heads_gradient_normalizer = GradientNormalizer(clip=1e-6, momentum=self.stats_momentum)
            self.shared_gradient_normalizer = GradientNormalizer(clip=1e-6, momentum=self.stats_momentum)
        if algorithm == "sgd":
            self.hp_gradient_normalizer = GradientNormalizer(clip=1e-12, momentum=self.stats_momentum)
        self.hp_simple_gd = hp_simple_gd
        self.clip_hp_grad = clip_hp_grad
        self.warmup_steps = warmup_steps
        self.z_grad_lr = z_grad_lr
        self.maxiters = maxiters
        self.eps = eps

        # TODO: use optimizer state for gradient caches.
        self._grads_cache = {HPO_STAGE_DOWNSTREAM: None, "weights": None, "covs": None, "products": None} | {i: None for i in range(self.n_weights)}
        if encoder_decoder:
            self._grads_cache["jacobian"] = None
            if (self.align == "train-val") and self.encoder_decoder:
                self._grads_cache["encoder"] = None
        self._buffers = {
            "n_skipped_steps": 0,
            "n_updates": 0,
            "weights": None,
            "ema_weights": None,
            "avg_weights": None,
            "tune_grad_norm_downstream": None,
            "tune_grad_norms": None
        }

        if self.algorithm != "none":
            self._buffers["correlations"] = None
            self._buffers["ema_correlations"] = None
            self._buffers["avg_correlations"] = None

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
        if self._grads_cache.get("jacobian", None) is not None:
            result["jacobian_norm"] = self._grads_cache["jacobian"]
        if (self.algorithm not in {"none"}) and (self._buffers["correlations"] is not None):
            for name, c in zip(self.weights_names, self._buffers["correlations"]):
                result[f"grad_correlations_{name}"] = c
            for name, c in zip(self.weights_names, self._buffers["avg_correlations"]):
                result[f"avg_grad_correlations_{name}"] = c
            for name, c in zip(self.weights_names, self._buffers["ema_correlations"]):
                result[f"ema_grad_correlations_{name}"] = c
        if self._buffers["weights"] is not None:
            for name, c in zip(self.weights_names, self._buffers["weights"]):
                result[f"weights_{name}"] = c
            for name, c in zip(self.weights_names, self._buffers["avg_weights"]):
                result[f"avg_weights_{name}"] = c
            for name, c in zip(self.weights_names, self._buffers["ema_weights"]):
                result[f"ema_weights_{name}"] = c
        if self._buffers["tune_grad_norms"] is not None:
            for name, c in zip(self.weights_names, self._buffers["tune_grad_norms"]):
                result[f"tune_grad_norm_{name}"] = c
        if self._buffers["tune_grad_norm_downstream"] is not None:
            result[f"tune_grad_norm_downstream"] = self._buffers["tune_grad_norm_downstream"]
        if self.apply_gradient_normalizer:
            if not self.heads_gradient_normalizer.is_first:
                result["heads_gradient_moving_norm"] = self.heads_gradient_normalizer.moving_norm
            if not self.shared_gradient_normalizer.is_first:
                result["shared_gradient_moving_norm"] = self.shared_gradient_normalizer.moving_norm
        if self.algorithm == "sgd":
            if not self.hp_gradient_normalizer.is_first:
                result["hp_gradient_moving_norm"] = self.hp_gradient_normalizer.moving_norm
        return result

    def step(self, closure=None, *, inner=False):
        if not inner:
            raise ValueError("Please, use 'hpo_step' function.")
        self.base_optimizer.step(closure=closure)

    @property
    def _unnormalized_weights(self):
        if self.weights_parametrization == "abs":
            weights = torch.abs(self.logits)
        elif self.weights_parametrization == "linear":
            weights = self.logits
        else:
            raise RuntimeError(f"Unknown parametrization: {self.weights_parametrization}")
        return weights

    def _normalize_weights(self, weights, pretrain_covariances=None):
        if torch.linalg.norm(weights) < self.eps:
            return weights
        if isinstance(self.weights_normalization, Number):
            return weights / self.weights_normalization
        elif self.weights_normalization == "sum":
            return self.n_weights * weights / (weights.sum() + self.eps)
        elif self.weights_normalization == "norm":
            return math.sqrt(self.n_weights) * weights / (torch.linalg.norm(weights) + self.eps)
        elif self.weights_normalization in {"grad-norm", "grad-norm-scaled"}:
            if pretrain_covariances is None:
                raise ValueError("Need covariances for grad-norm")
            pretrain_covariances = pretrain_covariances.detach()

            # Pre-scale covariance to O(1) for numerical stability with tiny gradients.
            # All intermediate computations use the pre-scaled matrix; the final result
            # is corrected so that it equals the one obtained without pre-scaling.
            cov_scale = pretrain_covariances.diag().max().clamp(min=self.eps ** 2)

            # Scale to the gradient norm of equal weights (use original covariance).
            target_norm = pretrain_covariances.sum().sqrt() if self.weights_normalization == "grad-norm-scaled" else 1

            pretrain_covariances = pretrain_covariances / cov_scale

            # Scale weights and covariances.
            scales = torch.diag(pretrain_covariances).sqrt().clamp(min=self.eps)  # (W).
            pretrain_covariances = pretrain_covariances / (scales[:, None] * scales[None, :])
            weights = RescaleWeights.apply(weights, scales)

            # Normalize. The pre-scaled norm² = w^T (C/cov_scale) w = (w^T C w) / cov_scale,
            # so multiply by sqrt(cov_scale) to recover the true norm.
            norm = (weights[None] @ pretrain_covariances @ weights).sqrt() * cov_scale.sqrt()
            norm = norm.clamp(min=self.eps)
            weights = weights / norm
            weights = weights * target_norm

            # Unscale.
            weights = weights / scales

            return weights
        else:
            assert self.weights_normalization == "none"
            return weights

    def _update_grads_cache(self, grads, stage=None):
        if stage not in self._grads_cache:
            raise ValueError(f"Unknown stage: {stage}")
        if stage in {HPO_STAGE_DOWNSTREAM, "encoder"}:
            momentum = self.downstream_momentum
        elif stage == "weights":
            momentum = self.weights_momentum
        elif stage in {"jacobian"}:
            momentum = self.stats_momentum
        elif stage in {"covs", "products"}:
            momentum = self.covs_momentum
        else:
            assert isinstance(stage, Number)
            momentum = self.main_momentum
        # TODO: Don't store gradients if momentum = 0.

        if self._grads_cache[stage] is None:
            self._grads_cache[stage] = grads
        else:
            if momentum > 0:
                grads = self._grads_cache[stage] * momentum + grads * (1 - momentum)
            self._grads_cache[stage] = grads
        return self._grads_cache[stage]

    @torch.no_grad()
    def _get_loss_grads(self, closure):
        """Update weights (gradient or value) and return cached gradients."""
        loss_weights = torch.zeros_like(self.logits)

        # Below:
        # - heads grads: individual heads weights gradients.
        # - z grads: gradient w.r.t. encoder output.
        # - shared grads: shared decoder weights gradients before EMA.
        # - grads: HPO grads after EMA smoothing used for weights tuning.

        # Compute downstream grads.
        downstream_weight = 1
        z_down = closure(downstream_weight, loss_weights, retain_graph=True, stage=HPO_STAGE_DOWNSTREAM)
        if self.encoder_decoder and (z_down is None or z_down.grad is None):
            raise TypeError("In the encoder-decoder mode, closure must return embedding with gradient.")
        if self.encoder_decoder:
            z_down_grads = z_down.grad.flatten().clone()
            z_down = z_down.clone()
        heads_down_grads = self._gather_grads("heads")
        shared_down_grads = self._gather_grads("shared")

        # Caches for normalization differentiation.
        all_z_grads = []
        all_heads_grads = []
        all_shared_grads = []

        # Compute main losses grads.
        downstream_weight = 0
        for i in range(self.n_weights):
            loss_weights[i] = 1
            z = closure(downstream_weight, loss_weights, retain_graph=(i < self.n_weights - 1), stage=i)
            loss_weights[i] = 0
            if self.encoder_decoder and (z is None or z.grad is None):
                raise TypeError("In the encoder-decoder mode, closure must return gradient w.r.t. encoder output.")
            heads_grads = self._gather_grads("heads")
            shared_grads = self._gather_grads("shared")
            if self.encoder_decoder:
                all_z_grads.append(z.grad.flatten().clone())
            all_heads_grads.append(heads_grads)
            all_shared_grads.append(shared_grads)

        return {
            "heads_down_grads": heads_down_grads,
            "shared_down_grads": shared_down_grads,
            "z_down": z_down if self.encoder_decoder else None,
            "z_down_grads": z_down_grads if self.encoder_decoder else None,
            "all_heads_grads": all_heads_grads,
            "all_shared_grads": all_shared_grads,
            "all_z_grads": all_z_grads,
        }

    @torch.no_grad()
    def _compute_weights_gradients(self, all_grads_covs, products):
        if self.algorithm == "sgd":
            with torch.enable_grad():
                weights = self._normalize_weights(self._unnormalized_weights, pretrain_covariances=all_grads_covs)
            # Normalize products before backward to avoid tiny gradient magnitudes
            # propagating through the normalization graph when gradients are small.
            products_scale = products.norm().clamp(min=self.eps)
            self.logits.grad = None
            weights.backward(-products / products_scale)
            self.logits.grad = self.logits.grad * products_scale
        elif self.algorithm == "mse":
            if self.weights_parametrization == "abs":
                positive = True
            else:
                assert self.weights_parametrization == "linear"
                positive = False

            if self.weights_normalization not in {"grad-norm", "grad-norm-scaled"}:
                weights = solve_qp(all_grads_covs, -products, positive=positive,
                                   maxiters=self.maxiters, eps=self.eps)
                weights = self._normalize_weights(weights)
            else:
                scale = all_grads_covs.detach().sum().sqrt() if self.weights_normalization == "grad-norm-scaled" else 1
                weights = solve_qcqp(all_grads_covs, products, positive=positive) * scale
            self.logits.grad = self.logits - weights
        else:
            assert self.algorithm == "none"
            self.logits.grad = torch.zeros_like(self.logits)

    @torch.no_grad()
    def _tune_weights(self, closure, embed_fn=None):
        """Update weights (gradient or value) and return cached gradients."""
        grads = self._get_loss_grads(closure)

        if self.align in {"train", "val"}:
            # Downstream and pretrain gradients are computed on the same batch.
            down_grads = self._update_grads_cache(grads["shared_down_grads"], stage=HPO_STAGE_DOWNSTREAM)
            if self.apply_optimizer_correction:
                down_grads = down_grads.clone()
                self.apply_optimizer_correction_("shared", down_grads)
            if self.encoder_decoder:
                down_grads = torch.cat([grads["z_down_grads"], down_grads])
        else:
            assert self.align == "train-val"
            # Downstream and pretrain gradients are computed on different batches.
            # Now we process a training batch.
            down_grads = self._grads_cache[HPO_STAGE_DOWNSTREAM]
            if self.apply_optimizer_correction:
                down_grads = down_grads.clone()
                self.apply_optimizer_correction_("shared", down_grads)
            if self.encoder_decoder:
                # Estimate z gradient on the current batch.
                if embed_fn is None:
                    raise ValueError("Need embed_fn for train-val alignment in encoder_decoder mode")
                encoder_down_grads = self._grads_cache["encoder"]
                if self.apply_optimizer_correction:
                    encoder_down_grads = encoder_down_grads.clone()
                    self.apply_optimizer_correction_("encoder", encoder_down_grads)
                with self._tmp_encoder_update(encoder_down_grads, lr=self.z_grad_lr):
                    z_down_after = embed_fn()
                z_grad = (grads["z_down"] - z_down_after) / self.z_grad_lr
                down_grads = torch.cat([z_grad.flatten(), down_grads])

        # Caches for normalization differentiation.
        all_grads = []

        # Compute main losses grads.
        for i in range(self.n_weights):
            loss_grads = self._update_grads_cache(grads["all_shared_grads"][i], stage=i)
            if self.apply_optimizer_correction:
                loss_grads = loss_grads.clone()
                self.apply_optimizer_correction_("shared", loss_grads)
            loss_grads = torch.cat([grads["all_z_grads"][i], loss_grads]) if self.encoder_decoder else loss_grads
            all_grads.append(loss_grads)

        all_grads = torch.stack(all_grads, 0)  # (W, P).

        all_grads_scale = torch.linalg.norm(all_grads, dim=1).max().clamp(min=1e-12)
        all_grads /= all_grads_scale
        down_grads = torch.nn.functional.normalize(down_grads, dim=0)

        all_grads_covs = all_grads @ all_grads.T
        products = all_grads @ down_grads  # (W).

        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized() and (torch.distributed.get_world_size() > 1)
        if is_distributed:
            torch.distributed.all_reduce(all_grads_covs, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(products, op=torch.distributed.ReduceOp.SUM)
            all_grads_covs /= torch.distributed.get_world_size() ** 2
            products /= torch.distributed.get_world_size() ** 2

        all_grads_covs = self._update_grads_cache(all_grads_covs, "covs")
        products = self._update_grads_cache(products, "products")

        logits = self.logits
        self._buffers["correlations"] = products.detach().clone()
        self._buffers["tune_grad_norms"] = all_grads_covs.diag().sqrt()
        self._buffers["tune_grad_norm_downstream"] = torch.linalg.norm(down_grads)

        if self._buffers["n_updates"] < self.warmup_steps:
            self.logits.grad = torch.zeros_like(self.logits)
        else:
            self._compute_weights_gradients(all_grads_covs, products)
        if is_distributed:
            torch.distributed.all_reduce(self.logits.grad, op=torch.distributed.ReduceOp.SUM)
            self.logits.grad /= torch.distributed.get_world_size()

        if self.clip_hp_grad is not None:
            grad_norm = torch.linalg.norm(self.logits.grad)
            if grad_norm > self.clip_hp_grad:
                self.logits.grad *= self.clip_hp_grad / (grad_norm + self.eps)

        if self.algorithm != "sgd":
            # Set weights closed-form.
            self.logits.copy_(self.logits - self.logits.grad)
            self.logits.grad = None
            weights = self.logits
        else:
            weights = self._normalize_weights(self._unnormalized_weights, pretrain_covariances=all_grads_covs)
        weights = self._update_grads_cache(weights, stage="weights")

        self._buffers["n_updates"] += 1
        n_updates = self._buffers["n_updates"]

        if n_updates > 1:
            self._buffers["avg_correlations"] *= (n_updates - 1) / n_updates
            self._buffers["avg_correlations"] += self._buffers["correlations"] / n_updates
            self._buffers["ema_correlations"] *= self.stats_momentum
            self._buffers["ema_correlations"] += (1 - self.stats_momentum) * self._buffers["correlations"]
        else:
            self._buffers["avg_correlations"] = self._buffers["correlations"].clone()
            self._buffers["ema_correlations"] = self._buffers["correlations"].clone()

        self._buffers["weights"] = weights.detach().clone()
        if n_updates > 1:
            self._buffers["avg_weights"] *= (n_updates - 1) / n_updates
            self._buffers["avg_weights"] += self._buffers["weights"] / n_updates
            self._buffers["ema_weights"] *= self.stats_momentum
            self._buffers["ema_weights"] += (1 - self.stats_momentum) * self._buffers["weights"]
        else:
            self._buffers["avg_weights"] = self._buffers["weights"].clone()
            self._buffers["ema_weights"] = self._buffers["weights"].clone()

        return grads | {"all_grads_covs": all_grads_covs}

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
            # Cache encoder and shared grads.
            downstream_weight = 1
            loss_weights = torch.zeros_like(self.logits)
            z_down = torch.enable_grad()(closure)(downstream_weight, loss_weights, retain_graph=False, stage=HPO_STAGE_DOWNSTREAM)
            self._update_grads_cache(self._gather_grads("shared"), stage=HPO_STAGE_DOWNSTREAM)
            if self.encoder_decoder:
                closure_encoder(z_down.grad.flatten())
                encoder_grads = self._gather_grads("encoder")
                if torch.linalg.norm(encoder_grads) < 1e-12:
                    raise RuntimeError("Zero-norm encoder gradient")
                self._update_grads_cache(encoder_grads, stage="encoder")
            return

        # Tune weights.
        assert self.align == "val"
        output_weights = torch.empty_like(self.logits)

        @torch.no_grad()
        def inner_closure():
            grads = self._tune_weights(closure, embed_fn=embed_fn)
            weights = self._grads_cache["weights"]
            output_weights.copy_(weights)

            if self.algorithm == "sgd":
                self.hp_gradient_normalizer([self.logits])

            if after_backward_hook is not None:
                after_backward_hook()
        if (self.algorithm == "sgd") and self.hp_simple_gd:
            inner_closure()
            lr = self.param_groups[0].get("lr", self.defaults.get("lr", None))
            with torch.no_grad():
                self.logits.add_(- lr * self.logits.grad)
        elif self.algorithm == "sgd":
            # Use optimizer.
            self.step(inner_closure, inner=True)
        else:
            # Closed-form computation.
            assert self.algorithm in {"mse", "none"}
            inner_closure()
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
        logits_grads = torch.empty_like(self.logits)

        @torch.no_grad()
        def inner_closure():
            if self.align == "val":
                grads = self._get_loss_grads(closure)
            else:
                grads = self._tune_weights(closure, embed_fn=embed_fn)
            weights = self._grads_cache["weights"]

            if (self.algorithm != "sgd") and (self.skip_step_zero_weights_limit is not None) and (torch.linalg.norm(weights) < self.eps):
                skip_step = self._buffers["n_skipped_steps"] < self.skip_step_zero_weights_limit
                self._buffers["n_skipped_steps"] += 1
                if not skip_step:
                    weights = self._normalize_weights(torch.ones_like(weights), pretrain_covariances=grads.get("all_grads_covs", None))
            else:
                skip_step = False
                self._buffers["n_skipped_steps"] = 0

            # Cache returned value.
            output_weights.copy_(weights)

            # Set gradients for model weights.
            if self.encoder_decoder:
                # Set grads for the encoder (backbone) model. Keep logits grad intact.
                z_grad = sum([w * grads["all_z_grads"][i] for i, w in enumerate(weights)], self.encoder_downstream_weight * grads["z_down_grads"])
                logits_grad = self.logits.grad
                self.logits.grad = None
                closure_encoder(z_grad)
                self.logits.grad = logits_grad
                z_grad_norm = torch.linalg.norm(z_grad)
                if z_grad_norm > self.eps:
                    encoder_grad = self._gather_grads("encoder")
                    jacobian_norm = torch.linalg.norm(encoder_grad) / z_grad_norm
                    self._update_grads_cache(jacobian_norm, stage="jacobian")
                    del encoder_grad
                del z_grad

            # Set grads for the shared model.
            shared_grad = sum([w * grads["all_shared_grads"][i] for i, w in enumerate(weights[:-1])], weights[-1] * grads["all_shared_grads"][-1])
            if self.downstream_merge:
                shared_down_grads = grads["shared_down_grads"]
                mask = shared_grad == 0
                shared_grad = torch.where(mask, shared_down_grads, shared_grad)
                shared_down_grads.masked_fill_(mask, 0)
                del mask
            if self.encoder_decoder:
                param_groups = [self.param_groups[2]]
            else:
                param_groups = self.param_groups[2:]
            offset = 0
            for i, group in enumerate(param_groups):
                downstream_weight = self.shared_downstream_weight if i == 0 else self.encoder_downstream_weight
                for p in group["params"]:
                    numel = p.numel()
                    p.grad = shared_grad[offset:offset + numel].reshape(p.shape)
                    if downstream_weight > 0:
                        p.grad += downstream_weight * shared_down_grads[offset:offset + numel].reshape(p.shape)
                    offset += numel
            assert offset == len(shared_grad)
            del shared_grad

            # Set grads for individual heads model.
            heads_grad = sum(grads["all_heads_grads"], grads["heads_down_grads"])
            offset = 0
            for p in self.param_groups[1]["params"]:
                numel = p.numel()
                p.grad = heads_grad[offset:offset + numel].reshape(p.shape)
                offset += numel
            assert offset == len(heads_grad)
            del heads_grad

            if self.apply_gradient_normalizer:
                self.heads_gradient_normalizer(self.param_groups[1]["params"])
                self.shared_gradient_normalizer(itertools.chain(*[group["params"] for group in self.param_groups[2:]]))
            if self.algorithm == "sgd":
                self.hp_gradient_normalizer([self.logits])

            if after_backward_hook is not None:
                after_backward_hook()

            if self.algorithm == "sgd":
                logits_grads.copy_(self.logits.grad)

            if skip_step:
                raise ZeroWeightsException()
        try:
            logits_orig = self.logits.clone()
            self.step(inner_closure, inner=True)
            if self.hp_simple_gd and (self.algorithm == "sgd") and (self.align != "val"):
                lr = self.param_groups[0].get("lr", self.defaults.get("lr", None))
                with torch.no_grad():
                    self.logits.copy_(logits_orig - lr * logits_grads)
        except ZeroWeightsException:
            pass
        return output_weights

    @contextmanager
    def _tmp_encoder_update(self, grads, lr):
        """Save encoder parameters on entry and restore them on exit."""
        encoder_params = [p for group in self.param_groups[3:] for p in group["params"]]
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
        state["grads_cache"] = dict(self._grads_cache)
        state["buffers"] = dict(self._buffers)
        if self.apply_gradient_normalizer:
            state["heads_gradient_normalizer"] = self.heads_gradient_normalizer.state_dict()
            state["shared_gradient_normalizer"] = self.shared_gradient_normalizer.state_dict()
        if self.algorithm == "sgd":
            state["hp_gradient_normalizer"] = self.hp_gradient_normalizer.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
        p = self.logits
        self._grads_cache.update({k: (v.to(device=p.device, dtype=p.dtype) if v is not None else None)
                                  for k, v in state_dict.get("grads_cache", {}).items()})
        self._buffers.update({k: (v.to(device=p.device, dtype=p.dtype) if isinstance(v, torch.Tensor) else v)
                              for k, v in state_dict.get("buffers", {}).items()})
        if self.apply_gradient_normalizer:
             self.heads_gradient_normalizer.load_state_dict(state_dict["heads_gradient_normalizer"])
             self.shared_gradient_normalizer.load_state_dict(state_dict["shared_gradient_normalizer"])
        if self.algorithm == "sgd":
             self.hp_gradient_normalizer.load_state_dict(state_dict["hp_gradient_normalizer"])

    @torch.no_grad()
    def _gather_grads(self, part):
        """Get gradients vector.

        Model parts:
            heads: Individual losses heads.
            shared: Shared decoder in the encoder-decoder model or all weights except heads otherwise.
            encoder: Encoder part (before embedding) in the encoder-decoder model.

        Args:
            part: Part of the model to extract gradients for (`all`, `heads`, `shared`, or `encoder`).
        """
        if part == "all":
            param_groups = self.param_groups[1:]
        elif part == "heads":
            param_groups = [self.param_groups[1]]
        elif self.encoder_decoder:
            if part == "shared":
                param_groups = [self.param_groups[2]]
            else:
                assert part == "encoder"
                param_groups = self.param_groups[3:]
        else:
            assert part == "shared"
            # All except hyperparameters and individual heads.
            param_groups = self.param_groups[2:]
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
            param_groups = [self.param_groups[1]]
        elif self.encoder_decoder:
            if part == "shared":
                param_groups = [self.param_groups[2]]
            else:
                assert part == "encoder"
                param_groups = self.param_groups[3:]
        else:
            assert part == "shared"
            # All except hyperparameters and individual heads.
            param_groups = self.param_groups[2:]
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
