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


HPO_STAGE_DOWNSTREAM = "downstream"


DEFAULT_EMA_STATS = 0.9


class ZeroWeightsException(Exception):
    pass


class AlignedHPOptimizer(torch.optim.Optimizer):
    """Aligned Hyperparameter Optimizer.

    Args:
        params: Model parameters with 3 or more groups. See parameter groups note below.
        base_optimizer_cls: The optimizer to use.
        base_optimizer_params: Parameters of the base optimizer.
        weights_names: An optional list of names for hyperparameters (for logging).
        weights_parametrization: Either "linear" or "abs".
        weights_normalization: Weights normalization type ("sum", "norm", "grad-norm", "grad-norm-scaled", or "none"), or a number to divide weights by.
        encoder_downstream_weight: The weight of the downstream gradient in encoder optimization. Default is 0 (disable).
        downstream_merge: Fill zero values in encoder gradient with downsrtream gradient.
        encoder_decoder: Whether to use encoder-decoder decomposition and upper bound optimization for fast hyperparameter tuning.
            This flag affects an intereface of the provided closure. See notes below.
        algorithm: Either "sgd", "mse", or "none" to disable HPO.
        ema: Use momentum for smoothing statistics. Can be a dictionary with "covs", "weights", and "stats" keys. See notes below.
        align: Either `train` to tune weights on the train set only, `val` to tune weights on validation, or `train-val` to align training gradients with validation downstream grad.
        apply_gradient_normalizer: Normalize gradients using running statistics. Normalization is applied indipendently for loss weights, individual heads, and shared part.
        apply_optimizer_correction: Try to approximate an actual optimizer step rather than simple SGD.
        skip_step_zero_weights_limit: Skip optimizer step, when weights are zero. When the limit reached, continue training with equal weights.
        z_grad_lr: The encoder step size, when encoder-decoder is used with `train-val` alignment.
        maxiters: The maxmum number of iterations in the closed-form solver.
        eps: Roughly the square root of the minimum gradients correlation value.

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
    def __init__(self, params, base_optimizer_cls, base_optimizer_params=None, weights_names=None,
                 weights_parametrization="abs", weights_normalization="grad-norm-scaled",
                 encoder_downstream_weight=0, downstream_merge=False,
                 encoder_decoder=False, algorithm="sgd", ema=0, align="train",
                 apply_optimizer_correction=False, apply_gradient_normalizer=False,
                 skip_step_zero_weights_limit=5, z_grad_lr=0.001, maxiters=100, eps=1e-8):
        if (weights_parametrization == "linear") and (weights_normalization == "sum"):
            raise ValueError("A 'sum' normalization can be applied to positive weights only.")
        params = list(params)
        if len(params) < 3 or not isinstance(params[0], dict) or not isinstance(params[1], dict):
            raise ValueError("Expected at least three param groups with the first group being hyperparameters weights, the second group being projection heads weights, and the third group being encoder weights.")
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
        self.encoder_downstream_weight = encoder_downstream_weight
        self.downstream_merge = downstream_merge
        self.encoder_decoder = encoder_decoder
        self.algorithm = algorithm
        self.align = align
        self.skip_step_zero_weights_limit = skip_step_zero_weights_limit

        if isinstance(ema, Number):
            ema = {k: ema for k in ["covs", "stats"]}
        ema_defaults = {
            "covs": 0,
            "weights": 0,
            "stats": DEFAULT_EMA_STATS
        }
        ema = dict(ema_defaults, **ema)
        unknown_keys = set(ema) - {"covs", "weights", "stats"}
        if unknown_keys:
            raise ValueError(f"Unknown EMA keys: {unknown_keys}")
        self.covs_momentum = ema["covs"]
        self.weights_momentum = ema["weights"]
        self.stats_momentum = ema["stats"]

        self.apply_optimizer_correction = apply_optimizer_correction
        self.apply_gradient_normalizer = apply_gradient_normalizer
        if apply_gradient_normalizer:
            self.heads_gradient_normalizer = GradientNormalizer(clip=self.eps, momentum=self.stats_momentum)
            self.encoder_gradient_normalizer = GradientNormalizer(clip=self.eps, momentum=self.stats_momentum)
            if algorithm == "sgd":
                self.hp_gradient_normalizer = GradientNormalizer(clip=self.eps ** 2, momentum=self.stats_momentum)
        self.z_grad_lr = z_grad_lr
        self.maxiters = maxiters
        self.eps = eps

        # TODO: use optimizer state for gradient caches.
        self._grads_cache = {"covs": None, "products": None, "weights": None}
        if encoder_decoder:
            if (self.align == "train-val") and self.encoder_decoder:
                self._grads_cache["encoder"] = None
        self._buffers = {
            "n_skipped_steps": 0,
            "ema_n_skipped_steps": 0,
            "avg_n_skipped_steps": 0,
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
        if self._buffers["n_skipped_steps"] is not None:
            result[f"n_skipped_steps"] = self._buffers["n_skipped_steps"]
            result[f"avg_n_skipped_steps"] = self._buffers["avg_n_skipped_steps"]
            result[f"ema_n_skipped_steps"] = self._buffers["ema_n_skipped_steps"]
        if self._buffers["tune_grad_norms"] is not None:
            for name, c in zip(self.weights_names, self._buffers["tune_grad_norms"]):
                result[f"tune_grad_norm_{name}"] = c
        if self._buffers["tune_grad_norm_downstream"] is not None:
            result[f"tune_grad_norm_downstream"] = self._buffers["tune_grad_norm_downstream"]
        if self.apply_gradient_normalizer:
            if not self.heads_gradient_normalizer.is_first:
                result["heads_gradient_moving_norm"] = self.heads_gradient_normalizer.moving_norm
            if not self.encoder_gradient_normalizer.is_first:
                result["encoder_gradient_moving_norm"] = self.encoder_gradient_normalizer.moving_norm
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

            # Scale to the gradient norm of equal weights (use original covariance).
            target_norm = pretrain_covariances.sum().sqrt() if self.weights_normalization == "grad-norm-scaled" else 1

            norm = (weights[None] @ pretrain_covariances @ weights).sqrt()
            weights = weights * (target_norm / norm.clamp(min=self.eps ** 2))

            return weights
        else:
            assert self.weights_normalization == "none"
            return weights

    def _update_grads_cache(self, grads, stage=None):
        if stage not in self._grads_cache:
            raise ValueError(f"Unknown stage: {stage}")
        if stage == "weights":
            momentum = self.weights_momentum
        elif stage in {"covs", "products", "encoder"}:
            momentum = self.covs_momentum
        else:
            raise ValueError(f"Unknown stage: {stage}")

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
        if not self.encoder_decoder:
            encoder_down_grads = self._gather_grads("encoder")

        # Caches for normalization differentiation.
        all_z_grads = []
        all_heads_grads = []
        all_encoder_grads = []

        # Compute main losses grads.
        downstream_weight = 0
        for i in range(self.n_weights):
            loss_weights[i] = 1
            z = closure(downstream_weight, loss_weights, retain_graph=(i < self.n_weights - 1), stage=i)
            loss_weights[i] = 0
            if self.encoder_decoder and (z is None or z.grad is None):
                raise TypeError("In the encoder-decoder mode, closure must return gradient w.r.t. encoder output.")
            heads_grads = self._gather_grads("heads")
            if self.encoder_decoder:
                all_z_grads.append(z.grad.flatten().clone())
            else:
                encoder_grads = self._gather_grads("encoder")
                all_encoder_grads.append(encoder_grads)
            all_heads_grads.append(heads_grads)

        return {
            "heads_down_grads": heads_down_grads,
            "encoder_down_grads": encoder_down_grads if not self.encoder_decoder else None,
            "z_down": z_down if self.encoder_decoder else None,
            "z_down_grads": z_down_grads if self.encoder_decoder else None,
            "all_heads_grads": all_heads_grads,
            "all_encoder_grads": all_encoder_grads if not self.encoder_decoder else None,
            "all_z_grads": all_z_grads if self.encoder_decoder else None,
        }

    @torch.no_grad()
    def _compute_weights_gradients(self, all_grads_covs, products):
        if self.algorithm == "sgd":
            with torch.enable_grad():
                weights = self._normalize_weights(self._unnormalized_weights, pretrain_covariances=all_grads_covs)
            # Normalize products before backward to avoid tiny gradient magnitudes
            # propagating through the normalization graph when gradients are small.
            self.logits.grad = None
            weights.backward(-products)
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
            down_grads = self._grads_cache["encoder"]
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

        all_grads_scale = torch.linalg.norm(all_grads, dim=1).max().clamp(min=self.eps ** 2)
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

        self._compute_weights_gradients(all_grads_covs, products)
        if is_distributed:
            torch.distributed.all_reduce(self.logits.grad, op=torch.distributed.ReduceOp.SUM)
            self.logits.grad /= torch.distributed.get_world_size()

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
            # Cache encoder grads.
            downstream_weight = 1
            loss_weights = torch.zeros_like(self.logits)
            z_down = torch.enable_grad()(closure)(downstream_weight, loss_weights, retain_graph=False, stage=HPO_STAGE_DOWNSTREAM)
            if self.encoder_decoder:
                closure_encoder(z_down.grad.flatten())
            encoder_grads = self._gather_grads("encoder")
            if torch.linalg.norm(encoder_grads) < self.eps:
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

            if (self.algorithm == "sgd") and (self.apply_gradient_normalizer):
                self.hp_gradient_normalizer([self.logits])

            if after_backward_hook is not None:
                after_backward_hook()

            for group in self.param_groups[1:]:
                for p in group["params"]:
                    p.grad = None
        if self.algorithm == "sgd":
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

            n_updates = self._buffers["n_updates"]
            if n_updates > 1:
                self._buffers["avg_n_skipped_steps"] *= (n_updates - 1) / n_updates
                self._buffers["avg_n_skipped_steps"] += self._buffers["n_skipped_steps"] / n_updates
                self._buffers["ema_n_skipped_steps"] *= self.stats_momentum
                self._buffers["ema_n_skipped_steps"] += (1 - self.stats_momentum) * self._buffers["n_skipped_steps"]
            else:
                self._buffers["avg_n_skipped_steps"] = self._buffers["n_skipped_steps"]
                self._buffers["ema_n_skipped_steps"] = self._buffers["n_skipped_steps"]

            # Cache returned value.
            output_weights.copy_(weights)

            # Set gradients for the encoder model weights.
            if self.encoder_decoder:
                # Backprop with z grads. Keep logits grad intact.
                z_grad = sum([w * grads["all_z_grads"][i] for i, w in enumerate(weights)], self.encoder_downstream_weight * grads["z_down_grads"])
                logits_grad = self.logits.grad
                self.logits.grad = None
                closure_encoder(z_grad)
                self.logits.grad = logits_grad
                del z_grad
            else:
                encoder_grad = sum([w * grads["all_encoder_grads"][i] for i, w in enumerate(weights[:-1])], weights[-1] * grads["all_encoder_grads"][-1])
                if self.downstream_merge:
                    encoder_down_grads = grads["encoder_down_grads"]
                    mask = encoder_grad == 0
                    encoder_grad = torch.where(mask, encoder_down_grads, encoder_grad)
                    encoder_down_grads.masked_fill_(mask, 0)
                    del mask
                param_groups = self.param_groups[2:]
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
                self.encoder_gradient_normalizer(itertools.chain(*[group["params"] for group in self.param_groups[2:]]))
                if self.algorithm == "sgd":
                    self.hp_gradient_normalizer([self.logits])

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
        encoder_params = [p for group in self.param_groups[2:] for p in group["params"]]
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
            state["encoder_gradient_normalizer"] = self.encoder_gradient_normalizer.state_dict()
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
            self.encoder_gradient_normalizer.load_state_dict(state_dict["encoder_gradient_normalizer"])
            if self.algorithm == "sgd":
                self.hp_gradient_normalizer.load_state_dict(state_dict["hp_gradient_normalizer"])

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
            param_groups = [self.param_groups[1]]
        else:
            assert part == "encoder"
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
        else:
            assert part == "encoder"
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
