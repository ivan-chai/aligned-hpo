import itertools
import math
import numpy as np
import re
import scipy.optimize
import torch
import warnings
from copy import deepcopy
from numbers import Number

from .gradient import GradientNormalizer
from .solvers import solve_qp


HPO_STAGE_DOWNSTREAM = "downstream"


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
        weights_normalization: Weights normalization type ("sum", "norm", or "none"), or a number to divide weights by.
        downstream_weight: The weight of the downstream gradient in encoder optimization. Default is 0 (disable).
        downstream_merge: Fill zero values in encoder gradient with downsrtream gradient.
        encoder_decoder: Whether to use encoder-decoder decomposition and upper bound optimization for fast hyperparameter tuning.
            This flag affects an intereface of the provided closure. See notes below.
        algorithm: Either "sgd", "mse", "expected-error", or "none" to disable HPO.
        ema: Use momentum for gradient smoothing. Can be a dictionary with "cov", "main", "downstream", "z", and "weights" keys. See notes below.
        tune_on_val: Whether validation batches will be provided or not.
        apply_gradient_normalizer: Normalize gradients using running statistics.
        apply_optimizer_correction: Try to approximate an actual optimizer step rather than simple SGD.
        clip_hp_grad: Clipping value for hyperparameters gradients when "sgd" algorithm is used.
        maxiters: The maximum number of iterations in the QP solver, used for "mse" and "expected-error" algorithms.

    NOTE. Exponential Moving Average (EMA)

    There are multiple smoothing techinques that help to reduce overfitting on the val set.
    Two main parameters: "main" and "downstream", that control smoothing of the pretraining and downstream gradients respectivelly.
    There are some additional smoothing parameters:
    - "cov" can be used to control covariances estimation smoothing. By default, "cov" smoothing is equal to "main".
    - "z" controlls smoothing of the covariances, computed for embedding in the encoder-decoder mode. By default, "z" smoothing is equal to "downstream".
    - "weights" controlls smoothing of weights between batches. It is zero by default.
    - "jacobian" controlls smoothing of the Jacobian norm estimate. By default it is equal to main.

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
        return z.grad  # New.

    def closure_encoder(z_grad):
        optimizer.zero_grad()
        embeddings.backward(z_grad)

    optimizer.hpo_step(closure, closure_encoder)
    ```
    """
    def __init__(self, params, base_optimizer_cls, base_optimizer_params=None, weights_names=None,
                 weights_parametrization="abs", weights_normalization="norm",
                 encoder_downstream_weight=0, shared_downstream_weight=0, downstream_merge=False,
                 encoder_decoder=False, algorithm="expected-error", ema=None, tune_on_val=False,
                 apply_optimizer_correction=False, apply_gradient_normalizer=False, skip_step_zero_weights=True,
                 clip_hp_grad=None, maxiters=100, eps=1e-6):
        if (weights_parametrization == "linear") and (weights_normalization == "sum"):
            raise ValueError("A 'sum' normalization can be applied to positive weights only.")
        params = list(params)
        if len(params) < 3 or not isinstance(params[0], dict) or not isinstance(params[1], dict):
            raise ValueError("Expected at least three param groups with the first group being hyperparameters weights, second group being projectin heads weights, and third group being encoder weights.")
        if (len(params[0]["params"]) != 1) or (params[0]["params"][0].ndim != 1):
            raise ValueError("Weights must be flat.")
        if algorithm not in {"sgd", "dot", "mse", "expected-error", "none"}:
            raise ValueError(f"Unexpected algorithm: {algorithm}")
        if weights_parametrization not in ["linear", "abs"]:
            raise ValueError(f"Unknown weights parametrization method: {weights_parametrization}")
        if weights_normalization not in ["sum", "norm", "none"] and not isinstance(weights_normalization, Number):
            raise ValueError(f"Unknown weights normalization method: {weights_normalization}")
        defaults = dict(base_optimizer_params or {})
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **(base_optimizer_params or {}))
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.n_weights = len(self.param_groups[0]["params"][0])
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
        self.skip_step_zero_weights = skip_step_zero_weights

        if ema is None:
            ema = {"main": 0, "downstream": 0, "cov": 0.9}
        elif isinstance(ema, Number):
            ema = {k: ema for k in ["cov", "main", "downstream"]}
        elif ("main" not in ema) or ("downstream" not in ema):
            raise ValueError(f"ema: expected dictionary with 'main', 'downstream' and optional 'cov' keys.")
        if "cov" not in ema:
            ema["cov"] = ema["main"]
        if "z" not in ema:
            ema["z"] = ema["downstream"]
        if "weights" not in ema:
            ema["weights"] = 0
        if "jacobian" not in ema:
            ema["jacobian"] = ema["main"]
        unknown_keys = set(ema) - {"main", "downstream", "z", "cov", "weights", "jacobian"}
        if unknown_keys:
            raise ValueError(f"Unknown EMA keys: {unknown_keys}")
        self.downstream_momentum = ema["downstream"]
        self.main_momentum = ema["main"]
        self.cov_momentum = ema["cov"]
        self.z_momentum = ema["z"]
        self.weights_momentum = ema["weights"]
        self.jacobian_momentum = ema["jacobian"]

        self.tune_on_val = tune_on_val

        self.apply_optimizer_correction = apply_optimizer_correction
        self.apply_gradient_normalizer = apply_gradient_normalizer
        if apply_gradient_normalizer:
            self.heads_gradient_normalizer = GradientNormalizer(clip=1e-6, momentum=self.cov_momentum)
            self.shared_gradient_normalizer = GradientNormalizer(clip=1e-6, momentum=self.cov_momentum)
        self.clip_hp_grad = clip_hp_grad
        self.maxiters = maxiters
        self.eps = eps

        # TODO: use optimizer state for gradient caches.
        self._grads_cache = {HPO_STAGE_DOWNSTREAM: None} | {i: None for i in range(self.n_weights)}
        if tune_on_val and (self.algorithm not in {"sgd", "none"}):
            self._grads_cache.update({"z_C": None, "z_b": None})
        if algorithm == "expected-error":
            self._grads_cache.update({f"cov_{i}": None for i in range(self.n_weights)})
        if encoder_decoder:
            self._grads_cache["jacobian"] = None
        with torch.no_grad():
            self._grads_cache["weights"] = self.weights.clone()
        self._buffers = {
            "n_updates": 0,
            "weights": None,
            "ema_weights": None,
            "avg_weights": None
        }

        if self.algorithm != "none":
            self._buffers["correlations"] = None
            self._buffers["ema_correlations"] = None
            self._buffers["avg_correlations"] = None

    @property
    def unnormalized_weights(self):
        logits = self.param_groups[0]["params"][0]
        if self.weights_parametrization == "abs":
            weights = torch.abs(logits)
        elif self.weights_parametrization == "linear":
            weights = logits
        else:
            raise RuntimeError(f"Unknown parametrization: {self.weights_parametrization}")
        return weights

    @property
    def weights(self):
        return self._normalize_weights(self.unnormalized_weights)

    def step(self, closure=None, *, inner=False):
        if not inner:
            raise ValueError("Please, use 'hpo_step' function.")
        self.base_optimizer.step(closure=closure)

    def _update_grads_cache_impl(self, key, value, momentum=0):
        if self._grads_cache[key] is None:
            self._grads_cache[key] = value
        else:
            if momentum > 0:
                value = self._grads_cache[key] * momentum + value * (1 - momentum)
            self._grads_cache[key] = value

    def _update_grads_cache(self, grads, stage=None):
        return_smoothed = True
        if stage not in self._grads_cache:
            raise ValueError(f"Unknown stage: {stage}")
        if stage == HPO_STAGE_DOWNSTREAM:
            momentum = self.downstream_momentum
        elif stage in {"z_C", "z_b"}:
            momentum = self.z_momentum
        elif stage == "weights":
            momentum = self.weights_momentum
        elif stage == "jacobian":
            momentum = self.jacobian_momentum
        else:
            assert isinstance(stage, Number)
            momentum = self.main_momentum
            if (momentum == 0) and (self.algorithm == "expected-error"):
                # Smooth covariance estimation, but not the value.
                momentum = self.cov_momentum
                return_smoothed = False

        # Update covs.
        cov_key = f"cov_{stage}"
        if (self.algorithm == "expected-error") and (cov_key in self._grads_cache):
            if self._grads_cache[stage] is not None:
                delta = grads - self._grads_cache[stage]
                delta_sq = torch.linalg.norm(delta).square()
            else:
                delta_sq = torch.zeros([], dtype=grads.dtype, device=grads.device)
            self._update_grads_cache_impl(cov_key, delta_sq, momentum=self.cov_momentum)

        # Update means.
        self._update_grads_cache_impl(stage, grads, momentum=momentum)
        if return_smoothed:
            return self._grads_cache[stage]
        else:
            return grads

    @property
    def metrics(self):
        result = {}
        for k, v in self._grads_cache.items():
            if not isinstance(k, int) and k.startswith("cov_") and (v is not None):
                res = re.match(r"cov_([0-9]+)", k)
                if res:
                    k = f"cov_{self.weights_names[int(res.group(1))]}"
                result[k] = v.mean().item()
        if self._grads_cache.get("jacobian", None) is not None:
            result["jacobian_norm"] = self._grads_cache["jacobian"]
        if (self.algorithm not in {"sgd", "none"}) and (self._buffers["correlations"] is not None):
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
        if self.apply_gradient_normalizer:
            if not self.heads_gradient_normalizer.is_first:
                result["heads_gradient_moving_norm"] = self.heads_gradient_normalizer.moving_norm
            if not self.shared_gradient_normalizer.is_first:
                result["shared_gradient_moving_norm"] = self.shared_gradient_normalizer.moving_norm
        return result

    def _normalize_weights(self, weights):
        if torch.linalg.norm(weights) < self.eps:
            return weights
        if isinstance(self.weights_normalization, Number):
            return weights / self.weights_normalization
        elif self.weights_normalization == "sum":
            return self.n_weights * weights / (weights.sum() + self.eps)
        elif self.weights_normalization == "norm":
            return math.sqrt(self.n_weights) * weights / (torch.linalg.norm(weights) + self.eps)
        else:
            assert self.weights_normalization == "none"
            return weights

    def val_step(self, closure, after_backward_hook=None):
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
        if not self.tune_on_val:
            if not torch.distributed.is_initialized() or (torch.distributed.get_rank() == 0):
                warnings.warn("Calling val_step, when tune_on_val is disabled.")
            return
        assert closure is not None, "need closure"

        # Cache downstream shared grads.
        downstream_weight = 1
        loss_weights = torch.zeros_like(self.param_groups[0]["params"][0])
        z_down_grads = torch.enable_grad()(closure)(downstream_weight, loss_weights, retain_graph=True, stage=HPO_STAGE_DOWNSTREAM)
        shared_down_grads = self._gather_grads("shared")
        self._update_grads_cache(shared_down_grads, stage=HPO_STAGE_DOWNSTREAM)

        if self.encoder_decoder and (self.algorithm not in {"sgd", "none"}):
            # Cache embeddings covariances.
            if z_down_grads is None:
                raise TypeError("In the encoder-decoder mode, closure must return gradient w.r.t. encoder output.")
            downstream_weight = 0
            all_z_grads = []
            for i in range(self.n_weights):
                loss_weights[i] = 1
                all_z_grads.append(closure(downstream_weight, loss_weights, retain_graph=(i < self.n_weights - 1), stage=i))
                loss_weights[i] = 0
            all_z_grads = torch.stack(all_z_grads)  # (W, D).
            C = all_z_grads @ all_z_grads.T
            b = -(all_z_grads @ z_down_grads)  # (W).
            self._update_grads_cache(C, stage="z_C")
            self._update_grads_cache(b, stage="z_b")

    def hpo_step(self, closure, closure_encoder=None, after_backward_hook=None):
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

        output_weights = torch.empty_like(self.param_groups[0]["params"][0])

        @torch.no_grad()
        def inner_closure():
            logits = self.param_groups[0]["params"][0]
            with torch.enable_grad():
                weights = self.weights

            self._buffers["n_updates"] += 1
            n_updates = self._buffers["n_updates"]
            self._buffers["correlations"] = None

            loss_weights = torch.zeros_like(logits)

            # Below:
            # - heads grads: individual heads weights gradients.
            # - z grads: gradient w.r.t. encoder output.
            # - shared grads: shared decoder weights gradients before EMA.
            # - grads: HPO grads after EMA smoothing used for weights tuning.

            # Compute downstream grads.
            downstream_weight = 1
            z_down_grads = closure(downstream_weight, loss_weights, retain_graph=True, stage=HPO_STAGE_DOWNSTREAM)
            if self.encoder_decoder and (z_down_grads is None):
                raise TypeError("In the encoder-decoder mode, closure must return gradient w.r.t. encoder output.")
            heads_down_grads = self._gather_grads("heads")
            shared_down_grads = self._gather_grads("shared")

            if self.tune_on_val:
                down_grads = self._grads_cache[HPO_STAGE_DOWNSTREAM]
            else:
                down_grads = self._update_grads_cache(shared_down_grads, stage=HPO_STAGE_DOWNSTREAM)
            if self.apply_optimizer_correction:
                down_grads = down_grads.clone()
                self.apply_optimizer_correction_("shared", down_grads)
            if self.encoder_decoder and not self.tune_on_val:
                down_grads = torch.cat([z_down_grads, down_grads])

            # Caches for normalization differentiation.
            compute_products = self.algorithm in {"sgd"}
            if compute_products:
                products = torch.zeros(self.n_weights, dtype=down_grads[0].dtype, device=down_grads[0].device)
            store_all_grads = self.algorithm in {"dot", "mse", "expected-error"}
            if store_all_grads:
                all_grads = []
            store_z_grads = self.encoder_decoder
            if store_z_grads:
                all_z_grads = []
            all_heads_grads = []
            all_shared_grads = []

            # Compute main losses grads.
            downstream_weight = 0
            for i, w in enumerate(weights):
                loss_weights[i] = 1
                z_grads = closure(downstream_weight, loss_weights, retain_graph=(i < self.n_weights - 1), stage=i)
                loss_weights[i] = 0
                if self.encoder_decoder and (z_grads is None):
                    raise TypeError("In the encoder-decoder mode, closure must return gradient w.r.t. encoder output.")
                heads_grads = self._gather_grads("heads")
                shared_grads = self._gather_grads("shared")
                loss_grads = self._update_grads_cache(shared_grads, stage=i)
                if self.apply_optimizer_correction:
                    loss_grads = loss_grads.clone()
                    self.apply_optimizer_correction_("shared", loss_grads)
                loss_grads = torch.cat([z_grads, loss_grads]) if self.encoder_decoder and not self.tune_on_val else loss_grads
                if compute_products:
                    products[i] = down_grads @ loss_grads
                if store_all_grads:
                    all_grads.append(loss_grads)
                if store_z_grads:
                    all_z_grads.append(z_grads)
                all_heads_grads.append(heads_grads)
                all_shared_grads.append(shared_grads)

            if store_all_grads:
                all_grads = torch.stack(all_grads, 0)  # (W, P).

            if self.algorithm == "sgd":
                self._buffers["correlations"] = products.detach()
                actual_weights = weights

                logits.grad = None
                actual_weights.backward(-products)
                if self.clip_hp_grad is not None:
                    grad_norm = torch.linalg.norm(logits.grad)
                    if grad_norm > self.clip_hp_grad:
                        logihts.grad *= self.clip_hp_grad / (grad_norm + self.eps)
            elif self.algorithm == "dot":
                if self.weights_parametrization != "abs" or self.weights_normalization != "sum":
                    raise NotImplementedError(f"{self.weights_parametrization} {self.weights_normalization}")

                b = -(all_grads @ down_grads)  # (W).

                self._buffers["correlations"] = -b.detach()

                actual_weights = torch.from_numpy(scipy.optimize.linprog(b.float().cpu().numpy(), A_eq=np.ones([1, len(b)]), b_eq=np.ones([1])).x).float().to(b.device).to(b.dtype)

                actual_weights = self._normalize_weights(actual_weights)
            elif self.algorithm in {"mse", "expected-error"}:
                if self.weights_parametrization == "abs":
                    positive = True
                else:
                    assert self.weights_parametrization == "linear"
                    positive = False

                C = all_grads @ all_grads.T  # (W, W).
                b = -(all_grads @ down_grads)  # (W).

                if self.tune_on_val and self.encoder_decoder:
                    jacobian_norm = self._grads_cache["jacobian"]
                    if jacobian_norm is None:
                        jacobian_norm = 1
                    else:
                        jacobian_norm = jacobian_norm.clip(min=self.eps)
                    C = C + jacobian_norm * self._grads_cache["z_C"]
                    b = b + jacobian_norm * self._grads_cache["z_b"]

                self._buffers["correlations"] = -b.detach()

                if self.algorithm == "expected-error":
                    all_grads_covs = [self._grads_cache[f"cov_{i}"] for i in range(self.n_weights)]
                    if any([c is None for c in all_grads_covs]):
                        all_grads_covs = torch.zeros_like(all_grads[:, 0])  # (W).
                    else:
                        all_grads_covs = torch.stack(all_grads_covs)  # (W).
                    C = C + torch.diag(all_grads_covs)
                else:
                    assert self.algorithm == "mse"

                actual_weights = solve_qp(C, b, positive=positive,
                                          maxiters=self.maxiters, eps=self.eps)
                actual_weights = self._normalize_weights(actual_weights)
            else:
                assert self.algorithm == "none"
                actual_weights = torch.ones_like(weights)
                actual_weights = self._normalize_weights(actual_weights)

            actual_weights = self._update_grads_cache(actual_weights, stage="weights")
            output_weights.copy_(actual_weights)

            if self._buffers["correlations"] is not None:
                if n_updates > 1:
                    self._buffers["avg_correlations"] *= (n_updates - 1) / n_updates
                    self._buffers["avg_correlations"] += self._buffers["correlations"] / n_updates
                    self._buffers["ema_correlations"] *= self.cov_momentum
                    self._buffers["ema_correlations"] += (1 - self.cov_momentum) * self._buffers["correlations"]
                else:
                    self._buffers["avg_correlations"] = self._buffers["correlations"]
                    self._buffers["ema_correlations"] = self._buffers["correlations"]

            self._buffers["weights"] = actual_weights.detach()
            if n_updates > 1:
                self._buffers["avg_weights"] *= (n_updates - 1) / n_updates
                self._buffers["avg_weights"] += self._buffers["weights"] / n_updates
                self._buffers["ema_weights"] *= self.cov_momentum
                self._buffers["ema_weights"] += (1 - self.cov_momentum) * self._buffers["weights"]
            else:
                self._buffers["avg_weights"] = self._buffers["weights"]
                self._buffers["ema_weights"] = self._buffers["weights"]

            # Set hyperparameters and their grads.
            if self.algorithm == "sgd":
                assert logits.grad is not None
            else:
                if torch.distributed.is_available() and torch.distributed.is_initialized() and (torch.distributed.get_world_size() > 1):
                    # Synchronize weights.
                    torch.distributed.all_reduce(actual_weights, op=torch.distributed.ReduceOp.SUM)
                    actual_weights /= torch.distributed.get_world_size()
                logits.grad = None

            # Set gradients for model weights.
            if self.encoder_decoder:
                # Set grads for the encoder (backbone) model. Keep logits grad intact.
                z_grad = sum([w * all_z_grads[i] for i, w in enumerate(actual_weights)], self.encoder_downstream_weight * z_down_grads)
                logits_grad = logits.grad
                logits.grad = None
                closure_encoder(z_grad)
                logits.grad = logits_grad
                z_grad_norm = torch.linalg.norm(z_grad)
                if z_grad_norm > self.eps:
                    encoder_grad = self._gather_grads("encoder")
                    jacobian_norm = torch.linalg.norm(encoder_grad) / z_grad_norm
                    self._update_grads_cache(jacobian_norm, stage="jacobian")
                    del encoder_grad
                del z_grad

            # Set grads for the shared model.
            shared_grad = sum([w * all_shared_grads[i] for i, w in enumerate(actual_weights[:-1])], actual_weights[-1] * all_shared_grads[-1])
            if self.downstream_merge:
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
                shared_decoder = i == 0
                for p in group["params"]:
                    numel = p.numel()
                    p.grad = shared_grad[offset:offset + numel].reshape(p.shape)
                    if downstream_weight > 0:
                        p.grad += downstream_weight * shared_down_grads[offset:offset + numel].reshape(p.shape)
                    offset += numel
            assert offset == len(shared_grad)
            del shared_grad

            # Set grads for individual heads model.
            heads_grad = sum(all_heads_grads, heads_down_grads)
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

            if after_backward_hook is not None:
                after_backward_hook()

            if (self.algorithm != "sgd") and self.skip_step_zero_weights and (torch.linalg.norm(actual_weights) < self.eps):
                raise ZeroWeightsException()
        try:
            self.step(inner_closure, inner=True)
        except ZeroWeightsException:
            pass
        return output_weights

    def state_dict(self):
        state = self.base_optimizer.state_dict()
        state["grads_cache"] = dict(self._grads_cache)
        state["buffers"] = dict(self._buffers)
        if self.apply_gradient_normalizer:
            state["heads_gradient_normalizer"] = self.heads_gradient_normalizer.state_dict()
            state["shared_gradient_normalizer"] = self.shared_gradient_normalizer.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)
        self.param_groups = self.base_optimizer.param_groups
        p = self.param_groups[0]["params"][0]
        self._grads_cache.update({k: (v.to(device=p.device, dtype=p.dtype) if v is not None else None)
                                  for k, v in state_dict.get("grads_cache", {}).items()})
        self._buffers.update({k: (v.to(device=p.device, dtype=p.dtype) if isinstance(v, torch.Tensor) else v)
                              for k, v in state_dict.get("buffers", {}).items()})
        if self.apply_gradient_normalizer:
             self.heads_gradient_normalizer.load_state_dict(state["heads_gradient_normalizer"])
             self.shared_gradient_normalizer.load_state_dict(state["shared_gradient_normalizer"])

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
            return torch.zeros_like(self.param_groups[0]["params"][0][:0])
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
