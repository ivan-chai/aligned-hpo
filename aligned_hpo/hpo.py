import math
import re
import numpy as np
import torch
from copy import deepcopy
from numbers import Number

from .solvers import solve_qp


HPO_STAGE_DOWNSTREAM = "downstream"
HPO_STAGE_BACKBONE = "backbone"


class AlignedHPOptimizer(torch.optim.Optimizer):
    """Aligned Hyperparameter Optimizer.

    Args:
        params: Model parameters with 3 or more groups for encoder-decoder mode and 2 or more groups otherwise.
             The first group is for loss weights. In the encoder-decoder mode, second group is responsible for the
             decoder part of the model.
        base_optimizer_cls: The optimizer to use.
        base_optimizer_params: Parameters of the base optimizer.
        names: An optional list of names for hyperparameters (for logging).
        downstream_weight: The weight of the downstream loss in the backbone model optimization or "merge".
            The "merge" value means inserting downstream gradients for the weights, not updated by the main loss.
        weights_parametrization: Either "linear" or "abs".
        weights_normalization: Weights normalization type ("sum", "norm", or "none").
        weights_smoothing: Mix weights with uniform distribution with a given weight.
        encoder_decoder: Whether to use encoder-decoder decomposition and upper bound optimization for fast hyperparameter tuning.
            This flag affects an intereface of the provided closure. See notes below.
        algorithm: Either "sgd", "mse", "expected-error", or "none" to disable HPO.
        ema: Use momentum for gradient smoothing. Can be a dictionary with "main" and "downstream" keys
            for the main and downstream losses respectively.
        apply_optimizer_correction: Try to approximate an actual optimizer step rather than simple SGD.
        clip_hp_grad: Clipping value for hyperparameters gradients when "sgd" algorithm is used.
        kwargs: Base optimizer parameters.

    NOTE. Encoder-Decoder vs full gradients.

    In a simple "full" approach, closure must compute gradients for all model parameters, leading to multiple backward passes at each step.
    A more effective method decomposes the model into encoder and decoder part. A closure must be able to compute gradients w.r.t. to the
    embedding output of the encoder. A separate step is performed to pass aggregated gradient to the encoder part of the model.
    See examples below.

    Example usage (full gradients):
    ```
    optimizer = CorrHPOptimizer([{"params": [weights]},  # Weights for tuning.
                                 {"params": model.parameters()}],
                                torch.optim.Adam,
                                lr=0.01)

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
    optimizer = CorrHPOptimizer([{"params": [weights]},  # Weights for tuning.
                                 {"params": model.decoder.parameters()},   # Decoder.
                                 {"params": model.encoder.parameters()}],  # Encoder.
                                torch.optim.Adam,
                                lr=0.01)

    embeddings = model.encode(x)
    z = embeddings.detach().clone()
    z.requires_grad = True
    output = model.decode(z)
    down_loss, loss1, loss2 = criterion(output, y)

    def closure(down_weight, weights, retain_graph=False, stage=None):
        optimizer.zero_grad()
        loss = down_weight * down_loss + weights[0] * loss1 + weights[1] * loss2
        loss.backward(retain_graph=retain_graph)
        return z.grad  # New.

    def closure_encoder(z_grad):
        optimizer.zero_grad()
        embeddings.backward(z_grad)

    optimizer.hpo_step(closure, closure_encoder)
    ```
    """
    def __init__(self, params, base_optimizer_cls, base_optimizer_params=None, names=None,
                 downstream_weight="merge", weights_parametrization="abs", weights_normalization="norm", weights_smoothing=0,
                 encoder_decoder=False, algorithm="expected-error", ema=0, apply_optimizer_correction=False,
                 clip_hp_grad=None, eps=1e-6):
        if (weights_parametrization == "linear") and (weights_normalization == "sum"):
            raise ValueError("A 'sum' normalization can be applied to positive weights only.")
        params = list(params)
        if encoder_decoder:
            if len(params) < 3 or not isinstance(params[0], dict) or not isinstance(params[1], dict):
                raise ValueError("Expected at least three param groups with the first group being hyperparameters weights and second group being decoder weights.")
        else:
            if len(params) < 2 or not isinstance(params[0], dict):
                raise ValueError("Expected at least two param groups with the first group being hyperparameters weights.")
        if (len(params[0]["params"]) != 1) or (params[0]["params"][0].ndim != 1):
            raise ValueError("Weights must be flat.")
        if algorithm not in {"sgd", "mse", "expected-error", "none"}:
            raise ValueError(f"Unexpected algorithm: {algorithm}")
        if weights_parametrization not in ["linear", "abs"]:
            raise ValueError(f"Unknown weights parametrization method: {weights_parametrization}")
        if weights_normalization not in ["sum", "norm", "none"]:
            raise ValueError(f"Unknown weights normalization method: {weights_normalization}")
        defaults = dict(base_optimizer_params or {})
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **(base_optimizer_params or {}))
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.n_weights = len(self.param_groups[0]["params"][0])
        if names is None:
            names = [str(i) for i in range(self.n_weights)]
        elif len(names) != self.n_weights:
            raise ValueError("Names and weights lengths mismatch")
        if self.n_weights == 0:
            raise ValueError("Empty hyperparameters list.")
        if downstream_weight == "merge":
            self.downstream_merge = True
            self.downstream_weight = 0
        else:
            self.downstream_merge = False
            self.downstream_weight = float(downstream_weight)
        self.weights_parametrization = weights_parametrization
        self.weights_normalization = weights_normalization
        self.weights_smoothing = weights_smoothing
        self.encoder_decoder = encoder_decoder
        self.algorithm = algorithm
        if isinstance(ema, Number):
            ema = {k: ema for k in ["main", "downstream"]}
        if ("main" not in ema) or ("downstream" not in ema):
            raise ValueError(f"ema: expected dictionary with 'main' and 'downstream' keys.")
        self.main_momentum = ema["main"]
        self.downstream_momentum = ema["downstream"]

        self.apply_optimizer_correction = apply_optimizer_correction
        self.clip_hp_grad = clip_hp_grad
        self.eps = eps

        # TODO: use optimizer state for gradient caches.
        self._grads_cache = {HPO_STAGE_DOWNSTREAM: None} | {i: None for i in range(self.n_weights)}
        if algorithm == "expected-error":
            self._grads_cache.update({f"cov_{i}": None for i in range(self.n_weights)})

    def step(self, closure, *, inner=False):
        if not inner:
            raise ValueError("Please, use 'hpo_step' function.")
        self.base_optimizer.step(closure)

    def _update_grads_cache(self, grads, stage=None):
        if stage not in self._grads_cache:
            raise ValueError(f"Unknown stage: {stage}")
        if stage == HPO_STAGE_DOWNSTREAM:
            momentum = self.downstream_momentum
        else:
            assert isinstance(stage, Number)
            momentum = self.main_momentum

        # Update covs.
        cov_key = f"cov_{stage}"
        if (self.algorithm == "expected-error") and (cov_key in self._grads_cache):
            if self._grads_cache[stage] is not None:
                delta_sq = torch.linalg.norm(grads - self._grads_cache[stage]).square()
            else:
                delta_sq = torch.zeros_like(grads[0])
            if (self._grads_cache[cov_key] is not None) and momentum:
                self._grads_cache[cov_key] = self._grads_cache[cov_key] * momentum + delta_sq * (1 - momentum)
            else:
                self._grads_cache[cov_key] = delta_sq

        # Update means.
        if (self._grads_cache[stage] is not None) and momentum:
            self._grads_cache[stage] = self._grads_cache[stage] * momentum + grads * (1 - momentum)
        else:
            self._grads_cache[stage] = grads
        return self._grads_cache[stage]

    def cache_downstream(self, closure=None):
        """Cache downstream gradient for future computations.

        Use this method to tune hyperparamers on validation batches.
        """
        assert closure is not None, "need closure"
        downstream_weight = 1
        loss_weights = torch.zeros_like(self.param_groups[0]["params"][0])
        result = torch.enable_grad()(closure)(downstream_weight, loss_weights, retain_graph=False, stage=HPO_STAGE_DOWNSTREAM)
        if self.encoder_decoder:
            if result is None:
                raise RuntimeError("In the encoder-decoder mode, closure must return gradient w.r.t. encoder output.")
            grads = result
        else:
            grads = self._gather_grads(stage=HPO_STAGE_DOWNSTREAM)
        self._update_grads_cache(grads, stage=HPO_STAGE_DOWNSTREAM)

    def remove_cache(self, stage=None):
        if stage is None:
            self._grads_cache = {name: None for name in self._grads_cache}
        else:
            self._grads_cache[stage] = None
            key = f"cov_{stage}"
            if key in self._grads_cache:
                self._grads_cache[key] = None

    @property
    def metrics(self):
        result = {}
        for k, v in self._grads_cache.items():
            if not isinstance(k, int) and k.startswith("cov_") and (v is not None):
                result[k] = v.mean().item()
        return result

    def _normalize_weights(self, weights):
        if self.weights_normalization == "sum":
            return self.n_weights * weights / (weights.sum() + self.eps)
        elif self.weights_normalization == "norm":
            return math.sqrt(self.n_weights) * weights / (torch.linalg.norm(weights) + self.eps)
        else:
            assert self.weights_normalization == "none"
            return weights

    def hpo_step(self, closure, closure_encoder=None, use_cached_downstream=False):
        """Make a single step.

        Args:
            use_cached_downstream: Use gradients cache, don't recompute downstream gradients.

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
            if self.weights_parametrization == "abs":
                weights = torch.abs(logits)
            else:
                weights = logits
                assert self.weights_parametrization == "linear"

            loss_weights = torch.zeros_like(logits)

            # Compute downstream grads.
            downstream_weight = 1
            result = closure(downstream_weight, loss_weights, retain_graph=True, stage=HPO_STAGE_DOWNSTREAM)
            full_down_grads = self._gather_grads(stage=HPO_STAGE_DOWNSTREAM)
            if self.encoder_decoder and (result is None):
                raise TypeError("In the encoder-decoder mode, closure must return gradient w.r.t. encoder output.")

            down_grads = result if self.encoder_decoder else full_down_grads
            if use_cached_downstream:
                down_grads = self._grads_cache[HPO_STAGE_DOWNSTREAM]
            else:
                down_grads = self._update_grads_cache(down_grads, stage=HPO_STAGE_DOWNSTREAM)
            assert down_grads is not None

            # Caches for normalization differentiation.
            compute_products = self.algorithm in {"sgd"}
            if compute_products:
                products = torch.zeros(self.n_weights, dtype=down_grads[0].dtype, device=down_grads[0].device)
            compute_grad_sum = (self.algorithm == "sgd") and (self.weights_normalization in {"sum", "norm"})
            if compute_grad_sum:
                grad_sum = torch.zeros_like(down_grads)
            store_all_grads = self.algorithm in {"mse", "expected-error"}
            if store_all_grads:
                all_grads = []
            all_full_grads = []

            # Compute main losses grads.
            downstream_weight = 0
            for i, w in enumerate(weights):
                loss_weights[i] = 1
                result = closure(downstream_weight, loss_weights, retain_graph=(i < self.n_weights - 1), stage=i)
                loss_weights[i] = 0
                if self.encoder_decoder and (result is None):
                    raise TypeError("In the encoder-decoder mode, closure must return gradient w.r.t. encoder output.")
                full_grads = self._gather_grads(stage=i, apply_optimizer_correction=self.apply_optimizer_correction)
                loss_grads = result if self.encoder_decoder else full_grads
                loss_grads = self._update_grads_cache(loss_grads, stage=i)
                if compute_products:
                    products[i] = down_grads @ loss_grads
                if compute_grad_sum:
                    grad_sum += loss_grads * w
                if store_all_grads:
                    all_grads.append(loss_grads)
                all_full_grads.append(full_grads)

            if store_all_grads:
                all_grads = torch.stack(all_grads, 0)  # (W, P).

            if self.algorithm == "sgd":
                if self.weights_normalization == "sum":
                    scale = self.n_weights
                    norm = weights.sum() + self.eps
                    product = down_grads @ grad_sum
                    weight_grads = -products / norm + product / norm ** 2
                    weight_grads *= scale  # Scale by the number of weights.
                elif self.weights_normalization == "norm":
                    scale = math.sqrt(self.n_weights)
                    norm = torch.linalg.norm(weights) + self.eps
                    product = down_grads @ grad_sum
                    weight_grads = -products / norm + weights * product / (norm ** 3)
                    weight_grads *= scale  # Scale by the norm of union vector.
                else:
                    assert self.weights_normalization == "none"
                    scale = 1
                    norm = 1
                    weight_grads = -products
                actual_weights = scale * weights / norm

                if self.weights_parametrization == "abs":
                    weight_grads = torch.where(logits >= 0, weight_grads, -weight_grads)  # torch.sign freezes at zero.
                else:
                    assert self.weights_parametrization == "linear"
                if self.clip_hp_grad is not None:
                    grad_norm = torch.linalg.norm(weight_grads)
                    if grad_norm > self.clip_hp_grad:
                        weight_grads *= self.clip_hp_grad / (grad_norm + self.eps)
            elif self.algorithm in {"mse", "expected-error"}:
                if self.weights_parametrization == "abs":
                    positive = True
                else:
                    assert self.weights_parametrization == "linear"
                    positive = False
                dim = len(down_grads)

                C = all_grads @ all_grads.T  # (W, W).
                b = -(all_grads @ down_grads)  # (W).

                if self.algorithm == "expected-error":
                    all_grads_covs = [self._grads_cache[f"cov_{i}"] for i in range(self.n_weights)]
                    if any([c is None for c in all_grads_covs]):
                        all_grads_covs = torch.zeros_like(all_grads[:, 0])  # (W).
                    else:
                        all_grads_covs = torch.stack(all_grads_covs)  # (W).
                    C = C + 2 * torch.diag(all_grads_covs)
                else:
                    assert self.algorithm == "mse"

                actual_weights = solve_qp(C, b, eps=self.eps, positive=positive)
                if positive and (not actual_weights.isfinite().all()):
                    actual_weights = solve_qp(C, b, eps=self.eps, positive=False).clip(min=0)

                actual_weights = self._normalize_weights(actual_weights)
            else:
                assert self.algorithm == "none"
                actual_weights = self._normalize_weights(actual_weights)

            if self.weights_smoothing > 0:
                if self.algorithm == "sgd":
                    raise NotImplementedError("Weights smoothing for SGD HPO")
                actual_weights = self.weights_smoothing * torch.ones_like(actual_weights) + (1 - self.weights_smoothing) * actual_weights
                actual_weights = self._normalize_weights(actual_weights)


            # Set hyperparameters and their grads.
            if self.algorithm == "sgd":
                self.param_groups[0]["params"][0].grad = weight_grads
            else:
                self.param_groups[0]["params"][0].data.copy_(actual_weights)
                self.param_groups[0]["params"][0].grad = None

            # Set gradients for model weights.
            output_weights.copy_(actual_weights)
            grad = sum([w * all_full_grads[i] for i, w in enumerate(actual_weights)], self.downstream_weight * full_down_grads)
            if self.encoder_decoder:
                # Set grads for the encoder (backbone) model.
                z_grad = sum([w * all_grads[i] for i, w in enumerate(actual_weights)], self.downstream_weight * down_grads)
                closure_encoder(z_grad)

                # Set grads for the decoder model.
                offset = 0
                for p in self.param_groups[1]["params"]:
                    numel = p.numel()
                    p.grad = grad[offset:offset + numel].reshape(p.shape)
                    offset += numel
                assert grad.shape == (offset,)
            else:
                offset = 0
                for group in self.param_groups[1:]:
                    for p in group["params"]:
                        numel = p.numel()
                        p_grad = grad[offset:offset + numel].reshape(p.shape)
                        if (p.grad is None) or (not self.downstream_merge):
                            p.grad = p_grad
                        else:
                            down_p_grad = full_down_grads[offset:offset + numel].reshape(p.shape)
                            p.grad = torch.where(p_grad.abs() > 0, p_grad, down_p_grad)
                        offset += numel
                assert offset == len(grad)

        self.step(inner_closure, inner=True)
        return output_weights

    def state_dict(self):
        state = super().state_dict()
        state["grads_cache"] = dict(self._grads_cache)
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        p = self.param_groups[0]["params"][0]
        self._grads_cache.update({k: (v.to(device=p.device, dtype=p.dtype) if v is not None else None)
                                  for k, v in state_dict.get("grads_cache", {}).items()})

    def _gather_grads(self, stage, apply_optimizer_correction=False):
        if self.encoder_decoder and (stage != HPO_STAGE_BACKBONE):
            param_groups = [self.param_groups[1]]  # Decoder.
        else:
            param_groups = self.param_groups[1:]  # All except hyperparameters.
        grads = []
        for group in param_groups:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p).flatten())
                else:
                    grads.append(p.grad.flatten())
                    p.grad = None
        if apply_optimizer_correction:
            # We don't pass gradient to the velocity vector for simplicity.
            if isinstance(self.base_optimizer, torch.optim.Adam):
                i = 0
                for group in param_groups:
                    _, beta2 = group["betas"]
                    eps = group["eps"]
                    for p in group["params"]:
                        state = self.base_optimizer.state[p]
                        exp_avg_sq = state.get("exp_avg_sq", None)
                        if exp_avg_sq is None:
                            i += 1
                            continue
                        step = state["step"]
                        bias_correction2_sqrt = (1 - beta2 ** step) ** 0.5
                        grads[i] /= exp_avg_sq.sqrt().flatten() / bias_correction2_sqrt + eps
                        i += 1
                assert i == len(grads)
            elif isinstance(self.base_optimizer, torch.optim.SGD):
                pass  # No need for correction.
            else:
                raise NotImplementedError(f"Can't apply correction to {type(self.base_optimizer).__name__}")
        return torch.cat(grads)
