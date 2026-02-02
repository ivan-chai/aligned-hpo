import math
import re
import numpy as np
import torch
import torch.nn.functional as F
import warnings
from copy import deepcopy
from numbers import Number

from .solvers import solve_qp


HPO_STAGE_DOWNSTREAM = "downstream"


class AlignedHPOptimizer(torch.optim.Optimizer):
    """Aligned Hyperparameter Optimizer.

    Args:
        params: Model parameters with 3 or more groups for encoder-decoder mode and 2 or more groups otherwise.
             The first group is for loss weights. In the encoder-decoder mode, second group is responsible for the
             decoder part of the model.
        base_optimizer_cls: The optimizer to use.
        base_optimizer_params: Parameters of the base optimizer.
        weights_names: An optional list of names for hyperparameters (for logging).
        downstream_weight: The weight of the downstream loss in the backbone model optimization.
        weights_parametrization: Either "linear" or "abs".
        weights_normalization: Weights normalization type ("sum", "norm", or "none"), or a number to divide weights by.
        weights_smoothing: Mix weights with uniform distribution with a given weight.
        encoder_decoder: Whether to use encoder-decoder decomposition and upper bound optimization for fast hyperparameter tuning.
            This flag affects an intereface of the provided closure. See notes below.
        algorithm: Either "sgd", "mse", "expected-error", or "none" to disable HPO.
        ema: Use momentum for gradient smoothing. Can be a dictionary with "cov", "main", "downstream", "z", and "weights" keys. See notes below.
        tune_on_val: Whether validation batches will be provided or not.
        apply_optimizer_correction: Try to approximate an actual optimizer step rather than simple SGD.
        clip_hp_grad: Clipping value for hyperparameters gradients when "sgd" algorithm is used.
        kwargs: Base optimizer parameters.

    NOTE. Exponential Moving Average (EMA)

    There are multiple smoothing techinques that help to reduce overfitting on the val set.
    Two main parameters: "main" and "downstream", that control smoothing of the pretraining and downstream gradients respectivelly.
    There are some additional smoothing parameters:
    - "cov" can be used to control covariances estimation smoothing. By default, "cov" smoothing is equal to "main".
    - "z" controlls smoothing of the covariances, computed for embedding in the encoder-decoder mode. By default, "z" smoothing is equal to "downstream".
    - "weights" controlls smoothing of weights between batches. It is zero by default.

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
                 weights_parametrization="abs", weights_normalization="norm", weights_smoothing=0,
                 encoder_decoder=False, algorithm="expected-error", ema=0, downstream_weight=0, tune_on_val=False,
                 apply_optimizer_correction=False, clip_hp_grad=None, eps=1e-6, save_grad_params = None):
        if (weights_parametrization == "linear") and (weights_normalization == "sum"):
            raise ValueError("A 'sum' normalization can be applied to positive weights only.")
        if (algorithm == "sgd") and encoder_decoder:
            raise NotImplementedError("SGD optimization can't be used with an encoder-decoder architecture.")
        params = list(params)
        if len(params) < 3 or not isinstance(params[0], dict) or not isinstance(params[1], dict):
            raise ValueError("Expected at least three param groups with the first group being hyperparameters weights, second group being projectin heads weights, and third group being encoder weights.")
        if (len(params[0]["params"]) != 1) or (params[0]["params"][0].ndim != 1):
            raise ValueError("Weights must be flat.")
        if algorithm not in {"sgd", "mse", "expected-error", "none"}:
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
        self.weights_smoothing = weights_smoothing
        self.encoder_decoder = encoder_decoder
        self.algorithm = algorithm

        if isinstance(ema, Number):
            ema = {k: ema for k in ["cov", "main", "downstream"]}
        if ("main" not in ema) or ("downstream" not in ema):
            raise ValueError(f"ema: expected dictionary with 'main', 'downstream' and optional 'cov' keys.")
        if "cov" not in ema:
            ema["cov"] = ema["main"]
        if "z" not in ema:
            ema["z"] = ema["downstream"]
        if "weights" not in ema:
            ema["weights"] = 0
        unknown_keys = set(ema) - {"main", "downstream", "z", "cov", "weights"}
        if unknown_keys:
            raise ValueError(f"Unknown EMA keys: {unknown_keys}")
        self.downstream_momentum = ema["downstream"]
        self.main_momentum = ema["main"]
        self.cov_momentum = ema["cov"]
        self.z_momentum = ema["z"]
        self.weights_momentum = ema["weights"]

        self.downstream_weight = downstream_weight
        self.tune_on_val = tune_on_val

        self.apply_optimizer_correction = apply_optimizer_correction
        self.clip_hp_grad = clip_hp_grad
        self.eps = eps

        # TODO: use optimizer state for gradient caches.
        self._grads_cache = {HPO_STAGE_DOWNSTREAM: None} | {i: None for i in range(self.n_weights)}
        if tune_on_val and self.algorithm != "sgd":
            self._grads_cache.update({"z_C": None, "z_b": None})
        if algorithm == "expected-error":
            self._grads_cache.update({f"cov_{i}": None for i in range(self.n_weights)})
        self._grads_cache["weights"] = None

        #Grad caches
        self.save_grad_params = save_grad_params
        if save_grad_params:
            self._grads_cache["avg_grad_down"] = None
            for i in range(self.n_weights):
                self._grads_cache[f"avg_grad_{i}"] = None
            self._grads_cache["step"] = 0

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
        else:
            assert isinstance(stage, Number)
            momentum = self.main_momentum
            if (momentum == 0) and (self.algorithm == "expected-error"):
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
        return result

    def _normalize_weights(self, weights):
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
        shared_down_grads = self._gather_grads(gather_heads=False)
        self._update_grads_cache(shared_down_grads, stage=HPO_STAGE_DOWNSTREAM)

        if self.encoder_decoder and (self.algorithm != "sgd"):
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
            if self.weights_parametrization == "abs":
                weights = torch.abs(logits)
            else:
                weights = logits
                assert self.weights_parametrization == "linear"

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
            heads_down_grads = self._gather_grads(gather_heads=True)
            shared_down_grads = self._gather_grads(gather_heads=False)

            if self.tune_on_val:
                down_grads = self._grads_cache[HPO_STAGE_DOWNSTREAM]
            else:
                down_grads = self._update_grads_cache(shared_down_grads, stage=HPO_STAGE_DOWNSTREAM)
            if self.encoder_decoder and not self.tune_on_val:
                down_grads = torch.cat([z_down_grads, down_grads])

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
                heads_grads = self._gather_grads(gather_heads=True, apply_optimizer_correction=self.apply_optimizer_correction)
                shared_grads = self._gather_grads(gather_heads=False, apply_optimizer_correction=self.apply_optimizer_correction)
                loss_grads = self._update_grads_cache(shared_grads, stage=i)
                loss_grads = torch.cat([z_grads, loss_grads]) if self.encoder_decoder and not self.tune_on_val else loss_grads
                if compute_products:
                    products[i] = down_grads @ loss_grads
                if compute_grad_sum:
                    grad_sum += loss_grads * w
                if store_all_grads:
                    all_grads.append(loss_grads)
                if store_z_grads:
                    all_z_grads.append(z_grads)
                all_heads_grads.append(heads_grads)
                all_shared_grads.append(shared_grads)

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

                C = all_grads @ all_grads.T  # (W, W).
                b = -(all_grads @ down_grads)  # (W).
                if self.save_grad_params:
                    begin_step = self.save_grad_params.get('begin', 0)
                    end_step = self.save_grad_params.get('end', 1000)
                    file_name = self.save_grad_params.get('name', 'grad_logs.ckpt')
                    
                    self._grads_cache["step"] += 1
                    step = self._grads_cache["step"]
                    
                    if step != 1 and begin_step > step:
                        self._grads_cache["avg_grad_down"] = (
                            self._grads_cache["avg_grad_down"].detach().cpu() * step / (step + 1)
                            + F.normalize(down_grads.detach().cpu(), dim=0) / (step + 1)
                        )
                    elif begin_step >= step:
                        self._grads_cache["avg_grad_down"] = F.normalize(
                            down_grads.detach().cpu(), dim=0
                        )
                    
                    for i, grad in enumerate(all_shared_grads):
                        if step != 1 and begin_step > step:
                            self._grads_cache[f"avg_grad_{i}"] = (
                                self._grads_cache[f"avg_grad_{i}"].detach().cpu() * step / (step + 1)
                                + F.normalize(grad.detach().cpu(), dim=0) / (step + 1)
                            )
                        elif begin_step >= step:
                            self._grads_cache[f"avg_grad_{i}"] = F.normalize(
                                grad.detach().cpu(), dim=0
                            )
                    
                    
                    if step > end_step + begin_step:
                        torch.save(self._grads_cache, file_name)
                        print('Directions saved')
                        exit(0)

                if self.tune_on_val and self.encoder_decoder:
                    C = C + self._grads_cache["z_C"]
                    b = b + self._grads_cache["z_b"]

                if self.algorithm == "expected-error":
                    all_grads_covs = [self._grads_cache[f"cov_{i}"] for i in range(self.n_weights)]
                    if any([c is None for c in all_grads_covs]):
                        all_grads_covs = torch.zeros_like(all_grads[:, 0])  # (W).
                    else:
                        all_grads_covs = torch.stack(all_grads_covs)  # (W).
                    C = C + torch.diag(all_grads_covs)
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

            actual_weights = self._update_grads_cache(actual_weights, stage="weights")
            output_weights.copy_(actual_weights)

            # Set hyperparameters and their grads.
            if self.algorithm == "sgd":
                self.param_groups[0]["params"][0].grad = weight_grads
            else:
                self.param_groups[0]["params"][0].data.copy_(actual_weights)
                self.param_groups[0]["params"][0].grad = None
            # Set gradients for model weights.
            if self.encoder_decoder:
                # Set grads for the encoder (backbone) model.
                z_grad = sum([w * all_z_grads[i] for i, w in enumerate(actual_weights[:-1])], actual_weights[-1] * all_z_grads[-1])
                closure_encoder(z_grad)
                del z_grad

            # Set grads for the shared model.
            shared_grad = shared_down_grads
            if self.encoder_decoder:
                param_groups = [self.param_groups[2]]
            else:
                param_groups = self.param_groups[2:]
            offset = 0
            for group in param_groups:
                for p in group["params"]:
                    numel = p.numel()
                    p.grad = shared_grad[offset:offset + numel].reshape(p.shape)
                    offset += numel
            assert offset == len(shared_grad)
            del shared_grad

            # Set grads for individual heads model.
            heads_grad = sum([all_heads_grads[i] for i, w in enumerate(actual_weights)], heads_down_grads)
            offset = 0
            for p in self.param_groups[1]["params"]:
                numel = p.numel()
                p.grad = heads_grad[offset:offset + numel].reshape(p.shape)
                offset += numel
            assert offset == len(heads_grad)
            del heads_grad
            if after_backward_hook is not None:
                after_backward_hook()

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

    def _gather_grads(self, gather_heads, apply_optimizer_correction=False):
        if gather_heads:
            param_groups = [self.param_groups[1]]
        elif self.encoder_decoder:
            param_groups = [self.param_groups[2]]
        else:
            # All except hyperparameters and individual heads.
            param_groups = self.param_groups[2:]
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
        if not grads:
            return torch.zeros_like(self.param_groups[0]["params"][0][:0])
        return torch.cat(grads)
