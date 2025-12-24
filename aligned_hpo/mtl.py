import torch


class MultiTaskOptimizer(torch.optim.Optimizer):
    """Multi-task optimizer.

    Args:
        weights_names: A list of losses names (for logging).
        params: Model parameters with 2 or more groups (individual heads, shared decoder, *encoder). Some groups can be empty.
        base_optimizer_cls: The optimizer to use.
        base_optimizer_params: Parameters of the base optimizer.
        encoder_decoder: Whether to use encoder-decoder decomposition and upper bound optimization for fast hyperparameter tuning.
            This flag affects an intereface of the provided closure. See notes below.
        algorithm: TODO.

    NOTE. Encoder-Decoder vs full gradients.

    In a simple (default) approach, closure must compute gradients for all model parameters, leading to multiple backward passes at each step.
    A more effective method decomposes the model into encoder and decoder part. A closure must be able to compute gradients w.r.t. to the
    embedding output of the encoder. A separate step is performed to pass aggregated gradient to the encoder part of the model.
    See examples below.

    Example usage (default, full gradients):
    ```
    optimizer = MultiTaskOptimizer([{"params": heads.parameters()},   # Loss heads parameters.
                                    {"params": model.parameters()}],  # Model parameters (shared decoder).
                                   torch.optim.Adam,
                                   {"lr": 0.01})  # Optimizer parameters.

    output = model(x)
    loss1, loss2 = criterion(output, y)

    def closure(weights, retain_graph=False, stage=None):
        optimizer.zero_grad()
        loss = weights[0] * loss1 + weights[1] * loss2
        loss.backward(retain_graph=retain_graph)

    optimizer.mtl_step(closure)
    ```

    Example usage (encoder-decoder):
    ```
    optimizer = MultiTaskOptimizer([{"params": heads.parameters()},  # Loss heads parameters.
                                    {"params": model.decoder.parameters()},   # Shared decoder parameters (except individual heads), optional.
                                    {"params": model.encoder.parameters()}],  # Encoder.
                                   torch.optim.Adam,
                                   {"lr": 0.01})  # Optimizer parameters.

    embeddings = model.encode(x)
    z = embeddings.detach().clone()
    z.requires_grad = True
    output = model.decode(z)
    loss1, loss2 = criterion(output, y)

    def closure(weights, retain_graph=False, stage=None):
        optimizer.zero_grad()
        loss = weights[0] * loss1 + weights[1] * loss2
        loss.backward(retain_graph=retain_graph)
        return z.grad  # New.

    def closure_encoder(z_grad):
        optimizer.zero_grad()
        embeddings.backward(z_grad)

    optimizer.mtl_step(closure, closure_encoder)
    ```
    """
    def __init__(self, weights_names, params, base_optimizer_cls, base_optimizer_params=None,
                 encoder_decoder=False, algorithm="TODO"):
        params = list(params)
        if len(params) < 2 or not isinstance(params[0], dict) or not isinstance(params[1], dict):
            raise ValueError("Expected at least two param groups with the first group being individual heads parameters and the second group being shared weights.")
        defaults = dict(base_optimizer_params or {})
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **(base_optimizer_params or {}))
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.weights_names = weights_names
        self.n_weights = len(weights_names)
        if self.n_weights == 0:
            raise ValueError("Empty losses list.")
        self.encoder_decoder = encoder_decoder
        self.algorithm = algorithm

    def step(self, closure, *, inner=False):
        if not inner:
            raise ValueError("Please, use 'mtl_step' function.")
        self.base_optimizer.step(closure)

    @property
    def metrics(self):
        """Logging statistics."""
        return {}

    def mtl_step(self, closure, closure_encoder=None, after_backward_hook=None):
        """Make a single step.

        Args:
            closure: A closure to compute inidivdual gradients.
            closure_encoder: A closure to pass embedding gradients to the encoder in the encoder-decoder mode.
            after_backward_hook: A function to call after gradients are estimated (gradient clipping etc.).

        Returns:
            Weights used in current step.

        The closure is used like this: closure(*loss_weights, retain_graph=False, stage=i).
        The closure_encoder is used like this: closure_encoder(encoder_output_grad).

        Each closure must zero grads and compute gradients.
        """
        closure = torch.enable_grad()(closure)  # The closure should do a full forward-backward pass.
        if self.encoder_decoder:
            if closure_encoder is None:
                raise ValueError("Need encoder closure.")
            closure_encoder = torch.enable_grad()(closure_encoder)  # The closure should do a full forward-backward pass.

        p = self._get_some_parameter()
        output_weights = torch.empty(self.n_weights, dtype=p.dtype, device=p.device)

        @torch.no_grad()
        def inner_closure():
            # Compute losses grads.
            loss_weights = torch.zeros_like(output_weights)
            all_z_grads = []
            all_heads_grads = []
            all_shared_grads = []
            for i, w in enumerate(weights):
                loss_weights[i] = 1
                z_grads = closure(downstream_weight, loss_weights, retain_graph=(i < self.n_weights - 1), stage=i)
                loss_weights[i] = 0
                if self.encoder_decoder and (z_grads is None):
                    raise TypeError("In the encoder-decoder mode, closure must return gradient w.r.t. encoder output.")
                heads_grads = self._gather_grads(gather_heads=True)
                shared_grads = self._gather_grads(gather_heads=False)
                all_z_grads.append(z_grads)
                all_heads_grads.append(heads_grads)
                all_shared_grads.append(shared_grads)

            # Update gradients.
            # TODO: PUT YOUR CODE HERE.
            if self.algorithm == "none":
                weights = torch.ones_like(output_weights)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            output_weights.copy_(weights)

            # Set gradients for encoder weights.
            if self.encoder_decoder:
                # Set grads for the encoder (backbone) model.
                z_grad = sum([w * all_z_grads[i] for i, w in enumerate(weights[:-1])], weights[-1] * all_z_grads[-1])
                closure_encoder(z_grad)
                del z_grad

            # Set grads for the shared model.
            shared_grad = sum([w * all_shared_grads[i] for i, w in enumerate(weights[:-1])], weights[-1] * all_shared_grads[-1])
            if self.encoder_decoder:
                param_groups = [self.param_groups[1]]
            else:
                param_groups = self.param_groups[1:]
            offset = 0
            for group in param_groups:
                for p in group["params"]:
                    numel = p.numel()
                    p.grad = shared_grad[offset:offset + numel].reshape(p.shape)
                    offset += numel
            assert offset == len(shared_grad)
            del shared_grad

            # Set grads for individual heads model.
            heads_grad = sum(all_heads_grads[:-1], all_heads_grads[-1])
            offset = 0
            for p in self.param_groups[0]["params"]:
                numel = p.numel()
                p.grad = heads_grad[offset:offset + numel].reshape(p.shape)
                offset += numel
            assert offset == len(heads_grad)
            del heads_grad
            if after_backward_hook is not None:
                after_backward_hook()

        self.step(inner_closure, inner=True)
        return weights

    def state_dict(self):
        state = super().state_dict()
        state["grads_cache"] = dict(self._grads_cache)
        return state

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        p = self._get_some_parameter()
        self._grads_cache.update({k: (v.to(device=p.device, dtype=p.dtype) if v is not None else None)
                                  for k, v in state_dict.get("grads_cache", {}).items()})

    def _get_some_parameter(self):
        return next(p for p in group["params"] for group in self.param_groups)

    def _gather_grads(self, gather_heads):
        if gather_heads:
            param_groups = [self.param_groups[0]]
        elif self.encoder_decoder:
            param_groups = [self.param_groups[1]]
        else:
            param_groups = self.param_groups[1:]
        grads = []
        for group in param_groups:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p).flatten())
                else:
                    grads.append(p.grad.flatten())
                    p.grad = None
        if not grads:
            return torch.zeros_like(self.param_groups[0]["params"][0][:0])
        return torch.cat(grads)
