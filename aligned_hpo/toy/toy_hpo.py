import torch


class NoisyGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, distribution):
        ctx._distribution = distribution
        return src

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output + ctx._distribution.rsample(grad_output.shape), None


class ScaledGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src, scale):
        ctx._scale = scale
        return src

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx._scale, None


class ToyHPORosenbrock:
    def __init__(self, alpha=1, beta=100, positive=True):
        self.alpha = alpha
        self.beta = beta
        self.positive = positive

    @property
    def n_params(self):
        return 2

    @property
    def n_pretrain_weights(self):
        return 3 if self.positive else 2

    @property
    def solution(self):
        return torch.tensor([self.alpha, self.alpha ** 2]).float()

    @property
    def show_log(self):
        return True

    @property
    def show_range(self):
        rx = (-2 * self.alpha, 2 * self.alpha)
        ry = (-1, 2 + self.alpha ** 2)
        return (rx, ry)

    def loss_pretrain(self, params, weights):
        x, y = params
        if self.positive:
            alpha, beta, gamma = weights
            return 2 * alpha * x + 2 * beta * y - gamma * x - gamma * y
        else:
            alpha, beta = weights
            return - alpha * x + beta * y

    def loss_downstream(self, params):
        x, y = params
        return (self.alpha - x) ** 2 + self.beta * (y - x ** 2) ** 2


class ToyHPOLogRosenbrock(ToyHPORosenbrock):
    @property
    def show_log(self):
        return False

    def loss_downstream(self, params):
        return (super().loss_downstream(params) + 1e-6).log()


class ToyHPOQuadratic:
    def __init__(self, center=(0.5, 1), cov=((1, 0.5), (0.5, 1)), positive=True):
        self.center = torch.as_tensor(center).float()
        self.cov = torch.as_tensor(cov).float()
        self.positive = positive
        self.dim = len(self.center)
        assert self.center.shape == (self.dim,)
        assert self.cov.shape == (self.dim, self.dim)

    @property
    def n_params(self):
        return self.dim

    @property
    def n_pretrain_weights(self):
        return self.dim + 1 if self.positive else self.dim

    @property
    def solution(self):
        return self.center

    @property
    def show_log(self):
        return True

    @property
    def show_range(self):
        base_ranges = [(self.center[i] - 2 * self.cov[i, i].sqrt(), self.center[i] + 2 * self.cov[i, i].sqrt()) for i in range(self.dim)]
        return tuple((min(-1, vs[0]), max(1, vs[1])) for vs in base_ranges)

    def loss_pretrain(self, params, weights):
        if self.positive:
            return 2 * weights[:-1] @ params - weights[-1] * params.sum()
        else:
            return weights @ params

    def loss_downstream(self, params):
        params = torch.broadcast_tensors(*params)
        params = torch.stack(params, -1)  # (..., d).
        d = params - self.center  # (..., d).
        return ((d @ self.cov) * d).sum(-1)  # (...).


class ToyHPOQuadraticMixture:
    def __init__(self, weights=(0.3, 0.7), centers=((0, 0), (0.5, 1)), covs=(((3, 0), (0, 1)), ((1, 0.5), (0.5, 1)))):
        self.weights = torch.as_tensor(weights).float()
        self.centers = torch.as_tensor(centers).float()
        self.covs = torch.as_tensor(covs).float()
        self.nc, self.dim = self.centers.shape
        assert self.weights.shape == (self.nc,)
        assert self.covs.shape == (self.nc, self.dim, self.dim)

    @property
    def n_params(self):
        return self.dim

    @property
    def n_pretrain_weights(self):
        return self.nc

    @property
    def solution(self):
        mixed_covs = (self.weights[:, None, None] * self.covs).sum(0)  # (d, d).
        mixed_covs_means = (self.weights[:, None] * (self.covs @ self.centers.unsqueeze(-1)).squeeze(-1)).sum(0)  # (d).
        return torch.linalg.pinv(mixed_covs) @ mixed_covs_means  # (d).

    @property
    def show_log(self):
        return True

    @property
    def show_range(self):
        base_ranges = [(self.centers[:, i].min() - 2 * self.covs[:, i, i].max().sqrt(), self.centers[:, i].max() + 2 * self.covs[:, i, i].max().sqrt()) for i in range(self.dim)]
        return tuple((min(-1, vs[0]), max(1, vs[1])) for vs in base_ranges)

    def loss_pretrain(self, params, weights):
        d = params.unsqueeze(-2) - self.centers  # (k, d).
        r = ((d.unsqueeze(-2) @ self.covs).squeeze(-2) * d).sum(-1)  # (..., k).
        return (r * weights).sum(-1)  # (...).

    def loss_downstream(self, params):
        params = torch.broadcast_tensors(*params)
        params = torch.stack(params, -1)  # (..., d).
        d = params.unsqueeze(-2) - self.centers  # (..., k, d).
        r = ((d.unsqueeze(-2) @ self.covs).squeeze(-2) * d).sum(-1)  # (..., k).
        return (r * self.weights).sum(-1)  # (...).


class ToyHPONoisyGrads:
    def __init__(self, toy, pretrain_noise_distribution=None, downstream_noise_distribution=None):
        self.toy = toy
        self.pretrain_noise_distribution = pretrain_noise_distribution
        self.downstream_noise_distribution = downstream_noise_distribution

    @property
    def n_params(self):
        return self.toy.n_params

    @property
    def n_pretrain_weights(self):
        return self.toy.n_pretrain_weights

    @property
    def solution(self):
        return self.toy.solution

    @property
    def show_log(self):
        return self.toy.show_log

    @property
    def show_range(self):
        return self.toy.show_range

    def loss_pretrain(self, params, weights):
        if self.pretrain_noise_distribution is not None:
            params = NoisyGradient.apply(params, self.pretrain_noise_distribution)
        return self.toy.loss_pretrain(params, weights)

    def loss_downstream(self, params):
        if self.downstream_noise_distribution is not None:
            params = NoisyGradient.apply(params, self.downstream_noise_distribution)
        return self.toy.loss_downstream(params)


class ToyHPOScaledGrads:
    def __init__(self, toy, pretrain_scale=None, downstream_scale=None):
        self.toy = toy
        self.pretrain_scale = torch.as_tensor(pretrain_scale) if pretrain_scale is not None else None
        self.downstream_scale = torch.as_tensor(downstream_scale) if downstream_scale is not None else None

    @property
    def n_params(self):
        return self.toy.n_params

    @property
    def n_pretrain_weights(self):
        return self.toy.n_pretrain_weights

    @property
    def solution(self):
        return self.toy.solution

    @property
    def show_log(self):
        return self.toy.show_log

    @property
    def show_range(self):
        return self.toy.show_range

    def loss_pretrain(self, params, weights):
        if self.pretrain_scale is not None:
            params = ScaledGradient.apply(params, self.pretrain_scale)
        return self.toy.loss_pretrain(params, weights)

    def loss_downstream(self, params):
        if self.downstream_scale is not None:
            params = ScaledGradient.apply(params, self.downstream_scale)
        return self.toy.loss_downstream(params)
