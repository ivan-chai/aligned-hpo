import sklearn.datasets
import torch


class ToyCLSBinaryLinear:
    def __init__(self):
        X, y = sklearn.datasets.make_moons(random_state=0)
        self.X = torch.from_numpy(X).to(torch.float)
        self.y = torch.from_numpy(y).to(torch.long)

    @property
    def data(self):
        return self.X, self.y

    @property
    def n_features(self):
        return 2

    @property
    def n_pretrain_params(self):
        return 2

    @property
    def n_downstream_params(self):
        return 2

    @property
    def n_pretrain_weights(self):
        return 2

    def loss_pretrain(self, params, labels, weights):
        weighted = params * weights
        return weighted.take_along_dim(labels[:, None], 1).squeeze(1).mean()

    def loss_downstream(self, params, labels):
        return torch.nn.functional.cross_entropy(params, labels)


class ToyCLSBinaryLinearUnlabeled:
    def __init__(self, n_pretrain_losses=2):
        X, y = sklearn.datasets.make_moons(random_state=0)
        self.X = torch.from_numpy(X).to(torch.float)
        self.y = torch.from_numpy(y).to(torch.long)
        self.n_pretrain_losses = n_pretrain_losses

    @property
    def data(self):
        return self.X, self.y

    @property
    def n_features(self):
        return 2

    @property
    def n_pretrain_params(self):
        return self.n_pretrain_losses

    @property
    def n_downstream_params(self):
        return 2

    @property
    def n_pretrain_weights(self):
        return self.n_pretrain_losses

    def loss_pretrain(self, params, labels, weights):
        weighted = params * weights
        return weighted.sum(1).mean()

    def loss_downstream(self, params, labels):
        return torch.nn.functional.cross_entropy(params, labels)


class ToyCLSSurrogate:
    def __init__(self, noise=0):
        X, y = sklearn.datasets.make_moons(random_state=0)
        gen = torch.Generator()
        gen.manual_seed(0)
        self.X = torch.from_numpy(X).to(torch.float)
        self.X += torch.randn(*self.X.shape, generator=gen) * noise
        self.y = torch.from_numpy(y).to(torch.long)

    @property
    def data(self):
        return self.X, self.y

    @property
    def n_features(self):
        return 2

    @property
    def n_pretrain_params(self):
        return 1

    @property
    def n_downstream_params(self):
        return 2

    @property
    def n_pretrain_weights(self):
        return 1

    def loss_pretrain(self, params, labels, weights):
        assert params.shape[1] == 1
        return (params.squeeze(1) - labels.float()).square().mean()

    def loss_downstream(self, params, labels):
        return torch.nn.functional.cross_entropy(params, labels)
