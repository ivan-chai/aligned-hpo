#!/usr/bin/env python3
import math
import torch
from unittest import TestCase, main

from aligned_hpo import AlignedHPOptimizer, HPO_STAGE_DOWNSTREAM
from aligned_hpo.solvers import solve_qp
from aligned_hpo.toy import ToyHPOQuadratic


def f1(x):
    return (x - 5) ** 2


def f2(x):
    return (x + 3) ** 2


def loss(x, weights):
    alpha, beta = weights
    return alpha * f1(x) + beta * f2(x)


def downstream(x):
    return 0.3 * f1(x) + 0.9 * f2(x)


class TestAlignedHPOptimizer(TestCase):
    def test_quadratic_program(self):
        torch.manual_seed(0)
        # Test random.
        dim = 5
        v = torch.eye(dim).double()
        c = torch.randn(dim, dim).double()
        c = c @ c.T
        v = c @ v
        weights_gt = 5 * torch.rand(dim).double()  # Positive.
        target = v @ weights_gt

        weights = solve_qp(v @ v.T, - v @ target, positive=True)
        self.assertAlmostEqual(torch.linalg.norm(weights - weights_gt).item(), 0, places=2)

    def test_gradient(self):
        torch.manual_seed(0)
        x = torch.nn.Parameter(torch.randn([]))
        weights = torch.nn.Parameter(torch.rand([2]))

        def closure(down, weights, retain_graph=False, stage=None):
            optimizer.zero_grad()
            if down > 0:
                v = down * downstream(x)
            else:
                v = 0
            if (weights > 0).any():
                v = v + loss(x, weights)
            v.backward(retain_graph=retain_graph)

        for parametrization in ["abs", "linear"]:
            for normalization in ["none", "sum", "norm"]:
                if (parametrization == "linear") and (normalization == "sum"):
                    continue
                optimizer = AlignedHPOptimizer([{"params": [weights]},
                                                {"params": []},  # No heads.
                                                {"params": [x]}],
                                               torch.optim.Adam,
                                               {"lr": 0.01},
                                               algorithm="sgd",
                                               weights_parametrization=parametrization,
                                               weights_normalization=normalization,
                                               eps=0)

                if parametrization == "abs":
                    actual_weights = torch.abs(weights)
                else:
                    assert parametrization == "linear"
                    actual_weights = weights
                if normalization == "sum":
                    actual_weights = actual_weights / actual_weights.sum() * len(actual_weights)
                elif normalization == "norm":
                    actual_weights = actual_weights / torch.linalg.norm(actual_weights) * math.sqrt(len(actual_weights))
                else:
                    assert normalization == "none"
                w1, w2 = actual_weights

                grad = 2 * w1 * (x - 5) + 2 * w2 * (x + 3)
                new_x = (x + grad).detach() - grad
                weights.grad = None
                downstream(new_x).backward()
                grad_gt = weights.grad.clone()

                def mock_step(closure):
                    closure()
                    self.assertAlmostEqual(weights.grad[0].item(), grad_gt[0].item(), places=3)
                    self.assertAlmostEqual(weights.grad[1].item(), grad_gt[1].item(), places=3)

                optimizer.base_optimizer.step = mock_step
                optimizer.hpo_step(closure)

    def test_optimizer(self):
        torch.manual_seed(0)
        for algorithm in ["mse", "expected-error"]:
            for normalization in ["none", "sum", "norm"]:
                for parametrization in ["abs", "linear"]:
                    if (parametrization == "linear") and (normalization == "sum"):
                        continue
                    toy = ToyHPOQuadratic(positive=parametrization == "abs")
                    params = torch.nn.Parameter(torch.zeros([toy.n_params]))
                    weights = torch.nn.Parameter(torch.ones([toy.n_pretrain_weights]))
                    optimizer = AlignedHPOptimizer([{"params": [weights]},
                                                    {"params": []},  # No heads.
                                                    {"params": [params]}],
                                                   torch.optim.Adam,
                                                   {"lr": 0.01},
                                                   algorithm=algorithm,
                                                   weights_parametrization=parametrization,
                                                   weights_normalization=normalization)

                    def closure(down, weights, retain_graph=False, stage=None):
                        optimizer.zero_grad()
                        if down > 0:
                            v = down * toy.loss_downstream(params)
                        else:
                            v = 0
                        if any(w > 0 for w in weights):
                            v = v + toy.loss_pretrain(params, weights)
                        v.backward()
                    for step in range(2000):
                        optimizer.hpo_step(closure)
                    try:
                        self.assertAlmostEqual(torch.linalg.norm(params - toy.solution).item(), 0, delta=1e-2)
                    except AssertionError:
                        print(f"Test failed for {algorithm} {normalization} {parametrization}")
                        raise

    def test_optimizer_encoder_decoder(self):
        torch.manual_seed(0)
        for algorithm in ["mse", "expected-error"]:
            for normalization in ["none", "sum", "norm"]:
                for parametrization in ["abs", "linear"]:
                    if (parametrization == "linear") and (normalization == "sum"):
                        continue
                    toy = ToyHPOQuadratic(positive=parametrization == "abs")
                    params = torch.nn.Parameter(torch.zeros([toy.n_params]))
                    weights = torch.nn.Parameter(torch.ones([toy.n_pretrain_weights]))
                    decoder = torch.nn.Linear(toy.n_params, toy.n_params)
                    optimizer = AlignedHPOptimizer([{"params": [weights]},
                                                    {"params": decoder.parameters()},  # Head.
                                                    {"params": []},  # No shared decoder.
                                                    {"params": [params]}],  # Encoder.
                                                   torch.optim.SGD,
                                                   {"lr": 0.1},
                                                   encoder_decoder=True,
                                                   algorithm=algorithm,
                                                   weights_parametrization=parametrization,
                                                   weights_normalization=normalization)
                    grad_clip_fn = lambda: torch.nn.utils.clip_grad_norm_([weights, params] + list(decoder.parameters()), 1)

                    def closure(down, weights, retain_graph=False, stage=None):
                        optimizer.zero_grad()
                        z = params.detach().clone()
                        z.requires_grad = True
                        if down > 0:
                            v = down * toy.loss_downstream(z)
                        else:
                            v = 0
                        if any(w > 0 for w in weights):
                            v = v + toy.loss_pretrain(decoder(z), weights)
                        v.backward(retain_graph=retain_graph)
                        return z.grad

                    def closure_encoder(z_grad):
                        optimizer.zero_grad()
                        params.grad = z_grad

                    for step in range(2000):
                        optimizer.hpo_step(closure, closure_encoder,
                                           after_backward_hook=grad_clip_fn)
                    try:
                        self.assertAlmostEqual(torch.linalg.norm(params - toy.solution).item(), 0, delta=2e-2)
                    except AssertionError:
                        print(f"Test failed for {algorithm} {normalization} {parametrization}")
                        raise


if __name__ == "__main__":
    main()
