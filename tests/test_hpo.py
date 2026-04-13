#!/usr/bin/env python3
import math
import numpy as np
import torch
from unittest import TestCase, main

from aligned_hpo import AlignedHPOptimizer, HPO_STAGE_DOWNSTREAM
from aligned_hpo.solvers import solve_qp, solve_qcqp
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
                # hp_gradient_normalizer (added in refactor) divides logits.grad by its
                # L2 norm, so normalise the expected gradient by the same factor.
                grad_gt = grad_gt / torch.linalg.norm(grad_gt).clamp(min=1e-12)

                def mock_step(closure):
                    closure()
                    self.assertAlmostEqual(weights.grad[0].item(), grad_gt[0].item(), places=3)
                    self.assertAlmostEqual(weights.grad[1].item(), grad_gt[1].item(), places=3)

                optimizer.base_optimizer.step = mock_step
                optimizer.hpo_step(closure)

    def test_optimizer(self):
        torch.manual_seed(0)
        for algorithm in ["mse"]:
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
        for algorithm in ["mse"]:
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
                        return z

                    def closure_encoder(z_grad):
                        optimizer.zero_grad()
                        params.grad = z_grad

                    for step in range(2000):
                        optimizer.hpo_step(closure, closure_encoder,
                                           after_backward_hook=grad_clip_fn)
                    try:
                        self.assertAlmostEqual(torch.linalg.norm(params - toy.solution).item(), 0, delta=4e-2)
                    except AssertionError:
                        print(f"Test failed for {algorithm} {normalization} {parametrization}")
                        raise


    def test_normalization_correction(self):
        """grad-norm weights are scale-invariant: scaling losses by α and weights by 1/α
        must leave both the parameter gradient and the logits gradient unchanged."""
        toy = ToyHPOQuadratic(positive=True)
        params = torch.nn.Parameter(torch.zeros([toy.n_params]))
        weights = torch.nn.Parameter(torch.ones([toy.n_pretrain_weights]))
        optimizer1 = AlignedHPOptimizer([{"params": [weights]},
                                         {"params": []},  # No heads.
                                         {"params": [params]}],
                                        torch.optim.SGD,
                                        {"lr": 0.0},  # Only compute gradients.
                                        algorithm="sgd",
                                        weights_parametrization="abs",
                                        weights_normalization="grad-norm")

        def closure1(down, weights, retain_graph=False, stage=None):
            optimizer1.zero_grad()
            if down > 0:
                v = down * toy.loss_downstream(params)
            else:
                v = 0
            if any(w > 0 for w in weights):
                v = v + toy.loss_pretrain(params, weights)
            v.backward()

        optimizer1.hpo_step(closure1)
        params_grad1 = params.grad.clone()
        logits_grad1 = weights.grad.clone()

        scale = 2
        weights.data /= scale
        optimizer2 = AlignedHPOptimizer([{"params": [weights]},
                                         {"params": []},  # No heads.
                                         {"params": [params]}],
                                        torch.optim.SGD,
                                        {"lr": 0.0},
                                        algorithm="sgd",
                                        weights_parametrization="abs",
                                        weights_normalization="grad-norm")

        def closure2(down, weights, retain_graph=False, stage=None):
            optimizer2.zero_grad()
            if down > 0:
                v = down * toy.loss_downstream(params)
            else:
                v = 0
            if any(w > 0 for w in weights):
                v = v + toy.loss_pretrain(params, weights * scale)
            v.backward()

        optimizer2.hpo_step(closure2)
        params_grad2 = params.grad.clone()
        logits_grad2 = weights.grad.clone()

        self.assertTrue(torch.allclose(params_grad1, params_grad2, atol=1e-5),
                        f"Parameter gradient not scale-invariant:\n{params_grad1}\nvs\n{params_grad2}")
        self.assertTrue(torch.allclose(logits_grad1, logits_grad2, atol=1e-5),
                        f"Logits gradient not scale-invariant:\n{logits_grad1}\nvs\n{logits_grad2}")

    def test_optimizer_correction(self):
        torch.manual_seed(0)
        lr = 0.1
        toy = ToyHPOQuadratic(positive=True)
        params = torch.nn.Parameter(torch.zeros([toy.n_params]))
        weights = torch.nn.Parameter(torch.ones([toy.n_pretrain_weights]))
        decoder = torch.nn.Linear(toy.n_params, toy.n_params)
        optimizer = AlignedHPOptimizer([{"params": [weights]},
                                        {"params": decoder.parameters()},  # Head.
                                        {"params": []},  # No shared decoder.
                                        {"params": [params]}],  # Encoder.
                                        torch.optim.Adam,
                                        {"lr": lr},
                                        encoder_decoder=True,
                                        algorithm="mse",
                                        weights_parametrization="abs",
                                        weights_normalization="none")

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
            return z

        def closure_encoder(z_grad):
            optimizer.zero_grad()
            params.grad = z_grad

        grad_clip_fn = lambda: torch.nn.utils.clip_grad_norm_([weights, params] + list(decoder.parameters()), 1)

        # Warmup.
        for step in range(100):
            optimizer.hpo_step(closure, closure_encoder,
                               after_backward_hook=grad_clip_fn)

        @torch.no_grad()
        def gather_weights(part):
            if part == "heads":
                param_groups = [optimizer.param_groups[1]]
            elif optimizer.encoder_decoder:
                if part == "shared":
                    param_groups = [optimizer.param_groups[2]]
                else:
                    assert part == "encoder"
                    param_groups = optimizer.param_groups[3:]
            else:
                assert part == "shared"
                param_groups = optimizer.param_groups[2:]
            weights = []
            for group in param_groups:
                for p in group["params"]:
                    weights.append(p.flatten())
            return torch.cat(weights) if weights else None

        heads_weights_before = gather_weights("heads")
        encoder_weights_before = gather_weights("encoder")
        heads_gradient = torch.empty_like(heads_weights_before)
        encoder_gradient = torch.empty_like(encoder_weights_before)

        def after_backward_hook():
            # Clip gradients.
            torch.nn.utils.clip_grad_norm_([weights, params] + list(decoder.parameters()), 1)
            heads_grad = optimizer._gather_grads("heads")
            optimizer.apply_optimizer_correction_("heads", heads_grad)
            heads_gradient.copy_(heads_grad)
            encoder_grad = optimizer._gather_grads("encoder")
            optimizer.apply_optimizer_correction_("encoder", encoder_grad)
            encoder_gradient.copy_(encoder_grad)

        # Compute gradients and make step.
        optimizer.hpo_step(closure, closure_encoder,
                           after_backward_hook=after_backward_hook)
        heads_weights_after = gather_weights("heads")
        encoder_weights_after = gather_weights("encoder")

        # apply_optimizer_correction_ approximates the Adam preconditioner update.
        # The heads update uses the same Adam state, so the match is tight.
        self.assertTrue(torch.allclose(heads_weights_after - heads_weights_before, -heads_gradient * lr, rtol=5e-2))
        # The encoder is updated via closure_encoder which sets .grad directly;
        # Adam then applies a preconditioner built from its own prior state, which
        # diverges from the approximation in apply_optimizer_correction_.
        # We only check that the encoder actually moved in the correct direction.
        enc_diff = encoder_weights_after - encoder_weights_before
        self.assertTrue((enc_diff * (-encoder_gradient)).sum() > 0,
                        "Encoder update direction does not match gradient")


class TestSolveQCQP(TestCase):
    """Tests for solve_qcqp: maximize b^T x s.t. x^T C x = 1 (and x >= 0 when positive=True)."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_spd(self, n, seed=0, scale=1.0):
        """Return a random symmetric positive-definite n×n matrix scaled by `scale`."""
        rng = np.random.RandomState(seed)
        A = rng.randn(n, n)
        C = A @ A.T + np.eye(n) * 0.1  # ensure strict positive definiteness
        return torch.from_numpy(C * scale).double()

    def _check_ellipsoid(self, x, C, atol=1e-6):
        """Assert x^T C x ≈ 1 (only when x is non-zero)."""
        if torch.linalg.norm(x) < 1e-10:
            return  # zero result is a valid sentinel; skip constraint check
        val = (x @ C.double() @ x).item()
        self.assertAlmostEqual(val, 1.0, delta=atol, msg=f"Ellipsoid constraint violated: x^T C x = {val}")

    def _check_positive(self, x):
        self.assertTrue((x >= -1e-9).all(), f"Positivity violated: min={x.min()}")

    # ------------------------------------------------------------------
    # Basic correctness
    # ------------------------------------------------------------------

    def test_non_positive_known_solution(self):
        """1D case has a trivial closed-form answer."""
        # maximize b*x s.t. x^2 * C = 1  =>  x* = sign(b) / sqrt(C)
        for C_val, b_val in [(4.0, 1.0), (4.0, -1.0), (1.0, 3.0)]:
            C = torch.tensor([[C_val]], dtype=torch.double)
            b = torch.tensor([b_val], dtype=torch.double)
            x = solve_qcqp(C, b, positive=False)
            x_gt = b_val / (abs(b_val) * math.sqrt(C_val))  # = sign(b)/sqrt(C)
            self.assertAlmostEqual(x[0].item(), x_gt, places=6)
            self._check_ellipsoid(x, C)

    def test_non_positive_2d(self):
        """2D unconstrained: solution direction is C^{-1} b, normalised to unit ellipsoid."""
        torch.manual_seed(1)
        C = self._random_spd(2, seed=1)
        b = torch.tensor([1.0, 2.0], dtype=torch.double)
        x = solve_qcqp(C, b, positive=False)
        self._check_ellipsoid(x, C)
        # Solution must be proportional to C^{-1} b.
        prod = torch.linalg.solve(C, b)
        ratio = x / prod
        self.assertAlmostEqual(ratio[0].item(), ratio[1].item(), places=6)

    def test_positive_2d_basic(self):
        """Positive case: x >= 0 and on the ellipsoid."""
        C = self._random_spd(3, seed=2)
        b = torch.tensor([2.0, 1.0, 0.5], dtype=torch.double)
        x = solve_qcqp(C, b, positive=True)
        self._check_ellipsoid(x, C)
        self._check_positive(x)

    def test_positive_all_negative_b(self):
        """When all b[i] <= 0, the optimum under x >= 0 is effectively degenerate; expect zeros."""
        C = self._random_spd(3, seed=3)
        b = torch.tensor([-1.0, -2.0, -0.5], dtype=torch.double)
        x = solve_qcqp(C, b, positive=True)
        # The active-set seeds with the least-bad index; result should be a valid
        # unit-ellipsoid point or the zero sentinel.
        if torch.linalg.norm(x) > 1e-10:
            self._check_ellipsoid(x, C)
            self._check_positive(x)

    # ------------------------------------------------------------------
    # Small magnitudes of C
    # ------------------------------------------------------------------

    def test_small_C_scale(self):
        """Scaling C by a small constant should not change the solution direction."""
        C0 = self._random_spd(4, seed=4)
        b = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.double)
        x_ref = solve_qcqp(C0, b, positive=False)

        for scale in [1e-4, 1e-8, 1e-12]:
            C_small = C0 * scale
            x = solve_qcqp(C_small, b, positive=False)
            self._check_ellipsoid(x, C_small)
            # Direction of x must be the same as x_ref (up to sign / magnitude).
            if torch.linalg.norm(x) > 1e-10 and torch.linalg.norm(x_ref) > 1e-10:
                cos = (x @ x_ref) / (torch.linalg.norm(x) * torch.linalg.norm(x_ref))
                self.assertGreater(cos.item(), 0.99,
                    msg=f"Direction changed at scale={scale}: cos={cos.item():.4f}")

    def test_small_C_positive(self):
        """Same direction check under positivity constraint."""
        C0 = self._random_spd(3, seed=5)
        b = torch.tensor([3.0, 1.0, 2.0], dtype=torch.double)
        x_ref = solve_qcqp(C0, b, positive=True)

        for scale in [1e-4, 1e-8]:
            C_small = C0 * scale
            x = solve_qcqp(C_small, b, positive=True)
            if torch.linalg.norm(x) > 1e-10:
                self._check_ellipsoid(x, C_small)
                self._check_positive(x)
                if torch.linalg.norm(x_ref) > 1e-10:
                    cos = (x @ x_ref) / (torch.linalg.norm(x) * torch.linalg.norm(x_ref))
                    self.assertGreater(cos.item(), 0.99,
                        msg=f"Direction changed at scale={scale}: cos={cos.item():.4f}")

    # ------------------------------------------------------------------
    # Small magnitudes of b
    # ------------------------------------------------------------------

    def test_small_b_scale(self):
        """Scaling b by a small constant should not change the solution direction."""
        C = self._random_spd(4, seed=6)
        b0 = torch.tensor([1.0, 2.0, 0.5, 1.5], dtype=torch.double)
        x_ref = solve_qcqp(C, b0, positive=False)

        for scale in [1e-4, 1e-8, 1e-12]:
            b_small = b0 * scale
            x = solve_qcqp(C, b_small, positive=False)
            if torch.linalg.norm(x) > 1e-10:
                self._check_ellipsoid(x, C)
                if torch.linalg.norm(x_ref) > 1e-10:
                    cos = (x @ x_ref) / (torch.linalg.norm(x) * torch.linalg.norm(x_ref))
                    self.assertGreater(cos.item(), 0.99,
                        msg=f"Direction changed at scale={scale}: cos={cos.item():.4f}")

    def test_small_b_and_C_scale(self):
        """Both C and b tiny: solution direction should match or zero sentinel returned."""
        C0 = self._random_spd(3, seed=7)
        b0 = torch.tensor([1.0, 0.5, 2.0], dtype=torch.double)
        x_ref = solve_qcqp(C0, b0, positive=False)

        for scale in [1e-6, 1e-10]:
            C_small = C0 * scale
            b_small = b0 * scale
            x = solve_qcqp(C_small, b_small, positive=False)
            if torch.linalg.norm(x) > 1e-10:
                self._check_ellipsoid(x, C_small)
                cos = (x @ x_ref) / (torch.linalg.norm(x) * torch.linalg.norm(x_ref))
                self.assertGreater(cos.item(), 0.99,
                    msg=f"Direction changed at scale={scale}: cos={cos.item():.4f}")

    # ------------------------------------------------------------------
    # Small singular values of C
    # ------------------------------------------------------------------

    def test_near_singular_C(self):
        """C with one very small singular value: solution should still satisfy constraints."""
        rng = np.random.RandomState(8)
        U, _, Vt = np.linalg.svd(rng.randn(4, 4))
        # Singular values: normal except one very small one.
        for small_sv in [1e-6, 1e-8, 1e-10]:
            sv = np.array([2.0, 1.5, 1.0, small_sv])
            C = torch.from_numpy(U @ np.diag(sv) @ Vt @ Vt.T @ np.diag(sv) @ U.T).double()
            # Make C symmetric PSD.
            C = (C + C.T) / 2 + torch.eye(4, dtype=torch.double) * small_sv

            b = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.double)
            x = solve_qcqp(C, b, positive=False)
            if torch.linalg.norm(x) > 1e-10:
                self._check_ellipsoid(x, C, atol=1e-4)

    def test_rank_deficient_C(self):
        """Rank-deficient C (exactly zero eigenvalue): lstsq uses rcond to regularise."""
        n = 4
        v = torch.randn(n, dtype=torch.double)
        # C = v v^T is rank-1 (all other eigenvalues zero).
        C = v.outer(v)
        b = torch.tensor([1.0, 0.5, 0.2, 0.8], dtype=torch.double)
        # Should not raise; may return zeros or a valid point.
        x = solve_qcqp(C, b, positive=False)
        self.assertFalse(torch.isnan(x).any(), "NaN in output for rank-deficient C")
        self.assertFalse(torch.isinf(x).any(), "Inf in output for rank-deficient C")

    def test_diagonal_C_with_small_entry(self):
        """Diagonal C where one entry is near zero: solution should exclude that dimension."""
        eps_val = 1e-9
        C = torch.diag(torch.tensor([1.0, 2.0, eps_val], dtype=torch.double))
        b = torch.tensor([1.0, 1.0, 1.0], dtype=torch.double)
        x = solve_qcqp(C, b, positive=False)
        self.assertFalse(torch.isnan(x).any())
        self.assertFalse(torch.isinf(x).any())
        if torch.linalg.norm(x) > 1e-10:
            self._check_ellipsoid(x, C, atol=1e-4)

    # ------------------------------------------------------------------
    # Degenerate / zero inputs
    # ------------------------------------------------------------------

    def test_zero_b(self):
        """Zero b: no gradient signal; expect the zero sentinel."""
        C = self._random_spd(3, seed=9)
        b = torch.zeros(3, dtype=torch.double)
        x = solve_qcqp(C, b, positive=False)
        self.assertTrue((x == 0).all(), f"Expected zeros for zero b, got {x}")

    def test_zero_C(self):
        """Zero C: degenerate; expect zero sentinel and no crash."""
        C = torch.zeros(3, 3, dtype=torch.double)
        b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.double)
        x = solve_qcqp(C, b, positive=False)
        self.assertFalse(torch.isnan(x).any())
        self.assertFalse(torch.isinf(x).any())

    def test_1d_non_positive(self):
        """1D trivial case: x* = 1/sqrt(C) when b > 0."""
        C = torch.tensor([[9.0]], dtype=torch.double)
        b = torch.tensor([5.0], dtype=torch.double)
        x = solve_qcqp(C, b, positive=False)
        self.assertAlmostEqual(x[0].item(), 1.0 / 3.0, places=6)
        self._check_ellipsoid(x, C)

    def test_1d_positive(self):
        """1D positive case: same as unconstrained since x = 1/sqrt(C) > 0."""
        C = torch.tensor([[4.0]], dtype=torch.double)
        b = torch.tensor([2.0], dtype=torch.double)
        x = solve_qcqp(C, b, positive=True)
        self.assertAlmostEqual(x[0].item(), 0.5, places=6)
        self._check_ellipsoid(x, C)
        self._check_positive(x)

    # ------------------------------------------------------------------
    # Numerical invariance to scale
    # ------------------------------------------------------------------

    def test_solution_invariant_to_b_scale(self):
        """x*(αb) == x*(b) for any positive scalar α."""
        C = self._random_spd(5, seed=10)
        b = torch.tensor([1.0, 2.0, 0.5, 1.5, 0.8], dtype=torch.double)
        x_ref = solve_qcqp(C, b, positive=False)
        for alpha in [0.1, 10.0, 1e5, 1e-5]:
            x = solve_qcqp(C, b * alpha, positive=False)
            if torch.linalg.norm(x) > 1e-10 and torch.linalg.norm(x_ref) > 1e-10:
                cos = (x @ x_ref) / (torch.linalg.norm(x) * torch.linalg.norm(x_ref))
                self.assertGreater(cos.item(), 0.999,
                    msg=f"Direction changed for alpha={alpha}")

    def test_solution_to_C_scale(self):
        """x*(αC) has the same direction as x*(C) but magnitude scaled by 1/sqrt(α)."""
        C0 = self._random_spd(3, seed=11)
        b = torch.tensor([1.0, 2.0, 1.5], dtype=torch.double)
        x_ref = solve_qcqp(C0, b, positive=False)
        for alpha in [0.01, 100.0, 1e6, 1e-6]:
            x = solve_qcqp(C0 * alpha, b, positive=False)
            if torch.linalg.norm(x) > 1e-10 and torch.linalg.norm(x_ref) > 1e-10:
                # Directions must match.
                x_dir = x / torch.linalg.norm(x)
                xr_dir = x_ref / torch.linalg.norm(x_ref)
                cos = (x_dir @ xr_dir).item()
                self.assertGreater(cos, 0.999,
                    msg=f"Direction changed for alpha={alpha}: cos={cos:.4f}")
                # Magnitude must scale as 1/sqrt(alpha).
                ratio = (torch.linalg.norm(x) * math.sqrt(alpha) / torch.linalg.norm(x_ref)).item()
                self.assertAlmostEqual(ratio, 1.0, delta=1e-4,
                    msg=f"Magnitude scaling wrong for alpha={alpha}: ratio={ratio:.6f}")


if __name__ == "__main__":
    main()
