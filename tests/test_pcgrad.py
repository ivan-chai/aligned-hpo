#!/usr/bin/env python3
import torch
import torch.nn as nn
from unittest import TestCase, main

from aligned_hpo import PCGradOptimizer
from aligned_hpo.aligned_hpo import HPO_STAGE_DOWNSTREAM
from aligned_hpo.pcgrad import _pcgrad_surgery


def _make_model(seed=0):
    torch.manual_seed(seed)
    encoder = nn.Linear(4, 8)
    head1 = nn.Linear(8, 1)
    head2 = nn.Linear(8, 1)
    head_down = nn.Linear(8, 1)
    return encoder, head1, head2, head_down


def _make_optimizer(encoder, head1, head2, head_down, n_tasks=2, **kwargs):
    task_weights = torch.ones(n_tasks, requires_grad=True)
    params = [
        {"params": [task_weights]},
        {"params": list(head1.parameters()) + list(head2.parameters()) + list(head_down.parameters())},
        {"params": list(encoder.parameters())},
    ]
    return PCGradOptimizer(params, torch.optim.Adam, {"lr": 1e-3}, **kwargs)


def _make_batch(batch_size=8, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(batch_size, 4)
    y1 = torch.randn(batch_size, 1)
    y2 = torch.randn(batch_size, 1)
    y_down = torch.randn(batch_size, 1)
    return x, y1, y2, y_down


def _make_closure(optimizer, encoder, head1, head2, head_down, x, y1, y2, y_down):
    z = encoder(x)
    l1 = nn.functional.mse_loss(head1(z), y1)
    l2 = nn.functional.mse_loss(head2(z), y2)
    l_down = nn.functional.mse_loss(head_down(z), y_down)

    def closure(down_weight, weights, retain_graph=False, stage=None):
        optimizer.zero_grad()
        losses = torch.stack([l1, l2])
        loss = down_weight * l_down + (weights * losses).sum()
        loss.backward(retain_graph=retain_graph)
        return losses

    return closure


def _make_closure_encoder_decoder(optimizer, encoder, head1, head2, head_down, x, y1, y2, y_down):
    embeddings = encoder(x)
    z = embeddings.detach().clone()
    z.requires_grad = True
    l1 = nn.functional.mse_loss(head1(z), y1)
    l2 = nn.functional.mse_loss(head2(z), y2)
    l_down = nn.functional.mse_loss(head_down(z), y_down)

    def closure(down_weight, weights, retain_graph=False, stage=None):
        optimizer.zero_grad()
        z.grad = None
        losses = torch.stack([l1, l2])
        loss = down_weight * l_down + (weights * losses).sum()
        loss.backward(retain_graph=retain_graph)
        return z

    def closure_encoder(z_grad):
        optimizer.zero_grad()
        embeddings.backward(z_grad.reshape_as(embeddings))

    return closure, closure_encoder


class TestPCGradSurgery(TestCase):

    def test_no_conflict_unchanged(self):
        # Parallel gradients — no projection applied.
        g0 = torch.tensor([1.0, 0.0])
        g1 = torch.tensor([2.0, 0.0])
        result = _pcgrad_surgery([g0, g1])
        self.assertTrue(torch.allclose(result[0], g0))
        self.assertTrue(torch.allclose(result[1], g1))

    def test_opposite_gradients_projected(self):
        # g0 and g1 point in opposite directions: both should be projected to zero.
        g0 = torch.tensor([1.0, 0.0])
        g1 = torch.tensor([-1.0, 0.0])
        result = _pcgrad_surgery([g0, g1])
        self.assertTrue(torch.allclose(result[0], torch.zeros(2), atol=1e-6))
        self.assertTrue(torch.allclose(result[1], torch.zeros(2), atol=1e-6))

    def test_orthogonal_gradients_unchanged(self):
        g0 = torch.tensor([1.0, 0.0])
        g1 = torch.tensor([0.0, 1.0])
        result = _pcgrad_surgery([g0, g1])
        self.assertTrue(torch.allclose(result[0], g0, atol=1e-6))
        self.assertTrue(torch.allclose(result[1], g1, atol=1e-6))

    def test_partial_conflict_projects_conflicting_component(self):
        # g0 has a component along -g1 direction.
        g0 = torch.tensor([1.0, -1.0])
        g1 = torch.tensor([0.0, 1.0])
        # g0 · g1 = -1 < 0 → project: g0_new = g0 - (-1/1)*g1 = [1, -1] + [0,1] = [1, 0]
        result = _pcgrad_surgery([g0, g1])
        expected0 = torch.tensor([1.0, 0.0])
        self.assertTrue(torch.allclose(result[0], expected0, atol=1e-6))
        # g1 · g0 = -1 < 0 → project: g1_new = g1 - (-1/2)*g0 = [0,1] + [0.5,-0.5] = [0.5, 0.5]
        expected1 = torch.tensor([0.5, 0.5])
        self.assertTrue(torch.allclose(result[1], expected1, atol=1e-6))

    def test_three_tasks(self):
        torch.manual_seed(7)
        grads = [torch.randn(16) for _ in range(3)]
        result = _pcgrad_surgery(grads)
        self.assertEqual(len(result), 3)
        for r in result:
            self.assertEqual(r.shape, (16,))

    def test_uses_original_grads_for_projection(self):
        # Projection of g_i uses original g_j (not the modified version).
        # This is the PCGrad spec: projections are applied using the unmodified vectors.
        g0 = torch.tensor([1.0, -0.5])
        g1 = torch.tensor([-0.5, 1.0])
        # g0 · g1 = -0.5 - 0.5 = -1.0 < 0 → project g0 onto perp(g1)
        # g1 · g0 = same dot = -1.0 < 0 → project g1 onto perp(g0) using original g0
        result = _pcgrad_surgery([g0, g1])
        dot_orig = (g0 @ g1).item()
        norm_sq_g1 = (g1 @ g1).item()
        expected0 = g0 - (dot_orig / norm_sq_g1) * g1
        norm_sq_g0 = (g0 @ g0).item()
        expected1 = g1 - (dot_orig / norm_sq_g0) * g0
        self.assertTrue(torch.allclose(result[0], expected0, atol=1e-6))
        self.assertTrue(torch.allclose(result[1], expected1, atol=1e-6))


class TestPCGradOptimizerConstruction(TestCase):

    def test_requires_three_groups(self):
        with self.assertRaises(ValueError):
            task_weights = torch.ones(2, requires_grad=True)
            PCGradOptimizer(
                [{"params": [task_weights]}, {"params": [nn.Linear(2, 2)]}],
                torch.optim.SGD, {"lr": 1e-3},
            )

    def test_weights_must_be_1d(self):
        with self.assertRaises(ValueError):
            w = torch.ones(2, 2, requires_grad=True)
            PCGradOptimizer(
                [{"params": [w]},
                 {"params": list(nn.Linear(2, 1).parameters())},
                 {"params": list(nn.Linear(2, 1).parameters())}],
                torch.optim.SGD, {"lr": 1e-3},
            )

    def test_no_encoder_group_raises(self):
        encoder, head1, head2, head_down = _make_model()
        task_weights = torch.ones(2, requires_grad=True)
        with self.assertRaises(ValueError):
            PCGradOptimizer(
                [{"params": [task_weights]},
                 {"params": list(head1.parameters())},
                 {"params": list(head2.parameters())}],
                torch.optim.Adam, {"lr": 1e-3},
                heads_groups=(1, 2),
            )

    def test_group_0_in_heads_groups_raises(self):
        encoder, head1, head2, head_down = _make_model()
        task_weights = torch.ones(2, requires_grad=True)
        with self.assertRaises(ValueError):
            PCGradOptimizer(
                [{"params": [task_weights]},
                 {"params": list(head1.parameters())},
                 {"params": list(encoder.parameters())}],
                torch.optim.Adam, {"lr": 1e-3},
                heads_groups=(0, 1),
            )

    def test_weights_names_length_mismatch(self):
        encoder, head1, head2, head_down = _make_model()
        task_weights = torch.ones(2, requires_grad=True)
        with self.assertRaises(ValueError):
            PCGradOptimizer(
                [{"params": [task_weights]},
                 {"params": list(head1.parameters())},
                 {"params": list(encoder.parameters())}],
                torch.optim.Adam, {"lr": 1e-3},
                weights_names=["a"],
            )

    def test_encoder_decoder_requires_closure_encoder(self):
        encoder, head1, head2, head_down = _make_model()
        opt = _make_optimizer(encoder, head1, head2, head_down, encoder_decoder=True)
        x, y1, y2, y_down = _make_batch()
        closure, _ = _make_closure_encoder_decoder(opt, encoder, head1, head2, head_down, x, y1, y2, y_down)
        with self.assertRaises(ValueError):
            opt.hpo_step(closure)


class TestPCGradOptimizerStep(TestCase):

    def setUp(self):
        self.encoder, self.head1, self.head2, self.head_down = _make_model()
        self.optimizer = _make_optimizer(self.encoder, self.head1, self.head2, self.head_down)
        self.x, self.y1, self.y2, self.y_down = _make_batch()

    def _step(self):
        closure = _make_closure(
            self.optimizer, self.encoder, self.head1, self.head2, self.head_down,
            self.x, self.y1, self.y2, self.y_down,
        )
        return self.optimizer.hpo_step(closure)

    def test_step_runs(self):
        weights = self._step()
        self.assertEqual(weights.shape, (2,))

    def test_weights_are_uniform(self):
        weights = self._step()
        self.assertTrue(torch.allclose(weights, torch.full((2,), 0.5), atol=1e-6))

    def test_model_params_get_updated(self):
        initial = {name: p.clone() for name, p in self.encoder.named_parameters()}
        self._step()
        changed = any(
            not torch.allclose(p, initial[name])
            for name, p in self.encoder.named_parameters()
        )
        self.assertTrue(changed, "encoder parameters did not change after one step")

    def test_downstream_head_params_updated(self):
        initial = {name: p.clone() for name, p in self.head_down.named_parameters()}
        self._step()
        changed = any(
            not torch.allclose(p, initial[name])
            for name, p in self.head_down.named_parameters()
        )
        self.assertTrue(changed, "downstream head parameters did not change after one step")

    def test_n_updates_increments(self):
        self.assertEqual(self.optimizer._n_updates, 0)
        self._step()
        self.assertEqual(self.optimizer._n_updates, 1)
        self._step()
        self.assertEqual(self.optimizer._n_updates, 2)

    def test_step_raises_without_inner_flag(self):
        with self.assertRaises(ValueError):
            self.optimizer.step()

    def test_multiple_steps_stable(self):
        for _ in range(10):
            weights = self._step()
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(self.optimizer.weights).all())


class TestPCGradMetrics(TestCase):

    def test_metrics_empty_before_step(self):
        encoder, head1, head2, head_down = _make_model()
        opt = _make_optimizer(encoder, head1, head2, head_down)
        self.assertEqual(opt.metrics, {})

    def test_metrics_keys_after_step(self):
        encoder, head1, head2, head_down = _make_model()
        opt = _make_optimizer(encoder, head1, head2, head_down, weights_names=["t1", "t2"])
        x, y1, y2, y_down = _make_batch()
        opt.hpo_step(_make_closure(opt, encoder, head1, head2, head_down, x, y1, y2, y_down))
        m = opt.metrics
        for prefix in ("weights", "ema_weights", "avg_weights",
                       "grad_norms", "ema_grad_norms", "avg_grad_norms"):
            for name in ("t1", "t2"):
                self.assertIn(f"{prefix}_{name}", m, f"missing key {prefix}_{name}")

    def test_metrics_values_finite(self):
        encoder, head1, head2, head_down = _make_model()
        opt = _make_optimizer(encoder, head1, head2, head_down)
        x, y1, y2, y_down = _make_batch()
        for _ in range(3):
            opt.hpo_step(_make_closure(opt, encoder, head1, head2, head_down, x, y1, y2, y_down))
        for k, v in opt.metrics.items():
            self.assertTrue(torch.isfinite(torch.tensor(float(v))), f"{k} is not finite")


class TestPCGradStateDict(TestCase):

    def _run_steps(self, opt, encoder, head1, head2, head_down, n=5):
        x, y1, y2, y_down = _make_batch()
        for _ in range(n):
            opt.hpo_step(_make_closure(opt, encoder, head1, head2, head_down, x, y1, y2, y_down))

    def test_state_dict_roundtrip(self):
        encoder, head1, head2, head_down = _make_model()
        opt1 = _make_optimizer(encoder, head1, head2, head_down)
        self._run_steps(opt1, encoder, head1, head2, head_down, n=5)

        sd = opt1.state_dict()

        encoder2, head12, head22, head_down2 = _make_model()
        opt2 = _make_optimizer(encoder2, head12, head22, head_down2)
        opt2.load_state_dict(sd)

        self.assertEqual(opt2._n_updates, opt1._n_updates)
        self.assertTrue(torch.allclose(opt2.weights, opt1.weights))

    def test_state_dict_continues_in_sync(self):
        encoder, head1, head2, head_down = _make_model()
        opt1 = _make_optimizer(encoder, head1, head2, head_down)
        self._run_steps(opt1, encoder, head1, head2, head_down, n=5)

        encoder2, head12, head22, head_down2 = _make_model()
        encoder2.load_state_dict(encoder.state_dict())
        head12.load_state_dict(head1.state_dict())
        head22.load_state_dict(head2.state_dict())
        head_down2.load_state_dict(head_down.state_dict())

        opt2 = _make_optimizer(encoder2, head12, head22, head_down2)
        opt2.load_state_dict(opt1.state_dict())

        torch.manual_seed(99)
        for _ in range(3):
            x, y1, y2, y_down = _make_batch(seed=torch.randint(0, 1000, ()).item())
            w1 = opt1.hpo_step(_make_closure(opt1, encoder, head1, head2, head_down, x, y1, y2, y_down))
            w2 = opt2.hpo_step(_make_closure(opt2, encoder2, head12, head22, head_down2, x, y1, y2, y_down))
            self.assertTrue(torch.allclose(w1, w2, atol=1e-6),
                            f"weight divergence: {w1} vs {w2}")


class TestPCGradEncoderDecoder(TestCase):

    def _make_opt(self, encoder, head1, head2, head_down, **kwargs):
        return _make_optimizer(encoder, head1, head2, head_down, encoder_decoder=True, **kwargs)

    def test_step_runs(self):
        encoder, head1, head2, head_down = _make_model()
        opt = self._make_opt(encoder, head1, head2, head_down)
        x, y1, y2, y_down = _make_batch()
        closure, closure_encoder = _make_closure_encoder_decoder(
            opt, encoder, head1, head2, head_down, x, y1, y2, y_down)
        weights = opt.hpo_step(closure, closure_encoder)
        self.assertEqual(weights.shape, (2,))
        self.assertTrue(torch.allclose(weights, torch.full((2,), 0.5), atol=1e-6))

    def test_encoder_params_updated(self):
        encoder, head1, head2, head_down = _make_model()
        opt = self._make_opt(encoder, head1, head2, head_down)
        x, y1, y2, y_down = _make_batch()
        initial = {name: p.clone() for name, p in encoder.named_parameters()}
        closure, closure_encoder = _make_closure_encoder_decoder(
            opt, encoder, head1, head2, head_down, x, y1, y2, y_down)
        opt.hpo_step(closure, closure_encoder)
        changed = any(
            not torch.allclose(p, initial[name])
            for name, p in encoder.named_parameters()
        )
        self.assertTrue(changed, "encoder parameters did not change")

    def test_downstream_head_params_updated(self):
        encoder, head1, head2, head_down = _make_model()
        opt = self._make_opt(encoder, head1, head2, head_down)
        x, y1, y2, y_down = _make_batch()
        initial = {name: p.clone() for name, p in head_down.named_parameters()}
        closure, closure_encoder = _make_closure_encoder_decoder(
            opt, encoder, head1, head2, head_down, x, y1, y2, y_down)
        opt.hpo_step(closure, closure_encoder)
        changed = any(
            not torch.allclose(p, initial[name])
            for name, p in head_down.named_parameters()
        )
        self.assertTrue(changed, "downstream head parameters did not change")

    def test_multiple_steps_stable(self):
        encoder, head1, head2, head_down = _make_model()
        opt = self._make_opt(encoder, head1, head2, head_down)
        x, y1, y2, y_down = _make_batch()
        for _ in range(10):
            closure, closure_encoder = _make_closure_encoder_decoder(
                opt, encoder, head1, head2, head_down, x, y1, y2, y_down)
            weights = opt.hpo_step(closure, closure_encoder)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(opt.weights).all())

    def test_missing_z_grad_raises(self):
        encoder, head1, head2, head_down = _make_model()
        opt = self._make_opt(encoder, head1, head2, head_down)

        def bad_closure(down_weight, weights, retain_graph=False, stage=None):
            opt.zero_grad()
            z_dummy = torch.randn(8, 8, requires_grad=False)
            return z_dummy

        with self.assertRaises(TypeError):
            opt.hpo_step(bad_closure, lambda g: None)


class TestPCGradThreeTasks(TestCase):

    def test_three_tasks(self):
        torch.manual_seed(0)
        encoder = nn.Linear(4, 8)
        head1 = nn.Linear(8, 1)
        head2 = nn.Linear(8, 1)
        head3 = nn.Linear(8, 1)
        head_down = nn.Linear(8, 1)

        task_weights = torch.ones(3, requires_grad=True)
        params = [
            {"params": [task_weights]},
            {"params": list(head1.parameters()) + list(head2.parameters()) +
                       list(head3.parameters()) + list(head_down.parameters())},
            {"params": list(encoder.parameters())},
        ]
        opt = PCGradOptimizer(params, torch.optim.Adam, {"lr": 1e-3})

        x = torch.randn(8, 4)
        y = torch.randn(8, 1)
        y_down = torch.randn(8, 1)

        for _ in range(5):
            z = encoder(x)
            l1 = nn.functional.mse_loss(head1(z), y)
            l2 = nn.functional.mse_loss(head2(z), y)
            l3 = nn.functional.mse_loss(head3(z), y)
            l_down = nn.functional.mse_loss(head_down(z), y_down)

            def closure(down_weight, weights, retain_graph=False, stage=None):
                opt.zero_grad()
                losses = torch.stack([l1, l2, l3])
                loss = down_weight * l_down + (weights * losses).sum()
                loss.backward(retain_graph=retain_graph)
                return losses

            weights = opt.hpo_step(closure)

        self.assertEqual(weights.shape, (3,))
        expected = torch.full((3,), 1.0 / 3)
        self.assertTrue(torch.allclose(weights, expected, atol=1e-6))
        self.assertTrue(torch.isfinite(weights).all())


if __name__ == "__main__":
    main()
