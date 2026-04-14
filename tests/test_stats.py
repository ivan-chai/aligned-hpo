#!/usr/bin/env python3
import random
import statistics
import torch
from unittest import TestCase, main

from aligned_hpo.stats import StatsTracker


class TestStatsTracker(TestCase):

    def _tracker(self, ema=0.9):
        return StatsTracker("loss", ema)

    # ------------------------------------------------------------------
    # Basic scalar tracking
    # ------------------------------------------------------------------

    def test_single_update(self):
        t = self._tracker()
        t.update(3.0)
        self.assertEqual(t.last_value, 3.0)
        self.assertEqual(t.ema_value, 3.0)
        self.assertEqual(t.avg_value, 3.0)
        self.assertAlmostEqual(t.median_value, 3.0)

    def test_avg_and_ema(self):
        t = self._tracker(ema=0.9)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            t.update(v)
        self.assertAlmostEqual(float(t.avg_value), sum(values) / len(values))
        ema = values[0]
        for v in values[1:]:
            ema = ema * 0.9 + v * 0.1
        self.assertAlmostEqual(float(t.ema_value), ema)

    # ------------------------------------------------------------------
    # Median bootstrap (first 5 observations)
    # ------------------------------------------------------------------

    def test_median_available_from_first_update(self):
        t = self._tracker()
        for i, expected in enumerate([5.0, 3.0, 3.0, 2.5, 3.0], start=1):
            # Observations: 5, 1, 3, 2, 4 — running exact medians: 5, 3, 3, 2.5, 3
            t.update([5.0, 1.0, 3.0, 2.0, 4.0][i - 1])
            self.assertIsNotNone(t.median_value,
                                 f"median_value should not be None after {i} update(s)")
            self.assertAlmostEqual(t.median_value, expected)

    def test_median_after_5_updates(self):
        t = self._tracker()
        for v in [5.0, 1.0, 3.0, 2.0, 4.0]:
            t.update(v)
        # Sorted: [1, 2, 3, 4, 5] — median is exactly 3.0.
        self.assertAlmostEqual(t.median_value, 3.0)

    # ------------------------------------------------------------------
    # Median accuracy on streams
    # ------------------------------------------------------------------

    def test_median_random_stream(self):
        """P² estimate converges close to the true median on a random uniform stream."""
        rng = random.Random(42)
        values = [rng.uniform(0.0, 100.0) for _ in range(1000)]
        t = self._tracker()
        for v in values:
            t.update(v)
        true_median = statistics.median(values)
        self.assertAlmostEqual(t.median_value, true_median, delta=5.0)

    def test_median_sorted_stream(self):
        """P² estimate on an ascending stream still converges into a reasonable range."""
        values = list(range(1, 201))
        t = self._tracker()
        for v in values:
            t.update(float(v))
        true_median = statistics.median(values)  # 100.5
        self.assertAlmostEqual(t.median_value, true_median, delta=20.0)

    # ------------------------------------------------------------------
    # Scalar tensor input (0-d)
    # ------------------------------------------------------------------

    def test_torch_tensor_input(self):
        t = self._tracker()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            t.update(torch.tensor(v))
        self.assertAlmostEqual(t.median_value, 3.0)

    # ------------------------------------------------------------------
    # Flat (1-D) tensor input
    # ------------------------------------------------------------------

    def test_flat_tensor_bootstrap(self):
        """After 5 flat-tensor updates the per-element median equals the sorted middle value."""
        t = self._tracker()
        # Two-element tensors; element 0: [5,1,3,2,4] → median 3, element 1: [10,2,6,4,8] → median 6.
        rows = [[5.0, 10.0], [1.0, 2.0], [3.0, 6.0], [2.0, 4.0], [4.0, 8.0]]
        for row in rows:
            t.update(torch.tensor(row))
        self.assertIsInstance(t.median_value, torch.Tensor)
        self.assertEqual(t.median_value.shape, (2,))
        self.assertAlmostEqual(t.median_value[0].item(), 3.0)
        self.assertAlmostEqual(t.median_value[1].item(), 6.0)

    def test_flat_tensor_avg_and_ema(self):
        """avg_value and ema_value are computed element-wise for flat tensors."""
        t = self._tracker(ema=0.9)
        values = [torch.tensor([1.0, 2.0]),
                  torch.tensor([3.0, 4.0]),
                  torch.tensor([5.0, 6.0])]
        for v in values:
            t.update(v)
        expected_avg = torch.tensor([(1 + 3 + 5) / 3, (2 + 4 + 6) / 3])
        self.assertTrue(torch.allclose(t.avg_value, expected_avg, atol=1e-5))

    def test_flat_tensor_median_stream(self):
        """Per-element P² median converges close to the true per-element median."""
        torch.manual_seed(0)
        N, D = 1000, 4
        data = torch.rand(N, D) * 100.0
        t = self._tracker()
        for row in data:
            t.update(row)
        true_median = data.median(dim=0).values
        self.assertTrue(
            torch.allclose(t.median_value, true_median, atol=5.0),
            f"median estimate {t.median_value} too far from true {true_median}",
        )

    def test_flat_tensor_state_dict_roundtrip(self):
        """state_dict / load_state_dict preserves full P² state for flat tensors."""
        t1 = self._tracker()
        torch.manual_seed(1)
        for _ in range(20):
            t1.update(torch.randn(3))
        sd = t1.state_dict()

        t2 = self._tracker()
        t2.load_state_dict(sd)
        self.assertTrue(torch.allclose(t2.median_value, t1.median_value))
        self.assertTrue(torch.allclose(t2.avg_value.float(), t1.avg_value.float(), atol=1e-5))

        # Further updates must stay in sync.
        for _ in range(5):
            v = torch.randn(3)
            t1.update(v)
            t2.update(v)
        self.assertTrue(torch.allclose(t2.median_value, t1.median_value))

    # ------------------------------------------------------------------
    # get() dict
    # ------------------------------------------------------------------

    def test_get_keys(self):
        t = self._tracker()
        for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
            t.update(v)
        result = t.get()
        for key in ("loss", "ema_loss", "avg_loss", "median_loss"):
            self.assertIn(key, result)
        self.assertAlmostEqual(result["median_loss"], 3.0)

    # ------------------------------------------------------------------
    # state_dict / load_state_dict round-trip
    # ------------------------------------------------------------------

    def test_state_dict_roundtrip(self):
        t1 = self._tracker()
        for v in [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]:
            t1.update(v)
        sd = t1.state_dict()

        t2 = self._tracker()
        t2.load_state_dict(sd)
        self.assertEqual(t2.n_updates, t1.n_updates)
        self.assertAlmostEqual(float(t2.avg_value), float(t1.avg_value))
        self.assertAlmostEqual(float(t2.ema_value), float(t1.ema_value))
        self.assertAlmostEqual(t2.median_value, t1.median_value)

    def test_state_dict_continues_correctly(self):
        """After loading state, further updates must produce the same result as the original."""
        t1 = self._tracker()
        for v in [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]:
            t1.update(v)

        t2 = self._tracker()
        t2.load_state_dict(t1.state_dict())

        for v in [6.0, 5.0, 3.0]:
            t1.update(v)
            t2.update(v)
        self.assertAlmostEqual(float(t2.avg_value), float(t1.avg_value))
        self.assertAlmostEqual(t2.median_value, t1.median_value)


if __name__ == "__main__":
    main()
