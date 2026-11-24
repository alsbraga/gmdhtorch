"""Tests for neuron selection criteria (gmdhSelect*.m)."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from gmdhtorch import (
    SelectBest,
    SelectGaussMean,
    SelectGaussMedian,
    SelectMedian,
    SelectNone,
    SelectThreshold,
    gmdh_forecast_variance_selection,
    gmdh_select_best,
    gmdh_select_gauss_mean,
    gmdh_select_gauss_median,
    gmdh_select_median,
    gmdh_select_threshold,
)
from gmdhtorch.selection import _compute_batch_perfs_torch, resolve_selection_strategy


class TestSelection(unittest.TestCase):
    def test_select_best(self) -> None:
        perfs = np.array([0.5, 0.1, 0.9, 0.2, 0.4])
        rem = gmdh_select_best(perfs, 2)
        np.testing.assert_array_equal(rem, np.array([True, False, True, False, True]))

    def test_select_median_and_bounds(self) -> None:
        perfs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        rem = gmdh_select_median(perfs, max_neurons=10, min_neurons=1)
        np.testing.assert_array_equal(rem, np.array([False, False, False, True, True, True]))

        rem_max = gmdh_select_median(perfs, max_neurons=2, min_neurons=1)
        np.testing.assert_array_equal(rem_max, np.array([False, False, True, True, True, True]))

    def test_select_gauss_median_and_mean(self) -> None:
        perfs = np.linspace(0.01, 1.0, 10)
        rem_med = gmdh_select_gauss_median(perfs, max_neurons=4, min_neurons=2, distance=0.5)
        kept_med = int(np.sum(~rem_med))
        self.assertTrue(2 <= kept_med <= 4)

        rem_mean = gmdh_select_gauss_mean(perfs, max_neurons=4, min_neurons=2, distance=0.5)
        kept_mean = int(np.sum(~rem_mean))
        self.assertTrue(2 <= kept_mean <= 4)

    def test_select_threshold_and_fvs(self) -> None:
        perfs = np.array([0.05, 0.15, 0.25, 0.35])
        rem = gmdh_select_threshold(perfs, max_neurons=3, min_neurons=1, threshold=0.20)
        np.testing.assert_array_equal(rem, np.array([False, False, True, True]))

        targets = np.array([1.0, 2.0, 3.0, 4.0])
        good_neuron = targets + 0.01
        bad_neuron = np.zeros_like(targets)
        layer_outputs = np.column_stack([good_neuron, bad_neuron])
        fvs_keep = gmdh_forecast_variance_selection(targets, layer_outputs)
        np.testing.assert_array_equal(fvs_keep, np.array([True, False]))

    def test_selection_strategy_objects(self) -> None:
        perfs = np.array([0.5, 0.1, 0.9, 0.2, 0.4, 0.3])
        np.testing.assert_array_equal(
            SelectBest(2)(perfs, max_neurons=10, min_neurons=1),
            gmdh_select_best(perfs, 2),
        )
        np.testing.assert_array_equal(
            SelectMedian()(perfs, max_neurons=4, min_neurons=1),
            gmdh_select_median(perfs, max_neurons=4, min_neurons=1),
        )
        np.testing.assert_array_equal(
            SelectThreshold(0.35)(perfs, max_neurons=5, min_neurons=1),
            gmdh_select_threshold(perfs, max_neurons=5, min_neurons=1, threshold=0.35),
        )
        np.testing.assert_array_equal(
            SelectGaussMedian(0.5)(perfs, max_neurons=4, min_neurons=1),
            gmdh_select_gauss_median(perfs, max_neurons=4, min_neurons=1, distance=0.5),
        )
        np.testing.assert_array_equal(
            SelectGaussMean(0.5)(perfs, max_neurons=4, min_neurons=1),
            gmdh_select_gauss_mean(perfs, max_neurons=4, min_neurons=1, distance=0.5),
        )
        np.testing.assert_array_equal(
            SelectNone()(perfs, max_neurons=2, min_neurons=1),
            np.zeros(perfs.size, dtype=bool),
        )
        self.assertIsInstance(resolve_selection_strategy("select_median"), SelectMedian)
        self.assertIsInstance(resolve_selection_strategy("none"), SelectNone)

    def test_multi_output_batch_perfs_torch(self) -> None:
        s_t = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float64)
        # 2 candidates, 3 samples, 2 outputs
        pred = torch.zeros((3, 2, 2), dtype=torch.float64)
        pred[:, 0, 0] = s_t[:, 0]
        pred[:, 0, 1] = s_t[:, 1]
        pred[:, 1, 0] = s_t[:, 0] + 0.5
        pred[:, 1, 1] = s_t[:, 1] - 1.0
        perfs_mse = _compute_batch_perfs_torch(
            pred,
            s_t,
            is_r2=False,
            total_sum_of_squares=0.0,
            mse_div_4=False,
        )
        self.assertEqual(perfs_mse.shape, (2, 2))
        np.testing.assert_allclose(
            perfs_mse.numpy(),
            np.array([[0.0, 0.0], [0.25, 1.0]]),
            atol=1e-14,
        )

    def test_metrics_handle_non_finite_inputs(self) -> None:
        import math

        from gmdhtorch.metrics import nrmse

        yt = np.array([1.0, 2.0, 3.0])
        yp_clean = np.array([1.1, 1.9, 3.2])
        yp_nan = np.array([1.1, np.nan, 3.2])
        yp_inf = np.array([1.1, np.inf, 3.2])

        self.assertGreater(nrmse(yt, yp_clean), 0.0)
        self.assertTrue(math.isnan(nrmse(yt, yp_nan)))
        self.assertTrue(math.isnan(nrmse(yt, yp_inf)))



if __name__ == "__main__":
    unittest.main()
