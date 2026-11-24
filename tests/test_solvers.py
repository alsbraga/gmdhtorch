"""Tests for numerically stabilized PyTorch layer solvers (Truncated SVD, Pivoted QR, Ridge/GCV)."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from gmdhtorch import (
    GMDHLayer,
    StreamedQRAccumulator,
    fit_layer_neurons,
    gmdh_coeff_count,
    resolve_candidate_indices,
)
from sklearn.metrics import mean_squared_error


class TestSolvers(unittest.TestCase):
    def test_well_conditioned_all_solvers_agree(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.uniform(-1.0, 1.0, size=(80, 4))
        targets = (
            1.5
            - 0.8 * samples[:, 0]
            + 0.4 * samples[:, 0] ** 2
            + 1.2 * samples[:, 2]
            - 0.5 * samples[:, 0] * samples[:, 2]
        )

        i_mat = resolve_candidate_indices(4, 2) + 1
        c_count = gmdh_coeff_count(2, 2)

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for dev in devices:
            samples_t = torch.as_tensor(samples, dtype=torch.float64, device=dev)
            targets_t = torch.as_tensor(targets, dtype=torch.float64, device=dev)
            results: dict[str, np.ndarray] = {}
            for solver in ("svd", "qr", "ridge"):
                layer = GMDHLayer(
                    dtype=torch.float64,
                    device=dev,
                    layer_number=1,
                    degree=2,
                    input_count=2,
                    I=i_mat.copy(),
                    W=np.zeros((i_mat.shape[0], c_count)),
                )
                fit_layer_neurons(layer, samples_t, targets_t, solver=solver, alpha=1e-10)
                results[solver] = layer.W.detach().cpu().numpy().copy()

            for solver in ("qr", "ridge"):
                np.testing.assert_allclose(results[solver], results["svd"], atol=1e-6)

    def test_ill_conditioned_collinear_inputs_stabilized_solvers(self) -> None:
        rng = np.random.default_rng(99)
        n_train = 60
        n_val = 60
        base_signal = np.sin(np.linspace(0.0, 4.0, n_train + n_val))
        x1 = base_signal
        x2 = base_signal + 1e-14 * rng.normal(size=n_train + n_val)
        all_samples = np.column_stack([x1, x2])
        true_y = 0.5 + 1.2 * base_signal - 0.3 * (base_signal**2)
        noisy_y = true_y + 0.01 * rng.normal(size=n_train + n_val)

        train_s, val_s = all_samples[:n_train], all_samples[n_train:]
        train_y, val_y = noisy_y[:n_train], true_y[n_train:]

        def make_layer() -> GMDHLayer:
            return GMDHLayer(
                layer_number=2,
                degree=2,
                input_count=2,
                I=np.array([[1, 2]]),
                W=np.zeros((1, 6)),
            )

        train_s_t = torch.as_tensor(train_s, dtype=torch.float64)
        train_y_t = torch.as_tensor(train_y, dtype=torch.float64)
        for solver, alpha in [("svd", 1e-6), ("qr", 1e-6), ("ridge", 1e-4), ("ridge", "gcv")]:
            layer_stab = make_layer()
            fit_layer_neurons(
                layer_stab,
                train_s_t,
                train_y_t,
                solver=solver,
                rcond=1e-10,
                alpha=alpha,
                equilibrate=True,
            )
            stab_norm = float(torch.linalg.norm(layer_stab.W).item())
            val_pred = layer_stab(torch.as_tensor(val_s, dtype=torch.float64)).detach().cpu().numpy().ravel()
            val_mse = float(mean_squared_error(val_y, val_pred))
            self.assertLess(stab_norm, 50.0)
            self.assertLess(val_mse, 1e-3)

    def test_streamed_qr_accumulator_matches_full_batch_and_closed_form_stats(self) -> None:
        rng = np.random.default_rng(2026)
        samples = rng.uniform(-1.5, 1.5, size=(120, 4))
        targets = (
            0.7
            - 0.4 * samples[:, 0]
            + 0.9 * samples[:, 1] ** 2
            - 0.6 * samples[:, 2]
            + 0.3 * samples[:, 0] * samples[:, 3]
            + 0.02 * rng.normal(size=120)
        )
        i_mat = resolve_candidate_indices(4, 2) + 1
        idx0 = i_mat - 1
        c_count = gmdh_coeff_count(2, 2)

        samples_t = torch.as_tensor(samples, dtype=torch.float64)
        targets_t = torch.as_tensor(targets, dtype=torch.float64)
        for solver, alpha in [("svd", 1e-6), ("qr", 1e-6), ("ridge", 1e-3), ("ridge", "gcv")]:
            layer_full = GMDHLayer(
                layer_number=1,
                degree=2,
                input_count=2,
                I=i_mat.copy(),
                W=np.zeros((i_mat.shape[0], c_count)),
            )
            fit_layer_neurons(
                layer_full,
                samples_t,
                targets_t,
                solver=solver,
                rcond=1e-12,
                alpha=alpha,
                equilibrate=True,
            )

            acc = StreamedQRAccumulator(
                candidate_count=idx0.shape[0],
                coeff_count=c_count,
                target_count=1,
                device=torch.device("cpu"),
            )
            # Feed in 4 unequal mini-batches
            for start, end in [(0, 25), (25, 60), (60, 95), (95, 120)]:
                xb = torch.as_tensor(samples[start:end], dtype=torch.float64)
                yb = torch.as_tensor(targets[start:end], dtype=torch.float64)
                y64 = acc.update_targets(yb)
                design = layer_full.preprocess_batch(xb, idx0_subset=torch.as_tensor(idx0, dtype=torch.long))
                acc.update_chunk(0, idx0.shape[0], design, y64)

            w_stream, mean_cf, std_cf = acc.solve(
                solver=solver,
                rcond=1e-12,
                alpha=alpha,
                equilibrate=True,
            )
            w_s = w_stream[:, 0, :].numpy()
            w_f = layer_full.W.detach().cpu().numpy()
            np.testing.assert_allclose(w_s, w_f, rtol=1e-9, atol=1e-9)

            # Verify closed-form mean and std against explicit pass over samples
            layer_full.W.data = w_stream[:, 0, :].clone()
            explicit_out = layer_full(torch.as_tensor(samples, dtype=torch.float64)).detach().numpy()
            exp_mean = np.mean(explicit_out, axis=0)
            exp_std = np.std(explicit_out, axis=0, ddof=0)
            np.testing.assert_allclose(mean_cf[:, 0].numpy(), exp_mean, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(std_cf[:, 0].numpy(), exp_std, rtol=1e-11, atol=1e-11)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS backend not available")
    def test_mps_streamed_qr_accumulator_float32(self) -> None:
        rng = np.random.default_rng(321)
        samples = rng.normal(size=(80, 4))
        targets = 1.0 + 2.0 * samples[:, 0] - 0.5 * samples[:, 1]
        idx0 = resolve_candidate_indices(4, 2)
        c_count = gmdh_coeff_count(2, 2)
        mps_dev = torch.device("mps")
        samples_t = torch.as_tensor(samples, dtype=torch.float32, device=mps_dev)
        targets_t = torch.as_tensor(targets, dtype=torch.float32, device=mps_dev)

        for solver, alpha in [("svd", 1e-6), ("qr", 1e-6), ("ridge", 1e-4)]:
            layer_full = GMDHLayer(
                idx0=idx0,
                W=np.zeros((idx0.shape[0], c_count), dtype=np.float32),
                degree=2,
                dtype=torch.float32,
                device=mps_dev,
            )
            fit_layer_neurons(layer_full, samples_t, targets_t, solver=solver, alpha=alpha)

            acc = StreamedQRAccumulator(
                candidate_count=idx0.shape[0],
                coeff_count=c_count,
                target_count=1,
                device=mps_dev,
            )
            self.assertEqual(acc.dtype, torch.float32)
            for start, end in [(0, 40), (40, 80)]:
                xb = torch.as_tensor(samples[start:end], dtype=torch.float32, device=mps_dev)
                yb = torch.as_tensor(targets[start:end], dtype=torch.float32, device=mps_dev)
                y_acc = acc.update_targets(yb)
                design = layer_full.preprocess_batch(
                    xb, idx0_subset=torch.as_tensor(idx0, dtype=torch.long, device=mps_dev)
                )
                acc.update_chunk(0, idx0.shape[0], design, y_acc)

            w_stream, _m, _s = acc.solve(solver=solver, alpha=alpha)
            self.assertEqual(w_stream.dtype, torch.float32)
            self.assertEqual(w_stream.device.type, "mps")
            np.testing.assert_allclose(
                w_stream[:, 0, :].cpu().numpy(),
                layer_full.W.detach().cpu().numpy(),
                rtol=1e-4,
                atol=1e-4,
            )

    def test_qr_via_triangular_helper_and_singular_fallback(self) -> None:
        from gmdhtorch.solvers import _solve_qr_via_triangular, _solve_svd_batch_torch

        rng = np.random.default_rng(77)
        x_well = torch.as_tensor(rng.normal(size=(4, 40, 6)), dtype=torch.float32)
        y = torch.as_tensor(rng.normal(size=(40,)), dtype=torch.float32)
        y_rhs = y.view(1, -1, 1).expand(4, -1, 1)

        fallback_called = False

        def _no_fallback() -> torch.Tensor:
            nonlocal fallback_called
            fallback_called = True
            return _solve_svd_batch_torch(x_well, y, rcond=1e-5).unsqueeze(-1)

        sol_tri = _solve_qr_via_triangular(x_well, y_rhs, cond=1e-5, svd_fallback=_no_fallback)
        self.assertFalse(fallback_called)
        sol_svd = _solve_svd_batch_torch(x_well, y, rcond=1e-5).unsqueeze(-1)
        np.testing.assert_allclose(sol_tri.numpy(), sol_svd.numpy(), rtol=1e-4, atol=1e-4)

        # Rank-deficient / collinear matrix must trigger SVD fallback cleanly
        x_sing = x_well.clone()
        x_sing[1, :, 3] = 2.0 * x_sing[1, :, 1]
        fallback_hits = 0

        def _sing_fallback() -> torch.Tensor:
            nonlocal fallback_hits
            fallback_hits += 1
            return _solve_svd_batch_torch(x_sing, y, rcond=1e-5).unsqueeze(-1)

        sol_sing = _solve_qr_via_triangular(x_sing, y_rhs, cond=1e-5, svd_fallback=_sing_fallback)
        self.assertEqual(fallback_hits, 1)
        self.assertTrue(torch.all(torch.isfinite(sol_sing)))


if __name__ == "__main__":
    unittest.main()
