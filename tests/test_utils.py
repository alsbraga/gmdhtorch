"""Tests for PyTorch resource controls, gmdhtorch.utils helpers, and autocast."""

from __future__ import annotations

import sys
import unittest

import numpy as np
import torch
from gmdhtorch import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_BYTES,
    DEFAULT_MAX_OBSERVE_BYTES,
    GMDHLayer,
    GMDHModel,
    GMDHOutput,
    ModelIncrementer,
    SoftClip,
    default_layers,
    effective_batch_size,
    resolve_acc_dtype,
    resolve_dtype,
)
from torch import nn


class TestUtilsAndResourceControls(unittest.TestCase):
    def test_effective_batch_size_bounds_and_unbounded(self) -> None:
        # Both explicit requested_batch_size and max_chunk_bytes bounded
        b_cap = effective_batch_size(8192, n_samples=1000, coeff_count=10, elem_size=8, max_chunk_bytes=1024 * 1024)
        self.assertGreaterEqual(b_cap, 16)
        self.assertLessEqual(b_cap, 8192)

        # Requested None -> purely byte-bounded (exceeds DEFAULT_CHUNK_SIZE when N is tiny)
        b_unreq = effective_batch_size(None, n_samples=10, coeff_count=4, elem_size=4, max_chunk_bytes=64 * 1024 * 1024)
        self.assertGreater(b_unreq, DEFAULT_CHUNK_SIZE)

        # max_chunk_bytes None -> purely count-bounded
        self.assertEqual(
            effective_batch_size(1234, n_samples=100000, coeff_count=10, elem_size=8, max_chunk_bytes=None),
            1234,
        )

        # Both None -> effectively unbounded
        self.assertEqual(
            effective_batch_size(None, n_samples=1000, coeff_count=10, elem_size=8, max_chunk_bytes=None),
            sys.maxsize,
        )

    def test_resolve_dtype_and_strict_forward_mismatch(self) -> None:
        self.assertEqual(resolve_dtype(None, device="cpu"), torch.float64)
        self.assertEqual(resolve_dtype("float32", device="cpu"), torch.float32)
        self.assertEqual(resolve_acc_dtype("mps", torch.float64), torch.float32)
        self.assertEqual(resolve_acc_dtype("cpu", torch.float64), torch.float64)

        l1 = GMDHLayer(in_features=4, inputs=2, degree=1, interaction=2, init="xavier", dtype=torch.float32, rng=1)
        model = GMDHModel(l1, GMDHOutput(1, dense=True, in_features=l1.out_features, dtype=torch.float32), dtype=torch.float32)
        # Mismatched Tensor dtype must raise loudly rather than being silently cast
        x_f64 = torch.randn(8, 4, dtype=torch.float64)
        with self.assertRaises(RuntimeError):
            model(x_f64)

        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            x_cpu = torch.randn(8, 4, dtype=torch.float32)
            with self.assertRaises(RuntimeError):
                model_cuda(x_cpu)

    def test_incrementer_memory_budgets_parity_and_dense_noop(self) -> None:
        rng = np.random.default_rng(77)
        x_np = rng.normal(size=(120, 4))
        y_np = (0.5 + 1.1 * x_np[:, 0] - 0.7 * x_np[:, 1] * x_np[:, 2]).reshape(-1, 1)
        x = torch.as_tensor(x_np, dtype=torch.float64)
        y = torch.as_tensor(y_np, dtype=torch.float64)

        def _fit_with_budgets(chunk_size: int | None, max_chunk_bytes: int | None, max_observe_bytes: int | None) -> np.ndarray:
            net = GMDHModel(GMDHOutput(1, dtype=torch.float64), input_count=4, dtype=torch.float64)
            inc = ModelIncrementer(
                net,
                layer=GMDHLayer(inputs=2, degree=2, max_neurons=10, forward=True),
                max_layers=2,
                chunk_size=chunk_size,
                max_chunk_bytes=max_chunk_bytes,
                max_observe_bytes=max_observe_bytes,
                logs=False,
            )
            while inc.growing:
                # Feed in multiple small batches so max_observe_bytes=0 triggers streamed QR
                for s in range(0, 80, 20):
                    inc.observe(y[s : s + 20], x=x[s : s + 20])
                inc.step([(x[80:], y[80:])])
            return net.predict_numpy(x[80:])

        # Strict PyTorch API: raw NumPy inputs fail fast instead of silent coercion
        strict_net = GMDHModel(GMDHOutput(1, dtype=torch.float64), input_count=4, dtype=torch.float64)
        strict_inc = ModelIncrementer(strict_net, logs=False)
        with self.assertRaises(AttributeError):
            strict_inc.observe(y_np, x=x_np)  # type: ignore[arg-type]

        pred_mem = _fit_with_budgets(None, None, None)
        pred_stream = _fit_with_budgets(3, 1024, 0)
        np.testing.assert_allclose(pred_stream, pred_mem, rtol=1e-10, atol=1e-10)

        # Dense nn.Sequential wrapped by ModelIncrementer accepts identical resource params
        mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
        mlp_inc = ModelIncrementer(
            mlp,
            chunk_size=None,
            max_chunk_bytes=DEFAULT_MAX_CHUNK_BYTES,
            max_observe_bytes=DEFAULT_MAX_OBSERVE_BYTES,
            logs=False,
        )
        self.assertTrue(mlp_inc.stopped)
        self.assertIsNone(mlp_inc.chunk_size)
        self.assertEqual(mlp_inc.max_chunk_bytes, DEFAULT_MAX_CHUNK_BYTES)
        self.assertEqual(mlp_inc.max_observe_bytes, DEFAULT_MAX_OBSERVE_BYTES)

    def test_layer_chunk_override_and_autocast(self) -> None:
        l1 = GMDHLayer(
            in_features=6,
            inputs=2,
            degree=2,
            init="xavier",
            dtype=torch.float32,
            chunk_size=4,
            max_chunk_bytes=1024,
            rng=11,
        )
        model = GMDHModel(
            l1,
            SoftClip(3.0, dtype=torch.float32),
            GMDHOutput(2, dense=True, in_features=l1.out_features, dtype=torch.float32),
            dtype=torch.float32,
        )
        x = torch.randn(16, 6, dtype=torch.float32)
        out_chunked = model(x)
        out_single = l1(x, batch_size=100, max_chunk_bytes=None)
        self.assertEqual(out_chunked.shape, (16, 2))
        self.assertEqual(out_single.shape, (16, l1.out_features))

        if torch.cuda.is_available():
            model_cuda = model.to("cuda")
            x_cuda = x.to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                y_amp = model_cuda(x_cuda)
                loss = y_amp.sum()
            loss.backward()
            self.assertEqual(y_amp.shape, (16, 2))

        if torch.backends.mps.is_available():
            model_mps = model.to("mps")
            x_mps = x.to("mps")
            with torch.autocast("mps", dtype=torch.float16):
                y_mps = model_mps(x_mps)
                loss_mps = y_mps.sum()
            loss_mps.backward()
            self.assertEqual(y_mps.shape, (16, 2))
            self.assertEqual(y_mps.device.type, "mps")

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS backend not available")
    def test_mps_model_incrementer_streamed_qr_vs_in_memory_float32(self) -> None:
        rng = np.random.default_rng(123)
        x_tr = torch.as_tensor(rng.uniform(-1.0, 1.0, size=(80, 4)), dtype=torch.float32, device="mps")
        y_tr = (1.0 + 0.8 * x_tr[:, :1] - 0.5 * x_tr[:, 1:2] + 0.3 * x_tr[:, 0:1] * x_tr[:, 1:2])
        x_sel = torch.as_tensor(rng.uniform(-1.0, 1.0, size=(40, 4)), dtype=torch.float32, device="mps")
        y_sel = (1.0 + 0.8 * x_sel[:, :1] - 0.5 * x_sel[:, 1:2] + 0.3 * x_sel[:, 0:1] * x_sel[:, 1:2])

        net_inmem = GMDHModel(GMDHOutput(1, dtype=torch.float32, device="mps"), input_count=4, dtype=torch.float32, device="mps")
        inc_inmem = ModelIncrementer(
            net_inmem,
            layer=default_layers(degree=2, forward=False),
            max_layers=2,
            solver="qr",
            logs=False,
        )
        while inc_inmem.growing:
            inc_inmem.observe(y_tr, x=x_tr)
            inc_inmem.step([(x_sel, y_sel)])

        net_stream = GMDHModel(GMDHOutput(1, dtype=torch.float32, device="mps"), input_count=4, dtype=torch.float32, device="mps")
        inc_stream = ModelIncrementer(
            net_stream,
            layer=default_layers(degree=2, forward=False),
            max_layers=2,
            solver="qr",
            max_observe_bytes=128,
            logs=False,
        )
        while inc_stream.growing:
            inc_stream.observe(y_tr[:40], x=x_tr[:40])
            inc_stream.observe(y_tr[40:], x=x_tr[40:])
            inc_stream.step([(x_sel[:20], y_sel[:20]), (x_sel[20:], y_sel[20:])])

        out_inmem = net_inmem(x_sel).detach().cpu().numpy()
        out_stream = net_stream(x_sel).detach().cpu().numpy()
        np.testing.assert_allclose(out_stream, out_inmem, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
