"""Tests for GMDHModel, GMDHLayer, GMDHNorm, GMDHActivation, SoftClip, GMDHOutput,
ModelIncrementer, multi-output GMDH, composite nn.Sequential integration, and Scikit-Learn API.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gmdhtorch import (
    GMDHActivation,
    GMDHClassifier,
    GMDHLayer,
    GMDHModel,
    GMDHNorm,
    GMDHOutput,
    GMDHRegressor,
    LayerContext,
    ModelIncrementer,
    SelectBest,
    SelectMedian,
    SelectRelativeThreshold,
    SoftClip,
    combi_layers,
    default_layers,
    proximal_penalty,
)


def _make_synthetic_samples(
    rows: int,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic 5-input polynomial dataset with train, selection, and validation splits."""
    gen = np.random.default_rng(rng)
    inputs = 2.0 * gen.random((rows, 5)) - 1.0
    coef = 2.0 * gen.random((5, 1)) - 1.0
    tmp = np.column_stack(
        [inputs[:, 0] ** 4, inputs[:, 1] ** 3, inputs[:, 2] ** 2, inputs[:, 3], inputs[:, 3]]
    )
    outputs = (tmp @ coef).ravel()
    indexes = np.arange(1, rows + 1)
    val_mask = (indexes % 4) == 0
    sel_mask = (indexes % 4) == 3
    trn_mask = ~val_mask & ~sel_mask
    return (
        inputs[trn_mask],
        outputs[trn_mask],
        inputs[sel_mask],
        outputs[sel_mask],
        inputs[val_mask],
        outputs[val_mask],
    )


class TestNetworkAndTrain(unittest.TestCase):
    def test_default_and_combi_constructive_training_and_trimming(self) -> None:
        ts, tt, ss, st, *_ = _make_synthetic_samples(90, rng=10)

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")

        for dev in devices:
            with self.subTest(device=dev):
                ts_t = torch.as_tensor(ts, dtype=torch.float64, device=dev)
                tt_t = torch.as_tensor(tt, dtype=torch.float64, device=dev)
                ss_t = torch.as_tensor(ss, dtype=torch.float64, device=dev)
                st_t = torch.as_tensor(st, dtype=torch.float64, device=dev)

                # 1. Default O-GMDH
                model_o = GMDHModel(device=dev)
                inc_o = ModelIncrementer(
                    model_o,
                    layers=default_layers(degree=2, min_neurons=1, max_neurons=15),
                    max_layers=5,
                    logs=False,
                )
                while inc_o.growing:
                    inc_o.observe(tt_t, x=ts_t)
                    inc_o.step([(ss_t, st_t)])
                model_o.trim()
                self.assertEqual(model_o.layers[-1].neuron_count, 1)
                self.assertLess(inc_o.perf, 0.05)

                # 2. Combi C-GMDH
                model_c = GMDHModel(device=dev)
                inc_c = ModelIncrementer(
                    model_c,
                    layers=combi_layers(max_degree=3, min_neurons=1, max_neurons=15),
                    max_layers=4,
                    logs=False,
                )
                while inc_c.growing:
                    inc_c.observe(tt_t, x=ts_t)
                    inc_c.step([(ss_t, st_t)])
                model_c.trim()
                self.assertEqual(model_c.layers[-1].neuron_count, 1)
                out_c = model_c.predict_numpy(ss).ravel()
                self.assertLess(float(mean_squared_error(st, out_c)), 0.1)
                self.assertGreaterEqual(model_c.selected_inputs.size, 1)

                txt = model_c.describe(print_weights=True, print_output=False)
                self.assertIn("The network has", txt)

    def test_combi_max_degree_caps_input_count(self) -> None:
        ts, tt, ss, st, *_ = _make_synthetic_samples(90, rng=15)
        ts_t = torch.as_tensor(ts, dtype=torch.float64)
        tt_t = torch.as_tensor(tt, dtype=torch.float64)
        ss_t = torch.as_tensor(ss, dtype=torch.float64)
        st_t = torch.as_tensor(st, dtype=torch.float64)
        model = GMDHModel()
        inc = ModelIncrementer(
            model,
            layers=combi_layers(max_degree=3),
            max_layers=5,
            break_at_layer=5,
            min_grad=0.0,
            logs=False,
        )
        while inc.growing:
            inc.observe(tt_t, x=ts_t)
            inc.step([(ss_t, st_t)])

        self.assertEqual(model.layer_count, 5)
        for layer_idx, layer in enumerate(model.layers, start=1):
            expected_inputs = min(layer_idx, 3)
            self.assertEqual(layer.input_count, expected_inputs)
            self.assertEqual(layer.degree, expected_inputs)
            self.assertEqual(layer.I.shape[1], expected_inputs)

    def test_canonical_minibatch_loop_with_norm_softclip_and_tuning(self) -> None:
        ts, tt, ss, st, *_ = _make_synthetic_samples(100, rng=33)
        train_ds = TensorDataset(
            torch.as_tensor(ts, dtype=torch.float64),
            torch.as_tensor(tt, dtype=torch.float64),
        )
        select_ds = TensorDataset(
            torch.as_tensor(ss, dtype=torch.float64),
            torch.as_tensor(st, dtype=torch.float64),
        )
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        select_loader = DataLoader(select_ds, batch_size=50, shuffle=False)

        model = GMDHModel(GMDHNorm(), GMDHOutput(1))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def plan(ctx: LayerContext) -> list[nn.Module]:
            k = min(ctx.layer_index, 2)
            return [
                GMDHLayer(inputs=k, degree=2, forward=True, max_neurons=10),
                GMDHNorm(affine=True),
                GMDHActivation(SoftClip(bound=3.0)),
            ]

        inc = ModelIncrementer(
            model,
            optimizer=opt,
            plan_next=plan,
            select=SelectMedian(),
            max_layers=3,
            gradient_during_growth=True,
            logs=False,
        )

        loss_fn = nn.MSELoss()
        epoch = 0
        while not inc.stopped:
            for xb, yb in train_loader:
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb) + 1e-4 * proximal_penalty(model)
                if inc.tuning and pred.requires_grad:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if inc.growing:
                    inc.observe(yb, x=xb)
            if inc.growing:
                inc.step(select_loader)
            else:
                epoch += 1
                if epoch >= 3:
                    inc.stop()

        model.trim(optimizer=opt)
        self.assertEqual(model.layers[-1].neuron_count, 1)
        val_pred = model.predict_numpy(ss).ravel()
        self.assertLess(float(mean_squared_error(st, val_pred)), 0.05)

        # Verify O5 instrumentation
        self.assertGreater(model.parameter_count(trainable_only=True), 0)
        self.assertGreater(model.mac_count(), 0)
        c_stats = model.connectivity_stats()
        self.assertEqual(c_stats["layer_count"], model.layer_count)
        g_norms = model.gradient_norms()
        self.assertEqual(len(g_norms), model.layer_count)

    def test_passthrough_preserves_locked_forwarders_under_norm_and_activation(self) -> None:
        ts, tt, ss, st, *_ = _make_synthetic_samples(60, rng=7)
        model2 = GMDHModel()
        inc2 = ModelIncrementer(
            model2,
            layers=[GMDHLayer(inputs=2, degree=2, forward=True)],
            norm=True,
            activation=nn.Tanh(),
            max_layers=3,
            min_layers=3,
            logs=False,
        )
        # Grow 2 layers so [GMDHLayer(1), GMDHNorm, GMDHActivation, GMDHLayer(2), GMDHOutput] are all in model2
        ts_t = torch.as_tensor(ts[:, :4], dtype=torch.float64)
        tt_t = torch.as_tensor(tt, dtype=torch.float64)
        ss_t = torch.as_tensor(ss[:, :4], dtype=torch.float64)
        st_t = torch.as_tensor(st, dtype=torch.float64)
        inc2.observe(tt_t, x=ts_t)
        inc2.step([(ss_t, st_t)], final=False)
        inc2.observe(tt_t, x=ts_t)
        inc2.step([(ss_t, st_t)], final=False)

        layer1 = model2.layers[0]
        locked_mask = layer1.lock == 1
        self.assertEqual(int(np.sum(locked_mask)), 4)

        test_in = torch.tensor([[2.5, -3.0, 4.0, -5.0]], dtype=torch.float64)
        # Propagate through [GMDHLayer(1), GMDHNorm, GMDHActivation] (indices 0, 1, 2 in model2)
        h = test_in
        for idx in range(3):
            h = model2[idx](h)
        out = h.detach().cpu().numpy()
        self.assertTrue(np.all(np.abs(out[:, ~locked_mask]) <= 1.0))
        np.testing.assert_allclose(out[:, locked_mask], test_in.numpy(), atol=1e-15)

    def test_composite_sequential_with_frozen_upstream_feature_extractor(self) -> None:
        ts, tt, ss, st, *_ = _make_synthetic_samples(80, rng=19)
        upstream = nn.Linear(5, 4, dtype=torch.float64)
        with torch.no_grad():
            upstream.weight.copy_(torch.eye(4, 5, dtype=torch.float64))
            upstream.bias.zero_()
        for p in upstream.parameters():
            p.requires_grad_(False)

        gmdh_head = GMDHModel(GMDHOutput(1))
        composite = nn.Sequential(upstream, gmdh_head)

        inc = ModelIncrementer(
            gmdh_head,
            layers=default_layers(degree=2, max_neurons=10),
            max_layers=2,
            logs=False,
        )
        x_train = torch.as_tensor(ts, dtype=torch.float64)
        y_train = torch.as_tensor(tt, dtype=torch.float64)
        x_sel = torch.as_tensor(ss, dtype=torch.float64)
        y_sel = torch.as_tensor(st, dtype=torch.float64)

        while inc.growing:
            _ = composite(x_train)
            inc.observe(y_train)
            inc.step([(upstream(x_sel), y_sel)])

        gmdh_head.trim()
        pred = composite(x_sel).detach().numpy().ravel()
        self.assertEqual(pred.shape, st.shape)
        self.assertEqual(gmdh_head.input_count, 4)

    def test_multi_output_gmdh_option2_selection_and_pruning(self) -> None:
        rng = np.random.default_rng(77)
        x = rng.uniform(-1.0, 1.0, size=(120, 4))
        y1 = 0.5 + 1.2 * x[:, 0] - 0.4 * x[:, 1] ** 2
        y2 = -0.3 + 0.8 * x[:, 2] * x[:, 3] + 0.5 * x[:, 2]
        y = np.column_stack([y1, y2])

        t_x, s_x = x[:80], x[80:]
        t_y, s_y = y[:80], y[80:]

        t_x_t = torch.as_tensor(t_x, dtype=torch.float64)
        t_y_t = torch.as_tensor(t_y, dtype=torch.float64)
        s_x_t = torch.as_tensor(s_x, dtype=torch.float64)
        s_y_t = torch.as_tensor(s_y, dtype=torch.float64)

        model = GMDHModel(GMDHOutput(2))
        inc = ModelIncrementer(
            model,
            layers=combi_layers(max_degree=2, max_neurons=8),
            select=SelectBest(6),
            max_layers=3,
            logs=False,
        )
        while inc.growing:
            inc.observe(t_y_t, x=t_x_t)
            inc.step([(s_x_t, s_y_t)])
        model.trim()

        pred = model.predict_numpy(s_x)
        self.assertEqual(pred.shape, (40, 2))
        self.assertLess(float(mean_squared_error(s_y[:, 0], pred[:, 0])), 1e-4)
        self.assertLess(float(mean_squared_error(s_y[:, 1], pred[:, 1])), 1e-4)
        self.assertLessEqual(model.layers[-1].neuron_count, 2)

    def test_sklearn_regressor_and_classifier(self) -> None:
        ts, tt, ss, st, *_ = _make_synthetic_samples(90, rng=21)
        reg = GMDHRegressor(solver="svd", forward_inputs=True, max_layers=5, random_state=21)
        reg.fit(ts, tt, select_X=ss, select_y=st)
        r2 = reg.score(ss, st)
        self.assertGreater(r2, 0.94)

        x_iris, y_iris = load_iris(return_X_y=True)
        clf = GMDHClassifier(mode="ovr", solver="ridge", alpha=1e-4, shuffle_data=True, random_state=42)
        clf.fit(np.asarray(x_iris, dtype=np.float64), np.asarray(y_iris, dtype=int))
        acc = clf.score(np.asarray(x_iris, dtype=np.float64), np.asarray(y_iris, dtype=int))
        self.assertGreater(acc, 0.92)

    def test_evaluate_stopping_update_layer_and_remove_last_layer(self) -> None:
        ts, tt, ss, st, *_ = _make_synthetic_samples(80, rng=33)

        # 1. evaluate_stopping=False (via gradient_during_growth=True default) ignores min_grad/min_perf/MSE-increase
        #    but still stops at max_layers:
        model = GMDHModel(GMDHOutput(1))
        inc = ModelIncrementer(
            model,
            layer=GMDHLayer(inputs=2, degree=1, interaction=2, forward=False, max_neurons=6),
            activation=SoftClip(3.0),
            norm=True,
            min_layers=0,
            max_layers=3,
            min_grad=1e9,  # Impossible gradient threshold; ignored when evaluate_stopping resolves to False
            min_performance=1e9,  # Impossible perf threshold; ignored when evaluate_stopping resolves to False
            gradient_during_growth=True,
            logs=False,
        )

        # Cannot call remove_last_layer or update_layer before layer 1 exists
        ts_t = torch.as_tensor(ts, dtype=torch.float64)
        tt_t = torch.as_tensor(tt, dtype=torch.float64)
        ss_t = torch.as_tensor(ss, dtype=torch.float64)
        st_t = torch.as_tensor(st, dtype=torch.float64)
        with self.assertRaises(RuntimeError):
            inc.update_layer([(ss_t, st_t)])
        with self.assertRaises(RuntimeError):
            inc.remove_last_layer()

        # Cycle 1
        inc.observe(tt_t, x=ts_t)
        cont1 = inc.step([(ss_t, st_t)])
        self.assertTrue(cont1)
        self.assertEqual(model.gmdh_layer_count, 1)

        # With only 1 layer, remove_last_layer must raise RuntimeError
        with self.assertRaises(RuntimeError):
            inc.remove_last_layer()

        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        inc.optimizer = opt
        loss_fn = nn.MSELoss()

        # Fine-tune layer 1 and call update_layer
        x_t = ts_t
        y_t = tt_t.unsqueeze(1)
        for _ in range(5):
            opt.zero_grad()
            loss = loss_fn(model(x_t), y_t)
            loss.backward()
            opt.step()

        perf1 = inc.update_layer([(ss_t, st_t)], train_loader=[(ts_t, tt_t)])
        self.assertAlmostEqual(perf1, inc.current_performance, places=12)
        self.assertEqual(len(inc.layer_perfs_tuned), 1)
        # Verify manual MSE/4 matches update_layer return value
        pred1 = model.predict_numpy(ss).ravel()
        expected_mse_div4 = float(np.mean((st - pred1) ** 2) / 4.0)
        self.assertAlmostEqual(perf1, expected_mse_div4, places=12)

        # Cycle 2 (adds GMDHActivation(SoftClip) parameter + Layer 2 W parameter -> 3 params total)
        inc.observe(tt_t, x=ts_t)
        cont2 = inc.step([(ss_t, st_t)])
        self.assertTrue(cont2)
        self.assertEqual(model.gmdh_layer_count, 2)
        self.assertEqual(len(opt.param_groups[0]["params"]), 3)

        # Test remove_last_layer() while growing restores layer 1 state & cleans optimizer
        inc.remove_last_layer()
        self.assertEqual(model.gmdh_layer_count, 1)
        self.assertEqual(len(opt.param_groups[0]["params"]), 2)
        self.assertAlmostEqual(inc.current_performance, perf1, places=12)
        pred1_restored = model.predict_numpy(ss).ravel()
        np.testing.assert_allclose(pred1_restored, pred1, atol=1e-12)

        # Re-grow Cycle 2 and Cycle 3 (hits max_layers=3 and finishes growth)
        inc.observe(tt_t, x=ts_t)
        self.assertTrue(inc.step([(ss_t, st_t)]))
        self.assertEqual(model.gmdh_layer_count, 2)

        # Test stop(remove_last=True) while growing removes layer 2 before trimming
        inc.stop(remove_last=True)
        self.assertFalse(inc.growing)
        self.assertEqual(model.gmdh_layer_count, 1)
        self.assertEqual(model.layers[-1].neuron_count, 1)

        # Outside growing state, remove_last_layer() and stop(remove_last=True) must raise
        with self.assertRaises(RuntimeError):
            inc.remove_last_layer()
        with self.assertRaises(RuntimeError):
            inc.stop(remove_last=True)

    def test_multi_output_update_layer_and_remove_last_layer(self) -> None:
        rng = np.random.default_rng(91)
        x = rng.uniform(-1.0, 1.0, size=(100, 4))
        y1 = 0.5 + 1.1 * x[:, 0] - 0.3 * x[:, 1] ** 2
        y2 = -0.2 + 0.7 * x[:, 2] * x[:, 3] + 0.4 * x[:, 2]
        y = np.column_stack([y1, y2])
        t_x, s_x = x[:70], x[70:]
        t_y, s_y = y[:70], y[70:]

        model = GMDHModel(GMDHOutput(2))
        inc = ModelIncrementer(
            model,
            layer=GMDHLayer(inputs=2, degree=2, forward=False),
            select=SelectBest(4),
            norm=True,
            activation=SoftClip(3.0),
            max_layers=4,
            gradient_during_growth=True,
            logs=False,
        )

        # Layer 1
        x_t = torch.as_tensor(t_x, dtype=torch.float64)
        y_t = torch.as_tensor(t_y, dtype=torch.float64)
        sx_t = torch.as_tensor(s_x, dtype=torch.float64)
        sy_t = torch.as_tensor(s_y, dtype=torch.float64)

        inc.observe(y_t, x=x_t)
        inc.step([(sx_t, sy_t)])
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        inc.optimizer = opt
        loss_fn = nn.MSELoss()

        for _ in range(4):
            opt.zero_grad()
            loss = loss_fn(model(x_t), y_t)
            loss.backward()
            opt.step()

        perf_l1 = inc.update_layer([(sx_t, sy_t)])
        idx_l1 = model.output_module.index.detach().cpu().tolist()
        self.assertEqual(len(idx_l1), 2)
        pred_l1 = model.predict_numpy(s_x)
        expected_perf_l1 = float(np.mean(np.mean((s_y - pred_l1) ** 2, axis=0) / 4.0))
        self.assertAlmostEqual(perf_l1, expected_perf_l1, places=12)

        # Layer 2
        inc.observe(y_t, x=x_t)
        inc.step([(sx_t, sy_t)])
        self.assertEqual(model.gmdh_layer_count, 2)
        inc.update_layer([(sx_t, sy_t)])

        # Remove layer 2 and verify both target output indices are restored to layer 1
        inc.stop(remove_last=True)
        self.assertEqual(model.gmdh_layer_count, 1)
        self.assertLessEqual(model.layers[-1].neuron_count, 2)
        pred_after = model.predict_numpy(s_x)
        np.testing.assert_allclose(pred_after, pred_l1, atol=1e-12)

    def test_eager_layer_materialization_and_flexible_tails(self) -> None:
        rng = np.random.default_rng(123)
        x = torch.as_tensor(rng.normal(size=(64, 4)), dtype=torch.float32)
        y = torch.stack([x[:, 0] * x[:, 1] + 0.5 * x[:, 2], -0.4 * x[:, 3]], dim=1)

        # 1. Headless GMDHModel ending directly in an eager GMDHLayer
        l1 = GMDHLayer(in_features=4, inputs=2, degree=1, interaction=2, init="kaiming", dtype=torch.float32, rng=1)
        l2 = GMDHLayer(
            in_features=l1.out_features,
            out_features=2,
            inputs=2,
            degree=1,
            interaction=2,
            init="xavier",
            dtype=torch.float32,
            rng=2,
        )
        headless = GMDHModel(l1, nn.ReLU(), l2, dtype=torch.float32)
        self.assertFalse(headless.has_output_module)
        self.assertEqual(headless(x).shape, (64, 2))
        headless.trim()
        self.assertEqual(headless(x).shape, (64, 2))

        # 2. Unsupported module type in GMDHModel raises TypeError
        with self.assertRaises(TypeError):
            GMDHModel(nn.Conv1d(4, 2, 1))

        # 3. GMDHOutput(dense=True) gradient training, importance scoring, pruning, and warm resize
        l1_d = GMDHLayer(in_features=4, inputs=2, degree=1, interaction=2, init="xavier", dtype=torch.float32, rng=3)
        out_head = GMDHOutput(2, dense=True, in_features=l1_d.out_features, dtype=torch.float32)
        model = GMDHModel(l1_d, SoftClip(3.0), out_head, dtype=torch.float32)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        loss_fn = nn.MSELoss()

        for _ in range(15):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            h_sel = model.forward_to_insertion_point(x)
            scores = out_head.importance_scores(h_sel)
        self.assertEqual(scores.shape, (l1_d.out_features,))
        keep_mask = SelectRelativeThreshold(0.1).keep_mask(scores, higher_is_better=True)
        self.assertTrue(np.any(keep_mask))

        kept_count = int(np.sum(keep_mask))
        model.prune(0, keep_mask, optimizer=opt)
        self.assertEqual(l1_d.neuron_count, kept_count)
        self.assertEqual(out_head.linear.in_features, kept_count)  # type: ignore[union-attr]

        # Warm resize with forward=True preserves exact model predictions
        with torch.no_grad():
            pred_before = model(x).clone()
        out_head.resize_warm(
            kept_count + 4,
            forward_slice=slice(4, 4 + kept_count),
            optimizer=opt,
        )
        # Verify forward slice reproduces pred_before when new candidate slots are zero
        h_aug = torch.cat([torch.ones(64, 4, dtype=torch.float32), model.forward_to_insertion_point(x)], dim=1)
        with torch.no_grad():
            pred_after = out_head(h_aug)
        torch.testing.assert_close(pred_after, pred_before, atol=1e-6, rtol=1e-6)

    def test_stage2_noop_composite_plan_next_and_piecewise_grow(self) -> None:
        # 1. No-op mode on pure MLP (no ModelIncrementerNull needed)
        mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        mlp_inc = ModelIncrementer(mlp, logs=False)
        self.assertFalse(mlp_inc.growing)
        self.assertTrue(mlp_inc.stopped)
        self.assertFalse(mlp_inc.grow())
        self.assertFalse(mlp_inc.step())

        # 2. Composite nn.Sequential(upstream, GMDHModel) with raw loader in step()
        rng = np.random.default_rng(19)
        ts, tt, ss, st, *_ = _make_synthetic_samples(80, rng=19)
        upstream = nn.Linear(5, 4, dtype=torch.float64)
        with torch.no_grad():
            upstream.weight.copy_(torch.eye(4, 5, dtype=torch.float64))
            upstream.bias.zero_()
        for p in upstream.parameters():
            p.requires_grad_(False)
        composite = nn.Sequential(upstream, GMDHModel(GMDHOutput(1)))
        comp_inc = ModelIncrementer(
            composite,
            layer=GMDHLayer(inputs=2, degree=2, max_neurons=10),
            max_layers=2,
            logs=False,
        )
        ts_t = torch.as_tensor(ts, dtype=torch.float64)
        tt_t = torch.as_tensor(tt, dtype=torch.float64)
        ss_t = torch.as_tensor(ss, dtype=torch.float64)
        st_t = torch.as_tensor(st, dtype=torch.float64)
        while comp_inc.growing:
            comp_inc.observe(tt_t, x=upstream(ts_t))
            comp_inc.step([(ss_t, st_t)])  # raw [(ss_t, st_t)] without manual upstream(ss_t)
        self.assertEqual(composite[1].gmdh_layer_count, 2)  # type: ignore[union-attr]

        # 3. plan_next() overrides Layer 1 even when called after observe()
        m_plan = GMDHModel(GMDHOutput(1))
        inc_plan = ModelIncrementer(
            m_plan,
            layer=GMDHLayer(inputs=2, degree=1, interaction=2, max_neurons=10),
            max_layers=1,
            logs=False,
        )
        inc_plan.observe(tt_t, x=ts_t)
        inc_plan.plan_next(layer=GMDHLayer(inputs=1, degree=1, interaction=1, max_neurons=10))
        inc_plan.step([(ss_t, st_t)])
        self.assertEqual(m_plan.layers[0].input_count, 1)

        # 4. Solver-free piecewise growth via inc.grow()
        x_np = rng.normal(size=(90, 4))
        y_np = np.column_stack(
            [
                x_np[:, 0] * x_np[:, 1] + 0.5 * x_np[:, 2],
                -0.3 * x_np[:, 3] + 0.4 * x_np[:, 0],
            ]
        )
        tr_ds = TensorDataset(
            torch.as_tensor(x_np[:60], dtype=torch.float32),
            torch.as_tensor(y_np[:60], dtype=torch.float32),
        )
        sl_ds = TensorDataset(
            torch.as_tensor(x_np[60:], dtype=torch.float32),
            torch.as_tensor(y_np[60:], dtype=torch.float32),
        )
        tr_loader = DataLoader(tr_ds, batch_size=60, shuffle=False)
        sl_loader = DataLoader(sl_ds, batch_size=30, shuffle=False)

        l1 = GMDHLayer(in_features=4, inputs=2, degree=1, interaction=2, init="xavier", dtype=torch.float32, rng=5)
        out_head = GMDHOutput(2, dense=True, in_features=l1.out_features, dtype=torch.float32)
        g_model = GMDHModel(l1, out_head, dtype=torch.float32)
        opt = torch.optim.Adam(g_model.parameters(), lr=0.05)
        g_inc = ModelIncrementer(
            g_model,
            optimizer=opt,
            layer=GMDHLayer(inputs=2, degree=1, interaction=2, forward=True, max_neurons=16, init="xavier"),
            activation=SoftClip(3.0),
            norm=True,
            select=SelectRelativeThreshold(0.01),
            final_select=SelectRelativeThreshold(0.05),
            max_layers=3,
            gradient_during_growth=True,
            solver=None,
            logs=False,
            rng=5,
        )

        loss_fn = nn.MSELoss()
        for epoch in range(1, 31):
            g_model.train()
            for xb, yb in tr_loader:
                opt.zero_grad()
                loss = loss_fn(g_model(xb), yb)
                loss.backward()
                opt.step()
            if g_inc.growing and epoch % 10 == 0:
                g_inc.grow(sl_loader, train_loader=tr_loader)

        self.assertTrue(g_inc.tuning)
        self.assertEqual(g_model.gmdh_layer_count, 3)
        self.assertLess(loss.item(), 0.2)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS backend not available")
    def test_mps_dtype_resolution_and_validation(self) -> None:
        mps_dev = torch.device("mps")

        # Auto-default to float32 on MPS and child inference
        l1 = GMDHLayer(in_features=4, inputs=2, degree=1, interaction=2, init="xavier", device="mps", rng=1)
        self.assertEqual(l1.dtype, torch.float32)
        self.assertEqual(l1.device.type, "mps")
        self.assertEqual(l1.acc_dtype, torch.float32)

        inferred_model = GMDHModel(l1)
        self.assertEqual(inferred_model.dtype, torch.float32)
        self.assertEqual(inferred_model.acc_dtype, torch.float32)
        self.assertEqual(inferred_model.device.type, "mps")

        # Explicit float64 on MPS should fail fast
        with self.assertRaises(ValueError):
            GMDHModel(device="mps", dtype=torch.float64)
        with self.assertRaises(ValueError):
            GMDHLayer(in_features=4, device="mps", dtype=torch.float64)

        # Headless model .to(...) updates dtype, device, acc_dtype without crashing
        headless = GMDHModel(output="none")
        self.assertEqual(headless.dtype, torch.float64)
        headless.to(device=mps_dev, dtype=torch.float32)
        self.assertEqual(headless.dtype, torch.float32)
        self.assertEqual(headless.acc_dtype, torch.float32)
        self.assertEqual(headless.device.type, "mps")

        # SoftClip.to("mps") and proximal_penalty on empty MPS model
        sc = SoftClip(3.0).to("mps")
        self.assertEqual(sc.device.type, "mps")
        empty_mps = GMDHModel(device="mps")
        pen = proximal_penalty(empty_mps)
        self.assertEqual(pen.device.type, "mps")
        self.assertEqual(pen.dtype, torch.float32)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS backend not available")
    def test_mps_norm_grow_and_classical_step(self) -> None:
        rng = np.random.default_rng(7)
        mps_dev = torch.device("mps")

        # GMDHNorm accumulate_batch + finalize_stats preserves float32 on MPS
        norm = GMDHNorm(dtype=torch.float32, device=mps_dev)
        x_mps = torch.as_tensor(rng.normal(size=(32, 4)), dtype=torch.float32, device=mps_dev)
        norm.accumulate_batch(x_mps[:16])
        norm.accumulate_batch(x_mps[16:])
        norm.finalize_stats()
        self.assertEqual(norm.mean.dtype, torch.float32)
        self.assertEqual(norm.mean.device.type, "mps")

        # Solver-free grow() + AdamW on MPS
        x_np = rng.normal(size=(90, 4))
        y_np = np.column_stack(
            [
                x_np[:, 0] * x_np[:, 1] + 0.5 * x_np[:, 2],
                -0.3 * x_np[:, 3] + 0.4 * x_np[:, 0],
            ]
        )
        tr_x = torch.as_tensor(x_np[:60], dtype=torch.float32, device=mps_dev)
        tr_y = torch.as_tensor(y_np[:60], dtype=torch.float32, device=mps_dev)
        sl_x = torch.as_tensor(x_np[60:], dtype=torch.float32, device=mps_dev)
        sl_y = torch.as_tensor(y_np[60:], dtype=torch.float32, device=mps_dev)

        l1 = GMDHLayer(
            in_features=4, inputs=2, degree=1, interaction=2, init="xavier",
            dtype=torch.float32, device=mps_dev, rng=5,
        )
        out_head = GMDHOutput(2, dense=True, in_features=l1.out_features, dtype=torch.float32, device=mps_dev)
        g_model = GMDHModel(l1, out_head, dtype=torch.float32, device=mps_dev)
        opt = torch.optim.AdamW(g_model.parameters(), lr=0.05)
        g_inc = ModelIncrementer(
            g_model,
            optimizer=opt,
            layer=GMDHLayer(
                inputs=2, degree=1, interaction=2, forward=True, max_neurons=16, init="xavier",
                dtype=torch.float32, device=mps_dev,
            ),
            activation=SoftClip(3.0, dtype=torch.float32, device=mps_dev),
            norm=True,
            select=SelectRelativeThreshold(0.01),
            final_select=SelectRelativeThreshold(0.05),
            max_layers=3,
            gradient_during_growth=True,
            solver=None,
            logs=False,
            rng=5,
        )
        loss_fn = nn.MSELoss()
        for epoch in range(1, 21):
            g_model.train()
            opt.zero_grad()
            loss = loss_fn(g_model(tr_x), tr_y)
            loss.backward()
            opt.step()
            if g_inc.growing and epoch % 5 == 0:
                g_inc.grow([(sl_x, None)], train_loader=[(tr_x, None)])

        self.assertTrue(g_inc.tuning)
        self.assertEqual(g_model.gmdh_layer_count, 3)

        # Classical observe + step + update_layer on MPS
        c_model = GMDHModel(
            GMDHNorm(dtype=torch.float32, device=mps_dev),
            GMDHOutput(1, dtype=torch.float32, device=mps_dev),
            dtype=torch.float32,
            device=mps_dev,
        )
        c_inc = ModelIncrementer(
            c_model,
            layer=default_layers(inputs=2, degree=2, max_neurons=10),
            norm=True,
            activation=SoftClip(3.0, dtype=torch.float32, device=mps_dev),
            max_layers=2,
            logs=False,
            rng=5,
        )
        c_inc.prepare([(tr_x, tr_y[:, 0])])
        while c_inc.growing:
            c_inc.observe(tr_y[:, 0], x=tr_x)
            c_inc.step([(sl_x, sl_y[:, 0])])
        c_inc.update_layer([(sl_x, sl_y[:, 0])], train_loader=[(tr_x, tr_y[:, 0])])
        self.assertGreaterEqual(c_model.gmdh_layer_count, 1)
        self.assertEqual(c_model(sl_x).device.type, "mps")
        self.assertEqual(c_model(sl_x).dtype, torch.float32)

    def test_gmdh_checkpoint_config_round_trip(self) -> None:
        rng = np.random.default_rng(19)
        x_np = rng.normal(size=(80, 4))
        y_np = np.column_stack([x_np[:, 0] + 0.5 * x_np[:, 1], -0.2 * x_np[:, 2] * x_np[:, 3]])
        x_t = torch.as_tensor(x_np, dtype=torch.float64)
        y_t = torch.as_tensor(y_np, dtype=torch.float64)

        l1 = GMDHLayer(in_features=4, inputs=2, degree=1, interaction=2, max_neurons=8, init="xavier", rng=1)
        out_head = GMDHOutput(2, dense=True, in_features=l1.out_features)
        model = GMDHModel(l1, out_head)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        inc = ModelIncrementer(
            model,
            optimizer=opt,
            layer=GMDHLayer(inputs=2, degree=1, interaction=2, forward=True, max_neurons=16, init="xavier"),
            activation=SoftClip(3.0),
            norm=GMDHNorm(affine=True),
            select=SelectRelativeThreshold(0.01),
            final_select=SelectRelativeThreshold(0.05),
            max_layers=3,
            gradient_during_growth=True,
            solver=None,
            logs=False,
            rng=1,
        )

        # Grow to 2 layers (still growing=True), train 1 step, and save mid-growth
        loss_fn = nn.MSELoss()
        inc.grow([(x_t, None)], train_loader=[(x_t, None)])
        self.assertTrue(inc.growing)
        model.train()
        opt.zero_grad()
        loss_fn(model(x_t), y_t).backward()
        opt.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "gmdh_mid.pth"
            torch.save(
                {
                    "model_config": model.architecture_config(),
                    "model_state": model.state_dict(),
                    "optim_state": opt.state_dict(),
                    "inc_state": inc.state_dict(),
                },
                ckpt_path,
            )

            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            restored = GMDHModel.from_config(loaded["model_config"], dtype=torch.float64, device="cpu")
            restored.load_state_dict(loaded["model_state"])
            restored_opt = torch.optim.AdamW(restored.parameters(), lr=1e-2)
            restored_opt.load_state_dict(loaded["optim_state"])
            restored_inc = ModelIncrementer(
                restored,
                optimizer=restored_opt,
                layer=GMDHLayer(inputs=2, degree=1, interaction=2, forward=True, max_neurons=16, init="xavier"),
                activation=SoftClip(3.0),
                norm=GMDHNorm(affine=True),
                max_layers=3,
                gradient_during_growth=True,
                solver=None,
                logs=False,
                rng=1,
            )
            restored_inc.load_state_dict(loaded["inc_state"])

            model.eval()
            restored.eval()
            np.testing.assert_allclose(
                restored(x_t).detach().numpy(),
                model(x_t).detach().numpy(),
                rtol=1e-12,
                atol=1e-12,
            )
            self.assertTrue(restored_inc.growing)

            # Verify locked forwarder gradient hook and affine hook remain active after restore
            restored.train()
            restored_opt.zero_grad()
            loss_fn(restored(x_t), y_t).backward()
            l2 = restored.gmdh_layers[-1]
            self.assertTrue(bool(torch.any(l2.lock_mask).item()))
            self.assertTrue(torch.all(l2.W.grad[l2.lock_mask] == 0.0).item())
            restored_opt.step()

            # Grow to cap (switches to tuning / growing=False), save post-growth, and restore
            restored_inc.grow([(x_t, None)], train_loader=[(x_t, None)])
            restored_inc.grow([(x_t, None)], train_loader=[(x_t, None)])
            self.assertFalse(restored_inc.growing)
            self.assertTrue(restored_inc.tuning)

            ckpt_post = Path(tmpdir) / "gmdh_post.pth"
            torch.save(
                {
                    "model_config": restored.architecture_config(),
                    "model_state": restored.state_dict(),
                    "inc_state": restored_inc.state_dict(),
                },
                ckpt_post,
            )
            loaded_post = torch.load(ckpt_post, map_location="cpu", weights_only=True)
            post_model = GMDHModel.from_config(loaded_post["model_config"], dtype=torch.float64, device="cpu")
            post_model.load_state_dict(loaded_post["model_state"])
            post_inc = ModelIncrementer(post_model, max_layers=3, solver=None, logs=False)
            post_inc.load_state_dict(loaded_post["inc_state"])
            self.assertFalse(post_inc.growing)
            self.assertTrue(post_inc.tuning)

            if torch.backends.mps.is_available():
                mps_model = GMDHModel.from_config(
                    loaded["model_config"], dtype=torch.float32, device="mps"
                )
                mps_model.load_state_dict(loaded["model_state"])
                mps_out = mps_model(x_t.to(device="mps", dtype=torch.float32)).detach().cpu().numpy()
                np.testing.assert_allclose(mps_out, model(x_t).detach().numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
