"""Constructive GMDH growth and gradient-tuning lifecycle via ModelIncrementer."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import sys
from typing import Any, Callable, Iterable, Sequence
import warnings

import numpy as np
import torch
from torch import nn

from .network import (
    GMDHActivation,
    GMDHLayer,
    GMDHModel,
    GMDHNorm,
    GMDHOutput,
    LayerContext,
    PerformanceCriterion,
    default_layers,
    instantiate_activation_module,
)
from .polynomial import (
    gmdh_coeff_count,
    gmdh_preprocess_batch,
    precompute_layer_powers_torch,
    resolve_candidate_indices,
)
from .selection import (
    SelectMedian,
    SelectionStrategy,
    _compute_batch_perfs_torch,
    resolve_selection_strategy,
)
from .solvers import (
    Solver,
    StreamedQRAccumulator,
    _solve_batch_torch,
)
from .utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_BYTES,
    DEFAULT_MAX_OBSERVE_BYTES,
    effective_batch_size,
)

_UNSET: Any = object()


@dataclass
class TrainResults:
    """Training telemetry returned by ``ModelIncrementer.results``."""

    perf: float = -1.0
    peak: int = 0
    layer_perfs: list[float] = field(default_factory=list)
    layer_perfs_tuned: list[float] = field(default_factory=list)


@dataclass
class _LayerRecord:
    """Per-layer metadata enabling lightweight ``remove_last_layer()`` without state_dict snapshots."""

    best_out_indices: list[int]
    last_layer_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    closed_form_perf: float
    tuned_perf: float | None = None


def _instantiate_norm(
    spec: GMDHNorm | type[GMDHNorm] | Callable[[], GMDHNorm] | bool | None,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> GMDHNorm | None:
    if spec is None or spec is False:
        return None
    if spec is True:
        return GMDHNorm(dtype=dtype, device=device)
    if isinstance(spec, GMDHNorm):
        cloned = copy.deepcopy(spec)
        cloned.reset_accumulation()
        cloned.initialized.fill_(False)
        return cloned.to(dtype=dtype, device=device)
    if isinstance(spec, type) and issubclass(spec, GMDHNorm):
        return spec(dtype=dtype, device=device)
    if callable(spec):
        mod = spec()
        if isinstance(mod, GMDHNorm):
            return mod.to(dtype=dtype, device=device)
    raise TypeError(f"Invalid norm specification: {type(spec)!r}.")


class ModelIncrementer:
    """Orchestrates layer-by-layer closed-form GMDH growth and transitions to gradient tuning (§5.7)."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        layer: GMDHLayer | Callable[[LayerContext], Any] | Sequence[GMDHLayer] | None = None,
        layers: GMDHLayer | Callable[[LayerContext], Any] | Sequence[GMDHLayer] | None = None,
        plan_next: Callable[[LayerContext], Any] | None = None,
        activation: nn.Module | type[nn.Module] | Callable[[], nn.Module] | None = None,
        norm: GMDHNorm | type[GMDHNorm] | Callable[[], GMDHNorm] | bool | None = None,
        select: SelectionStrategy | str = SelectMedian(),
        final_select: SelectionStrategy | str | None = None,
        max_layers: int = 10,
        min_layers: int = 0,
        break_at_layer: int | None = None,
        min_grad: float = 1e-5,
        min_performance: float = 0.0,
        min_inputs: int = 1,
        criterion: PerformanceCriterion | str = "MSE",
        mse_div_4: bool = True,
        gradient_during_growth: bool = False,
        solver: Solver | str | None = "svd",
        rcond: float | None = None,
        alpha: float | str = 1e-6,
        equilibrate: bool = False,
        chunk_size: int | None = DEFAULT_CHUNK_SIZE,
        max_chunk_bytes: int | None = DEFAULT_MAX_CHUNK_BYTES,
        max_observe_bytes: int | None = DEFAULT_MAX_OBSERVE_BYTES,
        logs: bool = True,
        rng: int | np.random.Generator | None = None,
    ) -> None:
        self._root_model: nn.Module = model
        self.optimizer = optimizer
        self.chunk_size: int | None = int(chunk_size) if chunk_size is not None else None
        self.max_chunk_bytes: int | None = (
            int(max_chunk_bytes) if max_chunk_bytes is not None else None
        )
        self.max_observe_bytes: int | None = (
            int(max_observe_bytes) if max_observe_bytes is not None else None
        )
        self.layer_perfs: list[float] = []
        self.layer_perfs_tuned: list[float] = []
        self._layer_history: list[_LayerRecord] = []
        self.peak_memory_bytes: int = 0
        self._prev_min_perf: float = 0.0
        self._current_min_perf: float = -1.0
        self._last_layer_neuron_perfs: np.ndarray | None = None
        self._last_model_input_ptr: int | None = None
        self._last_model_input_shape: tuple[int, ...] | None = None
        self._last_pre_output: torch.Tensor | None = None
        self._model_hook: Any = None
        self._out_hook: Any = None
        self._pending_norm: GMDHNorm | None = None
        self._pending_activation: GMDHActivation | None = None
        self._last_layer_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._active_template: GMDHLayer | None = None
        self._active_cand_idx0: np.ndarray | None = None
        self._observed_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._observed_bytes: int = 0
        self._observed_samples: int = 0
        self._qr_acc: StreamedQRAccumulator | None = None
        self._target_y_sum: torch.Tensor | None = None
        self._target_y_count: int = 0

        if isinstance(model, GMDHModel):
            self.model: GMDHModel | None = model
            self._noop: bool = False
        else:
            found_gmdh = [m for m in model.modules() if isinstance(m, GMDHModel)]
            if len(found_gmdh) == 0:
                self.model = None
                self._noop = True
                self._state: str = "stopped"
                self.max_layers = 0
                self.gradient_during_growth = False
                return
            if len(found_gmdh) > 1:
                raise ValueError(
                    f"Found {len(found_gmdh)} GMDHModel instances inside {type(model).__name__}; "
                    "pass the target GMDHModel explicitly."
                )
            self.model = found_gmdh[0]
            self._noop = False

        assert self.model is not None

        eff_layer_spec = plan_next if plan_next is not None else (layers if layers is not None else layer)
        if eff_layer_spec is None:
            self._layer_factory: Callable[[LayerContext], Any] = default_layers()
        elif isinstance(eff_layer_spec, GMDHLayer):
            proto_layer = copy.deepcopy(eff_layer_spec)
            self._layer_factory = lambda ctx, _p=proto_layer: GMDHLayer(
                _p,
                dtype=self.model.dtype,
                device=self.model.device,
                layer_number=ctx.layer_no,
            )
        elif isinstance(eff_layer_spec, (list, tuple)):
            proto_list = [copy.deepcopy(item) for item in eff_layer_spec]
            if not proto_list:
                self._layer_factory = default_layers()
            elif all(isinstance(item, GMDHLayer) for item in proto_list):
                self._layer_factory = lambda ctx, _lst=proto_list: GMDHLayer(
                    _lst[min(max(0, ctx.layer_no - 1), len(_lst) - 1)],
                    dtype=self.model.dtype,
                    device=self.model.device,
                    layer_number=ctx.layer_no,
                )
            else:
                self._layer_factory = lambda _ctx, _lst=proto_list: [copy.deepcopy(m) for m in _lst]
        elif callable(eff_layer_spec):
            self._layer_factory = eff_layer_spec
        else:
            raise TypeError(
                f"`layer`/`layers` must be a GMDHLayer, sequence, or callable factory; got {type(eff_layer_spec)!r}."
            )

        self._default_activation_spec = activation
        self._default_norm_spec = norm
        self._default_select_strategy: SelectionStrategy = resolve_selection_strategy(select)
        self._final_select_strategy: SelectionStrategy = (
            resolve_selection_strategy(final_select)
            if final_select is not None
            else self._default_select_strategy
        )

        self._planned_layer: GMDHLayer | Callable[[LayerContext], Any] | None = None
        self._planned_activation_spec: Any = _UNSET
        self._planned_norm_spec: Any = _UNSET
        self._planned_select_strategy: SelectionStrategy | None = None

        self.max_layers = int(max_layers)
        self.min_layers = int(break_at_layer) if break_at_layer is not None else int(min_layers)
        self.min_grad = float(min_grad)
        self.min_performance = float(min_performance)
        self.min_inputs = int(min_inputs)

        crit_up = str(criterion).strip().upper()
        if crit_up == "R2":
            self._is_r2 = True
            self._crit_label = "1-R2"
        elif crit_up == "MSE":
            self._is_r2 = False
            self._crit_label = "MSE"
        else:
            raise ValueError("Valid `criterion` values are 'MSE' or 'R2'.")
        self.criterion = crit_up
        self.mse_div_4 = bool(mse_div_4)

        self.gradient_during_growth = bool(gradient_during_growth)
        self.solver: str | None = str(solver).strip().lower() if solver is not None else None
        self.rcond = float(rcond) if rcond is not None else None
        self.alpha = alpha
        self.equilibrate = bool(equilibrate)
        self.logs = bool(logs)
        self._rng = np.random.default_rng(rng)

        # Cached hook tensors for zero-copy reuse when model(xb) precedes observe(yb, x=xb)
        # or when GMDHModel is embedded inside an upstream nn.Sequential
        self._model_hook = self.model.register_forward_pre_hook(self._on_model_pre_forward)
        if self.model.has_output_module:
            self._out_hook = self.model.output_module.register_forward_pre_hook(self._on_output_pre_forward)

        # If model already has materialized GMDHLayers and a lazy dense GMDHOutput, adapt width now
        existing_layers = self.model.gmdh_layers
        if existing_layers and existing_layers[-1].is_materialized and self.model.has_output_module:
            out_mod = self.model.output_module
            if out_mod.dense and out_mod.linear is None:
                out_mod.adapt_width(existing_layers[-1].neuron_count, optimizer=self.optimizer)

        # Check if model starts with an uninitialized input-stage GMDHNorm
        has_uninit_input_norm = any(
            isinstance(m, GMDHNorm) and not bool(m.initialized.item())
            for m in self.model
        )
        if has_uninit_input_norm:
            self._state = "stats"
            if self.model.has_output_module:
                self.model.output_module.set_fallback(0.0)
        elif self.max_layers == 0 or (self.max_layers > 0 and self.model.gmdh_layer_count >= self.max_layers and self.solver is not None):
            self._state = "tuning"
        else:
            self._state = "accumulate"

    def _on_model_pre_forward(self, _mod: nn.Module, inputs: tuple[Any, ...]) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            x = inputs[0]
            self._last_model_input_ptr = x.data_ptr()
            self._last_model_input_shape = tuple(x.shape)

    def _on_output_pre_forward(self, _mod: nn.Module, inputs: tuple[Any, ...]) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            self._last_pre_output = inputs[0].detach()

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype if self.model is not None else torch.get_default_dtype()

    @property
    def acc_dtype(self) -> torch.dtype:
        return self.model.acc_dtype if self.model is not None else torch.float64

    @property
    def device(self) -> torch.device:
        return self.model.device if self.model is not None else torch.device("cpu")

    @property
    def growing(self) -> bool:
        return (not self._noop) and self._state in ("stats", "accumulate")

    @property
    def tuning(self) -> bool:
        if self._noop or self.model is None:
            return False
        return self._state == "tuning" or (
            self.gradient_during_growth and self.growing and self.model.gmdh_layer_count > 0
        )

    @property
    def stopped(self) -> bool:
        return self._noop or self._state == "stopped"

    @property
    def can_grow(self) -> bool:
        if self._noop or self.model is None or not self.growing:
            return False
        if self.max_layers > 0 and self.model.gmdh_layer_count >= self.max_layers:
            return False
        layers = self.model.gmdh_layers
        if layers and layers[-1].neuron_count < max(1, self.min_inputs):
            return False
        return True

    @property
    def perf(self) -> float:
        return self._current_min_perf

    @property
    def current_performance(self) -> float:
        return self._current_min_perf

    @property
    def results(self) -> TrainResults:
        return TrainResults(
            perf=self._current_min_perf,
            peak=self.peak_memory_bytes,
            layer_perfs=list(self.layer_perfs),
            layer_perfs_tuned=list(self.layer_perfs_tuned),
        )

    def state_dict(self) -> dict[str, Any]:
        """Return pure-primitive growth lifecycle state compatible with ``torch.load(..., weights_only=True)``."""
        return {
            "version": 1,
            "state": str(self._state),
            "layer_perfs": [float(v) for v in self.layer_perfs],
            "layer_perfs_tuned": [float(v) for v in self.layer_perfs_tuned],
            "current_min_perf": float(self._current_min_perf),
            "prev_min_perf": float(self._prev_min_perf),
            "last_layer_neuron_perfs": (
                [float(v) for v in self._last_layer_neuron_perfs.tolist()]
                if self._last_layer_neuron_perfs is not None
                else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> ModelIncrementer:
        """Restore growth lifecycle state saved by ``ModelIncrementer.state_dict()``."""
        if self._noop or self.model is None:
            if "state" in state:
                self._state = str(state["state"])
            return self
        restored_state = str(state.get("state", self._state))
        if restored_state not in ("stats", "accumulate", "tuning", "stopped"):
            raise ValueError(f"Invalid ModelIncrementer state {restored_state!r}.")
        self._state = restored_state
        self.layer_perfs = [float(v) for v in state.get("layer_perfs", [])]
        self.layer_perfs_tuned = [float(v) for v in state.get("layer_perfs_tuned", [])]
        self._current_min_perf = float(state.get("current_min_perf", -1.0))
        self._prev_min_perf = float(state.get("prev_min_perf", 0.0))
        raw_perfs = state.get("last_layer_neuron_perfs")
        self._last_layer_neuron_perfs = (
            np.asarray(raw_perfs, dtype=np.float64) if raw_perfs is not None else None
        )
        if self._state in ("tuning", "stopped"):
            self._reset_cycle_buffers()
            self._pending_norm = None
            self._pending_activation = None
        return self

    def plan_next(
        self,
        *,
        layer: GMDHLayer | Callable[[LayerContext], GMDHLayer] | None = None,
        activation: Any = _UNSET,
        norm: Any = _UNSET,
        select: SelectionStrategy | str | None = None,
    ) -> ModelIncrementer:
        """Override the layer template, activation, norm, or selection strategy for the **next** growth cycle (§5.7)."""
        if self._noop or self.model is None:
            return self
        if self._qr_acc is not None and self._observed_samples > 0:
            raise RuntimeError(
                "Cannot call plan_next() after StreamedQRAccumulator has already started accumulating batches."
            )
        if layer is not None:
            self._planned_layer = layer
        if activation is not _UNSET:
            self._planned_activation_spec = activation
        if norm is not _UNSET:
            self._planned_norm_spec = norm
        if select is not None:
            self._planned_select_strategy = resolve_selection_strategy(select)
        if self._qr_acc is None:
            self._active_template = None
            self._active_cand_idx0 = None
            if self._observed_samples == 0 and self.model.gmdh_layer_count > 0 and self._last_layer_stats is not None:
                mean_prev, std_prev, lock_prev = self._last_layer_stats
                self._prepare_pending_pre_modules(mean_prev, std_prev, lock_prev)
        return self

    def prepare(
        self,
        train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> ModelIncrementer:
        """Optional convenience: run the Cycle 0 input-stage statistics pass outside the main loop (§5.4)."""
        if self._noop or self.model is None or self._state != "stats":
            return self
        with torch.no_grad():
            for xb, yb in train_loader:
                self.observe(yb, x=xb)
            self.step(train_loader)
        return self

    def _resolve_cycle_template(self, previous_width: int) -> GMDHLayer:
        layer_no = self.model.gmdh_layer_count + 1
        in_width = self.model.input_count if self.model.input_count > 0 else previous_width
        ctx = LayerContext(
            layer_no=layer_no,
            previous_width=previous_width,
            input_width=in_width,
            perf_history=tuple(self.layer_perfs),
        )
        if self._planned_layer is not None:
            planned = self._planned_layer
            if isinstance(planned, GMDHLayer):
                raw_spec: Any = GMDHLayer(
                    planned,
                    dtype=self.model.dtype,
                    device=self.model.device,
                    layer_number=layer_no,
                )
            else:
                raw_spec = planned(ctx)
        else:
            raw_spec = self._layer_factory(ctx)

        if isinstance(raw_spec, (list, tuple)):
            found_layer: GMDHLayer | None = None
            found_norm: Any = _UNSET
            found_act: Any = _UNSET
            for item in raw_spec:
                if isinstance(item, GMDHLayer):
                    found_layer = item
                elif isinstance(item, GMDHNorm):
                    found_norm = item
                elif isinstance(item, GMDHActivation):
                    found_act = item.activation
                elif isinstance(item, nn.Module):
                    found_act = item
            if found_layer is None:
                raise ValueError("Sequence returned by layer factory/plan_next must contain a GMDHLayer.")
            tmpl = found_layer
            if found_norm is not _UNSET and self._planned_norm_spec is _UNSET:
                self._planned_norm_spec = found_norm
            if found_act is not _UNSET and self._planned_activation_spec is _UNSET:
                self._planned_activation_spec = found_act
            if layer_no > 1 and self._last_layer_stats is not None:
                self._prepare_pending_pre_modules(*self._last_layer_stats)
        elif isinstance(raw_spec, GMDHLayer):
            tmpl = raw_spec
        else:
            raise TypeError(f"Layer factory must return a GMDHLayer or sequence of modules; got {type(raw_spec)!r}.")

        if not tmpl.is_materialized and (tmpl.dtype != self.dtype or tmpl.device != self.device):
            tmpl = tmpl.to(dtype=self.dtype, device=self.device)

        fan_in = min(int(tmpl.input_count), int(previous_width))
        tmpl.input_count = max(1, fan_in)
        tmpl.layer_number = layer_no

        act_spec = (
            self._planned_activation_spec
            if self._planned_activation_spec is not _UNSET
            else self._default_activation_spec
        )
        has_any_act = any(isinstance(m, GMDHActivation) for m in self.model) or (
            self._pending_activation is not None
        )
        prev_layers = self.model.gmdh_layers
        prev_is_linear = bool(prev_layers) and all(
            ly.degree == 1 and ly.interaction <= 1 for ly in prev_layers
        )
        if (
            layer_no == 2
            and prev_is_linear
            and tmpl.degree == 1
            and tmpl.interaction <= 1
            and act_spec is None
            and not has_any_act
        ):
            warnings.warn(
                "Linear GMDH neurons (degree=1, interaction<=1) used across multiple layers without "
                "between-layer activation; the network will collapse to a linear model (§3.3).",
                UserWarning,
                stacklevel=3,
            )
        return tmpl

    def _apply_pending_pre_modules(self, h: torch.Tensor) -> torch.Tensor:
        out = h
        if self._pending_norm is not None:
            out = self._pending_norm(out)
        if self._pending_activation is not None:
            out = self._pending_activation(out)
        return out

    def _prepare_pending_pre_modules(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        lock_mask: torch.Tensor,
    ) -> None:
        norm_spec = (
            self._planned_norm_spec
            if self._planned_norm_spec is not _UNSET
            else self._default_norm_spec
        )
        act_spec = (
            self._planned_activation_spec
            if self._planned_activation_spec is not _UNSET
            else self._default_activation_spec
        )
        norm_mod = _instantiate_norm(
            norm_spec, dtype=self.model.dtype, device=self.model.device
        )
        if norm_mod is not None:
            norm_mod.set_stats(mean, std, passthrough_mask=lock_mask)
            if not self.gradient_during_growth and norm_mod.affine and norm_mod.weight is not None and norm_mod.bias is not None:
                norm_mod.weight.requires_grad_(False)
                norm_mod.bias.requires_grad_(False)
        self._pending_norm = norm_mod

        if act_spec is not None:
            act_mod = GMDHActivation(act_spec, passthrough_mask=lock_mask).to(
                device=self.model.device, dtype=self.model.dtype
            )
            if not self.gradient_during_growth:
                for p in act_mod.parameters():
                    p.requires_grad_(False)
            self._pending_activation = act_mod
        else:
            self._pending_activation = None

    def _compute_feature_stats(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-column ``(mean, std)`` in ``self.acc_dtype`` and cast to ``self.dtype``."""
        h_acc = features.to(dtype=self.acc_dtype, device=self.device)
        mean_acc = torch.mean(h_acc, dim=0)
        var_acc = torch.mean((h_acc - mean_acc.unsqueeze(0)) ** 2, dim=0)
        std_acc = torch.sqrt(torch.where(var_acc > 1e-24, var_acc, torch.ones_like(var_acc)))
        return mean_acc.to(dtype=self.dtype), std_acc.to(dtype=self.dtype)

    def _extract_pre_output_features(self, x: torch.Tensor | None) -> torch.Tensor:
        if x is not None:
            x_t = x.to(device=self.model.device, non_blocking=x.is_pinned())
            if x_t.ndim == 1:
                x_t = x_t.unsqueeze(0)
            if (
                self._last_model_input_ptr is not None
                and x_t.data_ptr() == self._last_model_input_ptr
                and tuple(x_t.shape) == self._last_model_input_shape
                and self._last_pre_output is not None
                and int(self._last_pre_output.shape[0]) == int(x_t.shape[0])
            ):
                cached = self._last_pre_output
                self._last_model_input_ptr = None
                self._last_model_input_shape = None
                self._last_pre_output = None
                return cached
            return self.model.forward_to_insertion_point(x_t)

        if self._last_pre_output is not None:
            cached = self._last_pre_output
            self._last_model_input_ptr = None
            self._last_model_input_shape = None
            self._last_pre_output = None
            return cached
        raise ValueError(
            "ModelIncrementer.observe(y, x=xb) requires `x` unless `model(xb)` was called immediately before."
        )

    def _flush_observed_to_qr_accumulator(self) -> None:
        if not self._observed_batches:
            return
        assert self._active_template is not None and self._active_cand_idx0 is not None
        tmpl = self._active_template
        cand_idx0 = self._active_cand_idx0
        n_cand = int(cand_idx0.shape[0])
        c_count = gmdh_coeff_count(tmpl.input_count, tmpl.degree, interaction=tmpl.interaction)
        target_count = int(self._observed_batches[0][1].shape[1])

        if self._qr_acc is None:
            self._qr_acc = StreamedQRAccumulator(
                candidate_count=n_cand,
                coeff_count=c_count,
                target_count=target_count,
                device=self.device,
                dtype=self.acc_dtype,
            )

        h_cat = torch.cat([b[0] for b in self._observed_batches], dim=0)
        y_cat = torch.cat([b[1] for b in self._observed_batches], dim=0)
        self._observed_batches.clear()
        self._observed_bytes = 0

        y64 = self._qr_acc.update_targets(y_cat)
        p_powers = precompute_layer_powers_torch(h_cat, tmpl.degree)
        idx0_t = torch.as_tensor(cand_idx0, dtype=torch.long, device=h_cat.device)
        b_step = effective_batch_size(
            self.chunk_size,
            int(h_cat.shape[0]),
            c_count,
            elem_size=h_cat.element_size(),
            max_chunk_bytes=self.max_chunk_bytes,
        )

        for start in range(0, n_cand, b_step):
            end = min(start + b_step, n_cand)
            x_chunk: torch.Tensor = gmdh_preprocess_batch(
                h_cat,
                idx0_t[start:end],
                tmpl.degree,
                precomputed_powers=p_powers,
                interaction=tmpl.interaction,
            )  # type: ignore[assignment]
            self._qr_acc.update_chunk(start, end, x_chunk, y64)
            del x_chunk

    def observe(
        self,
        y: torch.Tensor,
        x: torch.Tensor | None = None,
    ) -> None:
        """Accumulate batch tensors ``x`` and targets ``y`` for the current growth cycle (§5.7, Q-G).

        No-op when ``self.growing`` is ``False``.
        """
        if not self.growing:
            return

        with torch.no_grad():
            y_t = y.to(dtype=self.acc_dtype, device=self.device, non_blocking=y.is_pinned())
            if y_t.ndim == 1:
                y_t = y_t.unsqueeze(1)
            elif y_t.ndim > 2:
                y_t = y_t.reshape(y_t.shape[0], -1)

            if self._target_y_sum is None:
                self._target_y_sum = torch.sum(y_t, dim=0)
            else:
                self._target_y_sum = self._target_y_sum + torch.sum(y_t, dim=0)
            self._target_y_count += int(y_t.shape[0])

            if self._state == "stats":
                if x is not None:
                    x_in = x.to(device=self.model.device, non_blocking=x.is_pinned())
                    if x_in.ndim == 1:
                        x_in = x_in.unsqueeze(0)
                elif self._last_pre_output is not None:
                    x_in = self._last_pre_output
                else:
                    raise ValueError("ModelIncrementer.observe(y, x=xb) requires `x` during statistics pass.")

                if self.model._input_count == 0 and x_in.shape[1] > 0:
                    self.model._input_count = int(x_in.shape[1])

                cur = x_in
                for mod in self.model:
                    if isinstance(mod, GMDHOutput):
                        break
                    if isinstance(mod, GMDHNorm) and not bool(mod.initialized.item()):
                        mod.accumulate_batch(cur)
                        break
                    cur = mod(cur)
                return

            # State == "accumulate"
            if self.model.gmdh_layer_count == 0 and self._target_y_sum is not None and self._target_y_count > 0:
                self.model.output_module.set_fallback(self._target_y_sum / float(self._target_y_count))

            h_raw = self._extract_pre_output_features(x)

            if self._active_template is None:
                n_prev = int(h_raw.shape[1])
                m_targets_obs = int(y_t.shape[1])
                tmpl = self._resolve_cycle_template(n_prev)
                self._active_template = tmpl
                eff_max = (tmpl.max_neurons * m_targets_obs) if tmpl.max_neurons > 0 else 0
                if tmpl.forward_inputs and eff_max > 0 and n_prev > eff_max:
                    max_fwd = eff_max // 2
                    cand_cap: int | None = max(1, eff_max - max_fwd)
                else:
                    cand_cap = eff_max if eff_max > 0 else None
                scores_prev = (
                    self._last_layer_neuron_perfs
                    if (self._last_layer_neuron_perfs is not None and self._last_layer_neuron_perfs.size == n_prev)
                    else None
                )
                self._active_cand_idx0 = resolve_candidate_indices(
                    n_prev,
                    tmpl.input_count,
                    tmpl.candidates,
                    max_candidates=cand_cap,
                    input_scores=scores_prev,
                    rng=self._rng,
                )

            h_in = self._apply_pending_pre_modules(h_raw).detach()

            batch_bytes = int(h_in.numel() * h_in.element_size() + y_t.numel() * y_t.element_size())
            self._observed_batches.append((h_in, y_t.detach()))
            self._observed_bytes += batch_bytes
            self._observed_samples += int(h_in.shape[0])

            if self._qr_acc is not None or (
                self.max_observe_bytes is not None and self._observed_bytes > self.max_observe_bytes
            ):
                self._flush_observed_to_qr_accumulator()

    def _forward_to_insertion_point_from_root(self, xb_t: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        if self._root_model is self.model:
            return self.model.forward_to_insertion_point(xb_t)
        captured: list[torch.Tensor] = []

        def _cap_hook(_mod: nn.Module, inputs: tuple[Any, ...]) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                captured.append(inputs[0].detach())

        handle = self.model.register_forward_pre_hook(_cap_hook)
        try:
            self._root_model(xb_t)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError(
                f"Composite root {type(self._root_model).__name__} did not invoke its embedded GMDHModel."
            )
        return self.model.forward_to_insertion_point(captured[0])

    def _collect_selection_set(
        self,
        select_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.model is not None
        s_feats: list[torch.Tensor] = []
        s_targs: list[torch.Tensor] = []
        for xb, yb in select_loader:
            xb_t = xb.to(device=self.model.device, non_blocking=xb.is_pinned())
            if xb_t.ndim == 1:
                xb_t = xb_t.unsqueeze(0)
            yb_t = yb.to(device=self.model.device, non_blocking=yb.is_pinned())
            if yb_t.ndim == 1:
                yb_t = yb_t.unsqueeze(1)
            elif yb_t.ndim > 2:
                yb_t = yb_t.reshape(yb_t.shape[0], -1)
            h_raw = self._forward_to_insertion_point_from_root(xb_t)
            h_in = self._apply_pending_pre_modules(h_raw).detach()
            s_feats.append(h_in)
            s_targs.append(yb_t.detach())

        if not s_feats:
            raise ValueError("`select_loader` yielded no selection samples.")
        return torch.cat(s_feats, dim=0), torch.cat(s_targs, dim=0)

    def _reset_cycle_buffers(self) -> None:
        self._active_template = None
        self._active_cand_idx0 = None
        self._observed_batches.clear()
        self._observed_bytes = 0
        self._observed_samples = 0
        self._qr_acc = None
        self._planned_layer = None
        self._planned_activation_spec = _UNSET
        self._planned_norm_spec = _UNSET
        self._planned_select_strategy = None
        self._last_model_input_ptr = None
        self._last_model_input_shape = None
        self._last_pre_output = None

    def _register_parameters_in_optimizer(self, params: list[nn.Parameter]) -> None:
        if self.optimizer is None or not params:
            return
        existing_ids = {
            id(p)
            for group in self.optimizer.param_groups
            for p in group["params"]
        }
        new_params = [p for p in params if id(p) not in existing_ids]
        if not new_params:
            return
        if len(self.optimizer.param_groups) == 1:
            self.optimizer.param_groups[0]["params"].extend(new_params)
        else:
            self.optimizer.add_param_group({"params": new_params})

    def _remove_parameters_from_optimizer(self, params: list[nn.Parameter]) -> None:
        if self.optimizer is None or not params:
            return
        remove_ids = {id(p) for p in params}
        for p in params:
            if p in self.optimizer.state:
                del self.optimizer.state[p]
        for group in self.optimizer.param_groups:
            group["params"] = [p for p in group["params"] if id(p) not in remove_ids]

    def _finish_growth(self) -> None:
        self._reset_cycle_buffers()
        self._pending_norm = None
        self._pending_activation = None

        self.model.trim(optimizer=self.optimizer if self.gradient_during_growth else None)

        trainable_params: list[nn.Parameter] = []
        for mod in self.model:
            if isinstance(mod, GMDHLayer) and mod.is_materialized:
                mod.W.requires_grad_(True)
                mod._refresh_grad_hook()
                trainable_params.append(mod.W)
            elif isinstance(mod, GMDHNorm) and mod.affine and mod.weight is not None and mod.bias is not None:
                mod.weight.requires_grad_(True)
                mod.bias.requires_grad_(True)
                mod._refresh_affine_hooks()
                trainable_params.extend([mod.weight, mod.bias])
            elif isinstance(mod, GMDHActivation):
                for p in mod.parameters():
                    p.requires_grad_(True)
                    trainable_params.append(p)
            elif isinstance(mod, GMDHOutput) and mod.dense and mod.linear is not None:
                for p in mod.linear.parameters():
                    p.requires_grad_(True)
                    trainable_params.append(p)

        self._register_parameters_in_optimizer(trainable_params)
        self._state = "tuning"
        if self.logs:
            sys.stdout.write("Finished training\n")

    def remove_last_layer(self) -> ModelIncrementer:
        """Remove the most recently added ``GMDHLayer`` (and its preceding norm/activation),
        restore ``GMDHOutput.index`` and pending statistics to the previous layer, and clean up
        optimizer parameter groups.

        Only valid while ``self.growing`` is ``True`` and at least 2 ``GMDHLayer`` modules exist.
        """
        if not self.growing:
            raise RuntimeError(
                "Cannot call remove_last_layer() when the incrementer is not in the growing state."
            )
        layer_positions = [
            i for i, m in enumerate(self.model) if isinstance(m, GMDHLayer) and m.is_materialized
        ]
        if len(layer_positions) <= 1:
            raise RuntimeError(
                "Cannot remove the last layer when the model has fewer than 2 GMDH layers."
            )

        prev_layer_pos = layer_positions[-2]
        out_pos = self.model.output_index

        removed_params: list[nn.Parameter] = []
        for pos in range(out_pos - 1, prev_layer_pos, -1):
            mod = self.model[pos]
            removed_params.extend(list(mod.parameters()))
            del self.model[pos]
        self._remove_parameters_from_optimizer(removed_params)

        if self._layer_history:
            self._layer_history.pop()
        if self.layer_perfs:
            self.layer_perfs.pop()
        if len(self.layer_perfs_tuned) > len(self._layer_history):
            self.layer_perfs_tuned = self.layer_perfs_tuned[: len(self._layer_history)]

        prev_rec = self._layer_history[-1]
        self.model.output_module.set_index(prev_rec.best_out_indices)
        ref_perf = (
            prev_rec.tuned_perf if prev_rec.tuned_perf is not None else prev_rec.closed_form_perf
        )
        self._current_min_perf = ref_perf
        self._prev_min_perf = ref_perf
        self._last_layer_stats = prev_rec.last_layer_stats
        self._reset_cycle_buffers()
        self._prepare_pending_pre_modules(*prev_rec.last_layer_stats)
        return self

    def stop(self, *, remove_last: bool = False) -> ModelIncrementer:
        """End growth immediately: optionally remove the last layer, discard pending statistics,
        trim unused neurons, and switch to tuning (§5.7).
        If called while already in tuning state, transitions to stopped state.
        """
        if self._noop or self.model is None:
            self._state = "stopped"
            return self
        if self.growing:
            if remove_last:
                self.remove_last_layer()
            self._finish_growth()
        else:
            if remove_last:
                raise RuntimeError(
                    "Cannot call stop(remove_last=True) when the incrementer is not in the growing state."
                )
            self._state = "stopped"
        return self

    def grow(
        self,
        select_loader: Iterable[tuple[torch.Tensor, Any]] | None = None,
        *,
        train_loader: Iterable[tuple[torch.Tensor, Any]] | None = None,
        final: bool = False,
    ) -> bool:
        """Solver-free piecewise growth step for ``GMDHOutput(dense=True)`` (or headless) models.

        1. Scores current top-layer neurons using ``GMDHOutput.importance_scores``.
        2. Lightly prunes weak neurons via the active ``SelectionStrategy``.
        3. If ``max_layers`` is reached (or ``final=True``), applies ``final_select`` + ``trim()``
           and switches to ``"tuning"`` (returns ``False``).
        4. Otherwise inserts ``[GMDHNorm, GMDHActivation, GMDHLayer]`` with eager weights
           (and ``resize_warm`` on ``GMDHOutput(dense=True)`` when ``forward=True``).
        """
        if self._noop or self.model is None or not self.growing:
            return False

        was_training = self._root_model.training
        if was_training:
            self._root_model.eval()
        try:
            with torch.no_grad():
                layers = self.model.gmdh_layers
                if not layers or not layers[-1].is_materialized:
                    raise RuntimeError(
                        "grow() requires at least one materialized GMDHLayer in GMDHModel."
                    )

                last_layer = layers[-1]
                last_layer_pos = max(
                    i for i, m in enumerate(self.model) if m is last_layer
                )
                out_mod = self.model.output_module if self.model.has_output_module else None
                eval_src = select_loader if select_loader is not None else train_loader

                kept_scores: np.ndarray | None = None
                if out_mod is not None and out_mod.dense and out_mod.linear is not None:
                    if eval_src is None:
                        raise ValueError("`select_loader` or `train_loader` is required to score neurons in grow().")
                    h_list: list[torch.Tensor] = []
                    for batch in eval_src:  # type: ignore[union-attr]
                        xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                        xb_t = xb.to(device=self.model.device, non_blocking=xb.is_pinned())
                        if xb_t.ndim == 1:
                            xb_t = xb_t.unsqueeze(0)
                        h_list.append(self._forward_to_insertion_point_from_root(xb_t).detach())
                    h_sel = torch.cat(h_list, dim=0)
                    scores = out_mod.importance_scores(h_sel)

                    at_cap = (self.max_layers > 0 and len(layers) >= self.max_layers) or final
                    strat = (
                        self._final_select_strategy
                        if at_cap
                        else (self._planned_select_strategy or self._default_select_strategy)
                    )
                    keep_mask = strat.keep_mask(
                        scores,
                        max_neurons=last_layer.max_neurons,
                        min_neurons=last_layer.min_neurons,
                        higher_is_better=True,
                    )
                    if not np.any(keep_mask):
                        keep_mask[int(np.argmax(scores))] = True

                    # Respect 50% forwarder budget cap if upcoming layer uses forward=True
                    if not at_cap:
                        tmpl_preview = self._resolve_cycle_template(int(np.sum(keep_mask)))
                        if tmpl_preview.forward_inputs and tmpl_preview.max_neurons > 0:
                            max_fwd = tmpl_preview.max_neurons // 2
                            if int(np.sum(keep_mask)) > max_fwd:
                                top_idx = np.argsort(-scores, kind="mergesort")[:max_fwd]
                                keep_mask = np.zeros_like(keep_mask)
                                keep_mask[top_idx] = True

                    if not bool(np.all(keep_mask)):
                        self.model.prune(last_layer_pos, keep_mask, optimizer=self.optimizer)
                    kept_scores = scores[keep_mask]

                    if at_cap:
                        self._finish_growth()
                        return False
                elif (self.max_layers > 0 and len(layers) >= self.max_layers) or final:
                    self._finish_growth()
                    return False

                prev_width = self.model.gmdh_layers[-1].neuron_count
                layer_no = len(self.model.gmdh_layers) + 1
                tmpl = self._resolve_cycle_template(prev_width)

                # Prepare optional GMDHNorm & GMDHActivation using unlabeled x pass
                norm_spec = (
                    self._planned_norm_spec
                    if self._planned_norm_spec is not _UNSET
                    else self._default_norm_spec
                )
                act_spec = (
                    self._planned_activation_spec
                    if self._planned_activation_spec is not _UNSET
                    else self._default_activation_spec
                )
                norm_mod = _instantiate_norm(
                    norm_spec, dtype=self.model.dtype, device=self.model.device
                )
                lock_prev = self.model.gmdh_layers[-1].lock_mask.detach().clone()
                if norm_mod is not None:
                    stat_src = train_loader if train_loader is not None else eval_src
                    if stat_src is None:
                        mean_p = torch.zeros(prev_width, dtype=self.model.dtype, device=self.model.device)
                        std_p = torch.ones(prev_width, dtype=self.model.dtype, device=self.model.device)
                    else:
                        t_outs: list[torch.Tensor] = []
                        for batch in stat_src:  # type: ignore[union-attr]
                            xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                            xb_t = xb.to(device=self.model.device, non_blocking=xb.is_pinned())
                            if xb_t.ndim == 1:
                                xb_t = xb_t.unsqueeze(0)
                            t_outs.append(self._forward_to_insertion_point_from_root(xb_t).detach())
                        mean_p, std_p = self._compute_feature_stats(torch.cat(t_outs, dim=0))
                    norm_mod.set_stats(mean_p, std_p, passthrough_mask=lock_prev)

                act_mod: GMDHActivation | None = None
                if act_spec is not None:
                    act_mod = GMDHActivation(act_spec, passthrough_mask=lock_prev).to(
                        device=self.model.device, dtype=self.model.dtype
                    )

                eff_chunk_size = (
                    self.chunk_size if tmpl.chunk_size == DEFAULT_CHUNK_SIZE else tmpl.chunk_size
                )
                eff_max_bytes = (
                    self.max_chunk_bytes
                    if tmpl.max_chunk_bytes == DEFAULT_MAX_CHUNK_BYTES
                    else tmpl.max_chunk_bytes
                )
                new_layer = GMDHLayer(
                    in_features=prev_width,
                    inputs=tmpl.input_count,
                    degree=tmpl.degree,
                    interaction=tmpl.interaction,
                    forward=tmpl.forward_inputs,
                    min_neurons=tmpl.min_neurons,
                    max_neurons=tmpl.max_neurons,
                    candidates=tmpl.candidates,
                    init=tmpl.init,
                    init_std=tmpl.init_std,
                    rng=self._rng,
                    input_scores=kept_scores,
                    higher_is_better=True,
                    layer_number=layer_no,
                    dtype=self.model.dtype,
                    device=self.model.device,
                    chunk_size=eff_chunk_size,
                    max_chunk_bytes=eff_max_bytes,
                )

                new_params: list[nn.Parameter] = []
                if norm_mod is not None:
                    self.model.insert(self.model.output_index, norm_mod)
                    if norm_mod.affine and norm_mod.weight is not None and norm_mod.bias is not None:
                        new_params.extend([norm_mod.weight, norm_mod.bias])
                if act_mod is not None:
                    self.model.insert(self.model.output_index, act_mod)
                    new_params.extend(list(act_mod.parameters()))

                self.model.insert(self.model.output_index, new_layer)
                new_layer.W.requires_grad_(True)
                new_layer._refresh_grad_hook()
                new_params.append(new_layer.W)
                self._register_parameters_in_optimizer(new_params)

                if out_mod is not None and out_mod.dense:
                    if new_layer.forward_inputs:
                        n_cand = int(torch.sum(~new_layer.lock_mask).item())
                        out_mod.resize_warm(
                            new_layer.neuron_count,
                            forward_slice=slice(n_cand, new_layer.neuron_count),
                            optimizer=self.optimizer,
                        )
                    else:
                        out_mod.adapt_width(new_layer.neuron_count, optimizer=self.optimizer)

                if self.logs:
                    sys.stdout.write(
                        f"Grew layer {layer_no}. Neuron count: {new_layer.neuron_count}.\n"
                    )
                self._reset_cycle_buffers()
                return True
        finally:
            if was_training:
                self._root_model.train()

    def update_layer(
        self,
        select_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        train_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> float:
        """Synchronize the current layer's state after gradient fine-tuning:

        1. Re-scores all neurons of the last ``GMDHLayer`` on ``select_loader``.
        2. Updates ``GMDHOutput.index`` to the best post-tuning neuron (or best per target for multi-output).
        3. Updates the reference performance (``current_performance`` / ``layer_perfs_tuned``).
        4. Refreshes the outgoing pending ``GMDHNorm`` statistics (from ``train_loader`` if provided,
           otherwise from ``select_loader``) for the next growth cycle.

        Does not add/remove layers, does not trim, and does not change the growth state.
        """
        layers = self.model.gmdh_layers
        if not layers or not layers[-1].is_materialized:
            raise RuntimeError(
                "Cannot call update_layer() before at least one GMDHLayer has been materialized."
            )
        last_layer = layers[-1]

        was_training = self.model.training
        if was_training:
            self.model.eval()
        try:
            with torch.no_grad():
                s_outs: list[torch.Tensor] = []
                s_targs: list[torch.Tensor] = []
                for xb, yb in select_loader:
                    xb_t = xb.to(device=self.model.device, non_blocking=xb.is_pinned())
                    if xb_t.ndim == 1:
                        xb_t = xb_t.unsqueeze(0)
                    yb_t = yb.to(device=self.model.device, non_blocking=yb.is_pinned())
                    if yb_t.ndim == 1:
                        yb_t = yb_t.unsqueeze(1)
                    elif yb_t.ndim > 2:
                        yb_t = yb_t.reshape(yb_t.shape[0], -1)
                    h_out = self.model.forward_to_insertion_point(xb_t).detach()
                    s_outs.append(h_out)
                    s_targs.append(yb_t.detach())

                if not s_outs:
                    raise ValueError("`select_loader` yielded no selection samples.")
                out_sel = torch.cat(s_outs, dim=0)
                s_targets = torch.cat(s_targs, dim=0)
                m_targets = int(s_targets.shape[1])

                if self._is_r2:
                    s_mean = torch.mean(s_targets, dim=0, keepdim=True)
                    tss_vec = torch.sum((s_targets - s_mean) ** 2, dim=0)
                    tss_arg: float | torch.Tensor = (
                        float(tss_vec[0].item()) if m_targets == 1 else tss_vec
                    )
                else:
                    tss_arg = 0.0

                unlocked_np = (~last_layer.lock_mask).detach().cpu().numpy()
                if m_targets == 1:
                    perfs_np = (
                        _compute_batch_perfs_torch(
                            out_sel,
                            s_targets.reshape(-1),
                            is_r2=self._is_r2,
                            total_sum_of_squares=tss_arg,
                            mse_div_4=self.mse_div_4,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    cand_indices = np.flatnonzero(unlocked_np)
                    if cand_indices.size == 0:
                        cand_indices = np.arange(perfs_np.size)
                    cand_perfs = perfs_np[cand_indices]
                    tuned_perf = float(np.min(cand_perfs))
                    best_local = int(np.flatnonzero(cand_perfs == tuned_perf)[0])
                    best_out_indices = [int(cand_indices[best_local])]
                else:
                    tid_np = last_layer.target_id.detach().cpu().numpy()
                    best_out_indices = []
                    min_perfs_per_target: list[float] = []
                    tss_vec_t = tss_arg if isinstance(tss_arg, torch.Tensor) else None
                    for j in range(m_targets):
                        mask_j = unlocked_np & (tid_np == j)
                        idx_j = np.flatnonzero(mask_j)
                        if idx_j.size == 0:
                            idx_j = np.flatnonzero(unlocked_np)
                        if idx_j.size == 0:
                            idx_j = np.arange(int(out_sel.shape[1]))
                        idx_j_t = torch.as_tensor(idx_j, dtype=torch.long, device=out_sel.device)
                        tss_j = float(tss_vec_t[j].item()) if tss_vec_t is not None else 0.0
                        perfs_j = (
                            _compute_batch_perfs_torch(
                                out_sel[:, idx_j_t],
                                s_targets[:, j].reshape(-1),
                                is_r2=self._is_r2,
                                total_sum_of_squares=tss_j,
                                mse_div_4=self.mse_div_4,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        min_p_j = float(np.min(perfs_j))
                        min_perfs_per_target.append(min_p_j)
                        best_local_j = int(np.flatnonzero(perfs_j == min_p_j)[0])
                        best_out_indices.append(int(idx_j[best_local_j]))
                    tuned_perf = float(np.mean(min_perfs_per_target))

                if train_loader is not None:
                    t_outs: list[torch.Tensor] = []
                    for xb, _yb in train_loader:
                        xb_t = xb.to(device=self.model.device, non_blocking=xb.is_pinned())
                        if xb_t.ndim == 1:
                            xb_t = xb_t.unsqueeze(0)
                        t_outs.append(self.model.forward_to_insertion_point(xb_t).detach())
                    stat_source = torch.cat(t_outs, dim=0) if t_outs else out_sel
                else:
                    stat_source = out_sel

                mean_raw, std_raw = self._compute_feature_stats(stat_source)
                lock_mask = last_layer.lock_mask.detach().clone()
                new_mean = torch.where(lock_mask, torch.zeros_like(mean_raw), mean_raw)
                new_std = torch.where(lock_mask, torch.ones_like(std_raw), std_raw)

                self.model.output_module.set_index(best_out_indices)
                self._current_min_perf = tuned_perf
                self._prev_min_perf = tuned_perf

                layer_idx = len(layers) - 1
                while len(self.layer_perfs_tuned) <= layer_idx:
                    fallback_idx = len(self.layer_perfs_tuned)
                    fallback_val = (
                        self.layer_perfs[fallback_idx]
                        if fallback_idx < len(self.layer_perfs)
                        else tuned_perf
                    )
                    self.layer_perfs_tuned.append(fallback_val)
                self.layer_perfs_tuned[layer_idx] = tuned_perf

                if self._layer_history:
                    self._layer_history[-1].best_out_indices = list(best_out_indices)
                    self._layer_history[-1].tuned_perf = tuned_perf
                    self._layer_history[-1].last_layer_stats = (new_mean, new_std, lock_mask)

                if self.growing:
                    self._last_layer_stats = (new_mean, new_std, lock_mask)
                    if self._observed_samples == 0 and self._qr_acc is None:
                        self._prepare_pending_pre_modules(new_mean, new_std, lock_mask)

                self._last_model_input_ptr = None
                self._last_model_input_shape = None
                self._last_pre_output = None
                return tuned_perf
        finally:
            if was_training:
                self.model.train()

    def step(
        self,
        select_loader: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
        *,
        final: bool = False,
        evaluate_stopping: bool | None = None,
    ) -> bool:
        """Close the current growth cycle (§5.7).

        - In ``"stats"`` state (Cycle 0): finalizes input-stage ``GMDHNorm`` statistics and sets
          the Cycle 1 ``GMDHOutput`` fallback to the training target mean.
        - In ``"accumulate"`` state: solves all candidates from the observations accumulated since
          the last step, scores them on ``select_loader``, selects the kept neurons, checks stopping
          criteria (when ``evaluate_stopping`` resolves to ``True``), inserts ``[norm, activation] + layer``
          before ``GMDHOutput``, and returns whether growth continues (`True` if still growing,
          `False` if switched to tuning).
        """
        if not self.growing or self._noop or self.model is None:
            return False
        if self.solver is None:
            raise RuntimeError(
                "ModelIncrementer.step() requires a closed-form `solver`; use `inc.grow()` for solver-free piecewise growth."
            )

        eval_stop = (
            (not self.gradient_during_growth)
            if evaluate_stopping is None
            else bool(evaluate_stopping)
        )

        with torch.no_grad():
            if self._state == "stats":
                for mod in self.model:
                    if isinstance(mod, GMDHOutput):
                        break
                    if isinstance(mod, GMDHNorm) and not bool(mod.initialized.item()):
                        mod.finalize_stats()
                if self._target_y_sum is not None and self._target_y_count > 0:
                    self.model.output_module.set_fallback(
                        self._target_y_sum / float(self._target_y_count)
                    )
                self._state = "accumulate"
                if final or self.max_layers == 0:
                    self._finish_growth()
                    return False
                return True

            # State == "accumulate"
            if select_loader is None:
                raise ValueError("`select_loader` is required when stepping an accumulation cycle.")
            if self._observed_samples == 0 and self._qr_acc is None:
                raise RuntimeError("Cannot call ModelIncrementer.step() before any observe() calls.")

            if self._active_template is None or self._active_cand_idx0 is None:
                n_prev = int(self._observed_batches[0][0].shape[1])
                m_targets_obs = int(self._observed_batches[0][1].shape[1])
                tmpl = self._resolve_cycle_template(n_prev)
                self._active_template = tmpl
                eff_max = (tmpl.max_neurons * m_targets_obs) if tmpl.max_neurons > 0 else 0
                if tmpl.forward_inputs and eff_max > 0 and n_prev > eff_max:
                    max_fwd = eff_max // 2
                    cand_cap: int | None = max(1, eff_max - max_fwd)
                else:
                    cand_cap = eff_max if eff_max > 0 else None
                scores_prev = (
                    self._last_layer_neuron_perfs
                    if (self._last_layer_neuron_perfs is not None and self._last_layer_neuron_perfs.size == n_prev)
                    else None
                )
                self._active_cand_idx0 = resolve_candidate_indices(
                    n_prev,
                    tmpl.input_count,
                    tmpl.candidates,
                    max_candidates=cand_cap,
                    input_scores=scores_prev,
                    rng=self._rng,
                )

            tmpl = self._active_template
            cand_idx0_np = self._active_cand_idx0
            layer_no = self.model.gmdh_layer_count + 1
            fan_in = tmpl.input_count
            degree = tmpl.degree
            interaction = tmpl.interaction
            coeff_count = gmdh_coeff_count(fan_in, degree, interaction=interaction)

            if self._observed_samples < coeff_count:
                if self.logs:
                    sys.stdout.write(
                        f"Number of coefficients ({coeff_count}) in layer {layer_no} "
                        f"exceeded the amount of samples ({self._observed_samples}). Stop training. Layer is discarded.\n"
                    )
                self._finish_growth()
                return False

            s_in, s_targets = self._collect_selection_set(select_loader)
            previous_width = int(s_in.shape[1])
            unlocked_count = int(cand_idx0_np.shape[0])
            locked_count = previous_width if tmpl.forward_inputs else 0
            total_neuron_count = unlocked_count + locked_count
            m_targets = int(s_targets.shape[1])
            dev = self.model.device
            dt = self.model.dtype
            elem_size = s_in.element_size()

            if self.logs:
                sys.stdout.write(
                    f"Training layer {layer_no}. Neuron count: {total_neuron_count}.\n"
                )

            if self._is_r2:
                s_mean = torch.mean(s_targets, dim=0, keepdim=True)
                tss_vec = torch.sum((s_targets - s_mean) ** 2, dim=0)
                tss_arg: float | torch.Tensor = (
                    float(tss_vec[0].item()) if m_targets == 1 else tss_vec
                )
            else:
                tss_arg = 0.0

            s_powers = precompute_layer_powers_torch(s_in, degree)
            idx0_all_t = torch.as_tensor(cand_idx0_np, dtype=torch.long, device=dev)

            # Solve & score candidates in memory-bounded chunks
            if self._qr_acc is not None or m_targets > 1:
                if self._observed_batches:
                    self._flush_observed_to_qr_accumulator()
                assert self._qr_acc is not None
                w_all64, mean_all64, std_all64 = self._qr_acc.solve(
                    solver=self.solver,
                    rcond=self.rcond,
                    alpha=self.alpha,
                    equilibrate=self.equilibrate,
                    chunk_size=self.chunk_size,
                    max_chunk_bytes=self.max_chunk_bytes,
                )
                w_all = w_all64.to(dtype=dt)
                mean_all = mean_all64.to(dtype=dt)
                std_all = std_all64.to(dtype=dt)

                b_step_sel = effective_batch_size(
                    self.chunk_size,
                    int(s_in.shape[0]),
                    coeff_count,
                    elem_size=elem_size,
                    max_chunk_bytes=self.max_chunk_bytes,
                )
                perfs_t = torch.empty(
                    (unlocked_count, m_targets) if m_targets > 1 else (unlocked_count,),
                    dtype=dt,
                    device=dev,
                )
                for start in range(0, unlocked_count, b_step_sel):
                    end = min(start + b_step_sel, unlocked_count)
                    x_sel: torch.Tensor = gmdh_preprocess_batch(
                        s_in,
                        idx0_all_t[start:end],
                        degree,
                        precomputed_powers=s_powers,
                        interaction=interaction,
                    )  # type: ignore[assignment]
                    w_sub = w_all[start:end]
                    if m_targets == 1:
                        w_2d = w_sub.squeeze(1)
                        out_sel = torch.einsum("mnc,mc->nm", x_sel, w_2d)
                        perfs_t[start:end] = _compute_batch_perfs_torch(
                            out_sel,
                            s_targets.reshape(-1),
                            is_r2=self._is_r2,
                            total_sum_of_squares=tss_arg,
                            mse_div_4=self.mse_div_4,
                        )
                    else:
                        out_sel = torch.einsum("bnc,bmc->nbm", x_sel, w_sub)
                        perfs_t[start:end] = _compute_batch_perfs_torch(
                            out_sel,
                            s_targets,
                            is_r2=self._is_r2,
                            total_sum_of_squares=tss_arg,
                            mse_div_4=self.mse_div_4,
                        )
                    del x_sel, out_sel
            else:
                # Single-output in-memory path (fused chunked solve + selection scoring)
                acc_dt = self.acc_dtype
                t_in = (
                    self._observed_batches[0][0]
                    if len(self._observed_batches) == 1
                    else torch.cat([b[0] for b in self._observed_batches], dim=0)
                ).to(dtype=acc_dt, device=dev)
                t_targets_1d = (
                    self._observed_batches[0][1]
                    if len(self._observed_batches) == 1
                    else torch.cat([b[1] for b in self._observed_batches], dim=0)
                ).to(dtype=acc_dt, device=dev).reshape(-1)
                self._observed_batches.clear()
                self._observed_bytes = 0

                n_train = int(t_in.shape[0])
                n_sel = int(s_in.shape[0])
                t_powers = precompute_layer_powers_torch(t_in, degree)
                acc_elem_size = 4 if acc_dt == torch.float32 else 8
                b_step = effective_batch_size(
                    self.chunk_size,
                    max(n_train, n_sel),
                    coeff_count,
                    elem_size=acc_elem_size,
                    max_chunk_bytes=self.max_chunk_bytes,
                )

                w_all = torch.empty((unlocked_count, 1, coeff_count), dtype=dt, device=dev)
                mean_all = torch.empty((unlocked_count, 1), dtype=dt, device=dev)
                std_all = torch.empty((unlocked_count, 1), dtype=dt, device=dev)
                perfs_t = torch.empty(unlocked_count, dtype=dt, device=dev)
                s_targets_1d = s_targets.reshape(-1)

                for start in range(0, unlocked_count, b_step):
                    end = min(start + b_step, unlocked_count)
                    idx_sub = idx0_all_t[start:end]
                    x_train: torch.Tensor = gmdh_preprocess_batch(
                        t_in,
                        idx_sub,
                        degree,
                        precomputed_powers=t_powers,
                        interaction=interaction,
                    )  # type: ignore[assignment]
                    w_sub64 = _solve_batch_torch(
                        x_train,
                        t_targets_1d,
                        solver_low=self.solver,
                        rcond=self.rcond,
                        alpha=self.alpha,
                        equilibrate=self.equilibrate,
                    )
                    out_train = torch.einsum("mnc,mc->nm", x_train, w_sub64)
                    m_sub = torch.mean(out_train, dim=0)
                    v_sub = torch.mean((out_train - m_sub.unsqueeze(0)) ** 2, dim=0)
                    s_sub = torch.sqrt(torch.where(v_sub > 1e-24, v_sub, torch.ones_like(v_sub)))
                    del x_train, out_train

                    w_sub = w_sub64.to(dtype=dt)
                    w_all[start:end, 0, :] = w_sub
                    mean_all[start:end, 0] = m_sub.to(dtype=dt)
                    std_all[start:end, 0] = s_sub.to(dtype=dt)

                    x_sel = gmdh_preprocess_batch(
                        s_in,
                        idx_sub,
                        degree,
                        precomputed_powers=s_powers,
                        interaction=interaction,
                    )  # type: ignore[assignment]
                    out_sel = torch.einsum("mnc,mc->nm", x_sel, w_sub)
                    perfs_t[start:end] = _compute_batch_perfs_torch(
                        out_sel,
                        s_targets_1d,
                        is_r2=self._is_r2,
                        total_sum_of_squares=tss_arg,
                        mse_div_4=self.mse_div_4,
                    )
                    del x_sel, out_sel

                del t_in, t_targets_1d, t_powers

            del s_powers
            mem_now = int(
                (
                    self._observed_samples * previous_width
                    + int(s_in.numel())
                    + int(s_in.shape[0]) * total_neuron_count
                    + total_neuron_count * coeff_count
                )
                * elem_size
            )
            self.peak_memory_bytes = max(self.peak_memory_bytes, mem_now)

            is_nan_present = bool(torch.isnan(w_all).any().item())
            if layer_no == 1 and is_nan_present:
                raise RuntimeError(
                    "NaN weights found in the first layer. Sorry but it is impossible to continue training."
                )

            perfs_np = perfs_t.detach().cpu().numpy()
            del perfs_t

            # Select kept neurons (single-output or multi-output Option 2; §9)
            eff_select = (
                self._planned_select_strategy
                if self._planned_select_strategy is not None
                else self._default_select_strategy
            )
            if m_targets == 1:
                keep_mask = eff_select.keep_mask(
                    perfs_np,
                    max_neurons=tmpl.max_neurons,
                    min_neurons=tmpl.min_neurons,
                )
                kept_rows = np.flatnonzero(keep_mask)
                kept_rows_t = torch.as_tensor(kept_rows, dtype=torch.long, device=dev)
                sel_idx0 = idx0_all_t[kept_rows_t]
                sel_w = w_all[kept_rows_t, 0, :]
                sel_mean = mean_all[kept_rows_t, 0]
                sel_std = std_all[kept_rows_t, 0]
                sel_tid = torch.zeros(int(kept_rows_t.numel()), dtype=torch.long, device=dev)
                kept_perfs = perfs_np[kept_rows]
                kept_perfs_list = kept_perfs.tolist()
                current_layer_min_perf = float(np.min(kept_perfs))
                best_out_indices = [int(np.flatnonzero(kept_perfs == current_layer_min_perf)[0])]
            else:
                idx0_parts: list[torch.Tensor] = []
                w_parts: list[torch.Tensor] = []
                mean_parts: list[torch.Tensor] = []
                std_parts: list[torch.Tensor] = []
                tid_parts: list[torch.Tensor] = []
                best_out_indices = []
                min_perfs_per_target: list[float] = []
                kept_perfs_parts: list[np.ndarray] = []
                offset = 0

                for j in range(m_targets):
                    perfs_j = perfs_np[:, j]
                    keep_mask_j = eff_select.keep_mask(
                        perfs_j,
                        max_neurons=tmpl.max_neurons,
                        min_neurons=tmpl.min_neurons,
                    )
                    kept_j = np.flatnonzero(keep_mask_j)
                    kept_j_t = torch.as_tensor(kept_j, dtype=torch.long, device=dev)
                    idx0_parts.append(idx0_all_t[kept_j_t])
                    w_parts.append(w_all[kept_j_t, j, :])
                    mean_parts.append(mean_all[kept_j_t, j])
                    std_parts.append(std_all[kept_j_t, j])
                    tid_parts.append(torch.full((int(kept_j_t.numel()),), j, dtype=torch.long, device=dev))

                    kept_perfs_j = perfs_j[kept_j]
                    min_p_j = float(np.min(kept_perfs_j))
                    min_perfs_per_target.append(min_p_j)
                    # Normalize per-target scores so no target starves another in ranked selection
                    norm_j = kept_perfs_j / max(abs(min_p_j), 1e-12)
                    kept_perfs_parts.append(norm_j)
                    best_local_j = int(np.flatnonzero(kept_perfs_j == min_p_j)[0])
                    best_out_indices.append(offset + best_local_j)
                    offset += int(kept_j_t.numel())

                sel_idx0 = torch.cat(idx0_parts, dim=0)
                sel_w = torch.cat(w_parts, dim=0)
                sel_mean = torch.cat(mean_parts, dim=0)
                sel_std = torch.cat(std_parts, dim=0)
                sel_tid = torch.cat(tid_parts, dim=0)
                kept_perfs_list = np.concatenate(kept_perfs_parts).tolist()
                current_layer_min_perf = float(np.mean(min_perfs_per_target))

            n_selected_unlocked = int(sel_w.shape[0])
            eff_max_step = (tmpl.max_neurons * m_targets) if tmpl.max_neurons > 0 else 0
            if tmpl.forward_inputs:
                if eff_max_step > 0 and previous_width > eff_max_step:
                    max_fwd = eff_max_step // 2
                    if self._last_layer_neuron_perfs is not None and self._last_layer_neuron_perfs.size == previous_width:
                        fwd_indices_np = np.argsort(self._last_layer_neuron_perfs, kind="mergesort")[:max_fwd]
                    else:
                        fwd_indices_np = np.arange(max_fwd, dtype=np.int64)
                else:
                    fwd_indices_np = np.arange(previous_width, dtype=np.int64)
                n_fwd = int(fwd_indices_np.size)
                fwd_idx0 = torch.as_tensor(fwd_indices_np, dtype=torch.long, device=dev)[:, None].repeat(1, fan_in)
                fwd_w = torch.zeros((n_fwd, coeff_count), dtype=dt, device=dev)
                fwd_w[:, 1] = 1.0
                fwd_mean = torch.zeros(n_fwd, dtype=dt, device=dev)
                fwd_std = torch.ones(n_fwd, dtype=dt, device=dev)
                fwd_tid = torch.full((n_fwd,), -1, dtype=torch.long, device=dev)

                full_idx0 = torch.cat([sel_idx0, fwd_idx0], dim=0)
                full_w = torch.cat([sel_w, fwd_w], dim=0)
                full_mean = torch.cat([sel_mean, fwd_mean], dim=0)
                full_std = torch.cat([sel_std, fwd_std], dim=0)
                full_tid = torch.cat([sel_tid, fwd_tid], dim=0)
                full_lock = torch.zeros(full_w.shape[0], dtype=torch.bool, device=dev)
                full_lock[n_selected_unlocked:] = True
                if self._last_layer_neuron_perfs is not None and self._last_layer_neuron_perfs.size == previous_width:
                    fwd_perfs = self._last_layer_neuron_perfs[fwd_indices_np].tolist()
                else:
                    fwd_perfs = [current_layer_min_perf] * n_fwd
                next_layer_perfs = np.asarray(kept_perfs_list + fwd_perfs, dtype=np.float64)
            else:
                full_idx0 = sel_idx0
                full_w = sel_w
                full_mean = sel_mean
                full_std = sel_std
                full_tid = sel_tid
                full_lock = torch.zeros(n_selected_unlocked, dtype=torch.bool, device=dev)
                next_layer_perfs = np.asarray(kept_perfs_list, dtype=np.float64)

            kept_neuron_count = int(full_w.shape[0])
            self.layer_perfs.append(current_layer_min_perf)

            if self.logs:
                sys.stdout.write(
                    f"Done training layer {layer_no}. Neuron count: {kept_neuron_count}. "
                    f"Min {self._crit_label}: {current_layer_min_perf:e}.\n"
                )

            # Check rollback / stopping criteria
            if layer_no > 1:
                grad = self._prev_min_perf - current_layer_min_perf
                is_larger_perf = (
                    eval_stop
                    and (layer_no > self.min_layers)
                    and (current_layer_min_perf >= self._prev_min_perf)
                )
                is_min_grad = (
                    eval_stop
                    and (layer_no > self.min_layers)
                    and (grad < self.min_grad)
                )
                is_min_perf = (
                    eval_stop
                    and (layer_no > self.min_layers)
                    and (current_layer_min_perf <= self.min_performance)
                )
                is_max_layer = (self.max_layers > 0) and (layer_no >= self.max_layers)

                if is_nan_present:
                    if self.logs:
                        sys.stdout.write(
                            f"Layer {layer_no} got invalid weights (NaN). It will be removed.\n"
                        )
                    self._current_min_perf = self._prev_min_perf
                    self._finish_growth()
                    return False

                if is_larger_perf:
                    if self.logs:
                        sys.stdout.write(
                            f"Layer {layer_no} Performance ({current_layer_min_perf:e}) >= "
                            f"layer {layer_no - 1} Performance ({self._prev_min_perf:e}). "
                            f"Layer {layer_no} removed.\n"
                        )
                    self._current_min_perf = self._prev_min_perf
                    self._finish_growth()
                    return False
            else:
                grad = 0.0
                is_min_grad = False
                is_min_perf = (
                    eval_stop
                    and (self.min_layers == 0)
                    and (self.min_performance > 0.0 and current_layer_min_perf <= self.min_performance)
                )
                is_max_layer = (self.max_layers > 0) and (layer_no >= self.max_layers)

            # Materialize and insert [norm, activation] + layer before GMDHOutput
            tmpl.materialize(
                full_idx0,
                full_w,
                lock_mask=full_lock,
                target_id=full_tid,
                layer_number=layer_no,
            )
            if tmpl.chunk_size == DEFAULT_CHUNK_SIZE:
                tmpl.chunk_size = self.chunk_size
            if tmpl.max_chunk_bytes == DEFAULT_MAX_CHUNK_BYTES:
                tmpl.max_chunk_bytes = self.max_chunk_bytes
            if not self.gradient_during_growth:
                tmpl.W.requires_grad_(False)
            else:
                tmpl.W.requires_grad_(True)
                tmpl._refresh_grad_hook()

            new_params: list[nn.Parameter] = []
            if layer_no > 1:
                if self._pending_norm is not None:
                    self.model.insert(self.model.output_index, self._pending_norm)
                    if self.gradient_during_growth and self._pending_norm.affine:
                        if self._pending_norm.weight is not None and self._pending_norm.bias is not None:
                            self._pending_norm.weight.requires_grad_(True)
                            self._pending_norm.bias.requires_grad_(True)
                            self._pending_norm._refresh_affine_hooks()
                            new_params.extend([self._pending_norm.weight, self._pending_norm.bias])
                    self._pending_norm = None
                if self._pending_activation is not None:
                    self.model.insert(self.model.output_index, self._pending_activation)
                    if self.gradient_during_growth:
                        for p in self._pending_activation.parameters():
                            p.requires_grad_(True)
                            new_params.append(p)
                    self._pending_activation = None

            self.model.insert(self.model.output_index, tmpl)
            if self.gradient_during_growth:
                new_params.append(tmpl.W)
                self._register_parameters_in_optimizer(new_params)

            self.model.output_module.set_index(best_out_indices)
            self._current_min_perf = current_layer_min_perf
            self._prev_min_perf = current_layer_min_perf
            self._layer_history.append(
                _LayerRecord(
                    best_out_indices=list(best_out_indices),
                    last_layer_stats=(full_mean, full_std, full_lock),
                    closed_form_perf=current_layer_min_perf,
                )
            )

            if is_min_grad or is_min_perf or is_max_layer or final:
                if self.logs:
                    if is_min_grad:
                        sys.stdout.write(f"Minimum gradient reached: {grad:e}. Stop training.\n")
                    elif is_min_perf:
                        sys.stdout.write(
                            f"Minimum performance ({self.min_performance:e}) reached: {current_layer_min_perf:e}. Stop training\n "
                        )
                    elif is_max_layer:
                        sys.stdout.write(f"Maximum layer number reached: {layer_no}. Stop training\n ")
                self._finish_growth()
                return False

            if kept_neuron_count < self.min_inputs:
                if self.logs:
                    sys.stdout.write(
                        f"Layer {layer_no} remained with less than {self.min_inputs} neurons. Stop training.\n"
                    )
                self._finish_growth()
                return False

            # Prepare pending norm & activation for the next cycle (layer_no + 1)
            self._last_layer_neuron_perfs = next_layer_perfs
            self._last_layer_stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = (
                full_mean,
                full_std,
                full_lock,
            )
            self._reset_cycle_buffers()
            self._prepare_pending_pre_modules(full_mean, full_std, full_lock)
            return True
