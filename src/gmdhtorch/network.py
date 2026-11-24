"""PyTorch GMDH modules: GMDHModel, GMDHLayer, GMDHNorm, GMDHActivation, SoftClip, GMDHOutput,
and layer factory presets (combi_layers, default_layers).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import sys
from typing import Any, Callable, Literal, Sequence

import numpy as np
import torch
from torch import nn

from .polynomial import (
    _cached_combinations,
    comb_no,
    gmdh_coeff_count,
    gmdh_preprocess,
    precompute_layer_powers_torch,
    resolve_candidate_indices,
)
from .solvers import Solver
from .utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_BYTES,
    effective_batch_size,
    resolve_acc_dtype,
    resolve_dtype,
)

PerformanceCriterion = Literal["MSE", "R2"]
_UNSET: Any = object()


class SoftClip(nn.Module):
    """Smooth symmetric saturating activation :math:`c \\tanh(z / c)`."""

    def __init__(
        self,
        c: float = 3.0,
        *,
        bound: float | None = None,
        learnable: bool = False,
        dtype: str | torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        eff_c = float(bound) if bound is not None else float(c)
        if eff_c <= 0.0:
            raise ValueError("SoftClip bound `c` must be > 0.")
        self.learnable = bool(learnable)
        dev = torch.device(device) if device is not None else torch.device("cpu")
        dt = resolve_dtype(dtype, device=dev) if dtype is not None else torch.get_default_dtype()
        self.dtype: torch.dtype = dt
        self.device: torch.device = dev
        if self.learnable:
            self.c = nn.Parameter(torch.tensor(eff_c, dtype=dt, device=dev))
        else:
            self.register_buffer("c", torch.tensor(eff_c, dtype=dt, device=dev))

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> SoftClip:
        super()._apply(fn, recurse=recurse)
        self.dtype = self.c.dtype
        self.device = self.c.device
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bound = torch.clamp(self.c.to(dtype=x.dtype, device=x.device), min=1e-6)
        return bound * torch.tanh(x / bound)

    def extra_repr(self) -> str:
        val = float(self.c.detach().item())
        return f"c={val:g}, learnable={self.learnable}"


class GMDHNorm(nn.Module):
    """Per-neuron frozen standardization :math:`(z - \\mu) / \\sigma` with optional learnable affine
    parameters for fine-tuning. Forwarded channels (`passthrough_mask == True`) pass through untouched.
    """

    def __init__(
        self,
        *,
        affine: bool = False,
        eps: float = 1e-12,
        dtype: str | torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.affine = bool(affine)
        self.eps = float(eps)
        dev = torch.device(device)
        dt = resolve_dtype(dtype, device=dev)
        self.dtype: torch.dtype = dt
        self.device: torch.device = dev
        self.acc_dtype: torch.dtype = resolve_acc_dtype(dev)

        self.register_buffer("mean", torch.empty(0, dtype=dt, device=dev))
        self.register_buffer("std", torch.empty(0, dtype=dt, device=dev))
        self.register_buffer("passthrough_mask", torch.empty(0, dtype=torch.bool, device=dev))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool, device=dev))

        if self.affine:
            self.weight = nn.Parameter(torch.empty(0, dtype=dt, device=dev))
            self.bias = nn.Parameter(torch.empty(0, dtype=dt, device=dev))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self._weight_hook_handle: Any = None
        self._bias_hook_handle: Any = None

        self._accum_sum: torch.Tensor | None = None
        self._accum_sq_sum: torch.Tensor | None = None
        self._accum_count: int = 0
        self.register_load_state_dict_post_hook(_post_load_refresh_affine_hooks)

    @property
    def out_features(self) -> int:
        return int(self.mean.numel())

    def _refresh_affine_hooks(self) -> None:
        if self._weight_hook_handle is not None:
            self._weight_hook_handle.remove()
            self._weight_hook_handle = None
        if self._bias_hook_handle is not None:
            self._bias_hook_handle.remove()
            self._bias_hook_handle = None

        if self.affine and self.weight is not None and self.bias is not None and self.weight.requires_grad:
            if self.passthrough_mask.numel() > 0 and bool(torch.any(self.passthrough_mask)):

                def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
                    if self.passthrough_mask.numel() == grad.numel() and bool(torch.any(self.passthrough_mask)):
                        return grad.masked_fill(self.passthrough_mask, 0.0)
                    return grad

                self._weight_hook_handle = self.weight.register_hook(_mask_grad)
                self._bias_hook_handle = self.bias.register_hook(_mask_grad)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> GMDHNorm:
        super()._apply(fn, recurse=recurse)
        self.dtype = self.mean.dtype
        self.device = self.mean.device
        self.acc_dtype = resolve_acc_dtype(self.device)
        return self

    def set_stats(
        self,
        mean: torch.Tensor | np.ndarray,
        std: torch.Tensor | np.ndarray,
        passthrough_mask: torch.Tensor | np.ndarray | None = None,
    ) -> GMDHNorm:
        """Materialize frozen per-neuron ``mean`` and ``std`` statistics."""
        dev = self.device
        dt = self.dtype
        m_t = torch.as_tensor(mean, dtype=dt, device=dev).reshape(-1)
        s_t = torch.as_tensor(std, dtype=dt, device=dev).reshape(-1)
        s_t = torch.where(torch.abs(s_t) > self.eps, s_t, torch.ones_like(s_t))
        width = int(m_t.numel())

        if passthrough_mask is None:
            pm_t = torch.zeros(width, dtype=torch.bool, device=dev)
        else:
            pm_t = torch.as_tensor(passthrough_mask, dtype=torch.bool, device=dev).reshape(-1)

        if bool(torch.any(pm_t)):
            m_t = torch.where(pm_t, torch.zeros_like(m_t), m_t)
            s_t = torch.where(pm_t, torch.ones_like(s_t), s_t)

        self.mean = m_t.detach().clone()
        self.std = s_t.detach().clone()
        self.passthrough_mask = pm_t.detach().clone()
        self.initialized = self.initialized.to(device=dev)
        self.initialized.fill_(True)

        if self.affine and self.weight is not None and self.bias is not None:
            w_req = bool(self.weight.requires_grad)
            b_req = bool(self.bias.requires_grad)
            self.weight = nn.Parameter(
                torch.ones(width, dtype=m_t.dtype, device=m_t.device), requires_grad=w_req
            )
            self.bias = nn.Parameter(
                torch.zeros(width, dtype=m_t.dtype, device=m_t.device), requires_grad=b_req
            )
            self._refresh_affine_hooks()

        return self

    def reset_accumulation(self) -> None:
        self._accum_sum = None
        self._accum_sq_sum = None
        self._accum_count = 0

    def accumulate_batch(self, x: torch.Tensor) -> None:
        """Accumulate batch statistics in ``self.acc_dtype`` for input-stage normalization."""
        with torch.no_grad():
            x_acc = x.detach().to(dtype=self.acc_dtype, device=self.device)
            if x_acc.ndim == 1:
                x_acc = x_acc.unsqueeze(0)
            b_sum = torch.sum(x_acc, dim=0)
            b_sq = torch.sum(x_acc * x_acc, dim=0)
            if self._accum_sum is None or self._accum_sq_sum is None:
                self._accum_sum = b_sum
                self._accum_sq_sum = b_sq
            else:
                self._accum_sum = self._accum_sum + b_sum
                self._accum_sq_sum = self._accum_sq_sum + b_sq
            self._accum_count += int(x_acc.shape[0])

    def finalize_stats(self, passthrough_mask: torch.Tensor | np.ndarray | None = None) -> GMDHNorm:
        """Finalize accumulated batch statistics into frozen ``mean`` and ``std`` buffers."""
        if self._accum_sum is None or self._accum_sq_sum is None or self._accum_count == 0:
            raise RuntimeError("Cannot finalize GMDHNorm statistics: no batches were accumulated.")
        n = float(self._accum_count)
        mean_acc = self._accum_sum / n
        var_acc = torch.clamp(self._accum_sq_sum / n - mean_acc * mean_acc, min=0.0)
        std_acc = torch.sqrt(
            torch.where(var_acc > self.eps * self.eps, var_acc, torch.ones_like(var_acc))
        )
        self.set_stats(
            mean_acc.to(dtype=self.dtype, device=self.device),
            std_acc.to(dtype=self.dtype, device=self.device),
            passthrough_mask=passthrough_mask,
        )
        self.reset_accumulation()
        return self

    def prune(
        self,
        keep_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> GMDHNorm:
        """Slice neuron statistics (and affine parameters in place) to ``keep_indices``."""
        if not bool(self.initialized.item()):
            return self
        keep = keep_indices.to(dtype=torch.long, device=self.mean.device).reshape(-1)
        self.mean = self.mean[keep].contiguous()
        self.std = self.std[keep].contiguous()
        if self.passthrough_mask.numel() > 0:
            self.passthrough_mask = self.passthrough_mask[keep].contiguous()
        if self.affine and self.weight is not None and self.bias is not None:
            _slice_parameter_in_place(self, "weight", keep, dim=0, optimizer=optimizer)
            _slice_parameter_in_place(self, "bias", keep, dim=0, optimizer=optimizer)
            self._refresh_affine_hooks()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized.item()) or self.mean.numel() == 0:
            return x
        m = self.mean.to(dtype=x.dtype, device=x.device)
        s = self.std.to(dtype=x.dtype, device=x.device)
        normed = (x - m) / s
        if self.affine and self.weight is not None and self.bias is not None and self.weight.numel() == m.numel():
            w = self.weight.to(dtype=x.dtype, device=x.device)
            b = self.bias.to(dtype=x.dtype, device=x.device)
            normed = normed * w + b
        if self.passthrough_mask.numel() == x.shape[-1] and bool(torch.any(self.passthrough_mask)):
            pm = self.passthrough_mask.to(device=x.device)
            return torch.where(pm.unsqueeze(0), x, normed)
        return normed

    def extra_repr(self) -> str:
        init_flag = bool(self.initialized.item())
        return f"features={self.out_features}, affine={self.affine}, initialized={init_flag}"


def instantiate_activation_module(
    spec: nn.Module | type[nn.Module] | Callable[[], nn.Module],
) -> nn.Module:
    """Create a fresh ``nn.Module`` from a prototype instance, class, or factory (§5.5)."""
    if isinstance(spec, GMDHActivation):
        return copy.deepcopy(spec.activation)
    if isinstance(spec, nn.Module):
        return copy.deepcopy(spec)
    if isinstance(spec, type) and issubclass(spec, nn.Module):
        return spec()
    if callable(spec):
        mod = spec()
        if isinstance(mod, nn.Module):
            return mod
    raise TypeError(
        f"Activation specification must be an nn.Module instance, class, or factory; got {type(spec)!r}."
    )


class GMDHActivation(nn.Module):
    """Wraps any ``nn.Module`` activation and applies it only to newly fitted neurons
    (`passthrough_mask == False`), leaving forwarded channels untouched.
    """

    def __init__(
        self,
        activation: nn.Module | type[nn.Module] | Callable[[], nn.Module],
        passthrough_mask: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.activation: nn.Module = instantiate_activation_module(activation)
        if passthrough_mask is None:
            pm_t = torch.empty(0, dtype=torch.bool)
        else:
            pm_t = torch.as_tensor(passthrough_mask, dtype=torch.bool).reshape(-1).clone()
        self.register_buffer("passthrough_mask", pm_t)

    @property
    def out_features(self) -> int:
        return int(self.passthrough_mask.numel())

    def set_passthrough_mask(self, mask: torch.Tensor | np.ndarray) -> GMDHActivation:
        dev = mask.device if isinstance(mask, torch.Tensor) else self.passthrough_mask.device
        self.passthrough_mask = torch.as_tensor(
            mask, dtype=torch.bool, device=dev
        ).reshape(-1).clone()
        self.activation.to(device=dev)
        return self

    def prune(self, keep_indices: torch.Tensor) -> GMDHActivation:
        if self.passthrough_mask.numel() > 0:
            keep = keep_indices.to(dtype=torch.long, device=self.passthrough_mask.device).reshape(-1)
            self.passthrough_mask = self.passthrough_mask[keep].contiguous()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.passthrough_mask.numel() == x.shape[-1] and bool(torch.any(self.passthrough_mask)):
            if bool(torch.all(self.passthrough_mask)):
                return x
            pm = self.passthrough_mask.to(device=x.device)
            active = ~pm
            out = x.clone()
            out[:, active] = self.activation(x[:, active])
            return out
        return self.activation(x)

    def extra_repr(self) -> str:
        n_fwd = int(torch.sum(self.passthrough_mask).item()) if self.passthrough_mask.numel() > 0 else 0
        return f"passthrough={n_fwd}/{self.out_features}"


def _slice_parameter_in_place(
    module: nn.Module,
    attr_name: str,
    keep_indices: torch.Tensor,
    *,
    dim: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
) -> nn.Parameter:
    """Slice ``getattr(module, attr_name)`` along ``dim`` into a fresh ``nn.Parameter``
    and update any matching references and state tensors in ``optimizer``.
    """
    old_param: nn.Parameter = getattr(module, attr_name)
    idx = keep_indices.to(dtype=torch.long, device=old_param.device)
    old_len = int(old_param.shape[dim])
    was_req_grad = bool(old_param.requires_grad)
    old_grad = old_param.grad

    new_data = torch.index_select(old_param.detach(), dim, idx).contiguous()
    new_param = nn.Parameter(new_data, requires_grad=was_req_grad)
    if old_grad is not None and was_req_grad:
        new_param.grad = torch.index_select(old_grad.detach(), dim, idx).contiguous()

    setattr(module, attr_name, new_param)

    if optimizer is not None:
        for group in optimizer.param_groups:
            params_list = group["params"]
            for i, p in enumerate(params_list):
                if p is old_param:
                    params_list[i] = new_param
        if old_param in optimizer.state:
            state_dict = optimizer.state.pop(old_param)
            for key, val in list(state_dict.items()):
                if isinstance(val, torch.Tensor) and val.ndim >= 1 and val.shape[dim] == old_len:
                    state_dict[key] = torch.index_select(val, dim, idx.to(device=val.device)).contiguous()
            optimizer.state[new_param] = state_dict

    return new_param


def _init_gmdh_weights(
    w: torch.Tensor,
    *,
    init: str | Callable[[torch.Tensor], None] = "kaiming",
    init_std: float = 0.01,
    rng: int | np.random.Generator | None = None,
) -> torch.Tensor:
    """Initialize polynomial weights ``(M, P)`` with zero intercept ``W[:, 0] = 0``."""
    with torch.no_grad():
        w.zero_()
        if callable(init):
            init(w)
            return w
        init_key = str(init).strip().lower()
        if init_key == "zeros":
            return w
        f_in = max(1, int(w.shape[1]) - 1)
        if init_key == "kaiming":
            std = math.sqrt(2.0 / f_in)
        elif init_key == "xavier":
            std = math.sqrt(1.0 / f_in)
        elif init_key == "normal":
            std = float(init_std)
        else:
            raise ValueError(
                f"Unknown init scheme {init!r}; expected 'kaiming', 'xavier', 'normal', 'zeros', or callable."
            )
        if w.shape[1] > 1 and w.shape[0] > 0:
            gen = np.random.default_rng(rng)
            vals = gen.normal(0.0, std, size=(int(w.shape[0]), int(w.shape[1]) - 1))
            w[:, 1:] = torch.as_tensor(vals, dtype=w.dtype, device=w.device)
    return w


class GMDHLayer(nn.Module):
    """Differentiable PyTorch layer representing either a GMDH layer template or a materialized
    GMDH polynomial layer.

    Locked input forwarders (`lock_mask == True`, *"expedidores"*) have their weights frozen
    via a gradient hook and always bypass activations/normalization so forwarded signals remain
    strict identity pass-throughs.
    """

    def __init__(
        self,
        layer: GMDHLayer | None = None,
        dtype: str | torch.dtype | None = None,
        device: torch.device | str | None = None,
        *,
        inputs: int | None = None,
        input_count: int | None = None,
        in_features: int | None = None,
        out_features: int | None = None,
        degree: int | None = None,
        interaction: int | None = None,
        forward: bool = False,
        min_neurons: int = 1,
        max_neurons: int | None = 30,
        candidates: int | Callable[[int, int], np.ndarray | torch.Tensor] | None = None,
        init: str | Callable[[torch.Tensor], None] = "kaiming",
        init_std: float = 0.01,
        rng: int | np.random.Generator | None = None,
        input_scores: torch.Tensor | np.ndarray | Sequence[float] | None = None,
        higher_is_better: bool = False,
        layer_number: int = 1,
        idx0: torch.Tensor | np.ndarray | None = None,
        I: torch.Tensor | np.ndarray | None = None,
        W: torch.Tensor | np.ndarray | None = None,
        lock_mask: torch.Tensor | np.ndarray | None = None,
        lock: torch.Tensor | np.ndarray | None = None,
        target_id: torch.Tensor | np.ndarray | None = None,
        clone_tensors: bool = True,
        chunk_size: int | None = DEFAULT_CHUNK_SIZE,
        max_chunk_bytes: int | None = DEFAULT_MAX_CHUNK_BYTES,
    ) -> None:
        super().__init__()
        if device is not None:
            dev = torch.device(device)
        elif isinstance(layer, GMDHLayer):
            dev = layer.W.device
        elif isinstance(W, torch.Tensor):
            dev = W.device
        else:
            dev = torch.device("cpu")

        if dtype is not None:
            dt = resolve_dtype(dtype, device=dev)
        elif isinstance(layer, GMDHLayer):
            dt = resolve_dtype(layer.W.dtype, device=dev)
        elif isinstance(W, torch.Tensor) and W.is_floating_point():
            dt = resolve_dtype(W.dtype, device=dev)
        else:
            dt = resolve_dtype(None, device=dev)

        self._acc_dtype: torch.dtype = resolve_acc_dtype(dev)

        if isinstance(layer, GMDHLayer):
            self.layer_number = int(layer.layer_number)
            self.input_count = int(layer.input_count)
            self.degree = int(layer.degree)
            self.interaction = int(layer.interaction)
            self.forward_inputs = bool(layer.forward_inputs)
            self.min_neurons = int(layer.min_neurons)
            self.max_neurons = int(layer.max_neurons) if layer.max_neurons is not None else 0
            self.candidates = layer.candidates
            self.init = layer.init
            self.init_std = float(layer.init_std)
            self.chunk_size: int | None = layer.chunk_size
            self.max_chunk_bytes: int | None = layer.max_chunk_bytes
            idx0_t = layer.idx0.to(dtype=torch.long, device=dev)
            lock_t = layer.lock_mask.to(dtype=torch.bool, device=dev)
            tid_t = layer.target_id.to(dtype=torch.long, device=dev)
            w_init = layer.W.detach().to(dtype=dt, device=dev)
        else:
            eff_inputs = (
                int(inputs)
                if inputs is not None
                else (int(input_count) if input_count is not None else 2)
            )
            eff_degree = eff_inputs if (degree is None or int(degree) < 1) else int(degree)
            eff_interaction = (
                eff_inputs
                if interaction is None
                else min(eff_inputs, max(0, int(interaction)))
            )

            self.layer_number = int(layer_number)
            self.input_count = eff_inputs
            self.degree = eff_degree
            self.interaction = eff_interaction
            self.forward_inputs = bool(forward)
            self.min_neurons = int(min_neurons)
            self.max_neurons = int(max_neurons) if max_neurons is not None else 0
            self.candidates = candidates
            self.init = init
            self.init_std = float(init_std)
            self.chunk_size = int(chunk_size) if chunk_size is not None else None
            self.max_chunk_bytes = int(max_chunk_bytes) if max_chunk_bytes is not None else None

            if idx0 is None and I is not None:
                idx0 = (
                    np.asarray(I, dtype=np.int64) - 1
                    if not isinstance(I, torch.Tensor)
                    else (I.to(torch.long) - 1)
                )

            if idx0 is not None and W is not None:
                idx0_t = torch.as_tensor(idx0, dtype=torch.long, device=dev)
                if idx0_t.ndim == 1:
                    idx0_t = idx0_t.unsqueeze(1)
                self.input_count = int(idx0_t.shape[1])
                if degree is None:
                    self.degree = self.input_count
                if interaction is None:
                    self.interaction = self.input_count
                w_init = torch.as_tensor(W, dtype=dt, device=dev)
                if w_init.ndim == 1:
                    w_init = w_init.unsqueeze(0)
                if lock_mask is not None:
                    lock_t = torch.as_tensor(lock_mask, dtype=torch.bool, device=dev)
                elif lock is not None:
                    lock_t = torch.as_tensor(lock, device=dev) != 0
                else:
                    lock_t = torch.zeros(w_init.shape[0], dtype=torch.bool, device=dev)
                if target_id is not None:
                    tid_t = torch.as_tensor(target_id, dtype=torch.long, device=dev).reshape(-1)
                else:
                    tid_t = torch.zeros(w_init.shape[0], dtype=torch.long, device=dev)
            elif in_features is not None:
                n_prev = int(in_features)
                if n_prev < 1:
                    raise ValueError("`in_features` must be >= 1.")
                self.input_count = max(1, min(self.input_count, n_prev))
                c_count = gmdh_coeff_count(
                    self.input_count, self.degree, interaction=self.interaction
                )
                scores_np = (
                    input_scores.detach().cpu().numpy()
                    if isinstance(input_scores, torch.Tensor)
                    else (np.asarray(input_scores, dtype=np.float64) if input_scores is not None else None)
                )
                if self.forward_inputs:
                    max_fwd = (self.max_neurons // 2) if self.max_neurons > 0 else n_prev
                    if n_prev <= max_fwd:
                        fwd_indices = np.arange(n_prev, dtype=np.int64)
                    elif scores_np is not None:
                        order = np.argsort(-scores_np if higher_is_better else scores_np, kind="mergesort")
                        fwd_indices = order[:max_fwd]
                    else:
                        fwd_indices = np.arange(max_fwd, dtype=np.int64)
                    n_fwd = int(fwd_indices.size)
                    cand_budget = (self.max_neurons - n_fwd) if self.max_neurons > 0 else out_features
                    if out_features is not None and int(out_features) > 0:
                        cand_budget = min(cand_budget, int(out_features)) if cand_budget is not None else int(out_features)
                else:
                    fwd_indices = np.empty(0, dtype=np.int64)
                    n_fwd = 0
                    if out_features is not None and int(out_features) > 0:
                        cand_budget = int(out_features)
                        if self.max_neurons > 0:
                            cand_budget = min(cand_budget, self.max_neurons)
                    else:
                        cand_budget = self.max_neurons if self.max_neurons > 0 else None

                cand_idx0_np = resolve_candidate_indices(
                    n_prev,
                    self.input_count,
                    self.candidates,
                    max_candidates=cand_budget,
                    input_scores=scores_np,
                    higher_is_better=higher_is_better,
                    rng=rng,
                )
                n_cand = int(cand_idx0_np.shape[0])
                sel_idx0 = torch.as_tensor(cand_idx0_np, dtype=torch.long, device=dev)
                sel_w = torch.empty((n_cand, c_count), dtype=dt, device=dev)
                _init_gmdh_weights(sel_w, init=init, init_std=init_std, rng=rng)
                sel_lock = torch.zeros(n_cand, dtype=torch.bool, device=dev)
                sel_tid = torch.zeros(n_cand, dtype=torch.long, device=dev)

                if n_fwd > 0:
                    fwd_idx0 = torch.as_tensor(fwd_indices, dtype=torch.long, device=dev)[:, None].repeat(1, self.input_count)
                    fwd_w = torch.zeros((n_fwd, c_count), dtype=dt, device=dev)
                    fwd_w[:, 1] = 1.0
                    fwd_lock = torch.ones(n_fwd, dtype=torch.bool, device=dev)
                    fwd_tid = torch.full((n_fwd,), -1, dtype=torch.long, device=dev)

                    idx0_t = torch.cat([sel_idx0, fwd_idx0], dim=0)
                    w_init = torch.cat([sel_w, fwd_w], dim=0)
                    lock_t = torch.cat([sel_lock, fwd_lock], dim=0)
                    tid_t = torch.cat([sel_tid, fwd_tid], dim=0)
                else:
                    idx0_t = sel_idx0
                    w_init = sel_w
                    lock_t = sel_lock
                    tid_t = sel_tid
            else:
                c_count = gmdh_coeff_count(
                    self.input_count, self.degree, interaction=self.interaction
                )
                idx0_t = torch.empty((0, self.input_count), dtype=torch.long, device=dev)
                w_init = torch.empty((0, c_count), dtype=dt, device=dev)
                lock_t = torch.empty(0, dtype=torch.bool, device=dev)
                tid_t = torch.empty(0, dtype=torch.long, device=dev)

        self.neuron_count = int(w_init.shape[0])
        self.weight_count = int(w_init.shape[1])
        self._grad_hook_handle: Any = None

        self.register_buffer("idx0", idx0_t.clone() if clone_tensors else idx0_t)
        self.register_buffer("lock_mask", lock_t.clone() if clone_tensors else lock_t)
        self.register_buffer("target_id", tid_t.clone() if clone_tensors else tid_t)
        self.register_buffer("_W_init", None, persistent=False)
        self.W = nn.Parameter(w_init.clone() if clone_tensors else w_init)

        self._comb_degrees: list[int] = []
        self._rebuild_combination_buffers()
        self._refresh_grad_hook()
        self.register_load_state_dict_post_hook(_post_load_refresh_grad_hook)

    @property
    def dtype(self) -> torch.dtype:
        return self.W.dtype

    @property
    def device(self) -> torch.device:
        return self.W.device

    @property
    def acc_dtype(self) -> torch.dtype:
        return self._acc_dtype

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> GMDHLayer:
        super()._apply(fn, recurse=recurse)
        self._acc_dtype = resolve_acc_dtype(self.W.device)
        return self

    @property
    def inputs(self) -> int:
        return self.input_count

    @property
    def is_materialized(self) -> bool:
        return self.neuron_count > 0

    @property
    def out_features(self) -> int:
        return self.neuron_count

    @property
    def passthrough_mask(self) -> torch.Tensor:
        return self.lock_mask

    def _rebuild_combination_buffers(self) -> None:
        for d_old in self._comb_degrees:
            buf_name = f"_comb_{d_old}"
            if hasattr(self, buf_name):
                delattr(self, buf_name)
        self._comb_degrees = []
        q = min(self.input_count, max(0, int(self.interaction)))
        dev = self.W.device
        for i in range(2, q + 1):
            cidxs = _cached_combinations(self.input_count, i)
            buf_name = f"_comb_{i}"
            self.register_buffer(
                buf_name,
                torch.as_tensor(cidxs, dtype=torch.long, device=dev),
                persistent=False,
            )
            self._comb_degrees.append(i)

    def materialize(
        self,
        idx0: torch.Tensor | np.ndarray,
        W: torch.Tensor | np.ndarray,
        *,
        lock_mask: torch.Tensor | np.ndarray | None = None,
        target_id: torch.Tensor | np.ndarray | None = None,
        layer_number: int | None = None,
    ) -> GMDHLayer:
        """Populate this layer with selected connectivity ``idx0`` and fitted weights ``W``."""
        dev = W.device if isinstance(W, torch.Tensor) else self.W.device
        dt = W.dtype if isinstance(W, torch.Tensor) and W.is_floating_point() else self.W.dtype
        if layer_number is not None:
            self.layer_number = int(layer_number)

        idx0_t = torch.as_tensor(idx0, dtype=torch.long, device=dev)
        if idx0_t.ndim == 1:
            idx0_t = idx0_t.unsqueeze(1)
        w_t = torch.as_tensor(W, dtype=dt, device=dev)
        if w_t.ndim == 1:
            w_t = w_t.unsqueeze(0)

        n_neurons = int(w_t.shape[0])
        self.input_count = int(idx0_t.shape[1])
        self.neuron_count = n_neurons
        self.weight_count = int(w_t.shape[1])

        if lock_mask is None:
            lock_t = torch.zeros(n_neurons, dtype=torch.bool, device=dev)
        else:
            lock_t = torch.as_tensor(lock_mask, dtype=torch.bool, device=dev).reshape(-1)

        if target_id is None:
            tid_t = torch.zeros(n_neurons, dtype=torch.long, device=dev)
        else:
            tid_t = torch.as_tensor(target_id, dtype=torch.long, device=dev).reshape(-1)

        self.idx0 = idx0_t.contiguous()
        self.lock_mask = lock_t.contiguous()
        self.target_id = tid_t.contiguous()
        self._W_init = w_t.detach().clone()
        was_req = bool(self.W.requires_grad)
        if self._grad_hook_handle is not None:
            self._grad_hook_handle.remove()
            self._grad_hook_handle = None
        self.W = nn.Parameter(w_t.contiguous(), requires_grad=was_req)
        self._acc_dtype = resolve_acc_dtype(self.W.device)

        self._rebuild_combination_buffers()
        self._refresh_grad_hook()
        return self

    def copy(self) -> GMDHLayer:
        """Return a deep copy of this layer."""
        return GMDHLayer(self, dtype=self.W.dtype, device=self.W.device, clone_tensors=True)

    def to_gmdh_layer(self) -> GMDHLayer:
        """Backward-compatible helper returning a copy of this ``GMDHLayer``."""
        return self.copy()

    @property
    def W_init(self) -> torch.Tensor:
        """Lazily materialized initial weights snapshot for proximal trust-region fine-tuning."""
        if self._W_init is None or self._W_init.shape != self.W.shape:
            self._W_init = self.W.detach().clone()
        return self._W_init

    @property
    def I(self) -> np.ndarray:
        """1-based neuron input connectivity matrix ``(M, input_count)`` as a NumPy ``int64`` array."""
        return (self.idx0.detach().cpu().numpy() + 1).astype(np.int64)

    @I.setter
    def I(self, value: np.ndarray | torch.Tensor) -> None:
        idx0_new = (
            (value.to(dtype=torch.long, device=self.idx0.device) - 1)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(
                np.asarray(value, dtype=np.int64) - 1,
                dtype=torch.long,
                device=self.idx0.device,
            )
        )
        if idx0_new.ndim == 1:
            idx0_new = idx0_new.unsqueeze(1)
        self.idx0 = idx0_new
        self.neuron_count = int(idx0_new.shape[0])

    @property
    def lock(self) -> np.ndarray:
        """Lock mask ``(M,)`` (`1` for locked forwarders, `0` for unlocked) as a NumPy ``int64`` array."""
        return self.lock_mask.detach().cpu().numpy().astype(np.int64)

    @lock.setter
    def lock(self, value: np.ndarray | torch.Tensor) -> None:
        self.lock_mask = torch.as_tensor(value, device=self.lock_mask.device) != 0
        self._refresh_grad_hook()

    def _refresh_grad_hook(self) -> None:
        if self._grad_hook_handle is not None:
            self._grad_hook_handle.remove()
            self._grad_hook_handle = None
        if self.W.requires_grad and self.lock_mask.numel() > 0:
            self._grad_hook_handle = self.W.register_hook(self._mask_locked_grad)

    def _mask_locked_grad(self, grad: torch.Tensor) -> torch.Tensor:
        if self.lock_mask.numel() == grad.shape[0] and bool(torch.any(self.lock_mask)):
            return grad.masked_fill(self.lock_mask.unsqueeze(1), 0.0)
        return grad

    def keep_neurons(
        self,
        keep_indices: torch.Tensor,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        """Slice layer buffers and parameters in place to retain only ``keep_indices`` (0-based)."""
        keep = keep_indices.to(dtype=torch.long, device=self.W.device).reshape(-1)
        self.idx0 = self.idx0[keep].contiguous()
        self.lock_mask = self.lock_mask[keep].contiguous()
        if self.target_id.numel() > 0:
            self.target_id = self.target_id[keep].contiguous()
        if self._W_init is not None:
            self._W_init = self._W_init[keep].contiguous()
        if self._grad_hook_handle is not None:
            self._grad_hook_handle.remove()
            self._grad_hook_handle = None
        _slice_parameter_in_place(self, "W", keep, dim=0, optimizer=optimizer)
        self.neuron_count = int(self.W.shape[0])
        self._refresh_grad_hook()

    def preprocess_batch(
        self,
        x: torch.Tensor,
        idx0_subset: torch.Tensor | None = None,
        precomputed_powers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build 3D polynomial design tensor ``(B, N, coeff_count)`` in PyTorch."""
        idx = self.idx0 if idx0_subset is None else idx0_subset
        b_count = int(idx.shape[0])
        n_samples = int(x.shape[0])
        n_inputs = self.input_count
        d = self.degree

        if precomputed_powers is None:
            p_powers = precompute_layer_powers_torch(x, d)
        else:
            p_powers = precomputed_powers

        gathered_powers = p_powers[:, idx, :d].permute(1, 0, 2, 3)
        flat_powers = gathered_powers.reshape(b_count, n_samples, n_inputs * d)
        ones = torch.ones((b_count, n_samples, 1), dtype=x.dtype, device=x.device)

        if not self._comb_degrees:
            return torch.cat([ones, flat_powers], dim=2)

        parts = [ones, flat_powers]
        base_inputs = gathered_powers[:, :, :, 0]
        for i in self._comb_degrees:
            cidxs: torch.Tensor = getattr(self, f"_comb_{i}")
            if cidxs.shape[0] > 0:
                cross = base_inputs[:, :, cidxs[:, 0]] * base_inputs[:, :, cidxs[:, 1]]
                for k in range(2, i):
                    cross = cross * base_inputs[:, :, cidxs[:, k]]
                parts.append(cross)

        return torch.cat(parts, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        batch_size: int | None = None,
        *,
        max_chunk_bytes: int | None = _UNSET,
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        n_samples = int(x.shape[0])
        m_neurons = self.neuron_count
        if m_neurons == 0:
            return torch.empty((n_samples, 0), dtype=x.dtype, device=x.device)

        d = self.degree
        p_powers = precompute_layer_powers_torch(x, d)
        eff_req = self.chunk_size if batch_size is None else int(batch_size)
        eff_max_bytes = self.max_chunk_bytes if max_chunk_bytes is _UNSET else max_chunk_bytes
        if m_neurons <= 16 and (eff_req is None or m_neurons <= eff_req):
            b_step = m_neurons
        else:
            b_step = effective_batch_size(
                eff_req,
                n_samples,
                self.weight_count,
                elem_size=x.element_size(),
                max_chunk_bytes=eff_max_bytes,
            )

        if m_neurons <= b_step:
            x_batch = self.preprocess_batch(x, precomputed_powers=p_powers)
            w_use = self.W if self.W.dtype == x_batch.dtype else self.W.to(dtype=x_batch.dtype)
            raw_out = torch.einsum("mnc,mc->nm", x_batch, w_use)
        else:
            chunks = []
            for start in range(0, m_neurons, b_step):
                end = min(start + b_step, m_neurons)
                x_sub = self.preprocess_batch(
                    x,
                    idx0_subset=self.idx0[start:end],
                    precomputed_powers=p_powers,
                )
                w_sub = self.W[start:end]
                if w_sub.dtype != x_sub.dtype:
                    w_sub = w_sub.to(dtype=x_sub.dtype)
                chunks.append(torch.einsum("mnc,mc->nm", x_sub, w_sub))
            raw_out = torch.cat(chunks, dim=1)

        return raw_out

    def extra_repr(self) -> str:
        return (
            f"layer_number={self.layer_number}, inputs={self.input_count}, "
            f"degree={self.degree}, interaction={self.interaction}, "
            f"neurons={self.neuron_count}, weights={self.weight_count}"
        )


def _register_or_replace_optimizer_param(
    old_param: nn.Parameter | None,
    new_param: nn.Parameter | None,
    optimizer: torch.optim.Optimizer | None,
) -> None:
    if optimizer is None:
        return
    if old_param is not None:
        if old_param in optimizer.state:
            del optimizer.state[old_param]
        for group in optimizer.param_groups:
            group["params"] = [p for p in group["params"] if p is not old_param]
    if new_param is not None:
        existing_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if id(new_param) not in existing_ids:
            if optimizer.param_groups:
                optimizer.param_groups[0]["params"].append(new_param)
            else:
                optimizer.add_param_group({"params": [new_param]})


class GMDHOutput(nn.Module):
    """Terminal output head for ``GMDHModel``.

    - ``dense=False`` (default): parameter-free selector picking the best neuron for each of
      ``count`` outputs (`index` buffer).
    - ``dense=True``: learnable linear combiner ``nn.Linear(in_features, count)`` aggregating
      all neurons of the final ``GMDHLayer``.

    Before the first layer is materialized (or before ``in_features`` is set in dense mode),
    returns ``fallback``.
    """

    def __init__(
        self,
        count: int = 1,
        *,
        dense: bool = False,
        in_features: int | None = None,
        bias: bool = True,
        dtype: str | torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if int(count) < 1:
            raise ValueError("GMDHOutput `count` must be >= 1.")
        self.count = int(count)
        self.dense = bool(dense)
        self.use_bias = bool(bias)
        dev = torch.device(device)
        dt = resolve_dtype(dtype, device=dev)
        self.dtype: torch.dtype = dt
        self.device: torch.device = dev
        self.acc_dtype: torch.dtype = resolve_acc_dtype(dev)
        self.register_buffer("index", torch.full((self.count,), -1, dtype=torch.long, device=dev))
        self.register_buffer("fallback", torch.zeros(self.count, dtype=dt, device=dev))
        if self.dense and in_features is not None and int(in_features) > 0:
            self.linear: nn.Linear | None = nn.Linear(
                int(in_features), self.count, bias=self.use_bias, dtype=dt, device=dev
            )
        else:
            self.linear = None

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> GMDHOutput:
        super()._apply(fn, recurse=recurse)
        ref = self.linear.weight if (self.dense and self.linear is not None) else self.fallback
        self.dtype = ref.dtype
        self.device = ref.device
        self.acc_dtype = resolve_acc_dtype(self.device)
        return self

    @property
    def is_initialized(self) -> bool:
        if self.dense:
            return self.linear is not None
        return bool((self.index >= 0).all().item())

    def set_index(self, indices: int | Sequence[int] | np.ndarray | torch.Tensor) -> GMDHOutput:
        idx_t = torch.as_tensor(indices, dtype=torch.long, device=self.index.device).reshape(-1)
        if idx_t.numel() != self.count:
            raise ValueError(f"Expected {self.count} output indices, got {idx_t.numel()}.")
        self.index = idx_t.clone()
        return self

    def set_fallback(self, values: float | Sequence[float] | np.ndarray | torch.Tensor) -> GMDHOutput:
        val_t = torch.as_tensor(values, dtype=self.fallback.dtype, device=self.fallback.device).reshape(-1)
        if val_t.numel() == 1 and self.count > 1:
            val_t = val_t.expand(self.count).clone()
        elif val_t.numel() != self.count:
            raise ValueError(f"Expected {self.count} fallback values, got {val_t.numel()}.")
        self.fallback = val_t.clone()
        return self

    def adapt_width(
        self,
        in_features: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> GMDHOutput:
        """Lazily create ``self.linear`` for ``in_features`` when in ``dense=True`` mode."""
        if not self.dense:
            return self
        if self.linear is not None and self.linear.in_features == int(in_features):
            return self
        dt = self.fallback.dtype
        dev = self.fallback.device
        old_lin = self.linear
        new_lin = nn.Linear(int(in_features), self.count, bias=self.use_bias, dtype=dt, device=dev)
        if old_lin is not None:
            _register_or_replace_optimizer_param(old_lin.weight, None, optimizer)
            if old_lin.bias is not None:
                _register_or_replace_optimizer_param(old_lin.bias, None, optimizer)
        self.linear = new_lin
        _register_or_replace_optimizer_param(None, new_lin.weight, optimizer)
        if new_lin.bias is not None:
            _register_or_replace_optimizer_param(None, new_lin.bias, optimizer)
        return self

    def resize_warm(
        self,
        new_in_features: int,
        *,
        forward_slice: slice | torch.Tensor | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> GMDHOutput:
        """Function-preserving resize of ``dense=True`` head: copies existing weights into
        ``forward_slice`` and zero-initializes new candidate columns.
        """
        if not self.dense:
            return self
        if self.linear is None:
            return self.adapt_width(new_in_features, optimizer=optimizer)

        old_lin = self.linear
        dt = old_lin.weight.dtype
        dev = old_lin.weight.device
        new_lin = nn.Linear(int(new_in_features), self.count, bias=self.use_bias, dtype=dt, device=dev)
        with torch.no_grad():
            new_lin.weight.zero_()
            if forward_slice is not None:
                new_lin.weight[:, forward_slice] = old_lin.weight.detach()
            if new_lin.bias is not None and old_lin.bias is not None:
                new_lin.bias.copy_(old_lin.bias.detach())

        _register_or_replace_optimizer_param(old_lin.weight, new_lin.weight, optimizer)
        if old_lin.bias is not None and new_lin.bias is not None:
            _register_or_replace_optimizer_param(old_lin.bias, new_lin.bias, optimizer)
        self.linear = new_lin
        return self

    def prune(
        self,
        keep_indices: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> GMDHOutput:
        """Slice input columns of ``self.linear`` in place when ``dense=True``."""
        if not self.dense or self.linear is None:
            return self
        keep = keep_indices.to(dtype=torch.long, device=self.linear.weight.device).reshape(-1)
        _slice_parameter_in_place(self.linear, "weight", keep, dim=1, optimizer=optimizer)
        self.linear.in_features = int(keep.numel())
        return self

    def importance_scores(self, h: torch.Tensor) -> np.ndarray:
        """Compute scale-invariant column importance ``||W[:, j]||_2 * RMS(H[:, j])``."""
        if not self.dense or self.linear is None:
            raise RuntimeError("importance_scores() requires GMDHOutput(dense=True) with initialized linear head.")
        with torch.no_grad():
            w = self.linear.weight.detach().to(dtype=self.acc_dtype)
            h_acc = h.detach().to(dtype=self.acc_dtype, device=w.device)
            if h_acc.ndim == 1:
                h_acc = h_acc.unsqueeze(0)
            col_norms = torch.linalg.norm(w, dim=0)
            rms = torch.sqrt(torch.mean(h_acc * h_acc, dim=0))
            return (col_norms * rms).cpu().numpy()

    def remap_after_prune(self, keep_mask: torch.Tensor) -> GMDHOutput:
        if self.dense or not self.is_initialized:
            return self
        km = keep_mask.to(dtype=torch.long, device=self.index.device).reshape(-1)
        remap = torch.cumsum(km, dim=0) - 1
        self.index = remap[self.index]
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        n_samples = int(x.shape[0])
        if self.dense:
            if self.linear is None or int(x.shape[1]) == 0:
                fb = self.fallback.to(dtype=x.dtype, device=x.device).unsqueeze(0)
                return fb.expand(n_samples, self.count).contiguous()
            return self.linear(x)
        if not self.is_initialized or int(x.shape[1]) == 0:
            fb = self.fallback.to(dtype=x.dtype, device=x.device).unsqueeze(0)
            return fb.expand(n_samples, self.count).contiguous()
        idx = self.index.to(device=x.device)
        return x[:, idx]

    def extra_repr(self) -> str:
        if self.dense:
            in_f = self.linear.in_features if self.linear is not None else None
            return f"count={self.count}, dense=True, in_features={in_f}"
        idx_list = self.index.detach().cpu().tolist()
        return f"count={self.count}, dense=False, index={idx_list}"


_ALLOWED_GMDH_MODULES = (
    GMDHLayer,
    GMDHNorm,
    GMDHActivation,
    SoftClip,
    GMDHOutput,
    nn.Linear,
    nn.ReLU,
    nn.Hardtanh,
    nn.Tanh,
    nn.GELU,
    nn.SiLU,
    nn.LeakyReLU,
    nn.Sigmoid,
    nn.ELU,
    nn.Identity,
)

_SIMPLE_ACTIVATION_SPECS: dict[str, tuple[type[nn.Module], tuple[str, ...]]] = {
    "ReLU": (nn.ReLU, ("inplace",)),
    "Hardtanh": (nn.Hardtanh, ("min_val", "max_val", "inplace")),
    "Tanh": (nn.Tanh, ()),
    "GELU": (nn.GELU, ("approximate",)),
    "SiLU": (nn.SiLU, ("inplace",)),
    "LeakyReLU": (nn.LeakyReLU, ("negative_slope", "inplace")),
    "Sigmoid": (nn.Sigmoid, ()),
    "ELU": (nn.ELU, ("alpha", "inplace")),
    "Identity": (nn.Identity, ()),
}


def _post_load_refresh_affine_hooks(mod: nn.Module, _incompatible: Any) -> None:
    if isinstance(mod, GMDHNorm):
        mod._refresh_affine_hooks()


def _post_load_refresh_grad_hook(mod: nn.Module, _incompatible: Any) -> None:
    if isinstance(mod, GMDHLayer):
        mod._refresh_grad_hook()


def _serialize_module_config(mod: nn.Module) -> dict[str, Any]:
    """Serialize an allowed GMDH or PyTorch module into pure-primitive metadata."""
    if isinstance(mod, GMDHLayer):
        return {
            "type": "GMDHLayer",
            "layer_number": int(mod.layer_number),
            "input_count": int(mod.input_count),
            "neuron_count": int(mod.neuron_count),
            "degree": int(mod.degree),
            "interaction": int(mod.interaction),
            "forward_inputs": bool(mod.forward_inputs),
            "min_neurons": int(mod.min_neurons),
            "max_neurons": int(mod.max_neurons),
            "init": str(mod.init) if isinstance(mod.init, str) else "zeros",
            "init_std": float(mod.init_std),
            "requires_grad": bool(mod.W.requires_grad),
        }
    if isinstance(mod, GMDHNorm):
        return {
            "type": "GMDHNorm",
            "affine": bool(mod.affine),
            "eps": float(mod.eps),
            "num_features": int(mod.out_features),
            "requires_grad": bool(mod.weight.requires_grad)
            if (mod.affine and mod.weight is not None)
            else False,
        }
    if isinstance(mod, GMDHActivation):
        return {
            "type": "GMDHActivation",
            "num_features": int(mod.out_features),
            "activation": _serialize_module_config(mod.activation),
        }
    if isinstance(mod, SoftClip):
        return {
            "type": "SoftClip",
            "c": float(mod.c.detach().item()),
            "learnable": bool(mod.learnable),
            "requires_grad": bool(mod.c.requires_grad) if mod.learnable else False,
        }
    if isinstance(mod, GMDHOutput):
        return {
            "type": "GMDHOutput",
            "count": int(mod.count),
            "dense": bool(mod.dense),
            "in_features": int(mod.linear.in_features)
            if (mod.dense and mod.linear is not None)
            else None,
            "bias": bool(mod.use_bias),
            "requires_grad": bool(mod.linear.weight.requires_grad)
            if (mod.dense and mod.linear is not None)
            else False,
        }
    if isinstance(mod, nn.Linear):
        return {
            "type": "Linear",
            "in_features": int(mod.in_features),
            "out_features": int(mod.out_features),
            "bias": mod.bias is not None,
            "requires_grad": bool(mod.weight.requires_grad),
        }
    cls_name = type(mod).__name__
    if cls_name in _SIMPLE_ACTIVATION_SPECS:
        _, attr_names = _SIMPLE_ACTIVATION_SPECS[cls_name]
        cfg: dict[str, Any] = {"type": cls_name}
        for attr in attr_names:
            val = getattr(mod, attr)
            cfg[attr] = bool(val) if isinstance(val, bool) else (float(val) if isinstance(val, (int, float)) else str(val))
        return cfg
    raise TypeError(
        f"Cannot serialize unsupported module type {cls_name!r} in GMDHModel.architecture_config()."
    )


def _deserialize_module_config(
    cfg: dict[str, Any],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Reconstruct a module skeleton from ``_serialize_module_config`` metadata."""
    mod_type = str(cfg["type"])
    if mod_type == "GMDHLayer":
        n_in = int(cfg["input_count"])
        n_neurons = int(cfg["neuron_count"])
        deg = int(cfg["degree"])
        inter = int(cfg["interaction"])
        common_kwargs: dict[str, Any] = {
            "inputs": n_in,
            "degree": deg,
            "interaction": inter,
            "forward": bool(cfg.get("forward_inputs", False)),
            "min_neurons": int(cfg.get("min_neurons", 1)),
            "max_neurons": int(cfg.get("max_neurons", 30)),
            "init": str(cfg.get("init", "kaiming")),
            "init_std": float(cfg.get("init_std", 0.01)),
            "layer_number": int(cfg.get("layer_number", 1)),
            "dtype": dtype,
            "device": device,
        }
        if n_neurons > 0:
            c_count = gmdh_coeff_count(n_in, deg, interaction=inter)
            layer = GMDHLayer(
                idx0=torch.zeros((n_neurons, n_in), dtype=torch.long, device=device),
                W=torch.zeros((n_neurons, c_count), dtype=dtype, device=device),
                lock_mask=torch.zeros(n_neurons, dtype=torch.bool, device=device),
                target_id=torch.zeros(n_neurons, dtype=torch.long, device=device),
                clone_tensors=False,
                **common_kwargs,
            )
        else:
            layer = GMDHLayer(**common_kwargs)
        layer.W.requires_grad_(bool(cfg.get("requires_grad", False)))
        layer._refresh_grad_hook()
        return layer

    if mod_type == "GMDHNorm":
        norm = GMDHNorm(
            affine=bool(cfg.get("affine", False)),
            eps=float(cfg.get("eps", 1e-12)),
            dtype=dtype,
            device=device,
        )
        n_feat = int(cfg.get("num_features", 0))
        if n_feat > 0:
            norm.set_stats(
                torch.zeros(n_feat, dtype=dtype, device=device),
                torch.ones(n_feat, dtype=dtype, device=device),
            )
        if norm.affine and norm.weight is not None and norm.bias is not None:
            req = bool(cfg.get("requires_grad", False))
            norm.weight.requires_grad_(req)
            norm.bias.requires_grad_(req)
            norm._refresh_affine_hooks()
        return norm

    if mod_type == "GMDHActivation":
        inner = _deserialize_module_config(cfg["activation"], dtype=dtype, device=device)
        n_feat = int(cfg.get("num_features", 0))
        pm = torch.zeros(n_feat, dtype=torch.bool, device=device) if n_feat > 0 else None
        return GMDHActivation(inner, passthrough_mask=pm)

    if mod_type == "SoftClip":
        sc = SoftClip(
            float(cfg.get("c", 3.0)),
            learnable=bool(cfg.get("learnable", False)),
            dtype=dtype,
            device=device,
        )
        if sc.learnable:
            sc.c.requires_grad_(bool(cfg.get("requires_grad", False)))
        return sc

    if mod_type == "GMDHOutput":
        out = GMDHOutput(
            count=int(cfg.get("count", 1)),
            dense=bool(cfg.get("dense", False)),
            in_features=cfg.get("in_features"),
            bias=bool(cfg.get("bias", True)),
            dtype=dtype,
            device=device,
        )
        if out.dense and out.linear is not None:
            req = bool(cfg.get("requires_grad", True))
            for p in out.linear.parameters():
                p.requires_grad_(req)
        return out

    if mod_type == "Linear":
        lin = nn.Linear(
            int(cfg["in_features"]),
            int(cfg["out_features"]),
            bias=bool(cfg.get("bias", True)),
            dtype=dtype,
            device=device,
        )
        req = bool(cfg.get("requires_grad", True))
        for p in lin.parameters():
            p.requires_grad_(req)
        return lin

    if mod_type in _SIMPLE_ACTIVATION_SPECS:
        cls_type, attr_names = _SIMPLE_ACTIVATION_SPECS[mod_type]
        kwargs = {attr: cfg[attr] for attr in attr_names if attr in cfg}
        return cls_type(**kwargs)

    raise ValueError(f"Unsupported serialized module type {mod_type!r} in GMDHModel.from_config().")


def _infer_model_device_dtype(
    modules: Sequence[nn.Module],
) -> tuple[torch.device | None, torch.dtype | None]:
    """Infer device/dtype from primary child modules when GMDHModel args omit them."""
    for mod in modules:
        if isinstance(mod, GMDHLayer):
            return mod.W.device, mod.W.dtype
        if isinstance(mod, GMDHOutput):
            ref = mod.linear.weight if (mod.dense and mod.linear is not None) else mod.fallback
            return ref.device, ref.dtype
        if isinstance(mod, GMDHNorm):
            return mod.mean.device, mod.mean.dtype
        if isinstance(mod, nn.Linear):
            return mod.weight.device, mod.weight.dtype
    for mod in modules:
        for tensor in (*mod.parameters(), *mod.buffers()):
            if tensor.device.type != "cpu":
                dt = tensor.dtype if tensor.is_floating_point() else None
                return tensor.device, dt
    return None, None


class GMDHModel(nn.Sequential):
    """PyTorch ``nn.Sequential`` container for GMDH models with lazy input width inference,
    between-layer ``GMDHNorm`` / ``GMDHActivation`` modules, structural pruning/trimming,
    and research instrumentation (O5).
    """

    def __init__(
        self,
        *args: nn.Module,
        input_count: int | None = None,
        output: Literal["auto", "none"] | GMDHOutput = "auto",
        dtype: str | torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        modules = list(args)
        for mod in modules:
            if not isinstance(mod, _ALLOWED_GMDH_MODULES):
                raise TypeError(
                    f"Unsupported module type {type(mod).__name__!r} in GMDHModel. "
                    "Allowed types: GMDHLayer, GMDHNorm, GMDHActivation, SoftClip, GMDHOutput, "
                    "nn.Linear, and standard PyTorch activations."
                )

        inferred_dev, inferred_dt = _infer_model_device_dtype(modules)
        dev = (
            torch.device(device)
            if device is not None
            else (inferred_dev if inferred_dev is not None else torch.device("cpu"))
        )
        eff_dtype_spec = dtype if dtype is not None else inferred_dt
        dt = resolve_dtype(eff_dtype_spec, device=dev)

        if isinstance(output, GMDHOutput):
            if not modules or modules[-1] is not output:
                modules.append(output)
        elif output == "auto":
            if not modules:
                modules.append(GMDHOutput(1, dtype=dt, device=dev))
            else:
                last = modules[-1]
                is_materialized_tail = (
                    isinstance(last, (GMDHOutput, nn.Linear))
                    or (isinstance(last, GMDHLayer) and last.is_materialized)
                )
                if not is_materialized_tail:
                    modules.append(GMDHOutput(1, dtype=dt, device=dev))
        elif output != "none":
            raise ValueError(f"Invalid `output` mode {output!r}; expected 'auto', 'none', or GMDHOutput.")

        super().__init__(*modules)
        self._input_count: int = int(input_count) if input_count is not None else 0
        self.dtype: torch.dtype = dt
        self.device: torch.device = dev
        self.acc_dtype: torch.dtype = resolve_acc_dtype(dev)
        self._empty_param = nn.Parameter(
            torch.empty(0, dtype=self.dtype, device=self.device), requires_grad=False
        )
        self.to(device=self.device, dtype=self.dtype)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> GMDHModel:
        super()._apply(fn, recurse=recurse)
        layers = self.gmdh_layers
        if layers:
            ref = layers[0].W
        elif self.has_output_module:
            out_mod = self.output_module
            ref = (
                out_mod.linear.weight
                if (out_mod.dense and out_mod.linear is not None)
                else out_mod.fallback
            )
        else:
            ref = self._empty_param
        self.device = ref.device
        self.dtype = ref.dtype
        self.acc_dtype = resolve_acc_dtype(self.device)
        return self

    @property
    def input_count(self) -> int:
        if self._input_count > 0:
            return self._input_count
        for mod in self:
            if isinstance(mod, GMDHNorm) and mod.out_features > 0:
                return mod.out_features
        return 0

    @input_count.setter
    def input_count(self, value: int) -> None:
        self._input_count = int(value)

    @property
    def gmdh_layers(self) -> list[GMDHLayer]:
        return [m for m in self if isinstance(m, GMDHLayer)]

    @property
    def layers(self) -> list[GMDHLayer]:
        return self.gmdh_layers

    @property
    def gmdh_layer_count(self) -> int:
        return len(self.gmdh_layers)

    @property
    def layer_count(self) -> int:
        return len(self.gmdh_layers)

    @property
    def has_output_module(self) -> bool:
        return any(isinstance(m, GMDHOutput) for m in self)

    @property
    def output_module(self) -> GMDHOutput:
        for mod in reversed(self):
            if isinstance(mod, GMDHOutput):
                return mod
        raise RuntimeError("GMDHModel is missing a trailing GMDHOutput module.")

    @property
    def output_index(self) -> int:
        for i in range(len(self) - 1, -1, -1):
            if isinstance(self[i], (GMDHOutput, nn.Linear)):
                return i
        return len(self)

    @property
    def selected_inputs(self) -> np.ndarray:
        """Return sorted unique 0-based input indices active in the first ``GMDHLayer``."""
        layers = self.gmdh_layers
        if not layers or not layers[0].is_materialized:
            return np.empty(0, dtype=np.int64)
        return np.unique(layers[0].idx0.detach().cpu().numpy()).astype(np.int64)

    @property
    def selected_inputs_1based(self) -> np.ndarray:
        """Return sorted unique 1-based input indices active in the first ``GMDHLayer``."""
        sel0 = self.selected_inputs
        return sel0 + 1 if sel0.size > 0 else sel0

    def forward_to_insertion_point(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate ``x`` through all modules preceding the terminal output head (or all modules if headless)."""
        h = x.unsqueeze(0) if x.ndim == 1 else x
        if self._input_count == 0 and h.shape[1] > 0:
            self._input_count = int(h.shape[1])
        out_pos = self.output_index
        for i in range(out_pos):
            h = self[i](h)
        return h

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        h = input.unsqueeze(0) if input.ndim == 1 else input
        if self._input_count == 0 and h.shape[1] > 0:
            self._input_count = int(h.shape[1])
        for module in self:
            h = module(h)
        return h

    def predict_numpy(self, samples: np.ndarray | torch.Tensor) -> np.ndarray:
        """Evaluate ``samples`` without tracking gradients and return a NumPy array."""
        was_training = self.training
        if was_training:
            self.eval()
        try:
            with torch.no_grad():
                x_t = torch.as_tensor(samples, dtype=self.dtype, device=self.device)
                out_t = self.forward(x_t)
                return out_t.detach().cpu().numpy()
        finally:
            if was_training:
                self.train()

    def prune(
        self,
        position: int,
        keep: torch.Tensor | np.ndarray | Sequence[int] | Sequence[bool],
        *,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> GMDHModel:
        """Prune the ``GMDHLayer`` at ``self[position]`` to ``keep`` (0-based indices or boolean mask),
        slice subsequent elementwise ``GMDHNorm`` / ``GMDHActivation`` modules, and remap the next
        ``GMDHLayer.idx0`` or ``GMDHOutput.index`` (§5.3).
        """
        target_mod = self[position]
        if not isinstance(target_mod, GMDHLayer):
            raise TypeError(f"Module at position {position} is {type(target_mod).__name__}, expected GMDHLayer.")

        n_neurons = target_mod.neuron_count
        dev = target_mod.W.device
        keep_t = torch.as_tensor(keep, device=dev)
        if keep_t.dtype == torch.bool:
            keep_mask = keep_t.reshape(-1)
            keep_indices = torch.nonzero(keep_mask, as_tuple=False).view(-1)
        else:
            keep_indices = torch.sort(keep_t.to(dtype=torch.long).reshape(-1)).values
            keep_mask = torch.zeros(n_neurons, dtype=torch.bool, device=dev)
            keep_mask[keep_indices] = True

        target_mod.keep_neurons(keep_indices, optimizer=optimizer)

        remap = torch.cumsum(keep_mask.to(torch.long), dim=0) - 1
        for pos in range(position + 1, len(self)):
            mod = self[pos]
            if isinstance(mod, GMDHNorm):
                mod.prune(keep_indices, optimizer=optimizer)
            elif isinstance(mod, GMDHActivation):
                mod.prune(keep_indices)
            elif isinstance(mod, GMDHLayer):
                mod.idx0 = remap[mod.idx0]
                break
            elif isinstance(mod, GMDHOutput):
                if mod.dense:
                    mod.prune(keep_indices, optimizer=optimizer)
                else:
                    mod.remap_after_prune(keep_mask)
                break
            elif isinstance(mod, nn.Linear):
                _slice_parameter_in_place(mod, "weight", keep_indices, dim=1, optimizer=optimizer)
                mod.in_features = int(keep_indices.numel())
                break

        return self

    def trim(self, *, optimizer: torch.optim.Optimizer | None = None) -> GMDHModel:
        """Prune all unused neurons backward starting from the terminal output head (or final ``GMDHLayer``)."""
        layer_positions = [i for i, m in enumerate(self) if isinstance(m, GMDHLayer) and m.is_materialized]
        if not layer_positions:
            return self

        last_layer_pos = layer_positions[-1]
        out_pos = self.output_index
        has_selector = out_pos < len(self) and isinstance(self[out_pos], GMDHOutput) and not self[out_pos].dense

        if has_selector:
            out_mod: GMDHOutput = self[out_pos]  # type: ignore[assignment]
            if not out_mod.is_initialized:
                return self
            for pos in range(out_pos - 1, last_layer_pos, -1):
                if isinstance(self[pos], (GMDHNorm, GMDHActivation)):
                    del self[pos]
            last_layer: GMDHLayer = self[last_layer_pos]  # type: ignore[assignment]
            keep_last_mask = torch.zeros(last_layer.neuron_count, dtype=torch.bool, device=last_layer.W.device)
            keep_last_mask[out_mod.index.to(device=last_layer.W.device)] = True
            if not bool(torch.all(keep_last_mask)):
                self.prune(last_layer_pos, keep_last_mask, optimizer=optimizer)

        for idx in range(len(layer_positions) - 2, -1, -1):
            cur_pos = layer_positions[idx]
            next_pos = layer_positions[idx + 1]
            cur_layer: GMDHLayer = self[cur_pos]  # type: ignore[assignment]
            next_layer: GMDHLayer = self[next_pos]  # type: ignore[assignment]
            used_mask = torch.zeros(cur_layer.neuron_count, dtype=torch.bool, device=cur_layer.W.device)
            used_mask[next_layer.idx0.reshape(-1)] = True
            if not bool(torch.all(used_mask)):
                self.prune(cur_pos, used_mask, optimizer=optimizer)

        return self

    # --- O5: Research Instrumentation ---

    def parameter_count(self, *, trainable_only: bool = True) -> int:
        """Return total number of polynomial weights (excluding locked forwarders when ``trainable_only=True``)."""
        total = 0
        for mod in self:
            if isinstance(mod, GMDHLayer) and mod.is_materialized:
                if trainable_only:
                    unlocked = int(torch.sum(~mod.lock_mask).item())
                    total += unlocked * mod.weight_count
                else:
                    total += int(mod.W.numel())
            elif isinstance(mod, GMDHNorm) and mod.affine and mod.weight is not None:
                if trainable_only and mod.passthrough_mask.numel() > 0:
                    active = int(torch.sum(~mod.passthrough_mask).item())
                    total += 2 * active
                else:
                    total += 2 * mod.out_features
        return total

    def mac_count(self) -> int:
        """Estimate multiply-accumulate operations (MACs) per sample across all materialized layers."""
        macs = 0
        for layer in self.gmdh_layers:
            if not layer.is_materialized:
                continue
            unlocked = int(torch.sum(~layer.lock_mask).item())
            if unlocked == 0:
                continue
            n_in = layer.input_count
            d = layer.degree
            q = min(n_in, max(0, layer.interaction))
            power_muls = n_in * max(0, d - 1)
            cross_muls = sum(comb_no(n_in, k) * (k - 1) for k in range(2, q + 1))
            dot_macs = layer.weight_count - 1
            macs += unlocked * (power_muls + cross_muls + dot_macs)
        return macs

    def connectivity_stats(self) -> dict[str, Any]:
        """Return structural statistics per layer (dimensions, fan-in, unlocked vs forwarders, active inputs)."""
        layers = self.gmdh_layers
        dims = [layer.neuron_count for layer in layers]
        fan_ins = [layer.input_count for layer in layers]
        degrees = [layer.degree for layer in layers]
        interactions = [layer.interaction for layer in layers]
        unlocked_counts = [int(torch.sum(~layer.lock_mask).item()) for layer in layers]
        forwarder_counts = [int(torch.sum(layer.lock_mask).item()) for layer in layers]
        return {
            "layer_count": len(layers),
            "dimensions": dims,
            "fan_ins": fan_ins,
            "degrees": degrees,
            "interactions": interactions,
            "unlocked_neurons": unlocked_counts,
            "forwarder_neurons": forwarder_counts,
            "selected_inputs_0": self.selected_inputs.tolist(),
            "selected_inputs_1": self.selected_inputs_1based.tolist(),
            "trainable_parameters": self.parameter_count(trainable_only=True),
            "total_parameters": self.parameter_count(trainable_only=False),
            "macs_per_sample": self.mac_count(),
        }

    def gradient_norms(self) -> dict[str, float]:
        """Return the L2 gradient norm of unlocked parameters for each ``GMDHLayer`` (O5 / §3.4)."""
        norms: dict[str, float] = {}
        layer_idx = 0
        for mod in self:
            if isinstance(mod, GMDHLayer) and mod.is_materialized:
                layer_idx += 1
                key = f"layer_{layer_idx}"
                if mod.W.grad is None:
                    norms[key] = 0.0
                else:
                    g = mod.W.grad
                    if mod.lock_mask.numel() == g.shape[0] and bool(torch.any(mod.lock_mask)):
                        g = g[~mod.lock_mask]
                    norms[key] = float(torch.linalg.norm(g).item()) if g.numel() > 0 else 0.0
        return norms

    def describe(
        self,
        *,
        weights: bool = False,
        print_weights: bool = False,
        input_format: str = "%02d",
        weight_format: str = "%010.3e",
        file_name: str = "",
        print_output: bool = True,
        return_str: bool = True,
    ) -> str:
        """Format and optionally print an ASCII representation of the ``GMDHModel`` structure."""
        weights = bool(weights or print_weights)
        if return_str is True and not print_output:
            pass
        layers = self.gmdh_layers
        n_layers = len(layers)
        inp_len = len(input_format % 0)
        num_len = len(weight_format % 0.0)

        qtd_neurons = [layer.neuron_count for layer in layers]
        qtd_inputs = [int(layer.idx0.shape[1]) for layer in layers]
        qtd_weights = [int(layer.W.shape[1]) for layer in layers]

        lines: list[str] = []
        header_parts: list[str] = []
        for layer_idx in range(n_layers):
            width = qtd_inputs[layer_idx] * inp_len + qtd_inputs[layer_idx] + 1
            header_parts.append(f"L{layer_idx + 1:<{width}d}")
            if weights:
                w_pad = qtd_weights[layer_idx] * num_len + qtd_weights[layer_idx] + 1
                header_parts.append(" " * w_pad)
        lines.append("".join(header_parts))

        max_n = max(qtd_neurons) if n_layers > 0 else 0
        for neuron_no in range(1, max_n + 1):
            row_parts: list[str] = []
            for layer_idx, layer in enumerate(layers):
                if neuron_no <= qtd_neurons[layer_idx]:
                    w_row = layer.W.detach().cpu().numpy()[neuron_no - 1, :]
                    i_row = layer.I[neuron_no - 1, :]
                    inp_strs = [input_format % int(x) for x in i_row]
                    row_parts.append("(" + " ".join(inp_strs) + ")")
                    if not weights:
                        row_parts.append(" ")
                    else:
                        w_strs = [weight_format % float(w) for w in w_row]
                        row_parts.append("(" + " ".join(w_strs) + ") ")
                else:
                    i_pad = qtd_inputs[layer_idx] * inp_len + qtd_inputs[layer_idx] + 2
                    row_parts.append(" " * i_pad)
                    if weights:
                        w_pad = qtd_weights[layer_idx] * num_len + qtd_weights[layer_idx] + 1
                        row_parts.append(" " * w_pad)
            lines.append("".join(row_parts))

        dim_str = " x ".join(str(q) for q in qtd_neurons)
        lines.append(f"The network has {n_layers} layer(s), with the dimension(s): {dim_str}")
        output_text = "\n".join(lines) + "\n"

        if file_name:
            with open(file_name, "w", encoding="utf-8") as fh:
                fh.write(output_text)
            if print_output:
                sys.stdout.write(f'Network structure printed in the file "{file_name}".\n')
        elif print_output:
            sys.stdout.write(output_text)

        return output_text

    def architecture_config(self) -> dict[str, Any]:
        """Return a pure-primitive architecture descriptor compatible with ``torch.load(..., weights_only=True)``."""
        return {
            "version": 1,
            "input_count": int(self._input_count),
            "has_output_module": bool(self.has_output_module),
            "modules": [_serialize_module_config(mod) for mod in self],
        }

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        dtype: str | torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> GMDHModel:
        """Reconstruct a ``GMDHModel`` skeleton from ``architecture_config()`` prior to ``load_state_dict()``."""
        dev = torch.device(device) if device is not None else torch.device("cpu")
        dt = resolve_dtype(dtype, device=dev)
        modules = [
            _deserialize_module_config(mod_cfg, dtype=dt, device=dev)
            for mod_cfg in config.get("modules", [])
        ]
        output_mode: Literal["auto", "none"] = (
            "auto" if bool(config.get("has_output_module", True)) else "none"
        )
        return cls(
            *modules,
            input_count=int(config.get("input_count", 0)),
            output=output_mode,
            dtype=dt,
            device=dev,
        )


def proximal_penalty(model: nn.Module) -> torch.Tensor:
    """Compute proximal trust-region penalty :math:`\\sum_l \\|W^{(l)} - W_{\\text{init}}^{(l)}\\|_F^2`
    over all ``GMDHLayer`` modules in ``model`` (§6).
    """
    penalty: torch.Tensor | None = None
    for mod in model.modules():
        if isinstance(mod, GMDHLayer) and mod.is_materialized:
            term = torch.sum((mod.W - mod.W_init) ** 2)
            penalty = term if penalty is None else (penalty + term)
    if penalty is None:
        if isinstance(model, GMDHModel):
            return torch.zeros((), dtype=model.dtype, device=model.device)
        ref = next(model.parameters(), None)
        if ref is None:
            ref = next(model.buffers(), None)
        if ref is not None:
            dt = ref.dtype if ref.is_floating_point() else torch.get_default_dtype()
            return torch.zeros((), dtype=dt, device=ref.device)
        return torch.zeros((), dtype=torch.get_default_dtype())
    return penalty


@dataclass(frozen=True)
class LayerContext:
    """Context passed to layer factory functions at the start of each growth cycle (§5.6, §5.7)."""

    layer_no: int
    previous_width: int
    input_width: int
    perf_history: tuple[float, ...]

    @property
    def layer_index(self) -> int:
        return self.layer_no


def combi_layers(
    *,
    max_neurons: int = 5,
    min_neurons: int = 1,
    max_inputs: int = 0,
    min_inputs: int = 1,
    max_degree: int = 0,
    interaction: int | None = None,
    forward: bool = True,
    candidates: int | Callable[[int, int], np.ndarray | torch.Tensor] | None = None,
) -> Callable[[LayerContext], GMDHLayer]:
    """Return a layer factory reproducing C-GMDH (Combi) growth rules (§5.6, §5.9).

    At layer ``l``, fan-in is ``min(l, previous_width)`` (clamped by ``min_inputs``, ``max_inputs``,
    and ``max_degree`` when > 0), ``degree`` equals fan-in (capped by ``max_degree``), and forward
    nodes are enabled by default.
    """

    def _factory(ctx: LayerContext) -> GMDHLayer:
        fan_in = min(int(ctx.layer_no), int(ctx.previous_width))
        if max_inputs > 0 and fan_in > max_inputs:
            fan_in = int(max_inputs)
        if max_degree > 0 and fan_in > max_degree:
            fan_in = int(max_degree)
        if min_inputs > 0 and fan_in < min_inputs:
            fan_in = int(min_inputs)

        deg = fan_in
        if max_degree > 0 and deg > max_degree:
            deg = int(max_degree)

        return GMDHLayer(
            layer_number=ctx.layer_no,
            inputs=fan_in,
            degree=deg,
            interaction=interaction,
            forward=forward,
            min_neurons=min_neurons,
            max_neurons=max_neurons,
            candidates=candidates,
        )

    return _factory


def default_layers(
    *,
    inputs: int = 2,
    degree: int | None = 2,
    interaction: int | None = None,
    forward: bool = False,
    min_neurons: int = 1,
    max_neurons: int = 30,
    candidates: int | Callable[[int, int], np.ndarray | torch.Tensor] | None = None,
) -> Callable[[LayerContext], GMDHLayer]:
    """Return a layer factory reproducing classic O-GMDH rules (`inputs=2, degree=2, max_neurons=30`)."""

    def _factory(ctx: LayerContext) -> GMDHLayer:
        fan_in = min(int(inputs), int(ctx.previous_width)) if ctx.previous_width > 0 else int(inputs)
        eff_deg = fan_in if degree is None else int(degree)
        return GMDHLayer(
            layer_number=ctx.layer_no,
            inputs=fan_in,
            degree=eff_deg,
            interaction=interaction,
            forward=forward,
            min_neurons=min_neurons,
            max_neurons=max_neurons,
            candidates=candidates,
        )

    return _factory
