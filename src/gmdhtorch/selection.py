"""Neuron selection criteria, strategy objects, and chunked PyTorch selection scoring for GMDH layers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from .polynomial import precompute_layer_powers_torch
from .utils import DEFAULT_CHUNK_SIZE, DEFAULT_MAX_CHUNK_BYTES, effective_batch_size

if TYPE_CHECKING:
    from .network import GMDHLayer


def gmdh_select_best(current_perfs: np.ndarray, max_neurons: int) -> np.ndarray:
    """Return boolean removal mask keeping at most ``max_neurons`` best (lowest-error) neurons.

    ``True`` indicates a neuron to be **removed**, ``False`` indicates a neuron to be **kept**.
    """
    perfs = np.asarray(current_perfs, dtype=np.float64).ravel()
    qtd_neurons = perfs.size
    keep_count = min(int(max_neurons), qtd_neurons) if max_neurons > 0 else qtd_neurons
    removing = np.ones(qtd_neurons, dtype=bool)
    if keep_count > 0:
        order = np.argsort(perfs, kind="stable")
        removing[order[:keep_count]] = False
    return removing


def gmdh_select_median(
    current_perfs: np.ndarray,
    max_neurons: int = 0,
    min_neurons: int = 0,
) -> np.ndarray:
    """Return boolean removal mask excluding neurons with error above the layer median,
    clamped between ``min_neurons`` and ``max_neurons`` kept neurons.
    """
    if max_neurons > 0 and min_neurons > max_neurons:
        raise ValueError("min_neurons must be <= max_neurons")

    perfs = np.asarray(current_perfs, dtype=np.float64).ravel()
    if perfs.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(perfs, kind="stable")
    removing = perfs > np.median(perfs)
    kept_count = int(np.sum(~removing))

    if min_neurons > 0 and kept_count < min_neurons:
        removing[order[: int(min_neurons)]] = False
    elif max_neurons > 0 and kept_count > max_neurons:
        removing[:] = True
        removing[order[: int(max_neurons)]] = False

    return removing


def _gauss_window_selection(
    perfs: np.ndarray,
    center_1based: int,
    max_neurons: int,
    min_neurons: int,
    distance: float,
    order: np.ndarray,
) -> np.ndarray:
    qtd_neurons = perfs.size
    if qtd_neurons <= min_neurons:
        return np.zeros(qtd_neurons, dtype=bool)

    idx_std = float(np.std(np.arange(1, qtd_neurons + 1, dtype=np.float64), ddof=1)) if qtd_neurons > 1 else 0.0
    s = int(math.floor(idx_std * float(distance)))
    low_boundary = max(1, center_1based - s)
    top_boundary = min(qtd_neurons, center_1based + s)

    keeps = np.zeros(qtd_neurons, dtype=bool)
    keeps[low_boundary - 1 : top_boundary] = True
    qtd_keeps = int(np.sum(keeps))

    if qtd_keeps < min_neurons:
        diff = int(min_neurons) - qtd_keeps
        ldiff = int(math.ceil(diff / 2.0))
        hdiff = diff - ldiff
        l_idx = max(1, low_boundary - ldiff)
        t_idx = max(1, low_boundary - 1)
        if l_idx <= t_idx:
            keeps[l_idx - 1 : t_idx] = True
        hdiff = hdiff + (ldiff - (t_idx - l_idx + 1))
        l_idx = min(qtd_neurons, top_boundary + 1)
        t_idx = min(qtd_neurons, top_boundary + hdiff)
        if l_idx <= t_idx:
            keeps[l_idx - 1 : t_idx] = True
    elif max_neurons > 0 and qtd_keeps > max_neurons:
        diff = qtd_keeps - int(max_neurons)
        hdiff = int(math.ceil(diff / 2.0))
        ldiff = diff - hdiff
        l_idx = low_boundary
        t_idx = low_boundary + ldiff - 1
        if l_idx <= t_idx:
            keeps[l_idx - 1 : t_idx] = False
        hdiff = hdiff + (ldiff - (t_idx - l_idx + 1))
        t_idx = top_boundary
        l_idx = t_idx - hdiff + 1
        if l_idx <= t_idx:
            keeps[l_idx - 1 : t_idx] = False

    keep_idxs = order[keeps]
    removing = np.ones(qtd_neurons, dtype=bool)
    removing[keep_idxs] = False
    return removing


def gmdh_select_gauss_median(
    current_perfs: np.ndarray,
    max_neurons: int = 0,
    min_neurons: int = 0,
    distance: float = 0.5,
) -> np.ndarray:
    """Return boolean removal mask using a Gaussian-like index window around the median rank."""
    if max_neurons > 0 and min_neurons > max_neurons:
        raise ValueError("min_neurons must be <= max_neurons")
    perfs = np.asarray(current_perfs, dtype=np.float64).ravel()
    qtd_neurons = perfs.size
    if qtd_neurons <= min_neurons:
        return np.zeros(qtd_neurons, dtype=bool)
    order = np.argsort(perfs, kind="stable")
    perfs_median_1based = int(math.ceil((qtd_neurons + 1) / 2.0))
    return _gauss_window_selection(perfs, perfs_median_1based, max_neurons, min_neurons, distance, order)


def gmdh_select_gauss_mean(
    current_perfs: np.ndarray,
    max_neurons: int = 0,
    min_neurons: int = 0,
    distance: float = 0.5,
) -> np.ndarray:
    """Return boolean removal mask using a Gaussian-like index window around the mean performance."""
    if max_neurons > 0 and min_neurons > max_neurons:
        raise ValueError("min_neurons must be <= max_neurons")
    perfs = np.asarray(current_perfs, dtype=np.float64).ravel()
    qtd_neurons = perfs.size
    if qtd_neurons <= min_neurons:
        return np.zeros(qtd_neurons, dtype=bool)
    order = np.argsort(perfs, kind="stable")
    sorted_perfs = perfs[order]
    perfs_mean_1based = int(np.argmin(np.abs(sorted_perfs - np.mean(sorted_perfs)))) + 1
    return _gauss_window_selection(perfs, perfs_mean_1based, max_neurons, min_neurons, distance, order)


def gmdh_select_threshold(
    current_perfs: np.ndarray,
    max_neurons: int = 0,
    min_neurons: int = 0,
    threshold: float = 0.0,
) -> np.ndarray:
    """Return boolean removal mask excluding neurons with error above ``threshold``,
    clamped between ``min_neurons`` and ``max_neurons`` kept neurons.
    """
    if max_neurons > 0 and min_neurons > max_neurons:
        raise ValueError("min_neurons must be <= max_neurons")

    perfs = np.asarray(current_perfs, dtype=np.float64).ravel()
    if perfs.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(perfs, kind="stable")
    removing = perfs > float(threshold)
    kept_count = int(np.sum(~removing))

    if min_neurons > 0 and kept_count < min_neurons:
        removing[order[: int(min_neurons)]] = False
    elif max_neurons > 0 and kept_count > max_neurons:
        removing[:] = True
        removing[order[: int(max_neurons)]] = False

    return removing


def gmdh_forecast_variance_selection(
    select_targets: np.ndarray,
    layer_outputs: np.ndarray,
) -> np.ndarray:
    """Return boolean *keep* mask where Forecast Variance ratio ``D < 1``."""
    t = np.asarray(select_targets, dtype=np.float64).ravel()
    out = np.asarray(layer_outputs, dtype=np.float64)
    if out.ndim == 1:
        out = out[:, None]
    ss_tot = float(np.sum(np.square(t - np.mean(t))))
    if ss_tot == 0.0:
        return np.zeros(out.shape[1], dtype=bool)
    ss_err = np.sum(np.square(t[:, None] - out), axis=0)
    d = ss_err / ss_tot
    return d < 1.0


class SelectionStrategy(ABC):
    """Base strategy object for GMDH candidate neuron selection."""

    @abstractmethod
    def select(
        self,
        perfs: np.ndarray,
        *,
        max_neurons: int = 0,
        min_neurons: int = 0,
    ) -> np.ndarray:
        """Return boolean *removal* mask (`True` = remove, `False` = keep)."""

    def keep_mask(
        self,
        perfs: np.ndarray,
        *,
        max_neurons: int = 0,
        min_neurons: int = 0,
        higher_is_better: bool = False,
    ) -> np.ndarray:
        """Return boolean *keep* mask (`True` = keep, `False` = remove)."""
        arr = np.asarray(perfs, dtype=np.float64)
        if higher_is_better and not isinstance(self, SelectRelativeThreshold):
            arr = -arr
        return ~self.select(arr, max_neurons=max_neurons, min_neurons=min_neurons)

    def __call__(
        self,
        perfs: np.ndarray,
        *,
        max_neurons: int = 0,
        min_neurons: int = 0,
    ) -> np.ndarray:
        return self.select(perfs, max_neurons=max_neurons, min_neurons=min_neurons)


@dataclass(frozen=True)
class SelectBest(SelectionStrategy):
    """Keep at most ``n`` (or the layer's ``max_neurons``) lowest-error neurons."""

    n: int | None = None

    def select(
        self,
        perfs: np.ndarray,
        *,
        max_neurons: int = 0,
        min_neurons: int = 0,
    ) -> np.ndarray:
        eff_max = int(self.n) if self.n is not None else int(max_neurons)
        if min_neurons > 0 and eff_max > 0:
            eff_max = max(eff_max, int(min_neurons))
        return gmdh_select_best(perfs, eff_max)


@dataclass(frozen=True)
class SelectMedian(SelectionStrategy):
    """Keep neurons with error at or below the median, clamped to ``[min_neurons, max_neurons]``."""

    def select(
        self,
        perfs: np.ndarray,
        *,
        max_neurons: int = 0,
        min_neurons: int = 0,
    ) -> np.ndarray:
        return gmdh_select_median(perfs, max_neurons=max_neurons, min_neurons=min_neurons)


@dataclass(frozen=True)
class SelectThreshold(SelectionStrategy):
    """Keep neurons with error at or below ``threshold``, clamped to ``[min_neurons, max_neurons]``."""

    threshold: float = 0.0

    def select(
        self,
        perfs: np.ndarray,
        *,
        max_neurons: int = 0,
        min_neurons: int = 0,
    ) -> np.ndarray:
        return gmdh_select_threshold(
            perfs,
            max_neurons=max_neurons,
            min_neurons=min_neurons,
            threshold=self.threshold,
        )


@dataclass(frozen=True)
class SelectGaussMedian(SelectionStrategy):
    """Keep neurons within a Gaussian-like rank window around the median."""

    distance: float = 0.5

    def select(
        self,
        perfs: np.ndarray,
        *,
        max_neurons: int = 0,
        min_neurons: int = 0,
    ) -> np.ndarray:
        return gmdh_select_gauss_median(
            perfs,
            max_neurons=max_neurons,
            min_neurons=min_neurons,
            distance=self.distance,
        )


@dataclass(frozen=True)
class SelectGaussMean(SelectionStrategy):
    """Keep neurons within a Gaussian-like rank window around the mean performance."""

    distance: float = 0.5

    def select(
        self,
        perfs: np.ndarray,
        *,
        max_neurons: int = 0,
        min_neurons: int = 0,
    ) -> np.ndarray:
        return gmdh_select_gauss_mean(
            perfs,
            max_neurons=max_neurons,
            min_neurons=min_neurons,
            distance=self.distance,
        )


@dataclass(frozen=True)
class SelectRelativeThreshold(SelectionStrategy):
    """Keep neurons whose importance score is at least ``min_ratio * max(scores)``,
    clamped between ``min_neurons`` and ``max_neurons`` kept neurons.
    """

    min_ratio: float = 0.05

    def select(
        self,
        perfs: np.ndarray,
        *,
        max_neurons: int = 0,
        min_neurons: int = 0,
    ) -> np.ndarray:
        if max_neurons > 0 and min_neurons > max_neurons:
            raise ValueError("min_neurons must be <= max_neurons")
        scores = np.asarray(perfs, dtype=np.float64).ravel()
        if scores.size == 0:
            return np.zeros(0, dtype=bool)
        max_s = float(np.max(scores))
        cutoff = float(self.min_ratio) * max_s
        order = np.argsort(-scores, kind="stable")
        removing = scores < cutoff
        kept_count = int(np.sum(~removing))
        if min_neurons > 0 and kept_count < min_neurons:
            removing[order[: int(min_neurons)]] = False
        elif max_neurons > 0 and kept_count > max_neurons:
            removing[:] = True
            removing[order[: int(max_neurons)]] = False
        return removing


@dataclass(frozen=True)
class SelectNone(SelectionStrategy):
    """Keep all candidate neurons without pruning."""

    def select(
        self,
        perfs: np.ndarray,
        *,
        max_neurons: int = 0,
        min_neurons: int = 0,
    ) -> np.ndarray:
        return np.zeros(np.asarray(perfs).size, dtype=bool)


def resolve_selection_strategy(
    select: SelectionStrategy | str | None,
    *,
    threshold: float = 0.0,
    distance: float = 0.5,
    min_ratio: float = 0.05,
) -> SelectionStrategy:
    """Resolve a ``SelectionStrategy`` instance or string identifier."""
    if select is None:
        return SelectMedian()
    if isinstance(select, SelectionStrategy):
        return select
    sel_str = str(select).strip().lower()
    if sel_str in ("select_median", "median"):
        return SelectMedian()
    if sel_str in ("select_best", "best"):
        return SelectBest()
    if sel_str in ("select_threshold", "threshold"):
        return SelectThreshold(threshold=threshold)
    if sel_str in ("select_relative_threshold", "relative_threshold"):
        return SelectRelativeThreshold(min_ratio=min_ratio)
    if sel_str in ("select_gauss_median", "gauss_median"):
        return SelectGaussMedian(distance=distance)
    if sel_str in ("select_gauss_mean", "gauss_mean"):
        return SelectGaussMean(distance=distance)
    if sel_str == "none":
        return SelectNone()
    raise ValueError(f'Unrecognized select method: "{select}"')


def _compute_batch_perfs_torch(
    out_chunk: torch.Tensor,
    s_targets: torch.Tensor,
    *,
    is_r2: bool,
    total_sum_of_squares: float | torch.Tensor,
    mse_div_4: bool,
) -> torch.Tensor:
    """Compute selection performance for a chunk of neuron outputs.

    - Single-output: ``out_chunk`` has shape ``(N_sel, B)``, ``s_targets`` has shape ``(N_sel,)``,
      returns ``(B,)``.
    - Multi-output: ``out_chunk`` has shape ``(N_sel, B, M)``, ``s_targets`` has shape ``(N_sel, M)``,
      returns ``(B, M)``.
    """
    if out_chunk.ndim == 3:
        targets_view = s_targets.unsqueeze(1) if s_targets.ndim == 2 else s_targets.view(-1, 1, 1)
    else:
        targets_view = s_targets.view(-1, 1)

    err_chunk = (targets_view - out_chunk).square()
    if is_r2:
        finite_mask = torch.isfinite(err_chunk)
        ss_err = torch.sum(torch.where(finite_mask, err_chunk, torch.zeros_like(err_chunk)), dim=0)
        if isinstance(total_sum_of_squares, torch.Tensor):
            tss = total_sum_of_squares.to(device=ss_err.device, dtype=ss_err.dtype)
            if ss_err.ndim == 2:
                tss = tss.view(1, -1)
            zero_tss = tss == 0.0
            safe_tss = torch.where(zero_tss, torch.ones_like(tss), tss)
            ratio = ss_err / safe_tss
            return torch.where(
                zero_tss,
                torch.where(ss_err == 0.0, torch.zeros_like(ss_err), torch.full_like(ss_err, float("inf"))),
                ratio,
            )
        if float(total_sum_of_squares) == 0.0:
            return torch.where(
                ss_err == 0.0,
                torch.zeros_like(ss_err),
                torch.full_like(ss_err, float("inf")),
            )
        return ss_err / float(total_sum_of_squares)
    div = 4.0 if mse_div_4 else 1.0
    return torch.mean(err_chunk, dim=0) / div


def _compute_layer_perfs_torch(
    layer: GMDHLayer,
    s_samples: torch.Tensor,
    s_targets: torch.Tensor,
    *,
    unlocked_count: int,
    batch_size: int | None = DEFAULT_CHUNK_SIZE,
    max_chunk_bytes: int | None = DEFAULT_MAX_CHUNK_BYTES,
    is_r2: bool = False,
    total_sum_of_squares: float = 0.0,
    mse_div_4: bool = True,
    precomputed_powers: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate selection performance for the first ``unlocked_count`` neurons of ``layer`` in chunks."""
    if unlocked_count <= 0:
        return torch.empty(0, dtype=layer.W.dtype, device=layer.W.device)

    s_in = s_samples if s_samples.ndim > 1 else s_samples.unsqueeze(1)
    n_sel = int(s_in.shape[0])
    d = int(layer.degree)
    if precomputed_powers is None or precomputed_powers.shape[2] < d:
        precomputed_powers = precompute_layer_powers_torch(s_in, d)

    b_step = effective_batch_size(
        batch_size,
        n_sel,
        layer.weight_count,
        elem_size=layer.W.element_size(),
        max_chunk_bytes=max_chunk_bytes,
    )
    perfs_t = torch.empty(unlocked_count, dtype=layer.W.dtype, device=layer.W.device)

    for start in range(0, unlocked_count, b_step):
        end = min(start + b_step, unlocked_count)
        idx_sub = layer.idx0[start:end]
        x_batch = layer.preprocess_batch(
            s_in,
            idx0_subset=idx_sub,
            precomputed_powers=precomputed_powers,
        )
        w_sub = layer.W[start:end]
        out_chunk = torch.einsum("mnc,mc->nm", x_batch, w_sub)
        perfs_t[start:end] = _compute_batch_perfs_torch(
            out_chunk,
            s_targets,
            is_r2=is_r2,
            total_sum_of_squares=total_sum_of_squares,
            mse_div_4=mse_div_4,
        )
        del x_batch, out_chunk

    return perfs_t
