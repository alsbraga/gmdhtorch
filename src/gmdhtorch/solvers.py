"""Batched and streamed-QR PyTorch linear/polynomial least-squares solvers for GMDH layers
(Truncated SVD, Column-Pivoted QR, and Tikhonov / Ridge with GCV).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
import torch

from .polynomial import precompute_layer_powers_torch
from .utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MAX_CHUNK_BYTES,
    effective_batch_size,
    resolve_acc_dtype,
    resolve_dtype,
)

if TYPE_CHECKING:
    from .network import GMDHLayer


Solver = Literal["svd", "ridge", "qr"]


def _default_rcond(
    dtype: torch.dtype,
    rcond: float | None,
    *,
    positive_only: bool = False,
) -> float:
    if rcond is None or (positive_only and rcond <= 0.0):
        return 1e-5 if dtype in (torch.float16, torch.bfloat16, torch.float32) else 1e-12
    return float(rcond)


def _equilibrate_batch_torch(
    x_batch: torch.Tensor,
    eps: float = 1e-15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Column L2-equilibration for a 3D batch of matrices ``(B, N, C)`` in PyTorch."""
    norms = torch.linalg.norm(x_batch, dim=1, keepdim=True)
    safe_norms = torch.where(norms > eps, norms, torch.ones_like(norms))
    return x_batch / safe_norms, safe_norms.squeeze(1)


def _r_svd_proj_torch(
    x_work: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ``(U^T y, S, V^T)`` for a 3D batch ``x_work`` ``(B, N, C)``, using QR pre-reduction
    (R-SVD) when ``N > 2 * C`` to avoid running batched SVD across tall sample dimensions.
    """
    _b_count, n_samples, c_count = x_work.shape
    if n_samples > 2 * c_count and c_count > 0:
        q, r = torch.linalg.qr(x_work, mode="reduced")
        qty = torch.einsum("bnc,n->bc", q, y)
        del q
        u_r, s, vt = torch.linalg.svd(r, full_matrices=False)
        del r
        ut_y = torch.einsum("bck,bc->bk", u_r, qty)
        del u_r, qty
        return ut_y, s, vt

    u, s, vt = torch.linalg.svd(x_work, full_matrices=False)
    ut_y = torch.einsum("bnk,n->bk", u, y)
    del u
    return ut_y, s, vt


def _solve_qr_via_triangular(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    cond: float,
    svd_fallback: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Solve batched ``a @ x = b`` via QR + triangular back-substitution for devices
    (such as MPS) where ``torch.linalg.lstsq`` is unsupported.

    Falls back to ``svd_fallback()`` when ``a`` is underdetermined or numerically rank-deficient.
    """
    _b_count, n_rows, n_cols = a.shape
    if n_rows < n_cols or n_cols == 0:
        return svd_fallback()
    q, r = torch.linalg.qr(a, mode="reduced")
    qt_b = torch.einsum("bnk,bnm->bkm", q, b)
    diag = torch.diagonal(r, dim1=-2, dim2=-1).abs()
    max_diag = torch.amax(diag, dim=-1, keepdim=True)
    min_diag = torch.amin(diag, dim=-1, keepdim=True)
    # Require finite nonzero R diagonals above relative rcond before triangular solve.
    well_conditioned = bool(
        torch.all(torch.isfinite(diag))
        and torch.all(max_diag > 0)
        and torch.all(min_diag > cond * max_diag)
    )
    if not well_conditioned:
        return svd_fallback()
    sol = torch.linalg.solve_triangular(r, qt_b, upper=True)
    if not bool(torch.all(torch.isfinite(sol))):
        return svd_fallback()
    return sol


def _lstsq_qr_dispatch(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    cond: float,
    svd_fallback: Callable[[], torch.Tensor],
) -> torch.Tensor:
    """Dispatch batched QR/least-squares solves across CPU, CUDA, and MPS."""
    if a.device.type == "cpu":
        # Use 'gelsd' instead of 'gelsy' on CPU to avoid non-deterministic LAPACK dgelsy rank drops
        return torch.linalg.lstsq(a, b, rcond=cond, driver="gelsd").solution
    if a.device.type == "mps":
        return _solve_qr_via_triangular(a, b, cond=cond, svd_fallback=svd_fallback)
    return torch.linalg.lstsq(a, b, rcond=cond).solution


def _solve_from_qr_factors_torch(
    r_batch: torch.Tensor,
    qty_batch: torch.Tensor,
    *,
    y_norm_sq: torch.Tensor | float,
    n_samples: int,
    solver_low: str = "svd",
    rcond: float | None = 1e-12,
    alpha: float | str = 1e-6,
    equilibrate: bool = False,
    penalize_bias: bool = False,
) -> torch.Tensor:
    """Solve for candidate weights from reduced QR factors ``r_batch`` ``(B, K, C)``
    and ``qty_batch`` ``(B, K, M)`` in ``float64``.

    Returns ``w_batch`` of shape ``(B, M, C)`` (or ``(B, C)`` if ``qty_batch`` is 2D).
    """
    squeeze_m = qty_batch.ndim == 2
    qty = qty_batch.unsqueeze(-1) if squeeze_m else qty_batch
    b_count, k_dim, c_count = r_batch.shape
    m_targets = int(qty.shape[2])

    if equilibrate:
        r_work, col_scales = _equilibrate_batch_torch(r_batch)
        if solver_low in ("ridge", "tikhonov") and not penalize_bias and c_count > 0:
            col_scales[:, 0] = 1.0
            r_work[:, :, 0] = r_batch[:, :, 0]
    else:
        r_work = r_batch
        col_scales = None

    if solver_low in ("svd", "truncated_svd", "lstsq", "pinv"):
        u_r, s, vt = torch.linalg.svd(r_work, full_matrices=False)
        ut_y = torch.einsum("bjk,bjm->bkm", u_r, qty)
        eff_rcond = _default_rcond(r_work.dtype, rcond, positive_only=True)
        s_max = torch.amax(s, dim=1, keepdim=True)
        cutoff = eff_rcond * s_max
        mask = s > cutoff
        inv_s = torch.where(
            mask,
            1.0 / torch.where(mask, s, torch.ones_like(s)),
            torch.zeros_like(s),
        )
        scaled_ut_y = ut_y * inv_s.unsqueeze(-1)
        w_batch = torch.einsum("bkc,bkm->bmc", vt, scaled_ut_y)

    elif solver_low in ("qr", "qr_pivoted"):
        cond = _default_rcond(r_work.dtype, rcond)
        sol = _lstsq_qr_dispatch(
            r_work,
            qty,
            cond=cond,
            svd_fallback=lambda: _solve_from_qr_factors_torch(
                r_work,
                qty,
                y_norm_sq=y_norm_sq,
                n_samples=n_samples,
                solver_low="svd",
                rcond=rcond,
                equilibrate=False,
            )
            .transpose(1, 2)
            .contiguous(),
        )
        w_batch = sol.transpose(1, 2).contiguous()

    elif solver_low in ("ridge", "tikhonov"):
        if alpha == "gcv":
            lambdas = torch.logspace(-12, 2, 29, dtype=r_work.dtype, device=r_work.device)
            lam = lambdas.view(1, 1, 1, -1)  # (1, 1, 1, L)
            y_nsq = (
                y_norm_sq.to(dtype=r_work.dtype, device=r_work.device).view(1, m_targets, 1)
                if isinstance(y_norm_sq, torch.Tensor)
                else torch.full((1, m_targets, 1), float(y_norm_sq), dtype=r_work.dtype, device=r_work.device)
            )
            if not penalize_bias and c_count > 1 and k_dim > 1:
                r00 = r_work[:, 0, 0]  # (B,)
                r01 = r_work[:, 0, 1:]  # (B, C-1)
                r11 = r_work[:, 1:, 1:]  # (B, K-1, C-1)
                qty0 = qty[:, 0, :]  # (B, M)
                qty1 = qty[:, 1:, :]  # (B, K-1, M)

                u_r, s, vt = torch.linalg.svd(r11, full_matrices=False)
                ut_y = torch.einsum("bjk,bjm->bkm", u_r, qty1)  # (B, K-1, M)
                y_centered_nsq = torch.clamp(y_nsq.squeeze(-1) - qty0 * qty0, min=0.0).unsqueeze(-1)  # (B, M, 1)
                proj_norm_sq = torch.sum(ut_y * ut_y, dim=1).unsqueeze(-1)  # (B, M, 1)
                perp_err_sq = torch.clamp(y_centered_nsq - proj_norm_sq, min=0.0)  # (B, M, 1)

                s2 = (s * s).unsqueeze(2).unsqueeze(3)  # (B, K-1, 1, 1)
                shrink = lam / (s2 + lam)  # (B, K-1, 1, L)
                uty_sq = (ut_y * ut_y).unsqueeze(-1)  # (B, K-1, M, 1)
                rss_grid = perp_err_sq + torch.sum((shrink * shrink) * uty_sq, dim=1)  # (B, M, L)
                eff_df = 1.0 + torch.sum(s2 / (s2 + lam), dim=1)  # (B, 1, L)
                denom = torch.clamp(float(n_samples) - eff_df, min=1.0) ** 2
                gcv_scores = rss_grid / denom  # (B, M, L)
                best_lam_idx = torch.argmin(gcv_scores, dim=2)  # (B, M)
                best_lam = lambdas[best_lam_idx].unsqueeze(1)  # (B, 1, M)

                s_exp = s.unsqueeze(-1)  # (B, K-1, 1)
                filter_factors = s_exp / (s_exp * s_exp + best_lam)  # (B, K-1, M)
                w_nonbias = torch.einsum("bkc,bkm->bmc", vt, ut_y * filter_factors)  # (B, M, C-1)
                safe_r00 = torch.where(
                    torch.abs(r00) > 1e-15, r00, torch.ones_like(r00)
                ).unsqueeze(-1)  # (B, 1)
                w_bias = (qty0 - torch.einsum("bc,bmc->bm", r01, w_nonbias)) / safe_r00  # (B, M)
                w_batch = torch.cat([w_bias.unsqueeze(-1), w_nonbias], dim=2)
            else:
                u_r, s, vt = torch.linalg.svd(r_work, full_matrices=False)
                ut_y = torch.einsum("bjk,bjm->bkm", u_r, qty)
                proj_norm_sq = torch.sum(ut_y * ut_y, dim=1).unsqueeze(-1)
                perp_err_sq = torch.clamp(y_nsq - proj_norm_sq, min=0.0)
                s2 = (s * s).unsqueeze(2).unsqueeze(3)
                shrink = lam / (s2 + lam)
                uty_sq = (ut_y * ut_y).unsqueeze(-1)
                rss_grid = perp_err_sq + torch.sum((shrink * shrink) * uty_sq, dim=1)
                eff_df = torch.sum(s2 / (s2 + lam), dim=1)
                denom = torch.clamp(float(n_samples) - eff_df, min=1.0) ** 2
                gcv_scores = rss_grid / denom
                best_lam = lambdas[torch.argmin(gcv_scores, dim=2)].unsqueeze(1)
                s_exp = s.unsqueeze(-1)
                filter_factors = s_exp / (s_exp * s_exp + best_lam)
                w_batch = torch.einsum("bkc,bkm->bmc", vt, ut_y * filter_factors)
        else:
            lam_val = float(alpha)
            xx = torch.einsum("bki,bkj->bij", r_work, r_work)
            xy = torch.einsum("bki,bkm->bim", r_work, qty)
            reg_diag = torch.full((c_count,), lam_val, dtype=r_work.dtype, device=r_work.device)
            if not penalize_bias and c_count > 0:
                reg_diag[0] = 0.0
            xx = xx + torch.diag(reg_diag).unsqueeze(0)
            sol, info = torch.linalg.solve_ex(xx, xy)
            if bool(torch.any(info != 0)):
                w_svd = _solve_from_qr_factors_torch(
                    r_work,
                    qty,
                    y_norm_sq=y_norm_sq,
                    n_samples=n_samples,
                    solver_low="svd",
                    rcond=_default_rcond(r_work.dtype, None),
                    equilibrate=False,
                )
                w_batch = w_svd if w_svd.ndim == 3 else w_svd.unsqueeze(1)
            else:
                w_batch = sol.transpose(1, 2).contiguous()
    else:
        raise ValueError(
            f"Unknown solver '{solver_low}'. Supported solvers: 'svd', 'qr', 'ridge'."
        )

    if col_scales is not None:
        w_batch = w_batch / col_scales.unsqueeze(1)

    return w_batch.squeeze(1) if squeeze_m else w_batch


def compute_closed_form_norm_stats(
    r_batch: torch.Tensor,
    x_sum_batch: torch.Tensor,
    w_batch: torch.Tensor,
    n_samples: int,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact training mean and standard deviation of fitted neuron outputs ``z = X @ w``
    directly from the QR factor ``r_batch`` and column sums ``x_sum_batch`` without an extra data pass.

    - ``r_batch``: ``(B, K, C)``
    - ``x_sum_batch``: ``(B, C)`` (sum of rows of ``X``)
    - ``w_batch``: ``(B, C)`` or ``(B, M, C)``
    Returns ``(mean, std)`` matching the batch/target dimensions of ``w_batch``.
    """
    n = float(max(1, int(n_samples)))
    x_mean = x_sum_batch / n  # (B, C)
    if w_batch.ndim == 2:
        mean = torch.einsum("bc,bc->b", x_mean, w_batch)
        rw = torch.einsum("bkc,bc->bk", r_batch, w_batch)
        second_moment = torch.sum(rw * rw, dim=1) / n
    else:
        mean = torch.einsum("bc,bmc->bm", x_mean, w_batch)
        rw = torch.einsum("bkc,bmc->bkm", r_batch, w_batch)
        second_moment = torch.sum(rw * rw, dim=1) / n

    var = torch.clamp(second_moment - mean * mean, min=0.0)
    std = torch.sqrt(torch.where(var > eps * eps, var, torch.ones_like(var)))
    return mean, std


class StreamedQRAccumulator:
    """Streamed QR accumulator over candidate design matrices for a GMDH layer.

    Maintains upper-triangular ``R`` ``(B, C, C)``, ``Q^T Y`` ``(B, C, M)``, and column sums
    ``x_sum`` ``(B, C)`` per candidate chunk, updated incrementally across mini-batches.
    """

    def __init__(
        self,
        candidate_count: int,
        coeff_count: int,
        target_count: int = 1,
        *,
        device: torch.device | str = "cpu",
        dtype: str | torch.dtype | None = None,
    ) -> None:
        self.candidate_count = int(candidate_count)
        self.coeff_count = int(coeff_count)
        self.target_count = int(target_count)
        self.device = torch.device(device)
        self.dtype: torch.dtype = (
            resolve_acc_dtype(self.device)
            if dtype is None
            else resolve_dtype(dtype, device=self.device)
        )

        self.r_factors: torch.Tensor | None = None
        self.qty_factors: torch.Tensor | None = None
        self.x_sum: torch.Tensor | None = None
        self.y_sum = torch.zeros(self.target_count, dtype=self.dtype, device=self.device)
        self.y_sq_sum = torch.zeros(self.target_count, dtype=self.dtype, device=self.device)
        self.n_samples: int = 0

    def update_targets(self, y_batch: torch.Tensor) -> torch.Tensor:
        """Record batch targets ``y_batch`` ``(N_b,)`` or ``(N_b, M)`` in ``self.dtype``."""
        y_acc = y_batch.to(dtype=self.dtype, device=self.device)
        if y_acc.ndim == 1:
            y_acc = y_acc.unsqueeze(1)
        self.y_sum = self.y_sum + torch.sum(y_acc, dim=0)
        self.y_sq_sum = self.y_sq_sum + torch.sum(y_acc * y_acc, dim=0)
        self.n_samples += int(y_acc.shape[0])
        return y_acc

    def update_chunk(
        self,
        start: int,
        end: int,
        x_batch: torch.Tensor,
        y_acc: torch.Tensor,
    ) -> None:
        """Update QR factors for candidate slice ``[start:end]`` with ``x_batch`` ``(B_sub, N_b, C)``."""
        x_acc = x_batch.to(dtype=self.dtype, device=self.device)
        b_sub, n_b, c_count = x_acc.shape

        if self.r_factors is None:
            self.r_factors = torch.zeros(
                (self.candidate_count, c_count, c_count),
                dtype=self.dtype,
                device=self.device,
            )
            self.qty_factors = torch.zeros(
                (self.candidate_count, c_count, self.target_count),
                dtype=self.dtype,
                device=self.device,
            )
            self.x_sum = torch.zeros(
                (self.candidate_count, c_count),
                dtype=self.dtype,
                device=self.device,
            )

        assert self.r_factors is not None and self.qty_factors is not None and self.x_sum is not None
        self.x_sum[start:end] = self.x_sum[start:end] + torch.sum(x_acc, dim=1)

        r_prev = self.r_factors[start:end]
        qty_prev = self.qty_factors[start:end]

        if self.n_samples == n_b:
            q, r = torch.linalg.qr(x_acc, mode="reduced")
            qty = torch.einsum("bnc,nm->bcm", q, y_acc)
            k = int(r.shape[1])
            self.r_factors[start:end, :k, :] = r
            self.qty_factors[start:end, :k, :] = qty
        else:
            stacked_x = torch.cat([r_prev, x_acc], dim=1)
            y_exp = y_acc.unsqueeze(0).expand(b_sub, -1, -1)
            stacked_y = torch.cat([qty_prev, y_exp], dim=1)
            q, r = torch.linalg.qr(stacked_x, mode="reduced")
            qty = torch.einsum("bnk,bnm->bkm", q, stacked_y)
            k = int(r.shape[1])
            self.r_factors[start:end, :k, :] = r
            self.qty_factors[start:end, :k, :] = qty

    def solve(
        self,
        *,
        solver: str = "svd",
        rcond: float | None = None,
        alpha: float | str = 1e-6,
        equilibrate: bool = False,
        chunk_size: int | None = DEFAULT_CHUNK_SIZE,
        max_chunk_bytes: int | None = DEFAULT_MAX_CHUNK_BYTES,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve for all candidate weights and closed-form norm statistics `(W, mean, std)`."""
        if self.r_factors is None or self.qty_factors is None or self.x_sum is None or self.n_samples == 0:
            raise RuntimeError("No observations accumulated in StreamedQRAccumulator before solve().")

        solver_low = str(solver).strip().lower()
        k_eff = min(self.n_samples, self.coeff_count)
        w_all = torch.empty(
            (self.candidate_count, self.target_count, self.coeff_count),
            dtype=self.dtype,
            device=self.device,
        )
        mean_all = torch.empty(
            (self.candidate_count, self.target_count),
            dtype=self.dtype,
            device=self.device,
        )
        std_all = torch.empty(
            (self.candidate_count, self.target_count),
            dtype=self.dtype,
            device=self.device,
        )

        step = effective_batch_size(
            chunk_size,
            max(1, k_eff),
            max(1, self.coeff_count),
            elem_size=self.r_factors.element_size(),
            max_chunk_bytes=max_chunk_bytes,
        )
        for start in range(0, self.candidate_count, step):
            end = min(start + step, self.candidate_count)
            r_sub = self.r_factors[start:end, :k_eff, :]
            qty_sub = self.qty_factors[start:end, :k_eff, :]
            x_sum_sub = self.x_sum[start:end]
            w_sub = _solve_from_qr_factors_torch(
                r_sub,
                qty_sub,
                y_norm_sq=self.y_sq_sum,
                n_samples=self.n_samples,
                solver_low=solver_low,
                rcond=rcond,
                alpha=alpha,
                equilibrate=equilibrate,
            )
            m_sub, s_sub = compute_closed_form_norm_stats(
                r_sub, x_sum_sub, w_sub, self.n_samples
            )
            w_all[start:end] = w_sub
            mean_all[start:end] = m_sub
            std_all[start:end] = s_sub

        return w_all, mean_all, std_all


def _solve_svd_batch_torch(
    x_batch: torch.Tensor,
    y: torch.Tensor,
    rcond: float | None = 1e-12,
    equilibrate: bool = False,
) -> torch.Tensor:
    """Solve ``x_batch @ w = y`` in PyTorch using batched economy SVD (R-SVD when ``N > 2*C``) with ``rcond`` cutoff."""
    if equilibrate:
        x_work, col_scales = _equilibrate_batch_torch(x_batch)
    else:
        x_work = x_batch
        col_scales = None

    ut_y, s, vt = _r_svd_proj_torch(x_work, y)

    eff_rcond = _default_rcond(x_work.dtype, rcond, positive_only=True)
    s_max = torch.amax(s, dim=1, keepdim=True)
    cutoff = eff_rcond * s_max
    mask = s > cutoff
    inv_s = torch.where(
        mask,
        1.0 / torch.where(mask, s, torch.ones_like(s)),
        torch.zeros_like(s),
    )

    scaled_ut_y = ut_y * inv_s
    w_batch = torch.einsum("bkc,bk->bc", vt, scaled_ut_y)

    if col_scales is not None:
        w_batch = w_batch / col_scales

    return w_batch


def _solve_ridge_batch_torch(
    x_batch: torch.Tensor,
    y: torch.Tensor,
    alpha: float | str = 1e-6,
    equilibrate: bool = False,
    penalize_bias: bool = False,
) -> torch.Tensor:
    """Solve Tikhonov / Ridge regularized least squares in PyTorch for a batch of ``B`` neurons."""
    _b_count, n_samples, c_count = x_batch.shape
    if equilibrate:
        x_work, col_scales = _equilibrate_batch_torch(x_batch)
        if not penalize_bias:
            col_scales[:, 0] = 1.0
            x_work[:, :, 0] = x_batch[:, :, 0]
    else:
        x_work = x_batch
        col_scales = None

    if alpha == "gcv":
        lambdas = torch.logspace(-12, 2, 29, dtype=x_batch.dtype, device=x_batch.device)
        lam = lambdas.view(1, 1, -1)
        if not penalize_bias and c_count > 1:
            y_mean = torch.mean(y)
            y_centered = y - y_mean
            x_nonbias = x_work[:, :, 1:]
            x_means = torch.mean(x_nonbias, dim=1, keepdim=True)
            x_centered = x_nonbias - x_means
            ut_y, s, vt = _r_svd_proj_torch(x_centered, y_centered)
            y_norm_sq = torch.dot(y_centered, y_centered)
            proj_norm_sq = torch.sum(ut_y * ut_y, dim=1, keepdim=True)
            perp_err_sq = torch.clamp(y_norm_sq - proj_norm_sq, min=0.0)

            s2 = (s * s).unsqueeze(-1)
            shrink = lam / (s2 + lam)
            rss_grid = perp_err_sq + torch.sum((shrink * shrink) * (ut_y * ut_y).unsqueeze(-1), dim=1)
            eff_df = 1.0 + torch.sum(s2 / (s2 + lam), dim=1)
            denom = torch.clamp(float(n_samples) - eff_df, min=1.0) ** 2
            gcv_scores = rss_grid / denom
            best_lam_idx = torch.argmin(gcv_scores, dim=1)
            best_lam = lambdas[best_lam_idx].unsqueeze(-1)

            filter_factors = s / (s * s + best_lam)
            w_nonbias = torch.einsum("bkc,bk->bc", vt, ut_y * filter_factors)
            w_bias = y_mean - torch.einsum("bc,bc->b", x_means[:, 0, :], w_nonbias)
            w_batch = torch.cat([w_bias.unsqueeze(-1), w_nonbias], dim=1)
        else:
            ut_y, s, vt = _r_svd_proj_torch(x_work, y)
            s2 = (s * s).unsqueeze(-1)
            y_norm_sq = torch.dot(y, y)
            perp_err_sq = torch.clamp(y_norm_sq - torch.sum(ut_y * ut_y, dim=1, keepdim=True), min=0.0)
            shrink = lam / (s2 + lam)
            rss_grid = perp_err_sq + torch.sum((shrink * shrink) * (ut_y * ut_y).unsqueeze(-1), dim=1)
            eff_df = torch.sum(s2 / (s2 + lam), dim=1)
            denom = torch.clamp(float(n_samples) - eff_df, min=1.0) ** 2
            gcv_scores = rss_grid / denom
            best_lam = lambdas[torch.argmin(gcv_scores, dim=1)].unsqueeze(-1)
            filter_factors = s / (s * s + best_lam)
            w_batch = torch.einsum("bkc,bk->bc", vt, ut_y * filter_factors)
    else:
        lam_val = float(alpha)
        xx = torch.einsum("bni,bnj->bij", x_work, x_work)
        xy = torch.einsum("bni,n->bi", x_work, y)
        reg_diag = torch.full((c_count,), lam_val, dtype=x_batch.dtype, device=x_batch.device)
        if not penalize_bias and c_count > 0:
            reg_diag[0] = 0.0
        xx = xx + torch.diag(reg_diag).unsqueeze(0)
        sol, info = torch.linalg.solve_ex(xx, xy.unsqueeze(-1))
        if bool(torch.any(info != 0)):
            w_batch = _solve_svd_batch_torch(
                x_work, y, rcond=_default_rcond(x_work.dtype, None), equilibrate=False
            )
        else:
            w_batch = sol.squeeze(-1)

    if col_scales is not None:
        w_batch = w_batch / col_scales
    return w_batch


def _solve_qr_batch_torch(
    x_batch: torch.Tensor,
    y: torch.Tensor,
    rcond: float | None = 1e-12,
    equilibrate: bool = False,
) -> torch.Tensor:
    """Solve ``x_batch @ w = y`` (`gelsd` on CPU, `gels` on CUDA, QR + triangular solve on MPS)."""
    if equilibrate:
        x_work, col_scales = _equilibrate_batch_torch(x_batch)
    else:
        x_work = x_batch
        col_scales = None

    b_count = int(x_work.shape[0])
    cond = _default_rcond(x_work.dtype, rcond)
    y_rhs = y.view(1, -1, 1).expand(b_count, -1, 1)
    sol = _lstsq_qr_dispatch(
        x_work,
        y_rhs,
        cond=cond,
        svd_fallback=lambda: _solve_svd_batch_torch(x_work, y, rcond=rcond, equilibrate=False).unsqueeze(-1),
    )
    w_batch = sol.squeeze(-1)

    if col_scales is not None:
        w_batch = w_batch / col_scales
    return w_batch


def _solve_batch_torch(
    x_batch: torch.Tensor,
    y: torch.Tensor,
    *,
    solver_low: str,
    rcond: float | None = None,
    alpha: float | str = 1e-6,
    equilibrate: bool = False,
) -> torch.Tensor:
    """Dispatch a single chunk ``x_batch`` ``(B, N, C)`` to the requested stabilized PyTorch solver."""
    if solver_low in ("svd", "truncated_svd", "lstsq", "pinv"):
        eff_rcond = _default_rcond(x_batch.dtype, rcond)
        return _solve_svd_batch_torch(x_batch, y, rcond=eff_rcond, equilibrate=equilibrate)
    if solver_low in ("ridge", "tikhonov"):
        return _solve_ridge_batch_torch(x_batch, y, alpha=alpha, equilibrate=equilibrate)
    if solver_low in ("qr", "qr_pivoted"):
        eff_rcond = _default_rcond(x_batch.dtype, rcond)
        return _solve_qr_batch_torch(x_batch, y, rcond=eff_rcond, equilibrate=equilibrate)
    raise ValueError(
        f"Unknown solver '{solver_low}'. Supported solvers: 'svd', 'qr', 'ridge'."
    )


def fit_layer_neurons(
    layer: GMDHLayer,
    samples: torch.Tensor,
    targets: torch.Tensor,
    *,
    solver: str = "svd",
    rcond: float | None = None,
    alpha: float | str = 1e-6,
    equilibrate: bool = False,
    batch_size: int | None = DEFAULT_CHUNK_SIZE,
    max_chunk_bytes: int | None = DEFAULT_MAX_CHUNK_BYTES,
) -> GMDHLayer:
    """Fit all unlocked neurons in ``layer`` using batched PyTorch linear algebra."""
    dev = layer.W.device

    with torch.no_grad():
        s = samples.to(device=dev, non_blocking=samples.is_pinned())
        if s.ndim == 1:
            s = s.unsqueeze(1)
        y = targets.to(device=dev, non_blocking=targets.is_pinned()).reshape(-1)

        unlocked_rows = torch.nonzero(~layer.lock_mask, as_tuple=False).view(-1)
        if unlocked_rows.numel() == 0:
            return layer

        precomputed_powers = precompute_layer_powers_torch(s, layer.degree)
        b_step = effective_batch_size(
            batch_size,
            int(s.shape[0]),
            layer.weight_count,
            elem_size=layer.W.element_size(),
            max_chunk_bytes=max_chunk_bytes,
        )
        solver_low = str(solver).strip().lower()

        for start in range(0, int(unlocked_rows.numel()), b_step):
            batch_rows = unlocked_rows[start : start + b_step]
            idx_sub = layer.idx0[batch_rows]
            x_batch = layer.preprocess_batch(
                s,
                idx0_subset=idx_sub,
                precomputed_powers=precomputed_powers,
            )
            w_batch = _solve_batch_torch(
                x_batch,
                y,
                solver_low=solver_low,
                rcond=rcond,
                alpha=alpha,
                equilibrate=equilibrate,
            )
            layer.W.data[batch_rows, :] = w_batch
            del x_batch, w_batch

        layer._W_init = None

    return layer
