"""PyTorch device, dtype, batch-transfer, and memory-chunking utilities for GMDH."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch

DEFAULT_CHUNK_SIZE: int = 8192
DEFAULT_MAX_CHUNK_BYTES: int = 512 * 1024 * 1024  # 512 MiB target ceiling per 3D design tensor chunk
DEFAULT_MAX_OBSERVE_BYTES: int = 64 * 1024 * 1024  # 64 MiB threshold before spilling to StreamedQRAccumulator


def resolve_acc_dtype(
    device: torch.device | str,
    requested: torch.dtype = torch.float64,
) -> torch.dtype:
    """Return high-precision accumulation dtype supported on ``device``."""
    dev = torch.device(device)
    if dev.type == "mps" and requested == torch.float64:
        return torch.float32
    return requested


def resolve_dtype(
    dtype: str | torch.dtype | np.dtype[Any] | None,
    *,
    device: torch.device | str | None = None,
) -> torch.dtype:
    """Resolve a module/solver dtype once, defaulting to ``float32`` on MPS and ``float64`` elsewhere."""
    dev = torch.device(device) if device is not None else None
    if dtype is None:
        return torch.float32 if (dev is not None and dev.type == "mps") else torch.float64

    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        s = str(dtype).strip().lower()
        if "bfloat16" in s:
            resolved = torch.bfloat16
        elif "16" in s or "half" in s:
            resolved = torch.float16
        elif "32" in s or s == "float":
            resolved = torch.float32
        else:
            resolved = torch.float64

    if dev is not None and dev.type == "mps" and resolved == torch.float64:
        raise ValueError(
            "MPS does not support torch.float64; use dtype=torch.float32 or device='cpu'."
        )
    return resolved


def effective_batch_size(
    requested_batch_size: int | None,
    n_samples: int,
    coeff_count: int,
    elem_size: int = 8,
    max_chunk_bytes: int | None = DEFAULT_MAX_CHUNK_BYTES,
) -> int:
    """Bound neuron chunk size so ``(b_step, n_samples, coeff_count)`` stays within ``max_chunk_bytes``.

    When ``max_chunk_bytes`` is ``None`` (or ``<= 0``), no byte ceiling is applied and
    ``requested_batch_size`` governs chunking directly. When ``requested_batch_size`` is
    ``None`` (or ``<= 0``), only ``max_chunk_bytes`` bounds the chunk size.
    """
    req = (
        int(requested_batch_size)
        if (requested_batch_size is not None and int(requested_batch_size) > 0)
        else None
    )
    if max_chunk_bytes is None or int(max_chunk_bytes) <= 0:
        return max(1, req) if req is not None else sys.maxsize

    bytes_per_neuron = max(1, int(n_samples) * max(1, int(coeff_count)) * max(1, int(elem_size)))
    max_b = max(16, int(max_chunk_bytes) // bytes_per_neuron)
    if req is None:
        return max_b
    return max(1, min(req, max_b))
