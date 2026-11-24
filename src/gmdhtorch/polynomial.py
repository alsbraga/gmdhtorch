"""Polynomial design matrix construction, combinatorics, and vectorized PyTorch batch preprocessing."""

from __future__ import annotations

import functools
import heapq
import itertools
import math
from typing import Callable, Sequence

import numpy as np
import torch


def comb_no(n: int, m: int) -> int:
    """Return number of combinations C(n, m) using Python's standard math.comb."""
    if m < 0 or m > n:
        return 0
    return math.comb(int(n), int(m))


@functools.lru_cache(maxsize=256)
def _cached_combinations(n: int, m: int) -> np.ndarray:
    """Return all m-combinations of (0, ..., n-1) in lexicographical order, cached."""
    if m <= 0 or m > n:
        return np.empty((0, max(0, m)), dtype=np.int64)
    rows = comb_no(n, m)
    flat = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(n), m)),
        dtype=np.int64,
        count=rows * m,
    )
    return flat.reshape(rows, m)


def _ranked_candidate_indices(
    n_prev: int,
    n_in: int,
    k: int,
    input_scores: np.ndarray,
    *,
    higher_is_better: bool = False,
) -> np.ndarray:
    """Lazily generate up to ``k`` combinations prioritizing top-ranked inputs in ``O(k * n_in * log k)``."""
    scores = np.asarray(input_scores, dtype=np.float64).reshape(-1)
    if scores.size != n_prev:
        raise ValueError(f"Expected {n_prev} input_scores, got {scores.size}.")
    ranked = np.argsort(-scores if higher_is_better else scores, kind="mergesort")

    seen: set[tuple[int, ...]] = set()
    rows: list[tuple[int, ...]] = []

    def _add_rank_tuple(rt: tuple[int, ...]) -> None:
        if rt not in seen and len(rows) < k:
            seen.add(rt)
            rows.append(rt)

    # 1. Guarantee coverage of top-ranked inputs within budget k
    base = tuple(range(n_in - 1))
    _add_rank_tuple(tuple(range(n_in)))
    for r in range(n_in, min(n_prev, k + n_in - 1)):
        if len(rows) >= k:
            break
        _add_rank_tuple(base + (r,))

    # 2. Fill remaining budget in increasing rank-sum order via min-heap
    if len(rows) < k:
        init_t = tuple(range(n_in))
        heap: list[tuple[int, tuple[int, ...]]] = [(sum(init_t), init_t)]
        visited: set[tuple[int, ...]] = {init_t}
        while heap and len(rows) < k:
            _, cur = heapq.heappop(heap)
            _add_rank_tuple(cur)
            if len(rows) >= k:
                break
            for j in range(n_in):
                nxt_val = cur[j] + 1
                limit = cur[j + 1] if j + 1 < n_in else n_prev
                if nxt_val < limit:
                    nxt = cur[:j] + (nxt_val,) + cur[j + 1 :]
                    if nxt not in visited:
                        visited.add(nxt)
                        heapq.heappush(heap, (sum(nxt), nxt))

    # Map rank-space indices back to original 0-based input indices
    mapped = np.asarray([sorted(int(ranked[idx]) for idx in rt) for rt in rows], dtype=np.int64)
    return mapped


def resolve_candidate_indices(
    previous_count: int,
    input_count: int,
    candidates: int | Callable[[int, int], np.ndarray | torch.Tensor] | None = None,
    *,
    max_candidates: int | None = None,
    input_scores: np.ndarray | torch.Tensor | Sequence[float] | None = None,
    higher_is_better: bool = False,
    sort: bool = True,
    rng: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Generate 0-based candidate input index combinations ``(K, input_count)``.

    - ``candidates=None``: exhaustive combinations :math:`\\binom{\\text{previous\\_count}}{\\text{input\\_count}}`
      unless capped by ``max_candidates``.
    - ``candidates=k`` (``int``): up to ``k`` distinct combinations.
    - ``candidates=fn`` (callable): ``fn(previous_count, input_count)`` returning 0-based index rows.
    - When ``input_scores`` is provided and total combinations exceed the cap, tuples are prioritized
      lazily by input performance rank without materializing all combinations.
    """
    n_prev = int(previous_count)
    n_in = int(input_count)
    if n_in <= 0 or n_in > n_prev:
        return np.empty((0, max(0, n_in)), dtype=np.int64)

    if callable(candidates):
        raw = candidates(n_prev, n_in)
        arr = (
            raw.detach().cpu().numpy().astype(np.int64)
            if isinstance(raw, torch.Tensor)
            else np.asarray(raw, dtype=np.int64)
        )
        if arr.ndim == 1:
            arr = arr.reshape(-1, n_in)
        return arr

    total = comb_no(n_prev, n_in)
    cap_candidates = int(candidates) if (candidates is not None and int(candidates) > 0) else 0
    cap_max = int(max_candidates) if (max_candidates is not None and int(max_candidates) > 0) else 0
    if cap_candidates > 0 and cap_max > 0:
        k = min(cap_candidates, cap_max)
    else:
        k = cap_candidates or cap_max

    if k <= 0 or k >= total:
        combs = _cached_combinations(n_prev, n_in)
        if sort:
            return combs
        gen = np.random.default_rng(rng)
        return combs[gen.permutation(total)]

    if input_scores is not None:
        scores_np = (
            input_scores.detach().cpu().numpy()
            if isinstance(input_scores, torch.Tensor)
            else np.asarray(input_scores, dtype=np.float64)
        )
        return _ranked_candidate_indices(
            n_prev,
            n_in,
            k,
            scores_np,
            higher_is_better=higher_is_better,
        )

    gen = np.random.default_rng(rng)
    if total <= 500_000:
        all_combs = _cached_combinations(n_prev, n_in)
        chosen = np.sort(gen.choice(total, size=k, replace=False))
        return all_combs[chosen]

    seen: set[tuple[int, ...]] = set()
    rows: list[tuple[int, ...]] = []
    while len(rows) < k:
        comb = tuple(sorted(int(x) for x in gen.choice(n_prev, size=n_in, replace=False)))
        if comb not in seen:
            seen.add(comb)
            rows.append(comb)
    rows.sort()
    return np.asarray(rows, dtype=np.int64)


def gmdh_coeff_count(
    input_count: int,
    max_degree: int | None = None,
    *,
    interaction: int | None = None,
) -> int:
    """Compute the number of polynomial coefficients for ``input_count`` inputs,
    pure power degree ``max_degree``, and cross-product order ``interaction``.

    .. math::
        C = 1 + n \\, d + \\sum_{k=2}^{q} \\binom{n}{k}
    """
    n = int(input_count)
    d = n if (max_degree is None or int(max_degree) < 1) else int(max_degree)
    q = n if interaction is None else min(n, max(0, int(interaction)))

    if n < 2 or q < 2:
        cross_terms = 0
    elif q >= n:
        cross_terms = (1 << n) - n - 1
    else:
        cross_terms = sum(comb_no(n, k) for k in range(2, q + 1))
    return 1 + d * n + cross_terms


def precompute_layer_powers_torch(samples: torch.Tensor, degree: int) -> torch.Tensor:
    """Precompute pure powers ``samples ** k`` for ``k = 1..degree`` once for an entire layer in PyTorch.

    Returns tensor of shape ``(N, P, degree)`` where ``P = samples.shape[1]``.
    """
    if samples.ndim == 1:
        samples = samples.unsqueeze(1)
    d = max(1, int(degree))
    if d == 1:
        return samples.unsqueeze(-1)
    powers = torch.arange(1, d + 1, dtype=samples.dtype, device=samples.device).view(1, 1, d)
    return samples.unsqueeze(-1) ** powers


def precompute_layer_powers(
    samples: np.ndarray | torch.Tensor, degree: int
) -> np.ndarray | torch.Tensor:
    """Precompute pure powers ``samples ** k`` for ``k = 1..degree`` once for an entire layer."""
    if isinstance(samples, torch.Tensor):
        return precompute_layer_powers_torch(samples, degree)
    s_t = torch.as_tensor(np.asarray(samples, dtype=np.float64), dtype=torch.float64)
    return precompute_layer_powers_torch(s_t, degree).numpy()


def gmdh_preprocess(
    inputs: np.ndarray | torch.Tensor,
    max_degree: int | None = None,
    *,
    degree: int | None = None,
    interaction: int | None = None,
) -> np.ndarray | torch.Tensor:
    """Build the GMDH polynomial design matrix ``X`` for a single neuron's inputs ``(N, n)``."""
    is_tensor = isinstance(inputs, torch.Tensor)
    x_in = inputs if is_tensor else torch.as_tensor(np.asarray(inputs, dtype=np.float64), dtype=torch.float64)
    if x_in.ndim == 1:
        x_in = x_in.unsqueeze(1)
    n_samples, n_inputs = int(x_in.shape[0]), int(x_in.shape[1])
    eff_deg = degree if degree is not None else max_degree
    d = n_inputs if (eff_deg is None or int(eff_deg) < 1) else int(eff_deg)
    q = n_inputs if interaction is None else min(n_inputs, max(0, int(interaction)))

    coeff_count = gmdh_coeff_count(n_inputs, d, interaction=q)
    x_design = torch.empty((n_samples, coeff_count), dtype=x_in.dtype, device=x_in.device)
    x_design[:, 0] = 1.0

    w = 1
    if d == 1:
        x_design[:, 1 : 1 + n_inputs] = x_in
        w += n_inputs
    else:
        powers = torch.arange(1, d + 1, dtype=x_in.dtype, device=x_in.device).view(1, 1, d)
        x_design[:, 1 : 1 + n_inputs * d] = (x_in.unsqueeze(-1) ** powers).reshape(
            n_samples, n_inputs * d
        )
        w += n_inputs * d

    for i in range(2, q + 1):
        cidxs = torch.as_tensor(_cached_combinations(n_inputs, i), dtype=torch.long, device=x_in.device)
        num_combs = int(cidxs.shape[0])
        x_design[:, w : w + num_combs] = torch.prod(x_in[:, cidxs], dim=2)
        w += num_combs

    return x_design if is_tensor else x_design.detach().cpu().numpy()


def gmdh_preprocess_batch(
    samples: np.ndarray | torch.Tensor,
    neuron_indices_0based: np.ndarray | torch.Tensor,
    degree: int | None = None,
    precomputed_powers: np.ndarray | torch.Tensor | None = None,
    *,
    interaction: int | None = None,
) -> np.ndarray | torch.Tensor:
    """Construct a 3D batched design tensor ``X_batch`` of shape ``(B, N, coeff_count)``
    for ``B`` neurons simultaneously in PyTorch.
    """
    is_tensor = isinstance(samples, torch.Tensor)
    s = samples if is_tensor else torch.as_tensor(np.asarray(samples, dtype=np.float64), dtype=torch.float64)
    if s.ndim == 1:
        s = s.unsqueeze(1)
    idx = (
        neuron_indices_0based.to(dtype=torch.long, device=s.device)
        if isinstance(neuron_indices_0based, torch.Tensor)
        else torch.as_tensor(np.asarray(neuron_indices_0based, dtype=np.int64), dtype=torch.long, device=s.device)
    )
    if idx.ndim == 1:
        idx = idx.unsqueeze(0)

    b_count, n_inputs = int(idx.shape[0]), int(idx.shape[1])
    n_samples = int(s.shape[0])
    d = n_inputs if (degree is None or int(degree) < 1) else int(degree)
    q = n_inputs if interaction is None else min(n_inputs, max(0, int(interaction)))

    if precomputed_powers is None:
        p_powers = precompute_layer_powers_torch(s, d)
    else:
        p_powers = (
            precomputed_powers
            if isinstance(precomputed_powers, torch.Tensor)
            else torch.as_tensor(precomputed_powers, dtype=s.dtype, device=s.device)
        )
        if p_powers.shape[2] < d:
            p_powers = precompute_layer_powers_torch(s, d)

    coeff_count = gmdh_coeff_count(n_inputs, d, interaction=q)
    x_batch = torch.empty((b_count, n_samples, coeff_count), dtype=s.dtype, device=s.device)
    x_batch[:, :, 0] = 1.0

    gathered_powers = p_powers[:, idx, :d].permute(1, 0, 2, 3)
    x_batch[:, :, 1 : 1 + n_inputs * d] = gathered_powers.reshape(b_count, n_samples, n_inputs * d)

    w = 1 + n_inputs * d
    if n_inputs >= 2 and q >= 2:
        base_inputs = gathered_powers[:, :, :, 0]
        for i in range(2, q + 1):
            cidxs = torch.as_tensor(_cached_combinations(n_inputs, i), dtype=torch.long, device=s.device)
            num_combs = int(cidxs.shape[0])
            if num_combs > 0:
                cross_slice = x_batch[:, :, w : w + num_combs]
                torch.mul(
                    base_inputs[:, :, cidxs[:, 0]],
                    base_inputs[:, :, cidxs[:, 1]],
                    out=cross_slice,
                )
                for k in range(2, i):
                    cross_slice.mul_(base_inputs[:, :, cidxs[:, k]])
                w += num_combs

    return x_batch if is_tensor else x_batch.detach().cpu().numpy()


def gmdh_polynomial(
    inputs: np.ndarray | torch.Tensor,
    weights: np.ndarray | torch.Tensor,
    degree: int | None = None,
    *,
    interaction: int | None = None,
) -> np.ndarray | torch.Tensor:
    """Evaluate a GMDH polynomial expression ``X @ weights``."""
    is_tensor = isinstance(inputs, torch.Tensor) and isinstance(weights, torch.Tensor)
    w = (
        weights
        if isinstance(weights, torch.Tensor)
        else torch.as_tensor(np.asarray(weights, dtype=np.float64), dtype=torch.float64)
    )
    if w.ndim == 2:
        if w.shape[0] != 1:
            raise ValueError("weights must be a 1D vector or a single-row 2D array.")
        w = w.reshape(-1)
    elif w.ndim != 1:
        raise ValueError("weights must be a row vector.")

    x_in = (
        inputs.to(dtype=w.dtype, device=w.device)
        if isinstance(inputs, torch.Tensor)
        else torch.as_tensor(np.asarray(inputs, dtype=np.float64), dtype=w.dtype, device=w.device)
    )
    x_design: torch.Tensor = gmdh_preprocess(x_in, degree, interaction=interaction)  # type: ignore[assignment]
    out = (x_design @ w).unsqueeze(1)
    return out if is_tensor else out.detach().cpu().numpy()
