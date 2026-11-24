"""Tests for polynomial design matrix construction, combinatorics, and 3D batch preprocessing."""

from __future__ import annotations

import unittest

import numpy as np
from gmdhtorch import (
    comb_no,
    gmdh_coeff_count,
    gmdh_polynomial,
    gmdh_preprocess,
    gmdh_preprocess_batch,
    precompute_layer_powers,
    resolve_candidate_indices,
)


class TestPolynomial(unittest.TestCase):
    def test_comb_no(self) -> None:
        self.assertEqual(comb_no(5, 2), 10)
        self.assertEqual(comb_no(6, 3), 20)
        self.assertEqual(comb_no(10, 0), 1)
        self.assertEqual(comb_no(10, 10), 1)
        self.assertEqual(comb_no(4, 5), 0)

    def test_resolve_candidate_indices_ranked_and_shuffled(self) -> None:
        # Unsorted (shuffled) exhaustive combinations without twiddle
        shuffled = resolve_candidate_indices(4, 2, sort=False, rng=np.random.default_rng(7))
        self.assertEqual(shuffled.shape, (6, 2))
        self.assertEqual(np.unique(shuffled, axis=0).shape[0], 6)

        # Performance-ranked lazy selection when comb(100, 2) = 4950 > max_candidates = 10
        scores = np.arange(100, dtype=np.float64)
        scores[42] = -10.0  # best
        scores[7] = -5.0    # second best
        ranked = resolve_candidate_indices(100, 2, max_candidates=10, input_scores=scores)
        self.assertEqual(ranked.shape, (10, 2))
        # Best pair (7, 42) must be included first
        self.assertIn([7, 42], ranked.tolist())

    def test_gmdh_coeff_count_and_decoupled_interaction(self) -> None:
        self.assertEqual(gmdh_coeff_count(1, 1), 2)
        self.assertEqual(gmdh_coeff_count(2, 2), 6)
        self.assertEqual(gmdh_coeff_count(3, 3), 14)
        self.assertEqual(gmdh_coeff_count(3, 2), 11)
        # Decoupled interaction (§4.5):
        # Multilinear: 3 inputs, degree 1, interaction 3 -> 1 + 3 + 3 + 1 = 8
        self.assertEqual(gmdh_coeff_count(3, 1, interaction=3), 8)
        # Additive quadratic (no cross terms): 2 inputs, degree 2, interaction 0 -> 1 + 4 = 5
        self.assertEqual(gmdh_coeff_count(2, 2, interaction=0), 5)

    def test_decoupled_interaction_preprocess(self) -> None:
        rng = np.random.default_rng(42)
        samples = rng.normal(size=(12, 4))
        idx0 = resolve_candidate_indices(4, 3)
        # Multilinear: degree=1, interaction=3
        batch_ml = gmdh_preprocess_batch(samples, idx0, degree=1, interaction=3)
        self.assertEqual(batch_ml.shape, (idx0.shape[0], 12, 8))
        for b in range(idx0.shape[0]):
            single_ml = gmdh_preprocess(samples[:, idx0[b]], degree=1, interaction=3)
            np.testing.assert_allclose(batch_ml[b], single_ml, atol=1e-14)

    def test_resolve_candidate_indices(self) -> None:
        # Exhaustive
        ex = resolve_candidate_indices(6, 2, candidates=None)
        self.assertEqual(ex.shape, (15, 2))
        # Random int cap
        sampled = resolve_candidate_indices(10, 3, candidates=20, rng=np.random.default_rng(0))
        self.assertEqual(sampled.shape, (20, 3))
        # Unique rows & sorted columns
        self.assertEqual(np.unique(sampled, axis=0).shape[0], 20)
        np.testing.assert_array_equal(sampled, np.sort(sampled, axis=1))
        # Custom callable
        custom = resolve_candidate_indices(
            6, 2, candidates=lambda n_avail, k: np.array([[0, 1], [2, 5]], dtype=np.int64)
        )
        np.testing.assert_array_equal(custom, np.array([[0, 1], [2, 5]], dtype=np.int64))

    def test_gmdh_preprocess_column_ordering(self) -> None:
        x = np.array([[2.0, 3.0], [-1.0, 4.0]])
        design = gmdh_preprocess(x, 2)
        expected = np.array(
            [
                [1.0, 2.0, 4.0, 3.0, 9.0, 6.0],
                [1.0, -1.0, 1.0, 4.0, 16.0, -4.0],
            ]
        )
        np.testing.assert_allclose(design, expected, atol=1e-15)

    def test_batched_preprocess_matches_single_neuron(self) -> None:
        rng = np.random.default_rng(123)
        samples = rng.normal(size=(25, 6))
        for n_inputs, degree in [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 2), (4, 4)]:
            idx_0based = resolve_candidate_indices(6, n_inputs)

            powers = precompute_layer_powers(samples, degree)
            x_batch = gmdh_preprocess_batch(samples, idx_0based, degree, precomputed_powers=powers)

            self.assertEqual(
                x_batch.shape,
                (idx_0based.shape[0], 25, gmdh_coeff_count(n_inputs, degree)),
            )
            for b in range(idx_0based.shape[0]):
                single = gmdh_preprocess(samples[:, idx_0based[b]], degree)
                np.testing.assert_allclose(x_batch[b], single, atol=1e-14)

    def test_gmdh_polynomial_evaluation(self) -> None:
        rng = np.random.default_rng(7)
        inputs = rng.uniform(-2.0, 2.0, size=(15, 3))
        weights = rng.uniform(-1.0, 1.0, size=(1, gmdh_coeff_count(3, 2)))
        out = gmdh_polynomial(inputs, weights, degree=2)
        expected = (gmdh_preprocess(inputs, 2) @ weights.ravel())[:, None]
        np.testing.assert_allclose(out, expected, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
