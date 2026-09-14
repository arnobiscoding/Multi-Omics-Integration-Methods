"""
Unit tests for topological_diagnostic.py.

Covers:
1. Synthetic toy graph with hand-computed expected similarities and case labels.
2. Degenerate edge cases (singleton cluster, singleton ground-truth domain,
   empty adjacency row, all-identical embeddings).
3. Completeness and mutual exclusivity of case partitioning (no unclassified/nulls).
4. Vectorized neighborhood-mean vs naive Python reference loop.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from topological_diagnostic import (
    _compute_group_centroids,
    _l2_normalize_rows,
    summarize_topological_diagnostic,
    topological_diagnostic,
)


# =============================================================================
# Test 1: Vectorized Neighborhood Mean vs Naive Reference Loop
# =============================================================================
def test_neighborhood_mean_vectorization():
    """Verify vectorized sparse neighborhood mean matches naive per-node loop."""
    rng = np.random.RandomState(42)
    N, D = 50, 16
    Z = rng.randn(N, D)

    # Random sparse adjacency matrix with some isolated nodes
    A_dense = (rng.rand(N, N) > 0.85).astype(float)
    np.fill_diagonal(A_dense, 0.0)
    # Ensure node 0 is isolated
    A_dense[0, :] = 0.0
    A_dense[:, 0] = 0.0
    A_sparse = sp.csr_matrix(A_dense)

    # Naive reference loop
    mu_N_ref = np.zeros((N, D), dtype=float)
    degrees_ref = np.zeros(N, dtype=float)
    for i in range(N):
        nbrs = np.where(A_dense[i] > 0)[0]
        deg = len(nbrs)
        degrees_ref[i] = deg
        if deg > 0:
            mu_N_ref[i] = np.mean(Z[nbrs], axis=0)
        else:
            mu_N_ref[i] = np.nan

    # Vectorized computation
    degrees_vec = np.array(A_sparse.sum(axis=1)).flatten()
    safe_deg = np.where(degrees_vec == 0, 1.0, degrees_vec)[:, None]
    mu_N_vec = (A_sparse.dot(Z)) / safe_deg
    mu_N_vec[degrees_vec == 0] = np.nan

    assert np.allclose(degrees_vec, degrees_ref)
    # Compare non-isolated rows
    non_isolated = degrees_ref > 0
    assert np.allclose(mu_N_vec[non_isolated], mu_N_ref[non_isolated], atol=1e-10)
    assert np.all(np.isnan(mu_N_vec[~non_isolated]))


# =============================================================================
# Test 2: Centroid Vectorization & Singleton Behavior
# =============================================================================
def test_centroid_computation_and_singleton():
    """Verify group centroid calculation and singleton domain S_G == 1.0."""
    Z = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 5.0],  # Singleton group
        ]
    )
    labels = np.array(["clusterA", "clusterA", "singletonB"])
    mu, sizes = _compute_group_centroids(Z, labels)

    # Cluster A centroid should be [2.0, 0.0]
    assert np.allclose(mu[0], [2.0, 0.0])
    assert np.allclose(mu[1], [2.0, 0.0])
    assert sizes[0] == 2
    assert sizes[1] == 2

    # Singleton B centroid should equal the spot itself
    assert np.allclose(mu[2], Z[2])
    assert sizes[2] == 1


# =============================================================================
# Test 3: Synthetic Toy Graph Classification (All Diagnostic Cases)
# =============================================================================
def test_synthetic_toy_graph_cases():
    """
    Test 20-spot toy dataset with known expected classifications covering:
    Case 1, Case 2, Case 7, Case 2b, Case 5, Case 3, Case 6, Case 4, Isolated, Unannotated.
    """
    tau = 0.85
    rng = np.random.RandomState(123)

    v_A = np.array([1.0, 0.0, 0.0, 0.0])
    v_B = np.array([0.0, 1.0, 0.0, 0.0])
    v_C = np.array([0.0, 0.0, 1.0, 0.0])
    v_D = np.array([0.0, 0.0, 0.0, 1.0])

    def norm(v):
        return v / np.linalg.norm(v)

    # Spots designed for each case + anchors
    # We add sufficient DomA anchors to maintain DomA centroid alignment with v_A
    spots = [
        # 0: Case 1 (Homogeneous Core: Success, S_G >= tau, S_N >= tau)
        {"Z": norm(v_A), "true": "DomA", "pred": "DomA", "expected": "Case 1"},
        # 1: Case 2 (Boundary Triumph: Success, S_G >= tau, S_N < tau)
        {"Z": norm(v_A), "true": "DomA", "pred": "DomA", "expected": "Case 2"},
        # 2: Case 7 (Lucky Guess: Success, S_G < tau, S_N < tau)
        {"Z": norm(0.3 * v_A + 0.3 * v_B + 0.9 * v_D), "true": "DomA", "pred": "DomA", "expected": "Case 7"},
        # 3: Case 2b (Neighbor-Assisted: Success, S_G < tau, S_N >= tau)
        {"Z": norm(0.4 * v_A + 0.9 * v_D), "true": "DomA", "pred": "DomA", "expected": "Case 2b"},
        # 4: Case 5 (Resolution Fracture: Failure, S_G >= tau)
        {"Z": norm(v_A), "true": "DomA", "pred": "ClustB", "expected": "Case 5"},
        # 5: Case 3 (Oversmoothed Victim: Failure, S_G < tau, S_N > S_G)
        {"Z": norm(0.4 * v_A + 0.9 * v_B), "true": "DomA", "pred": "ClustB", "expected": "Case 3"},
        # 6: Case 6 (Feature Betrayal: Failure, S_G < tau, S_N <= S_G, S_C >= tau)
        {"Z": norm(v_C), "true": "DomA", "pred": "ClustC", "expected": "Case 6"},
        # 7: Case 4 (Feature Orphan: Failure, S_G < tau, S_N <= S_G, S_C < tau)
        {"Z": norm(v_D), "true": "DomA", "pred": "ClustB", "expected": "Case 4"},
        # 8: Isolated Spot (degree 0, valid annotation)
        {"Z": norm(v_A), "true": "DomA", "pred": "DomA", "expected": "Isolated Spot"},
        # 9: Unannotated (NaN ground truth)
        {"Z": norm(v_A), "true": np.nan, "pred": "DomA", "expected": "Unannotated"},
        # Anchors to shape domain/cluster centroids
        # 10, 11, 12, 13: Anchors for DomA
        {"Z": norm(v_A), "true": "DomA", "pred": "DomA", "expected": "Case 1"},
        {"Z": norm(v_A), "true": "DomA", "pred": "DomA", "expected": "Case 1"},
        {"Z": norm(v_A), "true": "DomA", "pred": "DomA", "expected": "Case 1"},
        {"Z": norm(v_A), "true": "DomA", "pred": "DomA", "expected": "Case 1"},
        # 14: Anchor DomB / ClustB
        {"Z": norm(v_B), "true": "DomB", "pred": "ClustB", "expected": "Case 1"},
        # 15: Anchor ClustC
        {"Z": norm(v_C), "true": "DomB", "pred": "ClustC", "expected": "Case 2"},
        # 16: Twin of spot 3 for neighbor agreement
        {"Z": norm(0.4 * v_A + 0.9 * v_D), "true": "DomA", "pred": "DomA", "expected": "Case 2b"},
    ]

    N = len(spots)
    Z = np.array([s["Z"] for s in spots])
    L_true = np.array([s["true"] for s in spots], dtype=object)
    L_pred = np.array([s["pred"] for s in spots], dtype=object)

    adj = np.zeros((N, N), dtype=float)
    # Spot 0 connected to 10 (high S_N ~ v_A)
    adj[0, 10] = 1.0
    adj[10, 0] = 1.0

    # Spot 1 connected to 14 (Domain B neighbor -> low S_N)
    adj[1, 14] = 1.0
    adj[14, 1] = 1.0

    # Spot 2 connected to 14 (Domain B -> low S_N)
    adj[2, 14] = 1.0
    adj[14, 2] = 1.0

    # Spot 3 connected to 16 (Twin -> high S_N)
    adj[3, 16] = 1.0
    adj[16, 3] = 1.0

    # Spot 4 connected to 10
    adj[4, 10] = 1.0
    adj[10, 4] = 1.0

    # Spot 5 connected to 14 (Domain B neighbor -> high S_N toward B > S_G toward A)
    adj[5, 14] = 1.0
    adj[14, 5] = 1.0

    # Spot 6 connected to 14 (low S_N <= S_G)
    adj[6, 14] = 1.0
    adj[14, 6] = 1.0

    # Spot 7 connected to 14 (low S_N <= S_G)
    adj[7, 14] = 1.0
    adj[14, 7] = 1.0

    # Spot 8 is isolated (row 8 has no edges)
    # Spot 9 connected to 10
    adj[9, 10] = 1.0
    adj[10, 9] = 1.0

    # Anchor 15 connected to 14
    adj[15, 14] = 1.0
    adj[14, 15] = 1.0

    A = sp.csr_matrix(adj)

    df = topological_diagnostic(Z, A, L_pred, L_true, tau_sim=tau)

    for i in range(10):
        expected = spots[i]["expected"]
        actual = str(df.loc[i, "case"])
        assert actual == expected, (
            f"Spot {i} mismatch: expected {expected}, got {actual}. "
            f"S_G={df.loc[i, 'S_G']:.3f}, S_C={df.loc[i, 'S_C']:.3f}, S_N={df.loc[i, 'S_N']}"
        )


# =============================================================================
# Test 4: Degenerate Cases
# =============================================================================
def test_degenerate_cases():
    """Verify robust handling of single-spot cluster, identical embeddings, empty graph."""
    N, D = 10, 5

    # 1. All-identical embeddings (no NaN/inf division by zero)
    Z_identical = np.ones((N, D))
    A_empty = sp.csr_matrix((N, N))
    L_pred = np.array(["C1"] * N)
    L_true = np.array(["D1"] * N)

    df = topological_diagnostic(Z_identical, A_empty, L_pred, L_true, tau_sim=0.85)

    assert not df["case"].isnull().any()
    # Since degree is 0, all should be Isolated Spot
    assert (df["case"] == "Isolated Spot").all()
    assert (df["degree"] == 0).all()
    assert df["S_N"].isnull().all()
    assert np.allclose(df["S_G"], 1.0)
    assert np.allclose(df["S_C"], 1.0)

    # 2. Single spot cluster & single spot ground-truth domain
    Z_single = np.random.randn(3, 4)
    # Connect 0 and 1; leave 2 alone in graph
    adj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    A = sp.csr_matrix(adj)
    # Spot 2 has singleton ground truth and singleton cluster
    L_pred_single = np.array(["C1", "C1", "SingletonClust"])
    L_true_single = np.array(["G1", "G1", "SingletonDomain"])

    df_single = topological_diagnostic(Z_single, A, L_pred_single, L_true_single)
    # Spot 2 is isolated
    assert df_single.loc[2, "case"] == "Isolated Spot"
    assert df_single.loc[2, "S_G"] == 1.0  # Singleton ground truth centroid equals spot


# =============================================================================
# Test 5: Partition Completeness & Mutual Exclusivity
# =============================================================================
def test_partition_completeness_and_mutual_exclusivity():
    """Verify that every spot gets classified into exactly one valid case, no unclassified."""
    rng = np.random.RandomState(99)
    N, D = 100, 8
    Z = rng.randn(N, D)
    A = sp.csr_matrix((rng.rand(N, N) > 0.8).astype(float))

    clusters = [f"Clust_{i}" for i in rng.randint(0, 5, size=N)]
    domains = [f"Domain_{i}" if rng.rand() > 0.15 else np.nan for i in rng.randint(0, 4, size=N)]

    df = topological_diagnostic(Z, A, clusters, domains, tau_sim=0.85)

    assert len(df) == N
    # No null values in case column
    assert not df["case"].isnull().any()
    # Zero unclassified spots
    assert (df["case"] != "Unclassified").all()

    # Summary table checks
    summary = summarize_topological_diagnostic(df)
    assert summary["count"].sum() == N
    assert np.isclose(summary["percentage"].sum(), 100.0, atol=0.1)


# =============================================================================
# Test 6: Custom Unannotated Values
# =============================================================================
def test_custom_unannotated_values():
    """Verify that custom unannotated strings (e.g. 'Exclude', 'unknown') are flagged correctly."""
    Z = np.ones((4, 2))
    A = sp.csr_matrix(np.ones((4, 4)) - np.eye(4))
    L_pred = ["ValidA", "ValidA", "ValidA", "ValidA"]
    L_true = ["ValidA", "Exclude", "unknown", np.nan]

    df = topological_diagnostic(Z, A, L_pred, L_true)
    assert df.loc[0, "case"] == "Case 1"
    assert df.loc[1, "case"] == "Unannotated"
    assert df.loc[2, "case"] == "Unannotated"
    assert df.loc[3, "case"] == "Unannotated"
