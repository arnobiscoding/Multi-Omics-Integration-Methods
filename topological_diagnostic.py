"""
Micro-Level Topological Diagnostic Framework for Spatial Multi-Omics.

Performs a spot-by-spot audit of spatial clustering results by computing:
- Four latent-space vectors: Z_i, mu_G_i, mu_C_i, mu_N_i
- Three cosine similarity scores: S_G (ground truth), S_C (predicted cluster), S_N (spatial neighborhood)
- Deterministic priority routing into 7 diagnostic cases (Case 1 - Case 7, Case 2b) and edge cases
  (Isolated Spot, Unannotated).

Designed as a post-processing diagnostic step compatible with spatial GNNs, scGPT,
and standard AnnData pipelines (scgptpipeline, smartpipeline, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger("topological_diagnostic")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Canonical Case Definitions and Metadata
CASE_METADATA: Dict[str, Dict[str, str]] = {
    "Case 1": {
        "name": "Homogeneous Core",
        "description": "Deep within ground-truth domain; high internal similarity and high local spatial agreement.",
        "branch": "Success",
    },
    "Case 2": {
        "name": "Boundary Triumph",
        "description": "At domain boundary with mixed physical neighbors, but model correctly anchored to ground truth.",
        "branch": "Success",
    },
    "Case 2b": {
        "name": "Neighbor-Assisted Correct",
        "description": "Residual correct spot agreeing with physical neighbors despite lower domain centroid similarity.",
        "branch": "Success",
    },
    "Case 3": {
        "name": "Oversmoothed Victim",
        "description": "Over-smoothing pulled embedding toward spatial neighbors and away from ground-truth centroid.",
        "branch": "Failure",
    },
    "Case 4": {
        "name": "Feature Orphan",
        "description": "Fails across all dimensions: no affinity to ground truth, neighborhood, or predicted cluster.",
        "branch": "Failure",
    },
    "Case 5": {
        "name": "Resolution Fracture",
        "description": "Spot aligns closely with true domain centroid, but was split into a different cluster (over-clustering).",
        "branch": "Failure",
    },
    "Case 6": {
        "name": "Feature Betrayal",
        "description": "Feature artifact dominates: aligns with incorrect cluster centroid over neighbors and ground truth.",
        "branch": "Failure",
    },
    "Case 7": {
        "name": "Lucky Guess",
        "description": "Low affinity to both ground-truth centroid and neighborhood, yet received correct label by chance.",
        "branch": "Success",
    },
    "Isolated Spot": {
        "name": "Isolated Spot",
        "description": "Zero physical neighbors (degree 0); spatial neighborhood comparisons are undefined.",
        "branch": "Edge Case",
    },
    "Unannotated": {
        "name": "Unannotated",
        "description": "Ground-truth annotation missing or excluded; cannot evaluate clustering success/failure.",
        "branch": "Edge Case",
    },
    "Unclassified": {
        "name": "Unclassified",
        "description": "Fell through diagnostic decision tree; indicates an unexpected logic gap.",
        "branch": "Safety Net",
    },
}

DEFAULT_UNANNOTATED_LABELS: Set[str] = {
    "nan",
    "none",
    "unknown",
    "exclude",
    "<na>",
    "unannotated",
    "na",
    "null",
}


def _l2_normalize_rows(
    matrix: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute row-wise L2 normalized matrix safely.

    Mathematical formula:
    $$Z_{\\text{norm}, i} = \\frac{Z_i}{\\max(\\|Z_i\\|_2, \\epsilon)}$$

    Parameters
    ----------
    matrix : np.ndarray
        (N, D) input matrix.
    eps : float
        Numerical epsilon to avoid division by zero.

    Returns
    -------
    np.ndarray
        (N, D) row-wise unit L2 normalized array.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Handle NaNs or near-zero norms safely
    valid_norms = np.where(np.isnan(norms) | (norms < eps), eps, norms)
    normed = matrix / valid_norms
    # Restore NaN rows if the input row was all NaN
    nan_rows = np.isnan(matrix).any(axis=1)
    if nan_rows.any():
        normed[nan_rows] = np.nan
    return normed


def _compute_group_centroids(
    Z: np.ndarray,
    labels: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute vectorized group centroids and broadcast back to (N, D).

    Mathematical formula:
    $$\\mu_{k} = \\frac{1}{|S_k|} \\sum_{j \\in S_k} Z_j$$
    $$\\mu_i = \\mu_{L_i}$$

    Uses pd.factorize and np.add.at for O(N * D) runtime without nested loops.
    For singleton groups (|S_k| = 1), mu_i == Z_i trivially.

    Parameters
    ----------
    Z : np.ndarray
        (N, D) latent embedding matrix.
    labels : np.ndarray
        (N,) categorical / integer / string labels.
    valid_mask : Optional[np.ndarray]
        (N,) boolean mask indicating valid rows to include in centroid calculation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - mu : (N, D) broadcasted centroid matrix (NaN for invalid rows).
        - group_sizes : (N,) size of the group each spot belongs to.
    """
    N, D = Z.shape
    mu = np.full((N, D), np.nan, dtype=float)
    group_sizes = np.zeros(N, dtype=int)

    if valid_mask is None:
        valid_mask = np.ones(N, dtype=bool)

    if not np.any(valid_mask):
        return mu, group_sizes

    valid_indices = np.where(valid_mask)[0]
    sub_labels = labels[valid_indices]
    sub_Z = Z[valid_indices]

    codes, uniques = pd.factorize(sub_labels)
    K = len(uniques)

    # Vectorized accumulation of sums and counts per group
    counts = np.bincount(codes, minlength=K)
    sums = np.zeros((K, D), dtype=float)
    np.add.at(sums, codes, sub_Z)

    # Compute group centroids: sums / counts
    safe_counts = np.maximum(counts[:, None], 1)
    centroids = sums / safe_counts

    # Broadcast centroids and group sizes back to valid rows
    mu[valid_indices] = centroids[codes]
    group_sizes[valid_indices] = counts[codes]

    return mu, group_sizes


def topological_diagnostic(
    Z: np.ndarray,
    A: sp.spmatrix,
    L_pred: Union[np.ndarray, pd.Series, Sequence],
    L_true: Union[np.ndarray, pd.Series, Sequence],
    tau_sim: float = 0.85,
    spot_ids: Optional[Union[np.ndarray, pd.Series, Sequence]] = None,
    unannotated_values: Optional[Set[Any]] = None,
) -> pd.DataFrame:
    """
    Perform a spot-by-spot topological diagnostic audit of spatial clustering.

    Parameters
    ----------
    Z : np.ndarray
        (N, D) latent feature/embedding matrix. Does NOT need to be pre-normalized.
    A : sp.spmatrix
        (N, N) physical spatial adjacency matrix (CSR, CSC, or COO; binary or weighted).
        May contain degree-0 rows (isolated spots).
    L_pred : np.ndarray | pd.Series | Sequence
        (N,) predicted cluster labels (e.g. from Leiden, KMeans, mclust).
    L_true : np.ndarray | pd.Series | Sequence
        (N,) ground-truth domain annotations (e.g. pathology annotations). May contain NaNs/unknowns.
    tau_sim : float, default=0.85
        Cosine similarity threshold for affinity classification.
    spot_ids : Optional[np.ndarray | pd.Series | Sequence], default=None
        Optional (N,) spot barcodes or identifiers for output DataFrame indexing.
    unannotated_values : Optional[Set[Any]], default=None
        Custom set of label strings or values to treat as unannotated. In addition to NaN/None,
        defaults to {'nan', 'none', 'unknown', 'exclude', '<na>', 'unannotated', 'na', 'null'}.

    Returns
    -------
    pd.DataFrame
        Annotated DataFrame containing one row per spot with columns:
        - `spot_id`: Spot identifier (from `spot_ids` or 0..N-1)
        - `L_pred`: Original predicted cluster label
        - `L_true`: Original ground-truth label (or NaN)
        - `S_G`: Cosine similarity between Z_i and ground-truth centroid mu_G_i
        - `S_C`: Cosine similarity between Z_i and predicted cluster centroid mu_C_i
        - `S_N`: Cosine similarity between Z_i and neighborhood mean mu_N_i
        - `degree`: Number of physical spatial neighbors (row-sum of A)
        - `case`: Categorical diagnostic case assignment
    """
    # -------------------------------------------------------------------------
    # Input Validation & Preprocessing
    # -------------------------------------------------------------------------
    if not isinstance(Z, np.ndarray):
        Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError(f"Z must be a 2D array of shape (N, D), got shape {Z.shape}")

    N, D = Z.shape

    # Ensure A is a scipy CSR matrix
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    else:
        A = A.tocsr()

    if A.shape != (N, N):
        raise ValueError(f"A must have shape ({N}, {N}) matching Z length {N}, got {A.shape}")

    # Normalize L_pred and L_true into consistent 1D arrays
    L_pred_arr = np.asarray(L_pred)
    L_true_arr = np.asarray(L_true)
    if L_pred_arr.shape[0] != N or L_true_arr.shape[0] != N:
        raise ValueError(
            f"L_pred ({L_pred_arr.shape[0]}) and L_true ({L_true_arr.shape[0]}) must have length N={N}"
        )

    # Spot IDs
    if spot_ids is None:
        spot_ids_arr = np.arange(N)
    else:
        spot_ids_arr = np.asarray(spot_ids)
        if len(spot_ids_arr) != N:
            raise ValueError(f"spot_ids length ({len(spot_ids_arr)}) must match N={N}")

    # Unannotated values set
    unanno_set = DEFAULT_UNANNOTATED_LABELS.copy()
    if unannotated_values:
        unanno_set.update({str(v).strip().lower() for v in unannotated_values})

    # Detect unannotated / missing ground truth labels
    def _is_unannotated_val(v: Any) -> bool:
        if pd.isna(v) or v is None:
            return True
        v_str = str(v).strip().lower()
        return v_str in unanno_set

    is_unannotated = np.array([_is_unannotated_val(v) for v in L_true_arr], dtype=bool)

    # Clean hashable label representations for comparisons
    L_pred_clean = np.array([str(x) if pd.notna(x) else "__NA__" for x in L_pred_arr])
    L_true_clean = np.array([str(x) if pd.notna(x) else "__NA__" for x in L_true_arr])

    # -------------------------------------------------------------------------
    # Part 1: Vector Computation (Fully Vectorized)
    # -------------------------------------------------------------------------
    # 1. Normalize embeddings:
    # $$Z_{\text{norm}, i} = \frac{Z_i}{\max(\|Z_i\|_2, \epsilon)}$$
    Z_norm = _l2_normalize_rows(Z, eps=1e-12)

    # 2. Neighborhood mean (mu_N):
    # $$\mu_{\mathcal{N}, i} = \frac{1}{d_i} \sum_{j} A_{ij} Z_j$$
    # Matrix form: \mu_{\mathcal{N}} = D^{-1} A Z
    degrees = np.array(A.sum(axis=1)).flatten().astype(float)
    is_isolated = degrees == 0

    # Sparse multiplication A @ Z
    A_dot_Z = A.dot(Z)
    safe_degrees = np.where(is_isolated, 1.0, degrees)[:, None]
    mu_N = A_dot_Z / safe_degrees
    mu_N[is_isolated] = np.nan

    # 3. Predicted cluster centroid (mu_C):
    # $$\mu_{C, k} = \frac{1}{|C_k|} \sum_{j \in C_k} Z_j$$
    mu_C, _ = _compute_group_centroids(Z, L_pred_clean)

    # 4. Ground-truth domain centroid (mu_G):
    # $$\mu_{G, m} = \frac{1}{|G_m|} \sum_{j \in G_m} Z_j$$
    # Excluding unannotated spots from centroid computation.
    # Note: Singleton ground-truth domains (|G_m| == 1) yield mu_G_i == Z_i trivially,
    # resulting in S(Z_i, mu_G_i) == 1.0. This is expected mathematical behavior.
    valid_anno_mask = ~is_unannotated
    mu_G, gt_sizes = _compute_group_centroids(Z, L_true_clean, valid_mask=valid_anno_mask)

    # 5. Cosine similarities:
    # $$S(Z_i, V_i) = Z_{\text{norm}, i} \cdot V_{\text{norm}, i}$$
    # Computed via row-wise einsum: np.einsum('ij,ij->i', Z_norm, V_norm)
    mu_N_norm = _l2_normalize_rows(mu_N, eps=1e-12)
    mu_C_norm = _l2_normalize_rows(mu_C, eps=1e-12)
    mu_G_norm = _l2_normalize_rows(mu_G, eps=1e-12)

    # Einsum row-wise dot products
    S_N = np.einsum("ij,ij->i", Z_norm, mu_N_norm)
    S_C = np.einsum("ij,ij->i", Z_norm, mu_C_norm)
    S_G = np.einsum("ij,ij->i", Z_norm, mu_G_norm)

    # Edge cases: isolated spots have undefined S_N -> NaN
    S_N[is_isolated] = np.nan
    # Unannotated spots have undefined S_G -> NaN
    S_G[is_unannotated] = np.nan

    # Numerical clip to [-1.0, 1.0] for non-NaN values
    S_N = np.where(np.isnan(S_N), np.nan, np.clip(S_N, -1.0, 1.0))
    S_C = np.where(np.isnan(S_C), np.nan, np.clip(S_C, -1.0, 1.0))
    S_G = np.where(np.isnan(S_G), np.nan, np.clip(S_G, -1.0, 1.0))

    # -------------------------------------------------------------------------
    # Part 2: Case Assignment Logic (Strict Priority Order via np.select)
    # -------------------------------------------------------------------------
    # Base masks
    # Success: L_pred == L_true (only defined when annotated)
    is_success = (~is_unannotated) & (L_pred_clean == L_true_clean)
    is_failure = (~is_unannotated) & (L_pred_clean != L_true_clean)

    # Non-edge standard spot mask
    is_standard = (~is_unannotated) & (~is_isolated)

    # ------------------
    # Priority Conditions:
    # ------------------
    # 1. Edge Case: Unannotated spot
    cond_unannotated = is_unannotated

    # 2. Edge Case: Isolated spot (with valid annotation)
    cond_isolated = (~is_unannotated) & is_isolated

    # Branch A: Success spots (L_pred == L_true)
    # A1. Case 1 (Homogeneous Core): S_G >= tau AND S_N >= tau
    cond_case1 = is_standard & is_success & (S_G >= tau_sim) & (S_N >= tau_sim)
    # A2. Case 2 (Boundary Triumph): S_G >= tau AND S_N < tau
    cond_case2 = is_standard & is_success & (S_G >= tau_sim) & (S_N < tau_sim)
    # A3. Case 7 (Lucky Guess): S_G < tau AND S_N < tau
    cond_case7 = is_standard & is_success & (S_G < tau_sim) & (S_N < tau_sim)
    # A4. Case 2b (Neighbor-Assisted Correct): S_G < tau AND S_N >= tau
    cond_case2b = is_standard & is_success & (S_G < tau_sim) & (S_N >= tau_sim)

    # Branch B: Failure spots (L_pred != L_true)
    # B1. Case 5 (Resolution Fracture): S_G >= tau (evaluated first)
    cond_case5 = is_standard & is_failure & (S_G >= tau_sim)
    # B2. Case 3 (Oversmoothed Victim): S_G < tau AND S_N > S_G
    cond_case3 = is_standard & is_failure & (S_G < tau_sim) & (S_N > S_G)
    # B3. Case 6 (Feature Betrayal): S_G < tau AND S_N <= S_G AND S_C >= tau
    cond_case6 = is_standard & is_failure & (S_G < tau_sim) & (S_N <= S_G) & (S_C >= tau_sim)
    # B4. Case 4 (Feature Orphan): S_G < tau AND S_N <= S_G AND S_C < tau
    cond_case4 = is_standard & is_failure & (S_G < tau_sim) & (S_N <= S_G) & (S_C < tau_sim)

    conditions = [
        cond_unannotated,
        cond_isolated,
        cond_case1,
        cond_case2,
        cond_case7,
        cond_case2b,
        cond_case5,
        cond_case3,
        cond_case6,
        cond_case4,
    ]

    choices = [
        "Unannotated",
        "Isolated Spot",
        "Case 1",
        "Case 2",
        "Case 7",
        "Case 2b",
        "Case 5",
        "Case 3",
        "Case 6",
        "Case 4",
    ]

    # Assign cases via np.select with default safety net
    assigned_cases = np.select(conditions, choices, default="Unclassified")

    # Audit safety net: check if any spot fell into Unclassified
    unclassified_count = int(np.sum(assigned_cases == "Unclassified"))
    if unclassified_count > 0:
        logger.error(
            f"Diagnostic logic gap detected: {unclassified_count} spots assigned to 'Unclassified'!"
        )

    # Log warning for Case 2b (residual neighbor-assisted correct)
    case2b_count = int(np.sum(assigned_cases == "Case 2b"))
    if case2b_count > 0:
        logger.warning(
            f"Case 2b (Neighbor-Assisted Correct) observed for {case2b_count} spots "
            f"({case2b_count / N * 100:.2f}%). These spots agree with physical neighbors "
            f"despite low domain centroid similarity."
        )

    # -------------------------------------------------------------------------
    # Part 3: Output DataFrame
    # -------------------------------------------------------------------------
    all_categories = [
        "Case 1",
        "Case 2",
        "Case 2b",
        "Case 3",
        "Case 4",
        "Case 5",
        "Case 6",
        "Case 7",
        "Isolated Spot",
        "Unannotated",
        "Unclassified",
    ]

    categorical_case = pd.Categorical(assigned_cases, categories=all_categories)

    df_out = pd.DataFrame(
        {
            "spot_id": spot_ids_arr,
            "L_pred": L_pred_arr,
            "L_true": L_true_arr,
            "S_G": S_G,
            "S_C": S_C,
            "S_N": S_N,
            "degree": degrees.astype(int),
            "case": categorical_case,
        }
    )

    return df_out


def summarize_topological_diagnostic(
    df_diagnostic: pd.DataFrame,
    sample_col: Optional[Union[str, Sequence]] = None,
) -> pd.DataFrame:
    """
    Generate summary statistics (spot count and percentage) per diagnostic case.

    Parameters
    ----------
    df_diagnostic : pd.DataFrame
        DataFrame produced by `topological_diagnostic`.
    sample_col : Optional[str | Sequence], default=None
        Optional column name in df_diagnostic or sequence of sample IDs for
        per-sample breakdown.

    Returns
    -------
    pd.DataFrame
        Summary table with counts and percentages per case.
    """
    total_spots = len(df_diagnostic)
    if total_spots == 0:
        return pd.DataFrame(columns=["case", "name", "count", "percentage"])

    if sample_col is not None and isinstance(sample_col, str) and sample_col in df_diagnostic.columns:
        # Per-sample breakdown
        grouped = (
            df_diagnostic.groupby([sample_col, "case"], observed=False)
            .size()
            .reset_index(name="count")
        )
        sample_totals = df_diagnostic.groupby(sample_col, observed=False).size()
        grouped["percentage"] = grouped.apply(
            lambda r: (r["count"] / sample_totals[r[sample_col]] * 100)
            if sample_totals[r[sample_col]] > 0
            else 0.0,
            axis=1,
        )
        grouped["name"] = grouped["case"].apply(
            lambda c: CASE_METADATA.get(str(c), {}).get("name", str(c))
        )
        return grouped

    counts = df_diagnostic["case"].value_counts(sort=False)
    summary_rows = []
    for case_label, count in counts.items():
        meta = CASE_METADATA.get(str(case_label), {})
        pct = (count / total_spots) * 100.0 if total_spots > 0 else 0.0
        summary_rows.append(
            {
                "case": str(case_label),
                "name": meta.get("name", str(case_label)),
                "branch": meta.get("branch", ""),
                "count": int(count),
                "percentage": round(float(pct), 2),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


# =============================================================================
# Pipeline / AnnData Integration Adapter
# =============================================================================
def topological_diagnostic_from_adata(
    adata: Any,
    cluster_key: str = "leiden",
    ground_truth_key: str = "ground_truth",
    embedding_key: str = "joint_feat",
    spatial_key: str = "spatial",
    adj_key: Optional[str] = None,
    n_neighbors: int = 6,
    tau_sim: float = 0.85,
    inplace: bool = True,
    unannotated_values: Optional[Set[Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adapter function to seamlessly run topological diagnostic on an AnnData object.

    Designed for compatibility with scgptpipeline, scgptspatialpipeline,
    smartpipeline, ucepipeline, and custom spatial multi-omics workflows.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    cluster_key : str, default='leiden'
        Key in `adata.obs` containing predicted cluster labels (e.g. 'leiden', 'kmeans', 'mclust').
    ground_truth_key : str, default='ground_truth'
        Key in `adata.obs` containing ground-truth domain annotations.
    embedding_key : str, default='joint_feat'
        Key in `adata.obsm` containing latent embeddings (e.g. 'joint_feat', 'feat', 'X_pca').
        If not found in `adata.obsm`, falls back to `adata.X`.
    spatial_key : str, default='spatial'
        Key in `adata.obsm` containing 2D spatial coordinates (or coordinates in `adata.obs[['x', 'y']]`).
    adj_key : Optional[str], default=None
        Key in `adata.uns` (e.g. 'adj' from SMART) or `adata.obsp` (e.g. 'spatial_connectivities').
        If None or not found, a KNN spatial graph is automatically constructed from spatial coordinates.
    n_neighbors : int, default=6
        Number of spatial neighbors if constructing KNN adjacency graph.
    tau_sim : float, default=0.85
        Cosine similarity threshold for diagnostic case routing.
    inplace : bool, default=True
        If True, stores S_G, S_C, S_N, and topological case into `adata.obs` and summary into `adata.uns`.
    unannotated_values : Optional[Set[Any]], default=None
        Custom set of label values to treat as unannotated.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (diagnostic_df, summary_df)
    """
    from sklearn.neighbors import kneighbors_graph

    N = adata.n_obs

    # 1. Extract Latent Embedding Matrix Z
    if embedding_key in adata.obsm:
        Z = adata.obsm[embedding_key]
    elif hasattr(adata, "X") and adata.X is not None:
        Z = adata.X
        if sp.issparse(Z):
            Z = Z.toarray()
    else:
        raise KeyError(f"Could not locate embedding '{embedding_key}' in adata.obsm or adata.X")

    if sp.issparse(Z):
        Z = Z.toarray()
    Z = np.asarray(Z, dtype=float)

    # 2. Extract or Construct Spatial Adjacency Matrix A
    A = None
    if adj_key is not None:
        if hasattr(adata, "uns") and adj_key in adata.uns:
            A = adata.uns[adj_key]
        elif hasattr(adata, "obsp") and adj_key in adata.obsp:
            A = adata.obsp[adj_key]

    if A is None:
        # Check standard pipeline locations (e.g., smartpipeline uns['adj'], scanpy obsp)
        if hasattr(adata, "uns") and "adj" in adata.uns and sp.issparse(adata.uns["adj"]):
            A = adata.uns["adj"]
        elif hasattr(adata, "obsp") and "spatial_connectivities" in adata.obsp:
            A = adata.obsp["spatial_connectivities"]
        elif hasattr(adata, "obsp") and "connectivities" in adata.obsp:
            A = adata.obsp["connectivities"]

    if A is None:
        # Build KNN adjacency from spatial coordinates
        coords = None
        if spatial_key in adata.obsm:
            coords = adata.obsm[spatial_key]
        elif "spatial_stereoseq" in adata.obsm:
            coords = adata.obsm["spatial_stereoseq"]
        elif "X_spatial" in adata.obsm:
            coords = adata.obsm["X_spatial"]
        else:
            for x_col, y_col in [
                ("x", "y"),
                ("X", "Y"),
                ("spatial_x", "spatial_y"),
                ("array_row", "array_col"),
            ]:
                if x_col in adata.obs and y_col in adata.obs:
                    coords = adata.obs[[x_col, y_col]].values
                    break

        if coords is not None:
            A = kneighbors_graph(
                coords,
                n_neighbors=n_neighbors,
                mode="connectivity",
                include_self=False,
            )
        else:
            logger.warning(
                "No spatial coordinates found. Creating empty spatial adjacency matrix."
            )
            A = sp.csr_matrix((N, N), dtype=float)

    # 3. Extract Labels
    if cluster_key in adata.obs:
        L_pred = adata.obs[cluster_key].values
    else:
        raise KeyError(f"Cluster key '{cluster_key}' not found in adata.obs")

    if ground_truth_key in adata.obs:
        L_true = adata.obs[ground_truth_key].values
    else:
        L_true = np.full(N, np.nan)

    spot_ids = adata.obs_names.values

    # 4. Run Topological Diagnostic
    diag_df = topological_diagnostic(
        Z=Z,
        A=A,
        L_pred=L_pred,
        L_true=L_true,
        tau_sim=tau_sim,
        spot_ids=spot_ids,
        unannotated_values=unannotated_values,
    )

    summary_df = summarize_topological_diagnostic(diag_df)

    # 5. Inplace Update
    if inplace:
        adata.obs["topological_case"] = diag_df["case"].values
        adata.obs["S_G"] = diag_df["S_G"].values
        adata.obs["S_C"] = diag_df["S_C"].values
        adata.obs["S_N"] = diag_df["S_N"].values
        adata.obs["topological_degree"] = diag_df["degree"].values
        adata.uns["topological_diagnostic_summary"] = summary_df

    return diag_df, summary_df


# =============================================================================
# Minimal Synthetic Self-Test / Demonstration Block
# =============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print("RUNNING TOPOLOGICAL DIAGNOSTIC SYNTHETIC DEMONSTRATION")
    print("=" * 78)

    # Construct a synthetic dataset explicitly engineering all 7 cases + edge cases
    # Latent dimension D = 4, threshold tau_sim = 0.85
    tau = 0.85
    rng = np.random.RandomState(42)

    # Prototype vectors for Domain A and Domain B
    v_A = np.array([1.0, 0.0, 0.0, 0.0])
    v_B = np.array([0.0, 1.0, 0.0, 0.0])
    v_C = np.array([0.0, 0.0, 1.0, 0.0])  # Distinct feature artifact direction
    v_noise = np.array([0.0, 0.0, 0.0, 1.0])

    embeddings = []
    l_true = []
    l_pred = []

    def make_vec(v, noise_scale=0.01):
        noisy = v + rng.randn(len(v)) * noise_scale
        return noisy / np.linalg.norm(noisy)

    # Spot 0: Case 1 (Homogeneous Core)
    embeddings.append(make_vec(v_A))
    l_true.append("A")
    l_pred.append("A")

    # Spot 1: Case 2 (Boundary Triumph)
    embeddings.append(make_vec(v_A))
    l_true.append("A")
    l_pred.append("A")

    # Spot 2: Case 7 (Lucky Guess)
    embeddings.append(make_vec(0.4 * v_A + 0.4 * v_B + 0.8 * v_noise))
    l_true.append("A")
    l_pred.append("A")

    # Spot 3: Case 2b (Neighbor-Assisted Correct: S_G < tau, S_N >= tau)
    v_2b = 0.5 * v_A + 0.866 * v_noise
    embeddings.append(make_vec(v_2b))
    l_true.append("A")
    l_pred.append("A")

    # Spot 4: Case 5 (Resolution Fracture: S_G >= tau, L_pred != L_true)
    embeddings.append(make_vec(v_A))
    l_true.append("A")
    l_pred.append("B")

    # Spot 5: Case 3 (Oversmoothed Victim: S_G < tau, S_N > S_G)
    embeddings.append(make_vec(0.4 * v_A + 0.9 * v_B))
    l_true.append("A")
    l_pred.append("B")

    # Spot 6: Case 6 (Feature Betrayal: S_C high >= tau, S_G low, S_N <= S_G)
    embeddings.append(make_vec(v_C))
    l_true.append("A")
    l_pred.append("C")

    # Spot 7: Case 4 (Feature Orphan: S_G low, S_N <= S_G, S_C low)
    embeddings.append(make_vec(v_noise))
    l_true.append("A")
    l_pred.append("B")

    # Spot 8: Isolated Spot (degree 0)
    embeddings.append(make_vec(v_A))
    l_true.append("A")
    l_pred.append("A")

    # Spot 9: Unannotated
    embeddings.append(make_vec(v_A))
    l_true.append(np.nan)
    l_pred.append("A")

    # Anchors:
    # Spot 10: Anchor for Domain A / Cluster A
    embeddings.append(make_vec(v_A))
    l_true.append("A")
    l_pred.append("A")

    # Spot 11: Anchor for Domain B / Cluster B
    embeddings.append(make_vec(v_B))
    l_true.append("B")
    l_pred.append("B")

    # Spot 12: Extra anchor for Cluster C to ensure cluster centroid aligns with v_C
    embeddings.append(make_vec(v_C))
    l_true.append("B")
    l_pred.append("C")

    # Spot 13: Twin of Spot 3 to ensure Spot 3's neighborhood has S_N >= tau
    embeddings.append(embeddings[3].copy())
    l_true.append("A")
    l_pred.append("A")

    Z_demo = np.array(embeddings)
    N_demo = len(Z_demo)
    L_true_demo = np.array(l_true)
    L_pred_demo = np.array(l_pred)

    # Build adjacency matrix A
    adj = np.zeros((N_demo, N_demo), dtype=float)

    # Spot 0 (Case 1) connected to anchor 10 (Domain A)
    adj[0, 10] = 1.0
    adj[10, 0] = 1.0

    # Spot 1 (Case 2) connected to anchor 11 (Domain B) -> S_N is low (< tau)
    adj[1, 11] = 1.0
    adj[11, 1] = 1.0

    # Spot 2 (Case 7) connected to anchor 11 -> S_N is low (< tau)
    adj[2, 11] = 1.0
    adj[11, 2] = 1.0

    # Spot 3 (Case 2b) connected to Spot 13 (its twin) -> S_N >= tau
    adj[3, 13] = 1.0
    adj[13, 3] = 1.0

    # Spot 4 (Case 5) connected to anchor 10
    adj[4, 10] = 1.0
    adj[10, 4] = 1.0

    # Spot 5 (Case 3) connected to anchor 11 (Domain B) -> S_N > S_G
    adj[5, 11] = 1.0
    adj[11, 5] = 1.0

    # Spot 6 (Case 6) connected to anchor 11 (Domain B) -> S_N is low
    adj[6, 11] = 1.0
    adj[11, 6] = 1.0

    # Spot 7 (Case 4) connected to anchor 11 -> S_N low
    adj[7, 11] = 1.0
    adj[11, 7] = 1.0

    # Spot 8 (Isolated Spot) has NO edges (degree 0)
    # Spot 9 (Unannotated) connected to anchor 10
    adj[9, 10] = 1.0
    adj[10, 9] = 1.0

    A_demo = sp.csr_matrix(adj)

    # Run diagnostic
    spot_barcodes = [f"Spot_{i:02d}" for i in range(N_demo)]
    df_results = topological_diagnostic(
        Z=Z_demo,
        A=A_demo,
        L_pred=L_pred_demo,
        L_true=L_true_demo,
        tau_sim=tau,
        spot_ids=spot_barcodes,
    )

    print("\n--- Diagnostic Results for Demonstration Spots ---")
    cols_to_show = ["spot_id", "L_pred", "L_true", "S_G", "S_C", "S_N", "degree", "case"]
    print(df_results.iloc[:10][cols_to_show].to_string(index=False))

    summary = summarize_topological_diagnostic(df_results)
    print("\n--- Diagnostic Summary Table ---")
    print(summary.to_string(index=False))
    print("\nDemonstration complete. Topological diagnostic is operational.")
