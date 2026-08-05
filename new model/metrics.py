"""
Evaluation & Spatial Early Stopping Metrics Module
==================================================

Computes:
1. Silhouette Score: Measures cluster separation and distinctness on latent embeddings Z_f.
2. Moran's I: Spatial autocorrelation coefficient measuring spatial coherence on graph/spot coordinates.
3. Composite Metric: lambda1 * Silhouette + lambda2 * Moran's I used for spatial early stopping on 5% spot subsample.
"""

import numpy as np
from sklearn.metrics import silhouette_score
import torch
from scipy.spatial.distance import cdist
from typing import Tuple, Optional, Union


def compute_silhouette(
    z_f: Union[torch.Tensor, np.ndarray],
    cluster_labels: Union[torch.Tensor, np.ndarray]
) -> float:
    """
    Computes the Silhouette Score on latent embeddings Z_f.

    Args:
        z_f: Latent embeddings (N, D)
        cluster_labels: Cluster assignment vector (N,)

    Returns:
        Silhouette score in range [-1, 1]
    """
    if isinstance(z_f, torch.Tensor):
        z_f = z_f.detach().cpu().numpy()
    if isinstance(cluster_labels, torch.Tensor):
        cluster_labels = cluster_labels.detach().cpu().numpy()

    num_clusters = len(np.unique(cluster_labels))
    if num_clusters <= 1 or num_clusters >= len(z_f):
        return -1.0

    try:
        score = silhouette_score(z_f, cluster_labels, metric='euclidean')
        return float(score)
    except Exception:
        return -1.0


def compute_morans_i(
    z_f: Union[torch.Tensor, np.ndarray],
    coords: Union[torch.Tensor, np.ndarray],
    k_spatial_neighbors: int = 6
) -> float:
    """
    Computes Moran's I spatial autocorrelation coefficient on latent embeddings z_f across spatial coordinates.

    Args:
        z_f: Latent embedding matrix (N, D)
        coords: Physical spot coordinates (N, 2)
        k_spatial_neighbors: Number of nearest neighbors to construct spatial weight matrix W

    Returns:
        Moran's I value (typically in [-1, 1], higher indicates higher spatial coherence)
    """
    if isinstance(z_f, torch.Tensor):
        z_f = z_f.detach().cpu().numpy()
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().cpu().numpy()

    N, D = z_f.shape
    if N < 5:
        return 0.0

    # Build inverse spatial distance weight matrix W
    dist_matrix = cdist(coords, coords, metric='euclidean')
    # Set diagonal to infinity to avoid division by zero
    np.fill_diagonal(dist_matrix, np.inf)

    # Keep k nearest neighbors per spot
    W = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        nearest_indices = np.argpartition(dist_matrix[i], k_spatial_neighbors)[:k_spatial_neighbors]
        W[i, nearest_indices] = 1.0 / (dist_matrix[i, nearest_indices] + 1e-6)

    # Make symmetric and row-normalize W
    W = (W + W.T) / 2.0
    W_sum = np.sum(W)
    if W_sum == 0:
        return 0.0

    # Compute Moran's I averaged over dimensions D
    moran_scores = []
    for d in range(min(D, 10)):  # Compute across up to top 10 dimensions for efficiency
        x = z_f[:, d]
        x_mean = np.mean(x)
        x_diff = x - x_mean
        denom = np.sum(x_diff ** 2) + 1e-8

        num = 0.0
        # W_ij * (x_i - mean) * (x_j - mean)
        num = np.sum(W * np.outer(x_diff, x_diff))

        moran_d = (N / W_sum) * (num / denom)
        moran_scores.append(moran_d)

    return float(np.mean(moran_scores))


def compute_composite_spatial_metric(
    z_f: Union[torch.Tensor, np.ndarray],
    cluster_labels: Union[torch.Tensor, np.ndarray],
    coords: Union[torch.Tensor, np.ndarray],
    lambda1: float = 0.5,
    lambda2: float = 0.5
) -> Tuple[float, float, float]:
    """
    Computes composite spatial evaluation metric: M = lambda1 * Silhouette + lambda2 * Moran's I

    Returns:
        composite_score, silhouette_score, morans_i
    """
    sil = compute_silhouette(z_f, cluster_labels)
    moran = compute_morans_i(z_f, coords)
    composite = lambda1 * sil + lambda2 * moran
    return composite, sil, moran
