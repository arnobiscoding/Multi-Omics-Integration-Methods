"""
Dual-Head Optimization: DEC Clustering Head & Pseudo-Label Contrastive Head
=============================================================================

1. DEC Clustering Head:
   - Learnable parameter matrix of K cluster centers C in R^(K x D).
   - Soft assignments Q: q_ij = (1 + ||z_i - C_j||^2)^(-1) / sum_k (1 + ||z_i - C_k||^2)^(-1)
   - Target distribution P: p_ij = (q_ij^2 / sum_i q_ij) / sum_k (q_ik^2 / sum_i q_ik)
   - Clustering Loss: L_DEC = sum_i sum_j p_ij log(p_ij / q_ij) (KL Divergence P || Q)

2. Pseudo-Label Spatially-Aware Contrastive Head:
   - Project Z_f through 2-layer MLP -> Z_proj.
   - Dynamic positive/negative masks based on soft assignments Q and confidence threshold tau = 0.95.
   - For neighbor (i, j) with A_ij = 1:
     * Strict Pos Mask M_pos(i,j) = 1 IF max(q_i) > tau AND max(q_j) > tau AND argmax(q_i) == argmax(q_j)
     * Strict Neg Mask M_neg(i,j) = 1 IF max(q_i) > tau AND max(q_j) > tau AND argmax(q_i) != argmax(q_j)
   - Masked InfoNCE loss pulling M_pos pairs together and penalizing M_neg pairs.
   - Also supports Spatial InfoNCE for Phase 1 warm-up.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
from typing import Tuple, Optional, Dict, Any


class DECHead(nn.Module):
    """
    Deep Embedded Clustering (DEC) Head maintaining learnable cluster centers C in R^(K x D).
    """
    def __init__(self, num_clusters: int, latent_dim: int, alpha: float = 1.0):
        super(DECHead, self).__init__()
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha  # Degrees of freedom for Student's t-distribution

        # Learnable cluster centers parameter
        self.cluster_centers = nn.Parameter(torch.Tensor(num_clusters, latent_dim))
        nn.init.xavier_uniform_(self.cluster_centers)

    def forward(self, z_f: torch.Tensor) -> torch.Tensor:
        """
        Computes soft cluster assignments Q for spot embeddings z_f.

        Args:
            z_f: Fused latent embedding (N, D)

        Returns:
            q: Soft cluster assignments probability matrix (N, K)
        """
        # Distances squared: ||z_i - C_j||^2 -> shape (N, K)
        dist_sq = torch.sum((z_f.unsqueeze(1) - self.cluster_centers.unsqueeze(0)) ** 2, dim=2)

        # Student's t-distribution kernel: q_ij = (1 + dist_sq / alpha)^(- (alpha+1)/2)
        numerator = 1.0 / (1.0 + dist_sq / self.alpha)
        if self.alpha != 1.0:
            numerator = torch.pow(numerator, (self.alpha + 1.0) / 2.0)

        # Normalize across clusters K
        q = numerator / (torch.sum(numerator, dim=1, keepdim=True) + 1e-8)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """
        Computes target distribution P from soft assignments Q:
        p_ij = (q_ij^2 / f_j) / sum_k (q_ik^2 / f_k) where f_j = sum_i q_ij

        Args:
            q: Soft assignment probabilities (N, K)

        Returns:
            p: Sharpened target distribution (N, K)
        """
        weight = q ** 2 / (torch.sum(q, dim=0, keepdim=True) + 1e-8)
        p = weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-8)
        return p.detach()

    @staticmethod
    def loss_dec(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Computes DEC clustering loss L_DEC = KL(P || Q) = sum_i sum_j p_ij * log(p_ij / q_ij)
        """
        loss = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=1)
        return torch.mean(loss)

    def init_centers_kmeans(self, z_f: torch.Tensor, random_state: int = 42) -> None:
        """
        Initializes cluster centers using K-Means++ algorithm on stabilized latent embedding z_f.
        """
        z_np = z_f.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_clusters, init='k-means++', n_init=20, random_state=random_state)
        kmeans.fit(z_np)
        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=z_f.device)
        self.cluster_centers.data.copy_(centers)


class PseudoLabelContrastiveHead(nn.Module):
    """
    Pseudo-Label Spatially-Aware Contrastive Head.
    Projects Z_f to Z_proj via 2-layer MLP and calculates masked InfoNCE loss.
    """
    def __init__(self, latent_dim: int, proj_dim: int = 64, temperature: float = 0.1, tau_threshold: float = 0.95):
        super(PseudoLabelContrastiveHead, self).__init__()
        self.temperature = temperature
        self.tau_threshold = tau_threshold

        # 2-layer MLP Projection Head
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim)
        )

    def forward(self, z_f: torch.Tensor) -> torch.Tensor:
        """
        Projects Z_f and l2-normalizes representations.
        """
        z_proj = self.projection(z_f)
        return F.normalize(z_proj, p=2, dim=1)

    def spatial_infonce_warmup(
        self,
        z_proj: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Phase 1 Warm-Up: Standard Spatially-Aware InfoNCE loss treating physical graph edges A_ij = 1
        as positive pairs.

        Args:
            z_proj: L2-normalized projections (N, proj_dim)
            edge_index: Spatial graph edge index (2, E)
        """
        row, col = edge_index[0], edge_index[1]
        sim_matrix = torch.matmul(z_proj, z_proj.T) / self.temperature # (N, N)

        # Positive pairs are physical neighbors in edge_index
        pos_sim = torch.exp(sim_matrix[row, col])

        # Denominator: sum of similarities to all nodes for central nodes
        exp_sim_matrix = torch.exp(sim_matrix)
        den_sim = exp_sim_matrix.sum(dim=1)[row]

        loss = -torch.log(pos_sim / (den_sim + 1e-8) + 1e-8)
        return torch.mean(loss)

    def pseudo_label_contrastive_loss(
        self,
        z_proj: torch.Tensor,
        q: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Phase 3: Pseudo-Label Spatially-Aware Contrastive Loss using soft assignments Q.

        Mask definitions for physical neighbor pair (i, j) with A_ij = 1:
        - M_pos(i,j) = 1 IF max(q_i) > tau AND max(q_j) > tau AND argmax(q_i) == argmax(q_j)
        - M_neg(i,j) = 1 IF max(q_i) > tau AND max(q_j) > tau AND argmax(q_i) != argmax(q_j)

        Args:
            z_proj: L2-normalized projections (N, proj_dim)
            q: Soft assignment probabilities (N, K)
            edge_index: Spatial graph edge index (2, E)
        """
        row, col = edge_index[0], edge_index[1]

        # Extract confidence scores and pseudo-label predictions
        max_q, pred_labels = torch.max(q, dim=1)

        # Filter edges by high-confidence spots
        conf_i = max_q[row] > self.tau_threshold
        conf_j = max_q[col] > self.tau_threshold
        high_conf_mask = conf_i & conf_j

        if not high_conf_mask.any():
            # Fallback to standard spatial InfoNCE if no pairs meet threshold tau
            return self.spatial_infonce_warmup(z_proj, edge_index)

        # Positive mask: same cluster prediction
        pos_mask = high_conf_mask & (pred_labels[row] == pred_labels[col])
        # Negative mask: different cluster prediction
        neg_mask = high_conf_mask & (pred_labels[row] != pred_labels[col])

        sim_matrix = torch.matmul(z_proj, z_proj.T) / self.temperature # (N, N)
        exp_sim = torch.exp(sim_matrix)

        loss_pos_list = []

        # Process positive pairs
        if pos_mask.any():
            row_pos = row[pos_mask]
            col_pos = col[pos_mask]

            pos_exp = torch.exp(sim_matrix[row_pos, col_pos])

            # For denominator: include negative pairs from high-confidence mask
            # Sum over row_pos
            den_exp = pos_exp.clone()
            for k_idx, r in enumerate(row_pos):
                neg_cols = col[neg_mask & (row == r)]
                if len(neg_cols) > 0:
                    den_exp[k_idx] = den_exp[k_idx] + torch.sum(exp_sim[r, neg_cols])

            loss_pos = -torch.log(pos_exp / (den_exp + 1e-8) + 1e-8)
            return torch.mean(loss_pos)
        else:
            return self.spatial_infonce_warmup(z_proj, edge_index)
