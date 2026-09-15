#!/usr/bin/env python3
"""
Progressive Multi-Omics Spatial Integration Pipeline: From SMART to ARISE
========================================================================
This pipeline benchmarks the 6 progressive hybrid models (M0 through M5)
across all 6 benchmark spatial multi-omics datasets with random seeds [42, 1234, 2024].

Progression Ladder:
-------------------
- M0: Base SMART (Spatial Graph, SAGEConv with L2 norm, Concat Fusion, MNN Triplet, PCA Recon)
- M1: Dual-Graph SMART (+ Dual Similarity/Distance Graphs + Common Overlap Graph)
- M2: Hierarchical SMART (+ 2-Stage Hierarchical MLP Fusion: Intra-RNA then Cross-Omics)
- M3: Contrastive SMART (+ Dense Spatial Contrastive Binary Cross-Entropy Loss, replacing MNN)
- M4: High-Dim Recon SMART (+ Multi-Head Decoders reconstructing raw 3000 HVGs + ADT/ATAC)
- M5: Full ARISE (+ Spectral GCNConv Backbone, Raw 3000 HVG input, Silhouette Checkpointing)
"""

import os
import sys
import argparse
import random
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns

import scanpy as sc
import anndata
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

try:
    from torch_geometric.nn import GCNConv, SAGEConv
except ImportError:
    from torch_geometric.nn import GCNConv
    class SAGEConv(nn.Module):
        """Native GraphSAGE fallback layer with optional L2 normalization."""
        def __init__(self, in_channels, out_channels, normalize=True):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.normalize = normalize
            self.lin_l = nn.Linear(in_channels, out_channels, bias=True)
            self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

        def forward(self, x, edge_index):
            row, col = edge_index
            num_nodes = x.size(0)
            deg = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
            deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
            deg = deg.clamp(min=1).unsqueeze(1)

            out = torch.zeros(num_nodes, self.in_channels, dtype=x.dtype, device=x.device)
            out.scatter_add_(0, row.unsqueeze(1).expand(-1, self.in_channels), x[col])
            out = out / deg

            out = self.lin_l(x) + self.lin_r(out)
            if self.normalize:
                out = F.normalize(out, p=2, dim=-1)
            return out

warnings.filterwarnings("ignore")

# ===========================================================================
# 1. SEED AND SYSTEM UTILITIES
# ===========================================================================
def set_seed(seed=2024):
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ===========================================================================
# 2. PREPROCESSING FUNCTIONS (ARISE & SMART)
# ===========================================================================
def clr_normalize_each_cell(adata, inplace=True):
    """Centered Log-Ratio (CLR) normalization on protein (ADT) data per cell."""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X))
    )
    return adata


def tfidf(X):
    """Compute TF-IDF matrix for sparse chromatin peak count matrix."""
    idf = X.shape[0] / X.sum(axis=0)
    if sp.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return sp.csr_matrix(tf.multiply(idf))
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def run_pca(X, n_comps=30):
    """Compute PCA projection matrix from array or sparse matrix."""
    pca_model = PCA(n_components=min(n_comps, X.shape[1], X.shape[0]))
    if sp.issparse(X):
        X = X.toarray()
    return pca_model.fit_transform(X)


def preprocess_dataset(adata_rna, adata_omics2, dataset_name, n_hvg=3000, n_comps_rna=30, n_comps_mod2=30):
    """
    Unified multi-omics preprocessing:
    Returns both high-dimensional expression matrices (ARISE target)
    and low-dimensional PCA projections (SMART target).
    """
    # 1. RNA Preprocessing
    adata_rna.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=n_hvg)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)

    hvg_mask = adata_rna.var['highly_variable']
    rna_raw = adata_rna[:, hvg_mask].X
    if sp.issparse(rna_raw):
        rna_raw = rna_raw.toarray()
    rna_pca = run_pca(rna_raw, n_comps=n_comps_rna)

    # 2. Second Modality Preprocessing (ADT Protein vs ATAC Chromatin)
    adata_omics2 = adata_omics2[adata_rna.obs_names].copy()
    if dataset_name.startswith("10x"):
        # Human Lymph Node: ADT Protein
        adata_omics2 = clr_normalize_each_cell(adata_omics2)
        sc.pp.scale(adata_omics2)
        mod2_raw = adata_omics2.X
        if sp.issparse(mod2_raw):
            mod2_raw = mod2_raw.toarray()
        mod2_pca = run_pca(mod2_raw, n_comps=min(n_comps_mod2, mod2_raw.shape[1]))
    else:
        # Mouse Brain: ATAC Chromatin
        adata_omics2.X = tfidf(adata_omics2.X)
        sc.pp.normalize_per_cell(adata_omics2, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_omics2)
        atac_comps = min(60, adata_omics2.shape[1])
        mod2_raw = run_pca(adata_omics2.X, n_comps=atac_comps)
        mod2_pca = mod2_raw[:, :min(n_comps_mod2, atac_comps)]

    return {
        'rna_raw': rna_raw.astype(np.float32),
        'mod2_raw': mod2_raw.astype(np.float32),
        'rna_pca': rna_pca.astype(np.float32),
        'mod2_pca': mod2_pca.astype(np.float32),
        'cell_positions': adata_rna.obsm['spatial'].astype(np.float32)
    }


# ===========================================================================
# 3. GRAPH BUILDERS & MNN TRIPLET MINING
# ===========================================================================
def build_spatial_knn_graph(cell_positions, num_neighbors=6, device='cpu'):
    """Build single physical coordinate spatial KNN graph (SMART convention)."""
    knn_graph = kneighbors_graph(cell_positions, n_neighbors=num_neighbors, mode='connectivity', include_self=False)
    knn_graph = knn_graph.maximum(knn_graph.T)
    edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long).to(device)
    return edge_index


def build_dual_graphs(rna_expression, cell_positions, num_neighbors=15, device='cpu'):
    """
    Build dual expression similarity and spatial proximity graphs plus their
    intersection (ARISE convention).
    """
    sim_mat = cosine_similarity(rna_expression)
    nbrs = NearestNeighbors(n_neighbors=num_neighbors + 1, metric='cosine').fit(rna_expression)
    _, indices = nbrs.kneighbors(rna_expression)

    adj_sim = np.zeros_like(sim_mat, dtype=int)
    for i in range(len(rna_expression)):
        for j in indices[i][1:]:
            adj_sim[i, j] = 1
            adj_sim[j, i] = 1

    sim_edge_index = torch.tensor(np.array(np.nonzero(adj_sim)), dtype=torch.long).to(device)
    sim_edge_weight = torch.tensor(sim_mat[adj_sim > 0], dtype=torch.float32).to(device)

    knn_graph = kneighbors_graph(cell_positions, n_neighbors=num_neighbors, mode='distance', include_self=False)
    knn_graph = knn_graph.maximum(knn_graph.T)

    dist_edge_index = torch.tensor(np.array(knn_graph.nonzero()), dtype=torch.long).to(device)
    dist_edge_weight = torch.tensor(knn_graph.data, dtype=torch.float32).to(device)

    sim_edges = set(zip(sim_edge_index[0].tolist(), sim_edge_index[1].tolist()))
    dist_edges = set(zip(dist_edge_index[0].tolist(), dist_edge_index[1].tolist()))
    common_edges = sim_edges.intersection(dist_edges)

    if len(common_edges) > 0:
        common_edge_index = torch.tensor(list(zip(*common_edges)), dtype=torch.long).to(device)
        common_edge_weight = torch.ones(common_edge_index.shape[1], dtype=torch.float32).to(device)
    else:
        common_edge_index = dist_edge_index
        common_edge_weight = torch.ones(dist_edge_index.shape[1], dtype=torch.float32).to(device)

    return (sim_edge_index, sim_edge_weight), (dist_edge_index, dist_edge_weight), (common_edge_index, common_edge_weight)


def mine_mnn_triplets(feature_matrix, n_nearest=3, farthest_ratio=0.6):
    """
    Mine Mutual Nearest Neighbor (MNN) triplets (anchors, positives, negatives)
    as implemented in SMART.
    """
    l = feature_matrix.shape[0]
    distances = pairwise_distances(feature_matrix)
    same_count = (distances == 0).sum(axis=1)

    sorted_idx = np.argsort(distances, axis=1).astype(np.int32)
    nearest_idx = []
    farthest_idx = []

    for i, j in enumerate(same_count):
        nearest_idx.append(sorted_idx[i, j:j + n_nearest])
        farthest_pool = max(1, int((l - j) * farthest_ratio))
        farthest_idx.append(sorted_idx[i, np.random.choice(np.arange(-farthest_pool, 0), n_nearest ** 2)])

    nn_dict = {i: set(nearest_idx[i]) for i in range(l)}
    anchors, positives, negatives = [], [], []

    for i, (nn_list, fn_list) in enumerate(zip(nearest_idx, farthest_idx)):
        for j in nn_list:
            if i in nn_dict.get(j, set()):
                anchors.append(i)
                positives.append(j)
                negatives.append(int(np.random.choice(fn_list, 1)[0]))

    return anchors, positives, negatives


# ===========================================================================
# 4. MODULAR GNN BUILDING BLOCKS
# ===========================================================================
class SAGEEncoderBlock(nn.Module):
    """2-Layer GraphSAGE encoder with L2 normalization (SMART)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True)
        self.conv2 = SAGEConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class SAGEDecoderBlock(nn.Module):
    """2-Layer GraphSAGE decoder for feature reconstruction (SMART)."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, in_channels, normalize=True)
        self.conv2 = SAGEConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


def cosine_similarity_matrix(emb):
    """Compute pairwise cosine similarity matrix with diagonal zeroed out (ARISE)."""
    mat = torch.matmul(emb, emb.T)
    norm = torch.norm(emb, p=2, dim=1).reshape((emb.shape[0], 1))
    mat = torch.div(mat, torch.matmul(norm, norm.T))
    mat = torch.where(torch.isnan(mat), torch.zeros_like(mat), mat)
    mat = mat - torch.diag_embed(torch.diag(mat))
    return mat


def compute_dense_spatial_contrastive_loss(emb, dist_edge_index, dist_edge_weight):
    """
    ARISE dense global spatial contrastive binary cross-entropy loss.
    Operates on spot cosine similarity matrix against spatial distance graph.
    """
    num_nodes = emb.size(0)
    graph_nei = torch.sparse_coo_tensor(
        dist_edge_index, torch.ones_like(dist_edge_weight), size=(num_nodes, num_nodes)
    ).coalesce().to_dense()
    graph_nei = torch.clamp(graph_nei, max=1.0)
    graph_neg = 1.0 - graph_nei

    sim_mat = torch.sigmoid(cosine_similarity_matrix(emb))

    neigh_loss = torch.mul(graph_nei, torch.log(sim_mat + 1e-10)).mean()
    neg_loss = torch.mul(graph_neg, torch.log(1.0 - sim_mat + 1e-10)).mean()
    return -(neigh_loss + neg_loss) / 2.0


# ===========================================================================
# 5. THE 6 PROGRESSIVE MODELS (M0 TO M5)
# ===========================================================================

# ----------------- Model 0: Base SMART -----------------
class Model0_BaseSMART(nn.Module):
    """
    M0: Base SMART Architecture
    - Input: PCA features
    - Topology: Single spatial coordinate graph E_spa
    - Backbone: SAGEConv with L2 normalization
    - Fusion: 1-stage concatenation + Linear FC
    - Decoders: SAGEConv decoders on PCA features
    - Regularization: MNN Triplet Margin Metric Loss
    """
    def __init__(self, in_rna_dim, in_mod2_dim, out_dim=64):
        super().__init__()
        self.enc_rna = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_mod2 = SAGEEncoderBlock(in_mod2_dim, out_dim)
        self.fc_fusion = nn.Linear(2 * out_dim, out_dim)

        self.dec_rna = SAGEDecoderBlock(out_dim, in_rna_dim)
        self.dec_mod2 = SAGEDecoderBlock(out_dim, in_mod2_dim)

    def forward(self, x_rna, x_mod2, edge_spa):
        h_rna = self.enc_rna(x_rna, edge_spa)
        h_mod2 = self.enc_mod2(x_mod2, edge_spa)

        z = self.fc_fusion(torch.cat([h_rna, h_mod2], dim=1))

        rec_rna = self.dec_rna(z, edge_spa)
        rec_mod2 = self.dec_mod2(z, edge_spa)
        return z, rec_rna, rec_mod2


# ----------------- Model 1: Dual-Graph SMART -----------------
class Model1_DualGraphSMART(nn.Module):
    """
    M1: SMART with ARISE Dual Graph Topology
    - Injects E_sim, E_dist, and E_common into SMART's SAGE backbone
    - RNA encodes along both similarity and distance graphs
    - Modality 2 encodes along the common intersection graph
    - 1-stage concatenation fusion + SAGE decoders on PCA features
    """
    def __init__(self, in_rna_dim, in_mod2_dim, out_dim=64):
        super().__init__()
        self.enc_sim = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_dist = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_mod2 = SAGEEncoderBlock(in_mod2_dim, out_dim)
        self.fc_fusion = nn.Linear(3 * out_dim, out_dim)

        self.dec_sim = SAGEDecoderBlock(out_dim, in_rna_dim)
        self.dec_dist = SAGEDecoderBlock(out_dim, in_rna_dim)
        self.dec_mod2 = SAGEDecoderBlock(out_dim, in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_common):
        h_sim = self.enc_sim(x_rna, e_sim)
        h_dist = self.enc_dist(x_rna, e_dist)
        h_pro = self.enc_mod2(x_mod2, e_common)

        z = self.fc_fusion(torch.cat([h_sim, h_dist, h_pro], dim=1))

        rec_sim = self.dec_sim(z, e_sim)
        rec_dist = self.dec_dist(z, e_dist)
        rec_mod2 = self.dec_mod2(z, e_common)
        return z, rec_sim, rec_dist, rec_mod2


# ----------------- Model 2: Hierarchical SMART -----------------
class Model2_HierarchicalSMART(nn.Module):
    """
    M2: Dual-Graph SMART with ARISE 2-Stage Hierarchical Fusion
    - Stage 1: Intra-RNA fusion of similarity and spatial embeddings -> Z_rna
    - Stage 2: Cross-modality fusion of Z_rna and Modality 2 -> Z_joint
    """
    def __init__(self, in_rna_dim, in_mod2_dim, out_dim=64):
        super().__init__()
        self.enc_sim = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_dist = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_mod2 = SAGEEncoderBlock(in_mod2_dim, out_dim)

        self.fusion1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim))
        self.fusion2 = nn.Linear(2 * out_dim, out_dim)

        self.dec_sim = SAGEDecoderBlock(out_dim, in_rna_dim)
        self.dec_dist = SAGEDecoderBlock(out_dim, in_rna_dim)
        self.dec_mod2 = SAGEDecoderBlock(out_dim, in_mod2_dim)
        self.dec_joint = nn.Linear(out_dim, in_rna_dim + in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_common):
        h_sim = self.enc_sim(x_rna, e_sim)
        h_dist = self.enc_dist(x_rna, e_dist)
        h_pro = self.enc_mod2(x_mod2, e_common)

        z_rna = self.fusion1(torch.cat([h_sim, h_dist], dim=1))
        z_joint = self.fusion2(torch.cat([z_rna, h_pro], dim=1))

        rec_sim = self.dec_sim(h_sim, e_sim)
        rec_dist = self.dec_dist(h_dist, e_dist)
        rec_mod2 = self.dec_mod2(h_pro, e_common)
        rec_joint = self.dec_joint(z_joint)

        return z_joint, z_rna, rec_joint, rec_sim, rec_dist, rec_mod2


# ----------------- Model 3: Contrastive SMART -----------------
class Model3_ContrastiveSMART(nn.Module):
    """
    M3: Dual-Graph Hierarchical SMART with Dense Spatial Contrastive BCE
    - Replaces MNN triplet margin loss with ARISE global dense spatial BCE loss
    - Retains low-dimensional PCA target efficiency
    """
    def __init__(self, in_rna_dim, in_mod2_dim, out_dim=64):
        super().__init__()
        self.enc_sim = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_dist = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_mod2 = SAGEEncoderBlock(in_mod2_dim, out_dim)

        self.fusion1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim))
        self.fusion2 = nn.Linear(2 * out_dim, out_dim)

        self.dec_shared = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU())
        self.dec_sim = nn.Linear(128, in_rna_dim)
        self.dec_dist = nn.Linear(128, in_rna_dim)
        self.dec_mod2 = nn.Linear(128, in_mod2_dim)
        self.dec_joint = nn.Linear(128, in_rna_dim + in_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_common):
        h_sim = self.enc_sim(x_rna, e_sim)
        h_dist = self.enc_dist(x_rna, e_dist)
        h_pro = self.enc_mod2(x_mod2, e_common)

        z_rna = self.fusion1(torch.cat([h_sim, h_dist], dim=1))
        z_joint = self.fusion2(torch.cat([z_rna, h_pro], dim=1))

        rec_sim = self.dec_sim(self.dec_shared(h_sim))
        rec_dist = self.dec_dist(self.dec_shared(h_dist))
        rec_mod2 = self.dec_mod2(self.dec_shared(h_pro))
        rec_joint = self.dec_joint(self.dec_shared(z_joint))

        return z_joint, z_rna, h_sim, h_dist, h_pro, rec_joint, rec_sim, rec_dist, rec_mod2


# ----------------- Model 4: High-Dim Direct Recon SMART -----------------
class Model4_HighDimSMART(nn.Module):
    """
    M4: SMART with High-Dimensional Raw Feature Reconstruction
    - Encoders take PCA/Raw features with SAGEConv
    - Decoders reconstruct original raw 3000 HVGs and raw ADT/ATAC features
    - Loss computed directly against full gene expression profiles
    """
    def __init__(self, in_rna_dim, in_mod2_dim, raw_rna_dim=3000, raw_mod2_dim=30, out_dim=64, hidden_dim=512):
        super().__init__()
        self.enc_sim = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_dist = SAGEEncoderBlock(in_rna_dim, out_dim)
        self.enc_mod2 = SAGEEncoderBlock(in_mod2_dim, out_dim)

        self.fusion1 = nn.Sequential(nn.Linear(2 * out_dim, out_dim))
        self.fusion2 = nn.Linear(2 * out_dim, out_dim)

        self.dec_shared = nn.Sequential(nn.Linear(out_dim, hidden_dim), nn.ReLU())
        self.dec_sim = nn.Linear(hidden_dim, raw_rna_dim)
        self.dec_dist = nn.Linear(hidden_dim, raw_rna_dim)
        self.dec_mod2 = nn.Linear(hidden_dim, raw_mod2_dim)
        self.dec_joint = nn.Linear(hidden_dim, raw_rna_dim + raw_mod2_dim)

    def forward(self, x_rna, x_mod2, e_sim, e_dist, e_common):
        h_sim = self.enc_sim(x_rna, e_sim)
        h_dist = self.enc_dist(x_rna, e_dist)
        h_pro = self.enc_mod2(x_mod2, e_common)

        z_rna = self.fusion1(torch.cat([h_sim, h_dist], dim=1))
        z_joint = self.fusion2(torch.cat([z_rna, h_pro], dim=1))

        rec_sim = self.dec_sim(self.dec_shared(h_sim))
        rec_dist = self.dec_dist(self.dec_shared(h_dist))
        rec_mod2 = self.dec_mod2(self.dec_shared(h_pro))
        rec_joint = self.dec_joint(self.dec_shared(z_joint))

        return z_joint, z_rna, h_sim, h_dist, h_pro, rec_joint, rec_sim, rec_dist, rec_mod2


# ----------------- Model 5: Full ARISE -----------------
class Model5_FullARISE(nn.Module):
    """
    M5: Full ARISE Architecture (Exact DualGCN & Dual model from reference ARISE)
    - Input: Raw 3000 HVGs directly
    - Backbone: Spectral GCNConv layers
    - Hierarchical 2-stage MLP fusion without non-linear clipping on stage 1
    - Global dense spatial contrastive BCE loss
    - Direct multi-target raw feature reconstruction
    - Parameter L1/L2 regularization
    - Cluster layer parameter
    """
    def __init__(self, in_channels, q, hidden_channels=512, out_channels=64, num_clusters=10,
                 beta=25.0, gamma=10.0, delta=1.0, dropout=0.0,
                 l1_lambda=1e-4, l2_lambda=1e-3):
        super().__init__()
        self.in_channels = in_channels
        self.q = q
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_clusters = num_clusters
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.dropout = dropout
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # RNA stream: similarity-based and distance-based GCN branches
        self.x_RNA1 = GCNConv(in_channels, hidden_channels)
        self.x_RNA2 = GCNConv(in_channels, hidden_channels)

        # ADT stream: initial embedding
        self.protein3 = GCNConv(q, out_channels)

        # Project RNA branches to embedding space
        self.sim_conv = GCNConv(hidden_channels, out_channels)
        self.dist_conv = GCNConv(hidden_channels, out_channels)

        # Fusion layers (exact ARISE: NO ReLU in fusion_layer1 or fusion_layer2)
        self.fusion_layer1 = nn.Sequential(nn.Linear(2 * out_channels, out_channels))
        self.fusion_layer2 = nn.Sequential(nn.Linear(2 * out_channels, out_channels))

        # Decoder layers for reconstruction
        self.deconv1 = nn.Linear(out_channels, hidden_channels)
        self.deconv2 = nn.Linear(hidden_channels, in_channels)
        self.deconv4 = nn.Linear(hidden_channels, q)
        self.deconv5 = nn.Linear(hidden_channels, q + in_channels)

        # Cluster parameter layer
        self.cluster_layer = nn.Parameter(torch.Tensor(num_clusters, out_channels))
        nn.init.xavier_uniform_(self.cluster_layer.data)

    def forward(self, x_rna, x_adt, e_sim, w_sim, e_dist, w_dist, e_common, w_common):
        xs = F.relu(self.x_RNA1(x_rna, e_sim, w_sim))
        xs = F.dropout(xs, self.dropout, training=self.training)

        xd = F.relu(self.x_RNA2(x_rna, e_dist, w_dist))
        xd = F.dropout(xd, self.dropout, training=self.training)

        x_sim = self.sim_conv(xs, e_sim, w_sim)
        x_dist = self.dist_conv(xd, e_dist, w_dist)
        pro = self.protein3(x_adt, e_common, w_common)

        combined = torch.cat([x_sim, x_dist], dim=1)
        fused = self.fusion_layer1(combined)

        combined_protein = torch.cat([fused, pro], dim=1)
        fused_pro = self.fusion_layer2(combined_protein)

        return x_sim, x_dist, fused, fused_pro, pro

    def reconstruct(self, z):
        return self.deconv2(F.relu(self.deconv1(z)))

    def reconstruct2(self, z):
        return self.deconv4(F.relu(self.deconv1(z)))

    def reconstruct3(self, z):
        return self.deconv5(F.relu(self.deconv1(z)))

    def compute_regularization_loss(self):
        l1_loss = sum(torch.sum(torch.abs(p)) for p in self.parameters())
        l2_loss = sum(torch.sum(p ** 2) for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss

    def compute_losses(self, x_rna, x_adt, sim_z, dist_z, fused_z, fused_pro, combined_raw, pro, dist_edge_index, dist_edge_weight):
        l_rec = F.mse_loss(combined_raw, self.reconstruct3(fused_pro))
        l_sim = F.mse_loss(x_rna, self.reconstruct(sim_z))
        l_dist = F.mse_loss(x_rna, self.reconstruct(dist_z))
        l_adt = F.mse_loss(x_adt, self.reconstruct2(pro))

        l_spatial = compute_dense_spatial_contrastive_loss(fused_z, dist_edge_index, dist_edge_weight)
        reg_loss = self.compute_regularization_loss()

        total_loss = self.beta * (l_rec + l_sim + l_dist + l_adt) + self.gamma * l_spatial + self.delta * reg_loss
        return total_loss, l_rec


# ===========================================================================
# 6. UNIFIED TRAINING AND EVALUATION ENGINE
# ===========================================================================
def train_and_evaluate_model(
    model_id: str,
    data_dict: dict,
    n_clusters: int,
    true_labels: np.ndarray,
    seed: int,
    epochs_smart: int = 500,
    epochs_arise: int = 350,
    lr: float = 1e-3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train any of the 6 progressive models (M0-M5) and return latent embeddings,
    cluster predictions, and quantitative metrics.
    """
    set_seed(seed)
    dev = torch.device(device)

    # Tensor preparation
    rna_raw = torch.tensor(data_dict['rna_raw'], dtype=torch.float32).to(dev)
    mod2_raw = torch.tensor(data_dict['mod2_raw'], dtype=torch.float32).to(dev)
    rna_pca = torch.tensor(data_dict['rna_pca'], dtype=torch.float32).to(dev)
    mod2_pca = torch.tensor(data_dict['mod2_pca'], dtype=torch.float32).to(dev)

    cell_positions = data_dict['cell_positions']
    edge_spa = build_spatial_knn_graph(cell_positions, num_neighbors=6, device=dev)
    (e_sim, w_sim), (e_dist, w_dist), (e_com, w_com) = build_dual_graphs(
        data_dict['rna_raw'], cell_positions, num_neighbors=15, device=dev
    )

    # ------------------ Model Instantiation & Training Loop ------------------
    if model_id == "M0":
        model = Model0_BaseSMART(rna_pca.shape[1], mod2_pca.shape[1]).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        anchors, positives, negatives = mine_mnn_triplets(data_dict['rna_pca'])
        triplet_fn = nn.TripletMarginLoss(margin=0.5, p=2)

        for epoch in range(epochs_smart):
            model.train()
            optimizer.zero_grad()
            z, rec_r, rec_m = model(rna_pca, mod2_pca, edge_spa)

            loss_rec = F.mse_loss(rna_pca, rec_r) + F.mse_loss(mod2_pca, rec_m)
            loss_tri = triplet_fn(z[anchors], z[positives], z[negatives]) if len(anchors) > 0 else 0.0
            loss = loss_rec + loss_tri
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_emb, _, _ = model(rna_pca, mod2_pca, edge_spa)
            final_emb = final_emb.cpu().numpy()

    elif model_id == "M1":
        model = Model1_DualGraphSMART(rna_pca.shape[1], mod2_pca.shape[1]).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        anchors, positives, negatives = mine_mnn_triplets(data_dict['rna_pca'])
        triplet_fn = nn.TripletMarginLoss(margin=0.5, p=2)

        for epoch in range(epochs_smart):
            model.train()
            optimizer.zero_grad()
            z, rec_sim, rec_dist, rec_mod = model(rna_pca, mod2_pca, e_sim, e_dist, e_com)

            loss_rec = F.mse_loss(rna_pca, rec_sim) + F.mse_loss(rna_pca, rec_dist) + F.mse_loss(mod2_pca, rec_mod)
            loss_tri = triplet_fn(z[anchors], z[positives], z[negatives]) if len(anchors) > 0 else 0.0
            loss = loss_rec + loss_tri
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_emb, _, _, _ = model(rna_pca, mod2_pca, e_sim, e_dist, e_com)
            final_emb = final_emb.cpu().numpy()

    elif model_id == "M2":
        model = Model2_HierarchicalSMART(rna_pca.shape[1], mod2_pca.shape[1]).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        anchors, positives, negatives = mine_mnn_triplets(data_dict['rna_pca'])
        triplet_fn = nn.TripletMarginLoss(margin=0.5, p=2)
        joint_pca = torch.cat([rna_pca, mod2_pca], dim=1)

        for epoch in range(epochs_smart):
            model.train()
            optimizer.zero_grad()
            z_joint, z_rna, rec_joint, rec_sim, rec_dist, rec_mod = model(rna_pca, mod2_pca, e_sim, e_dist, e_com)

            loss_rec = F.mse_loss(joint_pca, rec_joint) + F.mse_loss(rna_pca, rec_sim) + F.mse_loss(rna_pca, rec_dist) + F.mse_loss(mod2_pca, rec_mod)
            loss_tri = triplet_fn(z_joint[anchors], z_joint[positives], z_joint[negatives]) if len(anchors) > 0 else 0.0
            loss = loss_rec + loss_tri
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_emb, _, _, _, _, _ = model(rna_pca, mod2_pca, e_sim, e_dist, e_com)
            final_emb = final_emb.cpu().numpy()

    elif model_id == "M3":
        model = Model3_ContrastiveSMART(rna_pca.shape[1], mod2_pca.shape[1]).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        joint_pca = torch.cat([rna_pca, mod2_pca], dim=1)

        for epoch in range(epochs_smart):
            model.train()
            optimizer.zero_grad()
            z_joint, z_rna, h_sim, h_dist, h_pro, rec_joint, rec_sim, rec_dist, rec_mod = model(rna_pca, mod2_pca, e_sim, e_dist, e_com)

            loss_rec = F.mse_loss(joint_pca, rec_joint) + F.mse_loss(rna_pca, rec_sim) + F.mse_loss(rna_pca, rec_dist) + F.mse_loss(mod2_pca, rec_mod)
            loss_spatial = compute_dense_spatial_contrastive_loss(z_rna, e_dist, w_dist)
            loss = 10.0 * loss_rec + 5.0 * loss_spatial
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_emb, _, _, _, _, _, _, _, _ = model(rna_pca, mod2_pca, e_sim, e_dist, e_com)
            final_emb = final_emb.cpu().numpy()

    elif model_id == "M4":
        model = Model4_HighDimSMART(
            in_rna_dim=rna_pca.shape[1],
            in_mod2_dim=mod2_pca.shape[1],
            raw_rna_dim=rna_raw.shape[1],
            raw_mod2_dim=mod2_raw.shape[1]
        ).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        joint_raw = torch.cat([rna_raw, mod2_raw], dim=1)

        for epoch in range(epochs_arise):
            model.train()
            optimizer.zero_grad()
            z_joint, z_rna, h_sim, h_dist, h_pro, rec_joint, rec_sim, rec_dist, rec_mod = model(rna_pca, mod2_pca, e_sim, e_dist, e_com)

            loss_rec = F.mse_loss(joint_raw, rec_joint) + F.mse_loss(rna_raw, rec_sim) + F.mse_loss(rna_raw, rec_dist) + F.mse_loss(mod2_raw, rec_mod)
            loss_spatial = compute_dense_spatial_contrastive_loss(z_rna, e_dist, w_dist)
            loss = 25.0 * loss_rec + 10.0 * loss_spatial
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_emb, _, _, _, _, _, _, _, _ = model(rna_pca, mod2_pca, e_sim, e_dist, e_com)
            final_emb = final_emb.cpu().numpy()

    elif model_id == "M5":
        model = Model5_FullARISE(
            in_channels=rna_raw.shape[1],
            q=mod2_raw.shape[1],
            num_clusters=n_clusters,
            beta=25.0,
            gamma=10.0,
            delta=1.0,
            dropout=0.0
        ).to(dev)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        combined_raw = torch.cat([rna_raw, mod2_raw], dim=1)

        best_sil = -1.0
        best_emb = None
        best_labels = None

        for epoch in range(epochs_arise):
            optimizer.zero_grad()
            sim_z, dist_z, fused_z, fused_pro, pro = model(
                rna_raw, mod2_raw, e_sim, w_sim, e_dist, w_dist, e_com, w_com
            )

            loss, l_rec = model.compute_losses(
                rna_raw, mod2_raw,
                sim_z, dist_z, fused_z, fused_pro, combined_raw, pro,
                e_dist, w_dist
            )
            loss.backward()
            optimizer.step()

            # Canonical ARISE silhouette tracking: evaluated every single epoch
            model.eval()
            with torch.no_grad():
                _, _, _, eval_fused_pro, _ = model(
                    rna_raw, mod2_raw, e_sim, w_sim, e_dist, w_dist, e_com, w_com
                )
                curr_emb = eval_fused_pro.cpu().numpy()

            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            epoch_labels = km.fit_predict(curr_emb)
            sil = silhouette_score(curr_emb, epoch_labels)

            if sil > best_sil:
                best_sil = sil
                best_emb = curr_emb.copy()
                best_labels = epoch_labels.copy()

        final_emb = best_emb if best_emb is not None else curr_emb
        pred_labels = best_labels

    # ------------------ Clustering and Metric Evaluation ------------------
    if 'pred_labels' not in locals() or pred_labels is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        pred_labels = kmeans.fit_predict(final_emb)

    y_true = np.array(true_labels).astype(str)
    y_pred = pred_labels.astype(str)

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    v_meas = v_measure_score(y_true, y_pred)
    sil = silhouette_score(final_emb, pred_labels)
    chi = calinski_harabasz_score(final_emb, pred_labels)
    dbi = davies_bouldin_score(final_emb, pred_labels)

    metrics = {
        'ARI': float(ari),
        'NMI': float(nmi),
        'AMI': float(ami),
        'Homogeneity': float(homo),
        'V-measure': float(v_meas),
        'Silhouette': float(sil),
        'CHI': float(chi),
        'DBI': float(dbi)
    }

    return final_emb, pred_labels, metrics


# ===========================================================================
# 7. MAIN EXECUTION CONTROLLER
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Progressive SMART to ARISE Multi-Omics Pipeline")
    parser.add_argument('--models', nargs='+', default=['M0', 'M1', 'M2', 'M3', 'M4', 'M5'],
                        help="List of model IDs to benchmark: M0, M1, M2, M3, M4, M5")
    parser.add_argument('--datasets', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5],
                        help="Dataset indices to run (0 to 5)")
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 1234, 2024],
                        help="Random seeds for benchmark runs")
    parser.add_argument('--epochs_smart', type=int, default=500, help="Epochs for SMART-based models (M0-M3)")
    parser.add_argument('--epochs_arise', type=int, default=350, help="Epochs for ARISE-based models (M4-M5)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default=None, help="Root directory for saving all outputs")
    args = parser.parse_args()

    # Output Directory Setup
    if args.output_dir is not None:
        output_root = args.output_dir
    else:
        output_root = "/kaggle/working/progressive_ablation_results" if os.path.exists("/kaggle") else "./progressive_ablation_results"
    os.makedirs(output_root, exist_ok=True)

    # Dataset Root Search Paths
    KAGGLE_ROOT_CANDIDATES = [
        "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets",
        "/kaggle/input/multi-omics-datasets",
        "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue",
        "data",
        "."
    ]

    DATASET_CONFIGS = [
        {
            "name": "10x_human_lymph_node_A1",
            "type": "10x",
            "folder_candidates": ["10x_human_lymph_node_A1", "Human_Lymph_Node_A1", "Dataset11_Human_Lymph_Node_A1"],
            "other_file": "adata_ADT.h5ad",
            "anno_file": "annotation.csv",
            "gt_column": "manual-anno"
        },
        {
            "name": "10x_human_lymph_node_D1",
            "type": "10x",
            "folder_candidates": ["10x_human_lymph_node_D1", "Human_Lymph_Node_D1", "Dataset12_Human_Lymph_Node_D1"],
            "other_file": "adata_ADT.h5ad",
            "anno_file": "annotation.csv",
            "gt_column": "manual-anno"
        },
        {
            "name": "Mouse_Brain_E11_S1",
            "type": "Spatial-epigenome-transcriptome",
            "folder_candidates": ["Mouse_Brain_E11_S1", "Dataset7_Mouse_Brain_ATAC"],
            "other_file": "adata_ATAC.h5ad",
            "anno_file": "anno.csv",
            "gt_column": "cluster"
        },
        {
            "name": "Mouse_Brain_E13_S1",
            "type": "Spatial-epigenome-transcriptome",
            "folder_candidates": ["Mouse_Brain_E13_S1", "Dataset7_Mouse_Brain_ATAC"],
            "other_file": "adata_ATAC.h5ad",
            "anno_file": "anno.csv",
            "gt_column": "cluster"
        },
        {
            "name": "Mouse_Brain_E15_S1",
            "type": "Spatial-epigenome-transcriptome",
            "folder_candidates": ["Mouse_Brain_E15_S1", "Dataset7_Mouse_Brain_ATAC"],
            "other_file": "adata_ATAC.h5ad",
            "anno_file": "anno.csv",
            "gt_column": "cluster"
        },
        {
            "name": "Mouse_Brain_E18_S1",
            "type": "Spatial-epigenome-transcriptome",
            "folder_candidates": ["Mouse_Brain_E18_S1", "Dataset7_Mouse_Brain_ATAC"],
            "other_file": "adata_ATAC.h5ad",
            "anno_file": "anno.csv",
            "gt_column": "cluster"
        },
    ]

    all_benchmark_rows = []

    for d_idx in args.datasets:
        cfg = DATASET_CONFIGS[d_idx]
        d_name = cfg["name"]

        print("\n" + "="*80)
        print(f" LOADING DATASET: {d_name} ".center(80, "="))
        print("="*80)

        base_dir = None
        for root in KAGGLE_ROOT_CANDIDATES:
            for folder in cfg["folder_candidates"]:
                cand = os.path.join(root, folder)
                if os.path.isdir(cand):
                    base_dir = cand
                    break
            if base_dir is not None:
                break

        if base_dir is None:
            print(f"Warning: Could not find folder for dataset {d_name}. Skipping...")
            continue

        rna_path = os.path.join(base_dir, "adata_RNA.h5ad")
        other_path = os.path.join(base_dir, cfg["other_file"])
        anno_path = os.path.join(base_dir, cfg["anno_file"])

        if not (os.path.exists(rna_path) and os.path.exists(other_path) and os.path.exists(anno_path)):
            print(f"Warning: Missing required files in {base_dir}. Skipping...")
            continue

        adata_rna = sc.read_h5ad(rna_path)
        adata_mod2 = sc.read_h5ad(other_path)
        anno_df = pd.read_csv(anno_path, index_col=0)

        adata_rna.obs['ground_truth'] = anno_df[cfg["gt_column"]]
        valid_mask = adata_rna.obs['ground_truth'].notna()
        adata_rna = adata_rna[valid_mask].copy()
        adata_mod2 = adata_mod2[adata_rna.obs_names].copy()

        n_clusters = adata_rna.obs['ground_truth'].nunique()
        print(f"Dataset {d_name} loaded successfully: {adata_rna.n_obs} spots, {n_clusters} clusters.")

        # Unified Preprocessing
        data_dict = preprocess_dataset(adata_rna, adata_mod2, d_name)

        ds_out_dir = os.path.join(output_root, d_name)
        os.makedirs(ds_out_dir, exist_ok=True)
        os.makedirs(os.path.join(ds_out_dir, "plots"), exist_ok=True)

        ds_benchmark_rows = []

        # Loop over Models and Seeds
        for model_id in args.models:
            print("\n" + "-"*60)
            print(f" Running Architecture: {model_id} on {d_name} ".center(60, "-"))
            print("-"*60)

            for seed in args.seeds:
                print(f">> Executing {model_id} | Seed: {seed} ...")
                emb, labels, metrics = train_and_evaluate_model(
                    model_id=model_id,
                    data_dict=data_dict,
                    n_clusters=n_clusters,
                    true_labels=adata_rna.obs['ground_truth'].values,
                    seed=seed,
                    epochs_smart=args.epochs_smart,
                    epochs_arise=args.epochs_arise,
                    lr=args.lr,
                    device=args.device
                )

                print(f"   [{model_id} Seed {seed}] ARI: {metrics['ARI']:.4f} | NMI: {metrics['NMI']:.4f} | Sil: {metrics['Silhouette']:.4f}")

                # Save artifacts
                np.save(os.path.join(ds_out_dir, f"emb_{model_id}_seed_{seed}.npy"), emb)
                np.save(os.path.join(ds_out_dir, f"labels_{model_id}_seed_{seed}.npy"), labels)

                # Generate and save UMAP + Spatial plot
                try:
                    adata_plot = adata_rna.copy()
                    adata_plot.obsm['emb'] = emb
                    adata_plot.obs['pred'] = labels.astype(str)
                    sc.pp.neighbors(adata_plot, use_rep='emb', n_neighbors=10)
                    sc.tl.umap(adata_plot)

                    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
                    sc.pl.umap(adata_plot, color='pred', ax=axes[0], title=f'{model_id} (Seed {seed})', show=False, s=20)
                    sc.pl.embedding(adata_plot, basis='spatial', color='pred', ax=axes[1], title='Spatial Pred', show=False, s=20)
                    sc.pl.embedding(adata_plot, basis='spatial', color='ground_truth', ax=axes[2], title='Ground Truth', show=False, s=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(ds_out_dir, "plots", f"plot_{model_id}_seed_{seed}.png"), dpi=150)
                    plt.close(fig)
                except Exception as e:
                    print(f"Plotting warning for {model_id} seed {seed}: {e}")

                row = {
                    'dataset': d_name,
                    'model': model_id,
                    'seed': seed,
                    **metrics
                }
                ds_benchmark_rows.append(row)
                all_benchmark_rows.append(row)

        # Save dataset-level metrics
        if ds_benchmark_rows:
            df_ds = pd.DataFrame(ds_benchmark_rows)
            df_ds.to_csv(os.path.join(ds_out_dir, "metrics.csv"), index=False)

            # Generate Dataset Summary (Mean ± SD per Model)
            summary_ds = df_ds.groupby('model')[['ARI', 'NMI', 'AMI', 'Homogeneity', 'V-measure', 'Silhouette']].agg(['mean', 'std'])
            summary_ds.to_csv(os.path.join(ds_out_dir, "summary_mean_std.csv"))
            print(f"\nSaved dataset metrics and summary to: {ds_out_dir}")

    # ===========================================================================
    # 8. GLOBAL SUMMARY AND PROGRESSION BARPLOT EXPORT
    # ===========================================================================
    if all_benchmark_rows:
        df_all = pd.DataFrame(all_benchmark_rows)
        df_all.to_csv(os.path.join(output_root, "metrics_all_models_all_datasets_3_seeds.csv"), index=False)

        # Global Mean ± SD per Model across all datasets
        global_summary = df_all.groupby(['dataset', 'model'])[['ARI', 'NMI', 'Silhouette']].agg(['mean', 'std']).reset_index()
        global_summary.to_csv(os.path.join(output_root, "global_summary_mean_std.csv"), index=False)

        print("\n" + "="*80)
        print(" GLOBAL BENCHMARK SUMMARY (M0 -> M5 PROGRESSION) ".center(80, "="))
        print("="*80)
        print(global_summary.to_string())

        # Generate Progression Boxplot across Models
        try:
            plt.figure(figsize=(14, 6))
            sns.set_theme(style="whitegrid")
            ax = sns.boxplot(data=df_all, x='model', y='ARI', hue='dataset', palette='Set2', showfliers=False)
            plt.title("Stepwise Progression (M0 -> M5): ARI Across All Benchmark Datasets", fontsize=15, pad=12)
            plt.xlabel("Progressive Model", fontsize=12)
            plt.ylabel("Adjusted Rand Index (ARI)", fontsize=12)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_root, "global_progression_ari_boxplot.png"), dpi=300)
            plt.close()

            plt.figure(figsize=(14, 6))
            ax = sns.boxplot(data=df_all, x='model', y='NMI', hue='dataset', palette='Set2', showfliers=False)
            plt.title("Stepwise Progression (M0 -> M5): NMI Across All Benchmark Datasets", fontsize=15, pad=12)
            plt.xlabel("Progressive Model", fontsize=12)
            plt.ylabel("Normalized Mutual Information (NMI)", fontsize=12)
            plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
            plt.tight_layout()
            plt.savefig(os.path.join(output_root, "global_progression_nmi_boxplot.png"), dpi=300)
            plt.close()
            print(f"Saved global progression boxplots to: {output_root}")
        except Exception as e:
            print(f"Global plot export warning: {e}")

        print("\nBenchmarking complete! All results stored in:", output_root)


if __name__ == "__main__":
    main()
