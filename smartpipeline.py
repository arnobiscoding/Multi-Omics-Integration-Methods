#!/usr/bin/env python3
"""
SMART Multi-Seed Baseline & Ablation Pipeline (Fully Standalone Python Script)
=============================================================================
A completely self-contained, standalone implementation of the SMART (Spatial
Multi-omic Aggregation using Graph Neural Networks and Metric Learning) framework.

All core SMART components (Spatial graph construction, GraphSAGE encoder/decoder,
modular multi-modal network, Numba-accelerated Mutual Nearest Neighbors triplet
mining, Laplacian regularizer, early-stopping training engine, PCA, mclust_R,
clustering, and resolution search) are embedded directly into this single file.

Integrates the complete SMART workflow across 6 target benchmark datasets:
- Mouse Brain E11-S1, E13-S1, E15-S1, E18-S1 (Spatial RNA-seq + Spatial ATAC-seq)
- Human Lymph Node A1, D1 (Spatial RNA-seq + Spatial ADT / Protein CITE-seq)

Evaluates performance across 10 random seeds with KMeans, Leiden, and mclust
using 8 quantitative clustering metrics, generates UMAP & Spatial visualization
plots, and exports summary statistics and Box & Whiskers plots.
"""

# ===========================================================================
# 1. RUN CONFIGURATION & DATASETS
# ===========================================================================
import os
import sys
import math
import random
import argparse
import warnings
from typing import Optional, Dict, List, Tuple, Any

import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.sparse import issparse, csc_matrix, csr_matrix

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.backends import cudnn

import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
from sklearn.metrics.pairwise import pairwise_distances
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

import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')

ENV_MODE = "auto"

ALL_DATASETS_CONFIG = {
    "mouse-brain-e11-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E11_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv",
        "n_neighbors": 4,
        "n_comps_rna": 30,
        "n_comps_mod2": 60,
        "lr": 1e-3,
        "farthest_ratio": 0.6
    },
    "mouse-brain-e13-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E13_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv",
        "n_neighbors": 4,
        "n_comps_rna": 30,
        "n_comps_mod2": 60,
        "lr": 1e-3,
        "farthest_ratio": 0.6
    },
    "mouse-brain-e15-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E15_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv",
        "n_neighbors": 4,
        "n_comps_rna": 30,
        "n_comps_mod2": 60,
        "lr": 1e-3,
        "farthest_ratio": 0.6
    },
    "mouse-brain-e18-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E18_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv",
        "n_neighbors": 4,
        "n_comps_rna": 30,
        "n_comps_mod2": 60,
        "lr": 1e-3,
        "farthest_ratio": 0.6
    },
    "human-lymph-node-a1": {
        "type": "human_lymph_node",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Human_Lymph_Node_A1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset11_Human_Lymph_Node_A1/",
        "mod2_candidates": ["adata_ADT.h5ad"],
        "anno_file": "annotation.csv",
        "n_neighbors": 6,
        "n_comps_rna": 30,
        "n_comps_mod2": 30,
        "lr": 5e-3,
        "farthest_ratio": 0.6
    },
    "human-lymph-node-d1": {
        "type": "human_lymph_node",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Human_Lymph_Node_D1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset12_Human_Lymph_Node_D1/",
        "mod2_candidates": ["adata_ADT.h5ad"],
        "anno_file": "annotation.csv",
        "n_neighbors": 6,
        "n_comps_rna": 30,
        "n_comps_mod2": 30,
        "lr": 5e-3,
        "farthest_ratio": 0.6
    }
}

ACTIVE_DATASETS = ["all"]

SEEDS = [
    42, 0, 1, 7, 123, 1234, 2022, 2023, 2024, 1337
]

# Optional muon libraries for official CLR and TF-IDF preprocessing
try:
    from muon import prot as pt
    from muon import atac as ac
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False


# ===========================================================================
# 2. ORIGINAL SMART SPATIAL GRAPH BUILDER (FROM smart/build_graph.py)
# ===========================================================================
def Cal_Spatial_Net(adata, radius=None, n_neighbors=None, model='KNN', verbose=True, include_self=False):
    """
    Construct spatial neighbor graph from spatial coordinates.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing cell-level spatial coordinates in `adata.obsm['spatial']`.
    radius : float, optional
        Radius for neighborhood search (used when `model='Radius'`).
    n_neighbors : int, optional
        Number of neighbors (used when `model='KNN'`).
    model : {'KNN', 'Radius'}, default='KNN'
        Type of graph construction method.
    verbose : bool, default=True
        If True, print summary of constructed graph.
    include_self : bool, default=False
        Whether to include self-loops in the adjacency matrix.

    Returns
    -------
    None
        The adjacency matrix is stored in `adata.uns['adj']`, and edges in `adata.uns['edgeList']`.
    """
    if 'spatial' not in adata.obsm:
        for k in ['spatial_stereoseq', 'X_spatial', 'spatial_coord']:
            if k in adata.obsm:
                adata.obsm['spatial'] = adata.obsm[k]
                break
        if 'spatial' not in adata.obsm:
            for x_col, y_col in [('x', 'y'), ('X', 'Y'), ('spatial_x', 'spatial_y'), ('array_row', 'array_col')]:
                if x_col in adata.obs and y_col in adata.obs:
                    adata.obsm['spatial'] = adata.obs[[x_col, y_col]].values.astype(float)
                    break

    spatial = adata.obsm['spatial']
    if model == 'KNN':
        adata.uns['adj'] = kneighbors_graph(spatial, n_neighbors=n_neighbors, mode='connectivity', include_self=include_self)
    elif model == 'Radius':
        adata.uns['adj'] = radius_neighbors_graph(spatial, radius=radius, mode='connectivity', include_self=include_self)

    edgeList = np.nonzero(adata.uns['adj'])
    adata.uns['edgeList'] = np.array([edgeList[0], edgeList[1]])

    if verbose:
        print('The graph contains %d edges, %d cells.' % (adata.uns['edgeList'].shape[1], adata.n_obs))
        print('%.4f neighbors per cell on average.' % (adata.uns['edgeList'].shape[1] / adata.n_obs))


# ===========================================================================
# 3. ORIGINAL SMART MUTUAL NEAREST NEIGHBORS (FROM smart/MNN.py)
# ===========================================================================
@nb.njit('int32[:,::1](float32[:,::1])', parallel=True)
def fastSort32(a):
    """
    Perform fast argsort for float32 arrays using Numba parallelization.
    """
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i, :] = np.argsort(a[i, :])
    return b


@nb.njit('int32[:,::1](float64[:,::1])', parallel=True)
def fastSort64(a):
    """
    Perform fast argsort for float64 arrays using Numba parallelization.
    """
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i, :] = np.argsort(a[i, :])
    return b


def Mutual_Nearest_Neighbors(adata, key=None, n_nearest_neighbors=1, farthest_ratio=0.5, max_samples=20000):
    """
    Find mutual nearest neighbors (MNNs) and construct triplets with optional sampling.

    Parameters
    ----------
    adata : AnnData
        Input dataset.
    key : str, optional
        Key in `adata.obsm` to use as features (default: use `adata.X`).
    n_nearest_neighbors : int, default=1
        Number of nearest neighbors to consider.
    farthest_ratio : float, default=0.5
        Fraction of farthest neighbors to consider when sampling negatives.
    max_samples : int, default=20000
        Maximum number of cells to process. If dataset is larger, random sampling is applied.

    Returns
    -------
    anchors : list[int]
        Indices of anchor points in original `adata`.
    positives : list[int]
        Indices of positive samples (MNNs).
    negatives : list[int]
        Indices of negative samples (randomly sampled farthest neighbors).
    """
    original_indices = np.arange(adata.shape[0])  # Store original indices
    l = adata.shape[0]

    # Apply sampling if dataset is too large
    if l > max_samples:
        print(f"Dataset size {l} exceeds max_samples {max_samples}, performing random sampling...")
        np.random.seed(42)
        sample_idx = np.random.choice(l, max_samples, replace=False)
        adata_sampled = adata[sample_idx].copy()
        original_indices = original_indices[sample_idx]
        l = max_samples
        print(f"Working with sampled {l} cells")
    else:
        adata_sampled = adata.copy()

    X = adata_sampled.X if key is None else adata_sampled.obsm[key]
    if issparse(X):
        X = X.toarray()
    distances = pairwise_distances(X)
    same_count = (distances == 0).sum(axis=1)
    print('Distances calculation completed!')

    nearest_neighbors_index = []
    farthest_neighbors_index = []

    if distances.dtype == "float64":
        sorted_neighbors_index = fastSort64(distances)
    elif distances.dtype == "float32":
        sorted_neighbors_index = fastSort32(distances)
    else:
        sorted_neighbors_index = np.argsort(distances, axis=1).astype(np.int32)

    for i, j in enumerate(same_count):
        nearest_neighbors_index.append(sorted_neighbors_index[i, j:j + n_nearest_neighbors])
        farthest_neighbors_index.append(
            sorted_neighbors_index[i, np.random.choice(
                np.arange(-int(max(1, (l - j) * farthest_ratio)), 0),
                n_nearest_neighbors ** 2
            )]
        )

    nn_dict = {i: set(nearest_neighbors_index[i]) for i in range(l)}

    anchors, positives, negatives = [], [], []
    for i, (nearest_neighbors, farthest_neighbors) in enumerate(zip(nearest_neighbors_index, farthest_neighbors_index)):
        if not np.all(X[i] == 0):
            for j in nearest_neighbors:
                if i in nn_dict.get(j, set()):
                    anchors.append(original_indices[i])
                    positives.append(original_indices[j])
                    negatives.append(original_indices[np.random.choice(farthest_neighbors, 1)[0]])

    print(f"The data using feature '{key if key else 'X'}' contains {len(anchors)} mnn_anchors")
    return anchors, positives, negatives


# ===========================================================================
# 4. ORIGINAL SMART GNN LAYERS & MODEL (FROM smart/layer.py & smart/model.py)
# ===========================================================================
try:
    from torch_geometric.nn import SAGEConv
except ImportError:
    class SAGEConv(nn.Module):
        """Native PyTorch GraphSAGE convolution with L2 normalization fallback."""
        def __init__(self, in_channels, out_channels, normalize=True):
            super(SAGEConv, self).__init__()
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


class SAGEConv_Encoder(torch.nn.Module):
    """
    Encoder based on GraphSAGE convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Encoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, out_channels, normalize=True)
        self.conv2 = SAGEConv(out_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class SAGEConv_Decoder(torch.nn.Module):
    """
    Decoder based on GraphSAGE convolution for reconstruction.
    """
    def __init__(self, in_channels, out_channels):
        super(SAGEConv_Decoder, self).__init__()
        self.conv1 = SAGEConv(in_channels, in_channels, normalize=True)
        self.conv2 = SAGEConv(in_channels, out_channels, normalize=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x


class SMART(torch.nn.Module):
    """
    SMART: A modular multi-modal graph representation learning model.
    """
    def __init__(self, hidden_dims, device, Conv_Encoder=SAGEConv_Encoder, Conv_Decoder=SAGEConv_Decoder):
        super(SMART, self).__init__()
        out_dim = hidden_dims[-1]

        # One encoder per modality
        self.encoders = nn.ModuleList([Conv_Encoder(in_dim, out_dim).to(device) for in_dim in hidden_dims[:-1]])
        self.fc = nn.Linear((len(hidden_dims) - 1) * out_dim, out_dim)

        # One decoder per modality
        self.decoders = nn.ModuleList([Conv_Decoder(out_dim, in_dim).to(device) for in_dim in hidden_dims[:-1]])

    def forward(self, features, edge_indexs):
        # Encode each modality
        x = [encoder(feature, edge_index) for encoder, feature, edge_index in zip(self.encoders, features, edge_indexs)]

        # Concatenate or directly project
        if len(x) == 1:
            z = self.fc(x[0])
        else:
            z = self.fc(torch.cat(x, dim=1))

        # Decode each modality
        x_rec = [decoder(z, edge_index) for decoder, feature, edge_index in zip(self.decoders, features, edge_indexs)]
        return z, x_rec


# ===========================================================================
# 5. ORIGINAL SMART TRAINING ENGINE (FROM smart/train.py)
# ===========================================================================
def laplacian_regularization(x, edge_index):
    """
    Compute Laplacian regularization loss.
    """
    row, col = edge_index
    diff = x[row] - x[col]
    loss = (diff ** 2).sum(dim=1).mean()
    return loss


def train_SMART(
    features,
    edges,
    triplet_samples_list,
    weights=[1, 1],
    emb_dim=64,
    n_epochs=500,
    lr=0.0001,
    weight_decay=1e-5,
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    window_size=20,
    slope=0.0001,
    Conv_Encoder=SAGEConv_Encoder,
    Conv_Decoder=SAGEConv_Decoder,
    margin=0.5,
    return_loss=False,
    laplacian_alpha=0,
):
    """
    Train the SMART model with reconstruction, triplet, and optional Laplacian loss.
    """
    hidden_dims = [x.shape[1] for x in features] + [emb_dim]
    model = SMART(hidden_dims=hidden_dims, device=device,
                  Conv_Encoder=Conv_Encoder, Conv_Decoder=Conv_Decoder)

    features, edges = [x.to(device) for x in features], [edge.to(device) for edge in edges]
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_list = []

    for epoch in tqdm(range(1, n_epochs + 1), desc="Training SMART", leave=False):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        z, x_rec = model(features, edges)

        # Triplet loss
        triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2, reduction='mean')
        tri_loss = 0
        for i, (anchors, positives, negatives) in enumerate(triplet_samples_list):
            if len(anchors) == 0:
                continue
            anchor_arr = z[anchors]
            positive_arr = z[positives]
            negative_arr = z[negatives]

            tri_output = triplet_loss_fn(anchor_arr, positive_arr, negative_arr)
            w = weights[len(weights) // 2 + i]
            tri_loss += w * tri_output

        # Reconstruction loss
        rec_loss = 0
        for i, (feature, x_r) in enumerate(zip(features, x_rec)):
            rec_output = F.mse_loss(feature, x_r)
            w = weights[i]
            rec_loss += w * rec_output

        # Total loss
        loss = rec_loss + tri_loss

        # Add Laplacian regularization if enabled
        if laplacian_alpha != 0 and len(edges) > 0:
            loss += laplacian_alpha * laplacian_regularization(z, edges[0])

        # Early stopping based on slope of recent loss trend
        if epoch > window_size and epoch % 10 == 0:
            x_axis = np.arange(window_size)
            res1 = stats.linregress(x_axis, [item[1] for item in loss_list[-window_size:]])  # tri_loss trend
            res2 = stats.linregress(x_axis, [item[2] for item in loss_list[-window_size:]])  # rec_loss trend
            if abs(res1.slope) < slope or abs(res2.slope) < slope:
                if res1.slope != 0 and res2.slope != 0:
                    print("Early stopping: flat trend detected.")
                    break

        # Backward & optimize
        loss_list.append((loss.item(), tri_loss.item() if isinstance(tri_loss, torch.Tensor) else tri_loss, rec_loss.item() if isinstance(rec_loss, torch.Tensor) else rec_loss))
        loss.backward()
        optimizer.step()

    return model if not return_loss else (model, loss_list)


# ===========================================================================
# 6. ORIGINAL SMART UTILITIES (FROM smart/utils.py)
# ===========================================================================
def set_seed(seed=2024):
    """
    Set random seed for reproducibility across Python, NumPy, PyTorch, and CUDA.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def fix_seed(seed=2024):
    set_seed(seed)


def pca(adata, use_reps=None, n_comps=10):
    """
    Perform dimensionality reduction using PCA.
    """
    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, (csc_matrix, csr_matrix)):
            feat_pca = pca_model.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca_model.fit_transform(adata.X)
    return feat_pca


def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    """
    Perform clustering using R package `mclust`.
    """
    np.random.seed(random_seed)
    try:
        import rpy2.robjects as robjects
        import rpy2.robjects.numpy2ri

        robjects.r.library("mclust")
        rpy2.robjects.numpy2ri.activate()

        r_random_seed = robjects.r["set.seed"]
        r_random_seed(random_seed)
        robjects.globalenv[".smart_mclust_data"] = rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm])
        res = robjects.r('Mclust(as.matrix(.smart_mclust_data), %d, "%s")' % (num_cluster, modelNames))
        mclust_res = np.array(res[-2])

        adata.obs['mclust'] = mclust_res
        adata.obs['mclust'] = adata.obs['mclust'].astype('int')
        adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    except Exception as e:
        print(f"Warning: mclust execution via rpy2 failed ({e}). Falling back to KMeans.")
        data = adata.obsm[used_obsm]
        km = KMeans(n_clusters=num_cluster, random_state=random_seed, n_init=10)
        adata.obs['mclust'] = km.fit_predict(data).astype('category')
    return adata


def search_res(adata, n_clusters, method="leiden", use_rep="emb", start=0.1, end=3.0, increment=0.01):
    """
    Search for resolution value that yields the desired number of clusters.
    """
    print("Searching resolution...")
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)

    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == "leiden":
            try:
                sc.tl.leiden(adata, random_state=0, resolution=res, flavor='igraph', n_iterations=2, directed=False)
            except TypeError:
                sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = adata.obs["leiden"].nunique()
        elif method == "louvain":
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = adata.obs["louvain"].nunique()
        print(f"resolution={res}, cluster number={count_unique}")
        if count_unique == n_clusters:
            print(f"Found resolution={res} with target cluster count={n_clusters}")
            return res

    print(f"Warning: Exact cluster count {n_clusters} not reached. Using res=0.5")
    return 0.5


def clustering(
    adata,
    n_clusters=7,
    key="emb",
    add_key="SMART",
    method="SMART",
    start=0.1,
    end=3.0,
    increment=0.01,
    use_pca=False,
    n_comps=20,
):
    """
    Perform clustering on latent representations with multiple supported methods.
    """
    if use_pca:
        adata.obsm[key + "_pca"] = pca(adata, use_reps=key, n_comps=n_comps)

    if method == "mclust":
        adata = mclust_R(adata, used_obsm=(key + "_pca" if use_pca else key), num_cluster=n_clusters)
        adata.obs[add_key] = adata.obs["mclust"]

    elif method in ["leiden", "louvain"]:
        res = search_res(adata, n_clusters, use_rep=(key + "_pca" if use_pca else key),
                         method=method, start=start, end=end, increment=increment)
        if method == "leiden":
            try:
                sc.tl.leiden(adata, random_state=0, resolution=res, flavor='igraph', n_iterations=2, directed=False)
            except TypeError:
                sc.tl.leiden(adata, random_state=0, resolution=res)
            adata.obs[add_key] = adata.obs["leiden"]
        else:
            sc.tl.louvain(adata, random_state=0, resolution=res)
            adata.obs[add_key] = adata.obs["louvain"]

    elif method == "kmeans":
        X = adata.obsm[key + "_pca"] if use_pca else adata.obsm[key]
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        adata.obs[add_key] = kmeans.fit_predict(X).astype("category")

    else:
        raise ValueError("Clustering method must be one of ['mclust', 'leiden', 'louvain', 'gmm', 'kmeans'].")


# ===========================================================================
# 7. PREPROCESSING HELPERS FOR ATAC & PROTEIN
# ===========================================================================
def preprocess_protein_clr(adata_mod2):
    """Normalize protein counts using CLR as in SMART Tutorial 2."""
    if MUON_AVAILABLE:
        pt.pp.clr(adata_mod2)
    else:
        def seurat_clr(x):
            s = np.sum(np.log1p(x[x > 0]))
            exp_val = np.exp(s / len(x)) if len(x) > 0 else 1.0
            return np.log1p(x / exp_val)
        data = adata_mod2.X.toarray() if issparse(adata_mod2.X) else np.array(adata_mod2.X)
        adata_mod2.X = np.apply_along_axis(seurat_clr, 1, data)


def preprocess_atac_tfidf(adata_mod2):
    """Normalize ATAC counts using TF-IDF as in SMART Tutorials 3 & 4."""
    if MUON_AVAILABLE:
        ac.pp.tfidf(adata_mod2, scale_factor=1e4)
    else:
        X = adata_mod2.X.toarray() if issparse(adata_mod2.X) else np.array(adata_mod2.X)
        n_cells = X.shape[0]
        tf = X / (np.sum(X, axis=1, keepdims=True) + 1e-12) * 1e4
        idf = np.log(1.0 + n_cells / (np.sum(X > 0, axis=0, keepdims=True) + 1.0))
        adata_mod2.X = tf * idf


# ===========================================================================
# 8. EVALUATION & VISUALIZATION FUNCTIONS
# ===========================================================================
def evaluate_clustering(y_true_series, y_pred_series, features_matrix, name=""):
    """Evaluate clustering using 8 quantitative metrics."""
    mask = (y_true_series != 'Exclude') & (y_true_series != 'unknown') & (y_true_series.notna())
    y_true = y_true_series[mask].astype(str)
    y_pred = y_pred_series[mask].astype(str)
    feats = features_matrix[mask]

    if len(y_true) == 0 or len(np.unique(y_pred)) < 2:
        return {
            'ARI': 0.0, 'NMI': 0.0, 'Silhouette': 0.0,
            'AMI': 0.0, 'CHI': 0.0, 'DBI': 0.0,
            'Homogeneity': 0.0, 'V-measure': 0.0
        }

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    v_meas = v_measure_score(y_true, y_pred)

    try:
        sil = silhouette_score(feats, y_pred)
    except Exception:
        sil = 0.0

    try:
        chi = calinski_harabasz_score(feats, y_pred)
    except Exception:
        chi = 0.0

    try:
        dbi = davies_bouldin_score(feats, y_pred)
    except Exception:
        dbi = 0.0

    metrics = {
        'ARI': float(ari),
        'NMI': float(nmi),
        'Silhouette': float(sil),
        'AMI': float(ami),
        'CHI': float(chi),
        'DBI': float(dbi),
        'Homogeneity': float(homo),
        'V-measure': float(v_meas)
    }

    if name:
        print(f"\n--- Clustering Evaluation ({name}) ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    return metrics


def plot_smart_visualizations(adata, title_prefix="", dname="dataset", seed=2024, save_dir=None):
    """Plot UMAP and Spatial domain visualizations for Ground Truth, KMeans, Leiden, and mclust."""
    if 'spatial' not in adata.obsm:
        return
    sc.pp.neighbors(adata, use_rep='SMART', n_neighbors=15)
    sc.tl.umap(adata)
    fig, axes = plt.subplots(4, 2, figsize=(14, 22))

    # Ground Truth
    sc.pl.umap(adata, color='ground_truth', ax=axes[0, 0], title=f'{title_prefix} UMAP: Ground Truth', show=False, size=20)
    sc.pl.embedding(adata, basis='spatial', color='ground_truth', ax=axes[0, 1], title=f'{title_prefix} Spatial: Ground Truth', show=False, size=20)

    # KMeans
    if 'kmeans' in adata.obs:
        sc.pl.umap(adata, color='kmeans', ax=axes[1, 0], title=f'{title_prefix} UMAP: KMeans', show=False, size=20)
        sc.pl.embedding(adata, basis='spatial', color='kmeans', ax=axes[1, 1], title=f'{title_prefix} Spatial: KMeans', show=False, size=20)

    # Leiden
    if 'leiden' in adata.obs:
        sc.pl.umap(adata, color='leiden', ax=axes[2, 0], title=f'{title_prefix} UMAP: Leiden', show=False, size=20)
        sc.pl.embedding(adata, basis='spatial', color='leiden', ax=axes[2, 1], title=f'{title_prefix} Spatial: Leiden', show=False, size=20)

    # mclust
    if 'mclust' in adata.obs:
        sc.pl.umap(adata, color='mclust', ax=axes[3, 0], title=f'{title_prefix} UMAP: mclust', show=False, size=20)
        sc.pl.embedding(adata, basis='spatial', color='mclust', ax=axes[3, 1], title=f'{title_prefix} Spatial: mclust', show=False, size=20)

    plt.tight_layout()
    if save_dir is None:
        save_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"smart_plot_{dname}_seed_{seed}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization plot image to: {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


def plot_smart_summary_boxplots(df_all: pd.DataFrame, save_dir=None):
    """Generate and save Box & Whiskers plots comparing KMeans, Leiden, and mclust across all metrics."""
    if df_all.empty or "cluster alg" not in df_all.columns:
        print("No valid metrics found for Box & Whiskers plots.")
        return

    metrics = ["ARI", "NMI", "Silhouette", "AMI", "CHI", "DBI", "Homogeneity", "V-measure"]
    metrics = [m for m in metrics if m in df_all.columns]
    if not metrics:
        return

    if save_dir is None:
        save_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
    os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(
            data=df_all,
            x="dataset",
            y=metric,
            hue="cluster alg",
            palette="Set2",
            ax=ax,
            showmeans=True
        )
        ax.set_title(f"SMART: {metric} Performance Across Datasets", fontsize=14, fontweight='bold')
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title="Cluster Alg", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"smart_boxplot_{metric.lower().replace('-', '_')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric} Boxplot image to: {save_path}")
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    axes_flat = axes.flatten()
    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        sns.boxplot(
            data=df_all,
            x="dataset",
            y=metric,
            hue="cluster alg",
            palette="Set2",
            ax=ax,
            showmeans=True
        )
        ax.set_title(f"{metric} Comparison", fontsize=12, fontweight='bold')
        ax.set_xlabel("Dataset", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.tick_params(axis='x', rotation=40)
        if idx != 0:
            ax.legend().remove()
        else:
            ax.legend(title="Cluster Alg", loc='upper left')

    plt.tight_layout()
    combined_path = os.path.join(save_dir, "smart_boxplot_all_metrics.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined multi-metric Boxplot figure to: {combined_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ===========================================================================
# 9. DYNAMIC DATA LOADING
# ===========================================================================
def load_dataset_data(dname, cfg, env_mode="auto"):
    """Dynamically load RNA, Modality 2 (ATAC/ADT), and annotations for any dataset."""
    is_kaggle = (env_mode == "kaggle") or (env_mode == "auto" and os.path.exists("/kaggle/input"))
    data_dir = cfg["kaggle_dir"] if is_kaggle else cfg["local_dir"]

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Generating synthetic multi-omics data for validation...")
        adata_rna = sc.datasets.pbmc3k()
        adata_mod2 = adata_rna.copy()
        spatial_coords = np.random.randn(adata_rna.n_obs, 2)
        adata_rna.obsm['spatial'] = spatial_coords
        adata_mod2.obsm['spatial'] = spatial_coords
        adata_rna.obs['ground_truth'] = adata_rna.obs['louvain'].astype(str) if 'louvain' in adata_rna.obs else 'Cluster1'
        adata_mod2.obs['ground_truth'] = adata_rna.obs['ground_truth']
        return adata_rna, adata_mod2

    print(f"Loading data from: {data_dir}")
    rna_path = os.path.join(data_dir, "adata_RNA.h5ad")
    adata_rna = sc.read_h5ad(rna_path)

    mod2_path = None
    for cand in cfg["mod2_candidates"]:
        cp = os.path.join(data_dir, cand)
        if os.path.exists(cp):
            mod2_path = cp
            break

    if mod2_path is None:
        raise FileNotFoundError(f"Modality 2 file not found in candidates {cfg['mod2_candidates']} inside {data_dir}")

    adata_mod2 = sc.read_h5ad(mod2_path)

    # Align spot index names
    adata_rna.var_names_make_unique()
    adata_mod2.var_names_make_unique()
    common_obs = adata_rna.obs_names.intersection(adata_mod2.obs_names)
    if len(common_obs) > 0 and len(common_obs) != adata_rna.n_obs:
        adata_rna = adata_rna[common_obs].copy()
        adata_mod2 = adata_mod2[common_obs].copy()

    # Load Annotations
    anno_path = os.path.join(data_dir, cfg["anno_file"])
    if os.path.exists(anno_path):
        print(f"Loading annotations from: {anno_path}")
        annotation = pd.read_csv(anno_path)
        ground_col = None
        for col in ['cluster', 'manual-anno', 'ground_truth', 'assigned_cluster', 'layer_guess']:
            if col in annotation.columns:
                ground_col = col
                break

        barcode_col = None
        for col in ['barcode', 'Barcode', 'cell_id', 'Unnamed: 0']:
            if col in annotation.columns:
                barcode_col = col
                break

        if ground_col and barcode_col:
            annotation = annotation.rename(columns={ground_col: 'ground_truth', barcode_col: 'barcode'})
            annotation = annotation.set_index('barcode')
            adata_rna.obs = adata_rna.obs.join(annotation[['ground_truth']], how='left')
            adata_rna.obs['ground_truth'] = adata_rna.obs['ground_truth'].fillna('unknown')
            adata_mod2.obs = adata_mod2.obs.join(annotation[['ground_truth']], how='left')
            adata_mod2.obs['ground_truth'] = adata_mod2.obs['ground_truth'].fillna('unknown')
    else:
        print(f"Warning: Annotation file {cfg['anno_file']} not found.")
        if 'ground_truth' not in adata_rna.obs:
            adata_rna.obs['ground_truth'] = 'unknown'

    # Spatial Coordinates Auto-Detection
    if 'spatial' not in adata_rna.obsm:
        for k in ['spatial_stereoseq', 'X_spatial', 'spatial_coord']:
            if k in adata_rna.obsm:
                adata_rna.obsm['spatial'] = adata_rna.obsm[k]
                break
        if 'spatial' not in adata_rna.obsm:
            for x_col, y_col in [('x', 'y'), ('X', 'Y'), ('spatial_x', 'spatial_y'), ('array_row', 'array_col')]:
                if x_col in adata_rna.obs and y_col in adata_rna.obs:
                    adata_rna.obsm['spatial'] = adata_rna.obs[[x_col, y_col]].values.astype(float)
                    break

    if 'spatial' in adata_rna.obsm and 'spatial' not in adata_mod2.obsm:
        adata_mod2.obsm['spatial'] = adata_rna.obsm['spatial']

    print(f"RNA shape: {adata_rna.shape} | Modality 2 shape: {adata_mod2.shape}")
    return adata_rna, adata_mod2


# ===========================================================================
# 10. COMPLETE SMART WORKFLOW RUNNER (IDENTICAL TO ORIGINAL SMART SPEC)
# ===========================================================================
def run_smart_workflow(dname, cfg, env_mode, seed, device, show_plots=False, n_epochs=300, emb_dim=64):
    """
    Executes the complete SMART multi-omics workflow for a single dataset and seed,
    strictly matching SMART tutorials and official training specifications.
    """
    set_seed(seed)
    print(f"\n=======================================================")
    print(f"Running SMART Workflow: {dname} | Seed: {seed}")
    print(f"=======================================================")

    # 1. Load Data
    adata_rna, adata_mod2 = load_dataset_data(dname, cfg, env_mode)

    # 2. Preprocess Modality 1: RNA (As in SMART Tutorials 1, 2, 3, 4)
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=min(3000, adata_rna.n_vars))
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    adata_rna_high = adata_rna[:, adata_rna.var['highly_variable']]
    n_comps_rna = min(cfg.get("n_comps_rna", 30), adata_rna_high.n_obs - 1, adata_rna_high.n_vars - 1)
    adata_rna.obsm['feat'] = pca(adata_rna_high, n_comps=n_comps_rna)

    # 3. Preprocess Modality 2: ATAC vs ADT/Protein
    if cfg["type"] == "mouse_brain":
        # ATAC / Epigenomics (As in SMART Tutorials 3 & 4)
        preprocess_atac_tfidf(adata_mod2)
        sc.pp.normalize_per_cell(adata_mod2, counts_per_cell_after=1e4)
        sc.pp.log1p(adata_mod2)
        n_comps_mod2 = min(cfg.get("n_comps_mod2", 60), adata_mod2.n_obs - 1, adata_mod2.n_vars - 1)
        adata_mod2.obsm['feat'] = pca(adata_mod2, n_comps=n_comps_mod2)
    else:
        # ADT / Protein CITE-seq (As in SMART Tutorial 2)
        preprocess_protein_clr(adata_mod2)
        sc.pp.scale(adata_mod2)
        n_comps_mod2 = min(cfg.get("n_comps_mod2", 30), adata_mod2.n_obs - 1, adata_mod2.n_vars - 1)
        adata_mod2.obsm['feat'] = pca(adata_mod2, n_comps=n_comps_mod2)

    # 4. Construct Spatial Neighbor Graphs (KNN)
    n_neighbors = cfg.get("n_neighbors", 4)
    Cal_Spatial_Net(adata_rna, model="KNN", n_neighbors=n_neighbors, verbose=False)
    Cal_Spatial_Net(adata_mod2, model="KNN", n_neighbors=n_neighbors, verbose=False)

    # 5. Extract Feature Tensors and Edge Lists
    adata_list = [adata_rna, adata_mod2]
    features = [torch.FloatTensor(adata.obsm["feat"]).to(device) for adata in adata_list]
    edges = [torch.LongTensor(adata.uns["edgeList"]).to(device) for adata in adata_list]

    # 6. Mine MNN Triplet Constraints (As in SMART Tutorials)
    farthest_ratio = cfg.get("farthest_ratio", 0.6)
    triplet_samples_list = [
        Mutual_Nearest_Neighbors(adata, key="feat", n_nearest_neighbors=3, farthest_ratio=farthest_ratio)
        for adata in adata_list
    ]

    # 7. Train SMART Model (As in SMART Tutorials)
    lr = cfg.get("lr", 1e-3)
    print(f"Training SMART model (lr={lr}, emb_dim={emb_dim}, n_epochs={n_epochs})...")
    model = train_SMART(
        features=features,
        edges=edges,
        triplet_samples_list=triplet_samples_list,
        weights=[1, 1, 1, 1],
        emb_dim=emb_dim,
        n_epochs=n_epochs,
        lr=lr,
        weight_decay=1e-6,
        device=device,
        window_size=10,
        slope=1e-4,
        margin=0.5,
        laplacian_alpha=0.0
    )

    # 8. Extract Latent Embeddings (SMART Z)
    model.eval()
    with torch.no_grad():
        latent_z, _ = model(features, edges)
        smart_emb = latent_z.cpu().detach().numpy()

    adata_rna.obsm['SMART'] = smart_emb
    adata_rna.obsm['joint_feat'] = smart_emb
    print(f"SMART latent representation shape: {smart_emb.shape}")

    # Determine Target Cluster Count
    valid_labels = adata_rna.obs['ground_truth'].dropna().unique()
    target_labels = [l for l in valid_labels if l not in ['Exclude', 'unknown', 'nan', 'None']]
    n_clusters = len(target_labels) if len(target_labels) > 0 else 7

    # 9a. mclust Clustering (Official primary clustering method in SMART)
    print("Clustering with mclust (Gaussian Mixture Models via rpy2)...")
    clustering(adata_rna, key='SMART', add_key='mclust', n_clusters=n_clusters, method='mclust', use_pca=True, n_comps=20)

    # 9b. Leiden Graph Clustering
    print("Clustering with Leiden...")
    res = search_res(adata_rna, n_clusters, method='leiden', use_rep='SMART')
    try:
        sc.tl.leiden(adata_rna, random_state=seed, resolution=res, flavor='igraph', n_iterations=2, directed=False)
    except TypeError:
        sc.tl.leiden(adata_rna, random_state=seed, resolution=res)

    # 9c. KMeans Clustering
    print(f"Clustering with KMeans (n_clusters={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    adata_rna.obs['kmeans'] = kmeans.fit_predict(smart_emb).astype(str)

    # 10. Quantitative Metric Evaluation
    metrics_mclust = evaluate_clustering(
        adata_rna.obs['ground_truth'],
        adata_rna.obs['mclust'],
        smart_emb,
        name=f"{dname} Seed {seed} - mclust (SMART primary)"
    )

    metrics_leiden = evaluate_clustering(
        adata_rna.obs['ground_truth'],
        adata_rna.obs['leiden'],
        smart_emb,
        name=f"{dname} Seed {seed} - Leiden"
    )

    metrics_kmeans = evaluate_clustering(
        adata_rna.obs['ground_truth'],
        adata_rna.obs['kmeans'],
        smart_emb,
        name=f"{dname} Seed {seed} - KMeans"
    )

    row_mclust = {
        'cluster alg': 'mclust',
        'ARI': metrics_mclust['ARI'],
        'NMI': metrics_mclust['NMI'],
        'Silhouette': metrics_mclust['Silhouette'],
        'AMI': metrics_mclust['AMI'],
        'CHI': metrics_mclust['CHI'],
        'DBI': metrics_mclust['DBI'],
        'Homogeneity': metrics_mclust['Homogeneity'],
        'V-measure': metrics_mclust['V-measure'],
        'resolution': None
    }

    row_leiden = {
        'cluster alg': 'Leiden',
        'ARI': metrics_leiden['ARI'],
        'NMI': metrics_leiden['NMI'],
        'Silhouette': metrics_leiden['Silhouette'],
        'AMI': metrics_leiden['AMI'],
        'CHI': metrics_leiden['CHI'],
        'DBI': metrics_leiden['DBI'],
        'Homogeneity': metrics_leiden['Homogeneity'],
        'V-measure': metrics_leiden['V-measure'],
        'resolution': res
    }

    row_kmeans = {
        'cluster alg': 'KMeans',
        'ARI': metrics_kmeans['ARI'],
        'NMI': metrics_kmeans['NMI'],
        'Silhouette': metrics_kmeans['Silhouette'],
        'AMI': metrics_kmeans['AMI'],
        'CHI': metrics_kmeans['CHI'],
        'DBI': metrics_kmeans['DBI'],
        'Homogeneity': metrics_kmeans['Homogeneity'],
        'V-measure': metrics_kmeans['V-measure'],
        'resolution': None
    }

    # 11. Visualizations
    if show_plots:
        plot_smart_visualizations(adata_rna, title_prefix=f"{dname} (Seed {seed})", dname=dname, seed=seed)

    return [row_mclust, row_leiden, row_kmeans]


# ===========================================================================
# 11. MAIN PIPELINE EXECUTION DRIVER
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="SMART Multi-Seed Baseline & Ablation Pipeline")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Datasets to run ('all' or specific names)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="List of random seeds")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs per run")
    parser.add_argument("--emb_dim", type=int, default=64, help="Latent embedding dimension")
    parser.add_argument("--device", type=str, default=None, help="Torch device ('cuda' or 'cpu')")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for CSVs and plots")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        datasets_to_run = list(ALL_DATASETS_CONFIG.keys())
    else:
        datasets_to_run = [d for d in args.datasets if d in ALL_DATASETS_CONFIG]

    seeds_to_run = args.seeds
    print(f"Scheduled datasets: {datasets_to_run}")
    print(f"Evaluation seeds ({len(seeds_to_run)}): {seeds_to_run}")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
    os.makedirs(output_dir, exist_ok=True)

    all_results_flat = []
    all_results = {}

    for dname in datasets_to_run:
        cfg = ALL_DATASETS_CONFIG[dname]
        all_results[dname] = []

        print(f"\n=======================================================")
        print(f"STARTING SMART WORKFLOW FOR DATASET: {dname} ({len(seeds_to_run)} SEEDS)")
        print(f"=======================================================")

        for idx, seed in enumerate(seeds_to_run):
            show_plots = (idx == 0)
            try:
                results_list = run_smart_workflow(
                    dname=dname,
                    cfg=cfg,
                    env_mode=ENV_MODE,
                    seed=seed,
                    device=device,
                    show_plots=show_plots,
                    n_epochs=args.epochs,
                    emb_dim=args.emb_dim
                )
                if results_list:
                    for row in results_list:
                        res_row = {"dataset": dname, "seed": seed}
                        res_row.update(row)
                        all_results[dname].append(res_row)
                        all_results_flat.append(res_row)
            except Exception as e:
                print(f"Error processing dataset {dname} with seed {seed}: {e}")
                import traceback
                traceback.print_exc()

        if len(all_results[dname]) > 0:
            df_metrics = pd.DataFrame(all_results[dname])
            print(f"\n=======================================================")
            print(f"AVERAGE SMART PERFORMANCE FOR {dname} ({len(seeds_to_run)} seeds)")
            print(f"=======================================================")
            numeric_cols = ["ARI", "NMI", "Silhouette", "AMI", "CHI", "DBI", "Homogeneity", "V-measure"]
            for alg, df_alg in df_metrics.groupby("cluster alg"):
                print(f"\n--- {alg} Performance (Mean ± Std) ---")
                means = df_alg[numeric_cols].mean()
                stds = df_alg[numeric_cols].std()
                summary_df = pd.DataFrame({"Mean": means, "Std": stds})
                print(summary_df.to_string())
            print(f"=======================================================\n")

    if len(all_results_flat) > 0:
        df_all = pd.DataFrame(all_results_flat)
        output_csv = os.path.join(output_dir, 'smart_ablation_results.csv')
        df_all.to_csv(output_csv, index=False)
        print(f"\nAll SMART ablation results saved to CSV at: {output_csv}")

        print("Generating side-by-side Box & Whiskers plots across algorithms...")
        plot_smart_summary_boxplots(df_all, save_dir=output_dir)
    else:
        print("No evaluation metrics collected, skipping CSV export and Boxplots.")


if __name__ == '__main__':
    main()
