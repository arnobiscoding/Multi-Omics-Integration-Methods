#!/usr/bin/env python3
"""
================================================================================
🧬 SpatialGlue: Spatial Multi-Omics Graph Neural Network Pipeline
================================================================================
Deep Learning Spatial Multi-Omics Clustering Pipeline based on SpatialGlue:
  • Multi-graph integration: Spatial adjacency + feature correlation graphs
  • Modality-specific GNN encoders & decoders with intra- and inter-modality attention
  • Cross-modality consistency & reconstruction loss optimization
  • Comprehensive Evaluation: ARI, NMI, AMI, Silhouette, Homogeneity, V-measure, FMI, CHI, DBI
  • Multi-loss trajectory & metric curve tracking (Total, Recon, Correspondence, Silhouette, ARI)
  • Full Spatial & UMAP Embeddings export (100% of spots)
  • Direct Dashboard API Integration: Automatically pushes structured JSONs & metrics to MongoDB / Next.js
================================================================================
"""

import os
import sys
import copy
import json
import time
import random
import argparse
import warnings
import urllib.request
import urllib.error
from typing import Optional, Tuple, Dict, List, Union

# Suppress non-critical user and future warnings for clean terminal logs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup R environment paths if available
if 'R_HOME' not in os.environ and os.path.exists('/usr/lib/R'):
    os.environ['R_HOME'] = '/usr/lib/R'
    os.environ['PATH'] = '/usr/lib/R/bin:' + os.environ.get('PATH', '')

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from scipy.sparse import coo_matrix

import sklearn
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    fowlkes_mallows_score,
    silhouette_score,
    silhouette_samples,
    calinski_harabasz_score,
    davies_bouldin_score
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.backends import cudnn

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# 1. Dataset Registry & Global Settings
# ==============================================================================

CHOICES = [
    ("10x_human_lymph_node_A1", "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC"),
    ("10x_human_lymph_node_D1", "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz"),
    ("Mouse_Brain_E11_S1", "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo"),
    ("Mouse_Brain_E13_S1", "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud"),
    ("Mouse_Brain_E15_S1", "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD"),
    ("Mouse_Brain_E18_S1", "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB"),
]

DATASET_REGISTRY = {
    "10x_human_lymph_node_A1": {
        "index": 0,
        "url": "https://drive.google.com/drive/folders/10z1N4MwW8Y49o8GlkYGBKVx1N7fiMuyC",
        "type": "10x",
        "gt_col": "manual-anno",
        "other_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv",
        "default_epochs": 200,
        "weight_factors": [1.0, 5.0, 1.0, 10.0]
    },
    "10x_human_lymph_node_D1": {
        "index": 1,
        "url": "https://drive.google.com/drive/folders/1-g_Ca2XMaMXF-MisuVY-wobWDX86O6zz",
        "type": "10x",
        "gt_col": "manual-anno",
        "other_file": "adata_ADT.h5ad",
        "anno_file": "annotation.csv",
        "default_epochs": 200,
        "weight_factors": [1.0, 5.0, 1.0, 10.0]
    },
    "Mouse_Brain_E11_S1": {
        "index": 2,
        "url": "https://drive.google.com/drive/folders/1zRwDJrYnks0LRzlAVRqPU7jE_OcStgPo",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv",
        "default_epochs": 1600,
        "weight_factors": [1.0, 5.0, 1.0, 1.0]
    },
    "Mouse_Brain_E13_S1": {
        "index": 3,
        "url": "https://drive.google.com/drive/folders/1GOufwIRjjfcd9Bi2GKtebzKoPCg2jVud",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv",
        "default_epochs": 1600,
        "weight_factors": [1.0, 5.0, 1.0, 1.0]
    },
    "Mouse_Brain_E15_S1": {
        "index": 4,
        "url": "https://drive.google.com/drive/folders/1rHkTL5OF5qPsEERypRGMS51SjUQ69tdD",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv",
        "default_epochs": 1600,
        "weight_factors": [1.0, 5.0, 1.0, 1.0]
    },
    "Mouse_Brain_E18_S1": {
        "index": 5,
        "url": "https://drive.google.com/drive/folders/1Xj1LNIAY93biS6JIMKNRODn5GvtCKADB",
        "type": "Spatial-epigenome-transcriptome",
        "gt_col": "cluster",
        "other_file": "adata_ATAC.h5ad",
        "anno_file": "anno.csv",
        "default_epochs": 1600,
        "weight_factors": [1.0, 5.0, 1.0, 1.0]
    },
}

DATASET_LIST = list(DATASET_REGISTRY.keys())

DEFAULT_SEEDS = [42, 0, 1, 7, 123, 1234, 2022, 2023, 2024, 1337]


# ==============================================================================
# 2. Reproducibility & Device Configuration
# ==============================================================================

def fix_seed(seed: int = 42):
    """Set random seed across Python, NumPy, PyTorch, and CUDA for deterministic runs."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def get_compute_device(requested_device: Optional[str] = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using compute engine: {dev}")
    if torch.cuda.is_available():
        print(f"[GPU] Model: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    return dev


# ==============================================================================
# 3. Preprocessing Functions
# ==============================================================================

def clr_normalize_each_cell(adata, inplace=True):
    """Normalize count vector for each cell using Centered Log-Ratio (CLR)."""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x)) if len(x) > 0 else 1.0
        return np.log1p(x / (exp + 1e-12))

    if not inplace:
        adata = adata.copy()

    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    adata.X = np.apply_along_axis(seurat_clr, 1, X)
    return adata


def pca(adata, use_reps=None, n_comps=10):
    """Dimension reduction with PCA algorithm."""
    from sklearn.decomposition import PCA
    pca_model = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca_model.fit_transform(adata.obsm[use_reps])
    else:
        feat_pca = pca_model.fit_transform(adata.X.toarray() if sp.issparse(adata.X) else adata.X)
    return feat_pca


def tfidf(X):
    """TF-IDF normalization following Seurat approach."""
    idf = X.shape[0] / (X.sum(axis=0) + 1e-12)
    if sp.issparse(X):
        tf = X.multiply(1.0 / (X.sum(axis=1) + 1e-12))
        return sp.csr_matrix(tf.multiply(idf))
    else:
        tf = X / (X.sum(axis=1, keepdims=True) + 1e-12)
        return tf * idf


def lsi(adata: ad.AnnData, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs) -> None:
    """LSI analysis (TF-IDF + Truncated SVD) following Seurat v3."""
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= (X_lsi.std(axis=1, ddof=1, keepdims=True) + 1e-12)
    adata.obsm["X_lsi"] = X_lsi[:, 1:]


# ==============================================================================
# 4. Graph Construction & Preprocessing
# ==============================================================================

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    """Constructing spatial neighbor graph according to spatial coordinates."""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj


def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode="connectivity", metric="correlation", include_self=False):
    """Constructing feature neighbor graph according to expression profiles."""
    feature_graph_omics1 = kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics2 = kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    return feature_graph_omics1, feature_graph_omics2


def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3):
    """Construct neighbor graphs, including feature graph and spatial graph."""
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
        n_neighbors = 6

    # Omics 1 spatial graph
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1

    # Omics 2 spatial graph
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2

    # Feature graphs
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'] = feature_graph_omics1
    adata_omics2.obsm['adj_feature'] = feature_graph_omics2

    return {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}


def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def preprocess_graph(adj):
    """Normalize adjacency matrix: D^{-1/2} (A + I) D^{-1/2}."""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(np.maximum(rowsum, 1e-12), -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix for both modalities."""
    # 1. Spatial graphs
    adj_spatial_omics1 = transform_adjacent_matrix(adata_omics1.uns['adj_spatial']).toarray()
    adj_spatial_omics2 = transform_adjacent_matrix(adata_omics2.uns['adj_spatial']).toarray()

    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1 > 1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2 > 1, 1, adj_spatial_omics2)

    adj_spatial_omics1_t = preprocess_graph(adj_spatial_omics1)
    adj_spatial_omics2_t = preprocess_graph(adj_spatial_omics2)

    # 2. Feature graphs
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())

    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1 > 1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2 > 1, 1, adj_feature_omics2)

    adj_feature_omics1_t = preprocess_graph(adj_feature_omics1)
    adj_feature_omics2_t = preprocess_graph(adj_feature_omics2)

    return {
        'adj_spatial_omics1': adj_spatial_omics1_t,
        'adj_spatial_omics2': adj_spatial_omics2_t,
        'adj_feature_omics1': adj_feature_omics1_t,
        'adj_feature_omics2': adj_feature_omics2_t,
    }


# ==============================================================================
# 5. SpatialGlue Neural Architecture
# ==============================================================================

class Encoder(Module):
    """Modality-specific GNN encoder."""
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x


class Decoder(Module):
    """Modality-specific GNN decoder."""
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x


class AttentionLayer(Module):
    """Attention aggregation layer (intra- and cross-modality)."""
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(AttentionLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_omega)
        torch.nn.init.xavier_uniform_(self.u_omega)

    def forward(self, emb1, emb2):
        emb = []
        emb.append(torch.unsqueeze(torch.squeeze(emb1), dim=1))
        emb.append(torch.unsqueeze(torch.squeeze(emb2), dim=1))
        self.emb = torch.cat(emb, dim=1)

        self.v = torch.tanh(torch.matmul(self.emb, self.w_omega))
        self.vu = torch.matmul(self.v, self.u_omega)
        self.alpha = F.softmax(torch.squeeze(self.vu, dim=-1) + 1e-6, dim=-1)

        emb_combined = torch.matmul(torch.transpose(self.emb, 1, 2), torch.unsqueeze(self.alpha, -1))
        return torch.squeeze(emb_combined, dim=-1), self.alpha


class Encoder_overall(Module):
    """Overall SpatialGlue Architecture integrating intra- and cross-modality encoders/decoders."""
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act

        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.decoder_omics1 = Decoder(self.dim_out_feat_omics1, self.dim_in_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        self.decoder_omics2 = Decoder(self.dim_out_feat_omics2, self.dim_in_feat_omics2)

        self.atten_omics1 = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics1)
        self.atten_omics2 = AttentionLayer(self.dim_out_feat_omics2, self.dim_out_feat_omics2)
        self.atten_cross = AttentionLayer(self.dim_out_feat_omics1, self.dim_out_feat_omics2)

    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2):
        # 1. Spatial GNN embeddings
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)

        # 2. Feature GNN embeddings
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        # 3. Within-modality attention aggregation
        emb_latent_omics1, alpha_omics1 = self.atten_omics1(emb_latent_spatial_omics1, emb_latent_feature_omics1)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spatial_omics2, emb_latent_feature_omics2)

        # 4. Between-modality cross-attention aggregation
        emb_latent_combined, alpha_cross = self.atten_cross(emb_latent_omics1, emb_latent_omics2)

        # 5. Reverse integrated representation back to original spaces
        emb_recon_omics1 = self.decoder_omics1(emb_latent_combined, adj_spatial_omics1)
        emb_recon_omics2 = self.decoder_omics2(emb_latent_combined, adj_spatial_omics2)

        # 6. Cross-modality consistency encoding
        emb_latent_omics1_across_recon = self.encoder_omics2(self.decoder_omics2(emb_latent_omics1, adj_spatial_omics2), adj_spatial_omics2)
        emb_latent_omics2_across_recon = self.encoder_omics1(self.decoder_omics1(emb_latent_omics2, adj_spatial_omics1), adj_spatial_omics1)

        return {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon_omics1,
            'emb_recon_omics2': emb_recon_omics2,
            'emb_latent_omics1_across_recon': emb_latent_omics1_across_recon,
            'emb_latent_omics2_across_recon': emb_latent_omics2_across_recon,
            'alpha_omics1': alpha_omics1,
            'alpha_omics2': alpha_omics2,
            'alpha': alpha_cross
        }


# ==============================================================================
# 6. SpatialGlue Training Engine & Clustering
# ==============================================================================

class Train_SpatialGlue:
    def __init__(
        self,
        data,
        datatype='SPOTS',
        device=torch.device('cpu'),
        epochval=None,
        random_seed=2022,
        learning_rate=0.0001,
        weight_decay=0.00,
        epochs=600,
        dim_input=3000,
        dim_output=64,
        weight_factors=None,
        true_labels=None,
        num_clusters=10,
        eval_interval=10
    ):
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.true_labels = true_labels
        self.num_clusters = num_clusters
        self.eval_interval = eval_interval

        # Configure default epochs and weight factors
        if self.datatype == 'SPOTS':
            self.epochs = 600
            self.weight_factors = [1, 5, 1, 1]
        elif self.datatype == 'Stereo-CITE-seq':
            self.epochs = 1500
            self.weight_factors = [1, 10, 1, 10]
        elif self.datatype == '10x':
            self.epochs = 200
            self.weight_factors = [1, 5, 1, 10]
        elif self.datatype == 'Spatial-epigenome-transcriptome':
            self.epochs = 1600
            self.weight_factors = [1, 5, 1, 1]
        else:
            self.weight_factors = [1, 5, 1, 1]

        if weight_factors is not None:
            self.weight_factors = weight_factors

        if epochval is not None:
            self.epochs = epochval

        # Loss and trajectory tracking
        self.loss_history = []
        self.recon_loss_history = []
        self.corr_loss_history = []
        self.epoch_sil_history = []
        self.epoch_ari_history = []
        self.epoch_nmi_history = []

        # Graph Adjacency
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)

        # Features
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)

        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output

    def train(self, verbose=True):
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        best_sil = -float('inf')
        best_epoch = 1
        best_emb = None

        print(f"Training SpatialGlue for {self.epochs} epochs (LR: {self.learning_rate}, Weight factors: {self.weight_factors})...")

        for epoch in range(self.epochs):
            self.model.train()
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2
            )

            # Reconstruction loss
            loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
            loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])

            # Correspondence loss
            loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_latent_omics1_across_recon'])
            loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_latent_omics2_across_recon'])

            recon_total = self.weight_factors[0] * loss_recon_omics1 + self.weight_factors[1] * loss_recon_omics2
            corr_total = self.weight_factors[2] * loss_corr_omics1 + self.weight_factors[3] * loss_corr_omics2
            loss = recon_total + corr_total

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.loss_history.append(loss.item())
            self.recon_loss_history.append(recon_total.item())
            self.corr_loss_history.append(corr_total.item())

            # Evaluate intermediate embeddings periodically or on first/last epoch
            is_eval_epoch = (epoch % self.eval_interval == 0) or (epoch == self.epochs - 1) or (epoch == 0)
            if is_eval_epoch:
                with torch.no_grad():
                    self.model.eval()
                    res_eval = self.model(
                        self.features_omics1, self.features_omics2,
                        self.adj_spatial_omics1, self.adj_feature_omics1,
                        self.adj_spatial_omics2, self.adj_feature_omics2
                    )
                    emb_norm = F.normalize(res_eval['emb_latent_combined'], p=2, eps=1e-12, dim=1).cpu().numpy()

                try:
                    km_labels = KMeans(n_clusters=self.num_clusters, n_init=5, random_state=self.random_seed).fit_predict(emb_norm)
                    curr_sil = float(silhouette_score(emb_norm, km_labels))
                except Exception:
                    curr_sil = 0.0

                curr_ari = 0.0
                curr_nmi = 0.0
                if self.true_labels is not None:
                    try:
                        curr_ari = float(adjusted_rand_score(self.true_labels, km_labels))
                        curr_nmi = float(normalized_mutual_info_score(self.true_labels, km_labels))
                    except Exception:
                        pass

                self.epoch_sil_history.append(curr_sil)
                self.epoch_ari_history.append(curr_ari)
                self.epoch_nmi_history.append(curr_nmi)

                if curr_sil > best_sil:
                    best_sil = curr_sil
                    best_epoch = epoch + 1
                    best_emb = emb_norm.copy()

                if verbose and ((epoch + 1) % 50 == 0 or epoch == 0 or epoch == self.epochs - 1):
                    ari_str = f" | ARI: {curr_ari:.4f}" if self.true_labels is not None else ""
                    print(f"Epoch {epoch+1:4d}/{self.epochs} | Total Loss: {loss.item():.4f} | Recon: {recon_total.item():.4f} | Corr: {corr_total.item():.4f} | Sil: {curr_sil:.4f}{ari_str}")
            else:
                # Interpolate history point for smooth curves
                self.epoch_sil_history.append(self.epoch_sil_history[-1] if self.epoch_sil_history else 0.0)
                self.epoch_ari_history.append(self.epoch_ari_history[-1] if self.epoch_ari_history else 0.0)
                self.epoch_nmi_history.append(self.epoch_nmi_history[-1] if self.epoch_nmi_history else 0.0)

        print(f"Model training finished! Final Loss: {loss.item():.4f} | Best Silhouette: {best_sil:.4f} (Epoch {best_epoch})\n")

        with torch.no_grad():
            self.model.eval()
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2
            )

        emb_omics1 = F.normalize(results['emb_latent_omics1'], p=2, eps=1e-12, dim=1)
        emb_omics2 = F.normalize(results['emb_latent_omics2'], p=2, eps=1e-12, dim=1)
        emb_combined = F.normalize(results['emb_latent_combined'], p=2, eps=1e-12, dim=1)

        output = {
            'emb_latent_omics1': emb_omics1.detach().cpu().numpy(),
            'emb_latent_omics2': emb_omics2.detach().cpu().numpy(),
            'SpatialGlue': emb_combined.detach().cpu().numpy(),
            'alpha_omics1': results['alpha_omics1'].detach().cpu().numpy(),
            'alpha_omics2': results['alpha_omics2'].detach().cpu().numpy(),
            'alpha': results['alpha'].detach().cpu().numpy(),
            'loss_history': self.loss_history,
            'recon_loss_history': self.recon_loss_history,
            'corr_loss_history': self.corr_loss_history,
            'epoch_sil_history': self.epoch_sil_history,
            'epoch_ari_history': self.epoch_ari_history,
            'epoch_nmi_history': self.epoch_nmi_history,
            'best_epoch': best_epoch,
            'best_sil': best_sil,
            'best_embeddings': best_emb if best_emb is not None else emb_combined.detach().cpu().numpy()
        }
        return output


# ==============================================================================
# 7. Clustering Support (mclust, KMeans, Leiden, Louvain)
# ==============================================================================

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """Clustering using R package mclust via rpy2 with robust fallback."""
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri, default_converter
        from rpy2.robjects.conversion import localconverter
        import rpy2.robjects.conversion as cv

        cv.set_conversion(default_converter + pandas2ri.converter)
        robjects.r.options(warn=-1)

        # Attempt to install mclust if missing in R environment
        robjects.r("""
        if (!requireNamespace("mclust", quietly = TRUE)) {
            install.packages("mclust", repos="https://cloud.r-project.org", quiet=TRUE)
        }
        library(mclust)
        """)

        np.random.seed(random_seed)
        r_random_seed = robjects.r["set.seed"]
        r_random_seed(random_seed)
        rmclust = robjects.r["Mclust"]

        X = np.array(adata.obsm[used_obsm], dtype=np.float64)
        df = pd.DataFrame(X, columns=[f'PC{i+1}' for i in range(X.shape[1])])

        subset_size = min(300, X.shape[0])
        subset_indices = robjects.IntVector(list(np.random.choice(range(1, X.shape[0] + 1), subset_size, replace=False)))
        init_list = robjects.ListVector({'subset': subset_indices})

        with localconverter(default_converter + pandas2ri.converter):
            res = rmclust(df, G=num_cluster, modelNames=modelNames, initialization=init_list)

        if hasattr(res, 'rx2'):
            mclust_res = np.array(res.rx2('classification'))
        elif hasattr(res, 'getbyname'):
            mclust_res = np.array(res.getbyname('classification'))
        else:
            mclust_res = np.array(res['classification'])

        adata.obs['mclust'] = mclust_res
        adata.obs['mclust'] = adata.obs['mclust'].astype('int').astype('str').astype('category')
        return adata
    except Exception as e:
        print(f"[Warning] mclust via rpy2 not available ({e}). Falling back to KMeans clustering.")
        X = np.array(adata.obsm[used_obsm], dtype=np.float64)
        km = KMeans(n_clusters=num_cluster, n_init=10, random_state=random_seed).fit_predict(X)
        adata.obs['mclust'] = pd.Categorical(km.astype(str))
        return adata


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.02):
    """Search corresponding resolution for Leiden or Louvain according to target number of clusters."""
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    best_res = start
    min_diff = 999
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
            sc.tl.leiden(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        else:
            sc.tl.louvain(adata, random_state=0, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())

        if count_unique == n_clusters:
            return res
        if abs(count_unique - n_clusters) < min_diff:
            min_diff = abs(count_unique - n_clusters)
            best_res = res
    return best_res


def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0, increment=0.02, use_pca=False, n_comps=20, random_seed=2020):
    """Spatial clustering based on latent representations."""
    used_key = key
    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=min(n_comps, adata.obsm[key].shape[1] - 1))
        used_key = key + '_pca'

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm=used_key, num_cluster=n_clusters, random_seed=random_seed)
        adata.obs[add_key] = adata.obs['mclust']
    elif method == 'kmeans':
        X = adata.obsm[used_key]
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_seed).fit_predict(X)
        adata.obs[add_key] = pd.Categorical(km.astype(str))
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep=used_key, method=method, start=start, end=end, increment=increment)
        sc.tl.leiden(adata, random_state=random_seed, resolution=res)
        adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep=used_key, method=method, start=start, end=end, increment=increment)
        sc.tl.louvain(adata, random_state=random_seed, resolution=res)
        adata.obs[add_key] = adata.obs['louvain']


# ==============================================================================
# 8. Visualizations & Standard Plotting
# ==============================================================================

def plot_training_curves(training_results: dict, dataset_name="Dataset", save_path=None, show=False):
    """Plot Loss decomposition, Silhouette Score curve, and ARI curve across training epochs."""
    epochs_range = range(1, len(training_results['loss_history']) + 1)
    has_ari = len(training_results.get('epoch_ari_history', [])) > 0

    fig, axes = plt.subplots(1, 3 if has_ari else 2, figsize=(18 if has_ari else 12, 4.5), dpi=150)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # 1. Loss Decomposition Curves
    axes[0].plot(epochs_range, training_results['loss_history'], color='#1f77b4', linewidth=2.5, label='Total Loss')
    if 'recon_loss_history' in training_results and len(training_results['recon_loss_history']) == len(epochs_range):
        axes[0].plot(epochs_range, training_results['recon_loss_history'], color='#3b82f6', linewidth=1.5, linestyle='--', label='Recon Loss')
    if 'corr_loss_history' in training_results and len(training_results['corr_loss_history']) == len(epochs_range):
        axes[0].plot(epochs_range, training_results['corr_loss_history'], color='#10b981', linewidth=1.5, linestyle='-.', label='Corr Loss')

    axes[0].set_title(f'Loss Curves - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss Value', fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(frameon=True, fontsize='small', loc='upper right')

    # 2. Silhouette Score Curve
    best_epoch = training_results.get('best_epoch', len(epochs_range))
    best_sil = training_results.get('best_sil', training_results['epoch_sil_history'][-1] if training_results.get('epoch_sil_history') else 0.0)
    axes[1].plot(epochs_range, training_results['epoch_sil_history'], color='#2ca02c', linewidth=2.0, label='Silhouette Score')
    axes[1].scatter([best_epoch], [best_sil], color='#e11d48', s=50, zorder=5, label=f'Best Sil (Ep {best_epoch}): {best_sil:.4f}')
    axes[1].set_title(f'Silhouette Score Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(frameon=True, loc='lower right')

    # 3. ARI Curve
    if has_ari:
        best_ep_idx = min(max(0, best_epoch - 1), len(training_results['epoch_ari_history']) - 1)
        best_ep_ari = training_results['epoch_ari_history'][best_ep_idx] if training_results.get('epoch_ari_history') else 0.0
        axes[2].plot(epochs_range, training_results['epoch_ari_history'], color='#ff7f0e', linewidth=2.0, label='Epoch ARI')
        axes[2].scatter([best_epoch], [best_ep_ari], color='#e11d48', s=50, zorder=5, label=f'ARI at Best Sil (Ep {best_epoch}): {best_ep_ari:.4f}')
        axes[2].set_title(f'ARI Curve - {dataset_name}', fontsize=12, fontweight='bold', pad=10)
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Adjusted Rand Index (ARI)', fontsize=11)
        axes[2].grid(True, linestyle='--', alpha=0.5)
        axes[2].legend(frameon=True, loc='lower right')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def plot_all_visualizations(
    adata_RNA,
    final_embeddings: np.ndarray,
    final_labels: np.ndarray,
    sil: float,
    ari: float,
    training_results: dict,
    dataset_name="Dataset",
    seed=42,
    true_labels=None,
    output_dir="results",
    show=False
):
    """Generate and save complete visualizations (Curves, Spatial, UMAP, Violin)."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(os.path.join(plots_dir, "curves"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "spatial"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "umap"), exist_ok=True)
    os.makedirs(os.path.join(plots_dir, "violin"), exist_ok=True)

    if true_labels is not None:
        adata_RNA.obs['ground_truth'] = pd.Categorical(true_labels)
    elif 'ground_truth' not in adata_RNA.obs:
        adata_RNA.obs['ground_truth'] = pd.Categorical(final_labels.astype(str))

    adata_RNA.obsm['SpatialGlue_Emb'] = final_embeddings
    adata_RNA.obs['predicted_domain'] = pd.Categorical(final_labels.astype(str))

    # 1. Training Curves
    curve_path = os.path.join(plots_dir, "curves", f"{dataset_name}_seed{seed}_training_curves.png")
    plot_training_curves(training_results, dataset_name=f"{dataset_name} (Seed {seed})", save_path=curve_path, show=show)

    # 2. Spatial Domains Side-by-Side Comparison
    spatial_coords = adata_RNA.obsm.get('spatial', None)
    if spatial_coords is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        gt_cats = pd.Categorical(adata_RNA.obs['ground_truth'])
        pred_cats = pd.Categorical(adata_RNA.obs['predicted_domain'])
        axes[0].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=gt_cats.codes, cmap='tab20', s=10, alpha=0.9)
        axes[0].set_title(f'Ground Truth ({dataset_name})', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Spatial X')
        axes[0].set_ylabel('Spatial Y')

        axes[1].scatter(spatial_coords[:, 0], spatial_coords[:, 1], c=pred_cats.codes, cmap='tab20', s=10, alpha=0.9)
        axes[1].set_title(f'SpatialGlue Domains (ARI: {ari:.4f})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Spatial X')
        axes[1].set_ylabel('Spatial Y')

        plt.suptitle(f"Spatial Domains Comparison - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "spatial", f"{dataset_name}_seed{seed}_spatial.png"), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

    # 3. UMAP Representation
    sc.pp.neighbors(adata_RNA, use_rep='SpatialGlue_Emb')
    sc.tl.umap(adata_RNA)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sc.pl.umap(adata_RNA, color='ground_truth', ax=axes[0], show=False, title='UMAP: Ground Truth Annotation')
    sc.pl.umap(adata_RNA, color='predicted_domain', ax=axes[1], show=False, title=f'UMAP: Predicted Domains (Sil: {sil:.4f})')
    plt.suptitle(f"UMAP Joint Representation - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "umap", f"{dataset_name}_seed{seed}_umap.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    # 4. Violin Plots: Silhouette & Latent Profiles
    try:
        sample_sil_values = silhouette_samples(final_embeddings, final_labels)
        adata_RNA.obs['silhouette_coefficient'] = sample_sil_values
        adata_RNA.obs['Latent_Dim_1'] = final_embeddings[:, 0]
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
        sns.violinplot(
            data=adata_RNA.obs,
            x='predicted_domain',
            y='silhouette_coefficient',
            hue='predicted_domain',
            palette='Set2',
            legend=False,
            inner='quartile',
            ax=axes[0]
        )
        axes[0].axhline(sil, color='red', linestyle='--', label=f'Mean Sil: {sil:.4f}')
        axes[0].set_title("Silhouette Coefficient per Predicted Domain", fontsize=12, fontweight='bold')
        sns.violinplot(
            data=adata_RNA.obs,
            x='predicted_domain',
            y='Latent_Dim_1',
            hue='predicted_domain',
            palette='tab10',
            legend=False,
            inner='box',
            ax=axes[1]
        )
        axes[1].set_title("Latent Dimension 1 Distribution per Domain", fontsize=12, fontweight='bold')
        plt.suptitle(f"Violin Plots: Cluster Profiles - {dataset_name} (Seed {seed})", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "violin", f"{dataset_name}_seed{seed}_violin.png"), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    except Exception as e:
        print(f"Skipping Violin plots due to error: {e}")


# ==============================================================================
# 9. Standardized Dashboard Result Exporter & Live API Pusher
# ==============================================================================

def make_json_serializable(obj):
    """Recursively converts NumPy types, tensors, and arrays into native JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return [make_json_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (np.floating, np.float32, np.float64, np.float16)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    return obj


def export_dashboard_experiment(
    model_id: str,
    model_name: str,
    dataset_name: str,
    seed: int,
    metrics_dict: dict,
    training_results: dict,
    hyperparameters: dict,
    embeddings_data: Optional[dict] = None,
    output_dir: str = "results",
    api_url: Optional[str] = "https://model-performance.vercel.app/api/experiments/upload"
):
    """Export standardized JSON artifacts and send directly to Dashboard via API."""
    exp_dir = os.path.join(output_dir, "experiments", model_id, dataset_name, f"seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    # 1. Curves History
    loss_hist = training_results.get("loss_history", [])
    recon_hist = training_results.get("recon_loss_history", [])
    corr_hist = training_results.get("corr_loss_history", [])
    sil_hist = training_results.get("epoch_sil_history", [])
    ari_hist = training_results.get("epoch_ari_history", [])
    nmi_hist = training_results.get("epoch_nmi_history", [])

    history_points = []
    for ep in range(len(loss_hist)):
        history_points.append({
            "epoch": ep + 1,
            "losses": {
                "total_loss": float(loss_hist[ep]) if ep < len(loss_hist) else 0.0,
                "reconstruction_loss": float(recon_hist[ep]) if ep < len(recon_hist) else 0.0,
                "spatial_loss": float(corr_hist[ep]) if ep < len(corr_hist) else 0.0,
                "reg_loss": 0.0,
            },
            "metrics": {
                "Silhouette": float(sil_hist[ep]) if ep < len(sil_hist) else 0.0,
                "ARI": float(ari_hist[ep]) if ep < len(ari_hist) else 0.0,
                "NMI": float(nmi_hist[ep]) if ep < len(nmi_hist) else 0.0,
            },
            "total_loss": float(loss_hist[ep]) if ep < len(loss_hist) else None,
            "silhouette": float(sil_hist[ep]) if ep < len(sil_hist) else None,
            "ari": float(ari_hist[ep]) if ep < len(ari_hist) else None,
        })

    safe_metrics = make_json_serializable(metrics_dict)
    safe_hyperparameters = make_json_serializable(hyperparameters)
    safe_history = make_json_serializable(history_points)
    safe_embeddings = make_json_serializable(embeddings_data or {})

    best_epoch = int(training_results.get("best_epoch", len(loss_hist)))
    best_score = float(training_results.get("best_sil", metrics_dict.get("Silhouette", 0.0)))

    # 2. Local JSON Files Save
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump({
            "model_id": model_id,
            "dataset": dataset_name,
            "seed": int(seed),
            "final_epoch": len(loss_hist),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "metrics": safe_metrics
        }, f, indent=2)

    with open(os.path.join(exp_dir, "curves.json"), "w") as f:
        json.dump({"history": safe_history}, f, indent=2)

    with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "dataset": dataset_name,
            "seed": int(seed),
            "hyperparameters": safe_hyperparameters
        }, f, indent=2)

    print(f"📦 Standardized Dashboard JSONs saved to: {exp_dir}")

    # 3. Direct API Call to Next.js / MongoDB Dashboard
    if api_url:
        try:
            payload = {
                "modelId": model_id,
                "modelName": model_name,
                "datasetId": dataset_name,
                "datasetName": dataset_name.replace("_", " ").title(),
                "seed": int(seed),
                "bestEpoch": best_epoch,
                "bestScore": best_score,
                "finalMetrics": safe_metrics,
                "history": safe_history,
                "embeddingsData": safe_embeddings,
                "modelMetadata": {
                    "architecture": "Spatial Multi-Omics Dual-Attention GNN (SpatialGlue)",
                    "hyperparameters": safe_hyperparameters,
                }
            }

            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    print(f"🚀 Successfully sent experiment results directly to Dashboard via API ({api_url})!")
        except Exception as e:
            print(f"ℹ️ Direct API push skipped (dashboard server offline or unreachable: {e}). Data is safely stored in JSON files.")


# ==============================================================================
# 10. Dataset Resolver & Auto-Downloader
# ==============================================================================

def resolve_dataset_files(dataset_name: str, folder_url: str, data_dir: str = "data"):
    candidate_dirs = [
        os.path.join(data_dir, dataset_name),
        f"data/{dataset_name}",
        f"/content/data/{dataset_name}",
        f"/content/drive/MyDrive/Colab/data/{dataset_name}",
    ]
    base_dir = None
    for c_dir in candidate_dirs:
        if os.path.exists(c_dir):
            base_dir = c_dir
            break

    if base_dir is None:
        base_dir = os.path.join(data_dir, dataset_name)
        os.makedirs(base_dir, exist_ok=True)
        print(f"Downloading dataset {dataset_name} using gdown...")
        gdown_cmd = ".venv/bin/gdown" if os.path.exists(".venv/bin/gdown") else "gdown"
        os.system(f'{gdown_cmd} --folder "{folder_url}" --output "{base_dir}"')

    rna_path = os.path.join(base_dir, "adata_RNA.h5ad")
    if dataset_name.startswith("10x"):
        aux_path = os.path.join(base_dir, "adata_ADT.h5ad")
        anno_path = os.path.join(base_dir, "annotation.csv")
        gt_col = "manual-anno"
        data_type = '10x'
    else:
        aux_path = os.path.join(base_dir, "adata_ATAC.h5ad")
        anno_path = os.path.join(base_dir, "anno.csv")
        gt_col = "cluster"
        data_type = 'Spatial-epigenome-transcriptome'

    if not (os.path.exists(rna_path) and os.path.exists(aux_path) and os.path.exists(anno_path)):
        print(f"Re-downloading {dataset_name}...")
        gdown_cmd = ".venv/bin/gdown" if os.path.exists(".venv/bin/gdown") else "gdown"
        os.system(f'{gdown_cmd} --folder "{folder_url}" --output "{base_dir}"')

    return rna_path, aux_path, anno_path, gt_col, data_type


# ==============================================================================
# 11. Experiment Execution Engine
# ==============================================================================

def run_experiment(
    datasets: Union[str, int, List[Union[str, int]]] = "all",
    seeds: Optional[List[int]] = None,
    n_seeds: Optional[int] = None,
    epochs: Optional[int] = None,
    lr: float = 1e-4,
    weight_decay: float = 0.00,
    dim_output: int = 64,
    tool: str = 'mclust',
    device: Optional[str] = None,
    visualize: bool = True,
    show_plots: bool = False,
    output_dir: str = "results",
    data_dir: str = "data",
    api_url: Optional[str] = "https://model-performance.vercel.app/api/experiments/upload"
):
    dev = get_compute_device(device)
    os.makedirs(output_dir, exist_ok=True)

    if seeds is None or seeds == "all" or (isinstance(seeds, (list, tuple)) and len(seeds) > 0 and str(seeds[0]).lower() == "all"):
        run_seeds = DEFAULT_SEEDS
    else:
        run_seeds = [int(s) for s in seeds if isinstance(s, int) or (isinstance(s, str) and s.isdigit())]

    if n_seeds is not None and n_seeds > 0:
        run_seeds = run_seeds[:n_seeds]

    if datasets == "all" or (isinstance(datasets, (list, tuple)) and datasets and datasets[0] == "all"):
        dataset_names = [c[0] for c in CHOICES]
    else:
        dataset_names = []
        for d in (datasets if isinstance(datasets, (list, tuple)) else [datasets]):
            if isinstance(d, int) or (isinstance(d, str) and d.isdigit()):
                idx = int(d)
                if 0 <= idx < len(CHOICES):
                    dataset_names.append(CHOICES[idx][0])
            else:
                dataset_names.append(str(d))

    total_runs = len(dataset_names) * len(run_seeds)
    print("\n" + "=" * 80)
    print(" 🚀 STARTING SPATIALGLUE MULTI-OMICS EXPERIMENT ".center(80, "="))
    print("=" * 80)
    print(f"• Datasets     : {dataset_names}")
    print(f"• Seeds ({len(run_seeds)})   : {run_seeds}")
    print(f"• Clustering   : {tool}")
    print(f"• Output Dim   : {dim_output} | LR: {lr} | Weight Decay: {weight_decay}")
    print(f"• Device       : {dev}")
    print("=" * 80 + "\n")

    all_results = []

    for d_idx, dataset_name in enumerate(dataset_names, 1):
        folder_url = next((c[1] for c in CHOICES if c[0] == dataset_name), None)
        if not folder_url:
            print(f"Dataset {dataset_name} not found in registry. Skipping.")
            continue

        print("\n" + "#" * 80)
        print(f" DATASET: {dataset_name} ".center(80, "#"))
        print("#" * 80)

        rna_path, aux_path, anno_path, gt_col, data_type = resolve_dataset_files(dataset_name, folder_url, data_dir=data_dir)
        adata_omics1 = sc.read_h5ad(rna_path)
        adata_omics2 = sc.read_h5ad(aux_path)

        adata_omics1.var_names_make_unique()
        adata_omics2.var_names_make_unique()

        anno_df = pd.read_csv(anno_path, index_col=0)
        gt_values = anno_df[gt_col] if isinstance(anno_df[gt_col], pd.Series) else anno_df[gt_col].iloc[:, 0]
        adata_omics1.obs['ground_truth'] = gt_values.values
        adata_omics2.obs['ground_truth'] = gt_values.values

        # ------------------ Preprocessing ------------------
        sc.pp.filter_genes(adata_omics1, min_cells=10)
        if not dataset_name.startswith("10x"):
            sc.pp.filter_cells(adata_omics1, min_genes=200)

        sc.pp.highly_variable_genes(adata_omics1, flavor="seurat_v3", n_top_genes=3000)
        sc.pp.normalize_total(adata_omics1, target_sum=1e4)
        sc.pp.log1p(adata_omics1)
        sc.pp.scale(adata_omics1)

        adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]

        if dataset_name.startswith("10x"):
            adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=adata_omics2.n_vars - 1)
            adata_omics2 = clr_normalize_each_cell(adata_omics2)
            sc.pp.scale(adata_omics2)
            adata_omics2.obsm["feat"] = pca(adata_omics2, n_comps=adata_omics2.n_vars - 1)
        else:
            adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=50)
            adata_omics2 = adata_omics2[adata_omics1.obs_names].copy()
            if "X_lsi" not in adata_omics2.obsm:
                sc.pp.highly_variable_genes(adata_omics2, flavor="seurat_v3", n_top_genes=3000)
                lsi(adata_omics2, use_highly_variable=False, n_components=51)
            adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()

        # Construct spatial neighbor graphs
        data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=data_type)
        n_ground_truth = int(adata_omics1.obs["ground_truth"].dropna().nunique())
        n_cluster = n_ground_truth
        print(f"Number of ground truth classes: {n_ground_truth}")

        cfg = DATASET_REGISTRY.get(dataset_name, {})
        run_epochs = epochs if epochs is not None else cfg.get("default_epochs", 600)
        weight_factors = cfg.get("weight_factors", [1.0, 5.0, 1.0, 1.0])

        dataset_results = []

        for s_idx, seed in enumerate(run_seeds, 1):
            run_num = (d_idx - 1) * len(run_seeds) + s_idx
            print(f"\n[{run_num}/{total_runs}] Running SpatialGlue on {dataset_name} | Seed: {seed}...")
            fix_seed(seed)

            data_copy = {
                'adata_omics1': data['adata_omics1'].copy(),
                'adata_omics2': data['adata_omics2'].copy()
            }

            gt_labels_array = adata_omics1.obs['ground_truth'].astype(str).values
            start_time = time.time()

            # Train Model
            model_trainer = Train_SpatialGlue(
                data=data_copy,
                datatype=data_type,
                device=dev,
                epochval=run_epochs,
                random_seed=seed,
                learning_rate=lr,
                weight_decay=weight_decay,
                dim_output=dim_output,
                weight_factors=weight_factors,
                true_labels=gt_labels_array,
                num_clusters=n_cluster
            )
            output = model_trainer.train(verbose=True)
            duration = time.time() - start_time

            adata_eval = data_copy['adata_omics1'].copy()
            adata_eval.obsm['SpatialGlue'] = output['SpatialGlue'].copy()
            adata_eval.obsm['alpha'] = output['alpha']

            # Clustering
            clustering(
                adata_eval,
                key='SpatialGlue',
                add_key='SpatialGlue',
                n_clusters=n_cluster,
                method=tool,
                use_pca=True,
                random_seed=seed
            )

            y_true = adata_eval.obs['ground_truth'].astype(str).values
            y_pred = adata_eval.obs['SpatialGlue'].astype(str).values

            # Metric Calculations
            ari = float(adjusted_rand_score(y_true, y_pred))
            nmi = float(normalized_mutual_info_score(y_true, y_pred))
            ami = float(adjusted_mutual_info_score(y_true, y_pred))
            homo = float(homogeneity_score(y_true, y_pred))
            v_meas = float(v_measure_score(y_true, y_pred))
            fmi = float(fowlkes_mallows_score(y_true, y_pred))

            joint_feat = adata_eval.obsm['SpatialGlue']
            le = LabelEncoder()
            y_pred_int = le.fit_transform(y_pred)
            sil_score = float(silhouette_score(joint_feat, y_pred_int))
            chi_score = float(calinski_harabasz_score(joint_feat, y_pred_int))
            dbi_score = float(davies_bouldin_score(joint_feat, y_pred_int))

            print(f"\n✨ Evaluation Results for {dataset_name} | Seed: {seed}:")
            print(f"  • Silhouette : {sil_score:.4f}")
            print(f"  • ARI        : {ari:.4f} | NMI: {nmi:.4f} | AMI: {ami:.4f}")
            print(f"  • Homogeneity: {homo:.4f} | V-measure: {v_meas:.4f} | FMI: {fmi:.4f}")
            print(f"  • CHI        : {chi_score:.2f} | DBI: {dbi_score:.4f}")
            print(f"  • Duration   : {duration:.2f}s")

            # Visualization Plots
            if visualize:
                plot_all_visualizations(
                    adata_RNA=adata_eval,
                    final_embeddings=joint_feat,
                    final_labels=y_pred,
                    sil=sil_score,
                    ari=ari,
                    training_results=output,
                    dataset_name=dataset_name,
                    seed=seed,
                    true_labels=y_true,
                    output_dir=output_dir,
                    show=show_plots
                )

            # Compute UMAP coordinates on full spots
            umap_obj = sc.pp.neighbors(adata_eval, use_rep='SpatialGlue', copy=True)
            sc.tl.umap(umap_obj)
            umap_coords = umap_obj.obsm['X_umap']
            spatial_coords = adata_eval.obsm.get('spatial', None)

            embeddings_data = {
                "umapCoordinates": umap_coords.tolist(),
                "spatialCoordinates": spatial_coords.tolist() if spatial_coords is not None else [],
                "predictedLabels": y_pred.tolist(),
                "groundTruthLabels": y_true.tolist(),
                "sampleSilhouettes": silhouette_samples(joint_feat, y_pred_int).tolist(),
            }

            metrics_dict = {
                "ARI": ari,
                "NMI": nmi,
                "Silhouette": sil_score,
                "AMI": ami,
                "CHI": chi_score,
                "DBI": dbi_score,
                "Homogeneity": homo,
                "V-measure": v_meas,
                "FMI": fmi,
            }

            hyperparameters = {
                "epochs": run_epochs,
                "lr": lr,
                "weight_decay": weight_decay,
                "dim_output": dim_output,
                "weight_factors": weight_factors,
                "clustering_tool": tool,
            }

            # Export to Dashboard API & local JSONs
            export_dashboard_experiment(
                model_id="SpatialGlue",
                model_name="SpatialGlue",
                dataset_name=dataset_name,
                seed=seed,
                metrics_dict=metrics_dict,
                training_results=output,
                hyperparameters=hyperparameters,
                embeddings_data=embeddings_data,
                output_dir=output_dir,
                api_url=api_url
            )

            res_dict = {
                'dataset': dataset_name,
                'seed': seed,
                'best_epoch': output.get('best_epoch', run_epochs),
                'ARI': ari,
                'NMI': nmi,
                'AMI': ami,
                'Silhouette': sil_score,
                'CHI': chi_score,
                'DBI': dbi_score,
                'Homogeneity': homo,
                'V-measure': v_meas,
                'FMI': fmi,
                'no_cluster': n_cluster,
                'Duration_s': round(duration, 2)
            }
            dataset_results.append(res_dict)
            all_results.append(res_dict)

        # Save per-dataset CSV results
        df_ds = pd.DataFrame(dataset_results)
        df_ds.to_csv(os.path.join(output_dir, f"SpatialGlue_{dataset_name}_results.csv"), index=False)

    # Save overall summary CSV
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(output_dir, "SpatialGlue_all_results.csv"), index=False)

    summary_cols = ['ARI', 'Silhouette', 'NMI', 'AMI', 'CHI', 'DBI']
    print("\n" + "=" * 88)
    print(" ALL SPATIALGLUE EXPERIMENTS COMPLETED ".center(88, "="))
    print("=" * 88)
    print(df_all.groupby('dataset')[summary_cols].mean().to_string())
    print("-" * 88)
    print(" OVERALL MEAN ACROSS ALL DATASETS & SEEDS ".center(88, "-"))
    print(df_all[summary_cols].mean().to_frame().T.to_string(index=False))
    print("=" * 88 + "\n")

    return df_all


# ==============================================================================
# 12. Command-Line Interface (CLI)
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="SpatialGlue Spatial Multi-Omics Model CLI & Live Dashboard Integration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--datasets', nargs='+', default=['all'],
        help="Datasets to run. Can be index (0 to 5), name (e.g. 10x_human_lymph_node_A1), or 'all'."
    )
    parser.add_argument(
        '--seeds', nargs='+', default=None,
        help="List of random seeds to evaluate (e.g. --seeds 42 0 1 7 or --seeds all)."
    )
    parser.add_argument(
        '--n_seeds', type=int, default=None,
        help="Use first N seeds from default seed list."
    )
    parser.add_argument(
        '--epochs', type=int, default=None,
        help="Number of epochs to train. If not set, dataset-specific default is used."
    )
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.00, help="Weight decay for optimizer")
    parser.add_argument('--dim_output', type=int, default=64, help="Dimension of output representation")
    parser.add_argument('--tool', type=str, default='mclust', choices=['mclust', 'kmeans', 'leiden', 'louvain'], help="Clustering method")
    parser.add_argument('--device', type=str, default=None, help="Compute device ('cuda', 'cuda:0', or 'cpu')")
    parser.add_argument('--visualize', action='store_true', default=True, help="Generate and save all plots (Curves, Spatial, UMAP, Violin)")
    parser.add_argument('--show_plots', action='store_true', default=False, help="Display matplotlib interactive plots")
    parser.add_argument('--output_dir', type=str, default='results', help="Directory to save CSV results and plots")
    parser.add_argument('--data_dir', type=str, default='data', help="Directory to save/load datasets")
    parser.add_argument('--api_url', type=str, default="https://model-performance.vercel.app/api/experiments/upload", help="Live Dashboard endpoint URL")

    cli_args = parser.parse_args()

    run_experiment(
        datasets=cli_args.datasets if len(cli_args.datasets) > 1 or cli_args.datasets[0] != 'all' else 'all',
        seeds=cli_args.seeds,
        n_seeds=cli_args.n_seeds,
        epochs=cli_args.epochs,
        lr=cli_args.lr,
        weight_decay=cli_args.weight_decay,
        dim_output=cli_args.dim_output,
        tool=cli_args.tool,
        device=cli_args.device,
        visualize=cli_args.visualize,
        show_plots=cli_args.show_plots,
        output_dir=cli_args.output_dir,
        data_dir=cli_args.data_dir,
        api_url=cli_args.api_url
    )
