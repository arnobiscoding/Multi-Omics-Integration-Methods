#!/usr/bin/env python3
"""
spaLLM Multi-Seed Pipeline (Standalone Python Script)
Integrates joint spatial representation learning workflow for target datasets:
- Mouse Brain E11, E13, E15, E18 (RNA + ATAC)
- Human Lymph Node A1, D1 (RNA + Protein/ADT)
"""

# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 2
# ===========================================================================
# ==================================================================
# RUN CONFIGURATION
# ==================================================================
import numpy as np

ENV_MODE = "auto"

ALL_DATASETS_CONFIG = {
    "mouse-brain-e11-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/mouse-brain-e11-s1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "mouse-brain-e13-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/mouse-brain-e13-s1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "mouse-brain-e15-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/mouse-brain-e15-s1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "mouse-brain-e18-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/mouse-brain-e18-s1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "human-lymph-node-a1": {
        "type": "human_lymph_node",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/lymph-node-data/Dataset11_Human_Lymph_Node_A1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset11_Human_Lymph_Node_A1/",
        "mod2_candidates": ["adata_ADT.h5ad"],
        "anno_file": "annotation.csv"
    },
    "human-lymph-node-d1": {
        "type": "human_lymph_node",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/human-lymph-node-d1/10x_human_lymph_node_D1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset12_Human_Lymph_Node_D1/",
        "mod2_candidates": ["adata_ADT.h5ad"],
        "anno_file": "annotation.csv"
    }
}

ACTIVE_DATASETS = ["all"]

MASTER_SEED = 42
rng = np.random.default_rng(MASTER_SEED)
SEEDS = rng.integers(low=1, high=2**31 - 1, size=10).tolist()

if len(ACTIVE_DATASETS) == 1 and ACTIVE_DATASETS[0].lower() == "all":
    datasets_to_run = list(ALL_DATASETS_CONFIG.keys())
else:
    datasets_to_run = [d for d in ACTIVE_DATASETS if d in ALL_DATASETS_CONFIG]

print(f"Scheduled datasets: {datasets_to_run}")
print(f"Ablation seeds: {SEEDS}")


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 4
# ===========================================================================
# --- COMBINED spaLLM SOURCE CODE ---
import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.backends import cudnn
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from scipy.sparse import coo_matrix
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score
)

def init_weights(*params):
    """Initialize weights with Xavier uniform distribution."""
    for param in params:
        torch.nn.init.xavier_uniform_(param)

class DeepEncoder(Module):
    """Modality-specific GNN encoder."""
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super().__init__()
        self.dropout = dropout
        self.act = act
        self.hidden_dim = out_feat * 2

        self.weights = torch.nn.ParameterList([
            Parameter(torch.FloatTensor(in_feat, self.hidden_dim)),
            Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim)),
            Parameter(torch.FloatTensor(self.hidden_dim, out_feat))
        ])
        init_weights(*self.weights)

    def forward(self, feat, adj):
        x = self._apply_layer(feat, adj, self.weights[0])
        x = self._apply_layer(x, adj, self.weights[1])
        x = torch.spmm(adj, torch.mm(x, self.weights[2]))
        return x

    def _apply_layer(self, x, adj, weight):
        x = torch.spmm(adj, torch.mm(x, weight))
        x = self.act(x)
        return F.dropout(x, self.dropout, training=self.training)

class CellEmbedding(Module):
    """Modality-specific cell embedding encoder/decoder."""
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        init_weights(self.weight)

    def forward(self, feat, adj):
        return torch.spmm(adj, torch.mm(feat, self.weight))

class AttentionLayer(Module):
    """Generic Attention Layer."""
    def __init__(self, in_feat, out_feat):
        super().__init__()
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        init_weights(self.w_omega, self.u_omega)

    def forward(self, *embeddings):
        emb_stack = torch.cat([torch.unsqueeze(emb, dim=1) for emb in embeddings], dim=1)
        v = torch.tanh(torch.matmul(emb_stack, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(vu.squeeze(-1) + 1e-6, dim=1)
        emb_combined = torch.matmul(emb_stack.transpose(1, 2), alpha.unsqueeze(-1)).squeeze(-1)
        return emb_combined, alpha

class EncodingNetwork(Module):
    """Encoding network with modality-specific encoders, decoders, and attention layers."""
    def __init__(self, dim_in_omics1, dim_out_omics1, dim_in_omics2, dim_out_omics2):
        super().__init__()
        self.encoder_embedding = CellEmbedding(512, 64)
        self.decoder_embedding = CellEmbedding(64, 512)

        self.encoder_omics1 = DeepEncoder(dim_in_omics1, dim_out_omics1)
        self.decoder_omics1 = DeepEncoder(dim_out_omics1, dim_in_omics1)
        self.encoder_omics2 = DeepEncoder(dim_in_omics2, dim_out_omics2)
        self.decoder_omics2 = DeepEncoder(dim_out_omics2, dim_in_omics2)

        self.atten_feature1 = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_feature2 = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_feature = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_omics2 = AttentionLayer(dim_out_omics2, dim_out_omics2)
        self.atten_cross = AttentionLayer(dim_out_omics1, dim_out_omics2)

    def forward(self, f_omics1, f_omics2, adj_spa1, adj_fea1, adj_spa2, adj_fea2, cell_emb, adj_emb):
        emb_spa = self.encoder_embedding(cell_emb, adj_spa1)
        emb_fea = self.encoder_embedding(cell_emb, adj_emb)

        emb_latent_spa1 = self.encoder_omics1(f_omics1, adj_spa1)
        emb_latent_spa2 = self.encoder_omics2(f_omics2, adj_spa2)
        emb_latent_fea1 = self.encoder_omics1(f_omics1, adj_fea1)
        emb_latent_fea2 = self.encoder_omics2(f_omics2, adj_fea2)

        emb_att1, alpha_att1 = self.atten_feature1(emb_spa, emb_latent_spa1)
        emb_att2, alpha_att2 = self.atten_feature2(emb_fea, emb_latent_fea1)
        emb_latent_omics1, alpha_att_omics1 = self.atten_feature(emb_att1, emb_att2)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spa2, emb_latent_fea2)

        emb_latent_combined, alpha = self.atten_cross(emb_latent_omics1, emb_latent_omics2)

        emb_recon1 = self.decoder_omics1(emb_latent_combined, adj_spa1)
        emb_recon2 = self.decoder_omics2(emb_latent_combined, adj_spa2)
        emb_recon_spa = self.decoder_embedding(emb_spa, adj_spa1)
        emb_recon_fea = self.decoder_embedding(emb_fea, adj_emb)

        emb_cross1 = self.encoder_omics2(self.decoder_omics2(emb_latent_omics1, adj_spa2), adj_spa2)
        emb_cross2 = self.encoder_omics1(self.decoder_omics1(emb_latent_omics2, adj_spa1), adj_spa1)

        return {
            'emb_latent_omics1': emb_latent_omics1, 'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined, 'emb_recon_omics1': emb_recon1, 'emb_recon_omics2': emb_recon2,
            'emb_cross1': emb_cross1, 'emb_cross2': emb_cross2,
            'alpha_att1': alpha_att1, 'alpha_att2': alpha_att2, 'alpha_omics1': alpha_att_omics1,
            'alpha_omics2': alpha_omics2, 'alpha': alpha, 'emb_recon_spa': emb_recon_spa,
            'emb_recon_fea': emb_recon_fea
        }

def fix_seed(seed: int):
    """Fix random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='SPOTS', n_neighbors=3):
    """Construct spatial and feature neighbor graphs."""
    if datatype == 'Spatial-epigenome-transcriptome':
        n_neighbors = 6

    def _construct_spatial_graph(adata):
        cell_position = adata.obsm['spatial']
        return construct_graph_by_coordinate(cell_position, n_neighbors)

    adata_omics1.uns['adj_spatial'] = _construct_spatial_graph(adata_omics1)
    adata_omics2.uns['adj_spatial'] = _construct_spatial_graph(adata_omics2)

    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = construct_graph_by_feature(
        adata_omics1, adata_omics2
    )
    return {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}

def pca(adata: ad.AnnData, use_reps=None, n_comps=10):
    """Perform PCA for dimensionality reduction."""
    pca_model = PCA(n_components=n_comps)
    data = adata.obsm[use_reps] if use_reps else adata.X
    data = data.toarray() if sp.issparse(data) else data
    return pca_model.fit_transform(data)

def clr_normalize_each_cell(adata: ad.AnnData, inplace=True):
    """Normalize count vector for each cell using CLR normalization."""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x)) if len(x) > 0 else 1.0
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    adata.X = np.apply_along_axis(
        seurat_clr, 1, adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    )
    return adata

def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode="connectivity", metric="correlation"):
    """Construct feature neighbor graphs based on expression profiles."""
    graph_omics1 = kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=False)
    graph_omics2 = kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=False)
    return graph_omics1, graph_omics2

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    """Construct spatial graph based on spatial coordinates."""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(cell_position)
    _, indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    return pd.DataFrame({'x': x, 'y': y, 'value': 1})

def transform_adjacent_matrix(adj_df):
    """Transform adjacency dataframe into sparse matrix."""
    n_spot = adj_df['x'].max() + 1
    return coo_matrix((adj_df['value'], (adj_df['x'], adj_df['y'])), shape=(n_spot, n_spot))

def preprocess_graph(adj):
    """Normalize adjacency matrix for GNN input."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    degree_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    return sparse_mx_to_torch_sparse_tensor(adj.dot(degree_inv_sqrt).T.dot(degree_inv_sqrt))

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert scipy sparse matrix to torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    return torch.sparse.FloatTensor(indices, torch.from_numpy(sparse_mx.data), torch.Size(sparse_mx.shape))

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2, adj_emb):
    """Preprocess spatial and feature adjacency matrices for GNNs."""
    def _process_adj(adj):
        adj = adj.toarray() + adj.toarray().T
        adj = np.where(adj > 1, 1, adj)
        return preprocess_graph(adj)

    adj_spatial_omics1 = _process_adj(transform_adjacent_matrix(adata_omics1.uns['adj_spatial']))
    adj_spatial_omics2 = _process_adj(transform_adjacent_matrix(adata_omics2.uns['adj_spatial']))
    adj_emb = _process_adj(adj_emb)

    def _process_feature_adj(adj):
        adj = adj + adj.T
        return preprocess_graph(np.where(adj > 1, 1, adj))

    adj_feature_omics1 = _process_feature_adj(torch.FloatTensor(adata_omics1.obsm['adj_feature'].toarray()))
    adj_feature_omics2 = _process_feature_adj(torch.FloatTensor(adata_omics2.obsm['adj_feature'].toarray()))

    return {
        'adj_spatial_omics1': adj_spatial_omics1,
        'adj_spatial_omics2': adj_spatial_omics2,
        'adj_feature_omics1': adj_feature_omics1,
        'adj_feature_omics2': adj_feature_omics2,
        'adj_emb': adj_emb
    }

def run_mclust(data_matrix, n_clusters, seed=2024, max_dims=30):
    """Perform mclust clustering via rpy2 with automatic PCA dimension reduction for high-dim inputs."""
    data_mat = np.array(data_matrix, dtype=np.float64)
    if data_mat.shape[1] > max_dims:
        n_comps = min(max_dims, data_mat.shape[0] - 1, data_mat.shape[1])
        print(f"Reducing feature dimension from {data_mat.shape[1]} to {n_comps} PCA components for mclust...")
        pca_model = PCA(n_components=n_comps, random_state=seed)
        data_mat = pca_model.fit_transform(data_mat)

    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        
        try:
            robjects.r.library("mclust")
        except Exception:
            print("Installing R package 'mclust'...")
            robjects.r('install.packages("mclust", repos="https://cloud.r-project.org", quiet=TRUE)')
            robjects.r.library("mclust")
            
        robjects.r['set.seed'](seed)
        r_matrix = robjects.r['matrix'](
            robjects.FloatVector(data_mat.flatten()),
            nrow=data_mat.shape[0],
            ncol=data_mat.shape[1],
            byrow=True
        )
        res = robjects.r['Mclust'](r_matrix, n_clusters, "EEE")
        labels = np.array(res.rx2('classification')).astype(int)
        return labels.astype(str)
    except Exception as e:
        print(f"Warning: mclust execution via rpy2 failed ({e}). Returning fallback clusters.")
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(data_mat).astype(str)

def clustering(adata, n_clusters=7, key='emb', add_key='spaLLM', method='leiden', **kwargs):
    """Spatial clustering using `mclust`, `leiden`, or `louvain`."""
    use_pca = kwargs.get('use_pca', False)
    n_comps = kwargs.get('n_comps', 20)
    if use_pca:
        adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
        key = key + '_pca'

    if method == 'mclust':
        adata.obs['mclust'] = run_mclust(adata.obsm[key], n_clusters)
        adata.obs[add_key] = adata.obs['mclust']
    if method in ['leiden', 'louvain']:
        search_kwargs = {k: v for k, v in kwargs.items() if k not in ['method', 'use_rep']}
        res = search_res(adata, n_clusters, method=method, use_rep=key, **search_kwargs)
        try:
            sc.tl.leiden(adata, random_state=0, resolution=res, flavor='igraph', n_iterations=2, directed=False)
        except TypeError:
            sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs[method].astype(str) if method in adata.obs else None

def search_res(adata, n_clusters, method='leiden', use_rep='spaLLM', start=0.1, end=3.0, increment=0.01, **kwargs):
    """Search for resolution to achieve target cluster count using `leiden` or `louvain`."""
    print('Searching resolution...')
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in np.arange(start, end, increment):
        res = round(res, 3)
        if method == 'leiden':
            try:
                sc.tl.leiden(adata, random_state=0, resolution=res, flavor='igraph', n_iterations=2, directed=False)
            except TypeError:
                sc.tl.leiden(adata, random_state=0, resolution=res)
            clusters = adata.obs['leiden']
        elif method == 'louvain':
            sc.tl.louvain(adata, random_state=0, resolution=res)
            clusters = adata.obs['louvain']
        count_unique = clusters.nunique()
        if count_unique == n_clusters:
            return res
    print(f'Warning: Target cluster count {n_clusters} not reached exact match. Using res=0.5')
    return 0.5

def evaluate_clustering(y_true_series, y_pred_series, features_matrix, name=""):
    """Evaluate clustering performance using ARI, NMI, AMI, Homogeneity, V-measure, and Silhouette."""
    mask = (y_true_series != 'Exclude') & (y_true_series != 'unknown') & (y_true_series.notna())
    y_true = y_true_series[mask].astype(str)
    y_pred = y_pred_series[mask].astype(str)
    feats = features_matrix[mask]

    if len(y_true) == 0 or len(np.unique(y_pred)) < 2:
        return {
            'ARI': 0.0, 'NMI': 0.0, 'AMI': 0.0,
            'Homogeneity': 0.0, 'V-measure': 0.0, 'Silhouette': 0.0
        }

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    v_meas = v_measure_score(y_true, y_pred)
    sil = silhouette_score(feats, y_pred)

    metrics = {
        'ARI': float(ari),
        'NMI': float(nmi),
        'AMI': float(ami),
        'Homogeneity': float(homo),
        'V-measure': float(v_meas),
        'Silhouette': float(sil)
    }

    if name:
        print(f"\n--- Clustering Evaluation ({name}) ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    return metrics

def plot_spallm_visualizations(adata, title_prefix="", dname="dataset", seed=2024, save_dir=None):
    """Plot UMAP and Spatial visualizations for Ground Truth, KMeans, Leiden, and mclust, and save PNG image."""
    sc.pp.neighbors(adata, use_rep='spaLLM', n_neighbors=10)
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
    save_path = os.path.join(save_dir, f"spallm_plot_{dname}_seed_{seed}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization plot image to: {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)

def plot_spallm_summary_boxplots(df_all, save_dir=None):
    """Generate and save Box & Whiskers plots comparing KMeans, Leiden, and mclust performance side-by-side across all datasets."""
    if df_all.empty or "cluster alg" not in df_all.columns:
        print("No valid metrics found for Box & Whiskers plots.")
        return

    metrics = ["ARI", "NMI", "AMI", "Homogeneity", "V-measure", "Silhouette"]
    metrics = [m for m in metrics if m in df_all.columns]
    if not metrics:
        return

    if save_dir is None:
        save_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
    os.makedirs(save_dir, exist_ok=True)

    sns.set_theme(style="whitegrid")
    # 1. Individual Metric Boxplots
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
        ax.set_title(f"spaLLM Model: {metric} Performance Across Datasets", fontsize=14, fontweight='bold')
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title="Cluster Alg", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"spallm_boxplot_{metric.lower().replace('-', '_')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric} Boxplot image to: {save_path}")
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig)

    # 2. Combined 3x2 Multi-Panel Figure
    fig, axes = plt.subplots(3, 2, figsize=(18, 16))
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
    combined_path = os.path.join(save_dir, "spallm_boxplot_all_metrics.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined multi-metric Boxplot figure to: {combined_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)

# 4. spaLLM_util.py
class Train_spaLLM:
    def __init__(self, data, embedding, datatype='10x', device=torch.device('cpu'),
                 random_seed=2024, learning_rate=0.0001, weight_decay=0.0, epochs=600,
                 dim_input=3000, dim_output=64, weight_factors=None):
        self.device = device
        self.data = data.copy()
        self.embedding = torch.from_numpy(embedding).to(device)
        self.datatype = datatype
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_factors = weight_factors or [5, 5, 1, 10, 10, 10]

        self._init_adj_and_features()
        self.loss_history = []
        self._adjust_hyperparameters()

    def _init_adj_and_features(self):
        """Initialize adjacency matrices and input features."""
        adj = adjacent_matrix_preprocessing(self.data['adata_omics1'], self.data['adata_omics2'], self.data['adj_emb'])
        self.adj_spatial_omics1 = adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = adj['adj_feature_omics2'].to(self.device)
        self.adj_emb = adj['adj_emb'].to(self.device)

        self.features_omics1 = torch.FloatTensor(self.data['adata_omics1'].obsm['feat']).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.data['adata_omics2'].obsm['feat']).to(self.device)
        self.dim_input1, self.dim_input2 = self.features_omics1.shape[1], self.features_omics2.shape[1]

    def _adjust_hyperparameters(self):
        """Adjust hyperparameters based on data type."""
        if self.datatype == 'SPOTS':
            self.epochs, self.weight_factors = 600, [1, 5, 1, 1, 5, 5]
        elif self.datatype == '10x':
            self.epochs, self.weight_factors = 200, [5, 5, 1, 10, 10, 10]
        elif self.datatype == 'Spatial-epigenome-transcriptome':
            self.epochs, self.weight_factors = 1600, [1, 5, 1, 1, 10, 10]

    def _add_noise(self):
        """Apply Gaussian noise to features and embeddings."""
        features_omics1_noisy = add_gaussian_noise(self.features_omics1, mean=0, std=0.1)
        embedding_noisy = add_gaussian_noise(self.embedding, mean=0, std=0.01)
        return features_omics1_noisy, embedding_noisy

    def _calculate_losses(self, results):
        """Calculate reconstruction and correspondence losses."""
        loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
        loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
        loss_rec_es = F.mse_loss(self.embedding, results['emb_recon_spa'])
        loss_rec_ef = F.mse_loss(self.embedding, results['emb_recon_fea'])
        loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_cross1'])
        loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_cross2'])

        loss = (self.weight_factors[0] * loss_recon_omics1 +
                self.weight_factors[1] * loss_recon_omics2 +
                self.weight_factors[2] * loss_corr_omics1 +
                self.weight_factors[3] * loss_corr_omics2 +
                self.weight_factors[4] * loss_rec_es +
                self.weight_factors[5] * loss_rec_ef)
        return loss

    def train(self, epochs=None):
        epochs = epochs or self.epochs
        self.model = EncodingNetwork(self.dim_input1, self.dim_output, self.dim_input2, self.dim_output).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in range(epochs):
            optimizer.zero_grad()
            if random.random() < 0.5:
                features_omics1, embedding = self._add_noise()
            else:
                features_omics1, embedding = self.features_omics1, self.embedding

            results = self.model(features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1,
                                 self.adj_spatial_omics2, self.adj_feature_omics2, embedding, self.adj_emb)
            loss = self._calculate_losses(results)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

        return self._evaluate_model()

    def _evaluate_model(self):
        """Evaluate the model and return output embeddings."""
        self.model.eval()
        with torch.no_grad():
            results = self.model(self.features_omics1, self.features_omics2, self.adj_spatial_omics1,
                                 self.adj_feature_omics1, self.adj_spatial_omics2, self.adj_feature_omics2,
                                 self.embedding, self.adj_emb)

        return {
            'emb_latent_omics1': F.normalize(results['emb_latent_omics1'], p=2).cpu().numpy(),
            'emb_latent_omics2': F.normalize(results['emb_latent_omics2'], p=2).cpu().numpy(),
            'spaLLM': F.normalize(results['emb_latent_combined'], p=2).cpu().numpy(),
            'alpha_omics1': results['alpha_omics1'].cpu().numpy(),
            'alpha_omics2': results['alpha_omics2'].cpu().numpy(),
            'alpha': results['alpha'].cpu().numpy(),
            'alpha_att1': results['alpha_att1'].cpu().numpy(),
            'alpha_att2': results['alpha_att2'].cpu().numpy()
        }

    def plot_loss(self, save_path=None):
        """Plot and save the training loss curve."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(self.loss_history, label='Training Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curve')
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig)


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 6
# ===========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} (If this says 'cpu', make sure GPU is enabled!)")


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 8
# ===========================================================================
def run_spallm_workflow(dataset_name, dataset_cfg, env_mode, seed, device, show_plots=False):
    print(f"\n--- Running Seed: {seed} ---")
    fix_seed(seed)
    
    # 1. Resolve paths
    is_kaggle = os.path.exists('/kaggle/input')
    if env_mode == "kaggle" or (env_mode == "auto" and is_kaggle):
        data_dir = dataset_cfg["kaggle_dir"]
        model_dir = '/kaggle/input/datasets/sadmanbiazidarnob/scgpt-human/scGPT_human'
    else:
        data_dir = dataset_cfg["local_dir"]
        model_dir = r"D:/FYDP/spaLLM/spaLLM/scGPT_human"
        
    dataset_type = dataset_cfg["type"]
    
    print(f"Loading data from: {data_dir}")
    adata_rna = sc.read_h5ad(os.path.join(data_dir, 'adata_RNA.h5ad'))
    
    mod2_filename = None
    for cand in dataset_cfg["mod2_candidates"]:
        if os.path.exists(os.path.join(data_dir, cand)):
            mod2_filename = cand
            break
    if mod2_filename is None:
        mod2_filename = dataset_cfg["mod2_candidates"][0]
        
    adata_mod2 = sc.read_h5ad(os.path.join(data_dir, mod2_filename))
    
    annotation_filename = dataset_cfg["anno_file"]
    annotation_path = os.path.join(data_dir, annotation_filename)
    
    if os.path.exists(annotation_path):
        print(f"Loading annotations from: {annotation_path}")
        annotation = pd.read_csv(annotation_path)
        
        for col in ['Barcode', 'barcode']:
            if col in annotation.columns:
                annotation = annotation.rename(columns={col: 'barcode'})
                break
                
        for col in ['manual-anno', 'cluster', 'ground_truth']:
            if col in annotation.columns:
                annotation = annotation.rename(columns={col: 'ground_truth'})
                break
                
        annotation = annotation.set_index('barcode')
        
        adata_rna.obs = adata_rna.obs.join(annotation, how='left')
        adata_rna.obs['ground_truth'] = adata_rna.obs['ground_truth'].fillna('unknown')
        adata_mod2.obs = adata_mod2.obs.join(annotation, how='left')
        adata_mod2.obs['ground_truth'] = adata_mod2.obs['ground_truth'].fillna('unknown')
    else:
        print(f"Warning: Annotation file '{annotation_filename}' not found. Defaulting to 'unknown'.")
        adata_rna.obs['ground_truth'] = 'unknown'
        adata_mod2.obs['ground_truth'] = 'unknown'
        
    print("RNA shape:", adata_rna.shape)
    print("Modality 2 shape:", adata_mod2.shape)
    
    # 2. scGPT Embedding
    adata_rna.var_names_make_unique()
    adata_mod2.var_names_make_unique()
    adata_rna.var['gene_names'] = adata_rna.var.index.str.upper()
    
    if os.path.exists(model_dir):
        print(f"Running scGPT embedding on GPU using model at: {model_dir}")
        from scgpt.tasks.cell_emb import embed_data
        adata_emb = embed_data(
            adata_or_file=adata_rna.copy(), 
            model_dir=model_dir, 
            gene_col="gene_names", 
            max_length=1200, 
            batch_size=64, 
            obs_to_save=None, 
            device=device, 
            use_fast_transformer=False, 
            return_new_adata=False
        )
        embedding = adata_emb.obsm["X_scGPT"]
        print("scGPT embedding generated successfully.")
    else:
        print(f"Warning: scGPT model directory '{model_dir}' not found. Generating dummy mock embeddings...")
        embedding = np.random.normal(size=(adata_rna.n_obs, 512))
        
    # 3. Preprocessing
    print("Preprocessing RNA data...")
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna)
    
    adata_rna_high = adata_rna[:, adata_rna.var['highly_variable']]
    
    if dataset_type == "human_lymph_node":
        n_comps_rna = adata_mod2.n_vars - 1
        print(f"Using {n_comps_rna} components for RNA PCA (based on ADT n_vars)")
        adata_rna.obsm['feat'] = pca(adata_rna_high, n_comps=n_comps_rna)
        
        print("Preprocessing Protein (ADT) data...")
        adata_mod2 = clr_normalize_each_cell(adata_mod2)
        sc.pp.scale(adata_mod2)
        adata_mod2.obsm['feat'] = pca(adata_mod2, n_comps=adata_mod2.n_vars - 1)
    else:
        n_comps_rna = min(50, adata_rna_high.n_obs - 1, adata_rna_high.n_vars - 1)
        print(f"Using {n_comps_rna} components for RNA PCA")
        adata_rna.obsm['feat'] = pca(adata_rna_high, n_comps=n_comps_rna)
        
        print("Preprocessing Epigenome (ATAC) data...")
        sc.pp.normalize_total(adata_mod2, target_sum=1e4)
        sc.pp.log1p(adata_mod2)
        sc.pp.scale(adata_mod2)
        n_comps_mod2 = min(50, adata_mod2.n_obs - 1, adata_mod2.n_vars - 1)
        print(f"Using {n_comps_mod2} components for ATAC PCA")
        adata_mod2.obsm['feat'] = pca(adata_mod2, n_comps=n_comps_mod2)
        
    # 4. Training GNN
    print("Constructing neighbor graphs...")
    data = construct_neighbor_graph(adata_rna, adata_mod2, datatype='10x')
    data['adj_emb'] = kneighbors_graph(embedding, 20, mode="connectivity", metric="correlation", include_self=False)
    
    print("Training spaLLM...")
    model = Train_spaLLM(data, datatype='10x', device=device, embedding=embedding)
    output = model.train(epochs=800)
    
    adata = adata_rna.copy()
    for key, value in output.items():
        adata.obsm[key] = value
    print("Training complete.")
    
    # 5. Tri-Algorithm Clustering (KMeans, Leiden, mclust) on spaLLM Embedding
    valid_mask = (adata.obs['ground_truth'] != 'unknown') & (adata.obs['ground_truth'] != 'Exclude')
    valid_labels = adata.obs['ground_truth'][valid_mask]
    
    if len(valid_labels) > 0:
        n_clusters = len(np.unique(valid_labels))
        print(f"Clustering into {n_clusters} clusters (based on ground truth)...")
    else:
        n_clusters = 7
        print(f"No valid ground truth annotations. Clustering into default {n_clusters} clusters...")
        
    spallm_emb = adata.obsm['spaLLM']

    # 5a. KMeans
    print(f"Running KMeans on spaLLM embeddings with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    adata.obs['kmeans'] = kmeans.fit_predict(spallm_emb).astype(str)

    # 5b. Leiden
    print("Running Leiden on spaLLM embeddings...")
    clustering(adata, key='spaLLM', add_key='spaLLM', n_clusters=n_clusters, method='leiden', use_pca=False)
    adata.obs['leiden'] = adata.obs['spaLLM'].astype(str)

    # 5c. mclust
    print("Running mclust via rpy2 on spaLLM embeddings...")
    adata.obs['mclust'] = run_mclust(spallm_emb, n_clusters, seed=seed)

    # 6. Evaluation
    metrics_kmeans = evaluate_clustering(
        adata.obs['ground_truth'],
        adata.obs['kmeans'],
        spallm_emb,
        name=f"spaLLM {dataset_name} Seed {seed} - KMeans"
    )

    metrics_leiden = evaluate_clustering(
        adata.obs['ground_truth'],
        adata.obs['leiden'],
        spallm_emb,
        name=f"spaLLM {dataset_name} Seed {seed} - Leiden"
    )

    metrics_mclust = evaluate_clustering(
        adata.obs['ground_truth'],
        adata.obs['mclust'],
        spallm_emb,
        name=f"spaLLM {dataset_name} Seed {seed} - mclust"
    )

    row_kmeans = {
        'cluster alg': 'KMeans',
        'ARI': metrics_kmeans['ARI'],
        'NMI': metrics_kmeans['NMI'],
        'AMI': metrics_kmeans['AMI'],
        'Homogeneity': metrics_kmeans['Homogeneity'],
        'V-measure': metrics_kmeans['V-measure'],
        'Silhouette': metrics_kmeans['Silhouette'],
        'resolution': None
    }

    row_leiden = {
        'cluster alg': 'Leiden',
        'ARI': metrics_leiden['ARI'],
        'NMI': metrics_leiden['NMI'],
        'AMI': metrics_leiden['AMI'],
        'Homogeneity': metrics_leiden['Homogeneity'],
        'V-measure': metrics_leiden['V-measure'],
        'Silhouette': metrics_leiden['Silhouette'],
        'resolution': None
    }

    row_mclust = {
        'cluster alg': 'mclust',
        'ARI': metrics_mclust['ARI'],
        'NMI': metrics_mclust['NMI'],
        'AMI': metrics_mclust['AMI'],
        'Homogeneity': metrics_mclust['Homogeneity'],
        'V-measure': metrics_mclust['V-measure'],
        'Silhouette': metrics_mclust['Silhouette'],
        'resolution': None
    }

    # 7. Visualization Plot Saving (First seed run only)
    if show_plots:
        plot_spallm_visualizations(adata, title_prefix=f"{dataset_name} (Seed {seed})", dname=dataset_name, seed=seed)
        output_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
        model.plot_loss(save_path=os.path.join(output_dir, f"spallm_loss_{dataset_name}_seed_{seed}.png"))
        
    return [row_kmeans, row_leiden, row_mclust]


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 10
# ===========================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    all_results_flat = []
    all_results = {}

    for dname in datasets_to_run:
        cfg = ALL_DATASETS_CONFIG[dname]
        all_results[dname] = []
        
        print(f"\n=======================================================")
        print(f"STARTING WORKFLOW FOR DATASET: {dname} OVER {len(SEEDS)} SEEDS")
        print(f"=======================================================")
        
        for idx, seed in enumerate(SEEDS):
            show_plots = (idx == 0)
            try:
                results_list = run_spallm_workflow(dname, cfg, ENV_MODE, seed, device, show_plots=show_plots)
                if results_list:
                    for row in results_list:
                        res_row = {"dataset": dname, "seed": seed}
                        res_row.update(row)
                        all_results[dname].append(res_row)
                        all_results_flat.append(res_row)
            except Exception as e:
                print(f"Error processing dataset {dname} with seed {seed}: {e}")
                
        if len(all_results[dname]) > 0:
            df_metrics = pd.DataFrame(all_results[dname])
            print(f"\n=======================================================")
            print(f"AVERAGE PERFORMANCE FOR {dname} ({len(SEEDS)} seeds)")
            print(f"=======================================================")
            numeric_cols = ["ARI", "NMI", "AMI", "Homogeneity", "V-measure", "Silhouette"]
            for alg, df_alg in df_metrics.groupby("cluster alg"):
                print(f"\n--- {alg} Performance (Mean ± Std) ---")
                means = df_alg[numeric_cols].mean()
                stds = df_alg[numeric_cols].std()
                summary_df = pd.DataFrame({"Mean": means, "Std": stds})
                print(summary_df.to_string())
            print(f"=======================================================\n")
        else:
            print(f"No evaluation metrics collected for {dname}.")

    # Save all collected metrics to CSV and generate Box & Whiskers plots
    if len(all_results_flat) > 0:
        df_all = pd.DataFrame(all_results_flat)
        is_kaggle = os.path.exists('/kaggle/working')
        output_dir = '/kaggle/working' if is_kaggle else '.'
        output_csv = os.path.join(output_dir, 'spallm_ablation_results.csv')
        df_all.to_csv(output_csv, index=False)
        print(f"All ablation study results saved to CSV at: {output_csv}")
        
        print("Generating side-by-side Box & Whiskers plots for all metrics across algorithms...")
        plot_spallm_summary_boxplots(df_all, save_dir=output_dir)
    else:
        print("No results were generated, skipping CSV export and Boxplots.")

if __name__ == '__main__':
    main()


