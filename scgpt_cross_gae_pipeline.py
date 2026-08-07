#!/usr/bin/env python3
"""
scGPT-Cross-GAE Implementation Specification Pipeline
---------------------------------------------------
A lightweight, highly controlled hybrid architecture designed to integrate
Transcriptomic (RNA) and Proteomic/Epigenomic (Mod2) spatial data.
"""

import os
import random
import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Optional, Tuple
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

# Optional PyG import with clean native PyTorch fallback
try:
    from torch_geometric.nn import GATv2Conv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


# ==================================================================
# RUN CONFIGURATION & DATASETS
# ==================================================================
ENV_MODE = "auto"

ALL_DATASETS_CONFIG = {
    "mouse-brain-e11-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E11_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "mouse-brain-e13-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E13_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "mouse-brain-e15-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E15_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "mouse-brain-e18-s1": {
        "type": "mouse_brain",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E18_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "human-lymph-node-a1": {
        "type": "human_lymph_node",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Human_Lymph_Node_A1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset11_Human_Lymph_Node_A1/",
        "mod2_candidates": ["adata_ADT.h5ad"],
        "anno_file": "annotation.csv"
    },
    "human-lymph-node-d1": {
        "type": "human_lymph_node",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Human_Lymph_Node_D1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset12_Human_Lymph_Node_D1/",
        "mod2_candidates": ["adata_ADT.h5ad"],
        "anno_file": "annotation.csv"
    }
}

ACTIVE_DATASETS = ["all"]
SEEDS = [42, 0, 1, 7, 123, 1234, 2022, 2023, 2024, 1337]

if len(ACTIVE_DATASETS) == 1 and ACTIVE_DATASETS[0].lower() == "all":
    datasets_to_run = list(ALL_DATASETS_CONFIG.keys())
else:
    datasets_to_run = [d for d in ACTIVE_DATASETS if d in ALL_DATASETS_CONFIG]


# ==================================================================
# UTILITY & DATA PREPROCESSING FUNCTIONS
# ==================================================================
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

from sklearn.decomposition import PCA, TruncatedSVD

def pca(adata: ad.AnnData, use_reps=None, n_comps=50):
    """Perform memory-efficient PCA / TruncatedSVD for dimensionality reduction."""
    data = adata.obsm[use_reps] if use_reps else adata.X
    if sp.issparse(data):
        svd = TruncatedSVD(n_components=n_comps, random_state=42)
        return svd.fit_transform(data).astype(np.float32)
    else:
        n_comps = min(n_comps, data.shape[0] - 1, data.shape[1] - 1)
        pca_model = PCA(n_components=n_comps, random_state=42)
        return pca_model.fit_transform(data).astype(np.float32)

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

import warnings
warnings.filterwarnings("ignore")

def load_dataset_data(dname, cfg, env_mode="auto"):
    """Dynamically load RNA, Modality 2 (ATAC/ADT), and annotations for any specified dataset."""
    is_kaggle = (env_mode == "kaggle") or (env_mode == "auto" and os.path.exists("/kaggle/input"))
    data_dir = cfg["kaggle_dir"] if is_kaggle else cfg["local_dir"]
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Generating dummy spatial dataset for local verification...")
        n_obs = 500
        n_vars_rna = 1000
        n_vars_mod2 = 200
        adata_rna = ad.AnnData(X=np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars_rna)).astype(np.float32))
        adata_mod2 = ad.AnnData(X=np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars_mod2)).astype(np.float32))
        spatial_coords = np.random.uniform(0, 100, size=(n_obs, 2))
        adata_rna.obsm['spatial'] = spatial_coords
        adata_mod2.obsm['spatial'] = spatial_coords
        clusters = np.random.choice(['Region_A', 'Region_B', 'Region_C', 'Region_D', 'Region_E'], size=n_obs)
        adata_rna.obs['ground_truth'] = clusters
        adata_mod2.obs['ground_truth'] = clusters
        adata_rna.var_names_make_unique()
        adata_mod2.var_names_make_unique()
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

    anno_path = os.path.join(data_dir, cfg["anno_file"])
    if os.path.exists(anno_path):
        print(f"Loading annotations from: {anno_path}")
        annotation = pd.read_csv(anno_path)
        
        ground_col = None
        for col in ['cluster', 'manual-anno', 'ground_truth', 'assigned_cluster']:
            if col in annotation.columns:
                ground_col = col
                break
                
        barcode_col = None
        for col in ['barcode', 'Barcode', 'cell_id']:
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

    adata_rna.var_names_make_unique()
    adata_mod2.var_names_make_unique()

    print(f"RNA shape: {adata_rna.shape}")
    print(f"Modality 2 shape: {adata_mod2.shape}")
    return adata_rna, adata_mod2

def get_scgpt_embeddings(adata_rna, device):
    """Generate or load scGPT transcriptomic embeddings (E_RNA in R^{N x 512})."""
    KAGGLE_MODEL_DIR = '/kaggle/input/datasets/sadmanbiazidarnob/scgpt-human/scGPT_human'
    LOCAL_MODEL_DIR = 'D:/FYDP/GATCON/d1_test/scGPT_human'
    model_dir = KAGGLE_MODEL_DIR if os.path.exists(KAGGLE_MODEL_DIR) else LOCAL_MODEL_DIR
    
    try:
        from scgpt.tasks.cell_emb import embed_data
        adata_rna.var_names_make_unique()
        adata_rna.var['gene_names'] = adata_rna.var.index.str.upper()

        if os.path.exists(model_dir):
            print(f"Running scGPT embedding on device {device} using model at: {model_dir}")
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
            return adata_emb.obsm["X_scGPT"]
    except Exception as e:
        print(f"Notice: scGPT pretrained embedding loader skipped ({e}). Using simulated feature anchor.")

    np.random.seed(42)
    return np.random.randn(adata_rna.n_obs, 512).astype(np.float32)


# ==================================================================
# NATIVE GATv2 LAYER IMPLEMENTATION (Fallback & Pure PyTorch)
# ==================================================================
class NativeGATv2Layer(nn.Module):
    r"""
    Native PyTorch GATv2 implementation following the specification formula:
    Z_i = sigma( \sum_{j \in N_pruned(i) \cup {i}} alpha_{ij} W F_{joint, j} )
    alpha_{ij} = softmax_j( LeakyReLU( a^T LeakyReLU( W F_i + W F_j ) ) )
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.Tensor(1, out_features))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        h = self.W(x)  # (N, out_features)

        src, dst = edge_index[0], edge_index[1]
        
        # LeakyReLU( W x_i + W x_j )
        sum_h = self.leaky_relu(h[src] + h[dst])  # (E, out_features)
        
        # a^T LeakyReLU(...)
        e = self.leaky_relu(torch.sum(sum_h * self.a, dim=-1))  # (E,)

        # Softmax over neighborhood dst
        e_exp = torch.exp(e - torch.max(e))
        e_sum = torch.zeros(N, device=x.device).scatter_add_(0, dst, e_exp)
        alpha = e_exp / (e_sum[dst] + 1e-12)  # (E,)
        alpha = self.dropout(alpha)

        # Weighted message passing
        msg = h[src] * alpha.unsqueeze(-1)  # (E, out_features)
        out = torch.zeros(N, self.out_features, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(msg), msg)

        return F.elu(out)


# ==================================================================
# MODEL ARCHITECTURE: scGPT-Cross-GAE
# ==================================================================
class scGPTCrossGAE(nn.Module):
    """
    scGPT-Cross-GAE Architecture:
    - Phase 2: Cross-Attention Fusion (RNA Queries Mod2 Key/Value -> F_joint)
    - Phase 3 & 4: Dynamic Edge Pruning + 1-Layer GATv2 Encoder -> Latent Z
    - Phase 5: Dual Decoders for Feature Reconstruction Loss
    """
    def __init__(
        self,
        dim_rna: int = 512,
        dim_mod2: int = 50,
        latent_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        prune_threshold: float = 0.3
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.prune_threshold = prune_threshold

        # Phase 2: Linear Projections
        self.W_Q = nn.Linear(dim_rna, latent_dim)
        self.W_K = nn.Linear(dim_mod2, latent_dim)
        self.W_V = nn.Linear(dim_mod2, latent_dim)

        # Multi-Head Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Norms & Feed-Forward Network
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # Phase 4: Shallow Spatial Encoder (1-Layer GATv2)
        if PYG_AVAILABLE:
            self.gat_encoder = GATv2Conv(latent_dim, latent_dim, heads=1, concat=False, add_self_loops=True)
        else:
            self.gat_encoder = NativeGATv2Layer(latent_dim, latent_dim, dropout=dropout)

        # Phase 5: Independent Dual Decoders
        self.dec_rna = nn.Linear(latent_dim, dim_rna)
        self.dec_mod2 = nn.Linear(latent_dim, dim_mod2)

    def forward_cross_attention(self, E_rna: torch.Tensor, E_mod2: torch.Tensor) -> torch.Tensor:
        """Phase 2: Project RNA & Mod2 features and run Cross-Attention Fusion."""
        Q = self.W_Q(E_rna)    # (N, D)
        K = self.W_K(E_mod2)   # (N, D)
        V = self.W_V(E_mod2)   # (N, D)

        attn_out, _ = self.cross_attn(Q.unsqueeze(0), K.unsqueeze(0), V.unsqueeze(0))
        attn_out = attn_out.squeeze(0)  # (N, D)

        F_attn = self.norm1(Q + self.dropout(attn_out))
        F_joint = self.norm2(F_attn + self.dropout(self.ffn(F_attn)))
        return F_joint

    def prune_topology(self, F_joint: torch.Tensor, base_edge_index: torch.Tensor) -> torch.Tensor:
        """Phase 3: Dynamic Edge Pruning based on latent cosine similarity threshold theta."""
        src, dst = base_edge_index[0], base_edge_index[1]
        
        norm_F = F.normalize(F_joint, p=2, dim=-1)
        sim_ij = torch.sum(norm_F[src] * norm_F[dst], dim=-1)  # (E,)

        mask = sim_ij >= self.prune_threshold
        pruned_edge_index = base_edge_index[:, mask]
        
        N = F_joint.size(0)
        self_loops = torch.arange(N, device=F_joint.device).unsqueeze(0).repeat(2, 1)
        pruned_edge_index = torch.cat([pruned_edge_index, self_loops], dim=1)
        
        return pruned_edge_index

    def forward(
        self,
        E_rna: torch.Tensor,
        E_mod2: torch.Tensor,
        base_edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Phase 2: Cross Attention
        F_joint = self.forward_cross_attention(E_rna, E_mod2)

        # Phase 3: Dynamic Edge Pruning
        pruned_edge_index = self.prune_topology(F_joint, base_edge_index)

        # Phase 4: 1-Layer GATv2 Spatial Encoder
        if PYG_AVAILABLE:
            Z = self.gat_encoder(F_joint, pruned_edge_index)
        else:
            Z = self.gat_encoder(F_joint, pruned_edge_index)

        # Phase 5: Decoders for Reconstruction
        hat_E_rna = self.dec_rna(Z)
        hat_E_mod2 = self.dec_mod2(Z)

        return Z, hat_E_rna, hat_E_mod2, pruned_edge_index


# ==================================================================
# DUAL-OBJECTIVE LOSS ENGINE
# ==================================================================
def compute_spatially_aware_info_nce_loss(
    Z: torch.Tensor,
    pruned_edge_index: torch.Tensor,
    tau: float = 0.2
) -> torch.Tensor:
    """
    Phase 5: Spatially-Aware InfoNCE Contrastive Loss (L_NCE)
    Positive pairs are edges in pruned_edge_index, all other spots in batch as negative pairs.
    """
    N = Z.size(0)
    Z_norm = F.normalize(Z, p=2, dim=-1)
    
    sim_matrix = torch.matmul(Z_norm, Z_norm.T) / tau
    
    src, dst = pruned_edge_index[0], pruned_edge_index[1]
    non_self_mask = src != dst
    if non_self_mask.sum() > 0:
        src, dst = src[non_self_mask], dst[non_self_mask]
    
    pos_sim = torch.exp(sim_matrix[src, dst])
    
    self_sim = torch.exp(torch.ones(N, device=Z.device) / tau)
    denom = torch.sum(torch.exp(sim_matrix), dim=1) - self_sim
    denom_src = denom[src] + 1e-12
    
    loss_nce = -torch.log((pos_sim / denom_src) + 1e-12).mean()
    return loss_nce

def compute_total_loss(
    E_rna: torch.Tensor,
    E_mod2: torch.Tensor,
    hat_E_rna: torch.Tensor,
    hat_E_mod2: torch.Tensor,
    Z: torch.Tensor,
    pruned_edge_index: torch.Tensor,
    lambda_mod2: float = 1.0,
    gamma_nce: float = 1.0,
    tau: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase 5 Total Loss calculation: L_Total = L_Recon + gamma * L_NCE"""
    loss_rna = F.mse_loss(hat_E_rna, E_rna)
    loss_mod2 = F.mse_loss(hat_E_mod2, E_mod2)
    loss_recon = loss_rna + lambda_mod2 * loss_mod2

    loss_nce = compute_spatially_aware_info_nce_loss(Z, pruned_edge_index, tau=tau)
    loss_total = loss_recon + gamma_nce * loss_nce

    return loss_total, loss_recon, loss_nce


# ==================================================================
# CLUSTERING & EVALUATION ENGINE
# ==================================================================
def run_mclust(data_matrix, n_clusters, seed=2024, max_dims=30):
    """Perform mclust clustering via rpy2 using native R helper function with PCA reduction."""
    data_mat = np.array(data_matrix, dtype=np.float64)
    if data_mat.shape[1] > max_dims:
        n_comps = min(max_dims, data_mat.shape[0] - 1, data_mat.shape[1])
        pca_model = PCA(n_components=n_comps, random_state=seed)
        data_mat = pca_model.fit_transform(data_mat)

    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
        
        try:
            robjects.r.library("mclust")
        except Exception:
            robjects.r('install.packages("mclust", repos="https://cloud.r-project.org", quiet=TRUE)')
            robjects.r.library("mclust")
            
        r_code = '''
        run_mclust_native <- function(mat, n_clusters, seed) {
            suppressPackageStartupMessages(library(mclust))
            set.seed(seed)
            mat <- as.matrix(mat)
            dimnames(mat) <- NULL
            res <- Mclust(mat, G=n_clusters, modelNames="EEE")
            if (is.null(res)) {
                res <- Mclust(mat, G=n_clusters)
            }
            return(as.integer(res$classification))
        }
        '''
        robjects.r(r_code)
        robjects.globalenv['tmp_mclust_mat'] = numpy2ri.numpy2rpy(data_mat)
        res = robjects.r(f'run_mclust_native(tmp_mclust_mat, {n_clusters}, {seed})')
        labels = np.array(res).astype(int)
        return labels.astype(str)
    except Exception:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(data_mat).astype(str)

def search_res(adata, n_clusters, method='leiden', use_rep='scgpt_cross_gae', start=0.1, end=3.0, increment=0.02):
    """Search for resolution to achieve target cluster count."""
    try:
        sc.pp.neighbors(adata, n_neighbors=10, use_rep=use_rep)
        for res in np.arange(start, end, increment):
            res = round(res, 3)
            try:
                sc.tl.leiden(adata, random_state=0, resolution=res, flavor='igraph', n_iterations=2, directed=False)
            except Exception:
                try:
                    sc.tl.leiden(adata, random_state=0, resolution=res)
                except Exception:
                    sc.tl.louvain(adata, random_state=0, resolution=res)
            
            cluster_col = 'leiden' if 'leiden' in adata.obs else 'louvain'
            if adata.obs[cluster_col].nunique() == n_clusters:
                return res
    except Exception as e:
        print(f"Notice: Graph resolution search failed ({e}). Returning fallback res=0.5")
    return 0.5

def evaluate_clustering(y_true_series, y_pred_series, features_matrix, name=""):
    """Evaluate clustering performance using ARI, NMI, Silhouette, AMI, CHI, DBI, Homogeneity, and V-measure."""
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
    return metrics


# ==================================================================
# VISUALIZATION UTILITIES
# ==================================================================
def plot_scgpt_visualizations(adata, title_prefix="", dname="dataset", seed=2024, save_dir=None):
    """Plot UMAP and Spatial visualizations for Ground Truth, KMeans, Leiden, and mclust."""
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
    save_path = os.path.join(save_dir, f"scgpt_cross_gae_plot_{dname}_seed_{seed}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization plot image to: {save_path}")
    plt.close(fig)

def plot_scgpt_summary_boxplots(df_all, save_dir=None):
    """Generate and save Box & Whiskers plots comparing performance across all datasets."""
    if df_all.empty or "cluster alg" not in df_all.columns:
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
        ax.set_title(f"scGPT-Cross-GAE: {metric} Performance Across Datasets", fontsize=14, fontweight='bold')
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title="Cluster Alg", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"scgpt_cross_gae_boxplot_{metric.lower().replace('-', '_')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_scgpt_epoch_metrics(df_epoch: pd.DataFrame, dname: str, seed: int, save_dir: Optional[str] = None):
    """Plot ARI, NMI, and Silhouette vs Epoch in one combined graph to track model performance over training."""
    if df_epoch.empty:
        return

    if save_dir is None:
        save_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax.plot(df_epoch['epoch'], df_epoch['ARI'], marker='o', linewidth=2.5, color='#1f77b4', label='ARI')
    ax.plot(df_epoch['epoch'], df_epoch['NMI'], marker='s', linewidth=2.5, color='#2ca02c', label='NMI')
    ax.plot(df_epoch['epoch'], df_epoch['Silhouette'], marker='^', linewidth=2.5, color='#ff7f0e', label='Silhouette (SIL)')

    ax.set_title(f"scGPT-Cross-GAE Performance Across Epochs ({dname} | Seed {seed})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Metric Score", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(fontsize=12, loc='best', frameon=True)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"scgpt_cross_gae_epoch_metrics_{dname}_seed_{seed}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ARI, NMI, SIL vs Epoch plot image to: {save_path}")
    plt.close(fig)


# ==================================================================
# PIPELINE WORKFLOW EXECUTION
# ==================================================================
def run_scgpt_cross_gae_workflow(
    dname: str,
    cfg: dict,
    env_mode: str,
    seed: int,
    device: torch.device,
    n_epochs: int = 200,
    eval_interval: int = 5,
    show_plots: bool = False
):
    """Executes full scGPT-Cross-GAE pipeline for a dataset under a given seed."""
    fix_seed(seed)
    print(f"\n--- Running scGPT-Cross-GAE Dataset: {dname} | Seed: {seed} ---")

    # 1. Load Data
    adata_rna, adata_mod2 = load_dataset_data(dname, cfg, env_mode)

    # 2. Extract Phase 1 Embeddings
    # Transcriptomic Anchor (E_RNA in R^{N x 512})
    E_RNA_np = get_scgpt_embeddings(adata_rna, device)
    
    # Modality 2 Anchor (E_Mod2 in R^{N x 50})
    if cfg["type"] == "mouse_brain":
        sc.pp.normalize_total(adata_mod2, target_sum=1e4)
        sc.pp.log1p(adata_mod2)
        if sp.issparse(adata_mod2.X):
            sc.pp.scale(adata_mod2, max_value=10, zero_center=False)
        else:
            sc.pp.scale(adata_mod2, max_value=10)
        n_comps_mod2 = min(50, adata_mod2.n_obs - 1, adata_mod2.n_vars - 1)
        E_Mod2_np = pca(adata_mod2, n_comps=n_comps_mod2)
    else:
        clr_normalize_each_cell(adata_mod2)
        if sp.issparse(adata_mod2.X):
            sc.pp.scale(adata_mod2, max_value=10, zero_center=False)
        else:
            sc.pp.scale(adata_mod2, max_value=10)
        n_comps_mod2 = min(50, adata_mod2.n_obs - 1, adata_mod2.n_vars - 1)
        E_Mod2_np = pca(adata_mod2, n_comps=n_comps_mod2)

    # 3. Construct Base Spatial KNN Graph from physical (X, Y) coordinates
    if 'spatial' in adata_rna.obsm:
        coords = adata_rna.obsm['spatial']
    elif 'spatial' in adata_mod2.obsm:
        coords = adata_mod2.obsm['spatial']
    elif 'x' in adata_rna.obs and 'y' in adata_rna.obs:
        coords = adata_rna.obs[['x', 'y']].values
    elif 'spatial_x' in adata_rna.obs and 'spatial_y' in adata_rna.obs:
        coords = adata_rna.obs[['spatial_x', 'spatial_y']].values
    else:
        coords = np.random.uniform(0, 100, size=(adata_rna.n_obs, 2))
    
    adata_rna.obsm['spatial'] = coords
    adata_mod2.obsm['spatial'] = coords

    nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(coords)
    knn_graph = nbrs.kneighbors_graph(coords, mode='connectivity')
    coo = knn_graph.tocoo()
    base_edge_index = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long, device=device)

    # Convert anchors to PyTorch Tensors (Frozen Anchors)
    E_RNA_tensor = torch.tensor(E_RNA_np, dtype=torch.float32, device=device)
    E_Mod2_tensor = torch.tensor(E_Mod2_np, dtype=torch.float32, device=device)

    # 4. Initialize scGPT-Cross-GAE Model
    model = scGPTCrossGAE(
        dim_rna=E_RNA_tensor.size(1),
        dim_mod2=E_Mod2_tensor.size(1),
        latent_dim=256,
        n_heads=4,
        dropout=0.1,
        prune_threshold=0.3
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Determine target cluster count
    valid_labels = adata_rna.obs['ground_truth'].dropna().unique()
    target_labels = [l for l in valid_labels if l not in ['Exclude', 'unknown']]
    n_clusters = len(target_labels) if len(target_labels) > 0 else 7

    print(f"Training scGPT-Cross-GAE for {n_epochs} full epochs...")

    # Phase 6 Protocol: Train for full specified epochs without early stopping
    pbar = tqdm(range(1, n_epochs + 1), desc="Training scGPT-Cross-GAE")
    final_Z_np = None
    epoch_history = []

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        # Forward Pass
        Z, hat_E_rna, hat_E_mod2, pruned_edge_index = model(E_RNA_tensor, E_Mod2_tensor, base_edge_index)

        # Dual-Objective Loss Engine Computation
        loss_total, loss_recon, loss_nce = compute_total_loss(
            E_RNA_tensor, E_Mod2_tensor, hat_E_rna, hat_E_mod2, Z, pruned_edge_index,
            lambda_mod2=1.0, gamma_nce=1.0, tau=0.2
        )

        loss_total.backward()
        optimizer.step()

        pbar.set_postfix({"Loss": f"{loss_total.item():.4f}", "Recon": f"{loss_recon.item():.4f}", "NCE": f"{loss_nce.item():.4f}"})

        # Evaluate ARI, NMI, Silhouette vs Epoch
        if epoch == 1 or epoch % eval_interval == 0 or epoch == n_epochs:
            model.eval()
            with torch.no_grad():
                Z_eval, _, _, _ = model(E_RNA_tensor, E_Mod2_tensor, base_edge_index)
                Z_eval_np = Z_eval.cpu().numpy()
            
            km_eval = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            pred_eval = km_eval.fit_predict(Z_eval_np).astype(str)
            m_eval = evaluate_clustering(adata_rna.obs['ground_truth'], pd.Series(pred_eval, index=adata_rna.obs_names), Z_eval_np)
            
            epoch_history.append({
                'epoch': epoch,
                'ARI': m_eval['ARI'],
                'NMI': m_eval['NMI'],
                'Silhouette': m_eval['Silhouette']
            })

            if epoch == n_epochs:
                final_Z_np = Z_eval_np

    # Generate ARI, NMI, Silhouette vs Epoch Plot
    df_epoch_history = pd.DataFrame(epoch_history)
    output_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else '.'
    plot_scgpt_epoch_metrics(df_epoch_history, dname=dname, seed=seed, save_dir=output_dir)

    # Store final embedding in AnnData
    adata_rna.obsm['scgpt_cross_gae'] = final_Z_np

    # 5. Execute Downstream Clustering on final Z representation
    print(f"Running final evaluation clustering with n_clusters={n_clusters}...")
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    adata_rna.obs['kmeans'] = kmeans.fit_predict(final_Z_np).astype(str)

    # Leiden
    res = search_res(adata_rna, n_clusters, method='leiden', use_rep='scgpt_cross_gae')
    try:
        sc.tl.leiden(adata_rna, random_state=seed, resolution=res, flavor='igraph', n_iterations=2, directed=False)
    except Exception:
        try:
            sc.tl.leiden(adata_rna, random_state=seed, resolution=res)
        except Exception:
            try:
                sc.tl.louvain(adata_rna, random_state=seed, resolution=res)
                adata_rna.obs['leiden'] = adata_rna.obs['louvain']
            except Exception:
                adata_rna.obs['leiden'] = adata_rna.obs['kmeans']

    # mclust
    adata_rna.obs['mclust'] = run_mclust(final_Z_np, n_clusters, seed=seed)

    # Evaluate metrics
    metrics_kmeans = evaluate_clustering(
        adata_rna.obs['ground_truth'], adata_rna.obs['kmeans'], final_Z_np, name=f"{dname} Seed {seed} - KMeans"
    )
    metrics_leiden = evaluate_clustering(
        adata_rna.obs['ground_truth'], adata_rna.obs['leiden'], final_Z_np, name=f"{dname} Seed {seed} - Leiden"
    )
    metrics_mclust = evaluate_clustering(
        adata_rna.obs['ground_truth'], adata_rna.obs['mclust'], final_Z_np, name=f"{dname} Seed {seed} - mclust"
    )

    row_kmeans = {'cluster alg': 'KMeans', 'resolution': None}
    row_kmeans.update(metrics_kmeans)

    row_leiden = {'cluster alg': 'Leiden', 'resolution': res}
    row_leiden.update(metrics_leiden)

    row_mclust = {'cluster alg': 'mclust', 'resolution': None}
    row_mclust.update(metrics_mclust)

    if show_plots:
        plot_scgpt_visualizations(adata_rna, title_prefix=f"{dname} (scGPT-Cross-GAE Seed {seed})", dname=dname, seed=seed)

    return [row_kmeans, row_leiden, row_mclust]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="scGPT-Cross-GAE Pipeline")
    parser.add_argument("--test", action="store_true", help="Run fast test mode with 1 dataset, 1 seed, 5 epochs")
    parser.add_argument("--epochs", type=int, default=800, help="Number of training epochs")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    if args.test:
        print("--- RUNNING IN FAST TEST MODE ---")
        run_datasets = [datasets_to_run[0]] if datasets_to_run else ["mouse-brain-e11-s1"]
        run_seeds = [42]
        n_epochs = 5
    else:
        run_datasets = datasets_to_run
        run_seeds = SEEDS
        n_epochs = args.epochs

    all_results_flat = []
    all_results = {}

    for dname in run_datasets:
        cfg = ALL_DATASETS_CONFIG[dname]
        all_results[dname] = []

        print(f"\n=======================================================")
        print(f"STARTING scGPT-Cross-GAE WORKFLOW FOR DATASET: {dname}")
        print(f"=======================================================")

        for idx, seed in enumerate(run_seeds):
            show_plots = (idx == 0)
            try:
                results_list = run_scgpt_cross_gae_workflow(
                    dname, cfg, ENV_MODE, seed, device, n_epochs=n_epochs, eval_interval=5, show_plots=show_plots
                )
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
            print(f"scGPT-Cross-GAE AVERAGE PERFORMANCE FOR {dname}")
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
        is_kaggle = os.path.exists('/kaggle/working')
        output_dir = '/kaggle/working' if is_kaggle else '.'
        output_csv = os.path.join(output_dir, 'scgpt_cross_gae_ablation_results.csv')
        df_all.to_csv(output_csv, index=False)
        print(f"All results saved to CSV at: {output_csv}")
        plot_scgpt_summary_boxplots(df_all, save_dir=output_dir)

if __name__ == '__main__':
    main()
