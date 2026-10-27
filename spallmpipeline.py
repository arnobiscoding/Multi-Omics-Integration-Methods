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

SEEDS = [
    42, 0, 1, 7, 123, 1234, 2022, 2023, 2024, 1337
]

active_config_keys = list(ALL_DATASETS_CONFIG.keys())
datasets_to_run = active_config_keys if (len(ACTIVE_DATASETS) == 1 and ACTIVE_DATASETS[0].lower() == "all") else [d for d in ACTIVE_DATASETS if d in ALL_DATASETS_CONFIG]

print(f"Scheduled datasets: {datasets_to_run}")
print(f"Ablation seeds: {SEEDS}")


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 4
# ===========================================================================
# --- COMBINED spaLLM SOURCE CODE ---
import os
import sys
import random
import json

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import time
import argparse
import warnings
import urllib.request
import urllib.error

# Suppress noisy and deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchtext")
warnings.filterwarnings("ignore", category=UserWarning, module="scgpt")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Variable names are not unique.*")
warnings.filterwarnings("ignore", message=".*zero-centering a sparse array.*")
warnings.filterwarnings("ignore", message=".*Setting element.*of view.*")
try:
    import torchtext
    if hasattr(torchtext, "disable_torchtext_deprecation_warning"):
        torchtext.disable_torchtext_deprecation_warning()
except Exception:
    pass
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.backends.cudnn import deterministic
from torch.backends import cudnn
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Union
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    v_measure_score,
    silhouette_score,
    silhouette_samples,
    fowlkes_mallows_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# ==============================================================================
# Dashboard API Configuration & Authentication
# ==============================================================================
DEFAULT_DASHBOARD_URL = os.getenv("DASHBOARD_URL", "https://model-performance.vercel.app")
DEFAULT_API_URL = os.getenv("DASHBOARD_API_URL", f"{DEFAULT_DASHBOARD_URL}/api/experiments/upload")
DEFAULT_USERNAME = os.getenv("DASHBOARD_USERNAME", "student")
DEFAULT_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "fdsa4321")

DATASET_DASHBOARD_INFO = {
    "mouse-brain-e11-s1": {
        "datasetId": "Mouse_Brain_E11_S1",
        "datasetName": "Mouse Brain E11 S1"
    },
    "mouse-brain-e13-s1": {
        "datasetId": "Mouse_Brain_E13_S1",
        "datasetName": "Mouse Brain E13 S1"
    },
    "mouse-brain-e15-s1": {
        "datasetId": "Mouse_Brain_E15_S1",
        "datasetName": "Mouse Brain E15 S1"
    },
    "mouse-brain-e18-s1": {
        "datasetId": "Mouse_Brain_E18_S1",
        "datasetName": "Mouse Brain E18 S1"
    },
    "human-lymph-node-a1": {
        "datasetId": "10x_human_lymph_node_A1",
        "datasetName": "10X Human Lymph Node A1"
    },
    "human-lymph-node-d1": {
        "datasetId": "10x_human_lymph_node_D1",
        "datasetName": "10X Human Lymph Node D1"
    }
}


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


def authenticate_dashboard(
    base_url: str = DEFAULT_DASHBOARD_URL,
    username: str = DEFAULT_USERNAME,
    password: str = DEFAULT_PASSWORD
) -> Optional[str]:
    """Authenticate with the dashboard API using username and password to obtain a JWT bearer token."""
    login_url = f"{base_url.rstrip('/')}/api/auth/login"
    try:
        payload = json.dumps({"username": username, "password": password}).encode("utf-8")
        req = urllib.request.Request(
            login_url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                token = data.get("token")
                user = data.get("user", {})
                user_desc = user.get("username", username)
                role = user.get("role", "student")
                print(f"[Dashboard] Successfully authenticated as '{user_desc}' (Role: {role})")
                return token
    except Exception as e:
        print(f"[Dashboard Warning] Login failed ({e}). Proceeding without token...")
    return None


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
    api_url: Optional[str] = DEFAULT_API_URL,
    token: Optional[str] = None,
    dataset_display_name: Optional[str] = None
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
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_id": model_id,
            "dataset": dataset_name,
            "seed": int(seed),
            "final_epoch": len(loss_hist),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "metrics": safe_metrics
        }, f, indent=2)

    with open(os.path.join(exp_dir, "curves.json"), "w", encoding="utf-8") as f:
        json.dump({"history": safe_history}, f, indent=2)

    with open(os.path.join(exp_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "dataset": dataset_name,
            "seed": int(seed),
            "hyperparameters": safe_hyperparameters
        }, f, indent=2)

    print(f"[Dashboard] Standardized JSONs saved to: {exp_dir}")

    # 3. Direct API Call to Next.js / MongoDB Dashboard
    if api_url:
        try:
            display_name = dataset_display_name or dataset_name.replace("_", " ").replace("-", " ").title()
            payload = {
                "modelId": model_id,
                "modelName": model_name,
                "datasetId": dataset_name,
                "datasetName": display_name,
                "seed": int(seed),
                "bestEpoch": best_epoch,
                "bestScore": best_score,
                "finalMetrics": safe_metrics,
                "history": safe_history,
                "embeddingsData": safe_embeddings,
                "modelMetadata": {
                    "architecture": "Spatial Multi-Omics LLM + Dual-Attention GNN (spaLLM)",
                    "hyperparameters": safe_hyperparameters,
                }
            }

            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            req = urllib.request.Request(
                api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                if response.status == 200:
                    print(f"[Dashboard] Successfully uploaded experiment results to Dashboard via API ({api_url})!")
                    return True
        except Exception as e:
            print(f"[Dashboard Info] Direct API push skipped ({e}). Data is safely stored in JSON files.")
    return False

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
    values = torch.from_numpy(sparse_mx.data)
    return torch.sparse_coo_tensor(indices, values, torch.Size(sparse_mx.shape), dtype=torch.float32)

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

def add_gaussian_noise(matrix, mean=0.0, std=0.001):
    """Add Gaussian noise to the matrix."""
    noise = torch.normal(mean=mean, std=std, size=matrix.size()).to(matrix.device)
    return matrix + noise

def run_mclust(data_matrix, n_clusters, seed=2024, max_dims=30):
    """Perform mclust clustering via rpy2 using native R helper function with PCA reduction."""
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
        
        r_code = '''
        if (!requireNamespace("mclust", quietly = TRUE)) {
            install.packages("mclust", repos="https://cloud.r-project.org", quiet=TRUE)
        }
        suppressPackageStartupMessages(library(mclust))
        run_mclust_native <- function(mat, n_clusters, seed) {
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

    metrics = ["ARI", "NMI", "Silhouette", "AMI", "CHI", "DBI", "Homogeneity", "V-measure"]
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
    combined_path = os.path.join(save_dir, "spallm_boxplot_all_metrics.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined multi-metric Boxplot figure to: {combined_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


def export_spot_predictions_csv(adata, dataset_name, seed=0, out_dir="results"):
    """
    Exports cell/spot-level spatial coordinates, ground truth annotations, 
    clustering domains (KMeans, Leiden, mclust), and modality attention weights to CSV.
    """
    os.makedirs(out_dir, exist_ok=True)
    coords = adata.obsm.get('spatial', None)
    
    pred_data = {
        'barcode': adata.obs_names.tolist(),
        'ground_truth': adata.obs.get('ground_truth', 'unknown').astype(str).tolist(),
        'spaLLM_domain': adata.obs.get('spaLLM_domain', adata.obs.get('domain', '')).astype(str).tolist(),
    }
    
    for alg in ['kmeans', 'leiden', 'mclust']:
        if alg in adata.obs:
            pred_data[alg] = adata.obs[alg].astype(str).tolist()
            
    if 'mean_rna_attention' in adata.obs:
        pred_data['mean_rna_attention'] = adata.obs['mean_rna_attention'].tolist()
    elif 'alpha' in adata.obsm and adata.obsm['alpha'].ndim >= 2:
        pred_data['mean_rna_attention'] = adata.obsm['alpha'][:, 0].tolist()

    if 'mean_aux_attention' in adata.obs:
        pred_data['mean_aux_attention'] = adata.obs['mean_aux_attention'].tolist()
    elif 'alpha' in adata.obsm and adata.obsm['alpha'].ndim >= 2:
        pred_data['mean_aux_attention'] = adata.obsm['alpha'][:, 1].tolist()
        
    if coords is not None and coords.shape[0] == len(adata.obs_names) and coords.shape[1] >= 2:
        pred_data['spatial_x'] = coords[:, 0].tolist()
        pred_data['spatial_y'] = coords[:, 1].tolist()
        
    df_pred = pd.DataFrame(pred_data)
    csv_path = os.path.join(out_dir, f"spallm_spot_predictions_{dataset_name}.csv")
    df_pred.to_csv(csv_path, index=False)
    print(f"[CSV Saved] Spot-level predictions & attention weights saved to: {csv_path}")
    return df_pred


def compute_and_plot_cross_seed_linearity(
    df_results=None,
    out_dir="results",
    visuals_dir=None,
    show_plot=False
):
    """
    Computes cross-seed linearity & metric spread diagnostics (ARI & Silhouette) across seeds,
    tracks them in a CSV, and generates unified side-by-side plots (ARI vs Silhouette).
    Directly incorporates the functionality from Test.ipynb Cell 0.
    """
    os.makedirs(out_dir, exist_ok=True)
    v_dir = visuals_dir or os.path.join(out_dir, "visuals")
    os.makedirs(v_dir, exist_ok=True)

    # 1. Resolve results DataFrame
    if df_results is None or len(df_results) == 0:
        candidates = [
            os.path.join(out_dir, "spallm_benchmark_results.csv"),
            os.path.join(out_dir, "spallm_ablation_results.csv"),
            "results/spallm_benchmark_results.csv",
            "results/spallm_ablation_results.csv",
            "spallm_benchmark_results.csv",
            "spallm_ablation_results.csv"
        ]
        csv_fallback = next((p for p in candidates if os.path.exists(p)), None)
        if csv_fallback is not None:
            df_results = pd.read_csv(csv_fallback)
            print(f"[Loaded] Retrieved benchmark rows from: {csv_fallback}")
        else:
            print("[Notice] No benchmark results found to compute cross-seed linearity.")
            return None

    # Filter for primary algorithm if cluster alg column present
    if 'cluster alg' in df_results.columns:
        if (df_results['cluster alg'] == 'mclust').any():
            df_curr = df_results[df_results['cluster alg'] == 'mclust'].copy()
        elif (df_results['cluster alg'] == 'KMeans').any():
            df_curr = df_results[df_results['cluster alg'] == 'KMeans'].copy()
        else:
            df_curr = df_results.copy()
    else:
        df_curr = df_results.copy()

    if len(df_curr) == 0:
        print("[Notice] Filtered benchmark results are empty.")
        return None

    unique_datasets = df_curr['dataset'].unique()
    n_datasets = len(unique_datasets)
    if n_datasets == 0:
        return None

    fig, axes = plt.subplots(n_datasets, 2, figsize=(18, 4.8 * n_datasets), squeeze=False)

    print("\n" + "=" * 85)
    print(" CROSS-SEED LINEARITY & SPREAD DIAGNOSTICS (ARI & SILHOUETTE)")
    print("=" * 85)

    diagnostics_rows = []

    for idx, d_name in enumerate(unique_datasets):
        sub_df = df_curr[df_curr['dataset'] == d_name].copy()

        # Format seeds on X-axis as categorical labels to ensure uniform spacing
        sub_df['seed_str'] = sub_df['seed'].astype(str)
        seeds_order = sub_df['seed_str'].tolist()
        x_indices = np.arange(len(seeds_order))

        # -------------------------------------------------------------
        # Left Subplot: Adjusted Rand Index (ARI) across all seeds
        # -------------------------------------------------------------
        ari_vals = sub_df['ARI'].values.astype(float)
        mean_ari = float(np.mean(ari_vals))
        std_ari = float(np.std(ari_vals))
        min_ari = float(np.min(ari_vals))
        max_ari = float(np.max(ari_vals))
        range_ari = max_ari - min_ari

        ax_ari = axes[idx, 0]
        # Line + Marker plot across seeds
        ax_ari.plot(x_indices, ari_vals, marker='o', markersize=9, linewidth=2.4, color='#1f77b4', label='Seed ARI')
        # Horizontal reference line for Mean ARI (shows linearity/flatness)
        ax_ari.axhline(mean_ari, color='#d62728', linestyle='--', linewidth=2.0, label=f'Mean ARI: {mean_ari:.4f}')
        # Shaded +-1 Std confidence band
        ax_ari.fill_between(x_indices, mean_ari - std_ari, mean_ari + std_ari, color='#1f77b4', alpha=0.15, label=f'±1σ Band (±{std_ari:.4f})')
        # Annotate each seed point with its exact value
        for x_i, y_i in zip(x_indices, ari_vals):
            ax_ari.annotate(f"{y_i:.3f}", (x_i, y_i), textcoords="offset points", xytext=(0, 9), ha='center', fontsize=9, fontweight='bold', color='#1f77b4')

        ax_ari.set_xticks(x_indices)
        ax_ari.set_xticklabels([f"Seed {s}" for s in seeds_order], fontsize=10, rotation=25)
        ax_ari.set_ylabel("Adjusted Rand Index (ARI)", fontsize=11, fontweight='bold')
        ax_ari.set_title(f"{d_name}\n[ARI Linearity] Mean: {mean_ari:.4f} ± {std_ari:.4f} | Spread (Max-Min): {range_ari:.4f}", fontsize=12, fontweight='bold')
        ax_ari.grid(True, linestyle=':', alpha=0.6)
        ax_ari.legend(loc='lower right', fontsize=9)
        y_pad = max(0.04, range_ari * 0.4)
        ax_ari.set_ylim([max(0.0, min_ari - y_pad), min(1.0, max_ari + y_pad)])

        # -------------------------------------------------------------
        # Right Subplot: Silhouette Score across all seeds
        # -------------------------------------------------------------
        sil_vals = sub_df['Silhouette'].values.astype(float)
        mean_sil = float(np.mean(sil_vals))
        std_sil = float(np.std(sil_vals))
        min_sil = float(np.min(sil_vals))
        max_sil = float(np.max(sil_vals))
        range_sil = max_sil - min_sil

        ax_sil = axes[idx, 1]
        # Line + Marker plot across seeds
        ax_sil.plot(x_indices, sil_vals, marker='s', markersize=9, linewidth=2.4, color='#2ca02c', label='Seed Silhouette')
        # Horizontal reference line for Mean Silhouette
        ax_sil.axhline(mean_sil, color='#9467bd', linestyle='--', linewidth=2.0, label=f'Mean Sil: {mean_sil:.4f}')
        # Shaded +-1 Std confidence band
        ax_sil.fill_between(x_indices, mean_sil - std_sil, mean_sil + std_sil, color='#2ca02c', alpha=0.15, label=f'±1σ Band (±{std_sil:.4f})')
        # Annotate each seed point with its exact value
        for x_i, y_i in zip(x_indices, sil_vals):
            ax_sil.annotate(f"{y_i:.3f}", (x_i, y_i), textcoords="offset points", xytext=(0, 9), ha='center', fontsize=9, fontweight='bold', color='#2ca02c')

        ax_sil.set_xticks(x_indices)
        ax_sil.set_xticklabels([f"Seed {s}" for s in seeds_order], fontsize=10, rotation=25)
        ax_sil.set_ylabel("Silhouette Score", fontsize=11, fontweight='bold')
        ax_sil.set_title(f"{d_name}\n[Silhouette Linearity] Mean: {mean_sil:.4f} ± {std_sil:.4f} | Spread (Max-Min): {range_sil:.4f}", fontsize=12, fontweight='bold')
        ax_sil.grid(True, linestyle=':', alpha=0.6)
        ax_sil.legend(loc='lower right', fontsize=9)
        y_pad_sil = max(0.04, range_sil * 0.4)
        ax_sil.set_ylim([max(-1.0, min_sil - y_pad_sil), min(1.0, max_sil + y_pad_sil)])

        # Detailed Linearity Diagnostic
        print(f"Dataset: {d_name:<30} | Seeds Tested: {len(seeds_order)}")
        print(f"   * ARI Spread        -> Mean: {mean_ari:.4f} | Std: {std_ari:.4f} | Min: {min_ari:.4f} | Max: {max_ari:.4f} | Spread (Max-Min): {range_ari:.4f}")
        print(f"   * Silhouette Spread -> Mean: {mean_sil:.4f} | Std: {std_sil:.4f} | Min: {min_sil:.4f} | Max: {max_sil:.4f} | Spread (Max-Min): {range_sil:.4f}")
        if range_ari <= 0.03:
            verdict = "Ultra-stable flat line (Near-zero variance, high seed independence)."
        elif range_ari <= 0.06:
            verdict = "Near-linear trajectory across seeds (Good stability)."
        else:
            verdict = f"Noticeable variation across seeds ({range_ari*100:.1f}% spread)."
        print(f"   >> Linearity Verdict : {verdict}")

        if 'NMI' in sub_df.columns:
            nmi_vals = sub_df['NMI'].values.astype(float)
            mean_nmi = float(np.mean(nmi_vals))
            std_nmi = float(np.std(nmi_vals))
            range_nmi = float(np.max(nmi_vals) - np.min(nmi_vals))
        else:
            mean_nmi, std_nmi, range_nmi = None, None, None

        diagnostics_rows.append({
            "dataset": d_name,
            "seeds_tested": len(seeds_order),
            "mean_ari": round(mean_ari, 4),
            "std_ari": round(std_ari, 4),
            "min_ari": round(min_ari, 4),
            "max_ari": round(max_ari, 4),
            "range_ari": round(range_ari, 4),
            "mean_silhouette": round(mean_sil, 4),
            "std_silhouette": round(std_sil, 4),
            "min_silhouette": round(min_sil, 4),
            "max_silhouette": round(max_sil, 4),
            "range_silhouette": round(range_sil, 4),
            "mean_nmi": round(mean_nmi, 4) if mean_nmi is not None else np.nan,
            "std_nmi": round(std_nmi, 4) if std_nmi is not None else np.nan,
            "range_nmi": round(range_nmi, 4) if range_nmi is not None else np.nan,
            "linearity_verdict": verdict
        })

    # Save Cross-Seed Linearity Diagnostics to CSV
    df_diag = pd.DataFrame(diagnostics_rows)
    diag_csv_path = os.path.join(out_dir, "spallm_cross_seed_linearity_diagnostics.csv")
    df_diag.to_csv(diag_csv_path, index=False)
    print(f"\n[CSV Saved] Cross-seed linearity diagnostics saved to: {diag_csv_path}")

    # Unified plot saving
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.subplots_adjust(top=0.96, hspace=0.35)
    plt.suptitle("spaLLM: Side-by-Side Cross-Seed Linearity & Spread Diagnostics (ARI & Silhouette)", fontsize=16, fontweight='bold', y=0.995)
    save_path = os.path.join(v_dir, "spallm_cross_seed_linearity.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[Saved] Unified cross-seed linearity figure saved to: {save_path}")
    if v_dir != out_dir:
        plt.savefig(os.path.join(out_dir, "spallm_cross_seed_linearity.png"), dpi=300, bbox_inches='tight')

    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)
    return df_diag


def plot_spallm_spatial_domains_and_attention(
    loaded_adatas=None,
    df_results=None,
    save_dir=None,
    visuals_dir=None,
    show_plot=False
):
    """
    Side-by-Side Spatial Domain & Attention Map Visualization for ALL Datasets (Unified Figure).
    Directly incorporates the functionality from Test.ipynb Cell 1.
    """
    out_dir = save_dir or ('/kaggle/working' if os.path.exists('/kaggle/working') else 'results')
    v_dir = visuals_dir or os.path.join(out_dir, "visuals")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    adatas_dict = loaded_adatas.copy() if loaded_adatas is not None else {}

    # Check if AnnData needs to be loaded from disk
    if len(adatas_dict) == 0:
        for d_name in ALL_DATASETS_CONFIG.keys():
            search_paths = [
                os.path.join(out_dir, f"spallm_adata_{d_name}.h5ad"),
                os.path.join(out_dir, "results", f"spallm_adata_{d_name}.h5ad"),
                os.path.join("results", f"spallm_adata_{d_name}.h5ad"),
                os.path.join(out_dir, f"DualGate_DEC_adata_{d_name}.h5ad"),
                os.path.join("results", f"DualGate_DEC_adata_{d_name}.h5ad"),
            ]
            for p in search_paths:
                if os.path.exists(p):
                    try:
                        adatas_dict[d_name] = sc.read_h5ad(p)
                        break
                    except Exception:
                        pass

    if len(adatas_dict) == 0:
        print("[Notice] No spatial domain AnnData objects available to render. Skipping spatial domain & attention visualization.")
        return

    valid_datasets = list(adatas_dict.keys())
    n_datasets = len(valid_datasets)

    print("\n" + "=" * 85)
    print(f" SPATIAL DOMAIN PARTITIONING & MODALITY ATTENTION VISUALIZATION ({n_datasets} DATASETS)")
    print("=" * 85)

    fig, axes = plt.subplots(n_datasets, 3, figsize=(21, 5.2 * n_datasets), squeeze=False)

    for row_idx, d_name in enumerate(valid_datasets):
        adata_plot = adatas_dict[d_name]

        coords = adata_plot.obsm.get('spatial', None)
        if coords is None:
            coords = np.zeros((adata_plot.n_obs, 2))

        gt_series = adata_plot.obs.get('ground_truth', None)
        gt_labels = pd.factorize(gt_series)[0] if gt_series is not None else np.zeros(coords.shape[0])

        domain_col = None
        for cand in ['spaLLM_domain', 'domain', 'mclust', 'kmeans', 'leiden']:
            if cand in adata_plot.obs:
                domain_col = cand
                break
        pred_labels = pd.factorize(adata_plot.obs[domain_col])[0] if domain_col else gt_labels

        has_att = 'mean_rna_attention' in adata_plot.obs
        if not has_att and 'alpha' in adata_plot.obsm and adata_plot.obsm['alpha'].ndim >= 2:
            mean_att = adata_plot.obsm['alpha'][:, 0]
            has_att = True
        elif has_att:
            mean_att = adata_plot.obs['mean_rna_attention'].values
        else:
            mean_att = None

        row_metric_str = ""
        seed_label = adata_plot.uns.get('selected_seed', None)
        if df_results is not None and len(df_results) > 0:
            sub_m = df_results[df_results['dataset'] == d_name]
            if len(sub_m) > 0:
                if seed_label is not None and (sub_m['seed'] == seed_label).any():
                    row_s = sub_m[sub_m['seed'] == seed_label].iloc[0]
                    row_metric_str = f" (Best ARI Seed: {seed_label} | ARI: {row_s['ARI']:.4f}, Sil: {row_s['Silhouette']:.4f})"
                else:
                    top_row = sub_m.sort_values(by='ARI', ascending=False).iloc[0]
                    row_metric_str = f" (Best ARI Seed: {top_row['seed']} | ARI: {top_row['ARI']:.4f}, Sil: {top_row['Silhouette']:.4f})"
        elif seed_label is not None:
            row_metric_str = f" (Seed: {seed_label})"

        # -------------------------------------------------------------
        # 1. Ground Truth Annotations (Col 0)
        # -------------------------------------------------------------
        ax0 = axes[row_idx, 0]
        is_bg = (gt_labels == -1)
        if gt_series is not None:
            is_bg = is_bg | (gt_series == 'unknown') | (gt_series == 'Exclude') | (gt_series.isna())
        if np.any(is_bg):
            ax0.scatter(coords[is_bg, 0], coords[is_bg, 1], c='#e0e0e0', s=20, alpha=0.35, label='Unannotated')
        valid_mask = ~is_bg
        if np.any(valid_mask):
            ax0.scatter(coords[valid_mask, 0], coords[valid_mask, 1], c=gt_labels[valid_mask], cmap='tab20', s=20, alpha=0.9)
        ax0.set_title(f"{d_name}\nGround Truth Annotations", fontsize=12, fontweight='bold')
        ax0.invert_yaxis()
        ax0.axis('off')

        # -------------------------------------------------------------
        # 2. spaLLM Spatial Domains (Col 1)
        # -------------------------------------------------------------
        ax1 = axes[row_idx, 1]
        ax1.scatter(coords[:, 0], coords[:, 1], c=pred_labels, cmap='tab20', s=20, alpha=0.9)
        ax1.set_title(f"{d_name}\nspaLLM Spatial Domains{row_metric_str}", fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.axis('off')

        # -------------------------------------------------------------
        # 3. Spatial Modality Attention Heatmap (Col 2)
        # -------------------------------------------------------------
        ax2 = axes[row_idx, 2]
        if has_att and mean_att is not None:
            sc_att = ax2.scatter(coords[:, 0], coords[:, 1], c=mean_att, cmap='coolwarm', s=20, vmin=0.0, vmax=1.0, alpha=0.95)
            cbar = plt.colorbar(sc_att, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label('RNA Attention Weight (1 - Aux Weight)', fontsize=9, fontweight='bold')
            ax2.set_title(f"{d_name}\nModality Attention Map (RNA vs Aux)", fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, "Attention weights not extracted", ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title(f"{d_name}\nModality Attention Map", fontsize=12, fontweight='bold')
        ax2.invert_yaxis()
        ax2.axis('off')

        print(f"Dataset: {d_name:<30} | Cells/Spots: {coords.shape[0]} | Domains Found: {len(np.unique(pred_labels))} | Attention Mapped: {has_att}")

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.subplots_adjust(top=0.96, hspace=0.35)
    plt.suptitle("spaLLM: Side-by-Side Spatial Domain Partitioning & Modality Attention Heatmaps across Datasets", fontsize=16, fontweight='bold', y=0.995)

    save_path = os.path.join(v_dir, "spallm_spatial_domains_and_attention.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Unified spatial domains & attention map figure ({n_datasets} datasets) saved to: {save_path}")
    if v_dir != out_dir:
        plt.savefig(os.path.join(out_dir, "spallm_spatial_domains_and_attention.png"), dpi=300, bbox_inches='tight')

    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)


def plot_spallm_latent_umap_manifolds(
    loaded_adatas=None,
    df_results=None,
    save_dir=None,
    visuals_dir=None,
    show_plot=False
):
    """
    Latent Embedding Manifold Projections (Side-by-Side UMAP Suite for ALL Datasets).
    Directly incorporates the functionality from Test.ipynb Cell 2.
    """
    out_dir = save_dir or ('/kaggle/working' if os.path.exists('/kaggle/working') else 'results')
    v_dir = visuals_dir or os.path.join(out_dir, "visuals")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    adatas_dict = loaded_adatas.copy() if loaded_adatas is not None else {}

    # Check if AnnData needs to be loaded from disk
    if len(adatas_dict) == 0:
        for d_name in ALL_DATASETS_CONFIG.keys():
            search_paths = [
                os.path.join(out_dir, f"spallm_adata_{d_name}.h5ad"),
                os.path.join(out_dir, "results", f"spallm_adata_{d_name}.h5ad"),
                os.path.join("results", f"spallm_adata_{d_name}.h5ad"),
                os.path.join(out_dir, f"DualGate_DEC_adata_{d_name}.h5ad"),
                os.path.join("results", f"DualGate_DEC_adata_{d_name}.h5ad"),
            ]
            for p in search_paths:
                if os.path.exists(p):
                    try:
                        adatas_dict[d_name] = sc.read_h5ad(p)
                        break
                    except Exception:
                        pass

    valid_adatas = {}
    for d_name, adata_curr in adatas_dict.items():
        has_emb = False
        for emb_k in ['spaLLM', 'spaLLM_emb', 'DualGate_DEC_emb', 'feat']:
            if emb_k in adata_curr.obsm:
                has_emb = True
                break
        if has_emb or 'X_umap' in adata_curr.obsm:
            valid_adatas[d_name] = adata_curr

    if len(valid_adatas) == 0:
        print("[Notice] No latent embeddings found to compute UMAP. Skipping latent UMAP visualization.")
        return

    valid_datasets = list(valid_adatas.keys())
    n_datasets = len(valid_datasets)

    print("\n" + "=" * 85)
    print(f" LATENT EMBEDDING MANIFOLD VISUALIZATION (UMAP) ({n_datasets} DATASETS - BEST ARI SEED)")
    print("=" * 85)

    for d_name in valid_datasets:
        adata_curr = valid_adatas[d_name]
        if 'X_umap' not in adata_curr.obsm:
            print(f"   • Computing 2D UMAP projection for {d_name}...")
            emb_rep = None
            for cand in ['spaLLM', 'spaLLM_emb', 'DualGate_DEC_emb', 'feat']:
                if cand in adata_curr.obsm:
                    emb_rep = cand
                    break
            sc.pp.neighbors(adata_curr, use_rep=emb_rep, n_neighbors=15, metric='cosine', random_state=42)
            sc.tl.umap(adata_curr, min_dist=0.3, spread=1.0, random_state=42)

    fig, axes = plt.subplots(n_datasets, 3, figsize=(21, 5.8 * n_datasets), squeeze=False)

    for row_idx, d_name in enumerate(valid_datasets):
        adata_plot = valid_adatas[d_name]
        umap_coords = adata_plot.obsm['X_umap']

        gt_series = adata_plot.obs.get('ground_truth', None)
        gt_labels = pd.factorize(gt_series)[0] if gt_series is not None else np.zeros(umap_coords.shape[0])

        domain_col = None
        for cand in ['spaLLM_domain', 'domain', 'mclust', 'kmeans', 'leiden']:
            if cand in adata_plot.obs:
                domain_col = cand
                break
        pred_labels = pd.factorize(adata_plot.obs[domain_col])[0] if domain_col else gt_labels

        has_att = 'mean_rna_attention' in adata_plot.obs
        if not has_att and 'alpha' in adata_plot.obsm and adata_plot.obsm['alpha'].ndim >= 2:
            mean_att = adata_plot.obsm['alpha'][:, 0]
            has_att = True
        elif has_att:
            mean_att = adata_plot.obs['mean_rna_attention'].values
        else:
            mean_att = None

        row_metric_str = ""
        seed_label = adata_plot.uns.get('selected_seed', None)
        if df_results is not None and len(df_results) > 0:
            sub_m = df_results[df_results['dataset'] == d_name]
            if len(sub_m) > 0:
                if seed_label is not None and (sub_m['seed'] == seed_label).any():
                    row_s = sub_m[sub_m['seed'] == seed_label].iloc[0]
                    row_metric_str = f" (Best ARI Seed: {seed_label} | ARI: {row_s['ARI']:.4f}, Sil: {row_s['Silhouette']:.4f})"
                else:
                    top_row = sub_m.sort_values(by='ARI', ascending=False).iloc[0]
                    row_metric_str = f" (Best ARI Seed: {top_row['seed']} | ARI: {top_row['ARI']:.4f} | Sil: {top_row['Silhouette']:.4f})"
        elif seed_label is not None:
            row_metric_str = f" (Best ARI Seed: {seed_label})"

        # -------------------------------------------------------------
        # 1. Ground Truth Annotations in Latent UMAP (Col 0)
        # -------------------------------------------------------------
        ax0 = axes[row_idx, 0]
        is_bg = (gt_labels == -1)
        if gt_series is not None:
            is_bg = is_bg | (gt_series == 'unknown') | (gt_series == 'Exclude') | (gt_series.isna())
        if np.any(is_bg):
            ax0.scatter(umap_coords[is_bg, 0], umap_coords[is_bg, 1], c='#e0e0e0', s=16, alpha=0.35, label='Unannotated')
        valid_mask = ~is_bg
        if np.any(valid_mask):
            ax0.scatter(umap_coords[valid_mask, 0], umap_coords[valid_mask, 1], c=gt_labels[valid_mask], cmap='tab20', s=22, alpha=0.85)
        ax0.set_title(f"{d_name}\nLatent UMAP: Ground Truth Annotations", fontsize=12, fontweight='bold')
        ax0.set_xlabel("UMAP 1", fontsize=10)
        ax0.set_ylabel("UMAP 2", fontsize=10)
        ax0.grid(True, linestyle=':', alpha=0.4)

        # -------------------------------------------------------------
        # 2. spaLLM Discovered Spatial Domains in Latent UMAP (Col 1)
        # -------------------------------------------------------------
        ax1 = axes[row_idx, 1]
        ax1.scatter(umap_coords[:, 0], umap_coords[:, 1], c=pred_labels, cmap='tab20', s=22, alpha=0.85)
        ax1.set_title(f"{d_name}\nLatent UMAP: spaLLM Domains{row_metric_str}", fontsize=12, fontweight='bold')
        ax1.set_xlabel("UMAP 1", fontsize=10)
        ax1.set_ylabel("UMAP 2", fontsize=10)
        ax1.grid(True, linestyle=':', alpha=0.4)

        # -------------------------------------------------------------
        # 3. Spatial Modality Attention Heatmap on Latent UMAP (Col 2)
        # -------------------------------------------------------------
        ax2 = axes[row_idx, 2]
        if has_att and mean_att is not None:
            sc_att = ax2.scatter(umap_coords[:, 0], umap_coords[:, 1], c=mean_att, cmap='coolwarm', s=22, vmin=0.0, vmax=1.0, alpha=0.9)
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.08)
            cbar = fig.colorbar(sc_att, cax=cax)
            cbar.set_label('RNA Attention Weight α_rna', fontsize=9, fontweight='bold')
            ax2.set_title(f"{d_name}\nLatent UMAP: Modality Attention (RNA vs Aux)", fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, "Attention weights not extracted", ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title(f"{d_name}\nLatent UMAP: Modality Attention", fontsize=12, fontweight='bold')
        ax2.set_xlabel("UMAP 1", fontsize=10)
        ax2.set_ylabel("UMAP 2", fontsize=10)
        ax2.grid(True, linestyle=':', alpha=0.4)

        print(f"Dataset: {d_name:<30} | Manifold Spots: {umap_coords.shape[0]} | Clusters: {len(np.unique(pred_labels))} | Attention Mapped: {has_att}")

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.subplots_adjust(top=0.96, hspace=0.35)
    plt.suptitle("spaLLM: Latent Embedding Manifold Projections (UMAP) across Datasets (Best ARI Seed)", fontsize=16, fontweight='bold', y=0.995)

    save_path = os.path.join(v_dir, "spallm_latent_umap.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[Saved] Unified latent UMAP manifold figure saved to: {save_path}")
    if v_dir != out_dir:
        plt.savefig(os.path.join(out_dir, "spallm_latent_umap.png"), dpi=300, bbox_inches='tight')

    if show_plot:
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
        self.embedding = torch.as_tensor(embedding, dtype=torch.float32).to(device)
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
            'alpha_att2': results['alpha_att2'].cpu().numpy(),
            'loss_history': list(self.loss_history)
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
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
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
def run_spallm_workflow(dataset_name, dataset_cfg, env_mode, seed, device, show_plots=False,
                        dashboard_token=None, api_url=DEFAULT_API_URL, output_dir=None):
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
    adata_rna.var_names_make_unique()
    
    mod2_filename = None
    for cand in dataset_cfg["mod2_candidates"]:
        if os.path.exists(os.path.join(data_dir, cand)):
            mod2_filename = cand
            break
    if mod2_filename is None:
        mod2_filename = dataset_cfg["mod2_candidates"][0]
        
    adata_mod2 = sc.read_h5ad(os.path.join(data_dir, mod2_filename))
    adata_mod2.var_names_make_unique()
    
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
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[0] == adata.n_obs:
            adata.obsm[key] = value
        else:
            adata.uns[key] = value
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
        'Silhouette': metrics_kmeans['Silhouette'],
        'AMI': metrics_kmeans['AMI'],
        'CHI': metrics_kmeans['CHI'],
        'DBI': metrics_kmeans['DBI'],
        'Homogeneity': metrics_kmeans['Homogeneity'],
        'V-measure': metrics_kmeans['V-measure'],
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
        'resolution': None
    }

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

    # 7. Modality Attention Weights & Domain Assignment
    if 'alpha' in output and isinstance(output['alpha'], np.ndarray) and output['alpha'].ndim >= 2:
        adata.obs['mean_rna_attention'] = output['alpha'][:, 0]
        adata.obs['mean_aux_attention'] = output['alpha'][:, 1]
    elif 'alpha' in adata.obsm and adata.obsm['alpha'].ndim >= 2:
        adata.obs['mean_rna_attention'] = adata.obsm['alpha'][:, 0]
        adata.obs['mean_aux_attention'] = adata.obsm['alpha'][:, 1]
    else:
        adata.obs['mean_rna_attention'] = 0.5
        adata.obs['mean_aux_attention'] = 0.5

    primary_alg = 'mclust' if 'mclust' in adata.obs else 'kmeans'
    adata.obs['spaLLM_domain'] = adata.obs[primary_alg].astype(str)
    adata.obs['domain'] = adata.obs['spaLLM_domain']
    adata.uns['selected_seed'] = seed
    adata.obsm['spaLLM_emb'] = spallm_emb
    adata.obsm['spaLLM'] = spallm_emb

    # 8. Visualization Plot Saving (First seed run only)
    out_dir = output_dir or ('/kaggle/working' if os.path.exists('/kaggle/working') else 'results')
    os.makedirs(out_dir, exist_ok=True)
    if show_plots:
        plot_spallm_visualizations(adata, title_prefix=f"{dataset_name} (Seed {seed})", dname=dataset_name, seed=seed, save_dir=out_dir)
        model.plot_loss(save_path=os.path.join(out_dir, f"spallm_loss_{dataset_name}_seed_{seed}.png"))

    # 9. Export & Upload to Live Dashboard API
    if 'X_umap' not in adata.obsm:
        sc.pp.neighbors(adata, use_rep='spaLLM', n_neighbors=10)
        sc.tl.umap(adata)
    umap_coords = adata.obsm['X_umap']
    spatial_coords = adata.obsm.get('spatial', None)

    y_true_all = adata.obs['ground_truth'].astype(str).values
    y_pred_all = adata.obs[primary_alg].astype(str).values

    try:
        le = LabelEncoder()
        y_pred_int = le.fit_transform(y_pred_all)
        sample_sil_values = silhouette_samples(spallm_emb, y_pred_int).tolist()
    except Exception:
        sample_sil_values = []

    embeddings_data = {
        "umapCoordinates": umap_coords.tolist() if hasattr(umap_coords, 'tolist') else list(umap_coords),
        "spatialCoordinates": spatial_coords.tolist() if (spatial_coords is not None and hasattr(spatial_coords, 'tolist')) else [],
        "predictedLabels": y_pred_all.tolist() if hasattr(y_pred_all, 'tolist') else list(y_pred_all),
        "groundTruthLabels": y_true_all.tolist() if hasattr(y_true_all, 'tolist') else list(y_true_all),
        "sampleSilhouettes": sample_sil_values,
    }

    primary_metrics = metrics_mclust if primary_alg == 'mclust' else metrics_kmeans
    metrics_dict = {
        "ARI": float(primary_metrics.get('ARI', 0.0)),
        "NMI": float(primary_metrics.get('NMI', 0.0)),
        "Silhouette": float(primary_metrics.get('Silhouette', 0.0)),
        "AMI": float(primary_metrics.get('AMI', 0.0)),
        "CHI": float(primary_metrics.get('CHI', 0.0)),
        "DBI": float(primary_metrics.get('DBI', 0.0)),
        "Homogeneity": float(primary_metrics.get('Homogeneity', 0.0)),
        "V-measure": float(primary_metrics.get('V-measure', 0.0)),
    }
    try:
        mask = (adata.obs['ground_truth'] != 'Exclude') & (adata.obs['ground_truth'] != 'unknown') & (adata.obs['ground_truth'].notna())
        if mask.sum() > 0:
            metrics_dict["FMI"] = float(fowlkes_mallows_score(adata.obs['ground_truth'][mask], adata.obs[primary_alg][mask]))
    except Exception:
        pass

    hyperparameters = {
        "epochs": 800,
        "lr": 0.0001,
        "weight_decay": 0.0,
        "dim_output": 64,
        "clustering_tool": primary_alg,
        "all_algorithms": {
            "KMeans": metrics_kmeans,
            "Leiden": metrics_leiden,
            "mclust": metrics_mclust
        }
    }

    ds_info = DATASET_DASHBOARD_INFO.get(dataset_name, {
        "datasetId": dataset_name.replace("-", "_"),
        "datasetName": dataset_name.replace("-", " ").title()
    })

    export_dashboard_experiment(
        model_id="spaLLM",
        model_name="spaLLM",
        dataset_name=ds_info["datasetId"],
        seed=seed,
        metrics_dict=metrics_dict,
        training_results=output,
        hyperparameters=hyperparameters,
        embeddings_data=embeddings_data,
        output_dir=out_dir,
        api_url=api_url,
        token=dashboard_token,
        dataset_display_name=ds_info["datasetName"]
    )
        
    return [row_kmeans, row_leiden, row_mclust], adata


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 10
# ===========================================================================
def main(
    username=DEFAULT_USERNAME,
    password=DEFAULT_PASSWORD,
    api_url=DEFAULT_API_URL,
    datasets=None,
    seeds=None,
    output_dir=None,
    generate_plots=False,
    results_csv=None
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    is_kaggle = os.path.exists('/kaggle/working')
    out_dir = output_dir or ('/kaggle/working' if is_kaggle else 'results')
    visuals_dir = os.path.join(out_dir, "visuals")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    # -------------------------------------------------------------
    # Plot Generation & Diagnostic CSV Mode (From existing results)
    # -------------------------------------------------------------
    if generate_plots:
        print("\n" + "=" * 85)
        print(" GENERATING SPALLM PLOTS & TRACKING CSVs FROM EXISTING BENCHMARK DATA")
        print("=" * 85)

        # 1. Load benchmark CSV
        df_bench = None
        if results_csv and os.path.exists(results_csv):
            df_bench = pd.read_csv(results_csv)
            print(f"[Loaded] Retrieved benchmark CSV from: {results_csv}")
        else:
            candidates = [
                os.path.join(out_dir, "spallm_benchmark_results.csv"),
                os.path.join(out_dir, "spallm_ablation_results.csv"),
                "results/spallm_benchmark_results.csv",
                "results/spallm_ablation_results.csv",
                "results/DualGate_DEC_benchmark_results.csv",
                "results/scgpt_arise_results/scgpt_arise_benchmark_results.csv"
            ]
            cand = next((c for c in candidates if os.path.exists(c)), None)
            if cand:
                df_bench = pd.read_csv(cand)
                print(f"[Loaded] Retrieved benchmark CSV from: {cand}")

        if df_bench is not None and len(df_bench) > 0:
            if 'cluster alg' in df_bench.columns and (df_bench['cluster alg'] == 'mclust').any():
                df_primary = df_bench[df_bench['cluster alg'] == 'mclust'].copy()
            else:
                df_primary = df_bench.copy()

            compute_and_plot_cross_seed_linearity(df_primary, out_dir=out_dir, visuals_dir=visuals_dir)
            if 'cluster alg' in df_bench.columns:
                plot_spallm_summary_boxplots(df_bench, save_dir=visuals_dir)
        else:
            print("[Warning] No benchmark CSV found to plot linearity diagnostics.")

        plot_spallm_spatial_domains_and_attention(None, df_results=df_bench, save_dir=out_dir, visuals_dir=visuals_dir)
        plot_spallm_latent_umap_manifolds(None, df_results=df_bench, save_dir=out_dir, visuals_dir=visuals_dir)
        print("\n" + "=" * 85)
        print(" PLOT GENERATION COMPLETE")
        print("=" * 85 + "\n")
        return

    # Authenticate to dashboard API using provided credentials
    dashboard_token = None
    if api_url and username and password:
        base_url = api_url.split("/api/")[0] if "/api/" in api_url else DEFAULT_DASHBOARD_URL
        dashboard_token = authenticate_dashboard(base_url=base_url, username=username, password=password)

    active_datasets = datasets if datasets is not None else datasets_to_run
    active_seeds = seeds if seeds is not None else SEEDS

    all_results_flat = []
    all_results = {}
    best_adatas = {}
    best_aris = {}

    for dname in active_datasets:
        if dname not in ALL_DATASETS_CONFIG:
            print(f"Warning: Dataset '{dname}' not found in ALL_DATASETS_CONFIG. Skipping.")
            continue
        cfg = ALL_DATASETS_CONFIG[dname]
        all_results[dname] = []
        
        print(f"\n=======================================================")
        print(f"STARTING WORKFLOW FOR DATASET: {dname} OVER {len(active_seeds)} SEEDS")
        print(f"=======================================================")
        
        for idx, seed in enumerate(active_seeds):
            show_plots = (idx == 0)
            try:
                res = run_spallm_workflow(
                    dname, cfg, ENV_MODE, seed, device,
                    show_plots=show_plots,
                    dashboard_token=dashboard_token,
                    api_url=api_url,
                    output_dir=out_dir
                )
                if isinstance(res, tuple):
                    results_list, adata_curr = res
                else:
                    results_list, adata_curr = res, None

                if results_list:
                    for row in results_list:
                        res_row = {"dataset": dname, "seed": seed}
                        res_row.update(row)
                        all_results[dname].append(res_row)
                        all_results_flat.append(res_row)

                if adata_curr is not None:
                    mclust_row = next((r for r in results_list if r.get('cluster alg') == 'mclust'), None)
                    current_ari = mclust_row['ARI'] if mclust_row else (results_list[0]['ARI'] if results_list else 0.0)
                    if dname not in best_adatas or current_ari > best_aris.get(dname, -1.0):
                        best_adatas[dname] = adata_curr
                        best_aris[dname] = current_ari
            except Exception as e:
                import traceback
                print(f"Error processing dataset {dname} with seed {seed}: {e}")
                traceback.print_exc()
                
        if len(all_results[dname]) > 0:
            df_metrics = pd.DataFrame(all_results[dname])
            print(f"\n=======================================================")
            print(f"AVERAGE PERFORMANCE FOR {dname} ({len(active_seeds)} seeds)")
            print(f"=======================================================")
            numeric_cols = ["ARI", "NMI", "Silhouette", "AMI", "CHI", "DBI", "Homogeneity", "V-measure"]
            for alg, df_alg in df_metrics.groupby("cluster alg"):
                print(f"\n--- {alg} Performance (Mean ± Std) ---")
                means = df_alg[numeric_cols].mean()
                stds = df_alg[numeric_cols].std().fillna(0.0)
                summary_df = pd.DataFrame({"Mean": means, "Std": stds})
                print(summary_df.to_string())
            print(f"=======================================================\n")
        else:
            print(f"No evaluation metrics collected for {dname}.")

    # Save best AnnData objects and spot-level predictions CSV
    for dname, b_adata in best_adatas.items():
        try:
            adata_save_path = os.path.join(out_dir, f"spallm_adata_{dname}.h5ad")
            b_adata.write_h5ad(adata_save_path)
            print(f"Saved best AnnData object to: {adata_save_path}")
            export_spot_predictions_csv(b_adata, dname, b_adata.uns.get('selected_seed', 0), out_dir)
        except Exception as e:
            print(f"Warning: Failed to save AnnData or spot CSV for {dname}: {e}")

    # Save all collected metrics to CSV and generate plots
    if len(all_results_flat) > 0:
        df_all = pd.DataFrame(all_results_flat)

        # 1. Save all ablation results (KMeans, Leiden, mclust)
        output_csv = os.path.join(out_dir, 'spallm_ablation_results.csv')
        df_all.to_csv(output_csv, index=False)
        print(f"All ablation study results saved to CSV at: {output_csv}")

        # 2. Extract and save primary benchmark results (mclust by default, fallback to KMeans)
        if (df_all['cluster alg'] == 'mclust').any():
            df_benchmark = df_all[df_all['cluster alg'] == 'mclust'].copy()
        elif (df_all['cluster alg'] == 'KMeans').any():
            df_benchmark = df_all[df_all['cluster alg'] == 'KMeans'].copy()
        else:
            df_benchmark = df_all.copy()

        benchmark_csv = os.path.join(out_dir, 'spallm_benchmark_results.csv')
        df_benchmark.to_csv(benchmark_csv, index=False)
        print(f"Primary benchmark results saved to CSV at: {benchmark_csv}")
        
        # 3. Generate side-by-side Box & Whiskers plots
        print("Generating side-by-side Box & Whiskers plots for all metrics across algorithms...")
        plot_spallm_summary_boxplots(df_all, save_dir=visuals_dir)

        # 4. Cross-Seed Linearity & Spread Diagnostics Suite (CSV + Plot) from Test.ipynb Cell 0
        compute_and_plot_cross_seed_linearity(df_benchmark, out_dir=out_dir, visuals_dir=visuals_dir)

        # 5. Side-by-Side Spatial Domain & Attention Map Visualization from Test.ipynb Cell 1
        plot_spallm_spatial_domains_and_attention(best_adatas, df_results=df_benchmark, save_dir=out_dir, visuals_dir=visuals_dir)

        # 6. Latent Embedding Manifold Projections (UMAP) from Test.ipynb Cell 2
        plot_spallm_latent_umap_manifolds(best_adatas, df_results=df_benchmark, save_dir=out_dir, visuals_dir=visuals_dir)

        print("\n" + "=" * 80)
        print(f"Pipeline completed! Benchmark results automatically exported and uploaded to {DEFAULT_DASHBOARD_URL}".center(80))
        print("=" * 80 + "\n")
    else:
        print("No results were generated, skipping CSV export and Boxplots.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="spaLLM Multi-Seed Pipeline with Live Dashboard API Upload & Test Diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--username', type=str, default=DEFAULT_USERNAME, help="Dashboard account username")
    parser.add_argument('--password', type=str, default=DEFAULT_PASSWORD, help="Dashboard account password")
    parser.add_argument('--api_url', type=str, default=DEFAULT_API_URL, help="Dashboard upload API endpoint")
    parser.add_argument('--datasets', nargs='+', default=None, help="Datasets to run (e.g. mouse-brain-e11-s1 or 'all')")
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help="Specific seeds to evaluate")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save CSV results, plots, and JSONs")
    parser.add_argument('--generate_plots', action='store_true', help="Generate linearity CSV, summary plots, spatial domain/attention maps, and UMAP plots from existing benchmark results without re-training")
    parser.add_argument('--results_csv', type=str, default=None, help="Path to existing CSV containing benchmark results (for --generate_plots)")
    
    args, unknown = parser.parse_known_args()

    selected_datasets = None
    if args.datasets:
        if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
            selected_datasets = list(ALL_DATASETS_CONFIG.keys())
        else:
            selected_datasets = [d for d in args.datasets if d in ALL_DATASETS_CONFIG]

    main(
        username=args.username,
        password=args.password,
        api_url=args.api_url,
        datasets=selected_datasets,
        seeds=args.seeds,
        output_dir=args.output_dir,
        generate_plots=args.generate_plots,
        results_csv=args.results_csv
    )



