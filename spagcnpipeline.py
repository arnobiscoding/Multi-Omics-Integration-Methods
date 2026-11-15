#!/usr/bin/env python3
"""
SpaGCN Multi-Seed Pipeline (Standalone Python Script)
Integrates spatial representation learning & DEC domain identification for:
- Mouse Brain E11, E13, E15, E18 (MISAR-seq RNA + ATAC, square lattice)
- Human Lymph Node A1, D1 (10x Visium RNA + Protein/ADT, hexagonal lattice)

Faithful to official SpaGCN (Hu et al., Nature Methods 2021; SpaGCN_package):
- Uses simple_GC_DEC (1-layer linear GraphConvolution + Student-t DEC, alpha=0.2)
  as used in SpaGCN.py / ez_mode.py (with optional --model_type gc_dec for the
  2-layer ReLU+Dropout GC_DEC variant in models.py)
- Unnormalized Gaussian spatial kernel adjacency calibrated via search_l(p=0.5)
- Louvain + search_res cluster initialization (seed-aware), with KMeans fallback
- Platform-aware spatial refinement (shape="square" for MISAR-seq, "hexagon" for Visium)
- Multi-seed evaluation suite (10 seeds) covering:
  SpaGCN (Native DEC), SpaGCN_Refined, KMeans, Leiden, and mclust
- Complete evaluation metrics (ARI, NMI, Silhouette, AMI, CHI, DBI, Homogeneity, V-measure, FMI)
- Summary Box & Whiskers plots, cross-seed diagnostics, spatial/UMAP visualizations, CSVs,
  and standardized Dashboard JSON exports & API upload
"""

import os
import sys
import random
import json
import time
import argparse
import warnings
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, List

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

# Suppress noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Variable names are not unique.*")
warnings.filterwarnings("ignore", message=".*zero-centering a sparse array.*")
warnings.filterwarnings("ignore", message=".*Setting element.*of view.*")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.backends import cudnn

import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
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

import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt


# ==============================================================================
# RUN CONFIGURATION & DATASETS
# ==============================================================================
ENV_MODE = "auto"

ALL_DATASETS_CONFIG = {
    "mouse-brain-e11-s1": {
        "type": "mouse_brain",
        "shape": "square",  # MISAR-seq uses a square spatial grid (4 neighbors)
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E11_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Mouse_Brain_E11_S1/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "mouse-brain-e13-s1": {
        "type": "mouse_brain",
        "shape": "square",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E13_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Mouse_Brain_E13_S1/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "mouse-brain-e15-s1": {
        "type": "mouse_brain",
        "shape": "square",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E15_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Mouse_Brain_E15_S1/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "mouse-brain-e18-s1": {
        "type": "mouse_brain",
        "shape": "square",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E18_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Mouse_Brain_E18_S1/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "human-lymph-node-a1": {
        "type": "human_lymph_node",
        "shape": "hexagon",  # 10x Visium uses a hexagonal spatial grid (6 neighbors)
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Human_Lymph_Node_A1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset11_Human_Lymph_Node_A1/",
        "mod2_candidates": ["adata_ADT.h5ad"],
        "anno_file": "annotation.csv"
    },
    "human-lymph-node-d1": {
        "type": "human_lymph_node",
        "shape": "hexagon",
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
datasets_to_run = (
    active_config_keys
    if (len(ACTIVE_DATASETS) == 1 and ACTIVE_DATASETS[0].lower() == "all")
    else [d for d in ACTIVE_DATASETS if d in ALL_DATASETS_CONFIG]
)

DEFAULT_DASHBOARD_URL = os.getenv("DASHBOARD_URL", "https://model-performance.vercel.app")
DEFAULT_API_URL = os.getenv("DASHBOARD_API_URL", f"{DEFAULT_DASHBOARD_URL}/api/experiments/upload")
DEFAULT_USERNAME = os.getenv("DASHBOARD_USERNAME", "student")
DEFAULT_PASSWORD = os.getenv("DASHBOARD_PASSWORD", "fdsa4321")

DATASET_DASHBOARD_INFO = {
    "mouse-brain-e11-s1": {"datasetId": "Mouse_Brain_E11_S1", "datasetName": "Mouse Brain E11 S1"},
    "mouse-brain-e13-s1": {"datasetId": "Mouse_Brain_E13_S1", "datasetName": "Mouse Brain E13 S1"},
    "mouse-brain-e15-s1": {"datasetId": "Mouse_Brain_E15_S1", "datasetName": "Mouse Brain E15 S1"},
    "mouse-brain-e18-s1": {"datasetId": "Mouse_Brain_E18_S1", "datasetName": "Mouse Brain E18 S1"},
    "human-lymph-node-a1": {"datasetId": "10x_human_lymph_node_A1", "datasetName": "10X Human Lymph Node A1"},
    "human-lymph-node-d1": {"datasetId": "10x_human_lymph_node_D1", "datasetName": "10X Human Lymph Node D1"}
}


# ==============================================================================
# DASHBOARD HELPER FUNCTIONS
# ==============================================================================
def make_json_serializable(obj):
    """Recursively convert NumPy and PyTorch scalar/array types into native JSON types."""
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
    """Authenticate with the dashboard API using username and password to obtain a JWT token."""
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
) -> bool:
    """Export standardized JSON artifacts matching the dashboard schema and upload via API."""
    exp_dir = os.path.join(output_dir, "experiments", model_id, dataset_name, f"seed_{seed}")
    os.makedirs(exp_dir, exist_ok=True)

    loss_hist = training_results.get("loss_history", [])
    sil_hist = training_results.get("epoch_sil_history", [])
    ari_hist = training_results.get("epoch_ari_history", [])
    nmi_hist = training_results.get("epoch_nmi_history", [])

    history_points = []
    for ep in range(len(loss_hist)):
        loss_val = float(loss_hist[ep])
        sil_val = float(sil_hist[ep]) if ep < len(sil_hist) and sil_hist[ep] is not None else None
        ari_val = float(ari_hist[ep]) if ep < len(ari_hist) and ari_hist[ep] is not None else None
        nmi_val = float(nmi_hist[ep]) if ep < len(nmi_hist) and nmi_hist[ep] is not None else None
        history_points.append({
            "epoch": ep + 1,
            "losses": {
                "total_loss": loss_val,
                "reconstruction_loss": loss_val,
                "spatial_loss": 0.0,
                "reg_loss": 0.0,
            },
            "metrics": {
                "Silhouette": sil_val if sil_val is not None else 0.0,
                "ARI": ari_val if ari_val is not None else 0.0,
                "NMI": nmi_val if nmi_val is not None else 0.0,
            },
            "total_loss": loss_val,
            "silhouette": sil_val,
            "ari": ari_val,
        })

    safe_metrics = make_json_serializable(metrics_dict)
    safe_hyperparameters = make_json_serializable(hyperparameters)
    safe_history = make_json_serializable(history_points)
    safe_embeddings = make_json_serializable(embeddings_data or {})

    best_epoch = int(training_results.get("best_epoch", len(loss_hist)))
    best_score = float(training_results.get("best_score", metrics_dict.get("ARI", 0.0)))

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

    if safe_embeddings:
        with open(os.path.join(exp_dir, "embeddings.json"), "w", encoding="utf-8") as f:
            json.dump(safe_embeddings, f, indent=2)

    print(f"[Dashboard] Standardized JSONs saved to: {exp_dir}")

    if api_url:
        try:
            display_name = dataset_display_name or dataset_name.replace("_", " ").replace("-", " ").title()
            arch_label = training_results.get(
                "architecture",
                "Spatial Graph Convolutional Network + Deep Embedding Clustering (SpaGCN)"
            )
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
                    "architecture": arch_label,
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
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status in (200, 201):
                    print(f"[Dashboard] Successfully uploaded experiment results to Dashboard via API ({api_url})!")
                    return True
                print(f"[Dashboard] Upload returned HTTP status {response.status}")
        except Exception as e:
            print(f"[Dashboard Info] Direct API push skipped ({e}). Artifacts preserved locally.")
    return False


# ==============================================================================
# SpaGCN ARCHITECTURE (simple_GC_DEC + optional GC_DEC)
# ==============================================================================
class GraphConvolution(nn.Module):
    """Simple GCN layer: Z = A X W + b (matches SpaGCN/layers.py)."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(input_feat, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"


class _BaseGCDEC(nn.Module):
    """Shared DEC Student-t soft assignment, target distribution, and KL loss."""
    def __init__(self, nhid: int, alpha: float = 0.2):
        super().__init__()
        self.nhid = nhid
        self.alpha = alpha
        self.mu = None

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def soft_assign(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        if self.mu is None:
            return None
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2) / self.alpha + 1e-8)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        z = self.encode(x, adj)
        q = self.soft_assign(z)
        return z, q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    @staticmethod
    def loss_function(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8)), dim=1))


class SimpleGCDEC(_BaseGCDEC):
    """
    Official SpaGCN model (simple_GC_DEC in SpaGCN_package/SpaGCN/models.py):
    1 linear GraphConvolution layer + Student-t DEC (default alpha=0.2).
    """
    def __init__(self, nfeat: int, nhid: int, alpha: float = 0.2):
        super().__init__(nhid=nhid, alpha=alpha)
        self.gc = GraphConvolution(nfeat, nhid)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.gc(x, adj)


class GCDEC(_BaseGCDEC):
    """
    2-layer ReLU + Dropout GC_DEC variant (from SpaGCN_package/SpaGCN/models.py),
    with training-mode-aware dropout so eval()/predict() is deterministic.
    """
    def __init__(self, nfeat: int, nhid1: int, nhid2: int, dropout: float = 0.5, alpha: float = 0.2):
        super().__init__(nhid=nhid2, alpha=alpha)
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gc1(x, adj))
        h = F.dropout(h, self.dropout, training=self.training)
        return self.gc2(h, adj)


# ==============================================================================
# SpaGCN SPATIAL DISTANCE, ADJACENCY & REFINEMENT
# ==============================================================================
def calculate_spatial_adj_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix between spot spatial coordinates."""
    coords = np.column_stack([x, y]).astype(np.float32)
    return cdist(coords, coords).astype(np.float32)


def calculate_p(adj: np.ndarray, l: float) -> float:
    """Calculate average total edge weight contributed by spatial neighbors."""
    adj_exp = np.exp(-1.0 * (adj ** 2) / (2.0 * (l ** 2)))
    return float(np.mean(np.sum(adj_exp, axis=1)) - 1.0)


def search_l(p: float, adj: np.ndarray, start: float = 0.01, end: float = 1000.0, tol: float = 0.01, max_run: int = 100) -> float:
    """Binary search for characteristic spatial length scale l given target neighborhood contribution p."""
    p_low = calculate_p(adj, start)
    p_high = calculate_p(adj, end)
    if p_low > p + tol or np.abs(p_low - p) <= tol:
        return float(start)
    if p_high < p - tol or np.abs(p_high - p) <= tol:
        return float(end)

    cur_start, cur_end = start, end
    for _ in range(max_run):
        mid = (cur_start + cur_end) / 2.0
        p_mid = calculate_p(adj, mid)
        if np.abs(p_mid - p) <= tol:
            return float(mid)
        if p_mid <= p:
            cur_start = mid
        else:
            cur_end = mid
    return float((cur_start + cur_end) / 2.0)


def refine_spatial_domains(sample_id: List[str], pred: List[int], dis: np.ndarray, shape: str = "hexagon") -> List[int]:
    """
    Neighborhood spatial majority-voting refinement (matches SpaGCN/util.py::refine).
    Use shape="hexagon" (6 neighbors) for 10x Visium, shape="square" (4 neighbors) for MISAR-seq / ST.
    """
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        raise ValueError(f"Unsupported grid shape '{shape}'; expected 'hexagon' or 'square'.")

    refined_pred = []
    n = len(sample_id)

    for i in range(n):
        self_pred = int(pred[i])
        distances = dis[i]
        # Exclude self explicitly, take nearest num_nbs spatial neighbors, then include self
        other_order = np.argsort(distances, kind='stable')
        other_nbrs = [idx for idx in other_order if idx != i][:num_nbs]
        nbr_indices = [i] + other_nbrs
        nbr_preds = [int(pred[idx]) for idx in nbr_indices]

        val_counts = pd.Series(nbr_preds).value_counts()
        top_cluster = int(val_counts.idxmax())
        top_count = int(val_counts.max())

        if (int(val_counts.get(self_pred, 0)) < num_nbs / 2.0) and (top_count > num_nbs / 2.0):
            refined_pred.append(top_cluster)
        else:
            refined_pred.append(self_pred)
    return refined_pred


# ==============================================================================
# SpaGCN PIPELINE MODEL WRAPPER
# ==============================================================================
def _build_spagcn_net(model_type: str, nfeat: int, nhid: int, alpha: float = 0.2, dropout: float = 0.5) -> _BaseGCDEC:
    if model_type == "simple_gc_dec":
        return SimpleGCDEC(nfeat=nfeat, nhid=nhid, alpha=alpha)
    if model_type == "gc_dec":
        return GCDEC(nfeat=nfeat, nhid1=nhid, nhid2=nhid, dropout=dropout, alpha=alpha)
    raise ValueError(f"Unknown model_type '{model_type}'; expected 'simple_gc_dec' or 'gc_dec'.")


def _cluster_init_louvain(features_np: np.ndarray, res: float, n_neighbors: int, seed: int) -> np.ndarray:
    adata_tmp = sc.AnnData(features_np)
    sc.pp.neighbors(adata_tmp, n_neighbors=n_neighbors, random_state=seed)
    sc.tl.louvain(adata_tmp, resolution=res, random_state=seed)
    return adata_tmp.obs['louvain'].astype(int).to_numpy()


def _compute_centroids(features_np: np.ndarray, labels: np.ndarray) -> np.ndarray:
    df_feat = pd.DataFrame(features_np)
    df_feat["_group"] = labels
    return df_feat.groupby("_group").mean().sort_index().to_numpy(dtype=np.float32)


class SpaGCNModel:
    """SpaGCN GCN-DEC model matching SpaGCN_package (simple_GC_DEC or GC_DEC)."""
    def __init__(self, model_type: str = "simple_gc_dec", alpha: float = 0.2, dropout: float = 0.5, device: str = "cuda"):
        self.model_type = model_type
        self.alpha = alpha
        self.dropout = dropout
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.l = None
        self.model = None
        self.embed = None
        self.adj_exp = None
        self.loss_history = []

    def set_l(self, l: float):
        self.l = l

    def _fit_once(
        self,
        X_tensor: torch.Tensor,
        adj_tensor: torch.Tensor,
        n_clusters: int,
        init: str,
        res: float,
        n_neighbors: int,
        seed: int,
        lr: float,
        max_epochs: int,
        weight_decay: float,
        update_interval: int,
        tol: float,
        verbose: bool = True
    ):
        N, D = X_tensor.shape
        net = _build_spagcn_net(self.model_type, nfeat=D, nhid=D, alpha=self.alpha, dropout=self.dropout).to(self.device)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

        net.eval()
        with torch.no_grad():
            init_features = net.encode(X_tensor, adj_tensor).cpu().numpy()

        if init == "louvain":
            y_init = _cluster_init_louvain(init_features, res=res, n_neighbors=n_neighbors, seed=seed)
            cluster_centers = _compute_centroids(init_features, y_init)
        elif init == "kmeans":
            km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed)
            y_init = km.fit_predict(init_features)
            cluster_centers = km.cluster_centers_.astype(np.float32)
        else:
            raise ValueError(f"Unsupported init '{init}'; expected 'louvain' or 'kmeans'.")

        net.mu = Parameter(torch.FloatTensor(cluster_centers).to(self.device))
        loss_history = []
        net.train()
        p = None
        y_pred_last = y_init

        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                was_training = net.training
                net.eval()
                with torch.no_grad():
                    _, q_eval = net(X_tensor, adj_tensor)
                    p = net.target_distribution(q_eval).detach()
                    y_pred_cur = torch.argmax(q_eval, dim=1).cpu().numpy()
                if was_training:
                    net.train()

                delta_label = float(np.sum(y_pred_cur != y_pred_last)) / N
                y_pred_last = y_pred_cur
                if epoch > 0 and delta_label < tol:
                    if verbose:
                        print(f"  [SpaGCN Training] Early convergence at epoch {epoch} (delta_label={delta_label:.5f} < {tol})")
                    break

            optimizer.zero_grad()
            _, q = net(X_tensor, adj_tensor)
            loss = net.loss_function(p, q)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            loss_history.append(loss_val)

            if verbose and (epoch == 0 or (epoch + 1) % 25 == 0 or (epoch + 1) == max_epochs):
                print(f"  [SpaGCN Training] Epoch {epoch + 1:3d}/{max_epochs:3d} | KL Loss: {loss_val:.5f}", flush=True)

        net.eval()
        with torch.no_grad():
            _, q_final = net(X_tensor, adj_tensor)
            y_final = torch.argmax(q_final, dim=1).cpu().numpy()
        return net, loss_history, y_final

    def _search_louvain_res(
        self,
        X_tensor: torch.Tensor,
        adj_tensor: torch.Tensor,
        target_num: int,
        seed: int,
        lr: float,
        weight_decay: float,
        update_interval: int,
        n_neighbors: int = 10,
        start: float = 0.7,
        step: float = 0.1,
        tol: float = 5e-3,
        max_epochs: int = 20,
        max_run: int = 10
    ) -> Optional[float]:
        """Binary resolution search matching SpaGCN/util.py::search_res."""
        res = start
        _, _, y_pred = self._fit_once(
            X_tensor, adj_tensor, target_num, "louvain", res, n_neighbors, seed,
            lr, max_epochs, weight_decay, update_interval, tol, verbose=False
        )
        old_num = len(np.unique(y_pred))
        if old_num == target_num:
            return res

        for _ in range(max_run):
            old_sign = 1 if old_num < target_num else -1
            trial_res = max(0.01, res + step * old_sign)
            _, _, y_pred = self._fit_once(
                X_tensor, adj_tensor, target_num, "louvain", trial_res, n_neighbors, seed,
                lr, max_epochs, weight_decay, update_interval, tol, verbose=False
            )
            new_num = len(np.unique(y_pred))
            if new_num == target_num:
                return trial_res
            new_sign = 1 if new_num < target_num else -1
            if new_sign == old_sign:
                res = trial_res
                old_num = new_num
            else:
                step = step / 2.0
        return None

    def train(
        self,
        X_combined: np.ndarray,
        adj: np.ndarray,
        n_clusters: int,
        seed: int = 42,
        init: str = "louvain",
        n_neighbors: int = 10,
        lr: float = 0.005,
        max_epochs: int = 200,
        weight_decay: float = 0.0,
        update_interval: int = 3,
        tol: float = 1e-3
    ):
        """Train SpaGCN GCN-DEC using unnormalized Gaussian kernel adjacency (matching SpaGCN.py)."""
        assert X_combined.shape[0] == adj.shape[0] == adj.shape[1]

        if self.l is None:
            self.l = search_l(p=0.5, adj=adj)
        print(f"Using characteristic spatial length scale: l = {self.l:.4f}")

        # Unnormalized Gaussian spatial kernel matching SpaGCN.py line 59 (self-weight=1.0, mean neighbor sum~p)
        adj_exp = np.exp(-1.0 * (adj ** 2) / (2.0 * (self.l ** 2))).astype(np.float32)

        X_tensor = torch.FloatTensor(X_combined).to(self.device)
        adj_tensor = torch.FloatTensor(adj_exp).to(self.device)

        chosen_init = init
        res = 0.7
        if chosen_init == "louvain":
            try:
                found_res = self._search_louvain_res(
                    X_tensor, adj_tensor, target_num=n_clusters, seed=seed,
                    lr=lr, weight_decay=weight_decay, update_interval=update_interval,
                    n_neighbors=n_neighbors, tol=tol, max_epochs=min(20, max_epochs)
                )
                if found_res is not None:
                    res = found_res
                    print(f"Louvain resolution matched n_clusters={n_clusters}: res={res:.4f}")
                else:
                    print(f"Louvain search_res did not hit exact n_clusters={n_clusters}; falling back to KMeans init (seed={seed}).")
                    chosen_init = "kmeans"
            except Exception as e:
                print(f"Louvain init unavailable ({e}); falling back to KMeans init (seed={seed}).")
                chosen_init = "kmeans"

        # Reset seed right before final training so initialization & model weights match search_res
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        net, loss_history, y_final = self._fit_once(
            X_tensor, adj_tensor, n_clusters, chosen_init, res, n_neighbors, seed,
            lr, max_epochs, weight_decay, update_interval, tol, verbose=True
        )

        # Guard against rare empty-cluster collapse during Louvain DEC training
        if chosen_init == "louvain" and len(np.unique(y_final)) != n_clusters:
            print(f"Warning: Louvain DEC yielded {len(np.unique(y_final))} != {n_clusters} clusters; re-running with KMeans init (seed={seed}).")
            net, loss_history, _ = self._fit_once(
                X_tensor, adj_tensor, n_clusters, "kmeans", res, n_neighbors, seed,
                lr, max_epochs, weight_decay, update_interval, tol, verbose=True
            )

        self.model = net
        self.embed = X_tensor
        self.adj_exp = adj_tensor
        self.loss_history = loss_history

    def extract_latent_representation(self) -> np.ndarray:
        """Extract final GCN spot embeddings z."""
        self.model.eval()
        with torch.no_grad():
            z, _ = self.model(self.embed, self.adj_exp)
            return z.cpu().numpy()

    def predict(self):
        """Extract native DEC domain predictions (argmax q) and soft probabilities q."""
        self.model.eval()
        with torch.no_grad():
            _, q = self.model(self.embed, self.adj_exp)
            y_pred = torch.argmax(q, dim=1).cpu().numpy()
            prob = q.cpu().numpy()
            return y_pred, prob


# ==============================================================================
# CLUSTERING & EVALUATION ENGINE
# ==============================================================================
def run_mclust(data: np.ndarray, n_clusters: int, seed: int = 42, max_dims: int = 30) -> np.ndarray:
    """Clustering using R's mclust package with PCA dimension reduction and GMM fallback."""
    data_mat = np.array(data, dtype=np.float64)
    if data_mat.shape[1] > max_dims:
        n_comps = min(max_dims, data_mat.shape[0] - 1, data_mat.shape[1])
        data_mat = PCA(n_components=n_comps, random_state=seed).fit_transform(data_mat)

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
        robjects.globalenv['tmp_spagcn_mat'] = numpy2ri.numpy2rpy(data_mat)
        res = robjects.r(f'run_mclust_native(tmp_spagcn_mat, {n_clusters}, {seed})')
        return np.array(res).astype(int) - 1
    except Exception as e:
        print(f"Warning: R mclust failed ({e}). Falling back to GaussianMixture...")
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(n_components=n_clusters, random_state=seed).fit_predict(data_mat)


def run_kmeans(data: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """KMeans clustering on latent embeddings."""
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit_predict(data)


def run_leiden(adata: ad.AnnData, n_clusters: int, seed: int = 42, max_iter: int = 25) -> Optional[np.ndarray]:
    """Leiden clustering on X_spagcn with binary resolution search."""
    adata_tmp = adata.copy()
    sc.pp.neighbors(adata_tmp, use_rep='X_spagcn', n_neighbors=15, random_state=seed)
    res_min, res_max = 0.05, 3.0
    best_labels = None
    best_diff = float('inf')

    try:
        for _ in range(max_iter):
            res = (res_min + res_max) / 2.0
            try:
                sc.tl.leiden(adata_tmp, resolution=res, key_added='leiden_temp', random_state=seed,
                             flavor='igraph', n_iterations=2, directed=False)
            except TypeError:
                sc.tl.leiden(adata_tmp, resolution=res, key_added='leiden_temp', random_state=seed)

            n_found = len(adata_tmp.obs['leiden_temp'].unique())
            diff = abs(n_found - n_clusters)

            if diff < best_diff:
                best_diff = diff
                best_labels = adata_tmp.obs['leiden_temp'].astype(int).to_numpy()

            if n_found == n_clusters:
                return best_labels
            if n_found < n_clusters:
                res_min = res
            else:
                res_max = res
    except ImportError as e:
        print(f"Warning: Leiden skipped ({e}). Install python-igraph/leidenalg to enable Leiden.")
        return None

    return best_labels


def _valid_gt_mask(gt_series: pd.Series) -> np.ndarray:
    s_str = gt_series.astype(str).str.strip()
    return (
        gt_series.notna().to_numpy()
        & ~s_str.isin(['unknown', 'Exclude', '-1', 'nan', 'None', '']).to_numpy()
    )


def evaluate_clustering(y_true, y_pred, features: np.ndarray) -> Dict[str, float]:
    """Compute 9 clustering evaluation metrics safely for string/int/categorical annotations."""
    gt_series = pd.Series(y_true)
    valid_mask = _valid_gt_mask(gt_series)
    y_true_valid = gt_series[valid_mask].astype(str).to_numpy()
    y_pred_valid = np.asarray(y_pred)[valid_mask]
    features_valid = features[valid_mask]

    if len(y_true_valid) == 0 or len(np.unique(y_pred_valid)) < 2:
        return {
            'ARI': 0.0, 'NMI': 0.0, 'Silhouette': 0.0, 'AMI': 0.0,
            'CHI': 0.0, 'DBI': 0.0, 'Homogeneity': 0.0, 'V-measure': 0.0, 'FMI': 0.0
        }

    y_true_encoded = LabelEncoder().fit_transform(y_true_valid)
    y_pred_encoded = LabelEncoder().fit_transform(y_pred_valid.astype(str))

    ari = float(adjusted_rand_score(y_true_encoded, y_pred_encoded))
    nmi = float(normalized_mutual_info_score(y_true_encoded, y_pred_encoded))
    ami = float(adjusted_mutual_info_score(y_true_encoded, y_pred_encoded))
    homo = float(homogeneity_score(y_true_encoded, y_pred_encoded))
    v_meas = float(v_measure_score(y_true_encoded, y_pred_encoded))
    fmi = float(fowlkes_mallows_score(y_true_encoded, y_pred_encoded))

    try:
        sil = float(silhouette_score(features_valid, y_pred_encoded))
    except Exception:
        sil = 0.0
    try:
        chi = float(calinski_harabasz_score(features_valid, y_pred_encoded))
    except Exception:
        chi = 0.0
    try:
        dbi = float(davies_bouldin_score(features_valid, y_pred_encoded))
    except Exception:
        dbi = 0.0

    return {
        'ARI': ari,
        'NMI': nmi,
        'Silhouette': sil,
        'AMI': ami,
        'CHI': chi,
        'DBI': dbi,
        'Homogeneity': homo,
        'V-measure': v_meas,
        'FMI': fmi
    }


# ==============================================================================
# VISUALIZATION SUITE
# ==============================================================================
def plot_clustering_and_spatial(
    adata: ad.AnnData,
    title_prefix: str = "SpaGCN",
    save_path: Optional[str] = None,
    show_plots: bool = True
):
    """Plot UMAP and Spatial maps for Ground Truth, SpaGCN, SpaGCN Refined, KMeans, Leiden, and mclust."""
    if 'X_umap' not in adata.obsm:
        rep = 'X_spagcn' if 'X_spagcn' in adata.obsm else None
        try:
            sc.pp.neighbors(adata, use_rep=rep, n_neighbors=15)
            sc.tl.umap(adata)
        except Exception as e:
            print(f"Warning: Failed to compute UMAP in plot_clustering_and_spatial: {e}")

    panels = [
        ('ground_truth', 'Ground Truth'),
        ('spagcn_pred', 'SpaGCN (Native DEC)'),
        ('spagcn_refined', 'SpaGCN Refined'),
        ('kmeans', 'KMeans'),
        ('leiden', 'Leiden'),
        ('mclust', 'mclust'),
    ]
    panels = [(col, label) for col, label in panels if col in adata.obs]
    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5.5 * n_rows), squeeze=False)

    for row_idx, (col, label) in enumerate(panels):
        adata.obs[col] = adata.obs[col].astype(str).astype('category')
        sc.pl.umap(adata, color=col, ax=axes[row_idx, 0], title=f'{title_prefix} UMAP: {label}', show=False, size=20)
        if 'spatial' in adata.obsm:
            sc.pl.embedding(adata, basis='spatial', color=col, ax=axes[row_idx, 1], title=f'{title_prefix} Spatial: {label}', show=False, size=20)
            axes[row_idx, 1].invert_yaxis()
        else:
            axes[row_idx, 1].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization plot image to: {save_path}")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_loss(loss_history: List[float], save_path: Optional[str] = None):
    """Plot and save training KL divergence loss curve."""
    if not loss_history:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, label='SpaGCN KL Divergence Loss', color='#e377c2', linewidth=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('SpaGCN Training Loss Curve')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_summary_boxplots(df_benchmark: pd.DataFrame, save_dir: str = "results", visuals_dir: Optional[str] = None):
    """Plot Box & Whiskers plots summarizing metric distributions across algorithms and datasets."""
    if df_benchmark.empty:
        return
    v_dir = visuals_dir or save_dir
    os.makedirs(v_dir, exist_ok=True)
    metrics_to_plot = ['ARI', 'NMI', 'Silhouette', 'AMI', 'FMI']

    for metric in metrics_to_plot:
        if metric not in df_benchmark.columns:
            continue
        plt.figure(figsize=(11, 6))
        sns.boxplot(x='Dataset', y=metric, hue='Algorithm', data=df_benchmark, palette='Set2')
        plt.title(f'SpaGCN: {metric} Distribution across Datasets and Clustering Algorithms', fontsize=14, fontweight='bold')
        plt.xlabel('Dataset', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.xticks(rotation=25, ha='right')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        save_path = os.path.join(v_dir, f"spagcn_summary_boxplot_{metric}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_cross_seed_linearity_and_spread(df_benchmark: pd.DataFrame, save_dir: str = "results", visuals_dir: Optional[str] = None):
    """Generate cross-seed stability diagnostics (Mean, Std) across algorithms & datasets."""
    if df_benchmark.empty:
        return
    v_dir = visuals_dir or save_dir
    os.makedirs(v_dir, exist_ok=True)

    summary_df = df_benchmark.groupby(['Dataset', 'Algorithm'])[['ARI', 'NMI', 'Silhouette', 'AMI', 'FMI']].agg(['mean', 'std'])
    summary_df.columns = [f"{col[0]}_{col[1]}" for col in summary_df.columns]
    summary_df.reset_index(inplace=True)
    summary_path = os.path.join(save_dir, "spagcn_cross_seed_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Saved] SpaGCN cross-seed stability diagnostics table saved to: {summary_path}")


def plot_spatial_domains_and_confidence(adatas_dict: Dict[str, ad.AnnData], save_dir: str = "results", visuals_dir: Optional[str] = None):
    """Side-by-Side Ground Truth, SpaGCN Refined Domains, and DEC Assignment Confidence maps."""
    if not adatas_dict:
        return
    v_dir = visuals_dir or save_dir
    os.makedirs(v_dir, exist_ok=True)
    datasets = list(adatas_dict.keys())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(n_datasets, 3, figsize=(18, 5 * n_datasets), squeeze=False)

    for row_idx, d_name in enumerate(datasets):
        adata_plot = adatas_dict[d_name]
        coords = adata_plot.obsm.get('spatial', None)
        if coords is None:
            continue

        gt_series = adata_plot.obs.get('ground_truth', None)
        gt_labels = pd.factorize(gt_series)[0] if gt_series is not None else np.zeros(coords.shape[0])

        domain_col = next(
            (c for c in ['spagcn_refined', 'spagcn_pred', 'mclust', 'kmeans', 'leiden'] if c in adata_plot.obs),
            None
        )
        pred_labels = pd.factorize(adata_plot.obs[domain_col])[0] if domain_col else gt_labels
        confidence = adata_plot.obs['spagcn_confidence'].values if 'spagcn_confidence' in adata_plot.obs else np.ones(coords.shape[0])

        ax0 = axes[row_idx, 0]
        ax0.scatter(coords[:, 0], coords[:, 1], c=gt_labels, cmap='tab20', s=25, alpha=0.9)
        ax0.set_title(f"{d_name}\nSpatial: Ground Truth", fontsize=12, fontweight='bold')
        ax0.set_aspect('equal')
        ax0.invert_yaxis()
        ax0.axis('off')

        ax1 = axes[row_idx, 1]
        ax1.scatter(coords[:, 0], coords[:, 1], c=pred_labels, cmap='tab20', s=25, alpha=0.9)
        ax1.set_title(f"{d_name}\nSpatial: SpaGCN ({domain_col})", fontsize=12, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.invert_yaxis()
        ax1.axis('off')

        ax2 = axes[row_idx, 2]
        sc2 = ax2.scatter(coords[:, 0], coords[:, 1], c=confidence, cmap='viridis', vmin=0.0, vmax=1.0, s=25, alpha=0.9)
        ax2.set_title(f"{d_name}\nSpatial: DEC Max Probability", fontsize=12, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.invert_yaxis()
        ax2.axis('off')
        plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(v_dir, "spagcn_spatial_domains_and_confidence.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_latent_umap_manifolds(adatas_dict: Dict[str, ad.AnnData], save_dir: str = "results", visuals_dir: Optional[str] = None):
    """Latent UMAP Manifold projections (Ground Truth, SpaGCN Refined, DEC Confidence)."""
    if not adatas_dict:
        return
    v_dir = visuals_dir or save_dir
    os.makedirs(v_dir, exist_ok=True)
    datasets = list(adatas_dict.keys())
    n_datasets = len(datasets)

    fig, axes = plt.subplots(n_datasets, 3, figsize=(18, 5 * n_datasets), squeeze=False)

    for row_idx, d_name in enumerate(datasets):
        adata_plot = adatas_dict[d_name]
        if 'X_umap' not in adata_plot.obsm:
            sc.pp.neighbors(adata_plot, use_rep='X_spagcn', n_neighbors=15)
            sc.tl.umap(adata_plot)
        umap_coords = adata_plot.obsm['X_umap']

        gt_series = adata_plot.obs.get('ground_truth', None)
        gt_labels = pd.factorize(gt_series)[0] if gt_series is not None else np.zeros(umap_coords.shape[0])

        domain_col = next(
            (c for c in ['spagcn_refined', 'spagcn_pred', 'mclust', 'kmeans', 'leiden'] if c in adata_plot.obs),
            None
        )
        pred_labels = pd.factorize(adata_plot.obs[domain_col])[0] if domain_col else gt_labels
        confidence = adata_plot.obs['spagcn_confidence'].values if 'spagcn_confidence' in adata_plot.obs else np.ones(umap_coords.shape[0])

        ax1 = axes[row_idx, 0]
        ax1.scatter(umap_coords[:, 0], umap_coords[:, 1], c=gt_labels, cmap='tab20', s=20, alpha=0.85)
        ax1.set_title(f"{d_name}\nUMAP: Ground Truth", fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = axes[row_idx, 1]
        ax2.scatter(umap_coords[:, 0], umap_coords[:, 1], c=pred_labels, cmap='tab20', s=20, alpha=0.85)
        ax2.set_title(f"{d_name}\nUMAP: SpaGCN ({domain_col})", fontsize=12, fontweight='bold')
        ax2.axis('off')

        ax3 = axes[row_idx, 2]
        sc3 = ax3.scatter(umap_coords[:, 0], umap_coords[:, 1], c=confidence, cmap='viridis', vmin=0.0, vmax=1.0, s=20, alpha=0.85)
        ax3.set_title(f"{d_name}\nUMAP: DEC Max Probability", fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(v_dir, "spagcn_latent_umap_manifolds.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def export_predictions_csv(adata: ad.AnnData, dataset_name: str, seed: int, out_dir: str = "results"):
    """Export spot-level predictions, spatial coordinates, and cluster assignments to CSV."""
    coords = adata.obsm.get('spatial', None)
    pred_data = {
        'barcode': adata.obs_names.tolist(),
        'ground_truth': adata.obs['ground_truth'].tolist() if 'ground_truth' in adata.obs else ['unknown'] * adata.n_obs,
        'spagcn_pred': adata.obs['spagcn_pred'].tolist() if 'spagcn_pred' in adata.obs else [-1] * adata.n_obs,
        'spagcn_refined': adata.obs['spagcn_refined'].tolist() if 'spagcn_refined' in adata.obs else [-1] * adata.n_obs,
        'spagcn_confidence': adata.obs['spagcn_confidence'].tolist() if 'spagcn_confidence' in adata.obs else [0.0] * adata.n_obs,
        'kmeans': adata.obs['kmeans'].tolist() if 'kmeans' in adata.obs else [-1] * adata.n_obs,
        'leiden': adata.obs['leiden'].tolist() if 'leiden' in adata.obs else [-1] * adata.n_obs,
        'mclust': adata.obs['mclust'].tolist() if 'mclust' in adata.obs else [-1] * adata.n_obs,
    }
    if coords is not None and coords.shape[1] >= 2:
        pred_data['spatial_x'] = coords[:, 0].tolist()
        pred_data['spatial_y'] = coords[:, 1].tolist()

    df_preds = pd.DataFrame(pred_data)
    save_path = os.path.join(out_dir, f"predictions_spagcn_{dataset_name}_seed_{seed}.csv")
    df_preds.to_csv(save_path, index=False)
    print(f"Exported spot-level predictions to: {save_path}")


# ==============================================================================
# DATASET RESOLVER & PREPROCESSING
# ==============================================================================
def resolve_data_directory(dataset_name: str, dataset_cfg: dict, env_mode: str = "auto") -> str:
    """Resolve the dataset directory across Kaggle and local environments without cross-dataset fallback."""
    is_kaggle = (env_mode == "kaggle") or (env_mode == "auto" and os.path.exists('/kaggle/input'))

    primary_dir = dataset_cfg.get("kaggle_dir" if is_kaggle else "local_dir", "")
    tail = os.path.basename(os.path.normpath(primary_dir or dataset_name))
    candidate_paths = [
        primary_dir,
        dataset_cfg.get("local_dir" if is_kaggle else "kaggle_dir", ""),
        os.path.join("../../datasets", tail),
        os.path.join("../datasets", tail),
        os.path.join("datasets", tail),
        os.path.join("./datasets", tail),
        os.path.join("data", tail),
    ]

    for p in candidate_paths:
        if p and os.path.exists(p) and (
            os.path.exists(os.path.join(p, 'adata_RNA.h5ad'))
            or os.path.exists(os.path.join(p, 'mRNA_data.h5ad'))
        ):
            return p

    if is_kaggle:
        search_terms = [tail.lower(), dataset_name.lower().replace("-", "_"), dataset_name.lower().replace("-", "")]
        for root, _, files in os.walk('/kaggle/input'):
            if 'adata_RNA.h5ad' in files or 'mRNA_data.h5ad' in files:
                r_low = root.lower()
                if any(t in r_low for t in search_terms):
                    return root

    raise FileNotFoundError(
        f"Dataset directory for '{dataset_name}' not found. Checked: {[p for p in candidate_paths if p]}"
    )


def _load_annotations(adata_rna: ad.AnnData, data_dir: str, anno_file: str) -> pd.Series:
    """Load spot annotations safely without column collisions."""
    annotation_path = os.path.join(data_dir, anno_file)
    if not os.path.exists(annotation_path):
        print(f"Warning: Annotation file '{anno_file}' not found. Defaulting to 'unknown'.")
        return pd.Series('unknown', index=adata_rna.obs_names, dtype=object)

    print(f"Loading annotations from: {annotation_path}")
    annotation = pd.read_csv(annotation_path)

    barcode_col = next((c for c in ['Barcode', 'barcode', 'Unnamed: 0'] if c in annotation.columns), None)
    if barcode_col is not None:
        annotation = annotation.set_index(barcode_col)
    else:
        annotation = annotation.set_index(annotation.columns[0])

    label_col = next(
        (c for c in ['manual-anno', 'cluster', 'ground_truth', 'Annotation', 'annotation', 'cell_type']
         if c in annotation.columns),
        annotation.columns[0]
    )
    gt_map = annotation[label_col].to_dict()
    mapped = pd.Series([gt_map.get(b, 'unknown') for b in adata_rna.obs_names], index=adata_rna.obs_names)
    return mapped.fillna('unknown')


def _tfidf(X):
    """TF-IDF normalization for ATAC chromatin accessibility matrices."""
    col_sum = np.asarray(X.sum(axis=0)).ravel() + 1e-12
    row_sum = np.asarray(X.sum(axis=1)).ravel() + 1e-12
    idf = X.shape[0] / col_sum
    if sp.issparse(X):
        tf = X.multiply(1.0 / row_sum[:, None])
        return tf.multiply(idf)
    tf = X / row_sum[:, None]
    return tf * idf


def _clr_normalize(X: np.ndarray) -> np.ndarray:
    """Centered log-ratio (CLR) normalization per spot for CITE-seq ADT data."""
    out = np.empty_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        row = X[i]
        pos = row[row > 0]
        denom = np.exp(np.sum(np.log1p(pos)) / len(row)) if len(row) > 0 and len(pos) > 0 else 1.0
        out[i] = np.log1p(row / denom)
    return out


def _rms_balance(X: np.ndarray) -> np.ndarray:
    """Scale modality matrix by scalar RMS so PC variance decay is preserved while balancing modalities."""
    rms = float(np.sqrt(np.mean(X ** 2)))
    return (X / (rms + 1e-8)).astype(np.float32)


# ==============================================================================
# SpaGCN WORKFLOW ENGINE
# ==============================================================================
def run_spagcn_workflow(
    dataset_name: str,
    dataset_cfg: dict,
    env_mode: str,
    seed: int,
    device: str,
    model_type: str = "simple_gc_dec",
    init_method: str = "louvain",
    lr: float = 0.005,
    epochs: int = 200,
    show_plots: bool = True,
    out_dir: str = "results",
    token: Optional[str] = None,
    api_url: Optional[str] = DEFAULT_API_URL
):
    """Execute SpaGCN multi-omics representation learning, clustering, and evaluation for one seed."""
    start_time = time.time()
    print(f"\n--- Running Seed: {seed} ---")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    data_dir = resolve_data_directory(dataset_name, dataset_cfg, env_mode)
    print(f"Loading data from: {data_dir}")

    rna_filename = next((c for c in ['adata_RNA.h5ad', 'mRNA_data.h5ad'] if os.path.exists(os.path.join(data_dir, c))), 'adata_RNA.h5ad')
    adata_rna = sc.read_h5ad(os.path.join(data_dir, rna_filename))
    adata_rna.var_names_make_unique()

    mod2_filename = next((c for c in dataset_cfg["mod2_candidates"] if os.path.exists(os.path.join(data_dir, c))), dataset_cfg["mod2_candidates"][0])
    adata_mod2 = sc.read_h5ad(os.path.join(data_dir, mod2_filename))
    adata_mod2.var_names_make_unique()

    # Align spot barcodes between RNA and Modality 2
    if list(adata_rna.obs_names) != list(adata_mod2.obs_names):
        mod2_set = set(adata_mod2.obs_names)
        common_spots = [s for s in adata_rna.obs_names if s in mod2_set]
        if not common_spots:
            raise ValueError(f"No overlapping spot barcodes between RNA and {mod2_filename} in {data_dir}.")
        print(f"Aligning {len(common_spots)} shared spot barcodes between RNA ({adata_rna.n_obs}) and Mod2 ({adata_mod2.n_obs})...")
        adata_rna = adata_rna[common_spots].copy()
        adata_mod2 = adata_mod2[common_spots].copy()

    gt_series = _load_annotations(adata_rna, data_dir, dataset_cfg["anno_file"])
    adata_rna.obs['ground_truth'] = gt_series
    adata_mod2.obs['ground_truth'] = gt_series

    print("RNA shape:", adata_rna.shape)
    print("Modality 2 shape:", adata_mod2.shape)

    # 1. Extract Spatial Coordinates
    if 'spatial' in adata_rna.obsm:
        coords = np.array(adata_rna.obsm['spatial'], dtype=np.float32)
    elif 'x' in adata_rna.obs and 'y' in adata_rna.obs:
        coords = np.column_stack([adata_rna.obs['x'].values, adata_rna.obs['y'].values]).astype(np.float32)
        adata_rna.obsm['spatial'] = coords
    else:
        raise ValueError(f"Spatial coordinates missing in adata_rna.obsm['spatial'] and obs['x'/'y'] for {dataset_name}.")

    adj_dist = calculate_spatial_adj_matrix(coords[:, 0], coords[:, 1])

    # 2. Multi-Omics Feature Integration (preserving within-modality PC variance)
    print("Preprocessing & Integrating RNA and Modality 2 for SpaGCN...")
    adata_rna_norm = adata_rna.copy()
    sc.pp.filter_genes(adata_rna_norm, min_cells=3)
    sc.pp.normalize_total(adata_rna_norm, target_sum=1e4)
    sc.pp.log1p(adata_rna_norm)
    if adata_rna_norm.n_vars > 3000:
        sc.pp.highly_variable_genes(adata_rna_norm, flavor="seurat", n_top_genes=3000)
        adata_rna_norm = adata_rna_norm[:, adata_rna_norm.var['highly_variable']].copy()
    X_rna_mat = adata_rna_norm.X.toarray() if issparse(adata_rna_norm.X) else np.asarray(adata_rna_norm.X, dtype=np.float32)
    n_pcs_rna = min(32, X_rna_mat.shape[0] - 1, X_rna_mat.shape[1])
    X_rna_pca = PCA(n_components=n_pcs_rna, random_state=seed).fit_transform(X_rna_mat)

    if dataset_cfg.get("type") == "mouse_brain":
        if 'X_lsi' in adata_mod2.obsm:
            X_mod2_pca = np.asarray(adata_mod2.obsm['X_lsi'], dtype=np.float32)[:, :32]
        else:
            X_atac = adata_mod2.X.toarray() if issparse(adata_mod2.X) else np.asarray(adata_mod2.X, dtype=np.float32)
            # Run TF-IDF + log1p + SVD (LSI) if raw non-negative peak matrix
            if X_atac.min() >= 0 and "normalized" not in mod2_filename.lower():
                X_atac = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(_tfidf(X_atac))
                X_atac = np.log1p(X_atac * 1e4)
            n_pcs_mod2 = min(32, X_atac.shape[0] - 1, X_atac.shape[1] - 1)
            X_mod2_pca = sklearn.utils.extmath.randomized_svd(X_atac, n_components=n_pcs_mod2, random_state=seed)[0]
    else:
        X_adt = adata_mod2.X.toarray() if issparse(adata_mod2.X) else np.asarray(adata_mod2.X, dtype=np.float32)
        if X_adt.min() >= 0:
            X_adt = _clr_normalize(X_adt)
        n_pcs_mod2 = min(32, X_adt.shape[0] - 1, X_adt.shape[1])
        X_mod2_pca = PCA(n_components=n_pcs_mod2, random_state=seed).fit_transform(X_adt)

    X_combined = np.concatenate([_rms_balance(X_rna_pca), _rms_balance(X_mod2_pca)], axis=1)

    valid_mask = _valid_gt_mask(adata_rna.obs['ground_truth'])
    valid_labels = adata_rna.obs['ground_truth'][valid_mask].astype(str)
    n_clusters = int(valid_labels.nunique()) if len(valid_labels) > 0 else 8
    print(f"Target cluster count: {n_clusters} clusters (grid shape: {dataset_cfg.get('shape', 'square')})")

    # 3. Train SpaGCN Model
    print(f"Initializing SpaGCN ({model_type}, init={init_method}) on device: {device}...")
    spagcn_model = SpaGCNModel(model_type=model_type, alpha=0.2, dropout=0.5, device=device)
    spagcn_model.train(
        X_combined=X_combined,
        adj=adj_dist,
        n_clusters=n_clusters,
        seed=seed,
        init=init_method,
        lr=lr,
        max_epochs=epochs,
        weight_decay=0.0,
        update_interval=3,
        tol=1e-3
    )

    latent_emb = spagcn_model.extract_latent_representation()
    y_spagcn_pred, prob = spagcn_model.predict()
    grid_shape = dataset_cfg.get("shape", "square")
    sample_ids = adata_rna.obs_names.tolist()
    y_spagcn_refined = refine_spatial_domains(sample_ids, y_spagcn_pred.tolist(), adj_dist, shape=grid_shape)

    print(f"SpaGCN joint latent embeddings extracted with shape: {latent_emb.shape}")

    adata = adata_rna.copy()
    adata.obsm['X_spagcn'] = latent_emb
    adata.obsm['feat'] = latent_emb
    adata.obs['spagcn_pred'] = y_spagcn_pred
    adata.obs['spagcn_refined'] = y_spagcn_refined
    adata.obs['spagcn_confidence'] = np.max(prob, axis=1)
    adata.uns['selected_seed'] = seed

    print("Computing UMAP embeddings on SpaGCN latent representations...")
    try:
        sc.pp.neighbors(adata, use_rep='X_spagcn', n_neighbors=15, random_state=seed)
        sc.tl.umap(adata, random_state=seed)
    except Exception as e:
        print(f"Warning: UMAP on X_spagcn failed ({e}).")

    # 4. Post-hoc Clustering Benchmarking on X_spagcn
    print(f"Running KMeans on SpaGCN embeddings with n_clusters={n_clusters}...")
    adata.obs['kmeans'] = run_kmeans(latent_emb, n_clusters=n_clusters, seed=seed)

    print("Running Leiden on SpaGCN embeddings...")
    leiden_labels = run_leiden(adata, n_clusters=n_clusters, seed=seed)
    if leiden_labels is not None:
        adata.obs['leiden'] = leiden_labels

    print("Running mclust on SpaGCN embeddings...")
    adata.obs['mclust'] = run_mclust(latent_emb, n_clusters=n_clusters, seed=seed)

    # 5. Evaluate Metrics (Native DEC, Native DEC Refined, KMeans, Leiden, mclust)
    metrics_spagcn_pred = evaluate_clustering(adata.obs['ground_truth'].values, y_spagcn_pred, latent_emb)
    metrics_spagcn_refined = evaluate_clustering(adata.obs['ground_truth'].values, np.array(y_spagcn_refined), latent_emb)
    metrics_kmeans = evaluate_clustering(adata.obs['ground_truth'].values, adata.obs['kmeans'].values, latent_emb)
    metrics_leiden = evaluate_clustering(adata.obs['ground_truth'].values, leiden_labels, latent_emb) if leiden_labels is not None else None
    metrics_mclust = evaluate_clustering(adata.obs['ground_truth'].values, adata.obs['mclust'].values, latent_emb)

    eval_items = [
        ("SpaGCN", "SpaGCN (Native DEC)", metrics_spagcn_pred),
        ("SpaGCN_Refined", f"SpaGCN Refined ({grid_shape})", metrics_spagcn_refined),
        ("KMeans", "KMeans (on X_spagcn)", metrics_kmeans),
    ]
    if metrics_leiden is not None:
        eval_items.append(("Leiden", "Leiden (on X_spagcn)", metrics_leiden))
    eval_items.append(("mclust", "mclust (on X_spagcn)", metrics_mclust))

    for _, label, m_dict in eval_items:
        print(f"\n--- Clustering Evaluation (SpaGCN {dataset_name} Seed {seed} - {label}) ---")
        for k, v in m_dict.items():
            print(f"{k}: {v:.4f}")

    # 6. Visualizations & Local Exports
    plot_save_path = os.path.join(out_dir, f"spagcn_plot_{dataset_name}_seed_{seed}.png")
    plot_clustering_and_spatial(adata, title_prefix=f"SpaGCN ({dataset_name} Seed {seed})", save_path=plot_save_path, show_plots=show_plots)

    loss_save_path = os.path.join(out_dir, f"spagcn_loss_{dataset_name}_seed_{seed}.png")
    plot_loss(spagcn_model.loss_history, save_path=loss_save_path)

    export_predictions_csv(adata, dataset_name, seed, out_dir=out_dir)

    # 7. Dashboard API Sync (primary SpaGCN metrics = SpaGCN_Refined)
    y_true_all = adata.obs['ground_truth'].astype(str).tolist()
    y_pred_primary = np.asarray(y_spagcn_refined).astype(str)
    try:
        pred_int = LabelEncoder().fit_transform(y_pred_primary)
        sample_sil = silhouette_samples(latent_emb, pred_int).tolist() if len(np.unique(pred_int)) > 1 else []
    except Exception:
        sample_sil = []

    embeddings_data = {
        "umapCoordinates": adata.obsm['X_umap'].tolist() if 'X_umap' in adata.obsm else [],
        "spatialCoordinates": coords.tolist(),
        "predictedLabels": y_pred_primary.tolist(),
        "groundTruthLabels": y_true_all,
        "sampleSilhouettes": sample_sil,
    }

    dash_info = DATASET_DASHBOARD_INFO.get(dataset_name, {"datasetId": dataset_name, "datasetName": dataset_name})
    training_results = {
        "loss_history": spagcn_model.loss_history,
        "runtime_seconds": time.time() - start_time,
        "device": str(spagcn_model.device),
        "best_epoch": len(spagcn_model.loss_history),
        "best_score": metrics_spagcn_refined["ARI"],
        "architecture": f"Spatial Graph Convolutional Network + Deep Embedding Clustering ({model_type})"
    }
    hyperparams = {
        "model_type": model_type,
        "init_method": init_method,
        "grid_shape": grid_shape,
        "alpha": 0.2,
        "epochs": epochs,
        "learning_rate": lr,
        "weight_decay": 0.0,
        "n_latent": int(latent_emb.shape[1]),
        "l": float(spagcn_model.l) if spagcn_model.l is not None else 0.0,
        "n_clusters": n_clusters,
        "seed": seed,
        "all_algorithms": {algo_key: m_dict for algo_key, _, m_dict in eval_items}
    }

    export_dashboard_experiment(
        model_id="SpaGCN",
        model_name="SpaGCN",
        dataset_name=dash_info["datasetId"],
        seed=seed,
        metrics_dict=metrics_spagcn_refined,
        training_results=training_results,
        hyperparameters=hyperparams,
        embeddings_data=embeddings_data,
        output_dir=out_dir,
        api_url=api_url,
        token=token,
        dataset_display_name=dash_info["datasetName"]
    )

    results_entry = [
        {"Dataset": dataset_name, "Seed": seed, "Algorithm": algo_key, **m_dict}
        for algo_key, _, m_dict in eval_items
    ]

    return results_entry, adata


# ==============================================================================
# MULTI-SEED PIPELINE ORCHESTRATION
# ==============================================================================
def run_multi_seed_pipeline(
    datasets: List[str] = datasets_to_run,
    seeds: List[int] = SEEDS,
    env_mode: str = ENV_MODE,
    device: Optional[str] = None,
    model_type: str = "simple_gc_dec",
    init_method: str = "louvain",
    lr: float = 0.005,
    epochs: int = 200,
    show_plots: bool = False,
    out_dir: str = "results",
    token: Optional[str] = None,
    api_url: Optional[str] = DEFAULT_API_URL
):
    """Execute the full SpaGCN pipeline across all scheduled datasets and seeds."""
    os.makedirs(out_dir, exist_ok=True)
    visuals_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(visuals_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device}")

    all_benchmark_results = []
    best_adatas_per_dataset = {}

    for d_name in datasets:
        if d_name not in ALL_DATASETS_CONFIG:
            print(f"Skipping unrecognized dataset: {d_name}")
            continue

        d_cfg = ALL_DATASETS_CONFIG[d_name]
        print("\n" + "=" * 55)
        print(f"STARTING SpaGCN WORKFLOW FOR DATASET: {d_name} OVER {len(seeds)} SEEDS")
        print("=" * 55)

        dataset_entries = []
        best_adata = None
        best_ari = -float('inf')

        for seed in seeds:
            try:
                entries, adata_out = run_spagcn_workflow(
                    dataset_name=d_name,
                    dataset_cfg=d_cfg,
                    env_mode=env_mode,
                    seed=seed,
                    device=device,
                    model_type=model_type,
                    init_method=init_method,
                    lr=lr,
                    epochs=epochs,
                    show_plots=show_plots,
                    out_dir=out_dir,
                    token=token,
                    api_url=api_url
                )
                dataset_entries.extend(entries)
                all_benchmark_results.extend(entries)

                spagcn_ari = next((e["ARI"] for e in entries if e["Algorithm"] == "SpaGCN_Refined"), 0.0)
                if spagcn_ari > best_ari:
                    best_ari = spagcn_ari
                    best_adata = adata_out.copy()

            except Exception as e:
                print(f"Error executing SpaGCN on {d_name} (Seed {seed}): {e}")
                import traceback
                traceback.print_exc()

        if best_adata is not None:
            best_adatas_per_dataset[d_name] = best_adata

        if dataset_entries:
            df_ds = pd.DataFrame(dataset_entries)
            print("\n" + "=" * 55)
            print(f"AVERAGE PERFORMANCE FOR {d_name} ({len(seeds)} seeds)")
            print("=" * 55)
            for algo in ["SpaGCN", "SpaGCN_Refined", "KMeans", "Leiden", "mclust"]:
                algo_df = df_ds[df_ds["Algorithm"] == algo]
                if not algo_df.empty:
                    print(f"\n--- {algo} Performance (Mean ± Std) ---")
                    cols = ['ARI', 'NMI', 'Silhouette', 'AMI', 'CHI', 'DBI', 'Homogeneity', 'V-measure', 'FMI']
                    summary_view = pd.DataFrame({'Mean': algo_df[cols].mean(), 'Std': algo_df[cols].std()})
                    print(summary_view.to_string())
            print("=" * 55 + "\n")

    if all_benchmark_results:
        df_benchmark = pd.DataFrame(all_benchmark_results)
        csv_path = os.path.join(out_dir, "spagcn_benchmark_results.csv")
        df_benchmark.to_csv(csv_path, index=False)
        print(f"\n[Saved] Full benchmark metrics saved to: {csv_path}")

        plot_summary_boxplots(df_benchmark, save_dir=out_dir, visuals_dir=visuals_dir)
        plot_cross_seed_linearity_and_spread(df_benchmark, save_dir=out_dir, visuals_dir=visuals_dir)
        plot_spatial_domains_and_confidence(best_adatas_per_dataset, save_dir=out_dir, visuals_dir=visuals_dir)
        plot_latent_umap_manifolds(best_adatas_per_dataset, save_dir=out_dir, visuals_dir=visuals_dir)

    print("\n" + "=" * 80)
    print(" SpaGCN MULTI-SEED PIPELINE EXECUTION COMPLETED")
    print("=" * 80)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SpaGCN Multi-Seed Benchmark Pipeline")
    parser.add_argument("--datasets", nargs="+", default=["all"], help="Datasets to execute (e.g. mouse-brain-e11-s1 or 'all')")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds to evaluate")
    parser.add_argument("--model_type", type=str, default="simple_gc_dec", choices=["simple_gc_dec", "gc_dec"],
                        help="DEC backbone: 'simple_gc_dec' (official SpaGCN) or 'gc_dec' (2-layer ReLU+Dropout)")
    parser.add_argument("--init", type=str, default="louvain", choices=["louvain", "kmeans"],
                        help="DEC centroid initialization ('louvain' matches official SpaGCN, falls back to kmeans if missing)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate for SpaGCN Adam optimizer")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs for SpaGCN GCN-DEC")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save benchmark metrics and plots")
    parser.add_argument("--no_dashboard", action="store_true", help="Disable uploading to dashboard")
    parser.add_argument("--generate_plots", action="store_true", help="Regenerate summary figures from existing benchmark CSV")

    args = parser.parse_args()

    if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        scheduled_datasets = list(ALL_DATASETS_CONFIG.keys())
    else:
        scheduled_datasets = [d for d in args.datasets if d in ALL_DATASETS_CONFIG]

    output_dir = args.out_dir
    if os.path.exists('/kaggle/working') and output_dir == "results":
        output_dir = "/kaggle/working"

    print("=" * 80)
    print(" SpaGCN MULTI-SEED PIPELINE (STANDALONE EXECUTION)")
    print(f" Default Scheduled Datasets ({len(scheduled_datasets)}): {scheduled_datasets}")
    print(f" Default Ablation Seeds ({len(args.seeds)}): {args.seeds}")
    print(f" Model: {args.model_type} | Init: {args.init} | LR: {args.lr} | Epochs: {args.epochs}")
    print("=" * 80)

    dashboard_token = None
    if not args.no_dashboard:
        dashboard_token = authenticate_dashboard(
            base_url=DEFAULT_DASHBOARD_URL,
            username=DEFAULT_USERNAME,
            password=DEFAULT_PASSWORD
        )

    if args.generate_plots:
        csv_path = os.path.join(output_dir, "spagcn_benchmark_results.csv")
        if os.path.exists(csv_path):
            print(f"Regenerating summary diagnostics from {csv_path}...")
            df_existing = pd.read_csv(csv_path)
            visuals_dir = os.path.join(output_dir, "visualizations")
            plot_summary_boxplots(df_existing, save_dir=output_dir, visuals_dir=visuals_dir)
            plot_cross_seed_linearity_and_spread(df_existing, save_dir=output_dir, visuals_dir=visuals_dir)
        else:
            print(f"Error: {csv_path} not found for --generate_plots.")
        sys.exit(0)

    run_multi_seed_pipeline(
        datasets=scheduled_datasets,
        seeds=args.seeds,
        env_mode=ENV_MODE,
        device=args.device,
        model_type=args.model_type,
        init_method=args.init,
        lr=args.lr,
        epochs=args.epochs,
        show_plots=False,
        out_dir=output_dir,
        token=dashboard_token,
        api_url=None if args.no_dashboard else DEFAULT_API_URL
    )
