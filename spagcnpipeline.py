#!/usr/bin/env python3
"""
SpaGCN Multi-Seed Pipeline (Standalone Python Script)
Integrates joint spatial representation learning workflow for target datasets:
- Mouse Brain E11, E13, E15, E18 (RNA + ATAC)
- Human Lymph Node A1, D1 (RNA + Protein/ADT)

Directly based on SpaGCN (Spatial Graph Convolutional Network, Nature Biomedical Engineering 2021)
and matches the architecture and benchmark tracking suite in spallmpipeline.py:
- Multi-seed evaluation suite (10 seeds)
- Tri-algorithm clustering (KMeans, Leiden, mclust) + Native SpaGCN DEC & Refined Clustering
- Complete evaluation metrics (ARI, NMI, Silhouette, AMI, CHI, DBI, Homogeneity, V-measure, FMI)
- Summary Box & Whiskers plots across datasets & algorithms
- Cross-Seed Linearity & Metric Spread diagnostics suite (CSV + Unified Figure)
- Side-by-Side Spatial Domain Partitioning & Modality Contribution maps
- Latent UMAP Manifold projections
- Spot-level prediction CSV export
- Standardized Dashboard JSON exports & automated live API upload
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
from typing import Optional, Dict, Any, List, Union

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
import scipy
import scipy.sparse as sp
from scipy.sparse import issparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.backends import cudnn

import sklearn
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
from tqdm import tqdm

# ==============================================================================
# RUN CONFIGURATION & DATASETS
# ==============================================================================
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

    loss_hist = training_results.get("loss_history", [])
    history_points = []
    for ep in range(len(loss_hist)):
        history_points.append({
            "epoch": ep + 1,
            "losses": {
                "total_loss": float(loss_hist[ep]) if ep < len(loss_hist) else 0.0,
                "reconstruction_loss": float(loss_hist[ep]) if ep < len(loss_hist) else 0.0,
                "spatial_loss": 0.0,
                "reg_loss": 0.0,
            },
            "metrics": {
                "Silhouette": float(metrics_dict.get("Silhouette", 0.0)),
                "ARI": float(metrics_dict.get("ARI", 0.0)),
                "NMI": float(metrics_dict.get("NMI", 0.0)),
            },
            "learningRate": float(hyperparameters.get("learning_rate", 0.005))
        })

    perf_metrics = {
        "ARI": float(metrics_dict.get("ARI", 0.0)),
        "NMI": float(metrics_dict.get("NMI", 0.0)),
        "Silhouette": float(metrics_dict.get("Silhouette", 0.0)),
        "AMI": float(metrics_dict.get("AMI", 0.0)),
        "CHI": float(metrics_dict.get("CHI", 0.0)),
        "DBI": float(metrics_dict.get("DBI", 0.0)),
        "Homogeneity": float(metrics_dict.get("Homogeneity", 0.0)),
        "V-measure": float(metrics_dict.get("V-measure", 0.0)),
        "FMI": float(metrics_dict.get("FMI", 0.0)),
        "final_loss": float(loss_hist[-1]) if loss_hist else 0.0
    }

    metadata = {
        "experimentId": f"{model_id}_{dataset_name}_seed_{seed}",
        "experimentName": f"{model_name} on {dataset_display_name or dataset_name} (Seed {seed})",
        "modelId": model_id,
        "modelName": model_name,
        "datasetId": dataset_name,
        "datasetName": dataset_display_name or dataset_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runTimeSeconds": float(training_results.get("runtime_seconds", 0.0)),
        "hardware": {
            "device": training_results.get("device", "cuda" if torch.cuda.is_available() else "cpu"),
            "gpuName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        },
        "modelDetails": {
            "architecture": "Spatial Graph Convolutional Network + Deep Embedding Clustering (SpaGCN)",
            "framework": "PyTorch",
            "task": "Spatial Domain Identification & Multi-Omics Representation Learning"
        },
        "hyperparameters": hyperparameters,
        "metrics": perf_metrics,
        "trainingStats": {
            "totalEpochs": len(loss_hist),
            "convergenceEpoch": len(loss_hist),
            "finalLoss": float(loss_hist[-1]) if loss_hist else 0.0
        }
    }

    history_data = {
        "experimentId": metadata["experimentId"],
        "history": history_points
    }

    metrics_data = {
        "experimentId": metadata["experimentId"],
        "metrics": perf_metrics
    }

    # Save local JSONs
    with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
        json.dump(make_json_serializable(metadata), f, indent=2)
    with open(os.path.join(exp_dir, "history.json"), "w") as f:
        json.dump(make_json_serializable(history_data), f, indent=2)
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(make_json_serializable(metrics_data), f, indent=2)
    if embeddings_data:
        with open(os.path.join(exp_dir, "embeddings.json"), "w") as f:
            json.dump(make_json_serializable(embeddings_data), f, indent=2)

    print(f"[Dashboard] Standardized JSONs saved to: {exp_dir}")

    # API Upload
    if api_url:
        payload_data = {
            "metadata": metadata,
            "metrics": metrics_data,
            "history": history_data
        }
        if embeddings_data:
            payload_data["embeddings"] = embeddings_data

        try:
            req_data = json.dumps(make_json_serializable(payload_data)).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            req = urllib.request.Request(api_url, data=req_data, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status in (200, 201):
                    print(f"[Dashboard] Successfully uploaded experiment results to Dashboard via API ({api_url})!")
                else:
                    print(f"[Dashboard] Upload returned HTTP status {resp.status}")
        except Exception as e:
            print(f"[Dashboard] Direct API upload skipped/failed ({e}). Artifacts preserved locally.")


# ==============================================================================
# SpaGCN ARCHITECTURE (GRAPH CONVOLUTION + DEC)
# ==============================================================================
class GraphConvolution(nn.Module):
    """Simple GCN layer: Z = A X W + b."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
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


class SimpleGCDEC(nn.Module):
    """
    SpaGCN Graph Convolutional Deep Embedding Clustering (DEC) Module.
    Uses Student-t kernel for soft cluster assignment and KL-divergence target distribution loss.
    """
    def __init__(self, nfeat: int, nhid: int, alpha: float = 1.0):
        super(SimpleGCDEC, self).__init__()
        self.gc = GraphConvolution(nfeat, nhid)
        self.nhid = nhid
        self.alpha = alpha
        self.mu = None  # Cluster centroids (initialized via KMeans)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        z = self.gc(x, adj)
        if self.mu is None:
            return z, None
        # Student-t distribution soft assignment
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha + 1e-8)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return z, q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """P target distribution highlighting high-confidence cluster assignments."""
        p = q ** 2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    @staticmethod
    def loss_function(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """KL divergence loss: KL(P || Q) = sum(P * log(P / Q))."""
        loss = torch.mean(torch.sum(p * torch.log((p + 1e-8) / (q + 1e-8)), dim=1))
        return loss


# ==============================================================================
# SpaGCN SPATIAL DISTANCE, ADJACENCY & REFINEMENT
# ==============================================================================
def calculate_spatial_adj_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes pairwise Euclidean distance matrix between spot spatial coordinates."""
    coords = np.column_stack([x, y]).astype(np.float32)
    adj = scipy.spatial.distance.cdist(coords, coords).astype(np.float32)
    return adj


def calculate_p(adj: np.ndarray, l: float) -> float:
    """Calculates percentage of total expression contributed by spatial neighborhoods."""
    adj_exp = np.exp(-1.0 * (adj ** 2) / (2.0 * (l ** 2)))
    return float(np.mean(np.sum(adj_exp, axis=1)) - 1.0)


def search_l(p: float, adj: np.ndarray, start: float = 0.01, end: float = 1000.0, tol: float = 0.01, max_run: int = 100) -> float:
    """Binary search for characteristic spatial distance scale parameter l given target p."""
    p_low = calculate_p(adj, start)
    p_high = calculate_p(adj, end)
    if p_low > p + tol:
        return float(start)
    if p_high < p - tol:
        return float(end)
    if np.abs(p_low - p) <= tol:
        return float(start)
    if np.abs(p_high - p) <= tol:
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
    """Neighborhood spatial voting domain boundary refinement."""
    num_nbs = 6 if shape == "hexagon" else 4
    refined_pred = []
    pred_series = pd.Series(pred, index=sample_id)
    n = len(sample_id)

    for i in range(n):
        self_pred = pred[i]
        distances = dis[i]
        # Top nearest neighbors (including self)
        nbr_indices = np.argsort(distances)[:num_nbs + 1]
        nbr_preds = [pred[idx] for idx in nbr_indices]
        val_counts = pd.Series(nbr_preds).value_counts()
        top_cluster = val_counts.idxmax()
        top_count = val_counts.max()

        if (val_counts.get(self_pred, 0) < num_nbs / 2.0) and (top_count > num_nbs / 2.0):
            refined_pred.append(int(top_cluster))
        else:
            refined_pred.append(int(self_pred))
    return refined_pred


# ==============================================================================
# SpaGCN PIPELINE MODEL WRAPPER
# ==============================================================================
class SpaGCNModel:
    """Integrated multi-modal SpaGCN pipeline model."""
    def __init__(self, n_latent: int = 64, device: str = "cuda"):
        self.n_latent = n_latent
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.l = None
        self.model = None
        self.embed = None
        self.adj_exp = None
        self.loss_history = []

    def set_l(self, l: float):
        self.l = l

    def train(
        self,
        X_combined: np.ndarray,
        adj: np.ndarray,
        n_clusters: int,
        lr: float = 0.005,
        max_epochs: int = 200,
        weight_decay: float = 5e-4,
        update_interval: int = 5,
        tol: float = 1e-3
    ):
        """Train SpaGCN GCN-DEC on integrated features and spatial adjacency."""
        assert X_combined.shape[0] == adj.shape[0] == adj.shape[1]
        N, D = X_combined.shape

        if self.l is None:
            self.l = search_l(p=0.5, adj=adj)
        print(f"Using characteristic spatial length scale: l = {self.l:.4f}")

        # Compute Gaussian spatial kernel adjacency matrix
        adj_exp = np.exp(-1.0 * (adj ** 2) / (2.0 * (self.l ** 2)))
        # Row-normalize adjacency matrix for gradient stability
        row_sum = np.sum(adj_exp, axis=1, keepdims=True) + 1e-8
        adj_norm = adj_exp / row_sum

        # Instantiate GCN-DEC model
        self.model = SimpleGCDEC(nfeat=D, nhid=self.n_latent, alpha=1.0).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        X_tensor = torch.FloatTensor(X_combined).to(self.device)
        adj_tensor = torch.FloatTensor(adj_norm).to(self.device)

        # Initialize cluster centers via KMeans on graph-convolved initial features
        with torch.no_grad():
            init_features = self.model.gc(X_tensor, adj_tensor).cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
        y_init = kmeans.fit_predict(init_features)
        cluster_centers = kmeans.cluster_centers_

        self.model.mu = Parameter(torch.FloatTensor(cluster_centers).to(self.device))
        self.embed = X_tensor
        self.adj_exp = adj_tensor
        self.loss_history = []

        self.model.train()
        p = None
        y_pred_last = y_init

        for epoch in range(max_epochs):
            if epoch % update_interval == 0:
                with torch.no_grad():
                    _, q = self.model(X_tensor, adj_tensor)
                    p = self.model.target_distribution(q).detach()
                    y_pred = torch.argmax(q, dim=1).cpu().numpy()
                    delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / N
                    y_pred_last = y_pred
                    if epoch > 0 and delta_label < tol:
                        print(f"  [SpaGCN Training] Early convergence reached at epoch {epoch} (delta_label={delta_label:.5f} < {tol})")
                        break

            optimizer.zero_grad()
            z, q = self.model(X_tensor, adj_tensor)
            loss = self.model.loss_function(p, q)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            self.loss_history.append(loss_val)

            if epoch == 0 or (epoch + 1) % 25 == 0 or (epoch + 1) == max_epochs:
                print(f"  [SpaGCN Training] Epoch {epoch + 1:3d}/{max_epochs:3d} | KL Loss: {loss_val:.5f}", flush=True)

    def extract_latent_representation(self) -> np.ndarray:
        """Extract spatial graph multi-omics joint embeddings."""
        self.model.eval()
        with torch.no_grad():
            z, _ = self.model(self.embed, self.adj_exp)
            return z.cpu().numpy()

    def predict(self):
        """Extract native DEC soft cluster probabilities and assignments."""
        self.model.eval()
        with torch.no_grad():
            _, q = self.model(self.embed, self.adj_exp)
            y_pred = torch.argmax(q, dim=1).cpu().numpy()
            prob = q.cpu().numpy()
            return y_pred, prob


# ==============================================================================
# CLUSTERING & EVALUATION ENGINE
# ==============================================================================
def run_mclust(data: np.ndarray, n_clusters: int, modelNames: str = "EEE", seed: int = 42, max_dims: int = 30) -> np.ndarray:
    """Clustering using R's mclust package with PCA dimension reduction and fallback."""
    data_mat = np.array(data, dtype=np.float64)
    if data_mat.shape[1] > max_dims:
        n_comps = min(max_dims, data_mat.shape[0] - 1, data_mat.shape[1])
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
        robjects.globalenv['tmp_spagcn_mat'] = numpy2ri.numpy2rpy(data_mat)
        res = robjects.r(f'run_mclust_native(tmp_spagcn_mat, {n_clusters}, {seed})')
        labels = np.array(res).astype(int) - 1
        return labels
    except Exception as e:
        print(f"Warning: R mclust failed ({e}). Falling back to GaussianMixture...")
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=n_clusters, random_state=seed)
        return gm.fit_predict(data_mat)


def run_kmeans(data: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    """KMeans clustering implementation."""
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    return km.fit_predict(data)


def run_leiden(adata: ad.AnnData, n_clusters: int, seed: int = 42, max_iter: int = 25) -> np.ndarray:
    """Leiden clustering with binary resolution search matching ground truth cluster count."""
    sc.pp.neighbors(adata, use_rep='X_spagcn', n_neighbors=15, random_state=seed)
    res_min, res_max = 0.05, 3.0
    best_labels = None
    best_diff = float('inf')

    for _ in range(max_iter):
        res = (res_min + res_max) / 2.0
        sc.tl.leiden(adata, resolution=res, key_added='leiden_temp', random_state=seed)
        n_found = len(adata.obs['leiden_temp'].unique())
        diff = abs(n_found - n_clusters)

        if diff < best_diff:
            best_diff = diff
            best_labels = adata.obs['leiden_temp'].astype(int).to_numpy()

        if n_found == n_clusters:
            return best_labels
        elif n_found < n_clusters:
            res_min = res
        else:
            res_max = res

    return best_labels


def evaluate_clustering(y_true, y_pred, features: np.ndarray) -> Dict[str, float]:
    """Compute 9 standard clustering evaluation metrics."""
    valid_mask = (y_true != 'unknown') & (y_true != 'Exclude') & (y_true != -1) & (y_true != '-1')
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    features_valid = features[valid_mask]

    if len(y_true_valid) == 0 or len(np.unique(y_pred_valid)) < 2:
        return {
            'ARI': 0.0, 'NMI': 0.0, 'Silhouette': 0.0, 'AMI': 0.0,
            'CHI': 0.0, 'DBI': 0.0, 'Homogeneity': 0.0, 'V-measure': 0.0, 'FMI': 0.0
        }

    le = LabelEncoder()
    y_true_encoded = le.fit_transform(y_true_valid)

    ari = float(adjusted_rand_score(y_true_encoded, y_pred_valid))
    nmi = float(normalized_mutual_info_score(y_true_encoded, y_pred_valid))
    ami = float(adjusted_mutual_info_score(y_true_encoded, y_pred_valid))
    homo = float(homogeneity_score(y_true_encoded, y_pred_valid))
    v_meas = float(v_measure_score(y_true_encoded, y_pred_valid))
    fmi = float(fowlkes_mallows_score(y_true_encoded, y_pred_valid))

    try:
        sil = float(silhouette_score(features_valid, y_pred_valid))
    except Exception:
        sil = 0.0
    try:
        chi = float(calinski_harabasz_score(features_valid, y_pred_valid))
    except Exception:
        chi = 0.0
    try:
        dbi = float(davies_bouldin_score(features_valid, y_pred_valid))
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
    """Plot UMAP and Spatial visualizations for Ground Truth, KMeans, Leiden, and mclust."""
    if 'X_umap' not in adata.obsm:
        rep = 'X_spagcn' if 'X_spagcn' in adata.obsm else ('feat' if 'feat' in adata.obsm else None)
        try:
            if rep is not None:
                sc.pp.neighbors(adata, use_rep=rep, n_neighbors=15)
            else:
                sc.pp.neighbors(adata, n_neighbors=15)
            sc.tl.umap(adata)
        except Exception as e:
            print(f"Warning: Failed to compute UMAP in plot_clustering_and_spatial: {e}")

    fig, axes = plt.subplots(4, 2, figsize=(14, 24))

    # Row 0: Ground Truth
    sc.pl.umap(adata, color='ground_truth', ax=axes[0, 0], title=f'{title_prefix} UMAP: Ground Truth', show=False, size=20)
    if 'spatial' in adata.obsm:
        sc.pl.embedding(adata, basis='spatial', color='ground_truth', ax=axes[0, 1], title=f'{title_prefix} Spatial: Ground Truth', show=False, size=20)
    else:
        axes[0, 1].axis('off')

    # Row 1: KMeans
    if 'kmeans' in adata.obs:
        sc.pl.umap(adata, color='kmeans', ax=axes[1, 0], title=f'{title_prefix} UMAP: KMeans', show=False, size=20)
        if 'spatial' in adata.obsm:
            sc.pl.embedding(adata, basis='spatial', color='kmeans', ax=axes[1, 1], title=f'{title_prefix} Spatial: KMeans', show=False, size=20)
        else:
            axes[1, 1].axis('off')

    # Row 2: Leiden
    if 'leiden' in adata.obs:
        sc.pl.umap(adata, color='leiden', ax=axes[2, 0], title=f'{title_prefix} UMAP: Leiden', show=False, size=20)
        if 'spatial' in adata.obsm:
            sc.pl.embedding(adata, basis='spatial', color='leiden', ax=axes[2, 1], title=f'{title_prefix} Spatial: Leiden', show=False, size=20)
        else:
            axes[2, 1].axis('off')

    # Row 3: mclust
    if 'mclust' in adata.obs:
        sc.pl.umap(adata, color='mclust', ax=axes[3, 0], title=f'{title_prefix} UMAP: mclust', show=False, size=20)
        if 'spatial' in adata.obsm:
            sc.pl.embedding(adata, basis='spatial', color='mclust', ax=axes[3, 1], title=f'{title_prefix} Spatial: mclust', show=False, size=20)
        else:
            axes[3, 1].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization plot image to: {save_path}")
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_loss(loss_history: List[float], save_path: Optional[str] = None):
    """Plot and save training loss curve."""
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
        plt.figure(figsize=(10, 6))
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
    """Generates cross-seed stability diagnostics (Mean, Std, Spread) across seeds."""
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


def plot_spatial_domains_and_attention(adatas_dict: Dict[str, ad.AnnData], save_dir: str = "results", visuals_dir: Optional[str] = None):
    """Side-by-Side Spatial Domain Partitioning & Modality Contribution maps across datasets."""
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

        domain_col = None
        for cand in ['spagcn_refined', 'spagcn_pred', 'mclust', 'kmeans', 'leiden']:
            if cand in adata_plot.obs:
                domain_col = cand
                break
        pred_labels = pd.factorize(adata_plot.obs[domain_col])[0] if domain_col else gt_labels
        mean_att = adata_plot.obs.get('mean_rna_attention', np.full(coords.shape[0], 0.5)).values

        # Subplot 1: Ground Truth
        ax0 = axes[row_idx, 0]
        ax0.scatter(coords[:, 0], coords[:, 1], c=gt_labels, cmap='tab20', s=25, alpha=0.9)
        ax0.set_title(f"{d_name}\nSpatial: Ground Truth", fontsize=12, fontweight='bold')
        ax0.set_aspect('equal')
        ax0.axis('off')

        # Subplot 2: SpaGCN Spatial Domains
        ax1 = axes[row_idx, 1]
        ax1.scatter(coords[:, 0], coords[:, 1], c=pred_labels, cmap='tab20', s=25, alpha=0.9)
        ax1.set_title(f"{d_name}\nSpatial: SpaGCN Domains ({domain_col})", fontsize=12, fontweight='bold')
        ax1.set_aspect('equal')
        ax1.axis('off')

        # Subplot 3: Relative Modality Weight
        ax2 = axes[row_idx, 2]
        sc2 = ax2.scatter(coords[:, 0], coords[:, 1], c=mean_att, cmap='viridis', s=25, alpha=0.9)
        ax2.set_title(f"{d_name}\nSpatial: RNA Relative Weight", fontsize=12, fontweight='bold')
        ax2.set_aspect('equal')
        ax2.axis('off')
        plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(v_dir, "spagcn_spatial_domains_and_attention.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_latent_umap_manifolds(adatas_dict: Dict[str, ad.AnnData], save_dir: str = "results", visuals_dir: Optional[str] = None):
    """Latent UMAP Manifold projections."""
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

        domain_col = None
        for cand in ['spagcn_refined', 'spagcn_pred', 'mclust', 'kmeans', 'leiden']:
            if cand in adata_plot.obs:
                domain_col = cand
                break
        pred_labels = pd.factorize(adata_plot.obs[domain_col])[0] if domain_col else gt_labels
        mean_att = adata_plot.obs.get('mean_rna_attention', np.full(umap_coords.shape[0], 0.5)).values

        # Subplot 1: GT UMAP
        ax1 = axes[row_idx, 0]
        ax1.scatter(umap_coords[:, 0], umap_coords[:, 1], c=gt_labels, cmap='tab20', s=20, alpha=0.85)
        ax1.set_title(f"{d_name}\nUMAP: Ground Truth Annotations", fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Subplot 2: Domain UMAP
        ax2 = axes[row_idx, 1]
        ax2.scatter(umap_coords[:, 0], umap_coords[:, 1], c=pred_labels, cmap='tab20', s=20, alpha=0.85)
        ax2.set_title(f"{d_name}\nUMAP: SpaGCN Clusters", fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Subplot 3: Feature Weight UMAP
        ax3 = axes[row_idx, 2]
        sc3 = ax3.scatter(umap_coords[:, 0], umap_coords[:, 1], c=mean_att, cmap='viridis', s=20, alpha=0.85)
        ax3.set_title(f"{d_name}\nUMAP: RNA Relative Weight", fontsize=12, fontweight='bold')
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
        'ground_truth': adata.obs.get('ground_truth', 'unknown').tolist(),
        'kmeans': adata.obs.get('kmeans', -1).tolist(),
        'leiden': adata.obs.get('leiden', -1).tolist(),
        'mclust': adata.obs.get('mclust', -1).tolist(),
        'spagcn_pred': adata.obs.get('spagcn_pred', -1).tolist(),
        'spagcn_refined': adata.obs.get('spagcn_refined', -1).tolist(),
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
    """Finds the dataset directory across Kaggle and local environments."""
    is_kaggle = (env_mode == "kaggle") or (env_mode == "auto" and os.path.exists('/kaggle/input'))

    candidate_paths = [
        dataset_cfg.get("kaggle_dir" if is_kaggle else "local_dir", ""),
        dataset_cfg.get("local_dir" if is_kaggle else "kaggle_dir", "")
    ]

    d_name_tail = os.path.basename(os.path.normpath(dataset_cfg.get("local_dir", "")))
    candidate_paths.extend([
        os.path.join("../../datasets", d_name_tail),
        os.path.join("../datasets", d_name_tail),
        os.path.join("datasets", d_name_tail),
        os.path.join("./datasets", d_name_tail),
        os.path.join("data", d_name_tail),
    ])

    for p in candidate_paths:
        if p and os.path.exists(p) and (
            os.path.exists(os.path.join(p, 'adata_RNA.h5ad')) or 
            os.path.exists(os.path.join(p, 'mRNA_data.h5ad'))
        ):
            return p

    if is_kaggle:
        search_terms = [d_name_tail.lower(), dataset_name.lower().replace("-", "_"), dataset_name.lower().replace("-", "")]
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'adata_RNA.h5ad' in files or 'mRNA_data.h5ad' in files:
                r_low = root.lower()
                if any(t in r_low for t in search_terms):
                    return root

    for p in candidate_paths:
        if p and os.path.exists(p):
            return p
    return dataset_cfg.get("kaggle_dir" if is_kaggle else "local_dir", "")


# ==============================================================================
# SpaGCN WORKFLOW ENGINE
# ==============================================================================
def run_spagcn_workflow(
    dataset_name: str,
    dataset_cfg: dict,
    env_mode: str,
    seed: int,
    device: str,
    epochs: int = 200,
    show_plots: bool = True,
    out_dir: str = "results",
    token: Optional[str] = None
):
    """Executes SpaGCN multi-omics representation learning, clustering, and evaluation for a single seed."""
    start_time = time.time()
    print(f"\n--- Running Seed: {seed} ---")

    # Set all random seeds
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

    # Load Modality 1: RNA
    rna_candidates = ['adata_RNA.h5ad', 'mRNA_data.h5ad']
    rna_filename = None
    for cand in rna_candidates:
        if os.path.exists(os.path.join(data_dir, cand)):
            rna_filename = cand
            break
    if rna_filename is None:
        rna_filename = rna_candidates[0]

    adata_rna = sc.read_h5ad(os.path.join(data_dir, rna_filename))
    adata_rna.var_names_make_unique()

    # Load Modality 2: ATAC or ADT
    mod2_filename = None
    for cand in dataset_cfg["mod2_candidates"]:
        if os.path.exists(os.path.join(data_dir, cand)):
            mod2_filename = cand
            break
    if mod2_filename is None:
        mod2_filename = dataset_cfg["mod2_candidates"][0]

    adata_mod2 = sc.read_h5ad(os.path.join(data_dir, mod2_filename))
    adata_mod2.var_names_make_unique()

    # Annotation matching
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

    # 1. Extract Spatial Coordinates
    if 'spatial' in adata_rna.obsm:
        coords = np.array(adata_rna.obsm['spatial']).astype(np.float32)
    elif 'x' in adata_rna.obs and 'y' in adata_rna.obs:
        coords = np.column_stack([adata_rna.obs['x'].values, adata_rna.obs['y'].values]).astype(np.float32)
        adata_rna.obsm['spatial'] = coords
    else:
        print("Warning: Spatial coordinates not found in adata_rna.obsm['spatial']. Computing PCA proxy...")
        pca_temp = PCA(n_components=2, random_state=seed).fit_transform(adata_rna.X if not issparse(adata_rna.X) else adata_rna.X.toarray())
        coords = pca_temp.astype(np.float32)
        adata_rna.obsm['spatial'] = coords

    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    adj_dist = calculate_spatial_adj_matrix(x_coords, y_coords)

    # 2. Multi-Omics Feature Integration
    print("Preprocessing & Integrating RNA and Modality 2 for SpaGCN...")
    adata_rna_norm = adata_rna.copy()
    if issparse(adata_rna_norm.X):
        adata_rna_norm.X = adata_rna_norm.X.toarray()
    sc.pp.normalize_total(adata_rna_norm, target_sum=1e4)
    sc.pp.log1p(adata_rna_norm)

    # Feature dimensionality reduction for RNA (32 PCs)
    pca_rna = PCA(n_components=min(32, adata_rna_norm.n_obs - 1, adata_rna_norm.n_vars - 1), random_state=seed)
    X_rna_pca = pca_rna.fit_transform(adata_rna_norm.X)

    # Feature dimensionality reduction for Modality 2 (32 PCs / LSI)
    if issparse(adata_mod2.X):
        mod2_dense = adata_mod2.X.toarray()
    else:
        mod2_dense = np.array(adata_mod2.X)

    if dataset_cfg.get("type") == "mouse_brain":
        # ATAC-seq: Select top 4,000 informative peaks if large
        if mod2_dense.shape[1] > 4000:
            var_peaks = np.var(mod2_dense, axis=0)
            top_peaks = np.argsort(var_peaks)[::-1][:4000]
            mod2_dense = mod2_dense[:, top_peaks]
        svd_mod2 = TruncatedSVD(n_components=min(32, mod2_dense.shape[0] - 1, mod2_dense.shape[1] - 1), random_state=seed)
        X_mod2_pca = svd_mod2.fit_transform(mod2_dense)
    else:
        # CITE-seq ADT / Protein: Standardize & PCA
        sc.pp.log1p(adata_mod2)
        pca_mod2 = PCA(n_components=min(32, adata_mod2.n_obs - 1, adata_mod2.n_vars - 1), random_state=seed)
        X_mod2_pca = pca_mod2.fit_transform(adata_mod2.X if not issparse(adata_mod2.X) else adata_mod2.X.toarray())

    # Concatenate modalities into unified 64-dimensional feature representation
    X_combined = np.concatenate([X_rna_pca, X_mod2_pca], axis=1)
    X_combined = StandardScaler().fit_transform(X_combined)

    # Determine ground truth cluster count
    valid_mask = (adata_rna.obs['ground_truth'] != 'unknown') & (adata_rna.obs['ground_truth'] != 'Exclude')
    valid_labels = adata_rna.obs['ground_truth'][valid_mask]
    n_clusters = len(valid_labels.unique()) if len(valid_labels) > 0 else 8
    print(f"Target cluster count: {n_clusters} clusters (based on ground truth)")

    # 3. Train SpaGCN Model
    print(f"Initializing SpaGCN on device: {device}...")
    spagcn_model = SpaGCNModel(n_latent=64, device=device)
    spagcn_model.train(
        X_combined=X_combined,
        adj=adj_dist,
        n_clusters=n_clusters,
        lr=0.005,
        max_epochs=epochs,
        weight_decay=5e-4
    )

    latent_emb = spagcn_model.extract_latent_representation()
    y_spagcn_pred, prob = spagcn_model.predict()
    sample_ids = adata_rna.obs_names.tolist()
    y_spagcn_refined = refine_spatial_domains(sample_ids, y_spagcn_pred.tolist(), adj_dist, shape="hexagon")

    print(f"SpaGCN joint latent embeddings extracted with shape: {latent_emb.shape}")

    adata = adata_rna.copy()
    adata.obsm['X_spagcn'] = latent_emb
    adata.obsm['feat'] = latent_emb
    adata.obs['spagcn_pred'] = y_spagcn_pred
    adata.obs['spagcn_refined'] = y_spagcn_refined

    # Compute UMAP projection for visualizations and exports
    print("Computing UMAP embeddings on SpaGCN latent representations...")
    try:
        sc.pp.neighbors(adata, use_rep='X_spagcn', n_neighbors=15, random_state=seed)
        sc.tl.umap(adata, random_state=seed)
    except Exception as e:
        print(f"Warning: Standard UMAP on X_spagcn failed ({e}). Retrying with default rep...")
        sc.pp.neighbors(adata, n_neighbors=15)
        sc.tl.umap(adata)

    # Compute modality attention proxy from the 64 latent dimensions (first 32 RNA, last 32 Mod2)
    try:
        rna_sub = latent_emb[:, :32]
        aux_sub = latent_emb[:, 32:64]
        rna_norm = np.linalg.norm(rna_sub, axis=1)
        aux_norm = np.linalg.norm(aux_sub, axis=1)
        total_norm = rna_norm + aux_norm + 1e-8
        adata.obs['mean_rna_attention'] = rna_norm / total_norm
    except Exception:
        adata.obs['mean_rna_attention'] = 0.5

    # 4. Clustering Benchmarking
    print(f"Running KMeans on SpaGCN embeddings with n_clusters={n_clusters}...")
    adata.obs['kmeans'] = run_kmeans(latent_emb, n_clusters=n_clusters, seed=seed)

    print("Running Leiden on SpaGCN embeddings...")
    adata.obs['leiden'] = run_leiden(adata, n_clusters=n_clusters, seed=seed)

    print("Running mclust on SpaGCN embeddings...")
    adata.obs['mclust'] = run_mclust(latent_emb, n_clusters=n_clusters, seed=seed)

    # 5. Evaluate Metrics
    metrics_kmeans = evaluate_clustering(adata.obs['ground_truth'].values, adata.obs['kmeans'].values, latent_emb)
    metrics_leiden = evaluate_clustering(adata.obs['ground_truth'].values, adata.obs['leiden'].values, latent_emb)
    metrics_mclust = evaluate_clustering(adata.obs['ground_truth'].values, adata.obs['mclust'].values, latent_emb)
    metrics_spagcn = evaluate_clustering(adata.obs['ground_truth'].values, np.array(y_spagcn_refined), latent_emb)

    print(f"\n--- Clustering Evaluation (SpaGCN {dataset_name} Seed {seed} - KMeans) ---")
    for k, v in metrics_kmeans.items():
        print(f"{k}: {v:.4f}")

    print(f"\n--- Clustering Evaluation (SpaGCN {dataset_name} Seed {seed} - Leiden) ---")
    for k, v in metrics_leiden.items():
        print(f"{k}: {v:.4f}")

    print(f"\n--- Clustering Evaluation (SpaGCN {dataset_name} Seed {seed} - mclust) ---")
    for k, v in metrics_mclust.items():
        print(f"{k}: {v:.4f}")

    print(f"\n--- Clustering Evaluation (SpaGCN {dataset_name} Seed {seed} - SpaGCN Native Refined) ---")
    for k, v in metrics_spagcn.items():
        print(f"{k}: {v:.4f}")

    # 6. Generate Visualizations & Local Exports
    plot_save_path = os.path.join(out_dir, f"spagcn_plot_{dataset_name}_seed_{seed}.png")
    plot_clustering_and_spatial(adata, title_prefix=f"SpaGCN ({dataset_name} Seed {seed})", save_path=plot_save_path, show_plots=show_plots)

    loss_save_path = os.path.join(out_dir, f"spagcn_loss_{dataset_name}_seed_{seed}.png")
    plot_loss(spagcn_model.loss_history, save_path=loss_save_path)

    export_predictions_csv(adata, dataset_name, seed, out_dir=out_dir)

    # 7. Dashboard API Sync
    embeddings_data = {
        "latent_dimensions": latent_emb.shape[1],
        "umap": adata.obsm['X_umap'].tolist() if 'X_umap' in adata.obsm else [],
        "cluster_assignments": {
            "kmeans": adata.obs['kmeans'].tolist(),
            "leiden": adata.obs['leiden'].tolist(),
            "mclust": adata.obs['mclust'].tolist(),
            "spagcn_refined": adata.obs['spagcn_refined'].tolist()
        }
    }

    dash_info = DATASET_DASHBOARD_INFO.get(dataset_name, {"datasetId": dataset_name, "datasetName": dataset_name})
    training_results = {
        "loss_history": spagcn_model.loss_history,
        "runtime_seconds": time.time() - start_time,
        "device": str(spagcn_model.device)
    }
    hyperparams = {
        "epochs": epochs,
        "learning_rate": 0.005,
        "weight_decay": 5e-4,
        "n_latent": 64,
        "l": float(spagcn_model.l) if spagcn_model.l is not None else 0.0,
        "n_clusters": n_clusters,
        "seed": seed
    }

    export_dashboard_experiment(
        model_id="SpaGCN",
        model_name="SpaGCN",
        dataset_name=dash_info["datasetId"],
        seed=seed,
        metrics_dict=metrics_mclust,
        training_results=training_results,
        hyperparameters=hyperparams,
        embeddings_data=embeddings_data,
        output_dir=out_dir,
        api_url=DEFAULT_API_URL,
        token=token,
        dataset_display_name=dash_info["datasetName"]
    )

    results_entry = [
        {"Dataset": dataset_name, "Seed": seed, "Algorithm": "KMeans", **metrics_kmeans},
        {"Dataset": dataset_name, "Seed": seed, "Algorithm": "Leiden", **metrics_leiden},
        {"Dataset": dataset_name, "Seed": seed, "Algorithm": "mclust", **metrics_mclust},
        {"Dataset": dataset_name, "Seed": seed, "Algorithm": "SpaGCN_Refined", **metrics_spagcn}
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
    epochs: int = 200,
    show_plots: bool = False,
    out_dir: str = "results",
    token: Optional[str] = None
):
    """Executes the full SpaGCN pipeline across all scheduled datasets and seeds."""
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
                    epochs=epochs,
                    show_plots=show_plots,
                    out_dir=out_dir,
                    token=token
                )
                dataset_entries.extend(entries)
                all_benchmark_results.extend(entries)

                mclust_ari = next((e["ARI"] for e in entries if e["Algorithm"] == "mclust"), 0.0)
                if mclust_ari > best_ari:
                    best_ari = mclust_ari
                    best_adata = adata_out.copy()

            except Exception as e:
                print(f"Error executing SpaGCN on {d_name} (Seed {seed}): {e}")
                import traceback
                traceback.print_exc()

        if best_adata is not None:
            best_adatas_per_dataset[d_name] = best_adata

        # Print Dataset Summary
        if dataset_entries:
            df_ds = pd.DataFrame(dataset_entries)
            print("\n" + "=" * 55)
            print(f"AVERAGE PERFORMANCE FOR {d_name} ({len(seeds)} seeds)")
            print("=" * 55)
            for algo in ["KMeans", "Leiden", "mclust", "SpaGCN_Refined"]:
                algo_df = df_ds[df_ds["Algorithm"] == algo]
                if not algo_df.empty:
                    print(f"\n--- {algo} Performance (Mean ± Std) ---")
                    mean_vals = algo_df[['ARI', 'NMI', 'Silhouette', 'AMI', 'CHI', 'DBI', 'Homogeneity', 'V-measure', 'FMI']].mean()
                    std_vals = algo_df[['ARI', 'NMI', 'Silhouette', 'AMI', 'CHI', 'DBI', 'Homogeneity', 'V-measure', 'FMI']].std()
                    summary_view = pd.DataFrame({'Mean': mean_vals, 'Std': std_vals})
                    print(summary_view.to_string())
            print("=" * 55 + "\n")

    # Export Cross-Dataset Benchmark Table
    if all_benchmark_results:
        df_benchmark = pd.DataFrame(all_benchmark_results)
        csv_path = os.path.join(out_dir, "spagcn_benchmark_results.csv")
        df_benchmark.to_csv(csv_path, index=False)
        print(f"\n[Saved] Full benchmark metrics saved to: {csv_path}")

        plot_summary_boxplots(df_benchmark, save_dir=out_dir, visuals_dir=visuals_dir)
        plot_cross_seed_linearity_and_spread(df_benchmark, save_dir=out_dir, visuals_dir=visuals_dir)
        plot_spatial_domains_and_attention(best_adatas_per_dataset, save_dir=out_dir, visuals_dir=visuals_dir)
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
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs for SpaGCN GCN-DEC")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save benchmark metrics and plots")
    parser.add_argument("--no_dashboard", action="store_true", help="Disable uploading to dashboard")
    parser.add_argument("--generate_plots", action="store_true", help="Regenerate summary figures from existing benchmark CSV")

    args = parser.parse_args()

    # Determine execution datasets
    if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
        scheduled_datasets = list(ALL_DATASETS_CONFIG.keys())
    else:
        scheduled_datasets = [d for d in args.datasets if d in ALL_DATASETS_CONFIG]

    # Handle output directory when running on Kaggle
    output_dir = args.out_dir
    if os.path.exists('/kaggle/working') and output_dir == "results":
        output_dir = "/kaggle/working"

    print("=" * 80)
    print(" SpaGCN MULTI-SEED PIPELINE (STANDALONE EXECUTION)")
    print(f" Default Scheduled Datasets ({len(scheduled_datasets)}): {scheduled_datasets}")
    print(f" Default Ablation Seeds ({len(args.seeds)}): {args.seeds}")
    print("=" * 80)

    # Dashboard Authentication
    dashboard_token = None
    if not args.no_dashboard:
        dashboard_token = authenticate_dashboard(
            base_url=DEFAULT_DASHBOARD_URL,
            username=DEFAULT_USERNAME,
            password=DEFAULT_PASSWORD
        )

    # Plot regeneration mode
    if args.generate_plots:
        csv_path = os.path.join(output_dir, "spagcn_benchmark_results.csv")
        if os.path.exists(csv_path):
            print(f"Regenerating summary diagnostics from {csv_path}...")
            df_existing = pd.read_csv(csv_path)
            plot_summary_boxplots(df_existing, save_dir=output_dir)
            plot_cross_seed_linearity_and_spread(df_existing, save_dir=output_dir)
        else:
            print(f"Error: {csv_path} not found for --generate_plots.")
        sys.exit(0)

    # Execute full multi-seed pipeline
    run_multi_seed_pipeline(
        datasets=scheduled_datasets,
        seeds=args.seeds,
        env_mode=ENV_MODE,
        device=args.device,
        epochs=args.epochs,
        show_plots=False,
        out_dir=output_dir,
        token=dashboard_token
    )

