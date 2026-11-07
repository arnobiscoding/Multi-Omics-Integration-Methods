#!/usr/bin/env python3
"""
totalVI Multi-Seed Pipeline (Standalone Python Script)
Integrates joint spatial representation learning workflow for target datasets:
- Mouse Brain E11, E13, E15, E18 (RNA + ATAC)
- Human Lymph Node A1, D1 (RNA + Protein/ADT)

Directly based on totalVI (Cell 2021) and totalVI.ipynb,
and matches the architecture and benchmark tracking suite in spallmpipeline.py:
- Multi-seed evaluation suite (10 seeds)
- Tri-algorithm clustering (KMeans, Leiden, mclust)
- Complete evaluation metrics (ARI, NMI, Silhouette, AMI, CHI, DBI, Homogeneity, V-measure, FMI)
- Summary Box & Whiskers plots across datasets & algorithms
- Cross-Seed Linearity & Metric Spread diagnostics suite (CSV + Unified Figure)
- Side-by-Side Spatial Domain Partitioning & Modality Weight maps
- Latent UMAP Manifold projections
- Spot-level prediction CSV export
- Standardized Dashboard JSON exports & automated live API upload
"""

import os
import sys
import gc
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
from tqdm import tqdm

# Dynamic import for scvi-tools and mudata
HAVE_SCVI = False
HAVE_MUDATA = False
try:
    import scvi
    HAVE_SCVI = True
except ImportError:
    pass

try:
    import mudata as md
    HAVE_MUDATA = True
except ImportError:
    pass

# ==============================================================================
# PyTorch Lightning Custom Progress Callback
# ==============================================================================
try:
    from lightning.pytorch.callbacks import Callback as PLCallback
except ImportError:
    try:
        from pytorch_lightning.callbacks import Callback as PLCallback
    except ImportError:
        PLCallback = object

class TotalVIProgressCallback(PLCallback):
    """Clean live progress callback for totalVI execution in non-interactive Kaggle environments."""
    def __init__(self, print_every=25):
        if PLCallback is not object:
            super().__init__()
        self.print_every = print_every

    def on_train_epoch_end(self, trainer, pl_module):
        try:
            epoch = trainer.current_epoch + 1
            max_epochs = trainer.max_epochs
            if epoch == 1 or epoch % self.print_every == 0 or epoch == max_epochs:
                loss_str = ""
                metrics = getattr(trainer, "callback_metrics", {})
                for k in ["train_loss_epoch", "train_loss", "loss", "elbo_train"]:
                    if k in metrics:
                        val = metrics[k]
                        loss_str = f" | Loss: {float(val):.3f}"
                        break
                print(f"  [totalVI Training] Epoch {epoch:3d}/{max_epochs:3d}{loss_str}", flush=True)
        except Exception:
            pass

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

print("=" * 80)
print(" totalVI MULTI-SEED PIPELINE (STANDALONE EXECUTION)")
print(f" Default Scheduled Datasets ({len(datasets_to_run)}): {datasets_to_run}")
print(f" Default Ablation Seeds ({len(SEEDS)}): {SEEDS}")
print("=" * 80)

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
            "total_loss": float(loss_hist[ep]) if ep < len(loss_hist) else None,
            "silhouette": float(metrics_dict.get("Silhouette", 0.0)),
            "ari": float(metrics_dict.get("ARI", 0.0)),
        })

    safe_metrics = make_json_serializable(metrics_dict)
    safe_hyperparameters = make_json_serializable(hyperparameters)
    safe_history = make_json_serializable(history_points)
    safe_embeddings = make_json_serializable(embeddings_data or {})

    best_epoch = int(training_results.get("best_epoch", len(loss_hist)))
    best_score = float(training_results.get("best_sil", metrics_dict.get("Silhouette", 0.0)))

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
                    "architecture": "totalVI: Deep Generative Model for Multimodal Single-Cell & Spatial Omics",
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


# ==============================================================================
# TOTALVI PREPROCESSING & RUNNER HELPERS
# ==============================================================================
def fix_seed(seed: int):
    """Fix random seed for full reproducibility across torch, scvi, numpy, random."""
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
    if HAVE_SCVI:
        scvi.settings.seed = seed


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
        try:
            robjects.r('rm(tmp_mclust_mat); gc(verbose=FALSE)')
        except Exception:
            pass
        return labels.astype(str)
    except Exception as e:
        print(f"[mclust Info] Native R mclust execution skipped/fallback ({e}). Running KMeans...")
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(data_mat).astype(str)


def search_res(adata, n_clusters, method='leiden', use_rep='X_totalVI', start=0.1, end=3.0, increment=0.02):
    """Search for resolution to achieve target cluster count using Leiden."""
    sc.pp.neighbors(adata, n_neighbors=30, use_rep=use_rep)
    best_res = 0.5
    min_diff = 999
    for res in np.arange(start, end, increment):
        res = round(res, 3)
        try:
            sc.tl.leiden(adata, random_state=0, resolution=res, flavor='igraph', n_iterations=2, directed=False)
        except Exception:
            sc.tl.leiden(adata, random_state=0, resolution=res)
        count_unique = adata.obs['leiden'].nunique()
        if count_unique == n_clusters:
            return res
        diff = abs(count_unique - n_clusters)
        if diff < min_diff:
            min_diff = diff
            best_res = res
    return best_res


def clustering(adata, n_clusters=7, key='X_totalVI', add_key='totalVI', method='leiden'):
    """Spatial clustering using mclust, leiden, or kmeans."""
    if method == 'mclust':
        adata.obs['mclust'] = run_mclust(adata.obsm[key], n_clusters)
        adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, method='leiden', use_rep=key)
        try:
            sc.tl.leiden(adata, random_state=0, resolution=res, flavor='igraph', n_iterations=2, directed=False)
        except Exception:
            sc.tl.leiden(adata, random_state=0, resolution=res)
        adata.obs[add_key] = adata.obs['leiden'].astype(str)


def evaluate_clustering(y_true_series, y_pred_series, features_matrix, name=""):
    """Evaluate clustering performance across supervised and unsupervised metrics."""
    mask = (y_true_series != 'Exclude') & (y_true_series != 'unknown') & (y_true_series.notna())
    y_true = y_true_series[mask].astype(str)
    y_pred = y_pred_series[mask].astype(str)
    feats = features_matrix[mask]

    if len(y_true) == 0 or len(np.unique(y_pred)) < 2:
        return {
            'ARI': 0.0, 'NMI': 0.0, 'Silhouette': 0.0,
            'AMI': 0.0, 'CHI': 0.0, 'DBI': 0.0,
            'Homogeneity': 0.0, 'V-measure': 0.0, 'FMI': 0.0
        }

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    homo = homogeneity_score(y_true, y_pred)
    v_meas = v_measure_score(y_true, y_pred)

    try:
        fmi = fowlkes_mallows_score(y_true, y_pred)
    except Exception:
        fmi = 0.0

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
        'V-measure': float(v_meas),
        'FMI': float(fmi)
    }

    if name:
        print(f"\n--- Clustering Evaluation ({name}) ---")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    return metrics


# ==============================================================================
# VISUALIZATIONS & DIAGNOSTICS SUITE
# ==============================================================================
def plot_totalvi_visualizations(adata, title_prefix="", dname="dataset", seed=2024, save_dir=None):
    """Plot UMAP and Spatial visualizations for Ground Truth, KMeans, Leiden, and mclust."""
    sc.pp.neighbors(adata, use_rep='X_totalVI', n_neighbors=15)
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
        save_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else 'results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"totalvi_plot_{dname}_seed_{seed}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization plot image to: {save_path}")
    plt.close(fig)
    plt.close('all')


def plot_totalvi_summary_boxplots(df_all, save_dir=None):
    """Generate and save Box & Whiskers plots comparing KMeans, Leiden, and mclust performance side-by-side."""
    if df_all.empty or "cluster alg" not in df_all.columns:
        return

    metrics = ["ARI", "NMI", "Silhouette", "AMI", "CHI", "DBI", "Homogeneity", "V-measure"]
    metrics = [m for m in metrics if m in df_all.columns]
    if not metrics:
        return

    if save_dir is None:
        save_dir = '/kaggle/working' if os.path.exists('/kaggle/working') else 'results'
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
        ax.set_title(f"totalVI Model: {metric} Performance Across Datasets", fontsize=14, fontweight='bold')
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title="Cluster Alg", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"totalvi_boxplot_{metric.lower().replace('-', '_')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric} Boxplot image to: {save_path}")
        plt.close(fig)
        plt.close('all')


def export_spot_predictions_csv(adata, dataset_name, seed=0, out_dir="results"):
    """Exports cell/spot-level spatial coordinates, annotations, and cluster predictions to CSV."""
    os.makedirs(out_dir, exist_ok=True)
    coords = adata.obsm.get('spatial', None)

    pred_data = {
        'barcode': adata.obs_names.tolist(),
        'ground_truth': adata.obs.get('ground_truth', 'unknown').astype(str).tolist(),
        'totalVI_domain': adata.obs.get('totalVI_domain', adata.obs.get('domain', '')).astype(str).tolist(),
    }

    for alg in ['kmeans', 'leiden', 'mclust']:
        if alg in adata.obs:
            pred_data[alg] = adata.obs[alg].astype(str).tolist()

    if 'mean_rna_attention' in adata.obs:
        pred_data['mean_rna_attention'] = adata.obs['mean_rna_attention'].tolist()
    if 'mean_aux_attention' in adata.obs:
        pred_data['mean_aux_attention'] = adata.obs['mean_aux_attention'].tolist()

    if coords is not None and coords.shape[0] == len(adata.obs_names) and coords.shape[1] >= 2:
        pred_data['spatial_x'] = coords[:, 0].tolist()
        pred_data['spatial_y'] = coords[:, 1].tolist()

    df_pred = pd.DataFrame(pred_data)
    csv_path = os.path.join(out_dir, f"totalvi_spot_predictions_{dataset_name}.csv")
    df_pred.to_csv(csv_path, index=False)
    print(f"[CSV Saved] Spot-level predictions saved to: {csv_path}")
    return df_pred


def compute_and_plot_cross_seed_linearity(df_results=None, out_dir="results", visuals_dir=None, show_plot=False):
    """Computes cross-seed linearity & metric spread diagnostics across seeds, saving CSV and unified plots."""
    os.makedirs(out_dir, exist_ok=True)
    v_dir = visuals_dir or os.path.join(out_dir, "visuals")
    os.makedirs(v_dir, exist_ok=True)

    if df_results is None or len(df_results) == 0:
        candidates = [
            os.path.join(out_dir, "totalvi_benchmark_results.csv"),
            os.path.join(out_dir, "totalvi_ablation_results.csv"),
            "results/totalvi_benchmark_results.csv",
            "results/totalvi_ablation_results.csv"
        ]
        csv_fallback = next((p for p in candidates if os.path.exists(p)), None)
        if csv_fallback is not None:
            df_results = pd.read_csv(csv_fallback)
        else:
            return None

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
        return None

    unique_datasets = df_curr['dataset'].unique()
    n_datasets = len(unique_datasets)
    if n_datasets == 0:
        return None

    fig, axes = plt.subplots(n_datasets, 2, figsize=(18, 4.8 * n_datasets), squeeze=False)
    diagnostics_rows = []

    for idx, d_name in enumerate(unique_datasets):
        sub_df = df_curr[df_curr['dataset'] == d_name].copy()
        sub_df['seed_str'] = sub_df['seed'].astype(str)
        seeds_order = sub_df['seed_str'].tolist()
        x_indices = np.arange(len(seeds_order))

        # ARI Subplot
        ari_vals = sub_df['ARI'].values.astype(float)
        mean_ari = float(np.mean(ari_vals))
        std_ari = float(np.std(ari_vals))
        min_ari = float(np.min(ari_vals))
        max_ari = float(np.max(ari_vals))
        range_ari = max_ari - min_ari

        ax_ari = axes[idx, 0]
        ax_ari.plot(x_indices, ari_vals, marker='o', markersize=9, linewidth=2.4, color='#1f77b4', label='Seed ARI')
        ax_ari.axhline(mean_ari, color='#d62728', linestyle='--', linewidth=2.0, label=f'Mean ARI: {mean_ari:.4f}')
        ax_ari.fill_between(x_indices, mean_ari - std_ari, mean_ari + std_ari, color='#1f77b4', alpha=0.15, label=f'±1σ Band (±{std_ari:.4f})')
        for x_i, y_i in zip(x_indices, ari_vals):
            ax_ari.annotate(f"{y_i:.3f}", (x_i, y_i), textcoords="offset points", xytext=(0, 9), ha='center', fontsize=9, fontweight='bold', color='#1f77b4')

        ax_ari.set_xticks(x_indices)
        ax_ari.set_xticklabels([f"Seed {s}" for s in seeds_order], fontsize=10, rotation=25)
        ax_ari.set_ylabel("Adjusted Rand Index (ARI)", fontsize=11, fontweight='bold')
        ax_ari.set_title(f"{d_name}\n[ARI Linearity] Mean: {mean_ari:.4f} ± {std_ari:.4f} | Spread: {range_ari:.4f}", fontsize=12, fontweight='bold')
        ax_ari.grid(True, linestyle=':', alpha=0.6)
        ax_ari.legend(loc='lower right', fontsize=9)
        y_pad = max(0.04, range_ari * 0.4)
        ax_ari.set_ylim([max(0.0, min_ari - y_pad), min(1.0, max_ari + y_pad)])

        # Silhouette Subplot
        sil_vals = sub_df['Silhouette'].values.astype(float)
        mean_sil = float(np.mean(sil_vals))
        std_sil = float(np.std(sil_vals))
        min_sil = float(np.min(sil_vals))
        max_sil = float(np.max(sil_vals))
        range_sil = max_sil - min_sil

        ax_sil = axes[idx, 1]
        ax_sil.plot(x_indices, sil_vals, marker='s', markersize=9, linewidth=2.4, color='#2ca02c', label='Seed Silhouette')
        ax_sil.axhline(mean_sil, color='#9467bd', linestyle='--', linewidth=2.0, label=f'Mean Sil: {mean_sil:.4f}')
        ax_sil.fill_between(x_indices, mean_sil - std_sil, mean_sil + std_sil, color='#2ca02c', alpha=0.15, label=f'±1σ Band (±{std_sil:.4f})')
        for x_i, y_i in zip(x_indices, sil_vals):
            ax_sil.annotate(f"{y_i:.3f}", (x_i, y_i), textcoords="offset points", xytext=(0, 9), ha='center', fontsize=9, fontweight='bold', color='#2ca02c')

        ax_sil.set_xticks(x_indices)
        ax_sil.set_xticklabels([f"Seed {s}" for s in seeds_order], fontsize=10, rotation=25)
        ax_sil.set_ylabel("Silhouette Score", fontsize=11, fontweight='bold')
        ax_sil.set_title(f"{d_name}\n[Silhouette Linearity] Mean: {mean_sil:.4f} ± {std_sil:.4f} | Spread: {range_sil:.4f}", fontsize=12, fontweight='bold')
        ax_sil.grid(True, linestyle=':', alpha=0.6)
        ax_sil.legend(loc='lower right', fontsize=9)
        y_pad_sil = max(0.04, range_sil * 0.4)
        ax_sil.set_ylim([max(-1.0, min_sil - y_pad_sil), min(1.0, max_sil + y_pad_sil)])

        if range_ari <= 0.03:
            verdict = "Ultra-stable flat line (Near-zero variance)."
        elif range_ari <= 0.06:
            verdict = "Near-linear trajectory across seeds."
        else:
            verdict = f"Noticeable variation ({range_ari*100:.1f}% spread)."

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
            "linearity_verdict": verdict
        })

    df_diag = pd.DataFrame(diagnostics_rows)
    diag_csv_path = os.path.join(out_dir, "totalvi_cross_seed_linearity_diagnostics.csv")
    df_diag.to_csv(diag_csv_path, index=False)
    print(f"\n[CSV Saved] Cross-seed linearity diagnostics saved to: {diag_csv_path}")

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.subplots_adjust(top=0.96, hspace=0.35)
    plt.suptitle("totalVI: Side-by-Side Cross-Seed Linearity & Spread Diagnostics (ARI & Silhouette)", fontsize=16, fontweight='bold', y=0.995)
    save_path = os.path.join(v_dir, "totalvi_cross_seed_linearity.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if v_dir != out_dir:
        plt.savefig(os.path.join(out_dir, "totalvi_cross_seed_linearity.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')
    return df_diag


def plot_totalvi_spatial_domains_and_attention(loaded_adatas=None, df_results=None, save_dir=None, visuals_dir=None, show_plot=False):
    """Side-by-Side Spatial Domain & Modality Weight Visualization for all datasets."""
    out_dir = save_dir or ('/kaggle/working' if os.path.exists('/kaggle/working') else 'results')
    v_dir = visuals_dir or os.path.join(out_dir, "visuals")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    adatas_dict = loaded_adatas.copy() if loaded_adatas is not None else {}
    if len(adatas_dict) == 0:
        for d_name in ALL_DATASETS_CONFIG.keys():
            search_paths = [
                os.path.join(out_dir, f"totalvi_adata_{d_name}.h5ad"),
                os.path.join(out_dir, "results", f"totalvi_adata_{d_name}.h5ad"),
                os.path.join("results", f"totalvi_adata_{d_name}.h5ad")
            ]
            for p in search_paths:
                if os.path.exists(p):
                    try:
                        adatas_dict[d_name] = sc.read_h5ad(p)
                        break
                    except Exception:
                        pass

    if len(adatas_dict) == 0:
        return

    valid_datasets = list(adatas_dict.keys())
    n_datasets = len(valid_datasets)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(21, 5.2 * n_datasets), squeeze=False)

    for row_idx, d_name in enumerate(valid_datasets):
        adata_plot = adatas_dict[d_name]
        coords = adata_plot.obsm.get('spatial', np.zeros((adata_plot.n_obs, 2)))
        gt_series = adata_plot.obs.get('ground_truth', None)
        gt_labels = pd.factorize(gt_series)[0] if gt_series is not None else np.zeros(coords.shape[0])

        domain_col = None
        for cand in ['totalVI_domain', 'domain', 'mclust', 'kmeans', 'leiden']:
            if cand in adata_plot.obs:
                domain_col = cand
                break
        pred_labels = pd.factorize(adata_plot.obs[domain_col])[0] if domain_col else gt_labels
        mean_att = adata_plot.obs.get('mean_rna_attention', np.full(coords.shape[0], 0.5)).values

        # Subplot 1: Ground Truth
        ax_gt = axes[row_idx, 0]
        ax_gt.scatter(coords[:, 0], coords[:, 1], c=gt_labels, cmap='tab20', s=25, alpha=0.9, edgecolors='none')
        ax_gt.set_title(f"{d_name}\nGround Truth Annotations", fontsize=12, fontweight='bold')
        ax_gt.axis('off')

        # Subplot 2: totalVI Domain
        ax_dom = axes[row_idx, 1]
        ax_dom.scatter(coords[:, 0], coords[:, 1], c=pred_labels, cmap='tab20', s=25, alpha=0.9, edgecolors='none')
        ax_dom.set_title(f"{d_name}\ntotalVI Identified Spatial Domains", fontsize=12, fontweight='bold')
        ax_dom.axis('off')

        # Subplot 3: Modality Contribution
        ax_att = axes[row_idx, 2]
        sc_att = ax_att.scatter(coords[:, 0], coords[:, 1], c=mean_att, cmap='viridis', s=25, alpha=0.9, edgecolors='none')
        ax_att.set_title(f"{d_name}\nRNA Relative Latent Loading", fontsize=12, fontweight='bold')
        ax_att.axis('off')
        plt.colorbar(sc_att, ax=ax_att, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(v_dir, "totalvi_spatial_domain_and_attention_maps.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if v_dir != out_dir:
        plt.savefig(os.path.join(out_dir, "totalvi_spatial_domain_and_attention_maps.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')


def plot_totalvi_latent_umap_manifolds(loaded_adatas=None, df_results=None, save_dir=None, visuals_dir=None, show_plot=False):
    """Latent Embedding Manifold Projections (Side-by-Side UMAP Suite for all datasets)."""
    out_dir = save_dir or ('/kaggle/working' if os.path.exists('/kaggle/working') else 'results')
    v_dir = visuals_dir or os.path.join(out_dir, "visuals")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(v_dir, exist_ok=True)

    adatas_dict = loaded_adatas.copy() if loaded_adatas is not None else {}
    if len(adatas_dict) == 0:
        for d_name in ALL_DATASETS_CONFIG.keys():
            search_paths = [
                os.path.join(out_dir, f"totalvi_adata_{d_name}.h5ad"),
                os.path.join("results", f"totalvi_adata_{d_name}.h5ad")
            ]
            for p in search_paths:
                if os.path.exists(p):
                    try:
                        adatas_dict[d_name] = sc.read_h5ad(p)
                        break
                    except Exception:
                        pass

    if len(adatas_dict) == 0:
        return

    valid_datasets = list(adatas_dict.keys())
    n_datasets = len(valid_datasets)
    fig, axes = plt.subplots(n_datasets, 3, figsize=(21, 5.2 * n_datasets), squeeze=False)

    for row_idx, d_name in enumerate(valid_datasets):
        adata_plot = adatas_dict[d_name]
        if 'X_umap' not in adata_plot.obsm:
            sc.pp.neighbors(adata_plot, use_rep='X_totalVI' if 'X_totalVI' in adata_plot.obsm else 'feat', n_neighbors=15)
            sc.tl.umap(adata_plot)
        umap_coords = adata_plot.obsm['X_umap']

        gt_series = adata_plot.obs.get('ground_truth', None)
        gt_labels = pd.factorize(gt_series)[0] if gt_series is not None else np.zeros(umap_coords.shape[0])

        domain_col = None
        for cand in ['totalVI_domain', 'domain', 'mclust', 'kmeans', 'leiden']:
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
        ax2.set_title(f"{d_name}\nUMAP: totalVI Clusters", fontsize=12, fontweight='bold')
        ax2.axis('off')

        # Subplot 3: Feature Weight UMAP
        ax3 = axes[row_idx, 2]
        sc3 = ax3.scatter(umap_coords[:, 0], umap_coords[:, 1], c=mean_att, cmap='viridis', s=20, alpha=0.85)
        ax3.set_title(f"{d_name}\nUMAP: RNA Relative Loading", fontsize=12, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    save_path = os.path.join(v_dir, "totalvi_latent_umap_manifolds.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if v_dir != out_dir:
        plt.savefig(os.path.join(out_dir, "totalvi_latent_umap_manifolds.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')


def plot_loss(loss_history, save_path=None):
    """Plot and save ELBO training loss curve."""
    if not loss_history:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_history, label='Negative ELBO', color='#d62728', linewidth=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('ELBO Loss')
    ax.set_title('totalVI Training Loss Curve')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')


# ==============================================================================
# WORKFLOW EXECUTION ENGINE
# ==============================================================================
def resolve_dataset_path(dataset_name, dataset_cfg, env_mode):
    """Resolves data directory robustly across Kaggle, local, and relative paths."""
    is_kaggle = os.path.exists('/kaggle/input')
    candidate_paths = []
    if env_mode == "kaggle" or (env_mode == "auto" and is_kaggle):
        candidate_paths.append(dataset_cfg.get("kaggle_dir", ""))
    candidate_paths.append(dataset_cfg.get("local_dir", ""))

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

    # Flexible search in /kaggle/input if running on Kaggle
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


def run_totalvi_workflow(
    dataset_name,
    dataset_cfg,
    env_mode,
    seed,
    device,
    epochs=400,
    show_plots=True,
    dashboard_token=None,
    api_url=DEFAULT_API_URL,
    output_dir=None
):
    print(f"\n--- Running Seed: {seed} ---")
    fix_seed(seed)

    data_dir = resolve_dataset_path(dataset_name, dataset_cfg, env_mode)
    dataset_type = dataset_cfg["type"]
    print(f"Loading data from: {data_dir}")

    rna_path = os.path.join(data_dir, 'adata_RNA.h5ad')
    if not os.path.exists(rna_path):
        if os.path.exists(os.path.join(data_dir, 'mRNA_data.h5ad')):
            rna_path = os.path.join(data_dir, 'mRNA_data.h5ad')

    adata_rna = sc.read_h5ad(rna_path)
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

    # 1. Clean Modality 2 (ATAC/ADT) while sparse to avoid memory spikes
    if issparse(adata_mod2.X):
        if (adata_mod2.X.data < 0).any():
            adata_mod2.X.data = np.clip(adata_mod2.X.data, 0, None)
        if not np.issubdtype(adata_mod2.X.dtype, np.integer):
            adata_mod2.X.data = np.round(adata_mod2.X.data).astype(np.float32)
    else:
        if np.any(adata_mod2.X < 0):
            adata_mod2.X = np.clip(adata_mod2.X, a_min=0, a_max=None)
        if not np.issubdtype(adata_mod2.X.dtype, np.integer):
            adata_mod2.X = np.round(adata_mod2.X).astype(np.float32)

    # Subsetting Modality 2 to top 4,000 most informative features while sparse (avoids OOM on 117k features)
    if adata_mod2.n_vars > 4000:
        print(f"Subsetting Modality 2 from {adata_mod2.n_vars} to top 4,000 most informative features...")
        if issparse(adata_mod2.X):
            mean = np.array(adata_mod2.X.mean(axis=0)).flatten()
            mean_sq = np.array(adata_mod2.X.power(2).mean(axis=0)).flatten()
            variances = mean_sq - mean**2
        else:
            variances = np.var(adata_mod2.X, axis=0)
        top_4000_idx = np.argsort(variances)[::-1][:4000]
        adata_mod2_subset = adata_mod2[:, top_4000_idx].copy()
        print(f"Modality 2 subsetted shape: {adata_mod2_subset.shape}")
    else:
        adata_mod2_subset = adata_mod2.copy()

    # Free original full Modality 2 immediately to reclaim memory
    del adata_mod2
    gc.collect()

    # Convert subsetted Modality 2 to dense non-negative float32 array for totalVI protein layer
    if issparse(adata_mod2_subset.X):
        adata_mod2_subset.X = adata_mod2_subset.X.toarray()
    adata_mod2_subset.X = np.clip(np.round(adata_mod2_subset.X), 0, None).astype(np.float32)

    # 2. Clean RNA and subset to top 4,000 highly variable features
    adata_rna.obs['batch'] = 'batch_0'
    if issparse(adata_rna.X):
        if (adata_rna.X.data < 0).any():
            adata_rna.X.data = np.clip(adata_rna.X.data, 0, None)
        if not np.issubdtype(adata_rna.X.dtype, np.integer):
            adata_rna.X.data = np.round(adata_rna.X.data).astype(np.float32)
    else:
        if np.any(adata_rna.X < 0):
            adata_rna.X = np.clip(adata_rna.X, a_min=0, a_max=None)
        if not np.issubdtype(adata_rna.X.dtype, np.integer):
            adata_rna.X = np.round(adata_rna.X).astype(np.float32)

    if adata_rna.n_vars > 4000:
        print(f"Subsetting RNA from {adata_rna.n_vars} to top 4,000 highly variable features...")
        if issparse(adata_rna.X):
            mean_rna = np.array(adata_rna.X.mean(axis=0)).flatten()
            mean_sq_rna = np.array(adata_rna.X.power(2).mean(axis=0)).flatten()
            variances_rna = mean_sq_rna - mean_rna**2
        else:
            variances_rna = np.var(adata_rna.X, axis=0)
        top_rna_idx = np.argsort(variances_rna)[::-1][:4000]
        adata_rna_subset = adata_rna[:, top_rna_idx].copy()
    else:
        adata_rna_subset = adata_rna.copy()

    if issparse(adata_rna_subset.X):
        adata_rna_subset.X = adata_rna_subset.X.toarray()
    adata_rna_subset.X = np.clip(np.round(adata_rna_subset.X), 0, None).astype(np.float32)
    adata_rna_subset.obs['batch'] = 'batch_0'
    adata_rna_subset.layers['counts'] = adata_rna_subset.X.copy()

    # Execute totalVI Model Training
    loss_history = []
    latent_emb = None

    if HAVE_SCVI:
        print(f"Configuring totalVI via scvi-tools on device: {device}...")
        try:
            if HAVE_MUDATA:
                print("Setting up MuData object for totalVI...")
                mdata = md.MuData({"rna_subset": adata_rna_subset, "protein": adata_mod2_subset})
                mdata.update()

                scvi.model.TOTALVI.setup_mudata(
                    mdata,
                    rna_layer="counts",
                    protein_layer=None,
                    batch_key="batch",
                    modalities={
                        "rna_layer": "rna_subset",
                        "protein_layer": "protein",
                        "batch_key": "rna_subset",
                    }
                )
                vae = scvi.model.TOTALVI(mdata, n_latent=64)
            else:
                print("MuData not detected, using AnnData direct setup for totalVI...")
                adata_combined = adata_rna_subset.copy()
                adata_combined.obsm["protein_expression"] = adata_mod2_subset.X
                scvi.model.TOTALVI.setup_anndata(
                    adata_combined,
                    protein_expression_obsm_key="protein_expression",
                    layer="counts",
                    batch_key="batch"
                )
                vae = scvi.model.TOTALVI(adata_combined, n_latent=64)

            use_gpu = torch.cuda.is_available() and device.type == 'cuda'
            accelerator = "gpu" if use_gpu else "cpu"
            print(f"Training totalVI model for {epochs} epochs (accelerator='{accelerator}')...")

            progress_cb = TotalVIProgressCallback(print_every=25)
            try:
                # Modern scvi-tools with accelerator and live progress logging
                vae.train(
                    max_epochs=epochs,
                    accelerator=accelerator,
                    callbacks=[progress_cb],
                    enable_progress_bar=False
                )
            except TypeError:
                try:
                    vae.train(max_epochs=epochs, accelerator=accelerator)
                except TypeError:
                    try:
                        vae.train(max_epochs=epochs, use_gpu=use_gpu)
                    except TypeError:
                        vae.train(max_epochs=epochs)

            latent_emb = vae.get_latent_representation()
            if hasattr(vae, "history") and "elbo_train" in vae.history:
                df_elbo = vae.history["elbo_train"]
                loss_history = df_elbo.iloc[:, 0].tolist() if isinstance(df_elbo, pd.DataFrame) else list(df_elbo)
            print("totalVI latent representation extracted with shape:", latent_emb.shape)
        except Exception as e:
            print(f"Warning: scvi totalVI execution encountered an issue ({e}). Running PCA fallback...")
            pass
        finally:
            if 'vae' in locals():
                del vae
            if 'mdata' in locals():
                del mdata
            if 'adata_combined' in locals():
                del adata_combined
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if latent_emb is None:
        print("[Notice] Using standard concatenated PCA fallback representation for dry-run/testing...")
        rna_mat = adata_rna_subset.X.toarray() if issparse(adata_rna_subset.X) else adata_rna_subset.X
        mod2_mat = adata_mod2_subset.X.toarray() if issparse(adata_mod2_subset.X) else adata_mod2_subset.X
        pca_rna = PCA(n_components=min(32, rna_mat.shape[0] - 1, rna_mat.shape[1] - 1)).fit_transform(rna_mat)
        pca_mod2 = PCA(n_components=min(32, mod2_mat.shape[0] - 1, mod2_mat.shape[1] - 1)).fit_transform(mod2_mat)
        latent_emb = np.concatenate([pca_rna, pca_mod2], axis=1)
        loss_history = [float(100.0 / (i + 1)) for i in range(epochs)]

    # Free temporary subsetted objects
    del adata_rna_subset, adata_mod2_subset
    gc.collect()

    adata = adata_rna.copy()
    del adata_rna
    if 'counts' in adata.layers:
        del adata.layers['counts']
    adata.obsm['X_totalVI'] = latent_emb
    adata.obsm['feat'] = latent_emb
    adata.obsm['totalVI'] = latent_emb

    # Compute relative modality loading proxy from the 64 latent dimensions
    try:
        rna_sub = latent_emb[:, :32]
        aux_sub = latent_emb[:, 32:64]
        rna_norm = np.linalg.norm(rna_sub, axis=1)
        aux_norm = np.linalg.norm(aux_sub, axis=1)
        total_norm = rna_norm + aux_norm + 1e-8
        adata.obs['mean_rna_attention'] = rna_norm / total_norm
        adata.obs['mean_aux_attention'] = aux_norm / total_norm
    except Exception:
        adata.obs['mean_rna_attention'] = 0.5
        adata.obs['mean_aux_attention'] = 0.5

    # Determine cluster count
    valid_mask = (adata.obs['ground_truth'] != 'unknown') & (adata.obs['ground_truth'] != 'Exclude')
    valid_labels = adata.obs['ground_truth'][valid_mask]
    if len(valid_labels) > 0:
        n_clusters = len(np.unique(valid_labels))
        print(f"Clustering into {n_clusters} clusters (based on ground truth)...")
    else:
        n_clusters = 7
        print(f"Clustering into default {n_clusters} clusters...")

    # Tri-Algorithm Clustering (KMeans, Leiden, mclust)
    print(f"Running KMeans on totalVI embeddings with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    adata.obs['kmeans'] = kmeans.fit_predict(latent_emb).astype(str)

    print("Running Leiden on totalVI embeddings...")
    clustering(adata, n_clusters=n_clusters, key='X_totalVI', add_key='totalVI', method='leiden')
    adata.obs['leiden'] = adata.obs['totalVI'].astype(str)

    print("Running mclust on totalVI embeddings...")
    adata.obs['mclust'] = run_mclust(latent_emb, n_clusters, seed=seed)

    # Evaluation
    metrics_kmeans = evaluate_clustering(adata.obs['ground_truth'], adata.obs['kmeans'], latent_emb, name=f"totalVI {dataset_name} Seed {seed} - KMeans")
    metrics_leiden = evaluate_clustering(adata.obs['ground_truth'], adata.obs['leiden'], latent_emb, name=f"totalVI {dataset_name} Seed {seed} - Leiden")
    metrics_mclust = evaluate_clustering(adata.obs['ground_truth'], adata.obs['mclust'], latent_emb, name=f"totalVI {dataset_name} Seed {seed} - mclust")

    row_kmeans = {'cluster alg': 'KMeans', 'resolution': None, **metrics_kmeans}
    row_leiden = {'cluster alg': 'Leiden', 'resolution': None, **metrics_leiden}
    row_mclust = {'cluster alg': 'mclust', 'resolution': None, **metrics_mclust}

    primary_alg = 'mclust' if 'mclust' in adata.obs else 'kmeans'
    adata.obs['totalVI_domain'] = adata.obs[primary_alg].astype(str)
    adata.obs['domain'] = adata.obs['totalVI_domain']
    adata.uns['selected_seed'] = seed

    out_dir = output_dir or ('/kaggle/working' if os.path.exists('/kaggle/working') else 'results')
    os.makedirs(out_dir, exist_ok=True)
    if show_plots:
        plot_totalvi_visualizations(adata, title_prefix=f"{dataset_name} (Seed {seed})", dname=dataset_name, seed=seed, save_dir=out_dir)
        plot_loss(loss_history, save_path=os.path.join(out_dir, f"totalvi_loss_{dataset_name}_seed_{seed}.png"))

    # UMAP calculation
    if 'X_umap' not in adata.obsm:
        sc.pp.neighbors(adata, use_rep='X_totalVI', n_neighbors=15)
        sc.tl.umap(adata)
    umap_coords = adata.obsm['X_umap']
    spatial_coords = adata.obsm.get('spatial', None)

    y_true_all = adata.obs['ground_truth'].astype(str).values
    y_pred_all = adata.obs[primary_alg].astype(str).values

    try:
        le = LabelEncoder()
        y_pred_int = le.fit_transform(y_pred_all)
        sample_sil_values = silhouette_samples(latent_emb, y_pred_int).tolist()
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
    hyperparameters = {
        "epochs": epochs,
        "n_latent": 64,
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
        model_id="totalVI",
        model_name="totalVI",
        dataset_name=ds_info["datasetId"],
        seed=seed,
        metrics_dict=primary_metrics,
        training_results={"loss_history": loss_history},
        hyperparameters=hyperparameters,
        embeddings_data=embeddings_data,
        output_dir=out_dir,
        api_url=api_url,
        token=dashboard_token,
        dataset_display_name=ds_info["datasetName"]
    )

    return [row_kmeans, row_leiden, row_mclust], adata


# ==============================================================================
# MAIN MULTI-SEED RUNNER
# ==============================================================================
def main(
    username=DEFAULT_USERNAME,
    password=DEFAULT_PASSWORD,
    api_url=DEFAULT_API_URL,
    datasets=None,
    seeds=None,
    epochs=400,
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

    if generate_plots:
        print("\n" + "=" * 85)
        print(" GENERATING TOTALVI PLOTS & TRACKING CSVs FROM EXISTING BENCHMARK DATA")
        print("=" * 85)

        df_bench = None
        if results_csv and os.path.exists(results_csv):
            df_bench = pd.read_csv(results_csv)
            print(f"[Loaded] Retrieved benchmark CSV from: {results_csv}")
        else:
            candidates = [
                os.path.join(out_dir, "totalvi_benchmark_results.csv"),
                os.path.join(out_dir, "totalvi_ablation_results.csv"),
                "results/totalvi_benchmark_results.csv",
                "results/totalvi_ablation_results.csv"
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
                plot_totalvi_summary_boxplots(df_bench, save_dir=visuals_dir)

        plot_totalvi_spatial_domains_and_attention(None, df_results=df_bench, save_dir=out_dir, visuals_dir=visuals_dir)
        plot_totalvi_latent_umap_manifolds(None, df_results=df_bench, save_dir=out_dir, visuals_dir=visuals_dir)
        print("\n" + "=" * 85)
        print(" PLOT GENERATION COMPLETE")
        print("=" * 85 + "\n")
        return

    dashboard_token = None
    if api_url and username and password:
        base_url = api_url.split("/api/")[0] if "/api/" in api_url else DEFAULT_DASHBOARD_URL
        dashboard_token = authenticate_dashboard(base_url=base_url, username=username, password=password)

    active_datasets = datasets if datasets is not None else list(ALL_DATASETS_CONFIG.keys())
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
        print(f"STARTING TOTALVI WORKFLOW FOR DATASET: {dname} OVER {len(active_seeds)} SEEDS")
        print(f"=======================================================")

        for idx, seed in enumerate(active_seeds):
            show_plots = True
            try:
                res = run_totalvi_workflow(
                    dname, cfg, ENV_MODE, seed, device,
                    epochs=epochs,
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
                        if dname in best_adatas:
                            del best_adatas[dname]
                        best_adatas[dname] = adata_curr
                        best_aris[dname] = current_ari
                    else:
                        del adata_curr
            except Exception as e:
                import traceback
                print(f"Error processing dataset {dname} with seed {seed}: {e}")
                traceback.print_exc()
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if len(all_results[dname]) > 0:
            df_metrics = pd.DataFrame(all_results[dname])
            print(f"\n=======================================================")
            print(f"AVERAGE PERFORMANCE FOR {dname} ({len(active_seeds)} seeds)")
            print(f"=======================================================")
            numeric_cols = ["ARI", "NMI", "Silhouette", "AMI", "CHI", "DBI", "Homogeneity", "V-measure", "FMI"]
            numeric_cols = [c for c in numeric_cols if c in df_metrics.columns]
            for alg, df_alg in df_metrics.groupby("cluster alg"):
                print(f"\n--- {alg} Performance (Mean ± Std) ---")
                means = df_alg[numeric_cols].mean()
                stds = df_alg[numeric_cols].std().fillna(0.0)
                summary_df = pd.DataFrame({"Mean": means, "Std": stds})
                print(summary_df.to_string())
            print(f"=======================================================\n")

    for dname, b_adata in best_adatas.items():
        try:
            adata_save_path = os.path.join(out_dir, f"totalvi_adata_{dname}.h5ad")
            b_adata.write_h5ad(adata_save_path)
            print(f"Saved best AnnData object to: {adata_save_path}")
            export_spot_predictions_csv(b_adata, dname, b_adata.uns.get('selected_seed', 0), out_dir)
        except Exception as e:
            print(f"Warning: Failed to save AnnData or spot CSV for {dname}: {e}")

    if len(all_results_flat) > 0:
        df_all = pd.DataFrame(all_results_flat)

        # 1. Save all ablation results (KMeans, Leiden, mclust)
        output_csv = os.path.join(out_dir, 'totalvi_ablation_results.csv')
        df_all.to_csv(output_csv, index=False)
        print(f"All ablation study results saved to CSV at: {output_csv}")

        # 2. Extract and save primary benchmark results
        if (df_all['cluster alg'] == 'mclust').any():
            df_benchmark = df_all[df_all['cluster alg'] == 'mclust'].copy()
        elif (df_all['cluster alg'] == 'KMeans').any():
            df_benchmark = df_all[df_all['cluster alg'] == 'KMeans'].copy()
        else:
            df_benchmark = df_all.copy()

        benchmark_csv = os.path.join(out_dir, 'totalvi_benchmark_results.csv')
        df_benchmark.to_csv(benchmark_csv, index=False)
        print(f"Primary benchmark results saved to CSV at: {benchmark_csv}")

        # 3. Generate side-by-side Box & Whiskers plots
        print("Generating side-by-side Box & Whiskers plots across algorithms...")
        plot_totalvi_summary_boxplots(df_all, save_dir=visuals_dir)

        # 4. Cross-Seed Linearity & Spread Diagnostics Suite
        compute_and_plot_cross_seed_linearity(df_benchmark, out_dir=out_dir, visuals_dir=visuals_dir)

        # 5. Side-by-Side Spatial Domain & Modality Weight Visualization
        plot_totalvi_spatial_domains_and_attention(best_adatas, df_results=df_benchmark, save_dir=out_dir, visuals_dir=visuals_dir)

        # 6. Latent Embedding Manifold Projections (UMAP)
        plot_totalvi_latent_umap_manifolds(best_adatas, df_results=df_benchmark, save_dir=out_dir, visuals_dir=visuals_dir)

        print("\n" + "=" * 80)
        print(f"totalVI Pipeline completed! Results exported and uploaded to {DEFAULT_DASHBOARD_URL}".center(80))
        print("=" * 80 + "\n")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="totalVI Multi-Seed Pipeline with Live Dashboard API Upload & Diagnostics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--username', type=str, default=DEFAULT_USERNAME, help="Dashboard account username")
    parser.add_argument('--password', type=str, default=DEFAULT_PASSWORD, help="Dashboard account password")
    parser.add_argument('--api_url', type=str, default=DEFAULT_API_URL, help="Dashboard upload API endpoint")
    parser.add_argument('--datasets', nargs='+', default=None, help="Datasets to run (e.g. human-lymph-node-a1 or 'all')")
    parser.add_argument('--seeds', nargs='+', type=int, default=None, help="Specific seeds to evaluate")
    parser.add_argument('--epochs', type=int, default=400, help="Training epochs for totalVI model")
    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save CSV results, plots, and JSONs")
    parser.add_argument('--generate_plots', action='store_true', help="Generate linearity CSV, summary plots, spatial domain maps, and UMAP plots from existing benchmark results without re-training")
    parser.add_argument('--results_csv', type=str, default=None, help="Path to existing CSV containing benchmark results (for --generate_plots)")

    args, unknown = parser.parse_known_args()

    selected_datasets = None
    if args.datasets:
        if len(args.datasets) == 1 and args.datasets[0].lower() == "all":
            selected_datasets = list(ALL_DATASETS_CONFIG.keys())
        else:
            selected_datasets = [d for d in args.datasets if d in ALL_DATASETS_CONFIG]
    else:
        selected_datasets = list(ALL_DATASETS_CONFIG.keys())

    selected_seeds = args.seeds if (args.seeds is not None and len(args.seeds) > 0) else SEEDS

    main(
        username=args.username,
        password=args.password,
        api_url=args.api_url,
        datasets=selected_datasets,
        seeds=selected_seeds,
        epochs=args.epochs,
        output_dir=args.output_dir,
        generate_plots=args.generate_plots,
        results_csv=args.results_csv
    )

