#!/usr/bin/env python3
"""
scGPT Multi-Seed Baseline Pipeline (Standalone Python Script)
Integrates baseline scGPT RNA embedding fusion workflow across 6 target datasets:
- Mouse Brain E11, E13, E15, E18 (scGPT RNA + RNA PCA + ATAC PCA)
- Human Lymph Node A1, D1 (scGPT RNA + RNA PCA + ADT/Protein CLR PCA)
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

SEEDS = [
    42, 0, 1, 7, 123, 1234, 2022, 2023, 2024, 1337
]

if len(ACTIVE_DATASETS) == 1 and ACTIVE_DATASETS[0].lower() == "all":
    datasets_to_run = list(ALL_DATASETS_CONFIG.keys())
else:
    datasets_to_run = [d for d in ACTIVE_DATASETS if d in ALL_DATASETS_CONFIG]

print(f"Scheduled datasets: {datasets_to_run}")
print(f"Ablation seeds: {SEEDS}")


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 4
# ===========================================================================
# --- COMBINED scGPT BASELINE UTILITIES ---
import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
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
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

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
        
        try:
            robjects.r.library("mclust")
        except Exception:
            print("Installing R package 'mclust'...")
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
    except Exception as e:
        print(f"Warning: mclust execution via rpy2 failed ({e}). Returning fallback clusters.")
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(data_mat).astype(str)

def search_res(adata, n_clusters, method='leiden', use_rep='joint_feat', start=0.1, end=3.0, increment=0.01):
    """Search for resolution to achieve target cluster count."""
    print('Searching resolution...')
    sc.pp.neighbors(adata, n_neighbors=10, use_rep=use_rep)
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
            print(f'Found resolution={res} with target cluster count={n_clusters}')
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

def plot_scgpt_visualizations(adata, title_prefix="", dname="dataset", seed=2024, save_dir=None):
    """Plot UMAP and Spatial visualizations for Ground Truth, KMeans, Leiden, and mclust, and save PNG image."""
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
    save_path = os.path.join(save_dir, f"scgpt_plot_{dname}_seed_{seed}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization plot image to: {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)

def plot_scgpt_summary_boxplots(df_all, save_dir=None):
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
        ax.set_title(f"scGPT Baseline: {metric} Performance Across Datasets", fontsize=14, fontweight='bold')
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title="Cluster Alg", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"scgpt_boxplot_{metric.lower().replace('-', '_')}.png")
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
    combined_path = os.path.join(save_dir, "scgpt_boxplot_all_metrics.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined multi-metric Boxplot figure to: {combined_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 6
# ===========================================================================
def load_dataset_data(dname, cfg, env_mode="auto"):
    """Dynamically load RNA, Modality 2 (ATAC/ADT), and annotations for any specified dataset."""
    is_kaggle = (env_mode == "kaggle") or (env_mode == "auto" and os.path.exists("/kaggle/input"))
    data_dir = cfg["kaggle_dir"] if is_kaggle else cfg["local_dir"]
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Returning dummy data for local validation...")
        adata_rna = sc.datasets.pbmc3k()
        adata_mod2 = adata_rna.copy()
        adata_rna.obsm['spatial'] = np.random.randn(adata_rna.n_obs, 2)
        adata_mod2.obsm['spatial'] = adata_rna.obsm['spatial']
        adata_rna.obs['ground_truth'] = adata_rna.obs['louvain'].astype(str) if 'louvain' in adata_rna.obs else 'Cluster1'
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

    print(f"RNA shape: {adata_rna.shape}")
    print(f"Modality 2 shape: {adata_mod2.shape}")
    return adata_rna, adata_mod2


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 8
# ===========================================================================
def get_scgpt_embeddings(adata_rna, device):
    """Generate scGPT embeddings for adata_rna using scGPT pretrained weights."""
    from scgpt.tasks.cell_emb import embed_data
    
    KAGGLE_MODEL_DIR = '/kaggle/input/datasets/sadmanbiazidarnob/scgpt-human/scGPT_human'
    LOCAL_MODEL_DIR = 'D:/FYDP/GATCON/d1_test/scGPT_human'
    
    model_dir = KAGGLE_MODEL_DIR if os.path.exists(KAGGLE_MODEL_DIR) else LOCAL_MODEL_DIR
    
    adata_rna.var_names_make_unique()
    adata_rna.var['gene_names'] = adata_rna.var.index.str.upper()

    print(f"Running scGPT embedding on GPU using model at: {model_dir}")
    if os.path.exists(model_dir):
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
    else:
        print(f"Warning: Model directory '{model_dir}' not found. Generating dummy random embeddings...")
        return np.random.randn(adata_rna.n_obs, 512)


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 10
# ===========================================================================
def run_scgpt_workflow(dname, cfg, env_mode, seed, device, show_plots=False):
    """Executes the complete baseline scGPT workflow for a single dataset under a specific seed using KMeans, Leiden, and mclust."""
    fix_seed(seed)
    print(f"\n--- Running Dataset: {dname} | Seed: {seed} ---")

    # 1. Load Data
    adata_rna, adata_mod2 = load_dataset_data(dname, cfg, env_mode)

    # 2. Extract scGPT Embedding
    X_scGPT = get_scgpt_embeddings(adata_rna, device)
    adata_rna.obsm['X_scGPT'] = X_scGPT

    # 3. Preprocess Omic 1 (RNA) & Omic 2 (ATAC vs ADT/Protein)
    sc.pp.filter_genes(adata_rna, min_cells=10)
    sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata_rna, target_sum=1e4)
    sc.pp.log1p(adata_rna)
    sc.pp.scale(adata_rna, max_value=10)
    adata_rna_high = adata_rna[:, adata_rna.var['highly_variable']]

    if cfg["type"] == "mouse_brain":
        n_comps_rna = min(50, adata_rna_high.n_obs - 1, adata_rna_high.n_vars - 1)
        feat_rna = pca(adata_rna_high, n_comps=n_comps_rna)
        
        sc.pp.normalize_total(adata_mod2, target_sum=1e4)
        sc.pp.log1p(adata_mod2)
        sc.pp.scale(adata_mod2, max_value=10)
        n_comps_mod2 = min(50, adata_mod2.n_obs - 1, adata_mod2.n_vars - 1)
        feat_mod2 = pca(adata_mod2, n_comps=n_comps_mod2)
    else:
        n_comps_rna = adata_mod2.n_vars - 1
        feat_rna = pca(adata_rna_high, n_comps=n_comps_rna)
        
        clr_normalize_each_cell(adata_mod2)
        sc.pp.scale(adata_mod2, max_value=10)
        n_comps_mod2 = adata_mod2.n_vars - 1
        feat_mod2 = pca(adata_mod2, n_comps=n_comps_mod2)

    adata_rna.obsm['feat'] = feat_rna
    adata_mod2.obsm['feat'] = feat_mod2

    # 5. Concatenate (Fuse) scGPT Embedding with RNA and Modality 2 Features
    joint_feat = np.concatenate((X_scGPT, feat_rna, feat_mod2), axis=1)
    adata_rna.obsm['joint_feat'] = joint_feat
    print(f"Fused feature matrix shape: {joint_feat.shape}")

    # Determine target number of clusters from ground truth
    valid_labels = adata_rna.obs['ground_truth'].dropna().unique()
    target_labels = [l for l in valid_labels if l not in ['Exclude', 'unknown']]
    n_clusters = len(target_labels) if len(target_labels) > 0 else 7

    # 6a. KMeans Clustering
    print(f"Running KMeans clustering with n_clusters={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    adata_rna.obs['kmeans'] = kmeans.fit_predict(joint_feat).astype(str)

    # 6b. Leiden Graph Clustering
    print("Running Leiden clustering...")
    res = search_res(adata_rna, n_clusters, method='leiden', use_rep='joint_feat')
    try:
        sc.tl.leiden(adata_rna, random_state=seed, resolution=res, flavor='igraph', n_iterations=2, directed=False)
    except TypeError:
        sc.tl.leiden(adata_rna, random_state=seed, resolution=res)

    # 6c. mclust Clustering
    print("Running mclust clustering via rpy2...")
    adata_rna.obs['mclust'] = run_mclust(joint_feat, n_clusters, seed=seed)

    # 7. Cluster Performance Evaluation (KMeans, Leiden, mclust)
    metrics_kmeans = evaluate_clustering(
        adata_rna.obs['ground_truth'],
        adata_rna.obs['kmeans'],
        joint_feat,
        name=f"{dname} Seed {seed} - KMeans"
    )

    metrics_leiden = evaluate_clustering(
        adata_rna.obs['ground_truth'],
        adata_rna.obs['leiden'],
        joint_feat,
        name=f"{dname} Seed {seed} - Leiden"
    )

    metrics_mclust = evaluate_clustering(
        adata_rna.obs['ground_truth'],
        adata_rna.obs['mclust'],
        joint_feat,
        name=f"{dname} Seed {seed} - mclust"
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
        'resolution': res
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

    # 8. Visualization & Plot Saving (First seed run only)
    if show_plots:
        plot_scgpt_visualizations(adata_rna, title_prefix=f"{dname} (Seed {seed})", dname=dname, seed=seed)

    return [row_kmeans, row_leiden, row_mclust]


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 12
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
                results_list = run_scgpt_workflow(dname, cfg, ENV_MODE, seed, device, show_plots=show_plots)
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
            numeric_cols = ["ARI", "NMI", "Silhouette", "AMI", "CHI", "DBI", "Homogeneity", "V-measure"]
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
        output_csv = os.path.join(output_dir, 'scgpt_ablation_results.csv')
        df_all.to_csv(output_csv, index=False)
        print(f"All ablation study results saved to CSV at: {output_csv}")
        
        print("Generating side-by-side Box & Whiskers plots for all metrics across algorithms...")
        plot_scgpt_summary_boxplots(df_all, save_dir=output_dir)
    else:
        print("No results were generated, skipping CSV export and Boxplots.")

if __name__ == '__main__':
    main()


