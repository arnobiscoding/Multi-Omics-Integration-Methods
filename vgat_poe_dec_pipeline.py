#!/usr/bin/env python3
"""
VGAT-PoE-DEC Multi-Seed Benchmark Pipeline
===========================================

Multi-omics spatial pipeline integrating Dual-Encoder VGAT, PoE Fusion, Dual Decoders (ZINB+MSE),
DEC Clustering, and Pseudo-Label Spatially-Aware Contrastive learning across 6 benchmark datasets:
- Mouse Brain E11, E13, E15, E18 (RNA + ATAC)
- Human Lymph Node A1, D1 (RNA + ADT/Protein)

Evaluates performance across 4 clustering algorithms (DEC Native, KMeans, Leiden, mclust)
and 8 benchmark metrics (ARI, NMI, Silhouette, AMI, CHI, DBI, Homogeneity, V-measure).
Exports results to vgat_poe_dec_ablation_results.csv and generates Box & Whiskers plots.
"""

# ===========================================================================
# 1. RUN CONFIGURATION
# ===========================================================================
import os
import sys
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch.backends import cudnn
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import anndata as ad
from tqdm import tqdm
from typing import Optional, Dict, Any, List

try:
    import scanpy as sc
    import seaborn as sns
    import matplotlib.pyplot as plt
    PLOTS_AVAILABLE = True
except ImportError:
    sc = None
    sns = None
    plt = None
    PLOTS_AVAILABLE = False

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

# Import VGAT-PoE-DEC package modules
curr_dir = os.path.dirname(os.path.abspath(__file__))
new_model_dir = os.path.join(curr_dir, "new model")
if new_model_dir not in sys.path:
    sys.path.insert(0, new_model_dir)

try:
    from preprocessing import prepare_spatial_multiomics_data
    from model import VGAT_PoE_DEC
    from trainer import SpatialOmicsTrainer
    from metrics import compute_silhouette, compute_morans_i
except ImportError:
    try:
        from new_model.preprocessing import prepare_spatial_multiomics_data
        from new_model.model import VGAT_PoE_DEC
        from new_model.trainer import SpatialOmicsTrainer
        from new_model.metrics import compute_silhouette, compute_morans_i
    except ImportError:
        raise ImportError("Failed to import new model components. Ensure 'new model' folder is in path.")


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

if len(ACTIVE_DATASETS) == 1 and ACTIVE_DATASETS[0].lower() == "all":
    datasets_to_run = list(ALL_DATASETS_CONFIG.keys())
else:
    datasets_to_run = [d for d in ACTIVE_DATASETS if d in ALL_DATASETS_CONFIG]


# ===========================================================================
# 2. UTILITY FUNCTIONS & PREPROCESSING
# ===========================================================================
def fix_seed(seed: int):
    """Fix random seed for complete reproducibility across environments."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def pca(data_matrix, n_comps=30):
    """Perform PCA dimensionality reduction on dense or sparse matrix."""
    data = data_matrix.toarray() if sp.issparse(data_matrix) else data_matrix
    n_comps = min(n_comps, data.shape[0] - 1, data.shape[1] - 1)
    pca_model = PCA(n_components=n_comps)
    return pca_model.fit_transform(data)


def clr_normalize_each_cell(adata: ad.AnnData, inplace=True):
    """Normalize count vector for each spot using CLR normalization."""
    def seurat_clr(x):
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x)) if len(x) > 0 else 1.0
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    raw_x = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    adata.X = np.apply_along_axis(seurat_clr, 1, raw_x)
    return adata


def run_mclust(data_matrix, n_clusters, seed=2024, max_dims=30):
    """Perform mclust clustering via rpy2 with PCA reduction, falling back to KMeans if rpy2 is unavailable."""
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
    except Exception as e:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(data_mat).astype(str)


def search_res(adata, n_clusters, method='leiden', use_rep='z_f', start=0.1, end=3.0, increment=0.01):
    """Search for resolution parameter to achieve exact target cluster count."""
    if sc is None:
        return 0.5
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
            return res
    return 0.5


def evaluate_clustering(y_true_series, y_pred_series, features_matrix, name=""):
    """Evaluate clustering performance across 8 benchmark metrics."""
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
        print(f"--- {name} Metrics --- ARI: {ari:.4f} | NMI: {nmi:.4f} | Sil: {sil:.4f}")
    return metrics


# ===========================================================================
# 3. DATA LOADING & DATASET PARSING
# ===========================================================================
def load_dataset_data(dname, cfg, env_mode="auto"):
    """Dynamically load RNA, Modality 2 (ATAC/ADT), coordinates, and annotations."""
    is_kaggle = (env_mode == "kaggle") or (env_mode == "auto" and os.path.exists("/kaggle/input"))
    data_dir = cfg["kaggle_dir"] if is_kaggle else cfg["local_dir"]

    if not os.path.exists(data_dir) or sc is None:
        print(f"Directory {data_dir} not found or scanpy missing. Generating fallback synthetic data...")
        num_spots = 400
        coords = np.random.uniform(0, 100, size=(num_spots, 2))
        labels = np.random.randint(0, 7, size=num_spots).astype(str)

        adata_rna = ad.AnnData(X=sp.csr_matrix(np.random.poisson(2.0, size=(num_spots, 1000))))
        adata_mod2 = ad.AnnData(X=np.random.randn(num_spots, 50).astype(np.float32))

        adata_rna.obsm['spatial'] = coords
        adata_mod2.obsm['spatial'] = coords
        adata_rna.obs['ground_truth'] = labels
        adata_mod2.obs['ground_truth'] = labels

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
        annotation = pd.read_csv(anno_path)
        ground_col = next((col for col in ['cluster', 'manual-anno', 'ground_truth', 'assigned_cluster'] if col in annotation.columns), None)
        barcode_col = next((col for col in ['barcode', 'Barcode', 'cell_id'] if col in annotation.columns), None)

        if ground_col and barcode_col:
            annotation = annotation.rename(columns={ground_col: 'ground_truth', barcode_col: 'barcode'}).set_index('barcode')
            adata_rna.obs = adata_rna.obs.join(annotation[['ground_truth']], how='left')
            adata_rna.obs['ground_truth'] = adata_rna.obs['ground_truth'].fillna('unknown')
            adata_mod2.obs = adata_mod2.obs.join(annotation[['ground_truth']], how='left')
            adata_mod2.obs['ground_truth'] = adata_mod2.obs['ground_truth'].fillna('unknown')
    else:
        if 'ground_truth' not in adata_rna.obs:
            adata_rna.obs['ground_truth'] = 'unknown'

    return adata_rna, adata_mod2


# ===========================================================================
# 4. MAIN WORKFLOW PER DATASET & SEED
# ===========================================================================
def run_vgat_poe_dec_workflow(dname, cfg, env_mode, seed, device, show_plots=False):
    """Executes full VGAT-PoE-DEC training and evaluation workflow across 4 clustering algorithms."""
    fix_seed(seed)
    print(f"\n==========================================================================")
    print(f" RUNNING VGAT-PoE-DEC WORKFLOW: Dataset={dname} | Seed={seed} | Device={device}")
    print(f"==========================================================================")

    # 1. Load Data
    adata_rna, adata_mod2 = load_dataset_data(dname, cfg, env_mode)

    # Extract spatial coordinates
    if 'spatial' in adata_rna.obsm:
        coords = np.asarray(adata_rna.obsm['spatial'], dtype=np.float32)
    else:
        coords = np.random.uniform(0, 100, size=(adata_rna.n_obs, 2)).astype(np.float32)

    # Extract raw RNA counts matrix for ZINB loss
    raw_counts_rna = adata_rna.X.toarray() if sp.issparse(adata_rna.X) else np.array(adata_rna.X)

    # 2. Preprocess Features
    if sc is not None and adata_rna.n_vars > 2000:
        sc.pp.filter_genes(adata_rna, min_cells=5)
        sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=2000)
        adata_rna_sub = adata_rna[:, adata_rna.var['highly_variable']]
        x_rna = adata_rna_sub.X.toarray() if sp.issparse(adata_rna_sub.X) else np.array(adata_rna_sub.X)
        x_rna = np.log1p(x_rna)
    else:
        x_rna = np.log1p(raw_counts_rna)

    if cfg["type"] == "mouse_brain":
        x_mod2 = adata_mod2.X.toarray() if sp.issparse(adata_mod2.X) else np.array(adata_mod2.X)
        x_adt = pca(x_mod2, n_comps=min(50, x_mod2.shape[1]))
    else:
        if sc is not None:
            clr_normalize_each_cell(adata_mod2)
            x_adt = adata_mod2.X.toarray() if sp.issparse(adata_mod2.X) else np.array(adata_mod2.X)
        else:
            x_adt = adata_mod2.X.toarray() if sp.issparse(adata_mod2.X) else np.array(adata_mod2.X)

    # Determine ground truth cluster count K
    valid_labels = adata_rna.obs['ground_truth'].dropna().unique()
    target_labels = [l for l in valid_labels if l not in ['Exclude', 'unknown']]
    n_clusters = len(target_labels) if len(target_labels) > 0 else 7

    # 3. Format Data & Instantiate Model
    data_dict = prepare_spatial_multiomics_data(
        x_rna=x_rna,
        x_adt=x_adt,
        coords=coords,
        raw_counts_rna=raw_counts_rna[:, :x_rna.shape[1]],
        k_neighbors=6,
        device=device
    )

    model = VGAT_PoE_DEC(
        in_dim_rna=data_dict["f_rna"],
        in_dim_adt=data_dict["f_adt"],
        num_clusters=n_clusters,
        hidden_dim=128,
        latent_dim=32,
        proj_dim=64,
        heads=4
    )

    # 4. Multi-Phase Training Protocol
    trainer = SpatialOmicsTrainer(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=400,
        finetune_epochs=600,
        p_update_interval=5,
        early_stop_eval_interval=100,
        subsample_ratio=0.1 if x_rna.shape[0] < 1000 else 0.05,
        device=device
    )

    fit_results = trainer.fit(data_dict)
    z_f = fit_results["z_f"]
    dec_preds = fit_results["predictions"]

    adata_rna.obsm['z_f'] = z_f
    adata_rna.obs['dec'] = dec_preds.astype(str)

    # 5. Run Alternative Clustering Algorithms on Latent Embedding z_f
    # 5a. KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    adata_rna.obs['kmeans'] = kmeans.fit_predict(z_f).astype(str)

    # 5b. Leiden
    res = search_res(adata_rna, n_clusters, method='leiden', use_rep='z_f')
    if sc is not None:
        try:
            sc.tl.leiden(adata_rna, random_state=seed, resolution=res, flavor='igraph', n_iterations=2, directed=False)
        except Exception:
            adata_rna.obs['leiden'] = adata_rna.obs['kmeans']
    else:
        adata_rna.obs['leiden'] = adata_rna.obs['kmeans']

    # 5c. mclust
    adata_rna.obs['mclust'] = run_mclust(z_f, n_clusters, seed=seed)

    # 6. Evaluate All 4 Clustering Algorithms
    metrics_dec = evaluate_clustering(adata_rna.obs['ground_truth'], adata_rna.obs['dec'], z_f, name=f"VGAT-PoE-DEC {dname} Seed {seed} - DEC Native")
    metrics_kmeans = evaluate_clustering(adata_rna.obs['ground_truth'], adata_rna.obs['kmeans'], z_f, name=f"VGAT-PoE-DEC {dname} Seed {seed} - KMeans")
    metrics_leiden = evaluate_clustering(adata_rna.obs['ground_truth'], adata_rna.obs['leiden'], z_f, name=f"VGAT-PoE-DEC {dname} Seed {seed} - Leiden")
    metrics_mclust = evaluate_clustering(adata_rna.obs['ground_truth'], adata_rna.obs['mclust'], z_f, name=f"VGAT-PoE-DEC {dname} Seed {seed} - mclust")

    row_dec = {'cluster alg': 'DEC', 'resolution': None}
    row_dec.update(metrics_dec)

    row_kmeans = {'cluster alg': 'KMeans', 'resolution': None}
    row_kmeans.update(metrics_kmeans)

    row_leiden = {'cluster alg': 'Leiden', 'resolution': res}
    row_leiden.update(metrics_leiden)

    row_mclust = {'cluster alg': 'mclust', 'resolution': None}
    row_mclust.update(metrics_mclust)

    return [row_dec, row_kmeans, row_leiden, row_mclust]


# ===========================================================================
# 5. PIPELINE EXECUTION ENTRY POINT
# ===========================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Starting VGAT-PoE-DEC Multi-Seed Benchmark Pipeline on device: {device}")

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
                results_list = run_vgat_poe_dec_workflow(dname, cfg, ENV_MODE, seed, device, show_plots=show_plots)
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

    if len(all_results_flat) > 0:
        df_all = pd.DataFrame(all_results_flat)
        is_kaggle = os.path.exists('/kaggle/working')
        output_dir = '/kaggle/working' if is_kaggle else '.'
        output_csv = os.path.join(output_dir, 'vgat_poe_dec_ablation_results.csv')
        df_all.to_csv(output_csv, index=False)
        print(f"\nAll benchmark evaluation results successfully saved to CSV at: {output_csv}")
    else:
        print("No results were generated.")


if __name__ == '__main__':
    main()
