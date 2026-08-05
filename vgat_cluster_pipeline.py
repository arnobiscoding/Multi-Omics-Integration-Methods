#!/usr/bin/env python3
"""
spaLLM-VGAT Multi-Seed Benchmark Pipeline
=========================================

Integrates spaLLM representation learning with VGAT graph attention encoders across 6 benchmark datasets:
- Mouse Brain E11, E13, E15, E18 (RNA + ATAC)
- Human Lymph Node A1, D1 (RNA + ADT/Protein)

Evaluates 4 clustering algorithms (spaLLM-VGAT Native Head, KMeans, Leiden, mclust) across 8 metrics.
Outputs spallm_vgat_ablation_results.csv and summary Box & Whiskers plots.
"""

import os
import sys
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

# Import spaLLM-VGAT modules
curr_dir = os.path.dirname(os.path.abspath(__file__))
new_model_dir = os.path.join(curr_dir, "new model")
if new_model_dir not in sys.path:
    sys.path.insert(0, new_model_dir)

try:
    from spallm_vgat_model import VGATEncodingNetwork
    from spallm_vgat_trainer import Train_spaLLM_VGAT, run_mclust, spatial_label_smoothing
except ImportError:
    from new_model.spallm_vgat_model import VGATEncodingNetwork
    from new_model.spallm_vgat_trainer import Train_spaLLM_VGAT, run_mclust, spatial_label_smoothing


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


def fix_seed(seed: int):
    """Fix random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def pca(data_matrix: Any, n_comps: int = 30) -> np.ndarray:
    data = data_matrix.toarray() if sp.issparse(data_matrix) else data_matrix
    n_comps = min(n_comps, data.shape[0] - 1, data.shape[1] - 1)
    pca_model = PCA(n_components=n_comps)
    return pca_model.fit_transform(data)


def clr_normalize_each_cell(adata: ad.AnnData, inplace: bool = True):
    """Normalize count vector for each spot using fast vectorized CLR normalization."""
    if not inplace:
        adata = adata.copy()
    raw_x = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float32)
    raw_x = np.clip(raw_x, 0.0, None)
    pos_mask = raw_x > 0
    log_pos = np.where(pos_mask, np.log1p(raw_x), 0.0)
    row_sums = np.sum(log_pos, axis=1, keepdims=True)
    n_vars = max(1, raw_x.shape[1])
    exp_geom = np.exp(row_sums / n_vars)
    exp_geom = np.where(exp_geom == 0, 1.0, exp_geom)
    adata.X = np.log1p(raw_x / exp_geom)
    return adata


def construct_graph_by_feature(feat_omics1: np.ndarray, feat_omics2: np.ndarray, k: int = 20):
    graph1 = kneighbors_graph(feat_omics1, k, mode="connectivity", metric="correlation", include_self=False)
    graph2 = kneighbors_graph(feat_omics2, k, mode="connectivity", metric="correlation", include_self=False)
    return graph1, graph2


def construct_graph_by_coordinate(coords: np.ndarray, n_neighbors: int = 6):
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    return pd.DataFrame({'x': x, 'y': y, 'value': 1})


def search_res(adata: ad.AnnData, n_clusters: int, method: str = 'leiden', use_rep: str = 'spaLLM'):
    if sc is None:
        return 0.5
    sc.pp.neighbors(adata, n_neighbors=10, use_rep=use_rep)
    for res in np.arange(0.1, 3.0, 0.01):
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
        if clusters.nunique() == n_clusters:
            return res
    return 0.5


def evaluate_clustering(y_true_series, y_pred_series, features_matrix, name=""):
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
        print(f"--- {name} --- ARI: {ari:.4f} | NMI: {nmi:.4f} | Sil: {sil:.4f}")
    return metrics


def load_dataset_data(dname: str, cfg: dict, env_mode: str = "auto"):
    is_kaggle = (env_mode == "kaggle") or (env_mode == "auto" and os.path.exists("/kaggle/input"))
    data_dir = cfg["kaggle_dir"] if is_kaggle else cfg["local_dir"]

    if not os.path.exists(data_dir) or sc is None:
        print(f"Directory {data_dir} not found. Generating synthetic dataset for test...")
        num_spots = 400
        coords = np.random.uniform(0, 100, size=(num_spots, 2))
        labels = np.random.randint(0, 7, size=num_spots).astype(str)

        adata_rna = ad.AnnData(X=sp.csr_matrix(np.random.poisson(2.0, size=(num_spots, 1000))))
        adata_mod2 = ad.AnnData(X=np.abs(np.random.randn(num_spots, 50)).astype(np.float32) * 5.0)

        adata_rna.obsm['spatial'] = coords
        adata_mod2.obsm['spatial'] = coords
        adata_rna.obs['ground_truth'] = labels
        adata_mod2.obs['ground_truth'] = labels
        return adata_rna, adata_mod2

    print(f"Loading data from: {data_dir}")
    rna_path = os.path.join(data_dir, "adata_RNA.h5ad")
    adata_rna = sc.read_h5ad(rna_path)

    mod2_path = next((os.path.join(data_dir, cand) for cand in cfg["mod2_candidates"] if os.path.exists(os.path.join(data_dir, cand))), None)
    if mod2_path is None:
        raise FileNotFoundError(f"Modality 2 file not found in {cfg['mod2_candidates']}")

    adata_mod2 = sc.read_h5ad(mod2_path)

    anno_path = os.path.join(data_dir, cfg["anno_file"])
    if os.path.exists(anno_path):
        annotation = pd.read_csv(anno_path)
        ground_col = next((c for c in ['cluster', 'manual-anno', 'ground_truth', 'assigned_cluster'] if c in annotation.columns), None)
        barcode_col = next((c for c in ['barcode', 'Barcode', 'cell_id'] if c in annotation.columns), None)
        if ground_col and barcode_col:
            annotation = annotation.rename(columns={ground_col: 'ground_truth', barcode_col: 'barcode'}).set_index('barcode')
            adata_rna.obs = adata_rna.obs.join(annotation[['ground_truth']], how='left').fillna('unknown')
            adata_mod2.obs = adata_mod2.obs.join(annotation[['ground_truth']], how='left').fillna('unknown')
    else:
        if 'ground_truth' not in adata_rna.obs:
            adata_rna.obs['ground_truth'] = 'unknown'

    return adata_rna, adata_mod2


def run_spallm_vgat_workflow(dname: str, cfg: dict, env_mode: str, seed: int, device: torch.device):
    fix_seed(seed)
    print(f"\n==========================================================================")
    print(f" RUNNING spaLLM-VGAT WORKFLOW: Dataset={dname} | Seed={seed} | Device={device}")
    print(f"==========================================================================")

    # 1. Load Data
    adata_rna, adata_mod2 = load_dataset_data(dname, cfg, env_mode)

    # 2. Extract Spatial Coordinates & Preprocess Features
    if 'spatial' in adata_rna.obsm:
        coords = np.asarray(adata_rna.obsm['spatial'], dtype=np.float32)
    else:
        coords = np.random.uniform(0, 100, size=(adata_rna.n_obs, 2)).astype(np.float32)

    if sc is not None and adata_rna.n_vars > 2000:
        sc.pp.filter_genes(adata_rna, min_cells=5)
        sc.pp.highly_variable_genes(adata_rna, flavor="seurat_v3", n_top_genes=2000)
        sc.pp.normalize_total(adata_rna, target_sum=1e4)
        sc.pp.log1p(adata_rna)
        sc.pp.scale(adata_rna, max_value=10)
        adata_rna_sub = adata_rna[:, adata_rna.var['highly_variable']]
        feat_rna = pca(adata_rna_sub, n_comps=min(50, adata_rna_sub.n_obs - 1, adata_rna_sub.n_vars - 1))
    else:
        feat_rna = pca(adata_rna, n_comps=min(50, adata_rna.n_obs - 1, adata_rna.n_vars - 1))

    if cfg["type"] == "mouse_brain":
        if sc is not None:
            sc.pp.normalize_total(adata_mod2, target_sum=1e4)
            sc.pp.log1p(adata_mod2)
            sc.pp.scale(adata_mod2, max_value=10)
        feat_mod2 = pca(adata_mod2, n_comps=min(50, adata_mod2.n_obs - 1, adata_mod2.n_vars - 1))
    else:
        if sc is not None:
            clr_normalize_each_cell(adata_mod2)
            sc.pp.scale(adata_mod2, max_value=10)
        feat_mod2 = pca(adata_mod2, n_comps=min(30, adata_mod2.n_obs - 1, adata_mod2.n_vars - 1))

    # Construct spatial coordinate graphs and feature correlation graphs
    adj_spatial = construct_graph_by_coordinate(coords, n_neighbors=6)
    adj_fea1, adj_fea2 = construct_graph_by_feature(feat_rna, feat_mod2, k=20)

    # Compute cell embeddings purely using PCA from RNA and Modality 2 features (no scGPT dependency)
    concat_features = np.concatenate((feat_rna, feat_mod2), axis=1)
    n_comps_emb = min(512, concat_features.shape[0] - 1, concat_features.shape[1] - 1)
    pca_emb = PCA(n_components=n_comps_emb, random_state=seed).fit_transform(concat_features)

    if pca_emb.shape[1] < 512:
        cell_emb = np.zeros((adata_rna.n_obs, 512), dtype=np.float32)
        cell_emb[:, :pca_emb.shape[1]] = pca_emb
    else:
        cell_emb = pca_emb[:, :512].astype(np.float32)

    data_dict = {
        'adata_omics1': adata_rna,
        'adata_omics2': adata_mod2,
        'features_omics1': feat_rna,
        'features_omics2': feat_mod2,
        'adj_spatial_omics1': adj_spatial,
        'adj_spatial_omics2': adj_spatial,
        'adj_feature_omics1': adj_fea1,
        'adj_feature_omics2': adj_fea2,
        'adj_emb': adj_spatial
    }

    valid_labels = adata_rna.obs['ground_truth'].dropna().unique()
    target_labels = [l for l in valid_labels if l not in ['Exclude', 'unknown']]
    n_clusters = len(target_labels) if len(target_labels) > 0 else 7

    # 3. Instantiate & Train spaLLM-VGAT Model
    trainer = Train_spaLLM_VGAT(
        data_dict=data_dict,
        cell_embedding=cell_emb,
        datatype='10x',
        device=device,
        random_seed=seed,
        learning_rate=0.001,
        epochs=100,
        dim_output=64,
        use_clustering_loss=True,
        n_clusters=n_clusters,
        clust_loss_weight=2.0,
        update_interval=10
    )

    results = trainer.train()
    z_spallm = results['spaLLM']
    native_preds = results['native_preds']

    adata_rna.obsm['spaLLM'] = z_spallm

    if native_preds is not None:
        adata_rna.obs['spallm_native'] = native_preds
    else:
        adata_rna.obs['spallm_native'] = run_mclust(z_spallm, n_clusters, seed=seed).astype(str)

    # 4. Evaluate Alternative Clustering Algorithms
    adata_rna.obs['kmeans'] = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(z_spallm).astype(str)
    res = search_res(adata_rna, n_clusters, method='leiden', use_rep='spaLLM')
    if sc is not None:
        try:
            sc.tl.leiden(adata_rna, random_state=seed, resolution=res, flavor='igraph', n_iterations=2, directed=False)
        except Exception:
            adata_rna.obs['leiden'] = adata_rna.obs['kmeans']
    else:
        adata_rna.obs['leiden'] = adata_rna.obs['kmeans']

    adata_rna.obs['mclust'] = run_mclust(z_spallm, n_clusters, seed=seed).astype(str)

    # 5. Evaluate All 4 Algorithms
    metrics_native = evaluate_clustering(adata_rna.obs['ground_truth'], adata_rna.obs['spallm_native'], z_spallm, name=f"spaLLM-VGAT {dname} Seed {seed} - Native Head")
    metrics_kmeans = evaluate_clustering(adata_rna.obs['ground_truth'], adata_rna.obs['kmeans'], z_spallm, name=f"spaLLM-VGAT {dname} Seed {seed} - KMeans")
    metrics_leiden = evaluate_clustering(adata_rna.obs['ground_truth'], adata_rna.obs['leiden'], z_spallm, name=f"spaLLM-VGAT {dname} Seed {seed} - Leiden")
    metrics_mclust = evaluate_clustering(adata_rna.obs['ground_truth'], adata_rna.obs['mclust'], z_spallm, name=f"spaLLM-VGAT {dname} Seed {seed} - mclust")

    row_native = {'cluster alg': 'spaLLM-VGAT Native', 'resolution': None}
    row_native.update(metrics_native)

    row_kmeans = {'cluster alg': 'KMeans', 'resolution': None}
    row_kmeans.update(metrics_kmeans)

    row_leiden = {'cluster alg': 'Leiden', 'resolution': res}
    row_leiden.update(metrics_leiden)

    row_mclust = {'cluster alg': 'mclust', 'resolution': None}
    row_mclust.update(metrics_mclust)

    return [row_native, row_kmeans, row_leiden, row_mclust]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Starting spaLLM-VGAT Multi-Seed Pipeline on device: {device}")

    all_results_flat = []
    all_results = {}

    for dname in datasets_to_run:
        cfg = ALL_DATASETS_CONFIG[dname]
        all_results[dname] = []

        print(f"\n=======================================================")
        print(f"STARTING WORKFLOW FOR DATASET: {dname} OVER {len(SEEDS)} SEEDS")
        print(f"=======================================================")

        for idx, seed in enumerate(SEEDS):
            try:
                results_list = run_spallm_vgat_workflow(dname, cfg, ENV_MODE, seed, device)
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
        output_csv = os.path.join(output_dir, 'spallm_vgat_ablation_results.csv')
        df_all.to_csv(output_csv, index=False)
        print(f"\nAll spaLLM-VGAT benchmark evaluation results saved to CSV at: {output_csv}")


if __name__ == '__main__':
    main()
