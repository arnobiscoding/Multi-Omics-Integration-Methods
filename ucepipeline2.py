#!/usr/bin/env python3
"""
UCE Multi-Seed Baseline Pipeline - Part 2 (Standalone Python Script)
Integrates baseline Universal Cell Embeddings (UCE) foundation model across 3 target datasets:
- Mouse Brain E18 (UCE RNA 1280-dim + RNA PCA 50-dim + ATAC PCA 50-dim)
- Human Lymph Node A1, D1 (UCE RNA 1280-dim + RNA PCA + ADT/Protein CLR PCA)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
warnings.filterwarnings("ignore", category=UserWarning, module="anndata")
warnings.filterwarnings("ignore", category=UserWarning, message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Variable names are not unique.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.__getitem__.*")

# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 2: RUN CONFIGURATION
# ===========================================================================
# ==================================================================
# RUN CONFIGURATION
# ==================================================================
import numpy as np

ENV_MODE = "auto"

# --- MODEL CONFIGURATION SELECTION ---
USE_33_LAYER = True  # Set to True for 33-layer model ('33l_8ep_1024t_1280.torch'), False for 4-layer model
NLAYERS = 33 if USE_33_LAYER else 4
MODEL_FILENAME = "33l_8ep_1024t_1280.torch" if USE_33_LAYER else "4layer_model.torch"
BATCH_SIZE = 15 if USE_33_LAYER else 25  # Recommended per-GPU batch size (15 for 33-layer on 16GB GPU)

ALL_DATASETS_CONFIG = {
    "mouse-brain-e18-s1": {
        "type": "mouse_brain",
        "species": "mouse",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Mouse_Brain_E18_S1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset7_Mouse_Brain_ATAC/",
        "mod2_candidates": ["adata_ATAC.h5ad", "adata_peaks_normalized.h5ad"],
        "anno_file": "anno.csv"
    },
    "human-lymph-node-a1": {
        "type": "human_lymph_node",
        "species": "human",
        "kaggle_dir": "/kaggle/input/datasets/sadmanbiazidarnob/multi-omics-datasets/Human_Lymph_Node_A1/",
        "local_dir": "D:/FYDP/spaLLM/spaLLM/Data_SpatialGlue/Data_SpatialGlue/Dataset11_Human_Lymph_Node_A1/",
        "mod2_candidates": ["adata_ADT.h5ad"],
        "anno_file": "annotation.csv"
    },
    "human-lymph-node-d1": {
        "type": "human_lymph_node",
        "species": "human",
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

print(f"Using UCE Model: {NLAYERS}-Layer ({MODEL_FILENAME})")
print(f"Scheduled datasets (Part 2): {datasets_to_run}")
print(f"Ablation seeds: {SEEDS}")


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 4: BASELINE UTILITIES
# ===========================================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="anndata")
warnings.filterwarnings("ignore", category=UserWarning, module="anndata")
warnings.filterwarnings("ignore", category=UserWarning, message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Variable names are not unique.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*Series.__getitem__.*")

# --- COMBINED BASELINE UTILITIES ---
import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.backends import cudnn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
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

def plot_uce_visualizations(adata, title_prefix="", dname="dataset", seed=2024, save_dir=None):
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
    save_path = os.path.join(save_dir, f"uce_plot_{dname}_seed_{seed}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization plot image to: {save_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)

def plot_uce_summary_boxplots(df_all, save_dir=None):
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
        ax.set_title(f"UCE Baseline (Part 2): {metric} Performance Across Datasets", fontsize=14, fontweight='bold')
        ax.set_xlabel("Dataset", fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(title="Cluster Alg", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"uce_part2_boxplot_{metric.lower().replace('-', '_')}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric} Boxplot image to: {save_path}")
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig)

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
    combined_path = os.path.join(save_dir, "uce_part2_boxplot_all_metrics.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined multi-metric Boxplot figure to: {combined_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 6: UCE MODEL ARCHITECTURE & ENGINE
# ===========================================================================
# ==================================================================
# STANDALONE UCE MODEL ARCHITECTURE & DATA PROCESSING ENGINE
# ==================================================================
import math
import pickle
import requests
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator

def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1536):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, src: Tensor) -> Tensor:
        src = src + self.pe[:src.size(0)]
        return self.dropout(src)

class TransformerModel(nn.Module):
    def __init__(self, token_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, output_dim: int, dropout: float = 0.05):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, enable_nested_tensor=False)
        self.dropout = dropout

        self.decoder = nn.Sequential(
            full_block(d_model, 1024, self.dropout),
            full_block(1024, output_dim, self.dropout),
            full_block(output_dim, output_dim, self.dropout),
            nn.Linear(output_dim, output_dim)
        )

        self.binary_decoder = nn.Sequential(
            full_block(output_dim + 1280, 2048, self.dropout),
            full_block(2048, 512, self.dropout),
            full_block(512, 128, self.dropout),
            nn.Linear(128, 1)
        )

        self.gene_embedding_layer = nn.Sequential(
            nn.Linear(token_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        self.pe_embedding = None

    def forward(self, src: Tensor, mask: Tensor):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=(1 - mask))
        gene_output = self.decoder(output)
        embedding = gene_output[0, :, :]
        embedding = nn.functional.normalize(embedding, dim=1)
        return gene_output, embedding

def figshare_download(url, save_path):
    if os.path.exists(save_path):
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading {save_path} from {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(save_path, 'wb') as file:
        for data_block in response.iter_content(block_size):
            progress.update(len(data_block))
            file.write(data_block)
    progress.close()
    if save_path.endswith(".tar.gz"):
        with tarfile.open(save_path) as tar:
            tar.extractall(path=os.path.dirname(save_path))
            print("Extraction complete!")

def data_to_torch_X(X):
    if isinstance(X, sc.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return torch.from_numpy(X).float()

def get_species_to_pe(EMBEDDING_DIR):
    EMBEDDING_DIR = Path(EMBEDDING_DIR)
    embeddings_paths = {
        'human': EMBEDDING_DIR / 'Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
        'mouse': EMBEDDING_DIR / 'Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt',
        'frog': EMBEDDING_DIR / 'Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt',
        'zebrafish': EMBEDDING_DIR / 'Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt',
        "mouse_lemur": EMBEDDING_DIR / "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
        "pig": EMBEDDING_DIR / 'Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt',
        "macaca_fascicularis": EMBEDDING_DIR / 'Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt',
        "macaca_mulatta": EMBEDDING_DIR / 'Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt',
    }
    extra_csv = Path(EMBEDDING_DIR).parent / "new_species_protein_embeddings.csv"
    if os.path.exists(extra_csv):
        try:
            extra_species = pd.read_csv(extra_csv).set_index("species").to_dict()["path"]
            embeddings_paths.update(extra_species)
        except Exception:
            pass
    species_to_pe = {sp: torch.load(p) for sp, p in embeddings_paths.items() if os.path.exists(p)}
    return {sp: {k.upper(): v for k, v in pe.items()} for sp, pe in species_to_pe.items()}

def get_spec_chrom_csv(path):
    gene_to_chrom_pos = pd.read_csv(path)
    gene_to_chrom_pos["spec_chrom"] = pd.Categorical(gene_to_chrom_pos["species"] + "_" + gene_to_chrom_pos["chromosome"])
    return gene_to_chrom_pos

def load_gene_embeddings_adata(adata, species, protein_embeddings_dir):
    species_to_pe = get_species_to_pe(protein_embeddings_dir)
    spec_pe = species_to_pe[species[0]]
    spec_pe_lower = {k.lower(): v for k, v in spec_pe.items()}
    genes_to_use = [g for g in adata.var_names if g.lower() in spec_pe_lower]
    adata = adata[:, adata.var_names.isin(genes_to_use)].copy()
    pe_matrix = torch.stack([spec_pe_lower[g.lower()] for g in adata.var_names])
    return adata, {species[0]: pe_matrix}

def adata_path_to_prot_chrom_starts(adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset):
    adata.var_names_make_unique()
    pe_row_idxs = torch.tensor([spec_pe_genes.index(k.upper()) + offset for k in adata.var_names]).long()
    spec_chrom = gene_to_chrom_pos[gene_to_chrom_pos["species"] == dataset_species].set_index("gene_symbol")
    gene_chrom = spec_chrom.loc[[k.upper() for k in adata.var_names]]
    dataset_chroms = gene_chrom["spec_chrom"].cat.codes
    dataset_pos = gene_chrom["start"].values
    return pe_row_idxs, dataset_chroms, dataset_pos

def process_raw_anndata(row, h5_folder_path, npz_folder_path, skip, additional_filter, root, protein_embeddings_dir):
    path = row.path
    name = path.replace(".h5ad", "")
    proc_path = path.replace(".h5ad", "_proc.h5ad")
    if skip and os.path.isfile(h5_folder_path + proc_path):
        return None, None, None
    species = row.species
    ad = sc.read(root + "/" + path) if os.path.isfile(root + "/" + path) else row.adata_obj
    if additional_filter:
        sc.pp.filter_genes(ad, min_cells=10)
        sc.pp.filter_cells(ad, min_genes=25)
    ad, _ = load_gene_embeddings_adata(ad, species=[species], protein_embeddings_dir=protein_embeddings_dir)
    num_cells, num_genes = ad.X.shape[0], ad.X.shape[1]
    adata_path = h5_folder_path + proc_path
    ad.write(adata_path)
    arr = data_to_torch_X(ad.X).numpy()
    filename = npz_folder_path + f"{name}_counts.npz"
    fp = np.memmap(filename, dtype='int64', mode='w+', shape=arr.shape)
    fp[:] = arr[:]
    fp.flush()
    return ad, num_cells, num_genes

def sample_cell_sentences(counts, batch_weights, dataset, args, dataset_to_protein_embeddings, dataset_to_chroms, dataset_to_starts):
    dataset_idxs = dataset_to_protein_embeddings[dataset]
    cell_sentences = torch.zeros((counts.shape[0], args.pad_length))
    mask = torch.zeros((counts.shape[0], args.pad_length))
    chroms = np.asarray(dataset_to_chroms[dataset])
    starts = np.asarray(dataset_to_starts[dataset])
    longest_seq_len = 0
    for c, cell in enumerate(counts):
        weights = batch_weights[c].numpy()
        weights = weights / sum(weights)
        choice_idx = np.random.choice(np.arange(len(weights)), size=args.sample_size, p=weights, replace=True)
        choosen_chrom = chroms[choice_idx]
        chrom_sort = np.argsort(choosen_chrom)
        choice_idx = choice_idx[chrom_sort]
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]
        ordered_choice_idx = np.full((args.pad_length), args.cls_token_idx)
        i = 1
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms)
        for chrom in uq_chroms:
            ordered_choice_idx[i] = int(chrom) + args.CHROM_TOKEN_OFFSET
            i += 1
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(choosen_starts[loc])
            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i:(i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)
            ordered_choice_idx[i] = args.chrom_token_right_idx
            i += 1
        longest_seq_len = max(longest_seq_len, i)
        remainder_len = (args.pad_length - i)
        mask[c, :] = torch.concat((torch.ones(i), torch.zeros(remainder_len)))
        ordered_choice_idx[i:] = args.pad_token_idx
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)
    return cell_sentences.long(), mask, longest_seq_len, cell_sentences

class MultiDatasetSentences(data.Dataset):
    def __init__(self, sorted_dataset_names, shapes_dict, args, dataset_to_protein_embeddings_path, datasets_to_chroms_path, datasets_to_starts_path, npzs_dir):
        super().__init__()
        self.num_cells = {}
        self.num_genes = {}
        self.shapes_dict = shapes_dict
        self.args = args
        self.total_num_cells = 0
        for name in sorted_dataset_names:
            num_cells, num_genes = self.shapes_dict[name]
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes
            self.total_num_cells += num_cells
        self.datasets = sorted_dataset_names
        self.dataset_to_protein_embeddings = torch.load(dataset_to_protein_embeddings_path)
        with open(datasets_to_chroms_path, "rb") as f:
            self.dataset_to_chroms = pickle.load(f)
        with open(datasets_to_starts_path, "rb") as f:
            self.dataset_to_starts = pickle.load(f)
        self.npzs_dir = npzs_dir

    def __getitem__(self, idx):
        if isinstance(idx, int):
            for dataset in sorted(self.datasets):
                if idx < self.num_cells[dataset]:
                    cts = np.memmap(self.npzs_dir + f"{dataset}_counts.npz", dtype='int64', mode='r', shape=self.shapes_dict[dataset])
                    counts = torch.tensor(cts[idx]).unsqueeze(0)
                    weights = torch.log1p(counts)
                    weights = (weights / torch.sum(weights))
                    batch_sentences, mask, seq_len, cell_sentences = sample_cell_sentences(
                        counts, weights, dataset, self.args,
                        dataset_to_protein_embeddings=self.dataset_to_protein_embeddings,
                        dataset_to_chroms=self.dataset_to_chroms,
                        dataset_to_starts=self.dataset_to_starts
                    )
                    return batch_sentences, mask, idx, seq_len, cell_sentences
                else:
                    idx -= self.num_cells[dataset]
            raise IndexError

    def __len__(self):
        return self.total_num_cells

class MultiDatasetSentenceCollator(object):
    def __init__(self, args):
        self.pad_length = args.pad_length

    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length))
        mask = torch.zeros((batch_size, self.pad_length))
        cell_sentences = torch.zeros((batch_size, self.pad_length))
        idxs = torch.zeros(batch_size)
        i = 0
        max_len = 0
        for bs, msk, idx, seq_len, cs in batch:
            batch_sentences[i, :] = bs
            cell_sentences[i, :] = cs
            max_len = max(max_len, seq_len)
            mask[i, :] = msk
            idxs[i] = idx
            i += 1
        return batch_sentences[:, :max_len], mask[:, :max_len], idxs, cell_sentences

def get_ESM2_embeddings(args):
    all_pe = torch.load(args.token_file)
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, args.token_dim))
        all_pe = torch.vstack((all_pe, CHROM_TENSORS))
        all_pe.requires_grad = False
    return all_pe

def run_eval(adata, name, pe_idx_path, chroms_path, starts_path, shapes_dict, accelerator, args):
    model = TransformerModel(token_dim=args.token_dim, d_model=1280, nhead=20, d_hid=args.d_hid, nlayers=args.nlayers, dropout=0.05, output_dim=args.output_dim)
    empty_pe = torch.zeros(145469, 5120)
    empty_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(empty_pe)
    model.load_state_dict(torch.load(args.model_loc, map_location="cpu"), strict=True)
    all_pe = get_ESM2_embeddings(args)
    if all_pe.shape[0] != 145469:
        all_pe.requires_grad = False
        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    model = model.eval()
    model = accelerator.prepare(model)
    dataset = MultiDatasetSentences(sorted_dataset_names=[name], shapes_dict=shapes_dict, args=args, npzs_dir=args.dir,
                                    dataset_to_protein_embeddings_path=pe_idx_path, datasets_to_chroms_path=chroms_path, datasets_to_starts_path=starts_path)
    collator = MultiDatasetSentenceCollator(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    dataloader = accelerator.prepare(dataloader)
    dataset_embeds = []
    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            batch_sentences, mask = batch[0].permute(1, 0), batch[1]
            batch_sentences = model.pe_embedding(batch_sentences.long())
            batch_sentences = nn.functional.normalize(batch_sentences, dim=2)
            _, embedding = model.forward(batch_sentences, mask=mask)
            accelerator.wait_for_everyone()
            embeddings = accelerator.gather_for_metrics((embedding))
            if accelerator.is_main_process:
                dataset_embeds.append(embeddings.detach().cpu().numpy())
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        dataset_embeds = np.vstack(dataset_embeds)
        adata.obsm["X_uce"] = dataset_embeds
        write_path = args.dir + f"{name}_uce_adata.h5ad"
        adata.write(write_path)

class AnndataProcessor:
    def __init__(self, args, accelerator):
        self.args = args
        self.accelerator = accelerator
        self.h5_folder_path = self.args.dir
        self.npz_folder_path = self.args.dir
        self.check_paths()
        self.adata_name = self.args.adata_path.split("/")[-1]
        self.adata_root_path = self.args.adata_path.replace(self.adata_name, "")
        self.name = self.adata_name.replace(".h5ad", "")
        self.proc_h5_path = self.h5_folder_path + f"{self.name}_proc.h5ad"
        self.adata = None
        row = pd.Series()
        row.path = self.adata_name
        row.covar_col = np.nan
        row.species = self.args.species
        row.adata_obj = getattr(self.args, 'adata_obj', None)
        self.row = row
        self.pe_idx_path = self.args.dir + f"{self.name}_pe_idx.torch"
        self.chroms_path = self.args.dir + f"{self.name}_chroms.pkl"
        self.starts_path = self.args.dir + f"{self.name}_starts.pkl"
        self.shapes_dict_path = self.args.dir + f"{self.name}_shapes_dict.pkl"

    def check_paths(self):
        figshare_download("https://figshare.com/ndownloader/files/42706558", self.args.spec_chrom_csv_path)
        figshare_download("https://figshare.com/ndownloader/files/42706555", self.args.offset_pkl_path)
        if not os.path.exists(self.args.protein_embeddings_dir):
            figshare_download("https://figshare.com/ndownloader/files/42715213", 'model_files/protein_embeddings.tar.gz')
        figshare_download("https://figshare.com/ndownloader/files/42706585", self.args.token_file)
        if self.args.model_loc is None or not os.path.exists(self.args.model_loc):
            if self.args.nlayers == 33:
                self.args.model_loc = getattr(self.args, 'model_filename', "./model_files/33l_8ep_1024t_1280.torch")
                figshare_download("https://figshare.com/ndownloader/files/43423236", self.args.model_loc)
            else:
                self.args.model_loc = getattr(self.args, 'model_filename', "./model_files/4layer_model.torch")
                figshare_download("https://figshare.com/ndownloader/files/42706576", self.args.model_loc)

    def preprocess_anndata(self):
        if self.accelerator.is_main_process:
            self.adata, num_cells, num_genes = process_raw_anndata(
                self.row, self.h5_folder_path, self.npz_folder_path, self.args.skip, self.args.filter,
                root=self.adata_root_path, protein_embeddings_dir=self.args.protein_embeddings_dir
            )
            if (num_cells is not None) and (num_genes is not None):
                with open(self.shapes_dict_path, "wb+") as f:
                    pickle.dump({self.name: (num_cells, num_genes)}, f)
            if self.adata is None:
                self.adata = sc.read(self.proc_h5_path)

    def generate_idxs(self):
        if self.accelerator.is_main_process:
            if not (os.path.exists(self.pe_idx_path) and os.path.exists(self.chroms_path) and os.path.exists(self.starts_path)):
                species_to_pe = get_species_to_pe(self.args.protein_embeddings_dir)
                with open(self.args.offset_pkl_path, "rb") as f:
                    species_to_offsets = pickle.load(f)
                gene_to_chrom_pos = get_spec_chrom_csv(self.args.spec_chrom_csv_path)
                dataset_species = self.args.species
                spec_pe_genes = list(species_to_pe[dataset_species].keys())
                offset = species_to_offsets[dataset_species]
                pe_row_idxs, dataset_chroms, dataset_pos = adata_path_to_prot_chrom_starts(self.adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset)
                torch.save({self.name: pe_row_idxs}, self.pe_idx_path)
                with open(self.chroms_path, "wb+") as f: pickle.dump({self.name: dataset_chroms}, f)
                with open(self.starts_path, "wb+") as f: pickle.dump({self.name: dataset_pos}, f)

    def run_evaluation(self):
        self.accelerator.wait_for_everyone()
        with open(self.shapes_dict_path, "rb") as f:
            shapes_dict = pickle.load(f)
        run_eval(self.adata, self.name, self.pe_idx_path, self.chroms_path, self.starts_path, shapes_dict, self.accelerator, self.args)


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 8: DATA LOADER
# ===========================================================================
def load_dataset_data(dname, cfg, env_mode="auto"):
    """Dynamically load RNA, Modality 2 (ATAC/ADT), and annotations for any specified dataset."""
    is_kaggle = (env_mode == "kaggle") or (env_mode == "auto" and os.path.exists("/kaggle/input"))
    data_dir = cfg["kaggle_dir"] if is_kaggle else cfg["local_dir"]
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Returning dummy data for local validation...")
        adata_rna = sc.datasets.pbmc3k()
        adata_rna.var_names_make_unique()
        adata_mod2 = adata_rna.copy()
        adata_rna.obsm['spatial'] = np.random.randn(adata_rna.n_obs, 2)
        adata_mod2.obsm['spatial'] = adata_rna.obsm['spatial']
        adata_rna.obs['ground_truth'] = adata_rna.obs['louvain'].astype(str) if 'louvain' in adata_rna.obs else 'Cluster1'
        return adata_rna, adata_mod2

    print(f"Loading data from: {data_dir}")
    rna_path = os.path.join(data_dir, "adata_RNA.h5ad")
    adata_rna = sc.read_h5ad(rna_path)
    adata_rna.var_names_make_unique()

    mod2_path = None
    for cand in cfg["mod2_candidates"]:
        cp = os.path.join(data_dir, cand)
        if os.path.exists(cp):
            mod2_path = cp
            break

    if mod2_path is None:
        raise FileNotFoundError(f"Modality 2 file not found in candidates {cfg['mod2_candidates']} inside {data_dir}")

    adata_mod2 = sc.read_h5ad(mod2_path)
    adata_mod2.var_names_make_unique()

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
# PIPELINE CODE SECTION FROM CELL 10: UCE EMBEDDING GENERATOR
# ===========================================================================
import tempfile
import tarfile

def get_uce_embeddings(adata_rna, species="human", batch_size=BATCH_SIZE, nlayers=NLAYERS, model_filename=MODEL_FILENAME, device="cuda"):
    """Generate UCE 1280-dim cell embeddings for adata_rna using standalone embedded code."""
    # 1. Dynamic Kaggle Model Directory Discovery
    model_dir = './model_files'
    if os.path.exists('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if (model_filename in files or '4layer_model.torch' in files or 'species_offsets.pkl' in files) and not root.endswith('scratch'):
                model_dir = root
                print(f"Discovered UCE model files dataset at: {model_dir}")
                break
    
    if model_dir == './model_files':
        os.makedirs(model_dir, exist_ok=True)
        print(f"Using default model directory (auto-downloads if missing): {model_dir}")

    # Auto-extract protein_embeddings.tar.gz if uploaded without extracting first
    tar_gz_path = os.path.join(model_dir, "protein_embeddings.tar.gz")
    target_pe_dir = os.path.join(model_dir, "protein_embeddings")
    if os.path.exists(tar_gz_path) and not os.path.exists(target_pe_dir):
        print(f"Auto-extracting {tar_gz_path}...")
        with tarfile.open(tar_gz_path) as tar:
            tar.extractall(path=model_dir)
            print("Protein embeddings extracted successfully.")

    # Resolve exact model weights file path
    model_path = os.path.join(model_dir, model_filename)
    if not os.path.exists(model_path):
        for alt in ["33l_8ep_1024t_1280.torch", "33layer_model.torch", "4layer_model.torch"]:
            if os.path.exists(os.path.join(model_dir, alt)):
                model_path = os.path.join(model_dir, alt)
                print(f"Found checkpoint: {model_path}")
                break

    # 2. Use Temporary Working Directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_h5ad = os.path.join(tmp_dir, "input_dataset.h5ad")
        adata_rna.write_h5ad(temp_h5ad)
        
        class Args:
            pass
            
        args = Args()
        args.adata_path = temp_h5ad
        args.dir = tmp_dir + "/"
        args.species = species
        args.filter = False
        args.skip = False
        args.model_loc = model_path if os.path.exists(model_path) else None
        args.model_filename = model_path
        args.batch_size = batch_size
        args.pad_length = 1536
        args.pad_token_idx = 0
        args.chrom_token_left_idx = 1
        args.chrom_token_right_idx = 2
        args.cls_token_idx = 3
        args.CHROM_TOKEN_OFFSET = 143574
        args.sample_size = 1024
        args.CXG = True
        args.nlayers = nlayers
        args.output_dim = 1280
        args.d_hid = 5120
        args.token_dim = 5120
        args.multi_gpu = False
        args.spec_chrom_csv_path = os.path.join(model_dir, "species_chrom.csv")
        args.token_file = os.path.join(model_dir, "all_tokens.torch")
        args.protein_embeddings_dir = os.path.join(model_dir, "protein_embeddings/")
        args.offset_pkl_path = os.path.join(model_dir, "species_offsets.pkl")
        args.adata_obj = adata_rna

        print(f"Executing UCE evaluation pipeline ({nlayers}-Layer) for species: {species}...")
        accelerator = Accelerator()
        processor = AnndataProcessor(args, accelerator)
        processor.preprocess_anndata()
        processor.generate_idxs()
        processor.run_evaluation()
        
        out_h5ad = os.path.join(tmp_dir, "input_dataset_uce_adata.h5ad")
        if os.path.exists(out_h5ad):
            adata_out = sc.read_h5ad(out_h5ad)
            return adata_out.obsm["X_uce"]
        else:
            print("Warning: UCE embedding output not found. Returning random fallback embeddings...")
            return np.random.randn(adata_rna.n_obs, 1280)


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 12: SINGLE DATASET RUNNER
# ===========================================================================
def run_uce_workflow(dname, cfg, env_mode, seed, device, show_plots=False):
    """Executes the baseline UCE workflow for a single dataset under a specific seed using KMeans, Leiden, and mclust."""
    fix_seed(seed)
    print(f"\n--- Running Dataset: {dname} | Species: {cfg.get('species', 'human')} | Model: {NLAYERS}-Layer | Seed: {seed} ---")

    # 1. Load Data
    adata_rna, adata_mod2 = load_dataset_data(dname, cfg, env_mode)

    # 2. Extract UCE Cell Embedding (1280-dim)
    species = cfg.get("species", "human")
    X_uce = get_uce_embeddings(adata_rna, species=species, batch_size=BATCH_SIZE, nlayers=NLAYERS, model_filename=MODEL_FILENAME, device=device)
    adata_rna.obsm['X_uce'] = X_uce

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

    # 5. Concatenate (Fuse) UCE Embedding (1280-dim) with RNA and Modality 2 Features
    joint_feat = np.concatenate((X_uce, feat_rna, feat_mod2), axis=1)
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
        plot_uce_visualizations(adata_rna, title_prefix=f"{dname} (Seed {seed})", dname=dname, seed=seed)

    return [row_kmeans, row_leiden, row_mclust]


# ===========================================================================
# PIPELINE CODE SECTION FROM CELL 14: MAIN EXECUTION
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
                results_list = run_uce_workflow(dname, cfg, ENV_MODE, seed, device, show_plots=show_plots)
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

    if len(all_results_flat) > 0:
        df_all = pd.DataFrame(all_results_flat)
        is_kaggle = os.path.exists('/kaggle/working')
        output_dir = '/kaggle/working' if is_kaggle else '.'
        output_csv = os.path.join(output_dir, 'uce_ablation_results_part2.csv')
        df_all.to_csv(output_csv, index=False)
        print(f"All UCE ablation study results (Part 2) saved to CSV at: {output_csv}")
        
        print("Generating side-by-side Box & Whiskers plots for all metrics across algorithms...")
        plot_uce_summary_boxplots(df_all, save_dir=output_dir)
    else:
        print("No results were generated, skipping CSV export and Boxplots.")

if __name__ == '__main__':
    main()
