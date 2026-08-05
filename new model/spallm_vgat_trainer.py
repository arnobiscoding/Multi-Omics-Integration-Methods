"""
spaLLM-VGAT Trainer & FACT-style Spatial Clustering Engine
=========================================================

Implements the exact spaLLM 3-Stage training workflow:
- Phase 1 (0 to 40% epochs): Multi-graph reconstruction & cross-correlation warm-up.
- Phase 2 (40% to 60% epochs): Freeze VGAT encoders, generate pseudo-labels via mclust/KMeans,
  apply spatial neighborhood label smoothing (majority voting), one-hot encode targets,
  and optimize KL divergence clustering loss (F.kl_div). Update pseudo-labels every 20 epochs.
- Phase 3 (60% to 100% epochs): Unfreeze VGAT encoders and train jointly.
"""

import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
from typing import Dict, Any, Tuple, Optional

try:
    from .spallm_vgat_model import VGATEncodingNetwork, init_weights
except ImportError:
    from spallm_vgat_model import VGATEncodingNetwork, init_weights


def spatial_label_smoothing(adj_spatial: Any, labels: np.ndarray) -> np.ndarray:
    """Refine spot cluster labels using spatial neighborhood majority voting."""
    if torch.is_tensor(adj_spatial):
        adj_mat = adj_spatial.to_dense().cpu().numpy() if adj_spatial.is_sparse else adj_spatial.cpu().numpy()
    elif sp.issparse(adj_spatial):
        adj_mat = adj_spatial.toarray()
    else:
        adj_mat = np.array(adj_spatial)

    N_spots = len(labels)
    smoothed_labels = labels.copy()

    for i in range(N_spots):
        neighbors = np.where(adj_mat[i] > 0)[0]
        neighbors = [j for j in neighbors if j != i]
        if len(neighbors) > 0:
            neigh_labels = [labels[j] for j in neighbors]
            vals, counts = np.unique(neigh_labels, return_counts=True)
            max_idx = np.argmax(counts)
            majority_label = vals[max_idx]
            majority_count = counts[max_idx]
            if majority_count >= len(neighbors) / 2:
                smoothed_labels[i] = majority_label

    return smoothed_labels


def run_mclust(data_matrix: np.ndarray, n_clusters: int, seed: int = 2024, max_dims: int = 30) -> np.ndarray:
    """Run mclust clustering via rpy2 with PCA reduction, falling back to KMeans if rpy2 is unavailable."""
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
        return labels
    except Exception:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(data_mat).astype(int)


def add_gaussian_noise(matrix: torch.Tensor, mean: float = 0.0, std: float = 0.001) -> torch.Tensor:
    """Apply Gaussian noise perturbation."""
    noise = torch.normal(mean=mean, std=std, size=matrix.size()).to(matrix.device)
    return matrix + noise


def preprocess_graph(adj: np.ndarray) -> torch.Tensor:
    """Symmetrize and row-normalize adjacency matrix for GNN/VGAT input."""
    adj_sp = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj_sp.sum(1))
    degree_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    norm_adj = adj_sp.dot(degree_inv_sqrt).T.dot(degree_inv_sqrt).tocoo().astype(np.float32)

    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    return torch.sparse.FloatTensor(indices, torch.from_numpy(norm_adj.data), torch.Size(norm_adj.shape))


def transform_adjacent_matrix(adj_df: pd.DataFrame) -> sp.coo_matrix:
    n_spot = adj_df['x'].max() + 1
    return coo_matrix((adj_df['value'], (adj_df['x'], adj_df['y'])), shape=(n_spot, n_spot))


class Train_spaLLM_VGAT:
    """
    3-Stage Training Engine for spaLLM-VGAT model.
    """
    def __init__(
        self,
        data_dict: Dict[str, Any],
        cell_embedding: np.ndarray,
        datatype: str = '10x',
        device: torch.device = torch.device('cpu'),
        random_seed: int = 2024,
        learning_rate: float = 0.0001,
        epochs: int = 600,
        dim_output: int = 64,
        weight_factors: Optional[list] = None,
        use_clustering_loss: bool = True,
        n_clusters: int = 7,
        clust_loss_weight: float = 2.0,
        update_interval: int = 20
    ):
        self.device = device
        self.data = data_dict
        self.embedding = torch.from_numpy(cell_embedding).float().to(device)
        self.datatype = datatype
        self.random_seed = random_seed
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.dim_output = dim_output
        self.weight_factors = weight_factors or [5, 5, 1, 10, 10, 10]
        self.use_clustering_loss = use_clustering_loss
        self.n_clusters = n_clusters
        self.clust_loss_weight = clust_loss_weight
        self.update_interval = update_interval

        self._init_adj_and_features()
        self.loss_history = []
        self._adjust_hyperparameters()

    def _init_adj_and_features(self):
        """Prepare spatial and feature graph adjacency matrices and feature tensors."""
        def _process_adj(adj):
            if isinstance(adj, pd.DataFrame):
                adj = transform_adjacent_matrix(adj).toarray()
            elif sp.issparse(adj):
                adj = adj.toarray()
            adj = adj + adj.T
            adj = np.where(adj > 1, 1, adj)
            return preprocess_graph(adj)

        self.adj_spatial_omics1 = _process_adj(self.data['adj_spatial_omics1']).to(self.device)
        self.adj_spatial_omics2 = _process_adj(self.data['adj_spatial_omics2']).to(self.device)
        self.adj_feature_omics1 = _process_adj(self.data['adj_feature_omics1']).to(self.device)
        self.adj_feature_omics2 = _process_adj(self.data['adj_feature_omics2']).to(self.device)

        if 'adj_emb' in self.data:
            self.adj_emb = _process_adj(self.data['adj_emb']).to(self.device)
        else:
            self.adj_emb = self.adj_spatial_omics1

        self.features_omics1 = torch.FloatTensor(self.data['features_omics1']).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.data['features_omics2']).to(self.device)
        self.dim_input1, self.dim_input2 = self.features_omics1.shape[1], self.features_omics2.shape[1]

    def _adjust_hyperparameters(self):
        if self.datatype == 'SPOTS':
            self.epochs, self.weight_factors = 600, [1, 5, 1, 1, 5, 5]
        elif self.datatype == '10x':
            self.epochs, self.weight_factors = 200, [5, 5, 1, 10, 10, 10]
        elif self.datatype == 'Spatial-epigenome-transcriptome':
            self.epochs, self.weight_factors = 1600, [1, 5, 1, 1, 10, 10]

    def _add_noise(self) -> Tuple[torch.Tensor, torch.Tensor]:
        features_omics1_noisy = add_gaussian_noise(self.features_omics1, mean=0, std=0.1)
        embedding_noisy = add_gaussian_noise(self.embedding, mean=0, std=0.01)
        return features_omics1_noisy, embedding_noisy

    def _calculate_losses(self, results: Dict[str, torch.Tensor], target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss_recon_omics1 = F.mse_loss(self.features_omics1, results['emb_recon_omics1'])
        loss_recon_omics2 = F.mse_loss(self.features_omics2, results['emb_recon_omics2'])
        loss_rec_es = F.mse_loss(self.embedding, results['emb_recon_spa'])
        loss_rec_ef = F.mse_loss(self.embedding, results['emb_recon_fea'])
        loss_corr_omics1 = F.mse_loss(results['emb_latent_omics1'], results['emb_cross1'])
        loss_corr_omics2 = F.mse_loss(results['emb_latent_omics2'], results['emb_cross2'])

        loss_spallm = (self.weight_factors[0] * loss_recon_omics1 +
                       self.weight_factors[1] * loss_recon_omics2 +
                       self.weight_factors[2] * loss_corr_omics1 +
                       self.weight_factors[3] * loss_corr_omics2 +
                       self.weight_factors[4] * loss_rec_es +
                       self.weight_factors[5] * loss_rec_ef)

        if target_labels is not None and results['cluster_probs'] is not None:
            eps = 1e-8
            log_probs = results['cluster_probs'].clamp_min(eps).log()
            loss_clust = F.kl_div(log_probs, target_labels.clamp_min(eps), reduction='batchmean')
            loss = loss_spallm + self.clust_loss_weight * loss_clust
        else:
            loss = loss_spallm

        return loss

    def train(self, epochs: Optional[int] = None) -> Dict[str, np.ndarray]:
        epochs = epochs or self.epochs
        n_clust_param = self.n_clusters if self.use_clustering_loss else None
        self.model = VGATEncodingNetwork(
            dim_in_omics1=self.dim_input1,
            dim_out_omics1=self.dim_output,
            dim_in_omics2=self.dim_input2,
            dim_out_omics2=self.dim_output,
            n_clusters=n_clust_param
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        phase1_end = int(0.4 * epochs)
        phase2_end = int(0.6 * epochs)
        current_target_labels = None

        print(f"[*] Training spaLLM-VGAT for {epochs} epochs (Phase 1: 0-{phase1_end}, Phase 2: {phase1_end}-{phase2_end}, Phase 3: {phase2_end}-{epochs})...")

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Phase 2 & 3 Target Pseudo-Label Generation
            if self.use_clustering_loss and epoch >= phase1_end:
                if epoch % self.update_interval == 0 or current_target_labels is None:
                    self.model.eval()
                    with torch.no_grad():
                        eval_res = self.model(
                            self.features_omics1, self.features_omics2,
                            self.adj_spatial_omics1, self.adj_feature_omics1,
                            self.adj_spatial_omics2, self.adj_feature_omics2,
                            self.embedding, self.adj_emb
                        )
                        z_emb = F.normalize(eval_res['emb_latent_combined'], p=2).cpu().numpy()

                    raw_pseudo = run_mclust(z_emb, n_clusters=self.n_clusters, seed=self.random_seed)

                    # Apply spatial neighborhood label smoothing
                    smoothed_pseudo = spatial_label_smoothing(self.adj_spatial_omics1, raw_pseudo)
                    le = LabelEncoder()
                    smoothed_pseudo_encoded = le.fit_transform(smoothed_pseudo)
                    current_target_labels = F.one_hot(torch.tensor(smoothed_pseudo_encoded).long(), num_classes=self.n_clusters).float().to(self.device)
                    self.model.train()

            # Phase 2: Freeze VGAT encoders for Omics 1 and Omics 2
            if self.use_clustering_loss and phase1_end <= epoch < phase2_end:
                for param in self.model.encoder_omics1.parameters():
                    param.requires_grad = False
                for param in self.model.encoder_omics2.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.encoder_omics1.parameters():
                    param.requires_grad = True
                for param in self.model.encoder_omics2.parameters():
                    param.requires_grad = True

            if random.random() < 0.5:
                features_omics1, embedding = self._add_noise()
            else:
                features_omics1, embedding = self.features_omics1, self.embedding

            results = self.model(
                features_omics1, self.features_omics2,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2,
                embedding, self.adj_emb
            )

            active_target = current_target_labels if (self.use_clustering_loss and epoch >= phase1_end) else None
            loss = self._calculate_losses(results, target_labels=active_target)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

        return self._evaluate_model()

    def _evaluate_model(self) -> Dict[str, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            results = self.model(
                self.features_omics1, self.features_omics2,
                self.adj_spatial_omics1, self.adj_feature_omics1,
                self.adj_spatial_omics2, self.adj_feature_omics2,
                self.embedding, self.adj_emb
            )

            if results['cluster_probs'] is not None:
                native_preds = torch.argmax(results['cluster_probs'], dim=1).cpu().numpy().astype(str)
            else:
                native_preds = None

        return {
            'emb_latent_omics1': F.normalize(results['emb_latent_omics1'], p=2).cpu().numpy(),
            'emb_latent_omics2': F.normalize(results['emb_latent_omics2'], p=2).cpu().numpy(),
            'spaLLM': F.normalize(results['emb_latent_combined'], p=2).cpu().numpy(),
            'native_preds': native_preds
        }
