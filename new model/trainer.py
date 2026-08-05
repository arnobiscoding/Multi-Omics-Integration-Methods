"""
Multi-Phase Training Execution Engine
=====================================

Implements the 4-Phase Protocol:
1. Phase 1: Reconstruction & Contrastive Warm-Up (100 epochs, DEC disabled, spatial InfoNCE)
2. Phase 2: Center Initialization (Freeze network, compute mu_f, run K-Means++, assign centroids C)
3. Phase 3: Joint Fine-Tuning (Unfreeze network, optimize L_total, update P target dist every 5 epochs)
4. Phase 4: Spatial Early Stopping (Subsample 5% spots every 10 epochs, compute Silhouette + Moran's I, save best checkpoint)
"""

import copy
import random
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm

try:
    from .model import VGAT_PoE_DEC
    from .clustering_and_contrastive import DECHead
    from .metrics import compute_composite_spatial_metric
except ImportError:
    from model import VGAT_PoE_DEC
    from clustering_and_contrastive import DECHead
    from metrics import compute_composite_spatial_metric


class SpatialOmicsTrainer:
    """
    4-Phase Execution Engine for Spatial Multi-Omics Pipeline.
    """
    def __init__(
        self,
        model: VGAT_PoE_DEC,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 100,
        finetune_epochs: int = 100,
        p_update_interval: int = 5,
        early_stop_eval_interval: int = 10,
        subsample_ratio: float = 0.05,
        alpha_init: float = 0.1,
        gamma_init: float = 0.1,
        alpha_max: float = 1.0,
        gamma_max: float = 1.0,
        lambda1_spatial: float = 0.5,
        lambda2_spatial: float = 0.5,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.device = device

        self.warmup_epochs = warmup_epochs
        self.finetune_epochs = finetune_epochs
        self.p_update_interval = p_update_interval
        self.early_stop_eval_interval = early_stop_eval_interval
        self.subsample_ratio = subsample_ratio

        self.alpha_init = alpha_init
        self.gamma_init = gamma_init
        self.alpha_max = alpha_max
        self.gamma_max = gamma_max

        self.lambda1 = lambda1_spatial
        self.lambda2 = lambda2_spatial

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.best_model_weights = None
        self.best_composite_score = -np.inf
        self.best_epoch = -1

    def fit(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full training protocol across all 4 phases.
        """
        x_rna = data_dict["x_rna"].to(self.device)
        x_adt = data_dict["x_adt"].to(self.device)
        edge_index = data_dict["edge_index"].to(self.device)
        raw_rna = data_dict["raw_counts_rna"].to(self.device)
        coords = data_dict["coords"].to(self.device)
        num_nodes = data_dict["num_nodes"]

        print("==========================================================================")
        print(" PHASE 1: Reconstruction & Spatial Contrastive Warm-Up (100 Epochs)")
        print("==========================================================================")
        self.model.train()
        for epoch in range(1, self.warmup_epochs + 1):
            self.optimizer.zero_grad()
            outputs = self.model(x_rna, x_adt, edge_index, raw_rna_counts=raw_rna)
            loss, loss_dict = self.model.compute_losses(
                outputs=outputs,
                x_rna_raw=raw_rna,
                x_adt=x_adt,
                edge_index=edge_index,
                phase=1,
                alpha=self.alpha_init
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            if epoch % 20 == 0 or epoch == self.warmup_epochs:
                print(f"[Phase 1 Warm-up | Epoch {epoch:03d}/{self.warmup_epochs}] "
                      f"Loss: {loss.item():.4f} (Recon: {loss_dict['l_recon'].item():.4f}, "
                      f"Contrast: {loss_dict['l_contrast'].item():.4f})")

        print("\n==========================================================================")
        print(" PHASE 2: DEC Center Initialization via K-Means++ on Stabilized Latent mu_f")
        print("==========================================================================")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_rna, x_adt, edge_index, raw_rna_counts=raw_rna)
            mu_f = outputs["mu_f"]

            # Initialize DEC cluster centers using K-Means++ algorithm
            self.model.dec_head.init_centers_kmeans(mu_f)
            print(f"[Phase 2] Successfully initialized {self.model.num_clusters} DEC cluster centers via K-Means++.")

        print("\n==========================================================================")
        print(" PHASE 3 & 4: Joint Fine-Tuning & Spatial Early Stopping Subsampling")
        print("==========================================================================")
        self.model.train()

        p_target = None
        self.best_composite_score = -np.inf
        self.best_model_weights = copy.deepcopy(self.model.state_dict())

        for epoch in range(1, self.finetune_epochs + 1):
            # Compute current loss scale factors (gradual ramp up)
            progress = epoch / self.finetune_epochs
            alpha = self.alpha_init + progress * (self.alpha_max - self.alpha_init)
            gamma = self.gamma_init + progress * (self.gamma_max - self.gamma_init)

            # Update target distribution P once every 5 epochs
            if (epoch - 1) % self.p_update_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(x_rna, x_adt, edge_index, raw_rna_counts=raw_rna)
                    q = outputs["q"]
                    p_target = DECHead.target_distribution(q)
                self.model.train()

            self.optimizer.zero_grad()
            outputs = self.model(x_rna, x_adt, edge_index, raw_rna_counts=raw_rna)

            loss, loss_dict = self.model.compute_losses(
                outputs=outputs,
                x_rna_raw=raw_rna,
                x_adt=x_adt,
                edge_index=edge_index,
                p_target=p_target,
                phase=3,
                alpha=alpha,
                gamma=gamma
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Phase 4: Spatial Early Stopping evaluation every 10 epochs on 5% spot subsample
            if epoch % self.early_stop_eval_interval == 0 or epoch == self.finetune_epochs:
                self.model.eval()
                with torch.no_grad():
                    eval_outputs = self.model(x_rna, x_adt, edge_index, raw_rna_counts=raw_rna)
                    z_f = eval_outputs["z_f"]
                    q = eval_outputs["q"]
                    pred_labels = torch.argmax(q, dim=1)

                    # Subsample 5% of spots for fast spatial evaluation
                    subsample_size = max(50, int(num_nodes * self.subsample_ratio))
                    sub_idx = np.random.choice(num_nodes, subsample_size, replace=False)

                    sub_z_f = z_f[sub_idx]
                    sub_pred = pred_labels[sub_idx]
                    sub_coords = coords[sub_idx]

                    composite, sil, moran = compute_composite_spatial_metric(
                        z_f=sub_z_f,
                        cluster_labels=sub_pred,
                        coords=sub_coords,
                        lambda1=self.lambda1,
                        lambda2=self.lambda2
                    )

                    print(f"[Phase 3 & 4 | Epoch {epoch:03d}/{self.finetune_epochs}] "
                          f"Loss: {loss.item():.4f} (DEC: {loss_dict['l_dec'].item():.4f}) | "
                          f"5% Subsample Composite Metric: {composite:.4f} (Sil: {sil:.4f}, Moran's I: {moran:.4f})")

                    # Checkpoint model if composite metric reaches new peak
                    if composite > self.best_composite_score:
                        self.best_composite_score = composite
                        self.best_model_weights = copy.deepcopy(self.model.state_dict())
                        self.best_epoch = epoch
                        print(f"   >>> Peak Composite Spatial Metric achieved ({composite:.4f})! Saved model weights.")

                self.model.train()

        # Load best model weights
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)
            print(f"\nTraining Complete. Loaded best model weights from epoch {self.best_epoch} "
                  f"with Peak Composite Metric: {self.best_composite_score:.4f}.")

        # Final evaluation pass
        self.model.eval()
        with torch.no_grad():
            final_outputs = self.model(x_rna, x_adt, edge_index, raw_rna_counts=raw_rna)
            z_f = final_outputs["z_f"]
            q = final_outputs["q"]
            predictions = torch.argmax(q, dim=1).cpu().numpy()

        results = {
            "model": self.model,
            "z_f": z_f.cpu().numpy(),
            "predictions": predictions,
            "best_composite_score": self.best_composite_score,
            "best_epoch": self.best_epoch,
            "final_outputs": final_outputs
        }
        return results
