"""
Top-Level Architecture Wrapper: VGAT_PoE_DEC
=============================================

Integrates Dual-Encoder VGAT, Product of Experts (PoE) Fusion, Dual Decoders (ZINB + MSE),
DEC Clustering Head, and Pseudo-Label Contrastive Head into a unified PyTorch nn.Module.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional

try:
    from .vgat_encoder import DualVGATEncoder
    from .poe_fusion import PoEFusion
    from .decoders_and_losses import RNADecoder, ADTDecoder, DualDecoderReconstructionLoss
    from .clustering_and_contrastive import DECHead, PseudoLabelContrastiveHead
except ImportError:
    from vgat_encoder import DualVGATEncoder
    from poe_fusion import PoEFusion
    from decoders_and_losses import RNADecoder, ADTDecoder, DualDecoderReconstructionLoss
    from clustering_and_contrastive import DECHead, PseudoLabelContrastiveHead


class VGAT_PoE_DEC(nn.Module):
    """
    End-to-End Dual-Encoder VGAT + Product of Experts (PoE) + Dual Decoders + DEC + Contrastive Architecture.
    """
    def __init__(
        self,
        in_dim_rna: int,
        in_dim_adt: int,
        num_clusters: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        proj_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        lambda_adt: float = 1.0,
        tau_threshold: float = 0.95,
        temperature: float = 0.1
    ):
        super(VGAT_PoE_DEC, self).__init__()
        self.in_dim_rna = in_dim_rna
        self.in_dim_adt = in_dim_adt
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim

        # 1. Dual-Encoder VGAT Architecture
        self.encoders = DualVGATEncoder(
            in_dim_rna=in_dim_rna,
            in_dim_adt=in_dim_adt,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            heads=heads,
            dropout=dropout
        )

        # 2. Product of Experts (PoE) Fusion
        self.poe_fusion = PoEFusion()

        # 3. Dual-Decoders
        self.rna_decoder = RNADecoder(latent_dim=latent_dim, out_features=in_dim_rna, hidden_dim=hidden_dim)
        self.adt_decoder = ADTDecoder(latent_dim=latent_dim, out_features=in_dim_adt, hidden_dim=hidden_dim)
        self.recon_loss_fn = DualDecoderReconstructionLoss(lambda_adt=lambda_adt)

        # 4. DEC Clustering Head
        self.dec_head = DECHead(num_clusters=num_clusters, latent_dim=latent_dim)

        # 5. Pseudo-Label Contrastive Head
        self.contrastive_head = PseudoLabelContrastiveHead(
            latent_dim=latent_dim,
            proj_dim=proj_dim,
            temperature=temperature,
            tau_threshold=tau_threshold
        )

    def forward(
        self,
        x_rna: torch.Tensor,
        x_adt: torch.Tensor,
        edge_index: torch.Tensor,
        raw_rna_counts: Optional[torch.Tensor] = None,
        library_size: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass of the complete architecture.

        Returns:
            Dictionary containing latent embeddings, decoded outputs, soft assignments Q, projection representations.
        """
        # Step 1: Dual-Encoder VGAT
        mu_rna, logvar_rna, mu_adt, logvar_adt = self.encoders(x_rna, x_adt, edge_index)

        # Step 2: PoE Fusion
        z_f, mu_f, var_f = self.poe_fusion(mu_rna, logvar_rna, mu_adt, logvar_adt)

        # Step 3: Decoders
        rna_mean, rna_theta, rna_pi = self.rna_decoder(z_f, library_size=library_size)
        x_adt_hat = self.adt_decoder(z_f)

        # Step 4: DEC Soft Assignments Q
        q = self.dec_head(z_f)

        # Step 5: Contrastive Projection
        z_proj = self.contrastive_head(z_f)

        outputs = {
            "z_f": z_f,
            "mu_f": mu_f,
            "var_f": var_f,
            "mu_rna": mu_rna,
            "logvar_rna": logvar_rna,
            "mu_adt": mu_adt,
            "logvar_adt": logvar_adt,
            "rna_mean": rna_mean,
            "rna_theta": rna_theta,
            "rna_pi": rna_pi,
            "x_adt_hat": x_adt_hat,
            "q": q,
            "z_proj": z_proj
        }
        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, Any],
        x_rna_raw: torch.Tensor,
        x_adt: torch.Tensor,
        edge_index: torch.Tensor,
        p_target: Optional[torch.Tensor] = None,
        phase: int = 1,
        alpha: float = 1.0,
        gamma: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute multi-loss objective according to training execution phase.

        Phase 1: Warm-up (Recon + Spatial InfoNCE)
        Phase 3: Joint Fine-tuning (Recon + alpha * InfoNCE_Pseudo + gamma * DEC_Loss)
        """
        z_f = outputs["z_f"]
        q = outputs["q"]
        z_proj = outputs["z_proj"]

        # Reconstruction Loss (L_Recon)
        l_recon, recon_dict = self.recon_loss_fn(
            raw_rna_counts=x_rna_raw,
            rna_mean=outputs["rna_mean"],
            rna_theta=outputs["rna_theta"],
            rna_pi=outputs["rna_pi"],
            x_adt=x_adt,
            x_adt_hat=outputs["x_adt_hat"]
        )

        loss_dict = recon_dict.copy()

        if phase == 1:
            # Warm-up phase: DEC disabled, standard spatial InfoNCE
            l_contrast = self.contrastive_head.spatial_infonce_warmup(z_proj, edge_index)
            l_total = l_recon + alpha * l_contrast
            loss_dict["l_contrast"] = l_contrast
            loss_dict["l_dec"] = torch.tensor(0.0, device=z_f.device)
            loss_dict["l_total"] = l_total
            return l_total, loss_dict

        elif phase == 3:
            # Joint Fine-Tuning Phase
            l_pseudo_contrast = self.contrastive_head.pseudo_label_contrastive_loss(z_proj, q, edge_index)

            if p_target is not None:
                l_dec = DECHead.loss_dec(q, p_target)
            else:
                p_target = DECHead.target_distribution(q)
                l_dec = DECHead.loss_dec(q, p_target)

            l_total = l_recon + alpha * l_pseudo_contrast + gamma * l_dec

            loss_dict["l_contrast"] = l_pseudo_contrast
            loss_dict["l_dec"] = l_dec
            loss_dict["l_total"] = l_total
            return l_total, loss_dict
        else:
            loss_dict["l_total"] = l_recon
            return l_recon, loss_dict
