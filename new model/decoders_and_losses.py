"""
Dual-Decoder & Reconstruction Loss (L_Recon) Module
===================================================

1. RNA Decoder (g_phi1): Reconstructs sparse RNA counts using Zero-Inflated Negative Binomial (ZINB) loss
   to model dropouts and overdispersion.
2. ADT Decoder (g_phi2): Reconstructs continuous ADT protein levels using Mean Squared Error (MSE) loss.
3. Total Reconstruction Loss: L_Recon = L_ZINB_RNA + lambda_ADT * L_MSE_ADT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class ZINBLoss(nn.Module):
    """
    Zero-Inflated Negative Binomial (ZINB) Loss Module.

    Calculates the negative log-likelihood of the ZINB distribution for sparse RNA count reconstruction.
    """
    def __init__(self, eps: float = 1e-10):
        super(ZINBLoss, self).__init__()
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        mean: torch.Tensor,
        theta: torch.Tensor,
        pi: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Ground truth count matrix (N, F)
            mean: Reconstructed mean mu (N, F), mu > 0
            theta: Reconstructed dispersion parameter theta (N, F), theta > 0
            pi: Dropout probability parameter pi (N, F), 0 <= pi <= 1

        Returns:
            Scalar ZINB negative log-likelihood loss
        """
        # Clamp inputs for numerical stability
        mean = torch.clamp(mean, min=self.eps, max=1e6)
        theta = torch.clamp(theta, min=self.eps, max=1e6)
        pi = torch.clamp(pi, min=self.eps, max=1.0 - self.eps)

        # Log probability of zero-count component: log(pi + (1-pi) * (theta / (theta + mean))^theta)
        softplus_pi = F.softplus(-torch.log(pi / (1.0 - pi + self.eps) + self.eps))

        nb_zero_log = theta * (torch.log(theta + self.eps) - torch.log(theta + mean + self.eps))

        # Log prob when x == 0
        zero_case = torch.log(pi + (1.0 - pi) * torch.exp(nb_zero_log) + self.eps)

        # Log prob when x > 0
        # NB log-pdf: log(1-pi) + log(Gamma(x+theta)) - log(Gamma(x+1)) - log(Gamma(theta))
        #             + theta*log(theta/(theta+mean)) + x*log(mean/(theta+mean))
        non_zero_case = (
            torch.log(1.0 - pi + self.eps)
            + torch.lgamma(x + theta + self.eps)
            - torch.lgamma(x + 1.0)
            - torch.lgamma(theta + self.eps)
            + theta * (torch.log(theta + self.eps) - torch.log(theta + mean + self.eps))
            + x * (torch.log(mean + self.eps) - torch.log(theta + mean + self.eps))
        )

        mask = (x < 1e-5).float()
        log_likelihood = mask * zero_case + (1.0 - mask) * non_zero_case

        loss = -torch.mean(log_likelihood)
        return loss


class RNADecoder(nn.Module):
    """
    RNA Decoder (g_phi1): Reconstructs ZINB parameters (mean, theta, pi) from latent fused embedding Z_f.
    """
    def __init__(self, latent_dim: int, out_features: int, hidden_dim: int = 128):
        super(RNADecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU()
        )

        # Heads for ZINB parameters
        self.decoder_mean = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.Softmax(dim=-1) # Reconstruct scale normalized across features
        )
        self.decoder_dispersion = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.Softplus() # theta > 0
        )
        self.decoder_pi = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.Sigmoid() # 0 <= pi <= 1
        )

    def forward(self, z_f: torch.Tensor, library_size: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_f: Latent embedding tensor (N, latent_dim)
            library_size: Optional spot library sizes (N, 1)

        Returns:
            mean: Predicted RNA mean count tensor (N, out_features)
            theta: Predicted dispersion tensor (N, out_features)
            pi: Predicted dropout probability tensor (N, out_features)
        """
        h = self.fc(z_f)
        scale = self.decoder_mean(h)

        if library_size is not None:
            if library_size.dim() == 1:
                library_size = library_size.unsqueeze(1)
            mean = scale * library_size
        else:
            mean = scale * 1000.0  # Default scale factor if library size is not explicitly passed

        theta = self.decoder_dispersion(h)
        pi = self.decoder_pi(h)

        return mean, theta, pi


class ADTDecoder(nn.Module):
    """
    ADT Decoder (g_phi2): Reconstructs continuous ADT protein levels from latent fused embedding Z_f.
    """
    def __init__(self, latent_dim: int, out_features: int, hidden_dim: int = 128):
        super(ADTDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, z_f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_f: Fused latent embedding (N, latent_dim)

        Returns:
            x_adt_hat: Reconstructed ADT continuous protein features (N, out_features)
        """
        return self.decoder(z_f)


class DualDecoderReconstructionLoss(nn.Module):
    """
    Combined Dual-Decoder Reconstruction Loss Anchor:
    L_Recon = L_ZINB_RNA + lambda_ADT * L_MSE_ADT
    """
    def __init__(self, lambda_adt: float = 1.0):
        super(DualDecoderReconstructionLoss, self).__init__()
        self.lambda_adt = lambda_adt
        self.zinb_loss = ZINBLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        raw_rna_counts: torch.Tensor,
        rna_mean: torch.Tensor,
        rna_theta: torch.Tensor,
        rna_pi: torch.Tensor,
        x_adt: torch.Tensor,
        x_adt_hat: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates L_Recon loss.

        Returns:
            total_recon_loss, dict of individual loss terms
        """
        l_zinb = self.zinb_loss(raw_rna_counts, rna_mean, rna_theta, rna_pi)
        l_mse_adt = self.mse_loss(x_adt_hat, x_adt)

        l_recon = l_zinb + self.lambda_adt * l_mse_adt

        loss_dict = {
            "l_recon": l_recon,
            "l_zinb_rna": l_zinb,
            "l_mse_adt": l_mse_adt
        }
        return l_recon, loss_dict
