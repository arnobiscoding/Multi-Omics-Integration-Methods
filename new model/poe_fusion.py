"""
Product of Experts (PoE) Fusion Module
======================================

Computes the joint latent distribution from RNA and ADT latent distributions:
1. Fused Variance: sigma^2_f = (1 / sigma^2_RNA + 1 / sigma^2_ADT)^(-1)
2. Fused Mean: mu_f = sigma^2_f * (mu_RNA / sigma^2_RNA + mu_ADT / sigma^2_ADT)
3. Reparameterization: Z_f = mu_f + epsilon * sqrt(sigma^2_f) where epsilon ~ N(0, I)
"""

import torch
import torch.nn as nn
from typing import Tuple


class PoEFusion(nn.Module):
    """
    Product of Experts (PoE) Multimodal Fusion Module.
    Combines gaussian posterior distributions (mu, logvar) from RNA and ADT encoders.
    """
    def __init__(self, eps: float = 1e-8):
        super(PoEFusion, self).__init__()
        self.eps = eps

    def forward(
        self,
        mu_rna: torch.Tensor,
        logvar_rna: torch.Tensor,
        mu_adt: torch.Tensor,
        logvar_adt: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mu_rna: (N, D) Mean from RNA encoder
            logvar_rna: (N, D) Log-variance from RNA encoder
            mu_adt: (N, D) Mean from ADT encoder
            logvar_adt: (N, D) Log-variance from ADT encoder

        Returns:
            z_f: (N, D) Reparameterized fused latent embedding Z_f
            mu_f: (N, D) Joint fused mean mu_f
            var_f: (N, D) Joint fused variance sigma^2_f
        """
        # Convert log-variance to variance: sigma^2 = exp(logvar)
        var_rna = torch.exp(logvar_rna) + self.eps
        var_adt = torch.exp(logvar_adt) + self.eps

        # Precision T = 1 / sigma^2
        prec_rna = 1.0 / var_rna
        prec_adt = 1.0 / var_adt

        # 1. Fused Variance: sigma^2_f = (1/sigma^2_RNA + 1/sigma^2_ADT)^(-1)
        prec_f = prec_rna + prec_adt
        var_f = 1.0 / (prec_f + self.eps)

        # 2. Fused Mean: mu_f = sigma^2_f * (mu_RNA / sigma^2_RNA + mu_ADT / sigma^2_ADT)
        mu_f = var_f * (mu_rna * prec_rna + mu_adt * prec_adt)

        # 3. Reparameterization Trick: Z_f = mu_f + epsilon * sqrt(sigma^2_f)
        if self.training:
            epsilon = torch.randn_like(mu_f)
            z_f = mu_f + epsilon * torch.sqrt(var_f + self.eps)
        else:
            z_f = mu_f

        return z_f, mu_f, var_f
