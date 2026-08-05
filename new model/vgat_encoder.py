"""
Dual-Encoder Variational Graph Attention Network (VGAT) Architecture
====================================================================

Implements two independent VGAT encoders (RNA and ADT).
Both encoders process their respective modality features across the shared spatial topology A.

Each VGAT encoder maps input node features into latent mean mu and log-variance log(sigma^2):
- RNA Encoder: outputs mu_RNA and log(sigma^2_RNA)
- ADT Encoder: outputs mu_ADT and log(sigma^2_ADT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

try:
    from torch_geometric.nn import GATConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class CustomVGATLayer(nn.Module):
    """
    Fallback custom Variational Graph Attention layer when PyTorch Geometric is unavailable,
    implementing dynamic LeakyReLU spatial attention coefficients alpha_ij over physical graph A.
    """
    def __init__(self, in_features: int, out_features: int, heads: int = 1, negative_slope: float = 0.2):
        super(CustomVGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.negative_slope = negative_slope

        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        self.a_src = nn.Parameter(torch.Tensor(1, heads, out_features))
        self.a_dst = nn.Parameter(torch.Tensor(1, heads, out_features))
        self.bias = nn.Parameter(torch.Tensor(heads * out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (N, in_features)
        N = x.size(0)
        h = self.W(x).view(N, self.heads, self.out_features) # (N, heads, out_features)

        row, col = edge_index[0], edge_index[1]

        # Compute attention scores: LeakyReLU(a^T [W h_i || W h_j])
        alpha_src = (h * self.a_src).sum(dim=-1) # (N, heads)
        alpha_dst = (h * self.a_dst).sum(dim=-1) # (N, heads)

        edge_alpha = alpha_src[row] + alpha_dst[col] # (E, heads)
        edge_alpha = F.leaky_relu(edge_alpha, negative_slope=self.negative_slope)

        # Softmax over neighborhood
        # Use scatter max/sum or loop for numerical stability
        edge_alpha_exp = torch.exp(edge_alpha - edge_alpha.max())
        alpha_sum = torch.zeros(N, self.heads, device=x.device)
        alpha_sum.index_add_(0, row, edge_alpha_exp)
        alpha_norm = edge_alpha_exp / (alpha_sum[row] + 1e-8) # (E, heads)

        # Message passing aggregation
        out = torch.zeros(N, self.heads, self.out_features, device=x.device)
        msg = h[col] * alpha_norm.unsqueeze(-1) # (E, heads, out_features)
        out.index_add_(0, row, msg)

        out = out.view(N, self.heads * self.out_features) + self.bias
        return out


class VGATEncoder(nn.Module):
    """
    Modality-Specific Variational Graph Attention Encoder.
    Processes node features through VGAT layers to produce latent mean mu and log(sigma^2).
    """
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2
    ):
        super(VGATEncoder, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dropout = dropout

        # Layer 1: Feature Projection to Graph Attention hidden space
        if PYG_AVAILABLE:
            self.gat1 = GATConv(
                in_channels=in_features,
                out_channels=hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                negative_slope=negative_slope
            )
            self.gat_mu = GATConv(
                in_channels=hidden_dim,
                out_channels=latent_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                negative_slope=negative_slope
            )
            self.gat_logvar = GATConv(
                in_channels=hidden_dim,
                out_channels=latent_dim,
                heads=1,
                concat=False,
                dropout=dropout,
                negative_slope=negative_slope
            )
        else:
            self.gat1 = CustomVGATLayer(in_features, hidden_dim // heads, heads=heads, negative_slope=negative_slope)
            self.gat_mu = CustomVGATLayer(hidden_dim, latent_dim, heads=1, negative_slope=negative_slope)
            self.gat_logvar = CustomVGATLayer(hidden_dim, latent_dim, heads=1, negative_slope=negative_slope)

        self.act = nn.ELU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for modality-specific VGAT encoder.

        Args:
            x: Node feature tensor (N, in_features)
            edge_index: Graph topology edge index tensor (2, E)

        Returns:
            mu: Latent mean tensor (N, latent_dim)
            logvar: Latent log-variance tensor (N, latent_dim)
        """
        h = self.gat1(x, edge_index)
        h = self.act(h)
        h = self.dropout_layer(h)

        mu = self.gat_mu(h, edge_index)
        logvar = self.gat_logvar(h, edge_index)

        # Clamp logvar for numerical stability in PoE fusion
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        return mu, logvar


class DualVGATEncoder(nn.Module):
    """
    Dual-Encoder VGAT Module containing independent encoders for RNA and ADT modalities.
    """
    def __init__(
        self,
        in_dim_rna: int,
        in_dim_adt: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super(DualVGATEncoder, self).__init__()
        self.rna_encoder = VGATEncoder(
            in_features=in_dim_rna,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            heads=heads,
            dropout=dropout
        )
        self.adt_encoder = VGATEncoder(
            in_features=in_dim_adt,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            heads=heads,
            dropout=dropout
        )

    def forward(
        self,
        x_rna: torch.Tensor,
        x_adt: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for both RNA and ADT encoders.

        Returns:
            mu_rna, logvar_rna, mu_adt, logvar_adt
        """
        mu_rna, logvar_rna = self.rna_encoder(x_rna, edge_index)
        mu_adt, logvar_adt = self.adt_encoder(x_adt, edge_index)
        return mu_rna, logvar_rna, mu_adt, logvar_adt
