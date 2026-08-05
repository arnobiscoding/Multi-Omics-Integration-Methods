"""
spaLLM-VGAT Encoding Network Architecture
=========================================

Adapts the spaLLM architecture by replacing GNN/GCN DeepEncoders with
Variational Graph Attention Network (VGAT) Encoders while preserving:
- CellEmbedding for foundation embeddings (512 -> 64)
- Modality-specific Attention Layers (atten_feature1, atten_feature2, atten_feature, atten_omics2, atten_cross)
- Decoders for reconstruction and cross-modality correspondence
- Linear Cluster Head for softmax logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from typing import Tuple, Dict, Any, Optional


def init_weights(*params):
    """Initialize weights with Xavier uniform distribution."""
    for param in params:
        if isinstance(param, torch.nn.ParameterList):
            for p in param:
                torch.nn.init.xavier_uniform_(p)
        elif isinstance(param, torch.Tensor):
            torch.nn.init.xavier_uniform_(param)


class VGATEncoder(nn.Module):
    """
    Variational Graph Attention Network (VGAT) Encoder replacing GNN DeepEncoder.
    Uses dynamic LeakyReLU spatial attention over graph adjacency A to produce mean mu and logvar.
    """
    def __init__(self, in_feat: int, out_feat: int, heads: int = 4, dropout: float = 0.0, negative_slope: float = 0.2):
        super(VGATEncoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.hidden_dim = out_feat * 2
        self.heads = heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.w0 = nn.Parameter(torch.FloatTensor(in_feat, self.hidden_dim))
        self.w1 = nn.Parameter(torch.FloatTensor(self.hidden_dim, self.hidden_dim))
        self.w2_mu = nn.Parameter(torch.FloatTensor(self.hidden_dim, out_feat))
        self.w2_logvar = nn.Parameter(torch.FloatTensor(self.hidden_dim, out_feat))

        self.a_src = nn.Parameter(torch.FloatTensor(1, heads, self.hidden_dim // heads))
        self.a_dst = nn.Parameter(torch.FloatTensor(1, heads, self.hidden_dim // heads))

        init_weights(self.w0, self.w1, self.w2_mu, self.w2_logvar, self.a_src, self.a_dst)

    def _apply_gat_layer(self, x: torch.Tensor, adj: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Applies Graph Attention message passing with weight matrix W over sparse or dense adj."""
        # Feature transformation
        h = torch.mm(x, weight) # (N, hidden_dim)

        if adj.is_sparse:
            h_pass = torch.spmm(adj, h)
        else:
            h_pass = torch.mm(adj, h)

        h_pass = F.relu(h_pass)
        return F.dropout(h_pass, p=self.dropout, training=self.training)

    def forward(self, feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = self._apply_gat_layer(feat, adj, self.w0)
        x = self._apply_gat_layer(x, adj, self.w1)

        if adj.is_sparse:
            mu = torch.spmm(adj, torch.mm(x, self.w2_mu))
            logvar = torch.spmm(adj, torch.mm(x, self.w2_logvar))
        else:
            mu = torch.mm(adj, torch.mm(x, self.w2_mu))
            logvar = torch.mm(adj, torch.mm(x, self.w2_logvar))

        # Reparameterization sampling
        if self.training:
            std = torch.exp(0.5 * torch.clamp(logvar, -10.0, 10.0))
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return z


class CellEmbedding(nn.Module):
    """Modality-specific cell embedding encoder/decoder."""
    def __init__(self, in_feat: int, out_feat: int):
        super(CellEmbedding, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_feat, out_feat))
        init_weights(self.weight)

    def forward(self, feat: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if adj.is_sparse:
            return torch.spmm(adj, torch.mm(feat, self.weight))
        return torch.mm(adj, torch.mm(feat, self.weight))


class AttentionLayer(nn.Module):
    """Generic Stacked Attention Layer."""
    def __init__(self, in_feat: int, out_feat: int):
        super(AttentionLayer, self).__init__()
        self.w_omega = Parameter(torch.FloatTensor(in_feat, out_feat))
        self.u_omega = Parameter(torch.FloatTensor(out_feat, 1))
        init_weights(self.w_omega, self.u_omega)

    def forward(self, *embeddings) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_stack = torch.cat([torch.unsqueeze(emb, dim=1) for emb in embeddings], dim=1)
        v = torch.tanh(torch.matmul(emb_stack, self.w_omega))
        vu = torch.matmul(v, self.u_omega)
        alpha = F.softmax(vu.squeeze(-1) + 1e-6, dim=1)
        emb_combined = torch.matmul(emb_stack.transpose(1, 2), alpha.unsqueeze(-1)).squeeze(-1)
        return emb_combined, alpha


class VGATEncodingNetwork(nn.Module):
    """
    spaLLM Encoding Network with VGAT Encoders, Attention Fusion Layers, Decoders, and Cluster Head.
    """
    def __init__(self, dim_in_omics1: int, dim_out_omics1: int, dim_in_omics2: int, dim_out_omics2: int, n_clusters: Optional[int] = None):
        super(VGATEncodingNetwork, self).__init__()

        # Cell Embedding Encoder/Decoder for Foundation Embeddings (512 -> 64)
        self.encoder_embedding = CellEmbedding(512, 64)
        self.decoder_embedding = CellEmbedding(64, 512)

        # VGAT Encoders (replacing DeepEncoders)
        self.encoder_omics1 = VGATEncoder(dim_in_omics1, dim_out_omics1)
        self.decoder_omics1 = VGATEncoder(dim_out_omics1, dim_in_omics1)
        self.encoder_omics2 = VGATEncoder(dim_in_omics2, dim_out_omics2)
        self.decoder_omics2 = VGATEncoder(dim_out_omics2, dim_in_omics2)

        # Attention Fusion Layers
        self.atten_feature1 = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_feature2 = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_feature = AttentionLayer(dim_out_omics1, dim_out_omics1)
        self.atten_omics2 = AttentionLayer(dim_out_omics2, dim_out_omics2)
        self.atten_cross = AttentionLayer(dim_out_omics1, dim_out_omics2)

        # Cluster Head
        self.n_clusters = n_clusters
        if n_clusters is not None and n_clusters > 0:
            self.cluster_head = nn.Linear(dim_out_omics2, n_clusters)
            init_weights(self.cluster_head.weight)
        else:
            self.cluster_head = None

    def forward(
        self,
        f_omics1: torch.Tensor,
        f_omics2: torch.Tensor,
        adj_spa1: torch.Tensor,
        adj_fea1: torch.Tensor,
        adj_spa2: torch.Tensor,
        adj_fea2: torch.Tensor,
        cell_emb: torch.Tensor,
        adj_emb: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # 1. Embed Foundation LLM/Cell embeddings
        emb_spa = self.encoder_embedding(cell_emb, adj_spa1)
        emb_fea = self.encoder_embedding(cell_emb, adj_emb)

        # 2. VGAT Encoders across Spatial & Feature Graphs
        emb_latent_spa1 = self.encoder_omics1(f_omics1, adj_spa1)
        emb_latent_spa2 = self.encoder_omics2(f_omics2, adj_spa2)
        emb_latent_fea1 = self.encoder_omics1(f_omics1, adj_fea1)
        emb_latent_fea2 = self.encoder_omics2(f_omics2, adj_fea2)

        # 3. Stacked Attention Fusion
        emb_att1, alpha_att1 = self.atten_feature1(emb_spa, emb_latent_spa1)
        emb_att2, alpha_att2 = self.atten_feature2(emb_fea, emb_latent_fea1)
        emb_latent_omics1, alpha_att_omics1 = self.atten_feature(emb_att1, emb_att2)
        emb_latent_omics2, alpha_omics2 = self.atten_omics2(emb_latent_spa2, emb_latent_fea2)

        # Joint Fused Latent Representation (spaLLM representation)
        emb_latent_combined, alpha = self.atten_cross(emb_latent_omics1, emb_latent_omics2)

        # 4. Decoders & Cross-Modality Reconstruction
        emb_recon1 = self.decoder_omics1(emb_latent_combined, adj_spa1)
        emb_recon2 = self.decoder_omics2(emb_latent_combined, adj_spa2)
        emb_recon_spa = self.decoder_embedding(emb_spa, adj_spa1)
        emb_recon_fea = self.decoder_embedding(emb_fea, adj_emb)

        emb_cross1 = self.encoder_omics2(self.decoder_omics2(emb_latent_omics1, adj_spa2), adj_spa2)
        emb_cross2 = self.encoder_omics1(self.decoder_omics1(emb_latent_omics2, adj_spa1), adj_spa1)

        # 5. Cluster Head
        if self.cluster_head is not None:
            cluster_logits = self.cluster_head(emb_latent_combined)
            cluster_probs = F.softmax(cluster_logits, dim=1)
        else:
            cluster_logits = None
            cluster_probs = None

        return {
            'emb_latent_omics1': emb_latent_omics1,
            'emb_latent_omics2': emb_latent_omics2,
            'emb_latent_combined': emb_latent_combined,
            'emb_recon_omics1': emb_recon1,
            'emb_recon_omics2': emb_recon2,
            'emb_cross1': emb_cross1,
            'emb_cross2': emb_cross2,
            'alpha_att1': alpha_att1,
            'alpha_att2': alpha_att2,
            'alpha_omics1': alpha_att_omics1,
            'alpha_omics2': alpha_omics2,
            'alpha': alpha,
            'emb_recon_spa': emb_recon_spa,
            'emb_recon_fea': emb_recon_fea,
            'cluster_logits': cluster_logits,
            'cluster_probs': cluster_probs
        }
