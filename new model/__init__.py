"""
Spatial Multi-Omics Dual-Encoder VGAT PoE DEC Pipeline Package
===============================================================

This package implements an end-to-end PyTorch/PyTorch Geometric architecture for spatial multi-omics:
- Dual-Encoder Variational Graph Attention Network (VGAT)
- Product of Experts (PoE) Fusion
- Dual-Decoders (RNA ZINB + ADT MSE)
- DEC Soft Clustering Head & Target Distribution
- Pseudo-Label Spatially-Aware Contrastive Head
- 4-Phase Training Execution Engine
"""

from .preprocessing import build_spatial_knn_graph, prepare_spatial_multiomics_data
from .vgat_encoder import VGATEncoder, DualVGATEncoder
from .poe_fusion import PoEFusion
from .decoders_and_losses import RNADecoder, ADTDecoder, ZINBLoss, DualDecoderReconstructionLoss
from .clustering_and_contrastive import DECHead, PseudoLabelContrastiveHead
from .metrics import compute_silhouette, compute_morans_i, compute_composite_spatial_metric
from .model import VGAT_PoE_DEC
from .trainer import SpatialOmicsTrainer

__all__ = [
    "build_spatial_knn_graph",
    "prepare_spatial_multiomics_data",
    "VGATEncoder",
    "DualVGATEncoder",
    "PoEFusion",
    "RNADecoder",
    "ADTDecoder",
    "ZINBLoss",
    "DualDecoderReconstructionLoss",
    "DECHead",
    "PseudoLabelContrastiveHead",
    "compute_silhouette",
    "compute_morans_i",
    "compute_composite_spatial_metric",
    "VGAT_PoE_DEC",
    "SpatialOmicsTrainer"
]
