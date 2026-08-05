"""
Spatial Multi-Omics Dual-Encoder VGAT PoE DEC & spaLLM-VGAT Pipeline Package
=============================================================================

Includes:
1. Dual-Encoder VGAT PoE DEC Architecture (model.py, vgat_encoder.py, poe_fusion.py, decoders_and_losses.py, clustering_and_contrastive.py)
2. spaLLM-VGAT Architecture & FACT-style Spatial Clustering Engine (spallm_vgat_model.py, spallm_vgat_trainer.py)
"""

from .preprocessing import build_spatial_knn_graph, prepare_spatial_multiomics_data
from .vgat_encoder import VGATEncoder, DualVGATEncoder
from .poe_fusion import PoEFusion
from .decoders_and_losses import RNADecoder, ADTDecoder, ZINBLoss, DualDecoderReconstructionLoss
from .clustering_and_contrastive import DECHead, PseudoLabelContrastiveHead
from .metrics import compute_silhouette, compute_morans_i, compute_composite_spatial_metric
from .model import VGAT_PoE_DEC
from .trainer import SpatialOmicsTrainer
from .spallm_vgat_model import VGATEncodingNetwork, CellEmbedding, AttentionLayer
from .spallm_vgat_trainer import Train_spaLLM_VGAT, spatial_label_smoothing, run_mclust

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
    "SpatialOmicsTrainer",
    "VGATEncodingNetwork",
    "CellEmbedding",
    "AttentionLayer",
    "Train_spaLLM_VGAT",
    "spatial_label_smoothing",
    "run_mclust"
]
