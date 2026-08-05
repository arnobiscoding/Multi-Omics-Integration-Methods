"""
Preprocessing & Input Formulation for Spatial Multi-Omics Pipeline
===================================================================

- RNA Node Features: X_RNA in R^(N x F1) (top F1 Spatially Variable / Highly Variable Genes)
- ADT Node Features: X_ADT in R^(N x F2) (normalized protein abundance features)
- Spatial Graph Topology: Undirected edge index tensor A in R^(2 x E) representing a
  K-Nearest Neighbors (KNN) spatial graph (K=4 or K=6) built strictly from physical (X, Y) spot coordinates.
"""

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, Union, Dict, Any

try:
    import torch_geometric
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    Data = None


def build_spatial_knn_graph(
    coords: np.ndarray,
    k_neighbors: int = 6,
    include_self_loops: bool = False
) -> torch.Tensor:
    """
    Construct an unweighted, undirected KNN spatial graph edge_index tensor
    strictly from physical (X, Y) spot coordinates.

    Args:
        coords: Spatial coordinates array of shape (N, 2)
        k_neighbors: Number of nearest neighbors K (default 4 or 6)
        include_self_loops: Whether to include self-loops in the graph

    Returns:
        edge_index: PyTorch LongTensor of shape (2, E)
    """
    coords = np.asarray(coords, dtype=np.float32)
    num_nodes = coords.shape[0]

    # Find k nearest neighbors based on physical Euclidean distance
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='kd_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)

    row_list = []
    col_list = []

    for i in range(num_nodes):
        for j_idx in range(1, k_neighbors + 1):  # Skip index 0 (itself)
            j = indices[i, j_idx]
            row_list.append(i)
            col_list.append(j)
            # Make graph undirected
            row_list.append(j)
            col_list.append(i)

    if include_self_loops:
        for i in range(num_nodes):
            row_list.append(i)
            col_list.append(i)

    edge_index = torch.tensor([row_list, col_list], dtype=torch.long)

    # Remove duplicate edges and sort
    edge_index = torch.unique(edge_index, dim=1)

    return edge_index


def prepare_spatial_multiomics_data(
    x_rna: Union[np.ndarray, torch.Tensor, sp.spmatrix],
    x_adt: Union[np.ndarray, torch.Tensor, sp.spmatrix],
    coords: Union[np.ndarray, torch.Tensor],
    raw_counts_rna: Optional[Union[np.ndarray, torch.Tensor, sp.spmatrix]] = None,
    k_neighbors: int = 6,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Prepare normalized feature tensors and spatial topology graph.

    Args:
        x_rna: RNA node features (N, F1)
        x_adt: ADT node features (N, F2)
        coords: Physical spot coordinates (N, 2)
        raw_counts_rna: Unnormalized sparse RNA count matrix (N, F1) for ZINB loss
        k_neighbors: K for KNN graph construction
        device: PyTorch device ('cpu' or 'cuda')

    Returns:
        Dictionary containing PyTorch Tensors and optional PyG Data object.
    """
    # Convert spmatrix to dense if sparse
    if sp.issparse(x_rna):
        x_rna = x_rna.toarray()
    if sp.issparse(x_adt):
        x_adt = x_adt.toarray()
    if raw_counts_rna is not None and sp.issparse(raw_counts_rna):
        raw_counts_rna = raw_counts_rna.toarray()

    # Convert arrays to PyTorch FloatTensors
    x_rna_tensor = torch.tensor(x_rna, dtype=torch.float32, device=device)
    x_adt_tensor = torch.tensor(x_adt, dtype=torch.float32, device=device)
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)

    if raw_counts_rna is not None:
        raw_rna_tensor = torch.tensor(raw_counts_rna, dtype=torch.float32, device=device)
    else:
        raw_rna_tensor = x_rna_tensor

    # Build spatial graph
    edge_index = build_spatial_knn_graph(coords, k_neighbors=k_neighbors).to(device)

    data_dict = {
        "x_rna": x_rna_tensor,
        "x_adt": x_adt_tensor,
        "raw_counts_rna": raw_rna_tensor,
        "edge_index": edge_index,
        "coords": coords_tensor,
        "num_nodes": x_rna_tensor.size(0),
        "f_rna": x_rna_tensor.size(1),
        "f_adt": x_adt_tensor.size(1)
    }

    if PYG_AVAILABLE:
        pyg_data = Data(
            x_rna=x_rna_tensor,
            x_adt=x_adt_tensor,
            edge_index=edge_index,
            pos=coords_tensor,
            raw_counts_rna=raw_rna_tensor
        )
        data_dict["pyg_data"] = pyg_data

    return data_dict
