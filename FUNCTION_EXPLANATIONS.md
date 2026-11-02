# ASTRA-Redo: Comprehensive Function & Class Reference Guide

This document provides an exhaustive, granular reference for every function, class, method, and module in [`astra-redo.py`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py).

---

## Table of Contents

1. [Environment & Determinism Utilities](#1-environment--determinism-utilities)
   - [`set_seed`](#set_seed)
   - [`init_r_mclust`](#init_r_mclust)
2. [Universal Multi-Omics Preprocessing Engine](#2-universal-multi-omics-preprocessing-engine)
   - [`seurat_clr`](#seurat_clr)
   - [`clr_normalize_each_cell`](#clr_normalize_each_cell)
   - [`tfidf`](#tfidf)
   - [`preprocess_universal`](#preprocess_universal)
3. [Dual-Graph Topology & Spatial Adjacency](#3-dual-graph-topology--spatial-adjacency)
   - [`DualGraphData`](#dualgraphdata)
   - [`build_dual_graph`](#build_dual_graph)
   - [`compute_spatial_adj_norm`](#compute_spatial_adj_norm)
4. [Deep Neural Architecture & Attention Fusion](#4-deep-neural-architecture--attention-fusion)
   - [`GatedAttentionFusion`](#gatedattentionfusion)
   - [`SpatialPottsDEC`](#spatialpottsdec)
   - [`DualGCN_Gated`](#dualgcn_gated)
   - [`ASTRA_v1_DEC_Gated`](#astra_v1_dec_gated)
5. [Clustering & Evaluation Engine](#5-clustering--evaluation-engine)
   - [`mclust_R`](#mclust_r)
   - [`cluster_embeddings`](#cluster_embeddings)
6. [Training & Optimization Pipeline](#6-training--optimization-pipeline)
   - [`train_model_dec`](#train_model_dec)
7. [Dataset Resolution & Download Manager](#7-dataset-resolution--download-manager)
   - [`resolve_dataset_files`](#resolve_dataset_files)
8. [Visualization & Plotting Engines](#8-visualization--plotting-engines)
   - [`plot_training_curves`](#plot_training_curves)
   - [`plot_all_visualizations`](#plot_all_visualizations)
9. [CLI Orchestration & Batch Execution](#9-cli-orchestration--batch-execution)
   - [`run_experiment`](#run_experiment)
   - [`__main__` CLI Block](#__main__-cli-block)

---

## 1. Environment & Determinism Utilities

### `set_seed`
[astra-redo.py:47-57](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L47-L57)

```python
def set_seed(seed=42):
```

- **Purpose**: Enforces deterministic execution across all pseudo-random number generators in the environment.
- **Parameters**:
  - `seed` (*int*, default: `42`): Random seed value applied to all libraries.
- **Returns**: `None`.
- **Implementation Details**:
  - Seeds `random.seed(seed)` and `np.random.seed(seed)`.
  - Seeds PyTorch CPU (`torch.manual_seed(seed)`).
  - If CUDA is available, seeds GPU devices via `torch.cuda.manual_seed(seed)` and `torch.cuda.manual_seed_all(seed)`.
  - Configures cuDNN determinism: `torch.backends.cudnn.deterministic = True` and disables non-deterministic profiling heuristics via `torch.backends.cudnn.benchmark = False`.
  - Sets environment variable `PYTHONHASHSEED = str(seed)` to guarantee consistent dictionary and set iteration order.
- **Callers**:
  - Module top-level at load time ([line 58](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L58)).
  - Inside [`run_experiment`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L868) prior to instantiating each model iteration.

---

### `init_r_mclust`
[astra-redo.py:60-68](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L60-L68)

```python
def init_r_mclust():
```

- **Purpose**: Dynamically verifies and initializes the R programming environment bridge (`rpy2`) and imports the R clustering package `mclust`.
- **Parameters**: None.
- **Returns**:
  - `bool`: `True` if `rpy2` is installed and the `mclust` R package was successfully loaded; `False` if any import error or runtime exception occurred.
- **Callers**:
  - Called by [`mclust_R`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L382).
- **Execution Status**: *Function is written but effectively unreachable in the standard CLI workflow* because `train_model_dec` hardcodes `tool='kmeans'` and no CLI parameter is provided to request `mclust`.

---

## 2. Universal Multi-Omics Preprocessing Engine

### `seurat_clr`
[astra-redo.py:71-74](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L71-L74)

```python
def seurat_clr(x):
```

- **Purpose**: Nested helper within [`clr_normalize_each_cell`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L70) implementing Seurat-style Centered Log Ratio (CLR) normalization on a 1D vector of feature counts for a single spot/cell.
- **Parameters**:
  - `x` (*numpy.ndarray*): 1D array representing raw feature values (e.g. antibody-derived tags) for one cell.
- **Returns**:
  - `numpy.ndarray`: Log-normalized and geometric-mean-centered feature vector of the same length as `x`.
- **Mathematical Formulation**:
  $$\mu_{\ln} = \frac{1}{|x|} \sum_{i: x_i > 0} \ln(1 + x_i)$$
  $$G(x) = \exp(\mu_{\ln})$$
  $$\text{seurat\_clr}(x) = \ln\left(1 + \frac{x}{G(x) + 10^{-12}}\right)$$
- **Callers**:
  - Applied along single-cell rows inside [`clr_normalize_each_cell`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L80) via `np.apply_along_axis`.

---

### `clr_normalize_each_cell`
[astra-redo.py:70-81](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L70-L81)

```python
def clr_normalize_each_cell(adata, inplace=True):
```

- **Purpose**: Applies Seurat-style CLR normalization independently across each cell for CITE-seq ADT (protein) expression matrices.
- **Parameters**:
  - `adata` (*anndata.AnnData*): Annotated data object containing the ADT expression matrix in `adata.X`.
  - `inplace` (*bool*, default: `True`): If `True`, overwrites `adata.X`; if `False`, operates on a deep copy of `adata`.
- **Returns**:
  - `adata` (*anndata.AnnData*): AnnData object with CLR-normalized `adata.X`.
- **Callers**:
  - Called inside [`preprocess_universal`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L113) when the dataset name starts with `"10x"`.

---

### `tfidf`
[astra-redo.py:83-90](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L83-L90)

```python
def tfidf(X):
```

- **Purpose**: Computes Term Frequency - Inverse Document Frequency (TF-IDF) normalization on chromatin accessibility peak counts (spatial ATAC-seq).
- **Parameters**:
  - `X` (*scipy.sparse.spmatrix* or *numpy.ndarray*): Cell-by-peak matrix of shape $[N_{\text{spots}}, M_{\text{peaks}}]$.
- **Returns**:
  - *scipy.sparse.csr_matrix* or *numpy.ndarray*: TF-IDF weighted matrix with matching sparsity/density.
- **Mathematical Formulation**:
  $$\text{IDF}_j = \frac{N_{\text{spots}}}{\sum_{i=1}^{N_{\text{spots}}} X_{ij} + 10^{-12}}$$
  $$\text{TF}_{ij} = \frac{X_{ij}}{\sum_{k=1}^{M_{\text{peaks}}} X_{ik} + 10^{-12}}$$
  $$\text{TF-IDF}(X)_{ij} = \text{TF}_{ij} \cdot \text{IDF}_j$$
- **Callers**:
  - Called in [`preprocess_universal`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L118) for non-10x datasets (Mouse Brain spatial ATAC-seq).

---

### `preprocess_universal`
[astra-redo.py:92-126](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L92-L126)

```python
def preprocess_universal(adata_RNA, adata_omics2, dataset_name, n_hvg=3000):
```

- **Purpose**: Universal, dual-modality preprocessing engine that harmonizes transcriptomic (RNA) data and auxiliary data (ADT protein or ATAC chromatin accessibility).
- **Parameters**:
  - `adata_RNA` (*anndata.AnnData*): Spatial transcriptomics data object.
  - `adata_omics2` (*anndata.AnnData*): Secondary modality data object (ADT or ATAC).
  - `dataset_name` (*str*): Identifier string used to branch between CITE-seq (`10x_*`) and ATAC (`Mouse_Brain_*`).
  - `n_hvg` (*int*, default: `3000`): Target number of Highly Variable Genes (HVGs) to select.
- **Returns**:
  - `RNA_expression` (*numpy.ndarray*): Dense matrix $[N_{\text{spots}}, n\_hvg]$ of normalized, scaled HVG expression.
  - `ADT_expression` (*numpy.ndarray*): Dense matrix $[N_{\text{spots}}, D_{\text{aux}}]$ of normalized auxiliary features (all scaled ADTs or top 50 ATAC principal components).
  - `adata_RNA` (*anndata.AnnData*): Preprocessed RNA AnnData object.
  - `adata_omics2` (*anndata.AnnData*): Preprocessed secondary modality AnnData object.
- **Algorithmic Flow**:
  1. Filters out genes expressed in fewer than 10 cells (`sc.pp.filter_genes`).
  2. Selects top `n_hvg` genes using Seurat v3 (or standard Seurat on total-count-normalized counts).
  3. Applies total count normalization ($10,000$), $\log(1 + x)$, and standard scaling (`sc.pp.scale`).
  4. Branches on `dataset_name`:
     - **10x Datasets (Protein ADT)**: Calls `clr_normalize_each_cell` $\to$ `sc.pp.scale`.
     - **Other Datasets (ATAC Epigenomics)**: Calls `tfidf` $\to$ per-cell normalization ($10,000$) $\to$ $\log(1 + x)$ $\to$ `PCA(n_components=50)`.
- **Callers**:
  - Called inside [`run_experiment`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L859).

---

## 3. Dual-Graph Topology & Spatial Adjacency

### `DualGraphData`
[astra-redo.py:128-140](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L128-L140)

```python
class DualGraphData(torch_geometric.data.Data):
    def __init__(self, x_RNA, x_ADT, sim_edge_index, sim_edge_weight,
                 dist_edge_index, dist_edge_weight, common_edge_index, common_edge_weight):
```

- **Purpose**: Custom PyTorch Geometric `Data` subclass that encapsulates node feature matrices and three distinct edge graphs required by the dual-GCN architecture.
- **Attributes**:
  - `x_RNA` (*torch.Tensor*): RNA feature matrix $[N, 3000]$.
  - `x_ADT` (*torch.Tensor*): Auxiliary omics feature matrix $[N, D_{\text{aux}}]$.
  - `sim_edge_index` (*torch.LongTensor*): Feature similarity graph connectivity $[2, E_{\text{sim}}]$.
  - `sim_edge_weight` (*torch.FloatTensor*): Cosine similarity weights $[E_{\text{sim}}]$.
  - `dist_edge_index` (*torch.LongTensor*): Physical spatial kNN graph connectivity $[2, E_{\text{dist}}]$.
  - `dist_edge_weight` (*torch.FloatTensor*): Physical distance weights $[E_{\text{dist}}]$.
  - `common_edge_index` (*torch.LongTensor*): Consensus intersection graph connectivity $[2, E_{\text{common}}]$.
  - `common_edge_weight` (*torch.FloatTensor*): Consensus edge weights $[E_{\text{common}}]$ (all ones).
- **Instantiated in**: [`build_dual_graph`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L173).

---

### `build_dual_graph`
[astra-redo.py:141-178](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L141-L178)

```python
def build_dual_graph(RNA_expression, ADT_expression, cell_positions, device='cpu', num_neighbors=15):
```

- **Purpose**: Constructs the three topological graphs (transcriptomic similarity, physical distance, and intersection consensus) and packs them into a `DualGraphData` container on the specified device.
- **Parameters**:
  - `RNA_expression` (*numpy.ndarray*): Processed RNA expression matrix $[N, 3000]$.
  - `ADT_expression` (*numpy.ndarray*): Processed auxiliary feature matrix $[N, D_{\text{aux}}]$.
  - `cell_positions` (*numpy.ndarray*): 2D physical spatial coordinates $[N, 2]$.
  - `device` (*str* or *torch.device*, default: `'cpu'`): Target compute device.
  - `num_neighbors` (*int*, default: `15`): Number of nearest neighbors $k$ for graph construction.
- **Returns**:
  - `DualGraphData`: Fully initialized PyG data object on `device`.
- **Algorithmic Steps**:
  1. Computes pairwise cosine similarity matrix on `RNA_expression` and queries `NearestNeighbors(n_neighbors=16, metric='cosine')`.
  2. Symmetrizes the similarity adjacency matrix: $A_{\text{sim}}[i, j] = 1$ if $j \in \text{kNN}(i)$ or $i \in \text{kNN}(j)$.
  3. Builds physical distance graph on `cell_positions` using `kneighbors_graph(..., mode='distance')` and symmetrizes via element-wise maximum: $A_{\text{dist}} = \max(A, A^T)$.
  4. Finds common edges: $\mathcal{E}_{\text{common}} = \mathcal{E}_{\text{sim}} \cap \mathcal{E}_{\text{dist}}$.
  5. Converts arrays to PyTorch tensors and instantiates `DualGraphData`.
- **Callers**:
  - Called inside [`run_experiment`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L861).

---

### `compute_spatial_adj_norm`
[astra-redo.py:180-186](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L180-L186)

```python
def compute_spatial_adj_norm(dist_edge_index, num_nodes, device='cpu'):
```

- **Purpose**: Constructs a row-normalized dense spatial adjacency matrix with self-loops, used to perform Spatial Potts Markov Random Field consensus smoothing during DEC target distribution updates.
- **Parameters**:
  - `dist_edge_index` (*torch.Tensor*): Spatial distance edge index $[2, E_{\text{dist}}]$.
  - `num_nodes` (*int*): Total number of spots $N$.
  - `device` (*str* or *torch.device*, default: `'cpu'`): Compute device.
- **Returns**:
  - `adj_norm` (*torch.Tensor*): Dense normalized adjacency matrix $[N, N]$ on `device`.
- **Mathematical Formulation**:
  $$A = A_{\text{dist}} + I_N$$
  $$D_{ii} = \sum_{j=1}^N A_{ij}$$
  $$A_{\text{norm}} = D^{-1} A$$
- **Callers**:
  - Called in [`run_experiment`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L862).

---

## 4. Deep Neural Architecture & Attention Fusion

### `GatedAttentionFusion`
[astra-redo.py:188-210](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L188-L210)

```python
class GatedAttentionFusion(nn.Module):
    def __init__(self, dim: int = 64):
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
```

- **Purpose**: Spot-adaptive, feature-wise gated attention fusion module. Dynamically balances information flow between two latent representations $z_1, z_2 \in \mathbb{R}^d$ on a per-spot and per-dimension basis.
- **Parameters (`__init__`)**:
  - `dim` (*int*, default: `64`): Latent embedding dimensionality.
- **Layers**:
  - `self.gate_net`: `nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())`
  - `self.out_proj`: `nn.Linear(dim, dim)`
- **Forward Logic**:
  $$g = \sigma(W_g [z_1 \parallel z_2] + b_g) \in (0, 1)^d$$
  $$z_{\text{fused}} = g \odot z_1 + (1 - g) \odot z_2$$
  $$z_{\text{out}} = W_{\text{proj}} z_{\text{fused}} + b_{\text{proj}}$$
- **Outputs**:
  - `fused` (*torch.Tensor*): Projected fused embedding $[N, d]$.
  - `gate` (*torch.Tensor*): Learned attention gate values $[N, d]$.
- **Usage**:
  - Instantiated twice inside [`DualGCN_Gated`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L268-L270):
    1. `fusion_layer1`: Fuses RNA similarity ($z_{\text{sim}}$) and RNA spatial ($z_{\text{dist}}$).
    2. `fusion_layer2`: Fuses intra-RNA fused representation ($z_{\text{fused}}$) and auxiliary representation ($z_{\text{pro}}$).

---

### `SpatialPottsDEC`
[astra-redo.py:211-254](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L211-L254)

```python
class SpatialPottsDEC(nn.Module):
    def __init__(self, num_clusters: int, latent_dim: int = 64, alpha: float = 1.0, lambda_spatial: float = 0.15):
```

- **Purpose**: Deep Embedding Clustering (DEC) module enhanced with a Spatial Potts Markov Random Field consensus mechanism. Suppresses salt-and-pepper spatial clustering noise while preserving transcriptomic cluster boundaries.
- **Parameters (`__init__`)**:
  - `num_clusters` (*int*): Number of spatial clusters/domains $K$.
  - `latent_dim` (*int*, default: `64`): Latent space embedding dimensionality.
  - `alpha` (*float*, default: `1.0`): Degrees of freedom in Student's t-distribution.
  - `lambda_spatial` (*float*, default: `0.15`): Spatial Potts consensus smoothing factor.
- **Parameters Learned**:
  - `self.cluster_centers`: Learnable prototype matrix $\mu \in \mathbb{R}^{K \times \text{latent\_dim}}$, initialized via Xavier uniform.
- **Methods**:

#### `compute_q(self, z: torch.Tensor) -> torch.Tensor`
- Computes Student's t-distribution soft cluster assignments:
  $$q_{ik} = \frac{\left(1 + \frac{\|z_i - \mu_k\|^2}{\alpha}\right)^{-\frac{\alpha + 1}{2}}}{\sum_{k'=1}^K \left(1 + \frac{\|z_i - \mu_{k'}\|^2}{\alpha}\right)^{-\frac{\alpha + 1}{2}}}$$
- Shape: $[N, K]$.

#### `compute_spatial_target_p(self, q: torch.Tensor, spatial_adj_norm: torch.Tensor) -> torch.Tensor`
- Computes the spatial Potts-smoothed target distribution $P$:
  1. Standard DEC sharpening:
     $$p_{\text{dec}, ik} = \frac{q_{ik}^2 / \sum_{i=1}^N q_{ik}}{\sum_{k'=1}^K (q_{ik'}^2 / \sum_{i=1}^N q_{ik'})}$$
  2. Spatial neighborhood consensus:
     $$\text{consensus} = A_{\text{norm}} \cdot Q, \quad \text{consensus}_{ik} = \sum_{j=1}^N (A_{\text{norm}})_{ij} q_{jk}$$
  3. Potts modulation and normalization:
     $$p_{\text{final}, ik} = \frac{p_{\text{dec}, ik} \cdot \exp(\lambda_{\text{spatial}} \cdot \text{consensus}_{ik})}{\sum_{k'=1}^K p_{\text{dec}, ik'} \cdot \exp(\lambda_{\text{spatial}} \cdot \text{consensus}_{ik'})}$$
- Shape: $[N, K]$.

#### `forward(self, z: torch.Tensor, p_target: Optional[torch.Tensor] = None)`
- Computes $Q = \text{compute\_q}(z)$.
- If `p_target` is provided, calculates Kullback-Leibler divergence:
  $$\mathcal{L}_{\text{KL}} = \text{KL}(Q \parallel P) = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p_{ik} \ln \left(\frac{p_{ik}}{q_{ik}}\right)$$
- Returns: `(q, kl_loss)`.

---

### `DualGCN_Gated`
[astra-redo.py:256-304](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L256-L304)

```python
class DualGCN_Gated(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, q_dim, dropout=0.0):
```

- **Purpose**: Core dual-branch Graph Convolutional Network with gated cross-modality fusion and multi-task reconstruction decoders.
- **Parameters (`__init__`)**:
  - `in_channels` (*int*): Input RNA HVG dimension ($3000$).
  - `hidden_channels` (*int*): Intermediate hidden feature dimension ($512$).
  - `out_channels` (*int*): Final latent embedding dimension ($64$).
  - `q_dim` (*int*): Input auxiliary omics feature dimension ($D_{\text{aux}}$).
  - `dropout` (*float*, default: `0.0`): Dropout probability applied between GCN layers.
- **Layers**:
  - `x_RNA1`: `GCNConv(in_channels, hidden_channels)` (RNA similarity branch layer 1)
  - `x_RNA2`: `GCNConv(in_channels, hidden_channels)` (RNA spatial distance branch layer 1)
  - `sim_conv`: `GCNConv(hidden_channels, out_channels)` (RNA similarity branch layer 2)
  - `dist_conv`: `GCNConv(hidden_channels, out_channels)` (RNA spatial distance branch layer 2)
  - `protein3`: `GCNConv(q_dim, out_channels)` (Auxiliary modality over common consensus graph)
  - `fusion_layer1`: `GatedAttentionFusion(dim=out_channels)`
  - `fusion_layer2`: `GatedAttentionFusion(dim=out_channels)`
  - `deconv1`: `nn.Linear(out_channels, hidden_channels)`
  - `deconv2`: `nn.Linear(hidden_channels, in_channels)` (RNA decoder)
  - `deconv4`: `nn.Linear(hidden_channels, q_dim)` (Auxiliary decoder)
  - `deconv5`: `nn.Linear(hidden_channels, in_channels + q_dim)` (Joint multi-omics decoder)
- **Methods**:

#### `forward(self, x_RNA, x_ADT, sim_edge_index, sim_edge_weight, dist_edge_index, dist_edge_weight, common_edge_index, common_edge_weight)`
- Propagates RNA through similarity and distance GCNs.
- Propagates auxiliary omics through consensus GCN.
- Executes two-level gated fusion.
- Returns: `(x_sim, x_dist, fused, fused_pro, pro, gate_spatial, gate_modality)`.

#### `reconstruct_joint(self, z)`
- Decodes $z \to \text{Linear}(64 \to 512) \to \text{ReLU} \to \text{Linear}(512 \to 3000 + D_{\text{aux}})$.

#### `reconstruct_rna(self, z)`
- Decodes $z \to \text{Linear}(64 \to 512) \to \text{ReLU} \to \text{Linear}(512 \to 3000)$.

#### `reconstruct_aux(self, z)`
- Decodes $z \to \text{Linear}(64 \to 512) \to \text{ReLU} \to \text{Linear}(512 \to D_{\text{aux}})$.

---

### `ASTRA_v1_DEC_Gated`
[astra-redo.py:305-378](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L305-L378)

```python
class ASTRA_v1_DEC_Gated(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, q_dim, num_clusters,
                 beta=25.0, gamma=10.0, delta=1.0, lambda_dec=1.0, lambda_spatial=0.15, dropout=0.0):
```

- **Purpose**: Top-level PyTorch model combining `DualGCN_Gated` and `SpatialPottsDEC`, managing loss computations and center initializations.
- **Hyperparameters**:
  - `beta` (*float*, default: `25.0`): Multi-task reconstruction loss weight.
  - `gamma` (*float*, default: `10.0`): Spatial embedding regularization loss weight.
  - `delta` (*float*, default: `1.0`): Weight decay ($L_1 / L_2$) regularization weight.
  - `lambda_dec` (*float*, default: `1.0`): DEC KL divergence loss weight.
  - `lambda_spatial` (*float*, default: `0.15`): Potts spatial consensus weight.
  - `l1_lambda` (*float*): $10^{-4}$ ($L_1$ weight sparsity penalty).
  - `l2_lambda` (*float*): $10^{-3}$ ($L_2$ weight decay penalty).
- **Methods**:

#### `init_cluster_centers(self, centers_numpy)`
- Copies external cluster centroids (from KMeans fitted on peak Stage 1 representations) into `self.dec.cluster_centers`.

#### `forward(self, data, compute_dec=False, p_target=None)`
- Invokes `self.gcn`.
- If `compute_dec=True`, calls `self.dec(fused_pro, p_target)` to compute $Q$ and KL loss.
- Returns embeddings, gates, and optionally $Q$ and $\mathcal{L}_{\text{KL}}$.

#### `cosine_similarity(self, emb)`
- Computes pairwise cosine similarity matrix of embedding matrix `emb` $[N, d]$:
  $$S_{ij} = \frac{z_i \cdot z_j}{\|z_i\|_2 \|z_j\|_2 + 10^{-12}}, \quad S_{ii} = 0$$

#### `spatial_regularization_loss(self, emb, dist_edge_index, dist_edge_weight)`
- Binary cross-entropy style contrastive loss over spatial graph edges:
  $$\mathcal{L}_{\text{spatial}} = -\frac{1}{2} \left( \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \ln \sigma(S_{ij}) + \frac{1}{|\mathcal{E}^c|} \sum_{(i,j) \notin \mathcal{E}} \ln(1 - \sigma(S_{ij})) \right)$$

#### `compute_losses(self, data, sim_z, dist_z, fused_z, fused_pro, pro, kl_loss=None, stage=1)`
- Computes multi-task objective:
  - $\mathcal{L}_{\text{rec}} = \text{MSE}([X_{\text{RNA}} \parallel X_{\text{ADT}}], \text{reconstruct\_joint}(z_{\text{fused\_pro}}))$
  - $\mathcal{L}_{\text{sim}} = \text{MSE}(X_{\text{RNA}}, \text{reconstruct\_rna}(z_{\text{sim}}))$
  - $\mathcal{L}_{\text{dist}} = \text{MSE}(X_{\text{RNA}}, \text{reconstruct\_rna}(z_{\text{dist}}))$
  - $\mathcal{L}_{\text{aux}} = \text{MSE}(X_{\text{ADT}}, \text{reconstruct\_aux}(z_{\text{pro}}))$
  - $\mathcal{L}_{\text{spatial}} = \text{spatial\_regularization\_loss}(z_{\text{fused}}, \text{dist\_edge\_index}, \text{dist\_edge\_weight})$
  - $\mathcal{L}_{\text{reg}} = 10^{-4} \sum |\Theta| + 10^{-3} \sum \Theta^2$
  - In Stage 2: weights reconstruction by $0.2 \cdot \beta$ and adds $\lambda_{\text{dec}} \cdot \mathcal{L}_{\text{KL}}$.
- Returns: `(total_loss, l_rec, l_spatial, kl_val)`.

---

## 5. Clustering & Evaluation Engine

### `mclust_R`
[astra-redo.py:380-414](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L380-L414)

```python
def mclust_R(adata_or_embeddings, num_cluster, modelNames='EEE', use_pca=True, n_comps=20, random_seed=42):
```

- **Purpose**: Gaussian Mixture Model clustering using R's `mclust` package via `rpy2`.
- **Parameters**:
  - `adata_or_embeddings` (*array-like*): Matrix of spot embeddings $[N, d]$.
  - `num_cluster` (*int*): Target number of clusters $K$.
  - `modelNames` (*str*, default: `'EEE'`): Model parameterization specification for R `Mclust` (ellipsoidal, equal volume, shape, and orientation).
  - `use_pca` (*bool*, default: `True`): Whether to reduce dimension to `n_comps` via PCA before clustering.
  - `n_comps` (*int*, default: `20`): PCA components.
  - `random_seed` (*int*, default: `42`): Seed for R and Python RNG.
- **Returns**:
  - `numpy.ndarray`: 1D array of integer cluster assignments $[N]$.
- **Execution Status**: *Effectively unreachable in the standard CLI workflow* (fallback to `sklearn.cluster.KMeans` is executed if invoked, but the function itself is bypassed by `tool='kmeans'`).

---

### `cluster_embeddings`
[astra-redo.py:416-420](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L416-L420)

```python
def cluster_embeddings(embeddings, num_clusters, method='kmeans', random_seed=42):
```

- **Purpose**: Dispatcher function for clustering latent embeddings.
- **Parameters**:
  - `embeddings` (*numpy.ndarray*): Latent representations $[N, d]$.
  - `num_clusters` (*int*): Number of clusters $K$.
  - `method` (*str*, default: `'kmeans'`): `'mclust'` or `'kmeans'`.
  - `random_seed` (*int*, default: `42`): Random seed.
- **Returns**:
  - `numpy.ndarray`: Cluster labels $[N]$.
- **Callers**:
  - Called at the end of each Stage 1 epoch in [`train_model_dec`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L486).

---

## 6. Training & Optimization Pipeline

### `train_model_dec`
[astra-redo.py:422-638](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L422-L638)

```python
def train_model_dec(
    model, data, spatial_adj_norm, gt_labels=None,
    pretrain_epochs=250, finetune_epochs=150, lr=1e-3,
    num_clusters=10, tool='kmeans', seed=42, p_update_interval=5
):
```

- **Purpose**: Core orchestration function for the two-stage training strategy (Pre-training $\to$ Bridge Transition $\to$ Potts Consensus DEC Fine-Tuning).
- **Parameters**:
  - `model` (*ASTRA_v1_DEC_Gated*): Neural network model.
  - `data` (*DualGraphData*): Multi-omics dual-graph dataset.
  - `spatial_adj_norm` (*torch.Tensor*): Row-normalized spatial adjacency matrix.
  - `gt_labels` (*numpy.ndarray*, optional): Ground truth domain labels for tracking ARI during training.
  - `pretrain_epochs` (*int*, default: `250`): Stage 1 epoch count.
  - `finetune_epochs` (*int*, default: `150`): Stage 2 epoch count.
  - `lr` (*float*, default: `1e-3`): Stage 1 learning rate.
  - `num_clusters` (*int*, default: `10`): Number of clusters $K$.
  - `tool` (*str*, default: `'kmeans'`): Clustering algorithm.
  - `seed` (*int*, default: `42`): Random seed.
  - `p_update_interval` (*int*, default: `5`): Frequency (in epochs) to refresh spatial target distribution $P$.
- **Returns**:
  - `best_embeddings` (*numpy.ndarray*): Best latent embeddings $[N, 64]$.
  - `best_labels` (*numpy.ndarray*): Best cluster domain predictions $[N]$.
  - `best_imputed_rna` (*numpy.ndarray*): Imputed RNA expression (unpacked but unused in caller).
  - `best_gates` (*tuple* of *numpy.ndarray*): Attention gate arrays `(g_sp, g_mod)` (unpacked but unused in caller).
  - `best_epoch` (*int*): Epoch number where peak Silhouette score occurred.
  - `best_stage` (*str*): Stage (`"Pre-train"` or `"DEC-Potts"`) of peak Silhouette score.
  - `history` (*dict*): Epoch-by-epoch loss and evaluation metric dictionary.
  - `best_sil` (*float*): Peak Silhouette score achieved across all epochs.
  - `ari_at_best_sil` (*float*): ARI value at peak Silhouette score.
  - `sil` (*float*): Final epoch Silhouette score.
  - `ari` (*float*): Final epoch ARI.
- **Callers**:
  - Called inside [`run_experiment`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L876).

---

## 7. Dataset Resolution & Download Manager

### `resolve_dataset_files`
[astra-redo.py:650-685](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L650-L685)

```python
def resolve_dataset_files(dataset_name: str, folder_url: str):
```

- **Purpose**: Locates dataset files locally across candidate cache folders; downloads missing folders from Google Drive via `gdown`.
- **Parameters**:
  - `dataset_name` (*str*): Dataset identifier string.
  - `folder_url` (*str*): Google Drive folder URL.
- **Returns**:
  - `rna_path` (*str*): File path to `adata_RNA.h5ad`.
  - `aux_path` (*str*): File path to `adata_ADT.h5ad` or `adata_ATAC.h5ad`.
  - `anno_path` (*str*): File path to `annotation.csv` or `anno.csv`.
  - `gt_col` (*str*): Name of the ground truth label column (`"manual-anno"` or `"cluster"`).
- **Callers**:
  - Called inside [`run_experiment`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L850).

---

## 8. Visualization & Plotting Engines

### `plot_training_curves`
[astra-redo.py:692-740](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L692-L740)

```python
def plot_training_curves(training_results, dataset_name="Dataset", save_path=None, show=True):
```

- **Purpose**: Renders multi-panel training trajectory figures displaying:
  1. Loss Curves: Total loss, joint reconstruction, spatial regularization, DEC KL loss, and component reconstructions ($L_{\text{sim}}, L_{\text{dist}}, L_{\text{aux}}$).
  2. Silhouette Score progression curve across all epochs.
  3. Adjusted Rand Index (ARI) progression curve across all epochs.
- **Parameters**:
  - `training_results` (*dict*): The `history` dictionary from `train_model_dec`.
  - `dataset_name` (*str*, default: `"Dataset"`): Title string.
  - `save_path` (*str*, optional): Output file path for PNG export.
  - `show` (*bool*, default: `True`): Whether to display the plot interactively.
- **Callers**:
  - Called inside [`plot_all_visualizations`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L770).

---

### `plot_all_visualizations`
[astra-redo.py:741-816](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L741-L816)

```python
def plot_all_visualizations(
    adata_RNA, best_emb, best_labels, sil, ari, history,
    dataset_name="Dataset", seed=42, true_labels=None,
    output_dir="results", show=False
):
```

- **Purpose**: Generates and saves four comprehensive visualization artifacts:
  1. **Curves**: Invokes `plot_training_curves` and writes to `plots/curves/`.
  2. **Spatial Domains**: 2D scatter plots of physical coordinates comparing Ground Truth vs. Predicted Spatial Domains (`plots/spatial/`).
  3. **UMAP Joint Representation**: Computes Scanpy UMAP on `best_emb` and plots Ground Truth vs. Predicted Domains (`plots/umap/`).
  4. **Violin Profiles**: Plots silhouette coefficient distributions and latent dimension 1 distributions across domains (`plots/violin/`).
- **Parameters**:
  - `adata_RNA` (*anndata.AnnData*): RNA AnnData object containing spatial coordinates in `.obsm['spatial']`.
  - `best_emb` (*numpy.ndarray*): Best latent embeddings $[N, 64]$.
  - `best_labels` (*numpy.ndarray*): Best cluster labels $[N]$.
  - `sil` (*float*): Best silhouette score.
  - `ari` (*float*): ARI score at best silhouette.
  - `history` (*dict*): Training metrics history.
  - `dataset_name` (*str*): Dataset identifier.
  - `seed` (*int*): Random seed.
  - `true_labels` (*numpy.ndarray*, optional): Ground truth annotation array.
  - `output_dir` (*str*, default: `"results"`): Directory where `plots/` subdirectories are saved.
  - `show` (*bool*, default: `False`): Interactive display toggle.
- **Callers**:
  - Called inside [`run_experiment`](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L883).

---

## 9. CLI Orchestration & Batch Execution

### `run_experiment`
[astra-redo.py:821-904](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L821-L904)

```python
def run_experiment(datasets="all", seeds=None, pretrain_epochs=250, finetune_epochs=150, lr=1e-3, 
                   lambda_dec=1.0, lambda_spatial=0.15, output_dir="astra_redo_Results", 
                   visualize=True):
```

- **Purpose**: Master experiment controller. Iterates over datasets and seeds, executing data loading, preprocessing, graph building, training, visualization, and tabular metric exports.
- **Parameters**:
  - `datasets` (*str* or *list* of *str*/*int*, default: `"all"`): Dataset selection.
  - `seeds` (*list* of *int*, default: `[42, 1234, 2024]`): List of random seeds.
  - `pretrain_epochs` (*int*, default: `250`): Stage 1 epochs.
  - `finetune_epochs` (*int*, default: `150`): Stage 2 epochs.
  - `lr` (*float*, default: `1e-3`): Initial learning rate.
  - `lambda_dec` (*float*, default: `1.0`): DEC loss weight.
  - `lambda_spatial` (*float*, default: `0.15`): Spatial Potts consensus weight.
  - `output_dir` (*str*, default: `"astra_redo_Results"`): Root output folder for CSVs and plots.
  - `visualize` (*bool*, default: `True`): Toggles visualization generation.
- **Returns**: `None` (exports files to disk).
- **Exports**:
  - `ASTRA_Redo_{dataset_name}_results.csv`: Per-dataset multi-seed summary.
  - `ASTRA_Redo_all_results.csv`: Combined benchmark table across all datasets.
- **Callers**:
  - Called in `__main__` ([line 920](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L920)).

---

### `__main__` CLI Block
[astra-redo.py:906-930](file:///Users/imran/Developer/FYDP/astra-redo/astra-redo.py#L906-L930)

- **Purpose**: Defines CLI interface via `argparse.ArgumentParser`, parses terminal flags, and launches `run_experiment`.
- **Supported Flags**:
  - `--datasets`: Datasets to run (names or `'all'`).
  - `--seeds`: Space-separated integer seeds (e.g. `--seeds 42 1234 2024`).
  - `--pretrain_epochs`: Stage 1 epochs.
  - `--finetune_epochs`: Stage 2 epochs.
  - `--lr`: Base learning rate.
  - `--lambda_dec`: DEC loss weight.
  - `--lambda_spatial`: Spatial consensus weight.
  - `--output_dir`: Target results directory.
  - `--visualize`: Action store true flag for generating figures.
