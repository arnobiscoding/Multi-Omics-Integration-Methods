# Spatial Multi-Omics Dual-Encoder VGAT PoE DEC Architecture Specification

## Overview & Executive Summary

This package implements an end-to-end PyTorch / PyTorch Geometric deep learning architecture for spatial multi-omics integration. The model integrates Transcriptomic (RNA) and Proteomic (ADT) modalities using a **Dual-Encoder Variational Graph Attention Network (VGAT)**, **Product of Experts (PoE) Fusion**, **Dual Decoders (ZINB + MSE)** for biologically accurate reconstruction, and a **Joint DEC / Pseudo-Label Spatially-Aware Contrastive Objective Framework**.

---

## 1. Input Formulation & Preprocessing

The model processes modality-specific node features across a shared spatial topology graph constructed from physical spot coordinates.

### Inputs
1. **RNA Node Features ($X_{\text{RNA}}$)**: $X_{\text{RNA}} \in \mathbb{R}^{N \times F_1}$, representing the top $F_1$ Spatially Variable Genes (SVGs) or Highly Variable Genes (HVGs) across $N$ spatial spots.
2. **ADT Node Features ($X_{\text{ADT}}$)**: $X_{\text{ADT}} \in \mathbb{R}^{N \times F_2}$, representing $F_2$ normalized protein abundance features.
3. **Spatial Graph Topology ($A$)**: An unweighted, undirected edge index tensor $A \in \mathbb{R}^{2 \times E}$ representing a $K$-Nearest Neighbors (KNN) spatial graph ($K=4$ or $K=6$) built strictly from physical $(X, Y)$ spot coordinates.

$$\text{Graph Topology: } A = (V, E), \quad e_{ij} \in E \iff j \in \text{KNN}(i) \lor i \in \text{KNN}(j)$$

---

## 2. Dual-Encoder VGAT Architecture

Two independent Variational Graph Attention Network (VGAT) encoders process RNA and ADT modalities across the shared spatial graph $A$.

### Layer Propagation Equation
For node $i$ and layer $k$:

$$h_{i}^{(k)} = \sigma \left( \sum_{j \in \mathcal{N}(i) \cup \{i\}} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} h_j^{(k-1)} \right)$$

### Dynamic Spatial Attention Coefficients ($\alpha_{ij}$)

$$\alpha_{ij}^{(k)} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i \parallel \mathbf{W} h_j]\right)\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W} h_i \parallel \mathbf{W} h_k]\right)\right)}$$

### Variational Output Distributions
Each encoder outputs mean and log-variance parameters mapping to latent space $\mathbb{R}^{N \times D}$:
- **RNA Encoder**: $\mu_{\text{RNA}}$ and $\log(\sigma^2_{\text{RNA}})$
- **ADT Encoder**: $\mu_{\text{ADT}}$ and $\log(\sigma^2_{\text{ADT}})$

---

## 3. Product of Experts (PoE) Multimodal Fusion

The `PoEFusion` module computes the joint latent Gaussian distribution by taking the product of RNA and ADT expert posteriors.

### 1. Fused Variance ($\sigma^2_f$)

$$\sigma^2_f = \left( \frac{1}{\sigma^2_{\text{RNA}}} + \frac{1}{\sigma^2_{\text{ADT}}} \right)^{-1} = \frac{\sigma^2_{\text{RNA}} \odot \sigma^2_{\text{ADT}}}{\sigma^2_{\text{RNA}} + \sigma^2_{\text{ADT}}}$$

### 2. Fused Mean ($\mu_f$)

$$\mu_f = \sigma^2_f \odot \left( \frac{\mu_{\text{RNA}}}{\sigma^2_{\text{RNA}}} + \frac{\mu_{\text{ADT}}}{\sigma^2_{\text{ADT}}} \right)$$

### 3. Reparameterization Trick ($Z_f$)

$$Z_f = \mu_f + \epsilon \odot \sqrt{\sigma^2_f} \quad \text{where} \quad \epsilon \sim \mathcal{N}(0, I)$$

---

## 4. Dual-Decoder & Reconstruction Loss ($\mathcal{L}_{Recon}$)

To prevent latent space collapse and denoise inputs, $Z_f$ is passed through modality-specific decoders.

### 1. RNA Decoder ($g_{\phi 1}$) with Zero-Inflated Negative Binomial (ZINB) Loss
Models sparse RNA counts with dropouts ($\pi$) and overdispersion ($\theta$):

$$\hat{X}_{\text{RNA}} = g_{\phi 1}(Z_f) \implies (\mu_{\text{RNA\_hat}}, \theta_{\text{RNA}}, \pi_{\text{RNA}})$$

$$\mathcal{L}_{ZINB\_RNA} = -\frac{1}{N \cdot F_1} \sum_{i=1}^N \sum_{j=1}^{F_1} \log P_{\text{ZINB}}(x_{ij} \mid \mu_{ij}, \theta_{ij}, \pi_{ij})$$

### 2. ADT Decoder ($g_{\phi 2}$) with Mean Squared Error (MSE) Loss
Reconstructs continuous ADT protein levels:

$$\hat{X}_{\text{ADT}} = g_{\phi 2}(Z_f), \quad \mathcal{L}_{MSE\_ADT} = \frac{1}{N \cdot F_2} \sum_{i=1}^N \| x_{i, \text{ADT}} - \hat{x}_{i, \text{ADT}} \|^2_2$$

### 3. Total Reconstruction Loss

$$\mathcal{L}_{Recon} = \mathcal{L}_{ZINB\_RNA} + \lambda_{ADT} \mathcal{L}_{MSE\_ADT}$$

---

## 5. Dual-Head Optimization Framework

### A. Deep Embedded Clustering (DEC) Head
Maintains learnable parameter matrix of $K$ cluster centers $C \in \mathbb{R}^{K \times D}$.

1. **Soft Assignments ($Q$)**:

$$q_{ij} = \frac{\left(1 + \|(Z_f)_i - C_j\|^2\right)^{-1}}{\sum_{k=1}^K \left(1 + \|(Z_f)_i - C_k\|^2\right)^{-1}}$$

2. **Sharpened Target Distribution ($P$)**:

$$p_{ij} = \frac{q_{ij}^2 / \sum_i q_{ij}}{\sum_k (q_{ik}^2 / \sum_i q_{ik})}$$

3. **Clustering Loss ($\mathcal{L}_{DEC}$)**:

$$\mathcal{L}_{DEC} = D_{KL}(P \parallel Q) = \sum_{i=1}^N \sum_{j=1}^K p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### B. Pseudo-Label Spatially-Aware Contrastive Head
Projects $Z_f$ through a 2-layer MLP to obtain $Z_{\text{proj}}$. Dynamic masks are computed based on soft assignments $Q$ and high-confidence threshold $\tau = 0.95$.

For central spot $i$ and physical neighbor $j$ ($A_{ij} = 1$):
- **Strict Positive Mask ($M_{pos}$)**:

$$M_{pos}(i,j) = 1 \iff \max(q_i) > \tau \land \max(q_j) > \tau \land \operatorname{argmax}(q_i) == \operatorname{argmax}(q_j)$$

- **Strict Negative Mask ($M_{neg}$)**:

$$M_{neg}(i,j) = 1 \iff \max(q_i) > \tau \land \max(q_j) > \tau \land \operatorname{argmax}(q_i) \neq \operatorname{argmax}(q_j)$$

Masked InfoNCE loss pulls $M_{pos}$ pairs together while penalizing $M_{neg}$ pairs:

$$\mathcal{L}_{InfoNCE\_Pseudo} = -\frac{1}{|M_{pos}|} \sum_{(i,j) \in M_{pos}} \log \frac{\exp(\operatorname{sim}(z_i, z_j)/\tau_{temp})}{\exp(\operatorname{sim}(z_i, z_j)/\tau_{temp}) + \sum_{k \in M_{neg}(i)} \exp(\operatorname{sim}(z_i, z_k)/\tau_{temp})}$$

---

## 6. Training Execution Protocol

```
+-----------------------------------------------------------------------+
| PHASE 1: Reconstruction & Spatial InfoNCE Warm-Up (100 Epochs)        |
| - DEC Head disabled.                                                  |
| - Optimize: L_total = L_Recon + alpha * L_InfoNCE_Spatial             |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| PHASE 2: Center Initialization (Freeze Network Weights)               |
| - Extract stabilized latent mean mu_f.                                |
| - Run K-Means++ to initialize K cluster centers C in DEC head.        |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| PHASE 3: Joint Fine-Tuning                                            |
| - Unfreeze network weights.                                           |
| - Composite Loss: L_total = L_Recon + alpha * L_Pseudo + gamma * L_DEC |
| - Target P distribution updated every 5 epochs.                      |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
| PHASE 4: Spatial Early Stopping Subsampling                           |
| - Evaluated every 10 epochs on 5% spot subsample.                     |
| - Composite Score: M = lambda1 * Silhouette + lambda2 * Moran's I     |
| - Model weights saved exclusively at peak composite score M.          |
+-----------------------------------------------------------------------+
```

---

## 7. Package Codebase Mapping

| File Name | Responsibility | Key Classes / Functions |
| :--- | :--- | :--- |
| [`preprocessing.py`](file:///d:/FYDP/GATCON/d1_test/new%20model/preprocessing.py) | Graph topology & feature tensor preparation | `build_spatial_knn_graph`, `prepare_spatial_multiomics_data` |
| [`vgat_encoder.py`](file:///d:/FYDP/GATCON/d1_test/new%20model/vgat_encoder.py) | Dual VGAT graph attention encoders | `VGATEncoder`, `DualVGATEncoder`, `CustomVGATLayer` |
| [`poe_fusion.py`](file:///d:/FYDP/GATCON/d1_test/new%20model/poe_fusion.py) | Product of Experts multimodal fusion | `PoEFusion` |
| [`decoders_and_losses.py`](file:///d:/FYDP/GATCON/d1_test/new%20model/decoders_and_losses.py) | RNA (ZINB) & ADT (MSE) decoders and losses | `RNADecoder`, `ADTDecoder`, `ZINBLoss`, `DualDecoderReconstructionLoss` |
| [`clustering_and_contrastive.py`](file:///d:/FYDP/GATCON/d1_test/new%20model/clustering_and_contrastive.py) | DEC clustering head & pseudo-label contrastive loss | `DECHead`, `PseudoLabelContrastiveHead` |
| [`metrics.py`](file:///d:/FYDP/GATCON/d1_test/new%20model/metrics.py) | Silhouette, Moran's I & early stopping metrics | `compute_silhouette`, `compute_morans_i`, `compute_composite_spatial_metric` |
| [`model.py`](file:///d:/FYDP/GATCON/d1_test/new%20model/model.py) | Master pipeline PyTorch module | `VGAT_PoE_DEC` |
| [`trainer.py`](file:///d:/FYDP/GATCON/d1_test/new%20model/trainer.py) | 4-phase execution engine | `SpatialOmicsTrainer` |
| [`__init__.py`](file:///d:/FYDP/GATCON/d1_test/new%20model/__init__.py) | Package initialization | Exported classes & modules |
| [`run_pipeline.py`](file:///d:/FYDP/GATCON/d1_test/new%20model/run_pipeline.py) | Executable CLI runner & synthetic benchmark | `run_pipeline`, `generate_synthetic_data` |
| [`requirements.txt`](file:///d:/FYDP/GATCON/d1_test/new%20model/requirements.txt) | Python dependencies specification | PyTorch, PyG, Scanpy, SciPy, scikit-learn |

---

## 8. Usage Instructions

### Synthetic Benchmark Verification
```bash
python "new model/run_pipeline.py" --synthetic --num_spots 500 --num_clusters 7
```

### Python API Integration
```python
from new_model import prepare_spatial_multiomics_data, VGAT_PoE_DEC, SpatialOmicsTrainer

# 1. Prepare data dictionary
data_dict = prepare_spatial_multiomics_data(
    x_rna=x_rna,
    x_adt=x_adt,
    coords=spot_coordinates,
    k_neighbors=6,
    device="cuda"
)

# 2. Instantiate master architecture
model = VGAT_PoE_DEC(
    in_dim_rna=data_dict["f_rna"],
    in_dim_adt=data_dict["f_adt"],
    num_clusters=10,
    hidden_dim=128,
    latent_dim=32
)

# 3. Train using 4-phase execution engine
trainer = SpatialOmicsTrainer(
    model=model,
    learning_rate=1e-3,
    warmup_epochs=100,
    finetune_epochs=100,
    device="cuda"
)

results = trainer.fit(data_dict)
latent_embeddings = results["z_f"]
cluster_predictions = results["predictions"]
```
