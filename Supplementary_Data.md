# Supplementary Data: Extended Derivations, Proofs, and Full Formulations for DualGate-DEC

This supplementary document provides complete mathematical derivations, formal propositions and proofs, expanded objective formulations, and algorithmic specifications accompanying Sections 2.6–2.9 of [`Fusion_Decoder_DEC_Training_Objective.md`](file:///d:/FYDP/GATCON/d1_test/Fusion_Decoder_DEC_Training_Objective.md) for the **DualGate-DEC** architecture (`DualGate_DEC`, `DualGCN_Gated`, `GatedAttentionFusion`, and `SpatialPottsDEC`).

---

## S1. Hierarchical Spot-Adaptive Gated Attention Fusion: Extended Formulations and Geometric Properties

Following dual-graph message passing ($k = 15$ nearest neighbors), let $\mathbf{Z}_{\text{sim}}, \mathbf{Z}_{\text{dist}} \in \mathbb{R}^{N \times d}$ ($d = 64$) denote the transcriptomic similarity and spatial-proximity latent matrices produced by the two-layer RNA Graph Convolutional Network (GCN) encoders (`x_RNA1` $\to$ `sim_conv` and `x_RNA2` $\to$ `dist_conv`), and let $\mathbf{Z}_{\text{pro}} \in \mathbb{R}^{N \times d}$ denote the auxiliary-modality embedding produced by the single-layer GCN encoder (`protein3`) over the unweighted shared-edge intersection graph $\mathcal{G}_{\text{common}} = \mathcal{G}_{\text{sim}} \cap \mathcal{G}_{\text{dist}}$.

### S1.1 Tier 1: Intra-Modality Spatial–Transcriptomic Gating

For each spot $i \in \{1, \dots, N\}$, the first `GatedAttentionFusion` module concatenates the two RNA embeddings and computes a continuous, feature-wise spatial boundary gate $\mathbf{g}_{\text{spatial}, i} \in (0, 1)^d$ (Main Paper Eq. (1)):

$$\mathbf{g}_{\text{spatial}, i} = \sigma\left( \mathbf{W}_{g1} [\mathbf{z}_{\text{sim}, i} \parallel \mathbf{z}_{\text{dist}, i}] + \mathbf{b}_{g1} \right) \tag{S1}$$

where $\mathbf{W}_{g1} \in \mathbb{R}^{d \times 2d}$, $\mathbf{b}_{g1} \in \mathbb{R}^d$, $[\cdot \parallel \cdot]$ denotes vector concatenation, and $\sigma(u) = (1 + e^{-u})^{-1}$ is applied element-wise. The gated convex combination and post-gating linear projection (Main Paper Eq. (2)) are defined sequentially as:

$$\tilde{\mathbf{z}}_{\text{fused}, i} = \mathbf{g}_{\text{spatial}, i} \odot \mathbf{z}_{\text{sim}, i} + (\mathbf{1} - \mathbf{g}_{\text{spatial}, i}) \odot \mathbf{z}_{\text{dist}, i} \tag{S2}$$

$$\mathbf{z}_{\text{fused}, i} = \mathbf{W}_{o1} \tilde{\mathbf{z}}_{\text{fused}, i} + \mathbf{b}_{o1} \tag{S3}$$

where $\odot$ denotes the Hadamard product, $\mathbf{W}_{o1} \in \mathbb{R}^{d \times d}$, and $\mathbf{b}_{o1} \in \mathbb{R}^d$. Stacking spot vectors row-wise yields the Tier 1 latent matrix $\mathbf{Z}_{\text{fused}} \in \mathbb{R}^{N \times d}$.

### S1.2 Tier 2: Cross-Modality Multi-Omics Gating

Given $\mathbf{z}_{\text{fused}, i}$ from Tier 1 and the auxiliary embedding $\mathbf{z}_{\text{pro}, i}$, the second `GatedAttentionFusion` module computes a spot-adaptive modality gate $\mathbf{g}_{\text{modality}, i} \in (0, 1)^d$ (Main Paper Eq. (3)):

$$\mathbf{g}_{\text{modality}, i} = \sigma\left( \mathbf{W}_{g2} [\mathbf{z}_{\text{fused}, i} \parallel \mathbf{z}_{\text{pro}, i}] + \mathbf{b}_{g2} \right) \tag{S4}$$

with learnable parameters $\mathbf{W}_{g2} \in \mathbb{R}^{d \times 2d}$ and $\mathbf{b}_{g2} \in \mathbb{R}^d$. The intermediate cross-modal interpolation and final affine refinement (Main Paper Eq. (4)) yield the unified multi-omics embedding $\mathbf{z}_{\text{fused\_pro}, i} \in \mathbb{R}^d$:

$$\tilde{\mathbf{z}}_{\text{fused\_pro}, i} = \mathbf{g}_{\text{modality}, i} \odot \mathbf{z}_{\text{fused}, i} + (\mathbf{1} - \mathbf{g}_{\text{modality}, i}) \odot \mathbf{z}_{\text{pro}, i} \tag{S5}$$

$$\mathbf{z}_{\text{fused\_pro}, i} = \mathbf{W}_{o2} \tilde{\mathbf{z}}_{\text{fused\_pro}, i} + \mathbf{b}_{o2} \tag{S6}$$

where $\mathbf{W}_{o2} \in \mathbb{R}^{d \times d}$ and $\mathbf{b}_{o2} \in \mathbb{R}^d$. Collectively, $\mathbf{Z}_{\text{fused\_pro}} \in \mathbb{R}^{N \times d}$ forms the unified spot-level representation passed to `SpatialPottsDEC` and joint reconstruction.

### S1.3 Geometric Analysis of Post-Gating Linear Projection

**Proposition S1 (Convex-Hull Relaxation and Spatial Regime Specializations).** *For any spot $i$, the intermediate vector $\tilde{\mathbf{z}}_{i} = \mathbf{g}_i \odot \mathbf{a}_i + (\mathbf{1} - \mathbf{g}_i) \odot \mathbf{b}_i$ with $\mathbf{g}_i \in (0, 1)^d$ is strictly confined to the coordinate-wise hyper-rectangle spanned by $\mathbf{a}_i$ and $\mathbf{b}_i$, whereas the projected representation $\mathbf{z}_i = \mathbf{W}_o \tilde{\mathbf{z}}_i + \mathbf{b}_o$ spans any affine image in $\mathbb{R}^d$ whenever $\mathbf{W}_o$ is full rank. Moreover, static slide-wide fusion arises as the degenerate zero-weight special case $\mathbf{W}_g = \mathbf{0}$.*

*Proof.* For each coordinate $c \in \{1, \dots, d\}$, since $g_{ic} \in (0, 1)$, the convex combination satisfies
$$\min(a_{ic}, b_{ic}) \le g_{ic} a_{ic} + (1 - g_{ic}) b_{ic} = \tilde{z}_{ic} \le \max(a_{ic}, b_{ic}).$$
Consequently, without $\mathbf{W}_o$, no channel of the fused embedding can represent constructive cross-branch interactions outside $[\min(a_{ic}, b_{ic}), \max(a_{ic}, b_{ic})]$ or rotate correlated features across dimensions. Applying an invertible affine map $\mathbf{W}_o \in \mathbb{R}^{d \times d}$ mixes coordinates via $z_{ic} = \sum_{r=1}^d W_{o,cr}\tilde{z}_{ir} + b_{oc}$, lifting the representation out of the axis-aligned interval box while preserving differentiability. Finally, if $\mathbf{W}_g = \mathbf{0}$, then $\mathbf{g}_i = \sigma(\mathbf{b}_g)$ is identical across all $N$ spots, reducing Eq. (S2) to static channel-wise averaging (and to isotropic scalar weighting $\alpha \mathbf{a}_i + (1-\alpha)\mathbf{b}_i$ when $\mathbf{b}_g = b_0 \mathbf{1}$). $\square$

---

## S2. Decoupled Multi-Task Reconstruction Decoders: Parameterization and Collapse Proof

### S2.1 Shared Semantic Dictionary and Linear Reconstruction Heads

All four reconstruction pathways share a single hidden dictionary layer (`deconv1`) parameterized by $\mathbf{W}_{\text{dec}1} \in \mathbb{R}^{512 \times d}$ and $\mathbf{b}_{\text{dec}1} \in \mathbb{R}^{512}$ ($d = 64$), mapping any $N \times d$ latent matrix $\mathbf{Z}$ into a unified 512-dimensional semantic subspace (Main Paper Eq. (5)):

$$\mathbf{h}(\mathbf{Z}) = \operatorname{ReLU}\left( \mathbf{Z} \mathbf{W}_{\text{dec}1}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}1}^T \right) \in \mathbb{R}^{N \times 512} \tag{S7}$$

where $\mathbf{1}_N \in \mathbb{R}^{N \times 1}$ broadcasts the bias across all $N$ spots. From $\mathbf{h}(\cdot)$, four linear heads (`deconv5`, `deconv2`, `deconv4`) reconstruct the multi-omics and branch-specific feature matrices (Main Paper Eqs. (6)–(7)):

1. **Joint Multimodal Head** (`deconv5`, $\mathbf{W}_{\text{dec}5} \in \mathbb{R}^{(G + D_{\text{aux}}) \times 512}$, $\mathbf{b}_{\text{dec}5} \in \mathbb{R}^{G + D_{\text{aux}}}$):
   $$\hat{\mathbf{X}}_{\text{joint}} = \mathbf{h}(\mathbf{Z}_{\text{fused\_pro}}) \mathbf{W}_{\text{dec}5}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}5}^T \in \mathbb{R}^{N \times (G + D_{\text{aux}})} \tag{S8}$$

2. **Weight-Tied Transcriptomic Similarity Head** (`deconv2`, $\mathbf{W}_{\text{dec}2} \in \mathbb{R}^{G \times 512}$, $\mathbf{b}_{\text{dec}2} \in \mathbb{R}^{G}$):
   $$\hat{\mathbf{X}}_{\text{sim}} = \mathbf{h}(\mathbf{Z}_{\text{sim}}) \mathbf{W}_{\text{dec}2}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}2}^T \in \mathbb{R}^{N \times G} \tag{S9}$$

3. **Weight-Tied Transcriptomic Spatial Head** (sharing `deconv2`, i.e., $\mathbf{W}_{\text{dec}2}, \mathbf{b}_{\text{dec}2}$):
   $$\hat{\mathbf{X}}_{\text{dist}} = \mathbf{h}(\mathbf{Z}_{\text{dist}}) \mathbf{W}_{\text{dec}2}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}2}^T \in \mathbb{R}^{N \times G} \tag{S10}$$

4. **Auxiliary Modality Head** (`deconv4`, $\mathbf{W}_{\text{dec}4} \in \mathbb{R}^{D_{\text{aux}} \times 512}$, $\mathbf{b}_{\text{dec}4} \in \mathbb{R}^{D_{\text{aux}}}$):
   $$\hat{\mathbf{X}}_{\text{aux}} = \mathbf{h}(\mathbf{Z}_{\text{pro}}) \mathbf{W}_{\text{dec}4}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}4}^T \in \mathbb{R}^{N \times D_{\text{aux}}} \tag{S11}$$

Weight tying of $(\mathbf{W}_{\text{dec}2}, \mathbf{b}_{\text{dec}2})$ across Eqs. (S9) and (S10) ensures that the cosine-similarity branch ($\mathbf{Z}_{\text{sim}}$) and the spatial-distance branch ($\mathbf{Z}_{\text{dist}}$) project onto an identical canonical gene-expression basis prior to Tier 1 fusion.

### S2.2 Proof of Auxiliary Collapse Prevention via Decoupled Normalization

**Proposition S2 (Gradient-Scale Imbalance in Concatenated vs. Decoupled Reconstruction).** *Let $G = \operatorname{dim}(\mathbf{X}_{\text{RNA}}) = 3000$ and $D_{\text{aux}} = \operatorname{dim}(\mathbf{X}_{\text{aux}}) \ll G$ (specifically, $D_{\text{aux}} = 31$ for CLR-normalized ADT panels and $D_{\text{aux}} = 50$ for PCA-reduced ATAC profiles). Under a single concatenated mean-squared error loss $\mathcal{L}_{\text{rec}}$ alone, the auxiliary error fraction scales as $\frac{D_{\text{aux}}}{G + D_{\text{aux}}}$ under equal per-feature residual variance, suppressing auxiliary contribution to $1.02\%$–$1.64\%$ while letting transcriptomics dominate $98.36\%$–$98.98\%$. Decoupling $\mathcal{L}_{\text{aux}}$ normalized by $(N D_{\text{aux}})^{-1}$ restores $\mathcal{O}(1)$ per-feature loss and gradient feedback directly to $\mathbf{Z}_{\text{pro}}$.*

*Proof.* Under the concatenated objective $\mathcal{L}_{\text{rec}} = \frac{1}{N(G + D_{\text{aux}})} \left( \|\mathbf{X}_{\text{RNA}} - \hat{\mathbf{X}}_{\text{joint, RNA}}\|_F^2 + \|\mathbf{X}_{\text{aux}} - \hat{\mathbf{X}}_{\text{joint, aux}}\|_F^2 \right)$, suppose standardized per-feature residuals satisfy $\mathbb{E}[(X_{\text{RNA},ij} - \hat{X}_{\text{joint},ij})^2] \approx \mathbb{E}[(X_{\text{aux},ik} - \hat{X}_{\text{joint},ik})^2] = \bar{\epsilon}^2$. Then the expected loss decomposes into
$$\mathbb{E}[\mathcal{L}_{\text{rec}}] = \frac{G}{G + D_{\text{aux}}} \bar{\epsilon}^2 + \frac{D_{\text{aux}}}{G + D_{\text{aux}}} \bar{\epsilon}^2.$$
For $G = 3000$ and $D_{\text{aux}} \in \{31, 50\}$, $\frac{D_{\text{aux}}}{G + D_{\text{aux}}} \in \{0.0102, 0.0164\}$; thus $98.36\%$ to $98.98\%$ of the concatenated reconstruction objective is governed by RNA residuals, permitting $\mathbf{Z}_{\text{pro}}$ to collapse or be ignored by the Tier 2 gate ($\mathbf{g}_{\text{modality}} \to \mathbf{1}$). By contrast, $\mathcal{L}_{\text{aux}} = \frac{1}{N D_{\text{aux}}} \|\mathbf{X}_{\text{aux}} - \hat{\mathbf{X}}_{\text{aux}}\|_F^2$ divides strictly by $N D_{\text{aux}}$ and depends exclusively on $\mathbf{Z}_{\text{pro}}$, ensuring $\mathbb{E}[\mathcal{L}_{\text{aux}}] = \bar{\epsilon}^2$ independently of $G$ and supplying non-vanishing gradients $\nabla_{\mathbf{Z}_{\text{pro}}} \mathcal{L}_{\text{aux}}$. Analogously, $\mathcal{L}_{\text{sim}}$ and $\mathcal{L}_{\text{dist}}$ prevent one RNA branch from degenerating while the other compensates. $\square$

---

## S3. Spatial Potts Markov Random Field Clustering: Complete Formulation

### S3.1 Student's $t$-Distribution Soft Assignment and Frequency Balancing

Let $\mathbf{Z} = \mathbf{Z}_{\text{fused\_pro}} \in \mathbb{R}^{N \times d}$ denote the unified latent embedding matrix and $\mathbf{M} = [\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K]^T \in \mathbb{R}^{K \times d}$ denote the $K$ trainable cluster prototypes (`cluster_centers`). In general DEC, the soft assignment probability $q_{ik}$ of spot $i$ to cluster $k$ is modeled via Student's $t$-distribution with $\alpha > 0$ degrees of freedom:

$$q_{ik} = \frac{\left( 1 + \frac{\|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2}{\alpha} \right)^{-\frac{\alpha + 1}{2}}}{\sum_{k'=1}^K \left( 1 + \frac{\|\mathbf{z}_i - \boldsymbol{\mu}_{k'}\|_2^2}{\alpha} \right)^{-\frac{\alpha + 1}{2}} + 10^{-12}} \tag{S12}$$

Setting $\alpha = 1.0$ yields the Cauchy kernel (Main Paper Eq. (8)):

$$q_{ik} = \frac{\left( 1 + \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2 \right)^{-1}}{\sum_{k'=1}^K \left( 1 + \|\mathbf{z}_i - \boldsymbol{\mu}_{k'}\|_2^2 \right)^{-1} + 10^{-12}} \tag{S13}$$

To self-sharpen high-confidence predictions while preventing large clusters from absorbing rare micro-domains, the auxiliary DEC target distribution $\mathbf{P}_{\text{dec}} \in \mathbb{R}^{N \times K}$ squares soft assignments and normalizes by empirical cluster mass $f_k = \sum_{i=1}^N q_{ik} + 10^{-12}$ (Main Paper Eq. (9)):

$$p_{\text{dec}, ik} = \frac{q_{ik}^2 / f_k}{\sum_{k'=1}^K \left( q_{ik'}^2 / f_{k'} \right) + 10^{-12}} \tag{S14}$$

### S3.2 Statistical-Physics Derivation of Potts MRF Modulation

Consider a $K$-state Potts Markov Random Field (MRF) defined over discrete spot labels $\mathbf{s} = (s_1, \dots, s_N) \in \{1, \dots, K\}^N$ on the spatial graph $\mathcal{G}_{\text{dist}} = (\mathcal{V}, \mathcal{E})$, governed by the Hamiltonian:

$$H(\mathbf{s}) = -\sum_{i=1}^N \sum_{j=1}^N J_{ij}\,\delta(s_i, s_j) - \sum_{i=1}^N h_i(s_i) \tag{S15}$$

where $\delta(\cdot, \cdot)$ is the Kronecker delta, $J_{ij} \ge 0$ is the normalized ferromagnetic spatial coupling strength, and $h_i(s_i)$ represents the local unary evidence from latent embeddings. Under the Gibbs distribution $P(\mathbf{s}) \propto \exp(-H(\mathbf{s}) / T)$ with temperature $T > 0$, a mean-field variational approximation replaces each neighbor's indicator $\delta(s_j, k)$ by its current soft assignment $q_{jk}$ and identifies the unary prior term $\exp(h_i(k)/T)$ with the latent-space DEC target $p_{\text{dec}, ik}$. Setting $J_{ij}/T = \lambda_{\text{spatial}} A_{\text{norm}, ij}$ yields the local mean-field posterior:

$$p_{\text{final}, ik} \propto p_{\text{dec}, ik} \exp\!\left( \lambda_{\text{spatial}} \sum_{j=1}^N A_{\text{norm}, ij} q_{jk} \right) \tag{S16}$$

To instantiate the transition weights $A_{\text{norm}, ij}$ (`compute_spatial_adj_norm`), we form the unweighted binary spatial adjacency $\mathbf{A}_{\text{dist}} \in \{0, 1\}^{N \times N}$ ($k = 15$, without self-loops), add self-loops $\mathbf{I}_N$, and apply row-stochastic degree normalization (Main Paper Eq. (10)):

$$\tilde{\mathbf{A}}_{\text{dist}} = \mathbf{A}_{\text{dist}} + \mathbf{I}_N, \qquad \mathbf{D}_{ii} = \sum_{j=1}^N \tilde{A}_{\text{dist}, ij}, \qquad \mathbf{A}_{\text{norm}} = (\mathbf{D} + 10^{-12}\mathbf{I}_N)^{-1} \tilde{\mathbf{A}}_{\text{dist}} \tag{S17}$$

Multiplying $\mathbf{A}_{\text{norm}}$ by $\mathbf{Q}$ produces the neighborhood consensus matrix $\mathbf{C} \in \mathbb{R}^{N \times K}$ (Main Paper Eq. (10)):

$$\mathbf{C} = \mathbf{A}_{\text{norm}} \mathbf{Q}, \qquad C_{ik} = \sum_{j=1}^N A_{\text{norm}, ij} q_{jk} \tag{S18}$$

Modulating $\mathbf{P}_{\text{dec}}$ by $\exp(\lambda_{\text{spatial}} C_{ik})$ and renormalizing across clusters produces the final spatially regularized target distribution $\mathbf{P}_{\text{final}} \in \mathbb{R}^{N \times K}$ (Main Paper Eq. (11)):

$$p_{\text{spatial}, ik} = p_{\text{dec}, ik} \exp\left( \lambda_{\text{spatial}} C_{ik} \right), \qquad p_{\text{final}, ik} = \frac{p_{\text{spatial}, ik}}{\sum_{k'=1}^K p_{\text{spatial}, ik'} + 10^{-12}} \tag{S19}$$

where $\lambda_{\text{spatial}} = 0.15$ balances spatial domain continuity against sharp transcriptomic boundary preservation.

---

## S4. Complete Loss Formulations, Analytical Gradients, and Attenuation Proof

### S4.1 Expanded Multi-Task Reconstruction Losses

To ensure that neither the joint bottleneck nor any individual encoder branch collapses under extreme transcriptomic dimensionality ($G = 3000$ vs. $D_{\text{aux}} \in \{31, 50\}$), reconstruction error is penalized simultaneously across the unified embedding and all three modality-specific branches using mean-squared error (`F.mse_loss`). Using the four decoder predictions in Eqs. (S8)–(S11), the individual reconstruction terms are:

$$\mathcal{L}_{\text{rec}} = \frac{1}{N (G + D_{\text{aux}})} \sum_{i=1}^N \sum_{j=1}^{G + D_{\text{aux}}} \left( [\mathbf{X}_{\text{RNA}} \parallel \mathbf{X}_{\text{aux}}]_{ij} - \hat{X}_{\text{joint}, ij} \right)^2 \tag{S20}$$

$$\mathcal{L}_{\text{sim}} = \frac{1}{N G} \sum_{i=1}^N \sum_{j=1}^{G} \left( X_{\text{RNA}, ij} - \hat{X}_{\text{sim}, ij} \right)^2, \qquad \mathcal{L}_{\text{dist}} = \frac{1}{N G} \sum_{i=1}^N \sum_{j=1}^{G} \left( X_{\text{RNA}, ij} - \hat{X}_{\text{dist}, ij} \right)^2 \tag{S21}$$

$$\mathcal{L}_{\text{aux}} = \frac{1}{N D_{\text{aux}}} \sum_{i=1}^N \sum_{k=1}^{D_{\text{aux}}} \left( X_{\text{aux}, ik} - \hat{X}_{\text{aux}, ik} \right)^2 \tag{S22}$$

Here, $\mathcal{L}_{\text{rec}}$ constrains $\mathbf{Z}_{\text{fused\_pro}}$ to preserve joint multi-omic evidence, $\mathcal{L}_{\text{sim}}$ and $\mathcal{L}_{\text{dist}}$ supervise both RNA branches through weight-tied decoder $(\mathbf{W}_{\text{dec}2}, \mathbf{b}_{\text{dec}2})$, and $\mathcal{L}_{\text{aux}}$ supplies independent $(N D_{\text{aux}})^{-1}$-normalized feedback to $\mathbf{Z}_{\text{pro}}$ (Proposition S2). Aggregating Eqs. (S20)–(S22) with equal unity weighting yields the composite multi-task reconstruction loss used in Main Paper Eqs. (12)–(13):

$$\mathcal{L}_{\text{multi-rec}} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{sim}} + \mathcal{L}_{\text{dist}} + \mathcal{L}_{\text{aux}} \tag{S23}$$

### S4.2 Spatial Coherence and Elastic-Net Parameter Regularization

To encourage adjacent spots in physical tissue space to maintain correlated transcriptomic latent representations while separating non-neighbors, off-diagonal pairwise cosine similarities on the Tier 1 spatial–transcriptomic manifold $\mathbf{Z}_{\text{fused}}$ are mapped into edge probabilities via the sigmoid function (`spatial_regularization_loss`):

$$S_{ij} = \frac{\mathbf{z}_{\text{fused}, i}^T \mathbf{z}_{\text{fused}, j}}{\|\mathbf{z}_{\text{fused}, i}\|_2 \|\mathbf{z}_{\text{fused}, j}\|_2 + 10^{-12}} \cdot (1 - \delta_{ij}), \qquad \tilde{S}_{ij} = \sigma(S_{ij}) \in (0, 1) \tag{S24}$$

where $\delta_{ij}$ zeros self-similarities ($S_{ii} = 0$, $\tilde{S}_{ii} = 0.5$). Using the unweighted binary spatial adjacency $\mathbf{A}_{\text{dist}} \in \{0, 1\}^{N \times N}$, the symmetric binary cross-entropy spatial smoothness loss used in Main Paper Eqs. (12)–(13) is:

$$\mathcal{L}_{\text{spatial}} = -\frac{1}{2 N^2} \sum_{i=1}^N \sum_{j=1}^N \left[ A_{\text{dist}, ij} \ln\left(\tilde{S}_{ij} + 10^{-10}\right) + (1 - A_{\text{dist}, ij}) \ln\left(1 - \tilde{S}_{ij} + 10^{-10}\right) \right] \tag{S25}$$

Applying $\mathcal{L}_{\text{spatial}}$ directly on the Tier 1 RNA manifold $\mathbf{Z}_{\text{fused}}$ preserves tissue-domain contiguity without forcing auxiliary modalities into over-smoothed boundaries.

Model capacity and parameter sparsity across the active parameter set $\Theta$ are further controlled via combined elastic-net ($L_1$ and $L_2$) penalties used in Main Paper Eqs. (12)–(13):

$$\mathcal{L}_{\text{reg}}(\Theta) = \lambda_{L_1} \sum_{\theta \in \Theta} |\theta| + \lambda_{L_2} \sum_{\theta \in \Theta} \theta^2, \qquad \lambda_{L_1} = 10^{-4},\ \lambda_{L_2} = 10^{-3} \tag{S26}$$

### S4.3 Analytical Gradients of the Spatial Potts KL Divergence

During Stage 2 fine-tuning, clustering refinement minimizes the batch-averaged KL divergence (`F.kl_div(q.log(), p_target, reduction='batchmean')`) from the soft assignment matrix $\mathbf{Q}$ (Eq. (S13)) to the Potts-modulated spatial target $\mathbf{P}_{\text{final}}$ (Eq. (S19)) appearing in Main Paper Eq. (13):

$$\mathcal{L}_{\text{KL}}(\mathbf{Q} \parallel \mathbf{P}_{\text{final}}) = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p_{\text{final}, ik} \ln\left( \frac{p_{\text{final}, ik}}{q_{ik}} \right) \tag{S27}$$

**Proposition S3 (Exact Analytical Gradients of $\mathcal{L}_{\text{KL}}$).** *Treating $\mathbf{P}_{\text{final}}$ as fixed during backpropagation, the gradients of Eq. (S27) with respect to the spot embedding $\mathbf{z}_i$ and cluster centroid $\boldsymbol{\mu}_k$ are:*

$$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \mathbf{z}_i} = \frac{2}{N} \sum_{k=1}^K \left( 1 + \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2 \right)^{-1} \left( p_{\text{final}, ik} - q_{ik} \right) \left( \mathbf{z}_i - \boldsymbol{\mu}_k \right) \tag{S28}$$

$$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \boldsymbol{\mu}_k} = -\frac{2}{N} \sum_{i=1}^N \left( 1 + \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2 \right)^{-1} \left( p_{\text{final}, ik} - q_{ik} \right) \left( \mathbf{z}_i - \boldsymbol{\mu}_k \right) \tag{S29}$$

*Proof.* Define $u_{ik} = \left(1 + \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2\right)^{-1}$ and $Z_i = \sum_{k'=1}^K u_{ik'}$, so that $q_{ik} = u_{ik}/Z_i$. Differentiating $\ln q_{ik} = \ln u_{ik} - \ln Z_i$ with respect to $\mathbf{z}_i$ gives:
$$\frac{\partial \ln u_{ik}}{\partial \mathbf{z}_i} = -2 u_{ik} (\mathbf{z}_i - \boldsymbol{\mu}_k), \qquad \frac{\partial \ln Z_i}{\partial \mathbf{z}_i} = \frac{1}{Z_i} \sum_{k'=1}^K \frac{\partial u_{ik'}}{\partial \mathbf{z}_i} = -2 \sum_{k'=1}^K q_{ik'} u_{ik'} (\mathbf{z}_i - \boldsymbol{\mu}_{k'}).$$
Since $\mathcal{L}_{\text{KL}} = \frac{1}{N}\sum_{i,k} p_{\text{final},ik}(\ln p_{\text{final},ik} - \ln q_{ik})$ and $\sum_{k=1}^K p_{\text{final},ik} = 1$, applying the chain rule gives
$$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \mathbf{z}_i} = -\frac{1}{N} \sum_{k=1}^K p_{\text{final},ik} \left[ -2 u_{ik}(\mathbf{z}_i - \boldsymbol{\mu}_k) + 2 \sum_{k'=1}^K q_{ik'} u_{ik'}(\mathbf{z}_i - \boldsymbol{\mu}_{k'}) \right] = \frac{2}{N} \sum_{k=1}^K u_{ik} \left( p_{\text{final},ik} - q_{ik} \right) (\mathbf{z}_i - \boldsymbol{\mu}_k),$$
which establishes Eq. (S28). When $p_{\text{final},ik} > q_{ik}$, gradient descent $\mathbf{z}_i \leftarrow \mathbf{z}_i - \eta \frac{\partial \mathcal{L}_{\text{KL}}}{\partial \mathbf{z}_i}$ pulls $\mathbf{z}_i$ toward $\boldsymbol{\mu}_k$. Because $\nabla_{\boldsymbol{\mu}_k} u_{ik} = -\nabla_{\mathbf{z}_i} u_{ik}$, replacing $(\mathbf{z}_i - \boldsymbol{\mu}_k)$ by $-(\mathbf{z}_i - \boldsymbol{\mu}_k)$ and summing over $i = 1, \dots, N$ immediately yields Eq. (S29). $\square$

### S4.4 Gradient Equilibrium Proof for Stage-2 Reconstruction Attenuation ($\beta: 25.0 \to 5.0$)

**Theorem S1 (Bounded Clustering Torque and Exact $5\times$ Attenuation Scaling).** *For any spot embedding $\mathbf{z}_i \in \mathbb{R}^d$ and centroids $\{\boldsymbol{\mu}_k\}_{k=1}^K$, the Euclidean and Frobenius norms of the clustering gradient in Eq. (S28) satisfy the uniform upper bounds*

$$\left\|\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \mathbf{z}_i}\right\|_2 \le \frac{1}{N} \sum_{k=1}^K \left| p_{\text{final}, ik} - q_{ik} \right| \le \frac{2}{N}, \qquad \left\|\nabla_{\mathbf{Z}} \mathcal{L}_{\text{KL}}\right\|_F \le \frac{2}{\sqrt{N}}. \tag{S30}$$

*Consequently, while $\nabla_{\mathbf{Z}} \mathcal{L}_{\text{KL}}$ is strictly bounded above and shrinks further as $\mathbf{Q}$ approaches $\mathbf{P}_{\text{final}}$, $\beta(t)\,\mathcal{L}_{\text{multi-rec}}$ scales linearly with $\beta(t)$ across all four reconstruction heads. Attenuating $\beta(t)$ from $\beta_{\text{pre}} = 25.0$ in Stage 1 to $\beta_{\text{ft}} = 0.2\beta_{\text{pre}} = 5.0$ in Stage 2 reduces the weighted reconstruction-to-clustering gradient norm ratio $\frac{\|\beta(t)\nabla_{\mathbf{Z}}\mathcal{L}_{\text{multi-rec}}\|_F}{\|\lambda_{\text{dec}}\nabla_{\mathbf{Z}}\mathcal{L}_{\text{KL}}\|_F}$ by an exact factor of $5\times$ ($80\%$ reduction), enabling $\lambda_{\text{dec}} \mathcal{L}_{\text{KL}}$ to actively sharpen spatial boundaries while reconstruction acts as a biological manifold regularizer.*

*Proof.* Let $r_{ik} = \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2 \ge 0$. Because $(1 - r_{ik})^2 \ge 0 \implies 2 r_{ik} \le 1 + r_{ik}^2$, the scalar function $\psi(r) = \frac{r}{1 + r^2}$ attains its global maximum $\psi(1) = \frac{1}{2}$ for all $r \ge 0$. Applying the triangle inequality to Eq. (S28) yields:
$$\left\|\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \mathbf{z}_i}\right\|_2 \le \frac{2}{N} \sum_{k=1}^K \left| p_{\text{final}, ik} - q_{ik} \right| \frac{r_{ik}}{1 + r_{ik}^2} \le \frac{1}{N} \sum_{k=1}^K \left| p_{\text{final}, ik} - q_{ik} \right|.$$
Since $\mathbf{p}_{\text{final}, i}$ and $\mathbf{q}_i$ lie on the probability simplex $\Delta^{K-1}$, $\sum_{k=1}^K |p_{\text{final}, ik} - q_{ik}| \le \sum_k p_{\text{final}, ik} + \sum_k q_{ik} = 2$, establishing $\left\|\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \mathbf{z}_i}\right\|_2 \le \frac{2}{N}$. Summing squared norms over all $N$ spots gives $\|\nabla_{\mathbf{Z}} \mathcal{L}_{\text{KL}}\|_F \le \sqrt{N \cdot (2/N)^2} = \frac{2}{\sqrt{N}}$, confirming Eq. (S30). Moreover, after $k$-means bridge initialization, $\mathbf{Q}$ already approximates the cluster structure so $\sum_k |p_{\text{final}, ik} - q_{ik}| \ll 2$. Because the Stage 2 reconstruction term enters linearly as $\beta(t) \nabla_{\mathbf{Z}} \mathcal{L}_{\text{multi-rec}}$ while the clustering term enters as $\lambda_{\text{dec}} \nabla_{\mathbf{Z}} \mathcal{L}_{\text{KL}}$ with $\lambda_{\text{dec}} = 1.0$, changing $\beta(t)$ from $25.0$ to $0.2\beta_{\text{pre}} = 5.0$ lowers the relative gradient ratio
$$\frac{\|\beta_{\text{ft}} \nabla_{\mathbf{Z}} \mathcal{L}_{\text{multi-rec}}\|_F}{\|\lambda_{\text{dec}} \nabla_{\mathbf{Z}} \mathcal{L}_{\text{KL}}\|_F} = \frac{5.0}{25.0} \cdot \frac{\|\beta_{\text{pre}} \nabla_{\mathbf{Z}} \mathcal{L}_{\text{multi-rec}}\|_F}{\|\lambda_{\text{dec}} \nabla_{\mathbf{Z}} \mathcal{L}_{\text{KL}}\|_F} = \frac{1}{5} \cdot \frac{\|\beta_{\text{pre}} \nabla_{\mathbf{Z}} \mathcal{L}_{\text{multi-rec}}\|_F}{\|\lambda_{\text{dec}} \nabla_{\mathbf{Z}} \mathcal{L}_{\text{KL}}\|_F},$$
precisely decreasing reconstruction gradient dominance by $5\times$ without destabilizing the learned biological manifold. $\square$

### S4.5 Unified Epoch-Indexed Two-Stage Objective

Combining Main Paper Eqs. (12)–(14) into a single indicator-gated form over epochs $t \in \{1, \dots, 180\}$ gives:

$$\mathcal{L}_{\text{total}}(t; \Theta) = \beta(t)\,\mathcal{L}_{\text{multi-rec}} + \gamma\,\mathcal{L}_{\text{spatial}} + \delta\,\mathcal{L}_{\text{reg}}(\Theta) + \mathbb{I}(t > 80)\,\lambda_{\text{dec}}\,\mathcal{L}_{\text{KL}}(\mathbf{Q} \parallel \mathbf{P}_{\text{final}}) \tag{S31}$$

During Stage 1 ($1 \le t \le 80$, $\mathbb{I}(t > 80) = 0$), omitting $\mathcal{L}_{\text{KL}}$ prevents random initial prototypes $\mathbf{M}$ from distorting unaligned latent coordinates, while $\beta_{\text{pre}} = 25.0$ and $\eta_1 = 10^{-3}$ drive rapid feature-space and spatial alignment. During Stage 2 ($81 \le t \le 180$, $\mathbb{I}(t > 80) = 1$), attenuating $\beta_{\text{ft}} = 5.0$ and decelerating the learning rate $10\times$ to $\eta_2 = 10^{-4}$ ensures gentle, boundary-preserving co-adaptation between embeddings $\mathbf{Z}_{\text{fused\_pro}}$ and prototypes $\mathbf{M}$.

---

## S5. Two-Stage Peak Silhouette Bridge Protocol and Complete Algorithm

### S5.1 Stage 1 Diagnostic Tracking and 5-Step Bridge Handover (Epochs $1$–$80$)

In standard two-stage deep clustering, prototypes $\mathbf{M}$ are initialized directly from the final pre-training checkpoint ($t = 80$). However, prolonged MSE optimization frequently causes **checkpoint decay**, where late-stage pre-training overfits to technical sequencing noise and over-smooths cluster boundaries prior to epoch $80$. To prevent initializing DEC inside a degraded local minimum, at every Stage 1 epoch $t \in \{1, \dots, 80\}$ the model extracts $\mathbf{Z}^{(t)} = \mathbf{Z}_{\text{fused\_pro}}^{(t)} \in \mathbb{R}^{N \times d}$, executes $k$-means clustering $\mathbf{y}^{(t)} = \operatorname{KMeans}(\mathbf{Z}^{(t)}, K, n_{\text{init}}=10, \text{seed})$ (with optional `mclust` support), and evaluates the mean Silhouette Coefficient:

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}, \qquad \operatorname{Sil}^{(t)} = \frac{1}{N} \sum_{i=1}^N s(i), \qquad \Theta_1^* = \arg\max_{\Theta_1^{(t)},\, t \in \{1,\dots,80\}} \operatorname{Sil}^{(t)} \tag{S32}$$

where $a(i)$ is spot $i$'s mean intra-cluster latent distance and $b(i)$ is its mean distance to the nearest neighboring cluster. Across all 180 epochs ($t \in \{1, \dots, 180\}$), a global checkpoint $\Theta_{\text{best}}$ is simultaneously maintained whenever $\operatorname{Sil}^{(t)} > \operatorname{Sil}_{\text{global\_best}}$.

At the completion of epoch $80$, the **5-Step Bridge Handover Protocol** executes:

1. **Parameter Rewind:** Restore Stage 1 parameters to the peak Stage 1 state:
   $$\Theta_1 \leftarrow \Theta_1^* \tag{S33}$$
2. **Aligned Manifold Extraction:** Recompute the unified embedding under $\Theta_1^*$:
   $$\mathbf{Z}_{\text{aligned}} = f_{\text{encoder}}\left(\mathbf{X}_{\text{RNA}}, \mathbf{X}_{\text{aux}}, \mathcal{G}_{\text{sim}}, \mathcal{G}_{\text{dist}}, \mathcal{G}_{\text{common}}; \Theta_1^*\right) \in \mathbb{R}^{N \times 64} \tag{S34}$$
3. **High-Fidelity Prototype Discovery:** Run $k$-means with $n_{\text{init}} = 20$ restarts and the run's random seed on $\mathbf{Z}_{\text{aligned}}$:
   $$\{\boldsymbol{\mu}_1^*, \dots, \boldsymbol{\mu}_K^*\} = \arg\min_{\{\boldsymbol{\mu}_k\}_{k=1}^K} \sum_{i=1}^N \min_{k \in \{1, \dots, K\}} \|\mathbf{z}_{\text{aligned}, i} - \boldsymbol{\mu}_k\|_2^2 \tag{S35}$$
4. **DEC Prototype Ingestion:** Overwrite the trainable centroid matrix $\mathbf{M} = [\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K]^T \in \mathbb{R}^{K \times 64}$ with the discovered centroids:
   $$\mathbf{M} \leftarrow [\boldsymbol{\mu}_1^*, \dots, \boldsymbol{\mu}_K^*]^T \tag{S36}$$
5. **Optimizer Deceleration and Target Reset:** Instantiate a fresh Stage 2 Adam optimizer with $10\times$ reduced learning rate $\eta_2 = 0.1\,\eta_1 = 10^{-4}$ and clear $\mathbf{P}_{\text{final}}$ prior to epoch $81$.

### S5.2 Asynchronous Target Updates, Dual-Cluster Competition, and Checkpoint Restoration (Epochs $81$–$180$)

During Stage 2 fine-tuning ($t \in \{81, \dots, 180\}$) over $\Theta_2 = \Theta_1 \cup \{\mathbf{M}\}$, simultaneously updating $\mathbf{P}_{\text{final}}$ and $\mathbf{Q}$ at every step can induce circular Expectation–Maximization (EM) oscillations. Therefore, the Potts target distribution is refreshed asynchronously every $p_{\text{update\_interval}} = 5$ epochs (at $t \in \{81, 86, \dots, 176\}$, yielding 20 target refreshes across 100 epochs) inside `torch.no_grad()`:

$$\mathbf{P}_{\text{final}}^{(t)} = \begin{cases} \operatorname{StopGradient}\left( \mathcal{F}_{\text{Potts}}(\mathbf{Q}^{(t)}, \mathbf{A}_{\text{norm}}) \right), & \text{if } (t - 81) \equiv 0 \pmod{5} \\ \mathbf{P}_{\text{final}}^{(t-1)}, & \text{otherwise} \end{cases} \tag{S37}$$

At every Stage 2 epoch $t$, two candidate partitions are evaluated by Silhouette compactness to protect against premature prototype collapse and select the geometrically cleanest clustering:

1. **Candidate A (DEC Soft Assignments):** $\hat{y}_{q, i}^{(t)} = \arg\max_k q_{ik}^{(t)}$ with a degenerate-cluster guard requiring at least two unique predicted clusters:
   $$\operatorname{Sil}_q^{(t)} = \begin{cases} \operatorname{Silhouette}\left(\mathbf{Z}^{(t)}, \hat{\mathbf{y}}_q^{(t)}\right), & \text{if } \left|\left\{\hat{y}_{q, i}^{(t)} : i=1,\dots,N\right\}\right| \ge 2 \\ -1.0, & \text{otherwise} \end{cases} \tag{S38}$$
2. **Candidate B (Latent $k$-Means):** $\hat{\mathbf{y}}_{\text{km}}^{(t)} = \operatorname{KMeans}(\mathbf{Z}^{(t)}, K, n_{\text{init}}=5, \text{seed})$, $\operatorname{Sil}_{\text{km}}^{(t)} = \operatorname{Silhouette}(\mathbf{Z}^{(t)}, \hat{\mathbf{y}}_{\text{km}}^{(t)})$.
3. **Competition Selection Rule:**
   $$\hat{\mathbf{y}}_{\text{chosen}}^{(t)} = \begin{cases} \hat{\mathbf{y}}_q^{(t)}, & \text{if } \operatorname{Sil}_q^{(t)} \ge \operatorname{Sil}_{\text{km}}^{(t)} \text{ and } \left|\left\{\hat{y}_{q, i}^{(t)}\right\}\right| \ge 2 \\ \hat{\mathbf{y}}_{\text{km}}^{(t)}, & \text{otherwise} \end{cases}, \qquad \operatorname{Sil}^{(t)} = \max\left(\operatorname{Sil}_q^{(t)}, \operatorname{Sil}_{\text{km}}^{(t)}\right) \tag{S39}$$
4. **Global Best Checkpoint Restoration:** Across **both** Stage 1 and Stage 2 ($t \in \{1, \dots, 180\}$), whenever $\operatorname{Sil}^{(t)} > \operatorname{Sil}_{\text{global\_best}}$, the model snapshots $\Theta_{\text{best}}$, $\hat{\mathbf{y}}_{\text{best}} = \hat{\mathbf{y}}_{\text{chosen}}^{(t)}$, unified embedding $\mathbf{Z}_{\text{best}} = \mathbf{Z}^{(t)}$, attention gates $(\mathbf{g}_{\text{spatial}}, \mathbf{g}_{\text{modality}})$, and denoised RNA reconstructions decoded from $\mathbf{Z}_{\text{sim}}$ (`reconstruct_rna(sim_z_eval)`). Consequently, if fine-tuning fails to surpass Stage 1's peak separability, restoration safely reverts to Stage 1's optimum. At epoch $180$, parameters rewind to the overall optimum:

$$\Theta \leftarrow \Theta_{\text{best}}, \qquad \hat{\mathbf{X}}_{\text{RNA}} = \mathbf{h}(\mathbf{Z}_{\text{sim}}) \mathbf{W}_{\text{dec}2}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}2}^T \tag{S40}$$

### S5.3 Hyperparameters and Complete Training Algorithm

**Table S1.** Default architecture and two-stage optimization hyperparameters matching `DualGate_DEC.ipynb`.

| Symbol | Description | Stage 1 ($t \in [1, 80]$) | Stage 2 ($t \in [81, 180]$) |
| :--- | :--- | :---: | :---: |
| $k$ | Neighbors for $\mathcal{G}_{\text{sim}}, \mathcal{G}_{\text{dist}}$ | $15$ | $15$ |
| $d$ | Latent embedding dimensionality | $64$ | $64$ |
| $d_{\text{dec}}$ | Shared decoder hidden dictionary dimension | $512$ | $512$ |
| $\beta(t)$ | Multi-task reconstruction loss weight | $25.0$ | $5.0$ |
| $\gamma$ | Spatial coherence regularization weight | $10.0$ | $10.0$ |
| $\delta$ | Parameter regularization weight | $1.0$ | $1.0$ |
| $\lambda_{L_1}, \lambda_{L_2}$ | $L_1$ and $L_2$ weight penalties | $10^{-4},\ 10^{-3}$ | $10^{-4},\ 10^{-3}$ |
| $\lambda_{\text{dec}}$ | Spatial Potts KL clustering weight | $0.0$ (dormant) | $1.0$ |
| $\lambda_{\text{spatial}}$ | Potts MRF neighborhood coupling strength | — | $0.15$ |
| $\alpha$ | Student's $t$-kernel degrees of freedom | — | $1.0$ |
| $p_{\text{update\_interval}}$ | Asynchronous target update interval (epochs) | — | $5$ |
| $\eta(t)$ | Adam optimizer learning rate | $10^{-3}$ | $10^{-4}$ |

---

**Algorithm S1** Two-Stage Fusion, Decoupled Reconstruction, and Spatial Potts MRF Clustering (`DualGate-DEC`)

**Input:** Feature matrices $\mathbf{X}_{\text{RNA}}, \mathbf{X}_{\text{aux}}$; graphs $\mathcal{G}_{\text{sim}}, \mathcal{G}_{\text{dist}}, \mathcal{G}_{\text{common}}$; cluster count $K$; schedules $\beta(t), \eta(t)$; weights $\gamma, \delta, \lambda_{\text{dec}}, \lambda_{\text{spatial}}$; interval $p_{\text{update\_interval}} = 5$; random seed.  
**Output:** Optimal spatial domain labels $\hat{\mathbf{y}}_{\text{best}}$, unified embedding $\mathbf{Z}_{\text{best}}$, denoised RNA $\hat{\mathbf{X}}_{\text{RNA}}$, attention gates $(\mathbf{g}_{\text{spatial}}, \mathbf{g}_{\text{modality}})$.

1. Compute row-normalized spatial transition operator $\mathbf{A}_{\text{norm}}$ via Eq. (S17).
2. **Stage 1 ($t = 1$ to $80$, Adam lr $\eta_1 = 10^{-3}$):**
   - *(a)* Encode $\mathbf{Z}_{\text{sim}}, \mathbf{Z}_{\text{dist}}, \mathbf{Z}_{\text{pro}}$ and compute hierarchical gated embeddings $\mathbf{Z}_{\text{fused}}, \mathbf{Z}_{\text{fused\_pro}}$ via Eqs. (S1)–(S6).
   - *(b)* Decode four reconstructions via Eqs. (S7)–(S11) and minimize $\mathcal{L}_{\text{total}}^{\text{pre}}(\Theta_1)$ (Main Paper Eq. (12); Eqs. (S20)–(S26) with $\beta_{\text{pre}} = 25.0$).
   - *(c)* Evaluate $\operatorname{Sil}^{(t)}$ via Eq. (S32) ($k$-means, $n_{\text{init}}=10$) and update peak Stage 1 state $\Theta_1^*$ and global best checkpoint $\Theta_{\text{best}}$.
3. **Peak Silhouette Bridge Handover (after epoch $80$):**
   - Execute Eqs. (S33)–(S36): rewind to $\Theta_1^*$, extract $\mathbf{Z}_{\text{aligned}}$, run $k$-means ($n_{\text{init}}=20$) to initialize $\mathbf{M}$, and initialize Stage 2 Adam with $\eta_2 = 10^{-4}$.
4. **Stage 2 ($t = 81$ to $180$, Adam lr $\eta_2 = 10^{-4}$):**
   - *(a)* If $(t - 81) \equiv 0 \pmod{p_{\text{update\_interval}}}$, compute $\mathbf{Q}^{(t)}$ via Eq. (S13) and refresh stopped-gradient Potts target $\mathbf{P}_{\text{final}}^{(t)}$ via Eqs. (S14)–(S19) and (S37).
   - *(b)* Compute forward embeddings, reconstructions, $\mathbf{Q}^{(t)}$, and $\mathcal{L}_{\text{KL}}$ via Eq. (S27); update $\Theta_2 = \Theta_1 \cup \{\mathbf{M}\}$ by minimizing $\mathcal{L}_{\text{total}}^{\text{ft}}(\Theta_2)$ (Main Paper Eq. (13); Eqs. (S20)–(S27) and (S31) with $\beta_{\text{ft}} = 5.0$).
   - *(c)* Run dual-cluster competition ($\hat{\mathbf{y}}_q^{(t)}$ vs. $\hat{\mathbf{y}}_{\text{km}}^{(t)}$ with $n_{\text{init}}=5$) via Eqs. (S38)–(S39) and update global best $\Theta_{\text{best}}, \hat{\mathbf{y}}_{\text{best}}, \mathbf{Z}_{\text{best}}, \hat{\mathbf{X}}_{\text{RNA}}$.
5. **Restore** $\Theta \leftarrow \Theta_{\text{best}}$ via Eq. (S40) and **Return** $\hat{\mathbf{y}}_{\text{best}}, \mathbf{Z}_{\text{best}}, \hat{\mathbf{X}}_{\text{RNA}}, (\mathbf{g}_{\text{spatial}}, \mathbf{g}_{\text{modality}})$.

---

## S6. Cross-Reference Index Between Main Paper and Supplementary Data

**Table S2.** Mapping between Main Paper equations ([`Fusion_Decoder_DEC_Training_Objective.md`](file:///d:/FYDP/GATCON/d1_test/Fusion_Decoder_DEC_Training_Objective.md)) and Supplementary derivations.

| Main Paper Section & Formula | Main Eq. | Supplementary Equations & Proofs |
| :--- | :---: | :--- |
| **2.6** Tier 1 Spatial–Transcriptomic Gate & Projection | Eqs. (1)–(2) | Eqs. (S1)–(S3); Proposition S1 |
| **2.6** Tier 2 Cross-Modality Gate & Joint Embedding | Eqs. (3)–(4) | Eqs. (S4)–(S6); Proposition S1 |
| **2.7** Shared Semantic Hidden Projection | Eq. (5) | Eq. (S7) |
| **2.7** Four Decoupled Decoder Heads | Eqs. (6)–(7) | Eqs. (S8)–(S11); Proposition S2 |
| **2.8** Student's $t$-Kernel Soft Assignment | Eq. (8) | Eqs. (S12)–(S13) |
| **2.8** Frequency-Normalized DEC Target | Eq. (9) | Eq. (S14) |
| **2.8** Spatial Transition & Neighborhood Consensus | Eq. (10) | Eqs. (S15)–(S18) |
| **2.8** Potts-Modulated Target Distribution | Eq. (11) | Eq. (S19) |
| **2.9** Stage 1 Pre-training Objective $\mathcal{L}_{\text{total}}^{\text{pre}}$ | Eq. (12) | Eqs. (S20)–(S26), (S31)–(S36); Table S1; Algorithm S1 |
| **2.9** Stage 2 Fine-tuning Objective $\mathcal{L}_{\text{total}}^{\text{ft}}$ | Eq. (13) | Eqs. (S20)–(S31); Proposition S3; Theorem S1; Eqs. (S37)–(S40); Algorithm S1 |
| **2.9** Two-Stage Epoch Ranges ($t$) & Schedule | Eq. (14) | Eq. (S31); Theorem S1; Table S1; Algorithm S1 |
