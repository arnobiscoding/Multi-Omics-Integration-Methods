# Methodology: Fusion, Decoding, Clustering, and Unified Optimization

---

## 2.6 Hierarchical Spot-Adaptive Gated Attention Fusion

In spatial multi-omics sequencing, biological tissues exhibit non-uniform structural organization that imposes conflicting constraints on representation learning:

1. **Intra-Domain Homogeneity**: Within expansive, contiguous anatomical structures (such as cortical layers or lymphoid follicles), transcriptomic expression profiles frequently suffer from severe technical dropouts and Poisson sampling noise. In these regions, physical spatial proximity ($\mathbf{z}_{\text{dist}}$) provides an essential inductive bias to enforce local smoothness and cluster continuity.
2. **Sharp Anatomical Boundaries**: At the interface between distinct histological compartments (such as the germinal center boundary or tumor-stroma margins), immediate physical neighbors belong to fundamentally distinct cell types. Over-reliance on spatial coordinates blurs sharp phenotypic boundaries; here, molecular transcriptomic variation ($\mathbf{z}_{\text{sim}}$) must govern the latent manifold.
3. **Modality Discordance**: In specific tissue regions, messenger RNA profiles capture definitive lineage markers; in other domains, cell-surface protein abundance (ADT) or chromatin accessibility peaks (ATAC) reveal phenotypic subtypes that remain indistinguishable in the transcriptomic space alone.

Static feature concatenation ($[\mathbf{z}_1 \parallel \mathbf{z}_2]$) or invariant scalar averaging ($\alpha \mathbf{z}_1 + (1-\alpha) \mathbf{z}_2$) imposes an inflexible, slide-wide trade-off. To dynamically resolve these spatial and multimodal trade-offs at cellular resolution, our architecture introduces a **Hierarchical Two-Tier Spot-Adaptive Gated Attention Fusion** mechanism. This architecture computes a continuous, per-spot, and feature-wise gating vector $\mathbf{g}_i \in (0, 1)^d$ for every individual spot $i \in \{1, \dots, N\}$.


### 2.6.1 Tier 1: Intra-Modality Spatial-Transcriptomic Fusion
The first gating tier adaptively resolves the tension between functional transcriptomic correlation and spatial coordinate proximity within the RNA modality.

#### Inputs:
- $\mathbf{Z}_{\text{sim}} \in \mathbb{R}^{N \times d}$ ($d=64$): The latent matrix output from the graph convolutional network operating over the transcriptomic cosine similarity graph $\mathcal{G}_{\text{sim}}$, capturing expression-driven functional co-regulation.
- $\mathbf{Z}_{\text{dist}} \in \mathbb{R}^{N \times d}$ ($d=64$): The latent matrix output from the graph convolutional network operating over the spatial coordinate graph $\mathcal{G}_{\text{dist}}$, encoding 2D spatial adjacency across histological neighbors.

#### Gating Mechanism & Output Projection:
For each spot $i$, our model evaluates the concatenated representations to infer a spatial boundary gate $\mathbf{g}_{\text{spatial}, i} \in (0, 1)^d$:

$$\mathbf{g}_{\text{spatial}, i} = \sigma\left( \mathbf{W}_{g1} [\mathbf{z}_{\text{sim}, i} \parallel \mathbf{z}_{\text{dist}, i}] + \mathbf{b}_{g1} \right)$$

where $\mathbf{W}_{g1} \in \mathbb{R}^{d \times 2d}$, $\mathbf{b}_{g1} \in \mathbb{R}^d$, and $\sigma(u) = \frac{1}{1 + e^{-u}}$ denotes the element-wise sigmoid activation function.

Using the learned gating vector, an intermediate convex combination is formed and subsequently refined through a trainable linear output projection layer $\mathbf{W}_{o1} \in \mathbb{R}^{d \times d}$:

$$\tilde{\mathbf{z}}_{\text{fused}, i} = \mathbf{g}_{\text{spatial}, i} \odot \mathbf{z}_{\text{sim}, i} + (\mathbf{1} - \mathbf{g}_{\text{spatial}, i}) \odot \mathbf{z}_{\text{dist}, i}$$

$$\mathbf{z}_{\text{fused}, i} = \mathbf{W}_{o1} \tilde{\mathbf{z}}_{\text{fused}, i} + \mathbf{b}_{o1}$$

where $\odot$ represents the Hadamard (element-wise) product. The projection operator $\mathbf{W}_{o1}$ enables our network to rotate the fused manifold outside the strict convex hull of $\mathbf{z}_{\text{sim}}$ and $\mathbf{z}_{\text{dist}}$, enabling non-linear coordinate transformations.

---

### 2.6.2 Tier 2: Cross-Modality Multi-Omics Integration
The second gating tier dynamically integrates the spatial-transcriptomic manifold derived from Tier 1 with auxiliary epigenomic or proteomic measurements.

#### Inputs:
- $\mathbf{Z}_{\text{fused}} \in \mathbb{R}^{N \times d}$ ($d=64$): The spatial-transcriptomic latent manifold output from Tier 1.
- $\mathbf{Z}_{\text{pro}} \in \mathbb{R}^{N \times d}$ ($d=64$): The latent representation derived from the graph convolutional network processing auxiliary omics ($\mathbf{X}_{\text{ADT}}$ for CITE-seq surface antibodies or $\mathbf{X}_{\text{ATAC}}$ for chromatin accessibility peaks) over the consensus intersection topology $\mathcal{G}_{\text{common}} = \mathcal{G}_{\text{sim}} \cap \mathcal{G}_{\text{dist}}$.

#### Gating Mechanism & Final Joint Embedding:
Spot-wise modality dominance is evaluated across channels via a cross-modality gating parameter $\mathbf{W}_{g2} \in \mathbb{R}^{d \times 2d}$ and $\mathbf{b}_{g2} \in \mathbb{R}^d$:

$$\mathbf{g}_{\text{modality}, i} = \sigma\left( \mathbf{W}_{g2} [\mathbf{z}_{\text{fused}, i} \parallel \mathbf{z}_{\text{pro}, i}] + \mathbf{b}_{g2} \right)$$

$$\tilde{\mathbf{z}}_{\text{fused\_pro}, i} = \mathbf{g}_{\text{modality}, i} \odot \mathbf{z}_{\text{fused}, i} + (\mathbf{1} - \mathbf{g}_{\text{modality}, i}) \odot \mathbf{z}_{\text{pro}, i}$$

$$\mathbf{z}_{\text{fused\_pro}, i} = \mathbf{W}_{o2} \tilde{\mathbf{z}}_{\text{fused\_pro}, i} + \mathbf{b}_{o2}$$

The resulting embedding vector $\mathbf{z}_{\text{fused\_pro}, i} \in \mathbb{R}^d$ serves as the **unified multi-omics latent code** for spot $i$. It simultaneously encapsulates gene expression dynamics, 2D histological coordinates, and auxiliary phenotypic signals.

---

## 2.7 Decoupled Multi-Task Reconstruction Decoders

A critical pathology in spatial multi-omics modeling arises from the extreme **Dimensionality Imbalance and Representation Collapse**:

$$\operatorname{dim}(\mathbf{X}_{\text{RNA}}) = 3000 \quad \gg \quad \operatorname{dim}(\mathbf{X}_{\text{aux}}) \in [10, 180]$$

When an autoencoder reconstructs only the concatenated feature matrix $[\mathbf{X}_{\text{RNA}} \parallel \mathbf{X}_{\text{aux}}]$ from a joint latent vector, the high-dimensional transcriptomic features contribute over $96\%$ to $99\%$ of the cumulative reconstruction error. Consequently, backpropagated gradients are dominated by RNA variance, causing the auxiliary branch to collapse into a trivial uninformative representation ($\mathbf{z}_{\text{pro}} \to \mathbf{0}$). Furthermore, intermediate representations $\mathbf{z}_{\text{sim}}$ and $\mathbf{z}_{\text{dist}}$ lack individual supervision, allowing one GCN branch to degenerate while the other compensates.

To resolve these failure modes, our architecture introduces **Four Decoupled Reconstruction Heads**, constraining every latent bottleneck to independently reconstruct its native biological feature space through a shared semantic subspace.


### 2.7.1 Shared Semantic Hidden Projection
All four reconstruction pathways project through a shared hidden dictionary layer $\mathbf{W}_{\text{dec}1} \in \mathbb{R}^{512 \times 64}$:

$$\mathbf{h}(\mathbf{z}) = \operatorname{ReLU}\left( \mathbf{z} \mathbf{W}_{\text{dec}1}^T + \mathbf{1} \mathbf{b}_{\text{dec}1}^T \right) \in \mathbb{R}^{N \times 512}$$

By passing the joint representation $\mathbf{z}_{\text{fused\_pro}}$, both single-modality RNA representations ($\mathbf{z}_{\text{sim}}, \mathbf{z}_{\text{dist}}$), and the auxiliary representation $\mathbf{z}_{\text{pro}}$ through the identical projection matrix $\mathbf{W}_{\text{dec}1}$, our network forces the intermediate 512-dimensional subspace to act as an aligned, cross-modal semantic dictionary.

---

### 2.7.2 Mathematical Formulation of Reconstruction Heads

#### 1. Joint Multimodal Decoder Head
Reconstructs the concatenated multi-omics matrix $[\mathbf{X}_{\text{RNA}} \parallel \mathbf{X}_{\text{aux}}] \in \mathbb{R}^{N \times (G + D_{\text{aux}})}$ from the Tier 2 unified latent code $\mathbf{z}_{\text{fused\_pro}}$ via linear head $\mathbf{W}_{\text{dec}5} \in \mathbb{R}^{(G + D_{\text{aux}}) \times 512}$:

$$\hat{\mathbf{X}}_{\text{joint}} = \mathbf{h}(\mathbf{Z}_{\text{fused\_pro}}) \mathbf{W}_{\text{dec}5}^T + \mathbf{1} \mathbf{b}_{\text{dec}5}^T$$

$$\mathcal{L}_{\text{rec}} = \frac{1}{N (G + D_{\text{aux}})} \sum_{i=1}^N \sum_{j=1}^{G + D_{\text{aux}}} \left( [\mathbf{X}_{\text{RNA}} \parallel \mathbf{X}_{\text{aux}}]_{ij} - \hat{\mathbf{X}}_{\text{joint}, ij} \right)^2$$

#### 2. Transcriptomic Similarity & Distance Decoder Heads
Both RNA branch representations ($\mathbf{z}_{\text{sim}}$ and $\mathbf{z}_{\text{dist}}$) are decoded back to the 3,000 highly variable gene space through the **identical projection matrix** $\mathbf{W}_{\text{dec}2} \in \mathbb{R}^{G \times 512}$:

$$\hat{\mathbf{X}}_{\text{sim}} = \mathbf{h}(\mathbf{Z}_{\text{sim}}) \mathbf{W}_{\text{dec}2}^T + \mathbf{1} \mathbf{b}_{\text{dec}2}^T$$

$$\hat{\mathbf{X}}_{\text{dist}} = \mathbf{h}(\mathbf{Z}_{\text{dist}}) \mathbf{W}_{\text{dec}2}^T + \mathbf{1} \mathbf{b}_{\text{dec}2}^T$$

$$\mathcal{L}_{\text{sim}} = \frac{1}{N G} \sum_{i=1}^N \sum_{j=1}^{G} \left( \mathbf{X}_{\text{RNA}, ij} - \hat{\mathbf{X}}_{\text{sim}, ij} \right)^2$$

$$\mathcal{L}_{\text{dist}} = \frac{1}{N G} \sum_{i=1}^N \sum_{j=1}^G \left( \mathbf{X}_{\text{RNA}, ij} - \hat{\mathbf{X}}_{\text{dist}, ij} \right)^2$$

Enforcing a shared projection matrix $\mathbf{W}_{\text{dec}2}$ across both $\mathbf{z}_{\text{sim}}$ and $\mathbf{z}_{\text{dist}}$ guarantees that the functional cosine GCN and the physical Euclidean GCN map their distinct inductive biases onto the same canonical transcriptomic manifold.

#### 3. Auxiliary Modality Decoder Head
Reconstructs the auxiliary protein or chromatin accessibility profile from $\mathbf{z}_{\text{pro}}$ via linear head $\mathbf{W}_{\text{dec}4} \in \mathbb{R}^{D_{\text{aux}} \times 512}$:

$$\hat{\mathbf{X}}_{\text{aux}} = \mathbf{h}(\mathbf{Z}_{\text{pro}}) \mathbf{W}_{\text{dec}4}^T + \mathbf{1} \mathbf{b}_{\text{dec}4}^T$$

$$\mathcal{L}_{\text{aux}} = \frac{1}{N D_{\text{aux}}} \sum_{i=1}^N \sum_{k=1}^{D_{\text{aux}}} \left( \mathbf{X}_{\text{aux}, ik} - \hat{\mathbf{X}}_{\text{aux}, ik} \right)^2$$

This decoupled auxiliary loss provides direct, non-vanishing gradient feedback to $\mathbf{z}_{\text{pro}}$, preventing auxiliary feature collapse.

#### Multi-Task Reconstruction Master Objective:
The decoupled reconstruction losses are aggregated with equal unity weighting into a composite reconstruction objective:

$$\mathcal{L}_{\text{multi-rec}} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{sim}} + \mathcal{L}_{\text{dist}} + \mathcal{L}_{\text{aux}}$$

---

## 2.8 Spatial Potts MRF Deep Embedding Clustering (DEC)

Standard geometric clustering algorithms (such as Euclidean $k$-means) identify cluster prototypes $\boldsymbol{\mu}_k$ by minimizing within-cluster sum-of-squares:

$$\min_{\{\boldsymbol{\mu}_k\}_{k=1}^K} \sum_{i=1}^N \min_{k \in \{1, \dots, K\}} \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2$$

In spatial transcriptomics, this classical formulation presents two critical drawbacks:
1. **Non-Differentiability**: The hard partition operator ($\arg\min_k$) has zero gradient almost everywhere, preventing end-to-end backpropagation into the underlying graph neural network.
2. **Spatial Blindness ("Salt-and-Pepper" Artifacts)**: Point-wise clustering in static latent space ignores histological organization, producing scattered, isolated misclassified spots that violate tissue architectural continuity.

Our model overcomes these limitations by integrating **Deep Embedding Clustering (DEC)** with a **Spatial Potts Markov Random Field (MRF)** consensus mechanism.

---

### 2.8.1 Soft Assignment Matrix $\mathbf{Q}$ via Student's t-Kernel
Let $\mathbf{Z} = \mathbf{z}_{\text{fused\_pro}} \in \mathbb{R}^{N \times d}$ denote the joint latent embeddings, and let $\mathbf{M} = [\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K]^T \in \mathbb{R}^{K \times d}$ be $K$ learnable cluster prototype vectors.

The soft assignment probability $q_{ik}$ that spot $i$ belongs to cluster $k$ is quantified via the Student's t-distribution kernel with degree of freedom $\alpha = 1.0$ (Cauchy distribution):

$$q_{ik} = \frac{\left( 1 + \frac{\|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2}{\alpha} \right)^{-\frac{\alpha + 1}{2}}}{\sum_{k'=1}^K \left( 1 + \frac{\|\mathbf{z}_i - \boldsymbol{\mu}_{k'}\|_2^2}{\alpha} \right)^{-\frac{\alpha + 1}{2}}}$$

Setting $\alpha = 1.0$:

$$q_{ik} = \frac{\left( 1 + \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2 \right)^{-1}}{\sum_{k'=1}^K \left( 1 + \|\mathbf{z}_i - \boldsymbol{\mu}_{k'}\|_2^2 \right)^{-1}}$$

The Cauchy distribution features heavy, power-law tails ($\mathcal{O}(d^{-2})$), preventing gradient vanishing for distant spots and providing stable attraction forces toward prototype centroids.

---

### 2.8.2 Target Distribution Sharpening ($\mathbf{P}_{\text{dec}}$)
To iteratively improve cluster purity, DEC constructs an auxiliary target distribution $\mathbf{P}_{\text{dec}} \in \mathbb{R}^{N \times K}$ that sharpens high-confidence predictions:

$$p_{\text{dec}, ik} = \frac{q_{ik}^2 / f_k}{\sum_{k'=1}^K \left( q_{ik'}^2 / f_{k'} \right)}, \quad \text{where } f_k = \sum_{i=1}^N q_{ik}$$

- **Probability Squaring ($q_{ik}^2$)**: Amplifies confident predictions while suppressing ambiguous boundary assignments.
- **Cluster Frequency Normalization ($/ f_k$)**: Normalizes by cluster mass to prevent large, dominant clusters from expanding and absorbing rare, biologically distinct micro-domains.

---

### 2.8.3 Spatial Potts Markov Random Field Consensus Modulation
Standard DEC sharpens distributions purely in latent space. Our architecture introduces spatial context by coupling the target distribution to a **Potts Markov Random Field**. In statistical physics, the Potts Hamiltonian over discrete spin states $s_i \in \{1, \dots, K\}$ on graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is:

$$H(\mathbf{s}) = -J \sum_{(i,j) \in \mathcal{E}} \delta(s_i, s_j) - \sum_{i \in \mathcal{V}} h_i(s_i)$$

where $J > 0$ represents the ferromagnetic coupling constant and $h_i$ is an external field. Under the Gibbs distribution $P(\mathbf{s}) \propto \exp(-H(\mathbf{s}) / k_B T)$, neighboring nodes are exponentially favored to adopt identical cluster labels.



#### Mathematical Formulation:
1. **Row-Normalized Spatial Transition Operator ($\mathbf{A}_{\text{norm}}$)**:
   Adding self-loops to the spatial distance adjacency $\mathbf{A}_{\text{dist}}$:
   $$\tilde{\mathbf{A}}_{\text{dist}} = \mathbf{A}_{\text{dist}} + \mathbf{I}_N \in \mathbb{R}^{N \times N}$$
   $$\mathbf{D}_{ii} = \sum_{j=1}^N \tilde{A}_{\text{dist}, ij} \implies \mathbf{A}_{\text{norm}} = \mathbf{D}^{-1} \tilde{\mathbf{A}}_{\text{dist}}$$
   where $\sum_{j=1}^N A_{\text{norm}, ij} = 1.0$ for every spot $i$.

2. **Neighborhood Soft Consensus Vector**:
   The expected cluster distribution within spot $i$'s physical neighborhood is computed by matrix multiplication:
   $$\mathbf{C} = \mathbf{A}_{\text{norm}} \cdot \mathbf{Q} \in \mathbb{R}^{N \times K}, \quad C_{ik} = \sum_{j=1}^N A_{\text{norm}, ij} q_{jk}$$

3. **Potts Exponential Modulation & Renormalization**:
   The DEC target is modulated by the exponential neighborhood consensus energy:
   $$p_{\text{spatial}, ik} = p_{\text{dec}, ik} \cdot \exp\left( \lambda_{\text{spatial}} \cdot C_{ik} \right) = p_{\text{dec}, ik} \cdot \exp\left( \lambda_{\text{spatial}} \sum_{j=1}^N A_{\text{norm}, ij} q_{jk} \right)$$
   Row-normalizing yields the final spatial Potts target distribution $\mathbf{P}_{\text{final}} \in \mathbb{R}^{N \times K}$:
   $$p_{\text{final}, ik} = \frac{p_{\text{spatial}, ik}}{\sum_{k'=1}^K p_{\text{spatial}, ik'} + 10^{-12}} = \frac{p_{\text{dec}, ik} \exp\left( \lambda_{\text{spatial}} \sum_{j=1}^N A_{\text{norm}, ij} q_{jk} \right)}{\sum_{k'=1}^K p_{\text{dec}, ik'} \exp\left( \lambda_{\text{spatial}} \sum_{j=1}^N A_{\text{norm}, ij} q_{jk'} \right) + 10^{-12}}$$
   where $\lambda_{\text{spatial}} = 0.15$ controls the strength of local spatial smoothing.

---

### 2.8.4 Kullback-Leibler Divergence Objective & Gradients
The clustering loss is formulated as the asymmetric Kullback-Leibler (KL) divergence from the model soft assignments $\mathbf{Q}$ to the Potts-modulated target distribution $\mathbf{P}_{\text{final}}$:

$$\mathcal{L}_{\text{KL}}(\mathbf{Q} \parallel \mathbf{P}_{\text{final}}) = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K p_{\text{final}, ik} \ln\left( \frac{p_{\text{final}, ik}}{q_{ik} + 10^{-10}} \right)$$

The analytical gradient of $\mathcal{L}_{\text{KL}}$ with respect to the latent embedding $\mathbf{z}_i$ and prototype $\boldsymbol{\mu}_k$ drives the neural feature representation:

$$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \mathbf{z}_i} = \frac{2}{N} \sum_{k=1}^K \left( 1 + \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2 \right)^{-1} \left( q_{ik} - p_{\text{final}, ik} \right) \left( \mathbf{z}_i - \boldsymbol{\mu}_k \right)$$

$$\frac{\partial \mathcal{L}_{\text{KL}}}{\partial \boldsymbol{\mu}_k} = -\frac{2}{N} \sum_{i=1}^N \left( 1 + \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2 \right)^{-1} \left( p_{\text{final}, ik} - q_{ik} \right) \left( \mathbf{z}_i - \boldsymbol{\mu}_k \right)$$

---

## 2.9 Stage 1 Representation Pre-training and Peak Silhouette Bridge Handover

In standard end-to-end deep clustering literature, prototype centroids $\boldsymbol{\mu}_k$ are typically initialized directly from the final epoch of pre-training (e.g., epoch 80). In multi-omics spatial biology, this direct handoff exhibits severe **Checkpoint Decay**:
- In prolonged training, continuous minimization of reconstruction MSE causes the GCN backbone to overfit on technical sequencing noise and dropouts, resulting in **geometric over-smoothing**.
- Empirically, the peak cluster separability (highest latent Silhouette score) occurs significantly earlier in training (often between epochs 35 and 65) than the final pre-training epoch ($t = 80$).
- Initializing centroids from a degraded final checkpoint traps the Student's t-kernel in sub-optimal, blurred local minima.

To ensure non-degenerate clustering, our proposed framework introduces a **Two-Stage Training Strategy linked by a Peak Silhouette Bridge Handover Protocol** across a structured 180-epoch budget.



### 2.9.1 Stage 1 Optimization Objective (Epochs 1–80)
During the initial 80 epochs, our model optimizes multi-task reconstruction and spatial contrastive topology without cluster prototype interference:

$$\min_{\Theta_1} \mathcal{L}_{\text{Stage1}}(\Theta_1) = \beta \cdot \mathcal{L}_{\text{multi-rec}}(\Theta_1) + \gamma \cdot \mathcal{L}_{\text{spatial}}(\Theta_1) + \delta \cdot \mathcal{L}_{\text{reg}}(\Theta_1)$$

where:
$$\beta = 25.0, \quad \gamma = 10.0, \quad \delta = 1.0, \quad \eta_1 = 10^{-3} \quad (\text{Adam optimizer})$$

#### Spatial Contrastive Graph Regularization:
Pairwise cosine similarities on Tier 1 fused RNA $\mathbf{Z}_{\text{fused}}$ are converted to connection probabilities via a sigmoid function:

$$\mathbf{S}_{ij} = \frac{\mathbf{z}_{\text{fused}, i}^T \mathbf{z}_{\text{fused}, j}}{\|\mathbf{z}_{\text{fused}, i}\|_2 \|\mathbf{z}_{\text{fused}, j}\|_2 + 10^{-12}} \cdot (1 - \delta_{ij}), \quad \tilde{\mathbf{S}}_{ij} = \sigma(\mathbf{S}_{ij}) \in (0, 1)$$

$$\mathcal{L}_{\text{spatial}} = -\frac{1}{2 N^2} \sum_{i=1}^N \sum_{j=1}^N \left[ \mathbf{A}_{\text{dist}, ij} \ln\left(\tilde{\mathbf{S}}_{ij} + 10^{-10}\right) + \left(1 - \mathbf{A}_{\text{dist}, ij}\right) \ln\left(1 - \tilde{\mathbf{S}}_{ij} + 10^{-10}\right) \right]$$

#### Weight Regularization:
$$\mathcal{L}_{\text{reg}} = \lambda_{L_1} \sum_{\theta \in \Theta_1} |\theta| + \lambda_{L_2} \sum_{\theta \in \Theta_1} \theta^2, \quad \lambda_{L_1} = 10^{-4}, \quad \lambda_{L_2} = 10^{-3}$$

---

### 2.9.2 Epoch-by-Epoch Diagnostic Tracking
At every epoch $t \in \{1, \dots, 80\}$:
1. Extract the current latent representation $\mathbf{Z}^{(t)} = \mathbf{z}_{\text{fused\_pro}}^{(t)} \in \mathbb{R}^{N \times 64}$.
2. Run fast $k$-means to obtain labels $\mathbf{y}^{(t)} = \operatorname{KMeans}(\mathbf{Z}^{(t)}, K, n_{\text{init}}=10)$.
3. Compute the Silhouette score $\text{Sil}^{(t)}$ across all $N$ spots:
   $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad \text{Sil}^{(t)} = \frac{1}{N} \sum_{i=1}^N s(i)$$
4. Checkpoint the peak model state:
   $$\Theta_1^* = \arg\max_{\Theta_1^{(t)}, t \in \{1, \dots, 80\}} \text{Sil}^{(t)}$$

---

### 2.9.3 The 5-Step Bridge Handover Protocol
At the completion of epoch 80, the bridge transition executes the following sequence:

1. **Parameter Rewind**: The model parameters are rewound to the state of peak Silhouette score:
   $$\Theta_1 \leftarrow \Theta_1^*$$
2. **Aligned Manifold Extraction**: A forward pass on the restored model extracts the peak joint latent coordinates:
   $$\mathbf{Z}_{\text{aligned}} = f_{\text{encoder}}\left(\mathbf{X}_{\text{RNA}}, \mathbf{X}_{\text{aux}}, \mathcal{G}_{\text{sim}}, \mathcal{G}_{\text{dist}}, \mathcal{G}_{\text{common}}; \Theta_1^*\right) \in \mathbb{R}^{N \times 64}$$
3. **High-Fidelity Prototype Discovery**: Runs $k$-means with **$n_{\text{init}} = 20$** on $\mathbf{Z}_{\text{aligned}}$ to locate global minimum centroid configurations:
   $$\{\boldsymbol{\mu}_1^*, \dots, \boldsymbol{\mu}_K^*\} = \arg\min_{\{\boldsymbol{\mu}_k\}_{k=1}^K} \sum_{i=1}^N \min_{k \in \{1, \dots, K\}} \|\mathbf{z}_{\text{aligned}, i} - \boldsymbol{\mu}_k\|_2^2$$
4. **DEC Prototype Parameter Ingestion**: Directly copies discovered centroids into the trainable parameter matrix $\mathbf{M}$:
   $$\mathbf{M} \leftarrow \begin{bmatrix} \boldsymbol{\mu}_1^{*T} \\ \vdots \\ \boldsymbol{\mu}_K^{*T} \end{bmatrix} \in \mathbb{R}^{K \times 64}$$
5. **Optimizer Deceleration & Target Reset**: Instantiates a fresh Stage 2 Adam optimizer with a $10\times$ reduced learning rate:
   $$\eta_2 = 0.1 \times \eta_1 = 10^{-4}, \quad \mathbf{P}_{\text{target}} \leftarrow \text{None}$$

---

## 2.10 Stage 2 Spatial Potts Markov Random Field Consensus DEC Fine-Tuning

In Stage 2 (epochs 81 to 180, comprising 100 epochs of fine-tuning), the network co-optimizes the continuous representations and the discrete spatial cluster boundaries under the joint parameter set $\Theta_2 = \Theta_1 \cup \{\mathbf{M}\}$.

### 2.10.1 Mathematical Formulation of Stage 2 Optimization
$$\min_{\Theta_2} \mathcal{L}_{\text{Stage2}}(\Theta_2) = \beta_{\text{curr}} \cdot \mathcal{L}_{\text{multi-rec}}(\Theta_1) + \gamma \cdot \mathcal{L}_{\text{spatial}}(\Theta_1) + \delta \cdot \mathcal{L}_{\text{reg}}(\Theta_2) + \lambda_{\text{dec}} \cdot \mathcal{L}_{\text{KL}}(\mathbf{Q} \parallel \mathbf{P}_{\text{final}})$$

where:
$$\beta_{\text{curr}} = 5.0, \quad \lambda_{\text{dec}} = 1.0, \quad \gamma = 10.0, \quad \delta = 1.0, \quad \eta_2 = 10^{-4}$$

---

### 2.10.2 Mathematical Proof of Reconstruction Attenuation ($\beta: 25.0 \to 5.0$)

During Stage 1, a large reconstruction weight ($\beta = 25.0$) is required to enforce high-fidelity feature recovery across 3,000 genes. However, maintaining $\beta = 25.0$ during Stage 2 leads to catastrophic optimization failure:

#### Gradient Magnitude Analysis:
Let $\mathbf{g}_{\text{rec}} = \nabla_{\mathbf{z}} \mathcal{L}_{\text{multi-rec}}$ and $\mathbf{g}_{\text{KL}} = \nabla_{\mathbf{z}} \mathcal{L}_{\text{KL}}$.

Empirically and analytically:
$$\|\mathbf{g}_{\text{rec}}\|_2 \sim \mathcal{O}(10^{-1}) \quad \text{due to } G = 3000 \text{ summed gene residuals}$$
$$\|\mathbf{g}_{\text{KL}}\|_2 \sim \mathcal{O}(10^{-2}) \quad \text{bounded by soft assignment probabilities } q_{ik} \in (0,1)$$

#### If $\beta = 25.0$ Were Maintained Unmodified:
$$\frac{\|\beta \mathbf{g}_{\text{rec}}\|_2}{\|\lambda_{\text{dec}} \mathbf{g}_{\text{KL}}\|_2} = \frac{25.0 \times 10^{-1}}{1.0 \times 10^{-2}} \approx 250:1$$

The reconstruction gradients overwhelm the clustering objective by more than two orders of magnitude. The Student's t-kernel centroids are rendered passive spectators, unable to deform the latent space to resolve spatial boundaries or conform to the Potts MRF consensus.

#### With Attenuation ($\beta_{\text{curr}} = 5.0$):
By scaling $\beta$ by an $80\%$ reduction ($25.0 \to 5.0$):
$$\frac{\|\beta_{\text{curr}} \mathbf{g}_{\text{rec}}\|_2}{\|\lambda_{\text{dec}} \mathbf{g}_{\text{KL}}\|_2} = \frac{5.0 \times 10^{-1}}{1.0 \times 10^{-2}} \approx 50:1$$

This attenuation restores dynamic gradient equilibrium:
1. Reconstruction gradients $\beta_{\text{curr}} \mathbf{g}_{\text{rec}}$ act as a stabilizing regularizer, preserving the global biological manifold topology.
2. The spatial Potts clustering gradient $\lambda_{\text{dec}} \mathbf{g}_{\text{KL}}$ exerts sufficient torque to compress intra-cluster variances and actively conform spot assignments to physical histological domains.

---

### 2.10.3 Asynchronous Target Updating Algorithm
To stabilize the Expectation-Maximization dynamic and prevent circular gradient chasing, the spatial target $\mathbf{P}_{\text{final}}$ is updated asynchronously every $p_{\text{update\_interval}}=5$ epochs:

$$\mathbf{P}_{\text{final}}^{(t)} = \begin{cases} \operatorname{StopGradient}\left( \mathcal{F}_{\text{Potts}}(\mathbf{Q}^{(t)}, \mathbf{A}_{\text{norm}}) \right) & \text{if } (t - 80) \equiv 0 \pmod 5 \text{ or } t = 81 \\ \mathbf{P}_{\text{final}}^{(t-1)} & \text{otherwise} \end{cases}$$

Across the 100 Stage 2 fine-tuning epochs ($t \in \{81, \dots, 180\}$), the target distribution is recomputed exactly 20 times. This asynchronous schedule prevents high-frequency oscillations while ensuring the Potts prior guides the latent trajectories.

---

### 2.10.4 Dual-Cluster Selection & Competition Strategy
At every epoch $t \in \{81, \dots, 180\}$, two candidate partitions are evaluated:

1. **Candidate A (DEC Soft Predictions)**:
   $$\hat{y}_{q, i}^{(t)} = \arg\max_{k \in \{1, \dots, K\}} q_{ik}^{(t)}, \quad \hat{\mathbf{y}}_q^{(t)} = [\hat{y}_{q, 1}^{(t)}, \dots, \hat{y}_{q, N}^{(t)}]^T$$
   $$\text{Sil}_q^{(t)} = \begin{cases} \operatorname{Silhouette}\left(\mathbf{Z}^{(t)}, \hat{\mathbf{y}}_q^{(t)}\right) & \text{if } |\{ \hat{y}_{q, i}^{(t)} \}| \ge 2 \\ -1.0 & \text{otherwise (degenerate protection)} \end{cases}$$

2. **Candidate B (Latent Space KMeans)**:
   $$\hat{\mathbf{y}}_{\text{km}}^{(t)} = \operatorname{KMeans}\left( \mathbf{Z}^{(t)}, K, n_{\text{init}}=5 \right), \quad \text{Sil}_{\text{km}}^{(t)} = \operatorname{Silhouette}\left(\mathbf{Z}^{(t)}, \hat{\mathbf{y}}_{\text{km}}^{(t)}\right)$$

3. **Competition Selection Rule**:
   $$\hat{\mathbf{y}}_{\text{chosen}}^{(t)} = \begin{cases} \hat{\mathbf{y}}_q^{(t)} & \text{if } \text{Sil}_q^{(t)} \ge \text{Sil}_{\text{km}}^{(t)} \text{ and } |\{ \hat{y}_{q, i}^{(t)} \}| \ge 2 \\ \hat{\mathbf{y}}_{\text{km}}^{(t)} & \text{otherwise} \end{cases}$$
   $$\text{Sil}^{(t)} = \max\left(\text{Sil}_q^{(t)}, \text{Sil}_{\text{km}}^{(t)}\right)$$

4. **Global Best State Checkpoint**:
   $$\text{If } \text{Sil}^{(t)} > \text{Sil}_{\text{global\_best}} \implies \Theta_{\text{best}} \leftarrow \operatorname{deepcopy}(\Theta_2^{(t)}), \quad \hat{\mathbf{y}}_{\text{best}} \leftarrow \hat{\mathbf{y}}_{\text{chosen}}^{(t)}$$

---

### 2.10.5 Global Checkpoint Restoration
At the completion of all 180 epochs, the network rewinds parameters to the global optimum:

$$\Theta_2 \leftarrow \Theta_{\text{best}}$$

This final restoration guarantees that all downstream domain labels, UMAP projections, and imputed RNA expression matrices $\hat{\mathbf{X}}_{\text{RNA}} = \mathbf{W}_{\text{dec}2} \cdot \operatorname{ReLU}(\mathbf{W}_{\text{dec}1} \mathbf{z}_{\text{sim}} + \mathbf{b}_{\text{dec}1}) + \mathbf{b}_{\text{dec}2}$ correspond strictly to the highest Silhouette partition achieved across the 180-epoch optimization continuum.

---

## 2.11 Unified Optimization Objective & Training Dynamics of Our Proposed Architecture

The unified mathematical objective function across the complete lifecycle of our proposed model is summarized below:

$$\mathcal{L}_{\text{total}}(\Theta) = \beta(t) \cdot \mathcal{L}_{\text{multi-rec}}(\Theta) + \gamma \cdot \mathcal{L}_{\text{spatial}}(\Theta) + \delta \cdot \mathcal{L}_{\text{reg}}(\Theta) + \mathbb{I}(t > 80) \cdot \lambda_{\text{dec}} \cdot \mathcal{L}_{\text{KL}}(\mathbf{Q} \parallel \mathbf{P}_{\text{final}})$$

### Dynamic Parameter Schedule:

$$\beta(t) = \begin{cases} 25.0 & \text{if } 1 \le t \le 80 \quad (\text{Stage 1: Representation Pre-training}) \\ 5.0 & \text{if } 81 \le t \le 180 \quad (\text{Stage 2: Potts Consensus DEC Fine-Tuning}) \end{cases}$$

$$\eta(t) = \begin{cases} 10^{-3} & \text{if } 1 \le t \le 80 \quad (\text{Standard learning rate}) \\ 10^{-4} & \text{if } 81 \le t \le 180 \quad (\mathbf{10\times \text{ decelerated fine-tuning rate}}) \end{cases}$$

$$\mathbb{I}(t > 80) = \begin{cases} 0 & \text{if } 1 \le t \le 80 \quad (\text{DEC module dormant}) \\ 1 & \text{if } 81 \le t \le 180 \quad (\text{DEC module active}) \end{cases}$$

---
