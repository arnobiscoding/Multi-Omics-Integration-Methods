# Methodology: Hierarchical Fusion, Decoupled Decoding, Clustering, and Unified Optimization (DualGate-DEC)

> **Supplementary Material:** Extended derivations, formal proofs (`Propositions S1–S3`, `Theorem S1`), complete loss expansions and training details (`Eqs. (S1)–(S40)`), hyperparameter configurations (`Table S1`), and the full training algorithm (`Algorithm S1`) are provided in [`Supplementary_Data.md`](file:///d:/FYDP/GATCON/d1_test/Supplementary_Data.md).

## 2.6 Hierarchical Spot-Adaptive Gated Attention Fusion

In spatial multi-omics sequencing, biological tissues impose three competing structural requirements across regions: (i) **intra-domain homogeneity**, where expansive anatomical compartments such as cortical layers or lymphoid follicles suffer from transcriptomic dropout and require physical spatial proximity ($\mathbf{Z}_{\text{dist}}$) for smoothing; (ii) **sharp anatomical boundaries**, where adjacent spots along histological interfaces belong to distinct cell states and must be governed by transcriptional variation ($\mathbf{Z}_{\text{sim}}$) to prevent boundary blurring; and (iii) **modality discordance**, where auxiliary proteomic (ADT) or chromatin-accessibility (ATAC) signals resolve cellular subtypes that remain indistinguishable in mRNA space alone. Static concatenation or global scalar averaging enforces an inflexible, slide-wide trade-off across these regimes. To dynamically balance molecular and spatial evidence at single-spot resolution, DualGate-DEC introduces a **two-tier spot-adaptive gated attention fusion** mechanism operating inside-out from intra-modal RNA consolidation to cross-modal integration.

**Intra-modality spatial–transcriptomic fusion (Tier 1).** Given the dual two-layer RNA graph convolutional network (GCN) embeddings ($k = 15$ nearest neighbors)—the transcriptomic similarity representation $\mathbf{Z}_{\text{sim}} \in \mathbb{R}^{N \times d}$ learned on $\mathcal{G}_{\text{sim}}$ and the spatial-proximity representation $\mathbf{Z}_{\text{dist}} \in \mathbb{R}^{N \times d}$ learned on $\mathcal{G}_{\text{dist}}$ ($d = 64$)—the first tier infers a continuous, feature-wise spatial boundary gate $\mathbf{g}_{\text{spatial}, i} \in (0, 1)^d$ for each spot $i \in \{1, \dots, N\}$:

$$\mathbf{g}_{\text{spatial}, i} = \sigma\left( \mathbf{W}_{g1} [\mathbf{z}_{\text{sim}, i} \parallel \mathbf{z}_{\text{dist}, i}] + \mathbf{b}_{g1} \right) \tag{1}$$

where $\mathbf{W}_{g1} \in \mathbb{R}^{d \times 2d}$ and $\mathbf{b}_{g1} \in \mathbb{R}^d$ are learnable parameters, $[\cdot \parallel \cdot]$ denotes concatenation, and $\sigma(\cdot)$ is the element-wise sigmoid function (Supplementary Eq. (S1)). The gated convex blend is then projected through an affine output mapping $(\mathbf{W}_{o1}, \mathbf{b}_{o1})$ to form the consolidated spatial–transcriptomic representation $\mathbf{z}_{\text{fused}, i} \in \mathbb{R}^d$:

$$\mathbf{z}_{\text{fused}, i} = \mathbf{W}_{o1} \Big( \mathbf{g}_{\text{spatial}, i} \odot \mathbf{z}_{\text{sim}, i} + (\mathbf{1} - \mathbf{g}_{\text{spatial}, i}) \odot \mathbf{z}_{\text{dist}, i} \Big) + \mathbf{b}_{o1} \tag{2}$$

where $\odot$ denotes the Hadamard product (Supplementary Eqs. (S2)–(S3)). As shown in Supplementary Proposition S1, the linear projection $\mathbf{W}_{o1} \in \mathbb{R}^{d \times d}$ lifts the representation outside the axis-aligned convex hull of $\mathbf{z}_{\text{sim}, i}$ and $\mathbf{z}_{\text{dist}, i}$, enabling cross-channel manifold rotations.

**Cross-modality multi-omics integration (Tier 2).** Let $\mathbf{Z}_{\text{pro}} \in \mathbb{R}^{N \times d}$ denote the auxiliary omics embedding produced by a one-layer GCN over the shared-edge intersection scaffold $\mathcal{G}_{\text{common}} = \mathcal{G}_{\text{sim}} \cap \mathcal{G}_{\text{dist}}$, and let $\mathbf{Z}_{\text{fused}} \in \mathbb{R}^{N \times d}$ denote the Tier 1 RNA core. A second spot-adaptive gate $\mathbf{g}_{\text{modality}, i} \in (0, 1)^d$ regulates modality contribution at each spot:

$$\mathbf{g}_{\text{modality}, i} = \sigma\left( \mathbf{W}_{g2} [\mathbf{z}_{\text{fused}, i} \parallel \mathbf{z}_{\text{pro}, i}] + \mathbf{b}_{g2} \right) \tag{3}$$

followed by gated fusion and linear refinement into the unified multi-omics latent embedding $\mathbf{z}_{\text{fused\_pro}, i} \in \mathbb{R}^d$:

$$\mathbf{z}_{\text{fused\_pro}, i} = \mathbf{W}_{o2} \Big( \mathbf{g}_{\text{modality}, i} \odot \mathbf{z}_{\text{fused}, i} + (\mathbf{1} - \mathbf{g}_{\text{modality}, i}) \odot \mathbf{z}_{\text{pro}, i} \Big) + \mathbf{b}_{o2} \tag{4}$$

where $\mathbf{W}_{g2} \in \mathbb{R}^{d \times 2d}$, $\mathbf{W}_{o2} \in \mathbb{R}^{d \times d}$, and $\mathbf{b}_{g2}, \mathbf{b}_{o2} \in \mathbb{R}^d$ are trainable parameters (Supplementary Eqs. (S4)–(S6)). Stacking $\mathbf{z}_{\text{fused\_pro}, i}$ row-wise produces the unified embedding matrix $\mathbf{Z}_{\text{fused\_pro}} \in \mathbb{R}^{N \times d}$ used for downstream clustering and joint reconstruction.

---

## 2.7 Decoupled Multi-Task Reconstruction Decoders

Spatial multi-omics inputs exhibit an acute dimensionality asymmetry between transcriptomics ($G = \operatorname{dim}(\mathbf{X}_{\text{RNA}}) = 3000$ highly variable genes) and auxiliary modalities ($D_{\text{aux}} = \operatorname{dim}(\mathbf{X}_{\text{aux}}) \ll G$, e.g., $D_{\text{aux}} = 31$ for CLR-normalized ADT panels and $D_{\text{aux}} = 50$ for PCA-reduced ATAC profiles). When an autoencoder decodes only the concatenated feature matrix $[\mathbf{X}_{\text{RNA}} \parallel \mathbf{X}_{\text{aux}}]$ from a joint bottleneck, RNA residuals dominate $98.36\%$–$98.98\%$ of the concatenated error signal (Supplementary Proposition S2), allowing the auxiliary embedding $\mathbf{Z}_{\text{pro}}$ to collapse and leaving the individual RNA branches ($\mathbf{Z}_{\text{sim}}, \mathbf{Z}_{\text{dist}}$) without direct supervision.

To prevent representation collapse while aligning all modalities within a common basis, we route four decoupled reconstruction pathways through a **shared 512-dimensional semantic dictionary layer** $(\mathbf{W}_{\text{dec}1} \in \mathbb{R}^{512 \times d}, \mathbf{b}_{\text{dec}1} \in \mathbb{R}^{512})$:

$$\mathbf{h}(\mathbf{Z}) = \operatorname{ReLU}\left( \mathbf{Z} \mathbf{W}_{\text{dec}1}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}1}^T \right) \in \mathbb{R}^{N \times 512} \tag{5}$$

where $\mathbf{1}_N \in \mathbb{R}^{N \times 1}$ broadcasts the bias across all $N$ spots (Supplementary Eq. (S7)). From this shared hidden subspace, four linear heads reconstruct both joint and modality-specific inputs:

$$\hat{\mathbf{X}}_{\text{joint}} = \mathbf{h}(\mathbf{Z}_{\text{fused\_pro}}) \mathbf{W}_{\text{dec}5}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}5}^T, \qquad \hat{\mathbf{X}}_{\text{aux}} = \mathbf{h}(\mathbf{Z}_{\text{pro}}) \mathbf{W}_{\text{dec}4}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}4}^T \tag{6}$$

$$\hat{\mathbf{X}}_{\text{sim}} = \mathbf{h}(\mathbf{Z}_{\text{sim}}) \mathbf{W}_{\text{dec}2}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}2}^T, \qquad \hat{\mathbf{X}}_{\text{dist}} = \mathbf{h}(\mathbf{Z}_{\text{dist}}) \mathbf{W}_{\text{dec}2}^T + \mathbf{1}_N \mathbf{b}_{\text{dec}2}^T \tag{7}$$

where $\mathbf{W}_{\text{dec}5} \in \mathbb{R}^{(G + D_{\text{aux}}) \times 512}$, $\mathbf{W}_{\text{dec}4} \in \mathbb{R}^{D_{\text{aux}} \times 512}$, and $\mathbf{W}_{\text{dec}2} \in \mathbb{R}^{G \times 512}$ denote linear output heads with matching bias vectors (Supplementary Eqs. (S8)–(S11)). Weight tying of $(\mathbf{W}_{\text{dec}2}, \mathbf{b}_{\text{dec}2})$ across $\hat{\mathbf{X}}_{\text{sim}}$ and $\hat{\mathbf{X}}_{\text{dist}}$ constrains both the cosine-similarity and spatial-distance GCN branches to map onto an identical canonical gene-expression manifold prior to Tier 1 fusion.

---

## 2.8 Spatial Potts MRF Deep Embedding Clustering

Classical centroid clustering such as Euclidean $k$-means relies on a non-differentiable hard assignment step that prevents end-to-end representation refinement and ignores tissue geometry, producing fragmented ("salt-and-pepper") domain assignments. To jointly refine the latent manifold and enforce contiguous tissue organization, we couple **Deep Embedding Clustering (DEC)** with a **Spatial Potts Markov Random Field (MRF)** consensus prior.

**Soft assignment and frequency-balanced DEC sharpening.** Let $\mathbf{Z} = \mathbf{Z}_{\text{fused\_pro}} \in \mathbb{R}^{N \times d}$ and let $\mathbf{M} = [\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K]^T \in \mathbb{R}^{K \times d}$ denote $K$ learnable cluster prototypes. Using a Student's $t$-kernel with degree of freedom $\alpha = 1.0$ (Cauchy kernel; general form in Supplementary Eq. (S12)), the soft probability $q_{ik}$ assigning spot $i$ to cluster $k$ is:

$$q_{ik} = \frac{\left( 1 + \|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^2 \right)^{-1}}{\sum_{k'=1}^K \left( 1 + \|\mathbf{z}_i - \boldsymbol{\mu}_{k'}\|_2^2 \right)^{-1} + 10^{-12}} \tag{8}$$

Because Cauchy tails decay polynomially ($\mathcal{O}(\|\mathbf{z}_i - \boldsymbol{\mu}_k\|_2^{-2})$), distant spots retain stable attractive gradients (Supplementary Eq. (S13)). To amplify high-confidence assignments while preventing dominant clusters from absorbing rare micro-domains, we construct the frequency-normalized auxiliary target $\mathbf{P}_{\text{dec}} \in \mathbb{R}^{N \times K}$ (Supplementary Eq. (S14)):

$$p_{\text{dec}, ik} = \frac{q_{ik}^2 / f_k}{\sum_{k'=1}^K \left( q_{ik'}^2 / f_{k'} \right) + 10^{-12}}, \qquad f_k = \sum_{i=1}^N q_{ik} + 10^{-12} \tag{9}$$

**Spatial Potts MRF consensus modulation.** Rather than sharpening assignments solely in latent space, we introduce spatial coherence via a mean-field Potts MRF derived from the Gibbs Hamiltonian on $\mathcal{G}_{\text{dist}}$ (Supplementary Section S3.2, Eqs. (S15)–(S16)). Augmenting the unweighted binary spatial adjacency $\mathbf{A}_{\text{dist}} \in \{0, 1\}^{N \times N}$ with self-loops yields a row-stochastic transition operator $\mathbf{A}_{\text{norm}}$, whose product with $\mathbf{Q} = [q_{ik}]$ computes the expected local neighborhood label distribution $\mathbf{C} \in \mathbb{R}^{N \times K}$:

$$\mathbf{C} = \mathbf{A}_{\text{norm}} \mathbf{Q}, \qquad \mathbf{A}_{\text{norm}} = (\mathbf{D} + 10^{-12}\mathbf{I}_N)^{-1}(\mathbf{A}_{\text{dist}} + \mathbf{I}_N) \tag{10}$$

where $\mathbf{D}_{ii} = \sum_{j=1}^N (\mathbf{A}_{\text{dist}} + \mathbf{I}_N)_{ij}$ (Supplementary Eqs. (S17)–(S18)). Exponentially reweighting $\mathbf{P}_{\text{dec}}$ by the neighborhood consensus and renormalizing across clusters produces the spatially smoothed target distribution $\mathbf{P}_{\text{final}} \in \mathbb{R}^{N \times K}$:

$$p_{\text{final}, ik} = \frac{p_{\text{dec}, ik} \exp\left( \lambda_{\text{spatial}} C_{ik} \right)}{\sum_{k'=1}^K p_{\text{dec}, ik'} \exp\left( \lambda_{\text{spatial}} C_{ik'} \right) + 10^{-12}} \tag{11}$$

where $\lambda_{\text{spatial}} = 0.15$ governs local spatial consensus strength without overriding sharp molecular boundaries (Supplementary Eq. (S19)).

---

## 2.9 Two-Stage Training Objective

DualGate-DEC is trained across a 180-epoch two-stage curriculum comprising representation pre-training followed by spatial Potts DEC fine-tuning, linked via a Silhouette-guided checkpoint handover.

**Stage 1: Representation pre-training ($1 \le t \le 80$).** During pre-training over the encoder, fusion, and decoder parameters $\Theta_1$, the network learns a modality-aligned and spatially coherent latent space without clustering prototype distortion. The pre-training objective combines the multi-task reconstruction loss $\mathcal{L}_{\text{multi-rec}}$ across all four decoder heads, the pairwise cosine-sigmoid spatial coherence loss $\mathcal{L}_{\text{spatial}}$ on $\mathbf{Z}_{\text{fused}}$, and combined $L_1/L_2$ weight regularization $\mathcal{L}_{\text{reg}}$ (expanded in Supplementary Section S4, Eqs. (S20)–(S26)):

$$\mathcal{L}_{\text{total}}^{\text{pre}}(\Theta_1) = \beta_{\text{pre}}\,\mathcal{L}_{\text{multi-rec}} + \gamma\,\mathcal{L}_{\text{spatial}} + \delta\,\mathcal{L}_{\text{reg}}(\Theta_1) \tag{12}$$

Because late-stage reconstruction can over-smooth cluster separability (*checkpoint decay*), Stage 1 tracks epoch-wise Silhouette scores and, after epoch 80, executes a **Peak Silhouette Bridge Handover**: parameters rewind to the peak-Silhouette checkpoint $\Theta_1^*$, cluster prototypes $\mathbf{M}$ are initialized via 20-restart $k$-means on the restored latent manifold, and the optimizer is reset for fine-tuning (Supplementary Eqs. (S32)–(S36)).

**Stage 2: Spatial Potts DEC fine-tuning ($81 \le t \le 180$).** During fine-tuning over $\Theta_2 = \Theta_1 \cup \{\mathbf{M}\}$, the spatial Potts KL divergence $\mathcal{L}_{\text{KL}}(\mathbf{Q} \parallel \mathbf{P}_{\text{final}})$ (Supplementary Eq. (S27)) is activated to jointly compact clusters and align spot assignments with local tissue neighborhoods, while the reconstruction weight is attenuated by $80\%$ ($\beta_{\text{ft}} = 5.0$) so reconstruction preserves biological manifold fidelity without overwhelming the bounded clustering gradient (Supplementary Proposition S3 and Theorem S1, Eqs. (S28)–(S30)):

$$\mathcal{L}_{\text{total}}^{\text{ft}}(\Theta_2) = \beta_{\text{ft}}\,\mathcal{L}_{\text{multi-rec}} + \gamma\,\mathcal{L}_{\text{spatial}} + \delta\,\mathcal{L}_{\text{reg}}(\Theta_2) + \lambda_{\text{dec}}\,\mathcal{L}_{\text{KL}}(\mathbf{Q} \parallel \mathbf{P}_{\text{final}}) \tag{13}$$

Every $5$ epochs, $\mathbf{P}_{\text{final}}$ is updated asynchronously with stopped gradients to prevent circular EM oscillations; at each epoch, dual-cluster Silhouette competition selects between soft DEC assignments and latent $k$-means, and the global best checkpoint across all 180 epochs is restored at convergence (Supplementary Eqs. (S37)–(S40), Algorithm S1).

**Stage schedule.** The epoch ranges, reconstruction weights $\beta(t)$, and Adam learning rates $\eta(t)$ across both phases are governed by:

$$\big(\text{Phase},\, \beta(t),\, \eta(t)\big) = \begin{cases} \big(\text{Pre-training},\, \beta_{\text{pre}} = 25.0,\, \eta_1 = 10^{-3}\big), & 1 \le t \le 80 \\ \big(\text{Fine-tuning},\, \beta_{\text{ft}} = 5.0,\, \eta_2 = 10^{-4}\big), & 81 \le t \le 180 \end{cases} \tag{14}$$

with remaining weights fixed at $\gamma = 10.0$, $\delta = 1.0$, $\lambda_{\text{dec}} = 1.0$, $\lambda_{L_1} = 10^{-4}$, and $\lambda_{L_2} = 10^{-3}$ (Supplementary Eq. (S31), Table S1).
