# Paper Writing Style, Formatting, and Structural Guide

**Analyzed Reference Corpus:**
- **Main Paper 1 (*Bioinformatics* / Oxford University Press):** [`arise.pdf`](file:///d:/FYDP/GATCON/d1_test/papers/arise.pdf) — *"ARISE: RNA-anchored shared-edge topology and hierarchical fusion for spatial multi-omics integration"* (Wang et al., *Bioinformatics*, 42(7), btag465, 2026; 12 pages)
- **Supplementary Data 1 (*Bioinformatics* LaTeX Companion):** [`arise_supplementary_data.pdf`](file:///d:/FYDP/GATCON/d1_test/papers/arise_supplementary_data.pdf) — Online Supplementary Material for ARISE (9 pages)
- **Main Paper 2 (*Information Fusion* / Elsevier):** [`SpaFusion.pdf`](file:///d:/FYDP/GATCON/d1_test/papers/SpaFusion.pdf) — *"SpaFusion: A multi-level fusion model for clustering spatial multi-omics data"* (Chen et al., *Information Fusion*, 124, 103372, 2025; 14 pages)

---

## 1. Executive Comparison: Two Publication Archetypes

Although all three documents address the same scientific domain—**graph-based spatial multi-omics representation learning and domain clustering**—the two main papers represent **two distinct academic publishing models**:

1. **The Compact Computational Biology Journal Model (`arise.pdf` + `arise_supplementary_data.pdf`):**
   - **Audience & Framing:** Computational biologists and bioinformaticians (*Bioinformatics*, *Nature Methods* / *Genome Biology* family).
   - **Design Philosophy:** Keep the main manuscript lean (~5,400 words total including captions), visually airy (7 multi-panel figures, **0 tables** in main text), and focused on core algorithmic intuition, key benchmarks, and deep biological interpretation. Offload formal mathematical proofs, dataset inventories, metric equations, exhaustive hyperparameter sweeps, runtime/memory tables, and secondary ablations into a dedicated **Supplementary Data** PDF (~3,826 words, 14 tables, 4 theorems + proofs, 2 extra figures).
   - **Narrative Order:** `Introduction` → `Materials and methods` → `Results` → `Discussion`.

2. **The Self-Contained Engineering / AI Journal Model (`SpaFusion.pdf`):**
   - **Audience & Framing:** Machine learning, multi-view learning, and data fusion researchers (*Information Fusion*, IEEE/Elsevier transactions).
   - **Design Philosophy:** Deliver a **single, self-contained monograph** (~8,987 words excluding references, 14 pages, **no supplementary file**) containing everything needed for verification inside one document: dedicated Related Works, explicit bulleted contributions, detailed multi-stage math, an `Algorithm 1` pseudocode block, dataset preprocessing, formal metric equations, 5 inline quantitative tables (`mean ± std` over 10 runs), parameter sensitivity, 4-variant ablation, hardware-pinned complexity profiling, synthetic noise stress-testing, and a biological case study.
   - **Narrative Order:** `Introduction` → `Related works` → `Methods` → `Experiments` → `Results` → `Discussions` → `Conclusion`.

### Side-by-Side Quantitative Comparison

| Dimension | [`arise.pdf`](file:///d:/FYDP/GATCON/d1_test/papers/arise.pdf) | [`arise_supplementary_data.pdf`](file:///d:/FYDP/GATCON/d1_test/papers/arise_supplementary_data.pdf) | [`SpaFusion.pdf`](file:///d:/FYDP/GATCON/d1_test/papers/SpaFusion.pdf) |
| :--- | :--- | :--- | :--- |
| **Publisher / Venue** | Oxford University Press (*Bioinformatics*) | OUP *Bioinformatics* Online Supplementary | Elsevier (*Information Fusion*) |
| **Length** | 12 pages | 9 pages | 14 pages |
| **Prose + Captions + Tables** | ~5,403 words | ~3,826 words | ~8,987 words |
| **Total Extracted Tokens** | 6,304 tokens (incl. refs/headers) | 4,027 tokens | 11,092 tokens |
| **Main Numbered Sections** | 4 (`1`–`4`) + 6 back-matter headings | 6 (`1`–`6`) + `References` | 7 (`1.`–`7.`) + 5 back-matter headings |
| **Abstract Format** | Structured (`Motivation / Results / Availability and implementation`), ~217 body words | None (starts directly at `1 Theorems`) | Unstructured 1-paragraph (~193 words) + `Keywords` box (3 terms) |
| **Related Work Placement** | Folded into `1 Introduction` (Paragraphs 2–3) | None | Dedicated `2. Related works` (3 thematic paragraphs + gap summary) |
| **Contribution Bullet List** | Prose only (no bullets in Intro) | Bullets used in `6.1`, `6.2`, `6.7` | 4 explicit bullet points at end of `1. Introduction` |
| **Figures** | 7 figures (`Figure 1`–`7`), multi-panel `(i)–(v)` and `(a)–(f)` | 2 figures (`Figure 1`–`2`) | 11 figures (`Fig. 1.`–`11.`), panels `(a)–(b)` and grid layouts |
| **Tables** | **0** (all pushed to Supplementary) | **14** (`Table 1`–`14`) | **5** (`Table 1`–`5`) |
| **Numbered Equations** | 20 (`(1)`–`(20)`) | 40 (`(1)`–`(40)`) | 32 (`(1)`–`(32)`) |
| **Pseudocode / Theorems** | Cross-references `Theorems T1–T4` | 4 formal `Theorem` statements + `Proof`s | `Algorithm 1` pseudocode float |
| **References** | 18 entries (unnumbered alphabetical) | 8 entries (unnumbered alphabetical) | 56 entries (`[1]`–`[56]` order of appearance) |
| **In-Text Citation Style** | Author-year, **no comma**: `(Hao et al. 2025)` | Author-year, **with comma**: `(Huang et al., 2024)` | Numeric brackets: `[1–4]`, `[9, 10]` |

---

## 2. Physical Layout, Page Geometry, and Typography

All three PDFs share a **two-column academic print layout** on A4-width (`595.28 pt` = `210.0 mm`) pages, transitioning from a **single-column full-width header/frontmatter block** on Page 1 to **two balanced columns** from Section 1 onward.

### 2.1 Page Dimensions, Margins, and Column Geometry

| Metric | `arise.pdf` | `arise_supplementary_data.pdf` | `SpaFusion.pdf` |
| :--- | :--- | :--- | :--- |
| **Page Size (pt / mm)** | `595.28 × 782.36 pt` (`210.0 × 276.0 mm`) | `595.28 × 782.36 pt` (`210.0 × 276.0 mm`) | `595.28 × 793.70 pt` (`210.0 × 280.0 mm`) |
| **Left / Right Margins** | `42.5 pt` / `48.2 pt` (body excl. watermark) | `35.4 pt` / `35.4 pt` (symmetric) | `37.6 pt` / `37.6 pt` (symmetric) |
| **Top / Bottom Margins** | `56.9 pt` / `47.4 pt` | `50.4 pt` / `51.3 pt` | `55.2 pt` / `52.0 pt` |
| **Column Width** | `243.8 pt` each (left `[42.5, 286.3]`, right `[303.3, 547.1]`) | `253.7 pt` each (left `[35.4, 289.1]`, right `[306.1, 559.8]`) | `251.1 pt` each (left `[37.6, 288.7]`, right `[306.6, 557.7]`) |
| **Column Gutter** | `17.0 pt` | `17.0 pt` | `17.9 pt` |
| **Line Spacing (Leading)** | `11.5–11.7 pt` on `9.0 pt` body (~1.28×) | `11.5 pt` on `8.0 pt` body (~1.44×) | `10.5 pt` on `7.97 pt` body (~1.32×) |
| **Production Engine** | Arbortext Advanced Print Publisher (`PDFlib+PDI`) | `pdfTeX-1.40.27` (LaTeX with `hyperref`) | Elsevier production (`Acrobat Distiller 8.1.0`) |

> **Key Layout Observation:** In both `arise.pdf` and `SpaFusion.pdf`, wide figures (`Figure 1`–`7` in ARISE; `Fig. 1`, `Fig. 2`, `Fig. 3`, `Fig. 4`, `Fig. 6`, `Fig. 7`, `Fig. 8`, `Fig. 9`, `Fig. 10`, `Fig. 11`, plus `Table 2`, `Table 3`, `Table 5` in SpaFusion) span **both columns** (`figure*` / `table*` in LaTeX) and are anchored strictly at the **top** or **bottom** of pages—never breaking prose mid-column awkwardly. Narrow tables (`Table 1`, `Table 4` in `SpaFusion`; `Tables 2–7, 9–13` in `arise_supp`) and `Algorithm 1` sit inside a **single column**.

### 2.2 Font Family, Sizing, and Color Hierarchy

#### A. [`arise.pdf`](file:///d:/FYDP/GATCON/d1_test/papers/arise.pdf) — Modern Sans-Serif (*Bioinformatics* Production Style)
- **Primary Typeface:** `SourceSansPro` family (`Regular`, `Bold`, `Semibold`, `Italic`, `SemiboldIt`, `Light`), paired with `TeX_CM_Maths_Symbols` for mathematics.
- **Hierarchy:**
  - **Paper Title (p. 1):** `20.0 pt` `SourceSansPro-Bold`, left-aligned, black `(0, 0, 0, 1)`.
  - **Author List:** `11.0 pt` `SourceSansPro-Semibold` with `7.33 pt` superscript affiliation numbers (`1`, `2`, `3`), equal-contribution dagger (`†`), corresponding-author asterisk (`*`), and green ORCID icon.
  - **Affiliations & Footnotes:** `8.0 pt` `SourceSansPro-Light`.
  - **Abstract Heading:** `13.0 pt` `SourceSansPro-Bold`; sub-labels (`Motivation:`, `Results:`, `Availability and implementation:`) in `9.5 pt` `SourceSansPro-SemiboldIt`.
  - **H1 (`1 Introduction`, `2 Materials and methods`, `3 Results`, `4 Discussion`, Back-matter):** `15.5 pt` `SourceSansPro-Bold` (notice **no period** after the section number).
  - **H2 (`2.1`, `2.2`, `3.1`, ...):** `13.0 pt` `SourceSansPro-Bold` (no period after subsection number).
  - **H3 / Inline Paragraph Lead-ins (`Primary modality (RNA).`, `Spatial regularization loss.`):** `9.0 pt` `SourceSansPro-Bold` followed immediately by a period and body text on the same line.
  - **Body Text:** `9.0 pt` `SourceSansPro-Regular` (`~11.5–11.7 pt` leading), fully justified, **no first-line paragraph indent** on the opening paragraph of a section, small indent on subsequent paragraphs.
  - **Hyperlinks & Citations:** Pure RGB blue `(0.0, 0.0, 1.0)` for all in-text citations (`(Hu et al. 2021)`), figure/equation cross-references (`Fig. 1`, `Equation (1)`), URLs/DOIs, and supplementary links (`supplementary data`).
  - **Captions:** `8.0 pt`; prefix `Figure 1` in `SourceSansPro-Semibold`, description in `SourceSansPro-Regular`.
  - **Page Chrome:** Top header shows `Bioinformatics, 2026, Vol, 42, Issue 7` (`6.5 pt`) + outer page number; bottom-right vertical grey `(0.4, 0.4, 0.4)` `ArialMT` download watermark; p. 1 footer holds `Received / Revised / Accepted` dates + CC-BY 4.0 license box separated by a thin horizontal rule.

#### B. [`arise_supplementary_data.pdf`](file:///d:/FYDP/GATCON/d1_test/papers/arise_supplementary_data.pdf) — Classic LaTeX Computer Modern (Author-Compiled OUP Template)
- **Primary Typeface:** LaTeX Computer Modern (`CMR8`, `CMBX10`, `CMBX8`, `CMSSBX10`, `CMSS12`, `CMSS8`, `CMMI8`, `CMSY8`).
- **Hierarchy:**
  - **Banner & Title (p. 1):** Gray/boxed template header (`Journal Title Here, YEAR, pp. 1–9` in `CMSSI8 8pt`), `Paper` tag (`CMSSBX10 7pt`), centered `20.0 pt` `CMSSBX10` title, `12.0 pt` `CMSS12` authors, `8.0 pt` affiliations.
  - **H1 (`1 Theorems`, `2 Proofs`, `3 Datasets`, `4 Metrics`, `5 Experimental Setup`, `6 Result`, `References`):** `11.46 pt` `CMBX10`.
  - **H2 (`2.1 Theorem 1`, `3.1`, `6.1`, ...):** `8.0 pt` `CMBX8`.
  - **H3 / Unnumbered Subheadings (`Mouse embryo Spatial-Mux-seq dataset`, `Why strict intersection prevails.`):** `8.0 pt` `CMBX8` (standalone line or bold inline lead-in).
  - **Theorem Statements:** Bold `Theorem 1`–`4` (`8.0 pt` `CMSSBX10` / `CMBX8`) with math-rich indented/tight prose and immediate one-sentence takeaway directly underneath.
  - **Body Text:** `8.0 pt` `CMR8` (`11.5 pt` leading).
  - **Tables & Captions:** `Table 1`–`14` and `Figure 1`–`2` labels in `CMSSBX10` (`7.5–8.0 pt`), table headers in `CMSSBX10 5.97 pt`, table cells in `CMSS8 5.97 pt` (sans-serif inside tables for compact tabular readability!).
  - **References (p. 9):** `7.75 pt` `CMR7` with italicized journal titles (`CMTI7`).

#### C. [`SpaFusion.pdf`](file:///d:/FYDP/GATCON/d1_test/papers/SpaFusion.pdf) — Traditional Serif (*Information Fusion* / Elsevier Style)
- **Primary Typeface:** `CharisSIL` (`Regular`, `Bold`, `Italic`) for all prose, headings, and tables; `STIXMath` and `LMRoman10` for mathematics.
- **Hierarchy:**
  - **Masthead (p. 1):** Full-width grey-shaded journal banner (`Contents lists available at ScienceDirect / Information Fusion / journal homepage`), Elsevier tree logo on left, journal cover thumbnail + CrossMark badge on right, `Full length article` label (`9.56 pt`).
  - **Paper Title (p. 1):** `13.45 pt` `CharisSIL`, left-aligned, black.
  - **Authors & Affiliations:** `10.56 pt` `CharisSIL` with italic superscript letters (`a`, `b`), green ORCID icon, `6.38 pt` `CharisSIL-Italic` affiliations.
  - **Abstract Box:** Framed between two horizontal rules under `A R T I C L E  I N F O` and `A B S T R A C T` (wide-tracked small-caps feel), split into a narrow left `Keywords:` column (`6.38 pt`) and wide right Abstract text (`7.17 pt`).
  - **H1 (`1. Introduction`, `2. Related works`, ..., `7. Conclusion`, Back-matter):** `7.97 pt` `CharisSIL-Bold` with a **trailing period after the section number** (`1. Introduction`).
  - **H2 (`3.1.`, `4.1.`, `5.1.`, ...) & H3 (`3.4.1.`, `5.7.1.`, ...):** `7.97 pt` `CharisSIL-Italic` with a trailing period (`3.1. High-order graph construction`).
  - **Body Text:** `7.97 pt` `CharisSIL` (`10.5 pt` leading), justified, with first-line paragraph indents on **every paragraph except the first paragraph** after a heading.
  - **Captions & Tables:** `6.38 pt` (`Fig. 1.` / `Table 1` label in `CharisSIL-Bold` on its own line for tables, inline for figures), table text `6.38 pt`.
  - **Accent Color:** Elsevier signature **teal/cyan** `(0.0, 0.502, 0.675)` (`#0080AC`) on DOIs, citations (`[1–4]`), `Fig.` / `Table` / `Eq.` cross-references, top journal citation bar, and `Algorithm 1` rules.
  - **Running Header / Footer (pp. 2–14):** Top-left `F. Chen et al.` (`CharisSIL-Italic 6.38 pt`), top-right `Information Fusion 124 (2025) 103372`, bottom-center page number (`6.38 pt`).

---

## 3. Macro-Structure and Word-Budget Allocation

### 3.1 Section Architecture & Measured Proportions

| Structural Component | `arise.pdf` (Main) | `arise_supplementary_data.pdf` | `SpaFusion.pdf` (Main) |
| :--- | :--- | :--- | :--- |
| **Frontmatter / Abstract** | ~113 + 221 words (p. 1) | ~34 words (title/authors p. 1) | ~62 + 201 words (p. 1) |
| **1. Introduction** | **639 words** (~11.8% of body) | — | **837 words** (~9.3% of body) |
| **2. Related Works** | *(folded into Intro)* | — | **419 words** (~4.7% of body) |
| **Theory & Proofs** | *(summarized in Intro/Methods)* | **1,234 words** (`1 Theorems` 539 + `2 Proofs` 695) | *(none)* |
| **Methods / Methodology** | **1,865 words** (~34.5% of body; incl. Fig. 1) | *(hyperparams + clustering + downstream in `5 Setup`: 331w)* | **2,730 words** (~30.4% of body; incl. Fig. 1 & Alg. 1) |
| **Datasets & Evaluation Setup** | *(260 words inside `3.1`)* | **523 words** (`3 Datasets` 393 + `4 Metrics` 130) | **1,185 words** (`4. Experiments`, incl. Table 1 & metrics) |
| **Results & Ablations** | **2,230 words** (~41.3% of body; incl. Figs. 2–7) | **1,738 words** (`6 Result`, incl. Tables 3–14 & Figs. 1–2) | **2,883 words** (~32.1% of body; incl. Tables 2–5 & Figs. 2–11) |
| **Discussion** | **332 words** (~6.1% of body) | — | **337 words** (~3.7% of body) |
| **Conclusion** | *(merged into Discussion)* | — | **296 words** (~3.3% of body) |
| **End-Matter Statements** | ~196 words (`Author contributions`, `Supplementary`, `Conflict`, `Funding`, `Data availability`) | — | ~157 words (`CRediT`, `Funding`, `Competing interest`, `Data availability`) |
| **References** | 18 entries (~481 words) | 8 entries (~135 words) | 56 entries (~1,804 words) |

> **Structural Takeaway:** In **both** papers, **Methods + Results account for ~63–76% of the entire manuscript's word budget**. Introductions are tightly capped under 850 words, and Discussions/Conclusions are remarkably concise (~300–340 words each). Never bloat the Introduction or Discussion at the expense of Methods clarity and Results evidence.

### 3.2 Complete Section Outline Blueprints

#### Blueprint A: `arise.pdf` + `arise_supplementary_data.pdf`
- **Main Paper (`arise.pdf`):**
  - `Abstract` (`Motivation`, `Results`, `Availability and implementation`)
  - `1 Introduction`
  - `2 Materials and methods`:
    - `2.1 Overview of ARISE` (+ `Figure 1`)
    - `2.2 RNA expression anchored shared-edge topology`
    - `2.3 RNA-anchored multi-modal graph encoding` (bold inline `Primary modality (RNA).`, `Auxiliary modalities.`)
    - `2.4 Hierarchical fusion of RNA and auxiliary omics representations`
    - `2.5 Optimization objective of ARISE` (bold inline `Spatial regularization loss.`, `Reconstruction loss.`, `L1/L2 regularization.`, `Total loss function.`)
  - `3 Results`:
    - `3.1 Datasets and evaluation`
    - `3.2 Identification of spatial domains in simulated multi-omics data` (+ `Figure 2`)
    - `3.3 Performance on bimodal spatial multi-omics datasets` (+ `Figure 3`, `Figure 4`)
    - `3.4 Extension to tri-modal spatial multi-omics integration` (+ `Figure 5`)
    - `3.5 Ablation and sensitivity analyses` (+ `Figure 6`)
    - `3.6 Biological interpretation of ARISE-derived spatial domains` (+ `Figure 7`)
  - `4 Discussion`
  - Back-matter: `Author contributions` (CRediT roles inline in parentheses), `Supplementary material`, `Conflict of interests`, `Funding`, `Data availability`, `References`
- **Supplementary Data (`arise_supplementary_data.pdf`):**
  - `1 Theorems` (`Theorem 1`–`4`, Eqs. 1–2)
  - `2 Proofs` (`2.1 Theorem 1` – `2.4 Theorem 4`, Eqs. 3–35)
  - `3 Datasets` (`3.1`–`3.5` + unnumbered `Mouse embryo Spatial-Mux-seq dataset` + `Table 1`)
  - `4 Metrics` (ARI, NMI, AMI, SC, DB; Eqs. 36–40)
  - `5 Experimental Setup` (`5.1 Baseline Implementation Details`, `5.2 Hyperparameter Configurations` + `Table 2`, `5.3 Clustering Procedure`, `5.4 Downstream Analysis Pipeline`)
  - `6 Result` (`6.1 Ablation on Feature Integration Hierarchy` + `Table 3`, `6.2 Comprehensive Evaluation of Graph Fusion Strategies` + `Tables 4–5`, `6.3 Hyperparameter Sensitivity Analysis on RNA+ATAC Data` + `Table 6`, `6.4 Impact of Spatial Neighborhood Definitions on Embedding Stability` + `Table 7`, `6.5 Sensitivity Analysis of Highly Variable Genes` + `Table 8`, `6.6 Computational Efficiency Comparison` + `Tables 9–11`, `6.7 Evaluation of Highly Variable Gene (HVG) Selection Methods` + `Tables 12–13`, plus `Table 14` and `Figures 1–2`)
  - `References`

#### Blueprint B: `SpaFusion.pdf`
- `Abstract` + `Keywords`
- `1. Introduction` (6 paragraphs + 4-bullet `The main contributions of this work are as follows:`)
- `2. Related works` (GNNs in spatial omics → GCN-based → GAE-based → Multi-modal integration → Gap statement)
- `3. Methods`:
  - `3.1. High-order graph construction` (+ `Fig. 1.`)
  - `3.2. Spatial graph construction`
  - `3.3. Local and global representation learning`
  - `3.4. Multi-level information fusion` → `3.4.1. Intra-omic information fusion` + `3.4.2. Cross-omics adaptive fusion`
  - `3.5. Consensus clustering`
  - `3.6. Implementation details` (+ `Algorithm 1`)
- `4. Experiments`:
  - `4.1. Datasets` (+ `Table 1`)
  - `4.2. Data pre-processing`
  - `4.3. Evaluation metrics` (ARI, NMI, ACC, AMI, F1; Eqs. 28–32)
  - `4.4. Compared methods` (10 baselines in `Name [ref]: 1-sentence summary.` format)
- `5. Results`:
  - `5.1. Clustering results` (+ `Tables 2–4`)
  - `5.2. Visualization of results` (+ `Figs. 2–4`)
  - `5.3. Parameter analysis` (`(1) Impact of K_s:` + `Fig. 5`, `(2) Impact of \lambda_1 and \lambda_2:` + `Fig. 6`)
  - `5.4. Ablation study` (+ `Fig. 7`)
  - `5.5. Computational complexity` (+ `Fig. 8`)
  - `5.6. Noise analysis` (+ `Table 5`)
  - `5.7. Case study in mouse breast cancer` (+ `Fig. 9`) → `5.7.1. Marker gene identification` (+ `Fig. 10`) + `5.7.2. Survival analysis` (+ `Fig. 11`)
- `6. Discussions`
- `7. Conclusion`
- Back-matter: `CRediT authorship contribution statement`, `Funding`, `Declaration of competing interest`, `Data availability`, `References` (`[1]`–`[56]`)

---

## 4. Section-by-Section Writing Mechanics

### 4.1 Title & Acronym Convention
Both papers follow the exact same **two-part title formula**:
> **`<ACRONYM>: <Key technical mechanism> for <Target task / data modality>`**
- `ARISE: RNA-anchored shared-edge topology and hierarchical fusion for spatial multi-omics integration`
- `SpaFusion: A multi-level fusion model for clustering spatial multi-omics data`

Notice:
- Sentence case after the colon (capitalizing only the first word and acronyms like `RNA`).
- The acronym is catchy, pronounceable, and expanded cleanly on first mention in the Abstract/Introduction (`Anchored RNA for Integrated Spatial Embedding (ARISE)`).

### 4.2 Abstract Anatomy
Both abstracts are ~193–221 words and execute the exact same 5-move sequence:
1. **Broad Context (1 sentence):** State what spatial multi-omics enables (`jointly profile transcriptomes, proteins and chromatin accessibility in situ` / `simultaneous profiling of transcriptomics and proteomics data at shared spatial coordinates`).
2. **Bottleneck / Gap (1–2 sentences):** Introduced with `"However, most existing..."` — pinpoint the exact architectural flaw of current methods (independent noisy kNN graphs causing topological discordance in `arise.pdf`; spatial transcriptomics-only focus or lack of global context in `SpaFusion.pdf`).
3. **Method Walkthrough (3–4 sentences):** Introduce the model (`We present...` / `In this study, we introduce...`) and walk sequentially through its core modules (`defines a shared-edge topology... encodes auxiliary modalities... integrates them through inside-out hierarchical fusion` / `construct a high-order cell graph... integrates graph autoencoders and transformer modules... multi-level information fusion... consensus clustering`).
4. **Theoretical & Quantitative Proof (1–2 sentences):** `arise.pdf` highlights its theorem (`We further show theoretically that...`) plus benchmarks (`spanning simulated and real datasets in bi-modal and tri-modal settings`); `SpaFusion.pdf` highlights its scale (`Extensive experiments on three real-world spatial transcriptome–proteome datasets demonstrate that SpaFusion consistently outperforms state-of-the-art methods`).
5. **Downstream Biological Utility (1 sentence):** Connect latent embeddings back to biology (`marker-based domain annotation, pathway enrichment, and cis-regulatory inference` / `valuable insights into the spatial organization of cell types and their potential roles in disease mechanisms`).

### 4.3 Introduction Funnel (Paragraph-by-Paragraph)
Both introductions use a disciplined **5-to-6-paragraph funnel** that moves from broad biological motivation down to the exact method gap:

- **Paragraph 1 (Biological Problem):** Why spatial cellular heterogeneity matters (`identifying cell types`, `tissue microenvironments`, single-cell lacks spatial context → spatial omics preserves it).
- **Paragraph 2 (Single-Omic Graph Foundations):** Survey classic spatial transcriptomics GNNs (`SpaGCN`, `STAGATE`, `SEDR`, `GraphST`, `BayesSpace`) and establish GNNs as the dominant paradigm, while noting single-omic noise/dropout limits.
- **Paragraph 3 (Shift to Multi-Omics & Specific Technical Flaw):** Introduce spatial multi-omics platforms (`Visium`, `SPOTS`, `Stereo-seq`) and existing baselines (`SpatialGlue`, `PRAGA`). Then diagnose the **exact structural flaw**:
  - `arise.pdf` splits this across Paras 3–4: modality-specific kNN graphs embed dropout/ATAC-sparsity/protein-noise directly into neighborhood topology, creating discordant topologies that propagate spurious edges.
  - `SpaFusion.pdf` targets `SpatialGlue` specifically (Para 4): focuses on local interactions while neglecting global context, and relies on 1st-order neighborhoods rather than higher-order motifs.
- **Paragraph 4/5 (Proposed Solution & Mechanism Preview):** Open with `"Here, we present..."` (`arise.pdf`) or `"To address these challenges, we propose..."` (`SpaFusion.pdf`), using ordered step markers (`First,... Second,... Third,... Finally,...` in `SpaFusion.pdf`; sequential narrative in `arise.pdf`).
- **Closing Paragraph / Bullet List:**
  - `arise.pdf` closes with a crisp **Theory + Empirics** pairing paragraph (`Complementing its architectural framework, ARISE is firmly grounded in both theoretical and empirical analyses. Theoretically, ... Empirically, ...`).
  - `SpaFusion.pdf` closes with **4 bold-free contribution bullets** (`The main contributions of this work are as follows:` covering (1) overall framework, (2) high-order motif graph, (3) local+global multi-level fusion, (4) benchmarks + disease biomarker case study).

### 4.4 Related Works Strategy
- **`SpaFusion.pdf` (`2. Related works`):** Uses a **three-cluster taxonomy** (`GCN-based methods`, `GAE-based methods`, `Integration strategies`) with 2–3 concrete method citations per paragraph (`SpaGCN`, `STMGCN`; `SEDR`, `stAA`; `stMVC`, `SpatialGlue`, `MAFN`), ending with a transition paragraph (`Notably, except for SpatialGlue, the methods mentioned above rely solely on spatial transcriptomics... These challenges underscore the need...`).
- **`arise.pdf`:** Omits a separate `Related works` section entirely to save space for biological results—compressing prior GNNs into Para 2 and multi-omic baselines into Para 3 of `1 Introduction`.

### 4.5 Methods Writing Pattern
Both papers follow a **strict top-down pipeline flow**:
1. **Overview + Architecture Diagram (`Fig. 1`) first:** Introduce the whole pipeline in prose matched 1-to-1 with the labeled blocks in Figure 1 before writing a single equation.
2. **Preprocessing first (`arise.pdf` `2.1`) or in Experiments (`SpaFusion.pdf` `4.2`):** Specify exact gene/spot QC, top `3,000` HVGs, `log1p`/scaling, `CLR` for ADT, `TF-IDF + LSI` for ATAC (ARISE) or `PCA` (`100` for RNA, `M^v - 1` for ADT in SpaFusion).
3. **Module-by-Module Equation Pattern:** Every mathematical step follows the strict micro-pattern:
   > **[Motivation sentence: *why* this step is needed]** → **[Notation setup: `Let F denote...`]** → **[Centered numbered Equation `(k)`]** → **[`where <symbol> denotes <meaning>` for every new symbol]** → **[Design takeaway: *what* property this guarantees]**.
4. **Progressive Loss Composition:** Break the objective into named sub-losses (`Spatial regularization loss`, `Reconstruction loss` at modality & fused levels, `L1/L2 regularization`, `Consensus clustering loss`), then finish with a single weighted `Total loss function` equation (Eq. `20` in ARISE; Eq. `27` in SpaFusion) + hyperparameter symbol list.
5. **Pretraining vs Joint Optimization + Pseudocode:** `SpaFusion.pdf` (`3.6` + `Algorithm 1`) explicitly separates Stage 1 (pretrain autoencoders + transformers on $L_{rec_l} + L_{rec_g}$) from Stage 2 (K-means initialization + iterative consensus clustering loop minimizing total $L$).

### 4.6 Experimental Setup & Supplementary Offloading
- **Datasets Summary Table (`Table 1` in both `SpaFusion.pdf` and `arise_supplementary_data.pdf`):** Always tabulate `Dataset | Resolution (if applicable) | Cells/Spots | Genes/Features per modality | #Clusters / GT | Species | Platform`.
- **Reproducibility Details (`arise_supplementary_data.pdf` Section `5`):** Notice how thorough *Bioinformatics* Supplementary `5.1–5.4` is—stating exact baseline policy (official code, default hyperparameters, no custom reimplementation), exact default hyperparameters per modality regime (`Table 2`: Dual-Modal vs Tri-Modal vs Simulation), exact clustering determinism (`n_init=10`, `random_state=42`), and exact bioinformatics package versions/dates (`Scanpy`, `org.Mm.eg.db v3.21.0`, `GRCm39/mm39`, `GENCODE vM34`, `clusterProfiler`, BH `adjusted p < 0.05`, `SciPy` two-sided Pearson).
- **Baseline Descriptions (`SpaFusion.pdf` `4.4`):** When listing many baselines (`10` baselines), format each as a tight one-line `<Bold/Plain Name [ref]: One-sentence architectural mechanism.>` entry, followed by a brief rule explaining how single-omic baselines were adapted to multi-omics (`applied independently to transcriptomics and proteomics`).

### 4.7 Results Narrative Structure
Every single Results subsection in both papers executes the same **4-beat narrative loop**:
1. **Experiment Objective:** State *what* hypothesis or regime is being tested (`To evaluate scalability to tri-modal settings...`, `We next examined whether the advantages observed in simulation generalize...`).
2. **Quantitative Win (with exact numbers & figure/table pointers):** Lead with the main finding (`ARISE achieved the best performance across all label-based metrics (Fig. 3a), with AMI = 0.4141, NMI = 0.4182, and ARI = 0.3427`), explicitly naming relative margins (`surpasses the second-best method by 7.5%, 1.2%, and 14.6%`).
3. **Diagnostic Comparison Against Competitors (Why others failed):** Don't just say competitors scored lower—diagnose **failure modes visually/mechanistically** (`affected by salt-and-pepper noise, suggesting insufficient spatial constraint`, `over-smoothed the tissue, collapsing heterogeneous regions`, `fragmented assignments`, `relatively large standard deviations suggest limited stability`).
4. **Biological / Mechanistic Takeaway:** Close the subsection with a concluding synthesis (`These results demonstrate that...`, `These findings underscore...`).

**Notice the progression of Results subsections:**
1. *Controlled / Main Benchmarks* (Simulation → Bimodal → Tri-modal)
2. *Spatial Domain Visualization + Marker Correspondence* (maps + UMAP/t-SNE + gene/protein markers)
3. *Ablation & Sensitivity Analyses* (component removal variants + key parameter sweeps)
4. *Robustness / Scalability* (runtime, GPU/CPU memory, synthetic noise injection, HVG/k stability)
5. *Deep Biological Interpretation / Case Study* (unlabeled tissue → DEGs → dot/heatmap/violin plots → GO/KEGG enrichment → *cis*-regulatory peak-to-gene arcs or TCGA clinical survival analysis).

### 4.8 Discussion & Conclusion Writing
- **Discussion (`4` in `arise.pdf`, `6.` in `SpaFusion.pdf`, ~332–337 words):**
  - *Paragraph 1:* Restate core conceptual contribution (never repeat raw numerical tables verbatim) + explain why theory and empirics align.
  - *Paragraph 2:* Synthesize biological interpretability and cross-modality extensibility.
  - *Paragraph 3:* Contextualize against emerging literature (e.g., foundational models `scMAGCA`, `GARDEN`, `stDCL` in ARISE) + **honest limitations** (reliance on RNA anchor / conservative intersection in ARISE; limited public spatial multi-omics datasets and Transformer quadratic overhead in SpaFusion) + concrete **future work** (image-informed/cross-sample/foundation-model embeddings; efficient attention architectures).
- **Conclusion (`7.` in `SpaFusion.pdf`, ~296 words):** Summarizes the domain need, recaps the proposed architecture's central mechanisms (`high-order motif graph`, `GAE + Transformer`, `two-stage fusion`, `consensus clustering`), and ends on extensibility to `additional omics modalities`.

---

## 5. Figures, Tables, Equations, and Algorithm Conventions

### 5.1 Architecture Diagrams (`Fig. 1`)
- **ARISE (`arise.pdf` p. 3):** Modular numbered pipeline `(i) RNA-Anchored Graph Encoding` → `(ii) Inside-Out Hierarchical Fusion` → `(iii) Multi-omics Reconstruction` & `(iv) Spatial Coherence Regularization` → `(v) Application`. Clean pastel-tinted rounded boxes, vector-style spot graphs, clear arrow flow, and small mathematical annotations ($\mathcal{L}_{\text{recon}}$, $Z_{\text{feat}}$, $Z_{\text{fused}}$, $Z$).
- **SpaFusion (`SpaFusion.pdf` p. 3):** Full-width color-coded horizontal flow from molecular/spatial inputs (`Spatial transcriptome`, `Spatial coordinates`, `Spatial proteome`) → dual encoders (`GAE_f`, `Transformer`, `GAE_s`) → zoomed-in `Information Fusion` center inset showing matrix operations ($\tilde{A}_f^v, Z_I^v, L^v, S^v, H^v, \tilde{Z}^v$) with `⊕ Element add` and `⊗ Multiply` legend → `Adaptive fusion` → `Z → Q / P / Q^1 / Q^2` clustering ring.

### 5.2 Results Figure Taxonomy & Visual Styling
- **Combined Multi-Modal Figure Layout (`arise.pdf` Figs. 2, 3, 4, 7):**
  - Instead of scattering 15 small figures, `arise.pdf` packs a whole story into one full-width figure:
    - Row `(a)`: Quantitative bar/line charts with `↑` / `↓` direction arrows in panel titles and a **red downward triangle marker (`▼`)** highlighting `ARISE` as the leftmost/top bar!
    - Row `(b)`: Side-by-side spatial domain maps (`Ground Truth` first on left, followed by `ARISE` and 6 baselines in consistent column order) + UMAP panels directly underneath (`Fig. 4b`).
    - Rows `(c)–(d)`: Biological validation triplets (`GT: cluster` | `Gene: marker` | `ADT/Peak: marker`) with colorbars showing expression/accessibility range (`0.0–max`).
  - **Complex Biological Storytelling (`arise.pdf` Fig. 7):** Uses 6 distinct bioinformatics plot idioms in a single cohesive page-wide figure: `(a)` paired RNA dot-plot + ATAC Z-score heatmap across clusters 0–13, `(b)` grouped horizontal lollipop plot for `BP / CC / MF` GO & KEGG enrichment (`-log10(p.adjust)` + dot size for `Count`), `(c)` genomic ATAC ridgeline plots around TSS (`-60` to `+20 kb`), `(d)` horizontal correlation bar chart (`-59.2kb`, `+24.7kb`), `(e)` genomic peak-to-gene arc diagram (`chr6`), `(f)` boxplot/violin comparison of Pearson $r$ with `n` pair counts.
- **Separated Single-Purpose Figures (`SpaFusion.pdf` Figs. 2–11):**
  - Separates tables from plots: spatial domain grids (`Figs. 2–3`, Ground Truth with named anatomical legend + 6-method grid), clustered DEG heatmap (`Fig. 4`), dual-y/line parameter sensitivity (`Fig. 5`), 3D grouped bar surface plots (`Fig. 6` for $\lambda_1 \times \lambda_2$), grouped ablation bars (`Fig. 7`), runtime line plot (`Fig. 8`), t-SNE grids (`Fig. 9`), dot + violin plots (`Fig. 10`), and Kaplan–Meier survival curve with shaded 95% confidence bands and `Logrank p = 0.0137` (`Fig. 11`).

### 5.3 Caption Grammar Rules
- **`arise.pdf`:** `Figure 2` (`Semibold`) + space + **`Performance on simulated spatial multi-omics (RNA+ATAC) data.`** (bold or semibold/regular title phrase ending in period) + `(a) ... (b) ...` (each subpanel starts with `(a)` bold/regular, ends with a period). Note: `arise_supplementary_data.pdf` uses identical `Figure 1` / `Table 1` format without period after number.
- **`SpaFusion.pdf`:** `Fig. 2.` (`Bold`, period after number!) + **`Spatial plots of the human lymph node A1 dataset. (left) ... (right) ...`**; Tables put `Table 2` on line 1 (`Bold`) and title sentence on line 2 (`Regular`).
- **Metric Direction Arrows:** Both papers place `↑` (higher is better) and `↓` (lower is better) directly inside plot titles and table headers (`ARI ↑`, `NMI ↑`, `AMI ↑`, `SC ↑`, `DB ↓`) so readers never have to guess polarity.

### 5.4 Table Conventions (Booktabs Standard)
Both `arise_supplementary_data.pdf` and `SpaFusion.pdf` strictly follow **LaTeX `booktabs` formatting**:
- **Three horizontal rules only:** Heavy/medium top rule, thin mid-rule separating column headers from data, bottom rule. **Zero vertical column borders.**
- **Hierarchical row grouping:**
  - `SpaFusion.pdf` Tables 2–3 add a left grouping column (`Omics`: `Single-omic scRNA` / `Single-omic ADT` / `Multi-omics`) with horizontal rules separating groups.
  - `arise_supplementary_data.pdf` Table 14 uses centered bold full-width sub-header banners (`Human Lymph Node Dataset (RNA+ADT)`, `Mouse Brain Dataset (RNA+ATAC)`) spanning all columns.
- **Precision & Uncertainty:**
  - `SpaFusion.pdf` reports **3 decimal places with `± std`** (`0.319 ± 0.002`) over 10 runs, **bolds** best values, **underlines** second-best values, and includes a bottom table note: `Optimal values are highlighted in bold and second best values are underlined.`
  - `arise_supplementary_data.pdf` reports **4 decimal places** (`0.4141`) in metric tables (`2 decimal places` in runtime/memory tables) and **bolds** the best value per row/column; `ARISE (Ours)` is placed as the **first method column** (`Table 9–11, 13–14`), whereas `SpaFusion` places `SpaFusion` as the **bottom row** of each multi-omics block (`Table 2–3`) or **rightmost column** (`Table 5`).

### 5.5 Mathematical Notation & Algorithm Styling
- **Notation Style:**
  - `arise.pdf`: Matrices in **bold italic/roman uppercase** ($\mathbf{F}_{\text{RNA}}, \mathbf{A}_f, \mathbf{Z}_{\text{feat}}, \mathbf{W}_1$), losses in calligraphic ($\mathcal{L}_{\text{spatial}}, \mathcal{L}_{\text{recon}}, \mathcal{L}_{\text{total}}$), concatenation as $\oplus$, sets in calligraphic ($\mathcal{G}_f$). Equations in `arise.pdf` omit trailing punctuation; equations in `arise_supplementary_data.pdf` mix punctuated and unpunctuated lines.
  - `SpaFusion.pdf`: Matrices in standard **math italic uppercase** with superscript modality index $v \in \{1, 2\}$ and subscript type ($X^v, A_f^v, A_s^v, Z_f^v, Z_s^v, Z_g^v, \tilde{Z}^v, L^v, S^v, H^v$), Hadamard product as $\odot$, losses in italic $L_{rec_l}, L_{rec_g}, L_{aux}, L_{kl}$. **Every displayed equation in `SpaFusion.pdf` ends with a comma `,` or period `.`** matching grammatical sentence flow!
- **Algorithm Block (`SpaFusion.pdf` p. 5):** Uses top/bottom horizontal rules (`booktabs`/`algorithm2e` style), `Input:` / `Output:` header lines, numbered outer steps `1.`–`4.`, lettered inner loop `(a)`–`(e)` referencing exact equation numbers (`Eq. (23)`, `Eq. (24)`, `Eq. (25)`, `Eq. (27)`, `Eq. (26)`), and `Return: Y`.

---

## 6. Writing Style, Voice, Rhetoric, and Phrasebook

### 6.1 Voice, Pronouns, and Tense Conventions
- **First-Person Active Voice Dominates:** Neither paper hides behind stiff third-person phrasing like *"the authors propose"*. Both routinely use active `"we"`:
  - `arise.pdf`: **49 occurrences** of `we/We` (`We present...`, `We construct...`, `We evaluated...`, `We next examined...`).
  - `SpaFusion.pdf`: **62 occurrences** of `we/We` (`we introduce...`, `we propose...`, `we conducted...`).
  - `arise_supplementary_data.pdf`: **16 `we/We` + 14 `our/Our`** (`We generated...`, `our theoretical analysis`).
- **Tense Discipline:**
  - **Present Tense** for established facts, model mechanics, equations, and figure/table descriptions: `ARISE defines...`, `Eq. (1) yields...`, `Fig. 5 shows...`, `higher SC indicates...`.
  - **Past Tense** for experiments executed, preprocessing performed, and observed outcomes: `We benchmarked...`, `ARISE achieved...`, `Data were preprocessed...`, `Performance remained stable...`.
  - **Future Tense** strictly reserved for the closing Limitations/Future Work sentences (`Future work will extend...`, `we plan to explore...`).

### 6.2 Sentence Structure & Rhythm
- **`arise.pdf` (~22.2 words/sentence in Intro):** Crisp, high-density, low-hedging academic prose. Avoids filler openers (*"It is well known that..."*, *"In order to..."*). Uses **semicolons and parallel verb phrases** to pack causal chains tightly: *"these graphs can become topologically discordant, propagate spurious edges, weaken cross-modal alignment, and reduce spatial domain resolution."*
- **`SpaFusion.pdf` (~25.4 words/sentence in Intro):** Slightly more explicit signposting and transition scaffolding (*"To address this limitation,..."*, *"Based on these observations,..."*, *"These findings underscore..."*). Uses numbered inline lists `(1)... (2)... (3)... (4)...` inside prose for ablation variants and parameter breakdowns.

### 6.3 Reusable Rhetorical Phrasebook (Observed Patterns)

| Rhetorical Move | `arise.pdf` Phrasing | `SpaFusion.pdf` Phrasing |
| :--- | :--- | :--- |
| **Introducing the Model** | *"Here, we present Anchored RNA for Integrated Spatial Embedding (ARISE), an RNA expression anchored framework for..."* | *"To address these challenges, we propose SpaFusion, a novel multi-level fusion model for..."* |
| **Motivating a Design Choice** | *"To ensure that the latent representations remain faithful to the observed data, we introduce..."* | *"To overcome this limitation, we introduce a Transformer module that..."* |
| **Transitioning to Next Experiment** | *"We next examined whether the advantages observed in simulation generalize to real biological data..."* | *"Comprehensive understanding of multi-omics clustering necessitates evaluating..."* |
| **Describing Qualitative Failure Modes** | *"affected by salt-and-pepper noise, suggesting insufficient spatial constraint"*; *"tended to over-smooth the signal and merge fine inner layers"* | *"its segmentation is relatively coarse and fails to delineate fine-grained structural details"*; *"lacks spatial modeling, leading to substantial deviations"* |
| **Synthesizing Biological Findings** | *"The learned latent space therefore functions not only as a clustering substrate, but also as a biologically grounded representation..."* | *"These findings underscore the capability of SpaFusion to learn biologically meaningful latent representations that preserve..."* |
| **Explaining Minor Discrepancies** | *"This minor variance is likely because the MB dataset possesses a higher peak-feature-to-cell ratio..."* | *"Although SpaFusion introduces additional computational overhead due to..., it consistently delivers superior clustering performance, thereby justifying..."* |
| **Scientific Hedging** | *"suggestive of active enhancer engagement"*; *"indicate that ARISE yields..."*; *"support the extensibility of..."* | *"likely due to their limited utilization of..."*; *"suggests the presence of..."*; *"underscoring its potential as..."* |

---

## 7. Main Paper vs. Supplementary Data Split & Pitfalls to Avoid

### 7.1 What Goes Where? (The ARISE Split Model)
If writing for *Bioinformatics*, *Briefings in Bioinformatics*, or *Nature*-style journals with strict page/word limits, copy ARISE's clean split:
- **Keep in Main Paper:**
  1. Motivation + clear problem diagnosis
  2. Complete forward architecture + loss equations (`Eqs. 1–20`) + `Fig. 1`
  3. Theorem *takeaways* in prose (1–2 sentences linking theory to design)
  4. High-level dataset list + metric names (`3.1`)
  5. Headline simulation, bimodal, trimodal, ablation/sensitivity plots (`Figs. 2–6`) + narrative
  6. Flagship biological interpretation (`Fig. 7`: markers, pathway enrichment, *cis*-regulation)
  7. Concise Discussion + limitations + future work
- **Offload to Supplementary Data (`arise_supplementary_data.pdf`):**
  1. Formal Theorem statements + step-by-step Proofs (`Sections 1–2`)
  2. Detailed per-dataset descriptions, spot/feature dimensions, platforms (`Section 3 + Table 1`)
  3. Metric equations (`Section 4`)
  4. Reproducibility configs: baseline policy, exact hyperparameter tables, K-means seeds, Bioconductor/GENCODE/Scanpy versions (`Section 5 + Table 2`)
  5. Secondary ablations (alternative fusion orders, weighted/attention graph fusion, `k`/HVG sweeps, HVG selector comparisons), full numeric score tables behind main-text bar plots (`Tables 3–8, 12–14`), computational runtime/memory tables (`Tables 9–11`), and extra qualitative domain plots (`Figs. 1–2`).

### 7.2 Real-World Editorial Pitfalls Observed in the Corpus (Checklist to Avoid)
Inspection of the three published/compiled PDFs revealed specific minor errors that slipped past production—use these as a **pre-submission audit list**:

1. **Unfilled Template Metadata:** `arise_supplementary_data.pdf` still displays `Journal Title Here, YEAR, pp. 1–9` and `© The Author(s) YEAR.` in its running header/footer. Always update or strip template placeholders in supplementary `.tex` files before uploading.
2. **Figure/Text & Legend Mismatches:**
   - In `arise.pdf`, `Fig. 4c` shows `Cald1` (Cluster 5) and `Fig. 4d` shows `Pde10a` (Cluster 1), matching p. 6 body prose—but the **Fig. 4 caption on p. 8 accidentally swapped `(c)` and `(d)`**.
   - In `SpaFusion.pdf`, `Fig. 11` caption states `red and green curves`, whereas the rendered plot shows red and teal/blue curves.
   - In `arise.pdf`, `Fig. 2b` leftmost column header contains a typo: `Groud Truth`.
3. **Cross-Reference Naming Drift Between Main & Supplementary:**
   - `arise.pdf` references `Theorems T1–T3`, `Theorem T4`, `Supplementary Fig. 1`, and `Supplementary Tables 4 and 5`—but `arise_supplementary_data.pdf` actually labels them `Theorem 1–4`, `Figure 1`, and `Table 4–5` (without `T` or `Supplementary` prefix). **Keep prefix labels (`Theorem S1`, `Table S1`, `Fig. S1`) strictly synchronized.**
4. **Notation Collisions & Variable Typos:**
   - `arise.pdf` Eq. (13) reuses $\mathbf{W}_m^0, \mathbf{W}_m^1$ for **decoder** weights after Eq. (6) already used $\mathbf{W}_m^0, \mathbf{W}_m^1$ for **encoder** weights; Eq. (9) writes $\mathbf{Z}_m$ (lowercase $m$) right before $\{Z_m\}_{m=1}^M$.
   - `SpaFusion.pdf` Eq. (28) defines $b_i$ instead of $b_j$ in the prose below the equation, and p. 11 writes `subtypes 10(a)` without `Fig.`.
5. **Typos & Duplicated Text:**
   - `arise_supplementary_data.pdf`: `define the edge-level false positive rate (FPR) The marginal false-positive rate (FPR)` (p. 1); `6 Result` singular (p. 5); `Tables Tables 9, 10, and 11` (p. 6); `Scanpy cell rangerr` (p. 7); `23081.67` GB CPU memory anomaly in Table 11 (p. 7).
   - `SpaFusion.pdf`: `For integrate spatial multi-omics` (p. 2); `to integration spatial transcriptomics` (p. 2); `can computed` (p. 3); `which is calculates` (p. 5); `SpataiGlue` in Table 5 (p. 11).
6. **Citation-Reference List Drift:**
   - `arise.pdf` p. 1 cites `SEDR (Fu et al. 2021)`, whereas p. 12 References lists the later journal version `Xu H, Fu H, Long Y et al. ... Genome Med 2024;16:12` without a `Fu et al. 2021` entry. Always run an automated BibTeX/reference consistency check.

---

## 8. Actionable Pre-Submission Writing & Formatting Checklist

1. **Choose Your Venue Archetype Early:**
   - *Bioinformatics / Nat. Methods style:* Structured Abstract (`Motivation / Results / Availability`), `Intro → Methods → Results → Discussion`, author-year citations, tables & proofs moved to Supplementary PDF, strong downstream biology figure (`GO/KEGG + cis-regulatory` or `DEGs + survival`).
   - *Information Fusion / IEEE / Elsevier style:* Unstructured Abstract + Keywords, `Intro (+ bullet contributions) → Related works → Methods (+ Algorithm) → Experiments (+ Table 1 + metrics + baselines) → Results (+ Tables + sensitivity + ablation + complexity + noise + case study) → Discussion → Conclusion`, numeric `[1]` citations, `mean ± std` bold+underline tables.
2. **Enforce Equation Discipline:** Define every matrix dimension ($N \times d$), every superscript/subscript, and every hyperparameter immediately after its equation; never reuse weight symbol names between encoders and decoders ($\mathbf{W}_{\text{enc}}, \mathbf{W}_{\text{dec}}$).
3. **Standardize Table & Plot Polarity:** Put `↑` / `↓` on every metric header, place your own method consistently first (or last), use booktabs (3 horizontal rules, 0 vertical rules), and highlight best (**bold**) and second-best (<u>underlined</u>).
4. **Lock Reproducibility Details:** Explicitly report preprocessing cutoffs (HVGs, CLR, TF-IDF/LSI, PCA dims), seeds (`random_state`), clustering initialization (`n_init`), software versions, and hardware specs alongside runtime/memory metrics.
5. **Run a Final Cross-Check:** Verify every `Fig. X` / `Table X` / `Eq. (X)` / `Supplementary` label, subpanel letter (`(a)`–`(f)`), color description, and citation entry matches 1-to-1 across the main manuscript and supplementary data.

