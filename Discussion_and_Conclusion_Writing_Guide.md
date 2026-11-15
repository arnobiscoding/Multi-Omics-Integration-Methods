# How to Write Discussion and Conclusion Sections: A Detailed Analysis from Reference Papers

This guide analyzes the structure, rhetorical moves, stylistic conventions, and writing mechanics of the **Discussion** and **Conclusion** sections in two spatial multi-omics reference papers:

1. **ARISE** ([`arise.pdf`](papers/arise.pdf), *Bioinformatics*, 2026) — Uses a **single combined `4 Discussion` section** (no separate Conclusion), supported by [`arise_supplementary_data.pdf`](papers/arise_supplementary_data.pdf).
2. **SpaFusion** ([`SpaFusion.pdf`](papers/SpaFusion.pdf), *Information Fusion*, 2025) — Uses **separate `6. Discussions` and `7. Conclusion` sections**.

---

## 1. Two Structural Paradigms: Combined vs. Separate

Before drafting, choose the structural pattern that fits your target venue:

| Feature | **ARISE** (*Bioinformatics*) | **SpaFusion** (*Information Fusion*) |
| :--- | :--- | :--- |
| **Sectioning** | `4 Discussion` only (3 paragraphs, ~350 words) | `6. Discussions` (2 paragraphs) + `7. Conclusion` (2 paragraphs) |
| **Opening Move** | States the core mechanism & *why* it solves the problem | Summarizes the experimental scope & *what* was achieved |
| **Results Synthesis** | High-level synthesis + biological interpretation (datasets, markers, pathways) | Quantitative metric highlights (ARI/NMI), ablation/sensitivity validation, downstream clinical link |
| **Limitations** | Brief, candid, integrated into the final paragraph | Dedicated second paragraph in `Discussions`, itemized ("First... Second...") |
| **Future Work** | Tied directly to limitations and emerging field trends (foundational models) | Tied directly to each stated limitation |
| **Closing** | Ends on future directions (cross-sample, image-informed, foundational models) | Dedicated `Conclusion` that re-frames the problem, recaps the architecture, and ends on extensibility |

> **Rule of thumb:**
> - **Bioinformatics / OUP / Nature Methods-style venues** prefer a tight, interpretive **Discussion** that blends synthesis, biological meaning, limitations, and outlook without repeating the Abstract.
> - **Elsevier / IEEE / Engineering & Fusion venues** expect an explicit **Discussions** (results synthesis + limitations + future work) followed by a standalone **Conclusion** (problem restatement + method recap + broader impact/extensibility).

---

## 2. Paragraph-by-Paragraph Anatomy of the **Discussion** Section

Across both papers, an effective Discussion answers **four core questions** in a deliberate order:
1. **What did we solve and why does our design work?** (Core mechanism & intuition)
2. **What do the results mean beyond benchmark numbers?** (Biological/practical interpretation)
3. **Where does our work sit in the broader field?** (Context & complementary methods)
4. **What are the honest limitations and next steps?** (Limitations + actionable future work)

### A. Paragraph 1: Core Mechanism & Empirical/Theoretical Justification
* **Goal:** Reconnect directly to the gap introduced in Section 1 (Introduction), without copying sentences verbatim. Explain *why* the method works, not just *that* it got higher numbers.
* **How ARISE does it:**
  - Opens with a crisp thesis: *"ARISE addresses cross-modal topological inconsistency through a single transcriptome-anchored shared-edge topology..."*
  - Explains the mechanism in plain conceptual terms (using RNA similarity + spatial proximity as a common scaffold so auxiliary modalities don't need graph reconciliation).
  - Anchors the claim to **both** pillars of evidence: *"Both the theoretical analysis and the empirical results support this design..."* — specifically noting *when* it helps most (*"particularly when auxiliary modalities are sparse, noisy, or weakly structured"*).
* **How SpaFusion does it:**
  - Opens with experimental breadth: *"We conducted extensive experiments to evaluate the effectiveness of SpaFusion, benchmarking it against a broad range of methods..."*
  - Summarizes what the metrics (ARI, NMI), parameter sensitivity, and ablation studies prove.
  - Pivots to the **causal explanation**: *"The strong performance of SpaFusion can be largely attributed to its integration of high-order graph structures and the incorporation of global contextual information."*
  - Highlights generalization to unlabeled data (mouse breast cancer → marker discovery + survival analysis).

**Key takeaway:** Never turn Paragraph 1 into a table-reading exercise (e.g., avoid *"On dataset A we got 0.65 vs 0.61"*). Instead, synthesize *why* the architectural components produced the observed gains.

### B. Paragraph 2 (ARISE): Biological & Domain Interpretation
* **Goal:** Prove that better clustering/embeddings translate into meaningful domain insights.
* **How ARISE does it:**
  - Opens with: *"Beyond improved clustering accuracy, ARISE yields biologically interpretable representations across molecular layers."*
  - Gives **2–3 concrete, named examples** from the Results:
    1. *Human Lymph Node:* domains aligned with stromal/immune compartments, supported by concordant RNA + ADT markers.
    2. *Mouse Brain:* preserved coordinated expression–chromatin accessibility patterns, enabling marker programs, enrichment, and cis-regulatory inference.
  - Notes **structural extensibility**: the same anchored design + hierarchical fusion scales from bi-modal to tri-modal without redefining the topology.

**Key takeaway:** If your paper includes downstream analyses (DEGs, pathway enrichment, survival analysis, spatial domain annotation), dedicate a focused paragraph in the Discussion to *what those findings establish scientifically*.

### C. Positioning, Limitations, and Future Work (ARISE Para 3 & SpaFusion Para 2)
* **Goal:** Show scholarly maturity by situating the method among emerging paradigms and stating limitations concretely—paired immediately with how to address them.
* **Two effective patterns observed:**
  1. **Complementary Positioning (ARISE):**
     - Acknowledges fast-moving alternatives (deep learning & foundational models like scMAGCA, GARDEN, stDCL) *without* being defensive.
     - Frames them as **complementary scales**: foundational models capture global molecular representations via pre-training, whereas ARISE models localized, fine-grained spatial topology.
     - Turns this contrast into a concrete future direction: using pre-trained embeddings as initial node features in the spatial graph framework.
  2. **Explicit Limitation → Solution Pairing (SpaFusion):**
     - Uses clear signposting: *"Despite the promising results, SpaFusion exhibits several limitations. First, ... Second, ..."*
     - **Limitation 1 (Data scope):** Limited publicly available spatial multi-omics datasets constrain evaluation across diverse tissues/conditions → *Future work:* expand evaluations to emerging high-throughput datasets.
     - **Limitation 2 (Computational cost):** Transformer self-attention + multi-level fusion introduce computational overhead, affecting scalability → *Future work:* explore more computationally efficient architectural designs.
  - **ARISE's concise limitation statement:** *"We note that ARISE is currently limited by its reliance on RNA as the anchor modality and by its conservative shared-edge design. Future work will extend the framework to image-informed and cross-sample settings..."*

**Key takeaway:** Every limitation should be:
1. **Specific to your method or evaluation** (e.g., reliance on an anchor modality, $O(N^2)$ attention cost, single-slice vs. cross-sample, lack of histology image integration).
2. **Paired with a concrete next step** (not vague hand-waving like *"more experiments will be done"*).

---

## 3. Anatomy of a Standalone **Conclusion** Section (SpaFusion Model)

When your venue expects a separate **Conclusion** (like *Information Fusion*, *IEEE*, or a thesis/FYDP report), its job is **closure**, not new analysis or limitations.

SpaFusion's `7. Conclusion` follows a clean **3-move arc** across two paragraphs:

### Move 1: Macro Context & Problem Restatement (Para 1)
- **What it does:** Zooms back out to the big picture—why spatial multi-omics matters and why existing single-omics or first-order methods fall short.
- **Example structure:**
  1. *Promise of the field:* Spatial multi-omics preserves spatial context while profiling multidimensional molecular layers.
  2. *Role of the task:* Clustering is foundational for delineating cellular diversity.
  3. *Remaining gap:* Existing methods adapted from spatial transcriptomics show significant limitations on multi-omics data, motivating integrative frameworks.

### Move 2: Method Recap — The "What & How" (Para 2, first half)
- **What it does:** Summarizes the proposed pipeline in **3–4 crisp sentences**, hitting each core innovation in order:
  1. High-order cell graph (3-node motif structures) → captures intricate local interactions.
  2. Graph autoencoders + transformers → learns local and global latent representations.
  3. Two-stage information fusion + consensus clustering → integrates representations for clustering.

### Move 3: Validation Summary & Extensibility (Para 2, second half)
- **What it does:** Closes with the scope of validation and forward-looking flexibility:
  - Evaluated on three real-world spatial transcriptome–proteome datasets → superior performance + biologically meaningful insights.
  - Final sentence highlights **generalizability**: *"Notably, the proposed framework is flexible and can be readily extended to integrate additional omics modalities beyond transcriptomics and proteomics."*

> **Important distinction:** Notice what is **NOT** in SpaFusion's Conclusion:
> - No new numbers, p-values, or figure references.
> - No citations to new related work (unlike the Discussion).
> - No detailed list of limitations (those already lived in `6. Discussions`).

---

## 4. Side-by-Side Rhetorical Templates

You can adapt these sentence templates directly when drafting your own Discussion and Conclusion:

### Discussion Templates
- **Opening (Mechanism-first, ARISE style):**
  > *"[Method] addresses [core problem, e.g., cross-modal topological inconsistency] through [key architectural innovation]. By [briefly how it works], [Method] [key benefit] without [pitfall of prior work]."*
- **Opening (Evidence-first, SpaFusion style):**
  > *"We conducted extensive experiments to evaluate [Method], benchmarking it against [categories of baselines]. The results consistently demonstrate that [Method] outperforms competing approaches on [datasets], as reflected by [metrics]. Ablation and sensitivity analyses further validate the contribution of each module."*
- **Causal Attribution:**
  > *"The strong performance of [Method] can be largely attributed to [Component A] and [Component B], particularly when [challenging condition, e.g., auxiliary modalities are sparse or noisy]."*
- **Biological / Downstream Interpretation:**
  > *"Beyond improved clustering accuracy, [Method] yields biologically interpretable representations. On [Dataset], the inferred domains aligned with [known anatomy/compartments] and were supported by [markers / pathway enrichment / survival analysis]."*
- **Positioning vs. Emerging Paradigms:**
  > *"The field is witnessing rapid development of [e.g., foundational models / generative architectures]. These approaches and [Method] function at complementary scales: the former [strength of other], whereas the latter [strength of yours]. A natural avenue for future work is to combine both by [concrete integration]."*
- **Limitations & Future Work:**
  > *"Despite these promising results, [Method] exhibits several limitations. First, [architectural/data limitation, e.g., reliance on X / computational overhead of Y]. Second, [evaluation limitation, e.g., single-section scope]. In future work, we plan to [specific remedy for 1] and [specific remedy for 2]."*

### Conclusion Templates
- **Context & Gap:**
  > *"[Domain/technology] provides unprecedented capabilities to [broad goal]. However, existing [task] methods exhibit limitations when [key challenge], underscoring the need for [type of framework]."*
- **Method & Validation Recap:**
  > *"In this study, we propose [Method], a [short characterization] framework for [task]. Central to our approach is [Innovation 1], complemented by [Innovation 2] and [Innovation 3]. Comprehensive evaluations on [datasets] demonstrate that [Method] achieves superior performance and uncovers biologically meaningful insights."*
- **Closing Extensibility Statement:**
  > *"Moreover, the proposed framework is modular and can be readily extended to [additional modalities / cross-sample integration / clinical settings]."*

---

## 5. Common Pitfalls to Avoid (Checklist)

Before finalizing your Discussion and Conclusion, verify the following:

- [ ] **No raw number dumping:** Don't repeat exact ARI/NMI tables from the Results section; summarize *trends*, *mechanisms*, and *why* the method wins.
- [ ] **No copy-pasting from the Abstract or Introduction:** Rephrase the problem and contributions at a higher level of synthesis (now backed by evidence).
- [ ] **No unsubstantiated claims:** Every claim in the Discussion must trace back to a figure, table, ablation, or theoretical result shown earlier in the paper.
- [ ] **No new results in the Discussion/Conclusion:** Never introduce new datasets, metrics, or experiments here.
- [ ] **Honest, specific limitations:** Avoid generic disclaimers (*"datasets are small"* without context). Name the exact architectural assumption (e.g., RNA-anchoring, quadratic attention cost, hyperparameter sensitivity, or missing histology/cross-sample alignment) and pair it with a realistic fix.
- [ ] **Match venue structure:**
  - Use **Discussion only** (3–4 tight paragraphs: Mechanism → Biological Meaning → Positioning + Limitations/Future) for *Bioinformatics* / *Nature*-family style.
  - Use **Discussion + Conclusion** (keep limitations and literature comparisons in Discussion; keep Conclusion as a crisp 1–2 paragraph synthesis of problem, pipeline, validation, and extensibility) for *Information Fusion* / Elsevier / thesis/FYDP style.
