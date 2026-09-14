# Implementation Prompt: Micro-Level Topological Diagnostic Framework for Spatial Multi-Omics

## Objective

Implement a Python function/module named `topological_diagnostic` that performs a **spot-by-spot** audit of a spatial transcriptomics clustering result. For every spot, the function must compute four latent-space vectors, three cosine-similarity scores, and route the spot into exactly one of seven diagnostic cases (Case 1–Case 7), as defined below. The output should be a single annotated DataFrame plus summary statistics, ready for downstream plotting (e.g., spatial scatter colored by case).

This must be built as a **post-processing step** — it runs after a spatial GNN model has produced embeddings and after a clustering algorithm has produced labels. It does not retrain or modify the model.

---

## Inputs

Implement the function with this signature:

```python
def topological_diagnostic(
    Z: np.ndarray,              # (N, D) latent embeddings
    A: sparse.spmatrix,         # (N, N) physical spatial adjacency matrix (unpruned, binary or weighted)
    L_pred: np.ndarray,         # (N,) predicted cluster labels (e.g., from Leiden)
    L_true: np.ndarray,         # (N,) ground-truth domain labels (manual pathology annotation)
    tau_sim: float = 0.85,      # similarity threshold
    spot_ids: np.ndarray | None = None,  # optional (N,) identifiers/barcodes for output indexing
) -> pd.DataFrame:
```

Assume:
- `Z` may need row-wise L2 normalization before similarity computation — do NOT assume it's pre-normalized.
- `A` is a `scipy.sparse` matrix (CSR preferred) and may contain zero-degree rows (isolated spots).
- `L_pred` and `L_true` may be strings, ints, or categorical — normalize both to a consistent hashable type internally.
- `L_true` may contain NaN/missing values for spots without pathology annotation — these must be excluded from centroid computation for their own group but still receive a diagnostic row (flag as `"Unannotated"` case, separate from Cases 1–7).

---

## Part 1 — Vector Computation (vectorized, no per-spot Python loops)

1. **Normalize embeddings**: `Z_norm = Z / ||Z||_2` per row (handle zero-norm rows safely — clip norm to a small epsilon, e.g. 1e-12).

2. **Neighborhood mean** ($\mu_{\mathcal{N}}$):
   - Compute via sparse matrix multiplication: `A @ Z`, then divide each row by its row-sum of `A` (node degree).
   - **Edge case**: spots with degree 0 (no physical neighbors) — set their `mu_N` to NaN and their similarity `S_N` to NaN (not 0), and route them to a dedicated `"Isolated Spot"` flag rather than forcing them into Cases 1–7, since the neighborhood comparison is undefined.

3. **Predicted cluster centroid** ($\mu_{C}$):
   - Group rows of `Z` by `L_pred`, compute per-group mean, and broadcast back to an `(N, D)` matrix via a label-to-index mapping (e.g., `pd.factorize` + `np.add.at` or `groupby` on a DataFrame of embeddings) — avoid `O(N*K)` loops.

4. **Ground-truth centroid** ($\mu_{G}$):
   - Same as above, using `L_true`, excluding NaN-labeled spots from centroid calculation.
   - **Edge case**: singleton ground-truth groups (only 1 spot in that domain) — the centroid equals the spot itself, so `S(Z_i, mu_G_i) = 1.0` trivially. Document this in code comments; do not treat it as an error.

5. **Cosine similarities**:
   - Compute `S_G`, `S_C`, `S_N` as row-wise cosine similarities between `Z_norm` and each of `mu_G`, `mu_C`, `mu_N` (which should also be normalized before the dot product, since they are averages and not unit-length).
   - Vectorize with `np.einsum('ij,ij->i', ...)` after normalizing both operands — do not use a Python loop or `sklearn.metrics.pairwise.cosine_similarity` on the full N×N matrix (memory-inefficient; only need the diagonal).

---

## Part 2 — Case Assignment Logic

**Important implementation note not fully specified in the source spec**: the seven case conditions are **not mutually exclusive** as literally written (e.g., a spot satisfying Case 5's condition `S_G ≥ τ` could also incidentally satisfy Case 3's `S_N > S_G` if `S_N` is even higher but still below `τ`... actually re-check — but similar ambiguities exist, e.g. Case 4 and Case 6 both require `L_pred ≠ L_true`, low `S_G`, low `S_N`, differing only on `S_C`). To make this deterministic, **evaluate cases in strict priority order** and assign the first match, short-circuiting on Success spots (`L_pred == L_true`) vs. Failure spots (`L_pred != L_true`) first:

**Branch A — Success spots (`L_pred == L_true`):**
1. Case 1 (Homogeneous Core): `S_G ≥ τ` AND `S_N ≥ τ`
2. Case 2 (Boundary Triumph): `S_G ≥ τ` AND `S_N < τ`
3. Case 7 (Lucky Guess): `S_G < τ` AND `S_N < τ`
4. **Residual**: `S_G < τ` AND `S_N ≥ τ` on a correct prediction — not covered by the spec. Tag as `"Case 2b: Neighbor-Assisted Correct"` (spot agrees with neighbors but not ground-truth centroid, yet still landed correctly) and log a warning count so the user can inspect these separately.

**Branch B — Failure spots (`L_pred != L_true`):**
1. Case 5 (Resolution Fracture): `S_G ≥ τ` — check this **first** among failure cases, since a high `S_G` on a wrong label is the most specific/informative signal (over-clustering) regardless of `S_C`/`S_N` values.
2. Case 3 (Oversmoothed Victim): `S_G < τ` AND `S_N > S_G`
3. Case 6 (Feature Betrayal): `S_G < τ` AND `S_N ≤ S_G` AND `S_C ≥ τ`
4. Case 4 (Feature Orphan): `S_G < τ` AND `S_N ≤ S_G` AND `S_C < τ`

Verify with assertions/unit tests that this ordering produces a **complete partition** (every spot gets exactly one label, no spot falls through). Add a final `else: "Unclassified"` bucket purely as a safety net that should log an error if ever populated (indicates a logic gap).

Implement this as a vectorized `np.select()` call with an ordered list of boolean masks and corresponding case labels — not `df.apply(row-wise-function)`.

---

## Part 3 — Output Structure

Return a DataFrame with one row per spot and these columns:

| Column | Type | Description |
|---|---|---|
| `spot_id` | str/int | From `spot_ids` or default `RangeIndex` |
| `L_pred` | original type | predicted label |
| `L_true` | original type | ground-truth label (or NaN) |
| `S_G`, `S_C`, `S_N` | float | the three similarity scores |
| `degree` | int | number of physical neighbors (for isolated-spot flagging) |
| `case` | str (categorical) | one of: `Case 1`…`Case 7`, `Case 2b`, `Isolated Spot`, `Unannotated`, `Unclassified` |

Also return (or log) a **summary table**: count and percentage of spots per case, and per-sample breakdown if a `sample_id` column is available in a future extension (note this as a TODO, not required now).

---

## Part 4 — Validation & Testing Requirements

Write `pytest` unit tests covering:
1. A synthetic toy graph (e.g., 20 spots, 2 clear domains, one boundary spot, one isolated spot, one NaN-labeled spot) with hand-computed expected similarities and expected case labels.
2. Degenerate cases: single-spot cluster, single-spot ground-truth domain, empty adjacency row, all-identical embeddings (should not produce NaN/inf from division by zero in cosine similarity).
3. Confirm the case assignment is a complete, mutually exclusive partition (`case` column has no nulls, every spot classified).
4. Confirm vectorized neighborhood-mean computation matches a slow reference loop implementation on the toy graph (regression test against naive per-node Python loop).

---

## Part 5 — Suggested Downstream Visualization (not required in this function, but note as follow-up)

- A spatial scatter plot (x, y coordinates) colored by `case`, so a user can see Case 3 (oversmoothed) spots clustering along tissue boundaries versus Case 6 (feature betrayal) spots scattered randomly — this is the diagnostic payoff of the whole framework.
- A stacked bar chart of case proportions per predicted cluster, to spot which clusters are most affected by which failure mode.

---

## Deliverable

A single Python module (`topological_diagnostic.py`) with:
- The `topological_diagnostic()` function as specified.
- Docstrings with the LaTeX-equivalent formulas as comments for each vector/similarity.
- No use of deprecated pandas `.append()`; use `pd.concat` or vectorized assignment throughout.
- Type hints on all functions.
- A `if __name__ == "__main__":` block with a minimal synthetic example demonstrating all seven cases plus the edge-case flags firing correctly.
