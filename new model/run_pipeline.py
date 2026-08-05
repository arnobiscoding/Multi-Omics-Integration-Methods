"""
Pipeline Execution & Synthetic Benchmark Runner
================================================

CLI entry point to execute the Dual-Encoder VGAT PoE DEC pipeline.
Supports running on synthetic spatial multi-omics benchmark datasets or real AnnData (.h5ad) datasets.

Usage:
  python -m "new model.run_pipeline" --synthetic --num_spots 500 --num_clusters 7
"""

import os
import sys
import argparse
import numpy as np
import torch

try:
    import scanpy as sc
    import anndata as ad
except Exception:
    sc = None
    ad = None

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Relative imports when executed as module or script
try:
    from .preprocessing import prepare_spatial_multiomics_data
    from .model import VGAT_PoE_DEC
    from .trainer import SpatialOmicsTrainer
    from .metrics import compute_silhouette, compute_morans_i
except ImportError:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(curr_dir)
    if curr_dir not in sys.path:
        sys.path.insert(0, curr_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    from preprocessing import prepare_spatial_multiomics_data
    from model import VGAT_PoE_DEC
    from trainer import SpatialOmicsTrainer
    from metrics import compute_silhouette, compute_morans_i


def generate_synthetic_data(
    num_spots: int = 500,
    f1_rna: int = 1000,
    f2_adt: int = 50,
    num_clusters: int = 7,
    seed: int = 42
):
    """
    Generate synthetic spatial multi-omics dataset for verification & unit testing.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Synthetic 2D spatial spot coordinates (N, 2)
    coords = np.random.uniform(0, 100, size=(num_spots, 2))

    # Synthetic ground-truth clusters
    labels = np.random.randint(0, num_clusters, size=num_spots)

    # RNA count matrix (Poisson / Negative Binomial distributed counts with dropouts)
    rna_means = np.random.uniform(1.0, 50.0, size=(num_clusters, f1_rna))
    raw_counts_rna = np.zeros((num_spots, f1_rna), dtype=np.float32)
    for i in range(num_spots):
        c = labels[i]
        raw_counts_rna[i] = np.random.poisson(rna_means[c])
        # Add 30% zero-inflation dropouts
        dropout_mask = np.random.binomial(1, 0.3, size=f1_rna)
        raw_counts_rna[i, dropout_mask == 1] = 0.0

    # Normalized RNA features
    x_rna = np.log1p(raw_counts_rna)

    # ADT continuous protein features
    adt_means = np.random.uniform(-1.0, 3.0, size=(num_clusters, f2_adt))
    x_adt = np.zeros((num_spots, f2_adt), dtype=np.float32)
    for i in range(num_spots):
        c = labels[i]
        x_adt[i] = adt_means[c] + np.random.normal(0, 0.5, size=f2_adt)

    return x_rna, x_adt, raw_counts_rna, coords, labels


def run_pipeline(
    x_rna: np.ndarray,
    x_adt: np.ndarray,
    coords: np.ndarray,
    num_clusters: int,
    raw_counts_rna: np.ndarray = None,
    ground_truth_labels: np.ndarray = None,
    k_neighbors: int = 6,
    warmup_epochs: int = 20,
    finetune_epochs: int = 30,
    device: str = "cpu"
):
    """
    Runs end-to-end preprocessing, model instantiation, multi-phase training, and metric evaluation.
    """
    print(f"[*] Preprocessing features: RNA ({x_rna.shape}), ADT ({x_adt.shape}), Coords ({coords.shape})...")
    data_dict = prepare_spatial_multiomics_data(
        x_rna=x_rna,
        x_adt=x_adt,
        coords=coords,
        raw_counts_rna=raw_counts_rna,
        k_neighbors=k_neighbors,
        device=device
    )

    in_dim_rna = data_dict["f_rna"]
    in_dim_adt = data_dict["f_adt"]

    print(f"[*] Instantiating VGAT_PoE_DEC model (RNA dim={in_dim_rna}, ADT dim={in_dim_adt}, Clusters={num_clusters})...")
    model = VGAT_PoE_DEC(
        in_dim_rna=in_dim_rna,
        in_dim_adt=in_dim_adt,
        num_clusters=num_clusters,
        hidden_dim=128,
        latent_dim=32,
        proj_dim=64,
        heads=4
    )

    trainer = SpatialOmicsTrainer(
        model=model,
        learning_rate=1e-3,
        warmup_epochs=warmup_epochs,
        finetune_epochs=finetune_epochs,
        p_update_interval=5,
        early_stop_eval_interval=5,
        subsample_ratio=0.2 if x_rna.shape[0] < 1000 else 0.05,
        device=device
    )

    print("[*] Starting multi-phase training protocol...")
    results = trainer.fit(data_dict)

    z_f = results["z_f"]
    predictions = results["predictions"]

    sil = compute_silhouette(z_f, predictions)
    moran = compute_morans_i(z_f, coords)

    print("\n==========================================================================")
    print(" FINAL EVALUATION SUMMARY")
    print("==========================================================================")
    print(f"  - Peak Composite Metric : {results['best_composite_score']:.4f}")
    print(f"  - Latent Silhouette     : {sil:.4f}")
    print(f"  - Moran's I Coherence  : {moran:.4f}")

    if ground_truth_labels is not None:
        ari = adjusted_rand_score(ground_truth_labels, predictions)
        nmi = normalized_mutual_info_score(ground_truth_labels, predictions)
        print(f"  - Ground-Truth ARI      : {ari:.4f}")
        print(f"  - Ground-Truth NMI      : {nmi:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Spatial Multi-Omics VGAT PoE DEC Pipeline Runner")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic benchmark dataset")
    parser.add_argument("--num_spots", type=int, default=500, help="Number of spots for synthetic run")
    parser.add_argument("--num_clusters", type=int, default=7, help="Number of clusters K")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Phase 1 Warm-up epochs")
    parser.add_argument("--finetune_epochs", type=int, default=30, help="Phase 3 Fine-tuning epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if args.synthetic or True:  # Default to synthetic verification run if no args
        print(f"--- Running Synthetic Pipeline Test on {args.device} ---")
        x_rna, x_adt, raw_counts, coords, labels = generate_synthetic_data(
            num_spots=args.num_spots,
            num_clusters=args.num_clusters
        )
        run_pipeline(
            x_rna=x_rna,
            x_adt=x_adt,
            coords=coords,
            num_clusters=args.num_clusters,
            raw_counts_rna=raw_counts,
            ground_truth_labels=labels,
            warmup_epochs=args.warmup_epochs,
            finetune_epochs=args.finetune_epochs,
            device=args.device
        )


if __name__ == "__main__":
    main()
