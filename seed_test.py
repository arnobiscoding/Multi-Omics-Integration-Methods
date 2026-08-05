#!/usr/bin/env python3
"""
seed_test.py

Loads a dataset using utilities from scgptpipeline.py, generates scGPT embeddings on RNA
with 3 random seeds, and checks whether the generated embeddings are all equal.
"""

import sys
import os
import argparse
import numpy as np
import torch

from scgptpipeline import (
    ALL_DATASETS_CONFIG,
    ENV_MODE,
    fix_seed,
    load_dataset_data,
    get_scgpt_embeddings,
)


def test_scgpt_seed_reproducibility(dataset_name: str = "mouse-brain-e11-s1", seeds: tuple = (42, 0, 123)):
    """
    Loads specified dataset, generates scGPT embeddings for 3 random seeds,
    and checks if the resulting embeddings are identical across all seeds.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using execution device: {device}")

    if dataset_name not in ALL_DATASETS_CONFIG:
        default_ds = list(ALL_DATASETS_CONFIG.keys())[0]
        print(f"Warning: Dataset '{dataset_name}' not found in configuration. Defaulting to '{default_ds}'.")
        dataset_name = default_ds

    cfg = ALL_DATASETS_CONFIG[dataset_name]
    print(f"\n=======================================================")
    print(f"LOADING DATASET: {dataset_name}")
    print(f"=======================================================")
    
    # Load dataset RNA & Modality 2 AnnData objects
    adata_rna, _ = load_dataset_data(dataset_name, cfg, ENV_MODE)

    embeddings = []
    print(f"\n=======================================================")
    print(f"GENERATING scGPT EMBEDDINGS ACROSS SEEDS: {list(seeds)}")
    print(f"=======================================================")
    
    for idx, seed in enumerate(seeds):
        print(f"\n--- [Run {idx + 1}/3] Seed: {seed} ---")
        fix_seed(seed)
        # Use copy of adata_rna to ensure clean input state for each seed
        emb = get_scgpt_embeddings(adata_rna.copy(), device)
        embeddings.append(emb)
        print(f"Seed {seed} embedding generated with shape: {emb.shape}")

    # Evaluate pairwise equality across all 3 seed embeddings
    emb1, emb2, emb3 = embeddings[0], embeddings[1], embeddings[2]

    eq_1_2 = np.array_equal(emb1, emb2)
    eq_2_3 = np.array_equal(emb2, emb3)
    eq_1_3 = np.array_equal(emb1, emb3)

    all_equal = eq_1_2 and eq_2_3 and eq_1_3

    diff_1_2 = float(np.max(np.abs(emb1 - emb2)))
    diff_2_3 = float(np.max(np.abs(emb2 - emb3)))
    diff_1_3 = float(np.max(np.abs(emb1 - emb3)))

    print("\n=======================================================")
    print("SEED REPRODUCIBILITY TEST SUMMARY FOR scGPT EMBEDDINGS")
    print("=======================================================")
    print(f"Dataset Tested : {dataset_name}")
    print(f"Seeds Used     : {list(seeds)}")
    print(f"Embedding Shape: {emb1.shape}")
    print("-------------------------------------------------------")
    print(f"Seed {seeds[0]} vs Seed {seeds[1]} | Equal: {eq_1_2} | Max Abs Diff: {diff_1_2:.6e}")
    print(f"Seed {seeds[1]} vs Seed {seeds[2]} | Equal: {eq_2_3} | Max Abs Diff: {diff_2_3:.6e}")
    print(f"Seed {seeds[0]} vs Seed {seeds[2]} | Equal: {eq_1_3} | Max Abs Diff: {diff_1_3:.6e}")
    print("-------------------------------------------------------")
    
    if all_equal:
        print("STATUS: SUCCESS - All 3 embeddings are EXACTLY EQUAL across random seeds.")
    else:
        print("STATUS: EMBEDDINGS DIFFER - Embeddings are NOT equal across random seeds.")
    print("=======================================================\n")

    return all_equal


def main():
    parser = argparse.ArgumentParser(description="Test scGPT RNA embedding equality across 3 random seeds.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mouse-brain-e11-s1",
        choices=list(ALL_DATASETS_CONFIG.keys()),
        help="Dataset key from ALL_DATASETS_CONFIG to test."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs=3,
        default=[42, 0, 123],
        help="Three integer random seeds to test (default: 42 0 123)."
    )
    args = parser.parse_args()

    test_scgpt_seed_reproducibility(dataset_name=args.dataset, seeds=tuple(args.seeds))


if __name__ == "__main__":
    main()
