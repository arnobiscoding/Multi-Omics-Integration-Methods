#!/usr/bin/env python3
"""
generate_comparison_assets.py
Processes benchmark results from results/arise_results and results/scgpt_arise_results,
generates aggregated comparison JSON and data.js for the web dashboard,
and copies image assets (boxplots, UMAPs) into web_dashboard/assets/.
"""

import os
import shutil
import json
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ARISE_DIR = os.path.join(RESULTS_DIR, "arise_results")
SCGPT_DIR = os.path.join(RESULTS_DIR, "scgpt_arise_results")
DASHBOARD_DIR = os.path.join(BASE_DIR, "web_dashboard")
ASSETS_DIR = os.path.join(DASHBOARD_DIR, "assets")

os.makedirs(ASSETS_DIR, exist_ok=True)

# 1. Load CSVs
arise_csv = os.path.join(ARISE_DIR, "metrics_all_datasets_10_seeds.csv")
scgpt_csv = os.path.join(SCGPT_DIR, "scgpt_arise_benchmark_results.csv")

df_arise = pd.read_csv(arise_csv)
df_scgpt = pd.read_csv(scgpt_csv)

print(f"Loaded ARISE records: {len(df_arise)}")
print(f"Loaded scGPT-ARISE records: {len(df_scgpt)}")

# Merge on dataset and seed
df_merged = pd.merge(df_arise, df_scgpt, on=["dataset", "seed"], suffixes=("_arise", "_scgpt"))
print(f"Merged paired records: {len(df_merged)}")

datasets = sorted(df_merged["dataset"].unique().tolist())
seeds = sorted(df_merged["seed"].unique().tolist())
metrics = ["ARI", "NMI", "AMI", "Homogeneity", "V-measure", "Silhouette"]

# Metadata for datasets
DATASET_INFO = {
    "10x_human_lymph_node_A1": {
        "title": "10x Human Lymph Node (Section A1)",
        "organism": "Human",
        "tissue": "Lymph Node",
        "modalities": "Spatial RNA + Protein (ADT)",
        "description": "Human lymph node tissue section profiling gene expression alongside 17 cell-surface protein markers. Captures germinal centers, B cell follicles, and T cell zones."
    },
    "10x_human_lymph_node_D1": {
        "title": "10x Human Lymph Node (Section D1)",
        "organism": "Human",
        "tissue": "Lymph Node",
        "modalities": "Spatial RNA + Protein (ADT)",
        "description": "Replicate human lymph node tissue section D1. Useful for cross-slice biological generalization and clustering stability assessment."
    },
    "Mouse_Brain_E11_S1": {
        "title": "Mouse Embryonic Brain E11 (Section 1)",
        "organism": "Mouse",
        "tissue": "Embryonic Brain E11.5",
        "modalities": "Spatial RNA + Chromatin Accessibility (ATAC)",
        "description": "Early embryonic mouse brain developmental stage. High neuroepithelial dynamism; scGPT foundation representations yield +8.57% ARI gain."
    },
    "Mouse_Brain_E13_S1": {
        "title": "Mouse Embryonic Brain E13 (Section 1)",
        "organism": "Mouse",
        "tissue": "Embryonic Brain E13.5",
        "modalities": "Spatial RNA + Chromatin Accessibility (ATAC)",
        "description": "Mid-gestation mouse embryonic brain section with extensive neurogenesis and regional differentiation across cerebral vesicles."
    },
    "Mouse_Brain_E15_S1": {
        "title": "Mouse Embryonic Brain E15 (Section 1)",
        "organism": "Mouse",
        "tissue": "Embryonic Brain E15.5",
        "modalities": "Spatial RNA + Chromatin Accessibility (ATAC)",
        "description": "Late-embryonic mouse brain. Distinct cortical layers, striatum, and thalamic regions; both models achieve highest overall ARI (~0.587)."
    },
    "Mouse_Brain_E18_S1": {
        "title": "Mouse Embryonic Brain E18 (Section 1)",
        "organism": "Mouse",
        "tissue": "Embryonic Brain E18.5",
        "modalities": "Spatial RNA + Chromatin Accessibility (ATAC)",
        "description": "Perinatal mouse brain stage shortly prior to birth, showing mature structural demarcation and highest homogeneity scores (>0.62)."
    }
}

# 2. Copy image assets
image_manifest = {}

# Global boxplot
global_boxplot_src = os.path.join(ARISE_DIR, "global_boxplot_comparison_10_seeds.png")
global_boxplot_dest = os.path.join(ASSETS_DIR, "global_boxplot_comparison.png")
if os.path.exists(global_boxplot_src):
    shutil.copy2(global_boxplot_src, global_boxplot_dest)
    image_manifest["global_boxplot"] = "assets/global_boxplot_comparison.png"

# Per dataset assets
for ds in datasets:
    ds_dir = os.path.join(ARISE_DIR, ds)
    ds_assets = {}
    
    # boxplot
    bp_src = os.path.join(ds_dir, "boxplot_metrics.png")
    if os.path.exists(bp_src):
        bp_dest = os.path.join(ASSETS_DIR, f"{ds}_boxplot.png")
        shutil.copy2(bp_src, bp_dest)
        ds_assets["boxplot"] = f"assets/{ds}_boxplot.png"
        
    # UMAP plots
    plots_dir = os.path.join(ds_dir, "plots")
    umaps = {}
    if os.path.exists(plots_dir):
        for seed in seeds:
            umap_src = os.path.join(plots_dir, f"umap_seed_{seed}.png")
            if os.path.exists(umap_src):
                umap_dest = os.path.join(ASSETS_DIR, f"{ds}_umap_seed_{seed}.png")
                shutil.copy2(umap_src, umap_dest)
                umaps[str(seed)] = f"assets/{ds}_umap_seed_{seed}.png"
    ds_assets["umaps"] = umaps
    image_manifest[ds] = ds_assets

print("Copied image assets successfully.")

# 3. Compute Per-Dataset Summaries
dataset_summaries = {}
for ds in datasets:
    sub = df_merged[df_merged["dataset"] == ds]
    metrics_summary = {}
    
    for m in metrics:
        a_vals = sub[f"{m}_arise"].values
        s_vals = sub[f"{m}_scgpt"].values
        
        a_mean, a_std = float(np.mean(a_vals)), float(np.std(a_vals, ddof=1)) if len(a_vals) > 1 else 0.0
        s_mean, s_std = float(np.mean(s_vals)), float(np.std(s_vals, ddof=1)) if len(s_vals) > 1 else 0.0
        
        diff = s_mean - a_mean
        pct_change = (diff / a_mean * 100) if a_mean != 0 else 0.0
        
        metrics_summary[m] = {
            "arise_mean": round(a_mean, 4),
            "arise_std": round(a_std, 4),
            "scgpt_mean": round(s_mean, 4),
            "scgpt_std": round(s_std, 4),
            "diff": round(diff, 4),
            "pct_change": round(pct_change, 2),
            "winner": "scGPT-ARISE" if diff > 0.0001 else ("ARISE" if diff < -0.0001 else "Tied")
        }
        
    # scGPT unique metrics (CHI, DBI)
    chi_vals = sub["CHI"].values
    dbi_vals = sub["DBI"].values
    
    scgpt_unsupervised = {
        "CHI_mean": round(float(np.mean(chi_vals)), 2),
        "CHI_std": round(float(np.std(chi_vals, ddof=1)) if len(chi_vals) > 1 else 0.0, 2),
        "DBI_mean": round(float(np.mean(dbi_vals)), 2),
        "DBI_std": round(float(np.std(dbi_vals, ddof=1)) if len(dbi_vals) > 1 else 0.0, 2)
    }
    
    dataset_summaries[ds] = {
        "info": DATASET_INFO.get(ds, {}),
        "metrics": metrics_summary,
        "scgpt_unsupervised": scgpt_unsupervised,
        "assets": image_manifest.get(ds, {})
    }

# 4. Global Overall Aggregations
overall_summary = {}
for m in metrics:
    a_all = df_merged[f"{m}_arise"].values
    s_all = df_merged[f"{m}_scgpt"].values
    
    a_mean, a_std = float(np.mean(a_all)), float(np.std(a_all, ddof=1))
    s_mean, s_std = float(np.mean(s_all)), float(np.std(s_all, ddof=1))
    diff = s_mean - a_mean
    pct_change = (diff / a_mean * 100) if a_mean != 0 else 0.0
    
    overall_summary[m] = {
        "arise_mean": round(a_mean, 4),
        "arise_std": round(a_std, 4),
        "scgpt_mean": round(s_mean, 4),
        "scgpt_std": round(s_std, 4),
        "diff": round(diff, 4),
        "pct_change": round(pct_change, 2),
        "winner": "scGPT-ARISE" if diff > 0 else "ARISE"
    }

# 5. Seed-level raw records list
seed_records = []
for _, row in df_merged.iterrows():
    rec = {
        "dataset": row["dataset"],
        "dataset_title": DATASET_INFO.get(row["dataset"], {}).get("title", row["dataset"]),
        "seed": int(row["seed"]),
        "metrics": {}
    }
    for m in metrics:
        a_val = float(row[f"{m}_arise"])
        s_val = float(row[f"{m}_scgpt"])
        rec["metrics"][m] = {
            "arise": round(a_val, 4),
            "scgpt": round(s_val, 4),
            "diff": round(s_val - a_val, 4),
            "pct_change": round(((s_val - a_val) / a_val * 100) if a_val != 0 else 0.0, 2)
        }
    rec["scgpt_CHI"] = round(float(row["CHI"]), 2)
    rec["scgpt_DBI"] = round(float(row["DBI"]), 2)
    seed_records.append(rec)

# Complete dataset object
dashboard_data = {
    "datasets": datasets,
    "seeds": seeds,
    "metrics": metrics,
    "dataset_info": DATASET_INFO,
    "dataset_summaries": dataset_summaries,
    "overall_summary": overall_summary,
    "seed_records": seed_records,
    "image_manifest": image_manifest
}

# Write to JSON and JS
json_path = os.path.join(DASHBOARD_DIR, "data.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(dashboard_data, f, indent=2)

js_path = os.path.join(DASHBOARD_DIR, "data.js")
with open(js_path, "w", encoding="utf-8") as f:
    f.write("// Auto-generated benchmark data for ARISE vs scGPT-ARISE Dashboard\n")
    f.write("window.BENCHMARK_DATA = ")
    json.dump(dashboard_data, f, indent=2)
    f.write(";\n")

print(f"Generated {json_path} and {js_path} successfully!")
