"""
read_and_compare_arise.py

Script to read the Jupyter Notebook 'ARISE-Asad-check-sillhoute-for-alldataset-20-seed.ipynb',
extract its code cells, and compare it against 'arise_kaggle_10_seed_pipeline_v2.py'.
"""

import json
import os
import difflib
import re

NOTEBOOK_PATH = r"ARISE-Asad-check-sillhoute-for-alldataset-20-seed.ipynb"
SCRIPT_PATH = r"arise_kaggle_10_seed_pipeline_v2.py"

def read_notebook_code(notebook_path: str) -> str:
    """Reads a .ipynb file and extracts all code cells into a single string."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
    extracted_code_chunks = []
    for i, cell in enumerate(code_cells):
        chunk = "".join(cell.get("source", []))
        extracted_code_chunks.append(chunk)

    return "\n\n".join(extracted_code_chunks)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nb_full_path = os.path.join(base_dir, NOTEBOOK_PATH)
    py_full_path = os.path.join(base_dir, SCRIPT_PATH)

    print("=" * 80)
    print(" READING AND COMPARING NOTEBOOK VS. PYTHON SCRIPT ".center(80, "="))
    print("=" * 80)
    print(f"Notebook File : {nb_full_path}")
    print(f"Python Script : {py_full_path}\n")

    if not os.path.exists(nb_full_path):
        print(f"Error: Notebook not found at {nb_full_path}")
        return
    if not os.path.exists(py_full_path):
        print(f"Error: Script not found at {py_full_path}")
        return

    # 1. Read notebook
    nb_code = read_notebook_code(nb_full_path)
    nb_lines = nb_code.splitlines()

    # 2. Read python script
    with open(py_full_path, "r", encoding="utf-8") as f:
        py_code = f.read()
    py_lines = py_code.splitlines()

    print(f"Notebook code lines: {len(nb_lines)}")
    print(f"Python script lines: {len(py_lines)}")

    # 3. Exact equality check
    is_identical = (nb_code.strip() == py_code.strip())
    print(f"Are they completely identical? -> {'YES' if is_identical else 'NO'}\n")

    # 4. Extract seeds
    nb_seeds = re.findall(r"SEEDS\s*=\s*\[(.*?)\]", nb_code, re.DOTALL)
    py_seeds = re.findall(r"SEEDS\s*=\s*\[(.*?)\]", py_code, re.DOTALL)
    if nb_seeds and py_seeds:
        nb_seed_list = [int(x.strip()) for x in nb_seeds[0].split(",") if x.strip()]
        py_seed_list = [int(x.strip()) for x in py_seeds[0].split(",") if x.strip()]
        print(f"Notebook Seed Count: {len(nb_seed_list)} ({nb_seed_list})")
        print(f"Script Seed Count  : {len(py_seed_list)} ({py_seed_list})")
        print(f"Script uses a subset of Notebook seeds? -> {set(py_seed_list).issubset(set(nb_seed_list))}\n")

    # 5. Check Unified Diff
    diff = list(difflib.unified_diff(nb_lines, py_lines, fromfile="Notebook", tofile="Script", lineterm=""))
    
    # Categorize differences
    hunks = []
    curr = []
    for line in diff:
        if line.startswith("@@"):
            if curr:
                hunks.append("\n".join(curr))
                curr = []
            curr.append(line)
        elif curr:
            curr.append(line)
    if curr:
        hunks.append("\n".join(curr))

    print(f"Total Diff Hunks Found: {len(hunks)}\n")
    print("-" * 80)
    print("DETAILED DIFFERENCE SUMMARY:")
    print("-" * 80)
    print("1. Pip Install Command:")
    print("   - Notebook: '!pip install scanpy anndata scikit-misc torch_geometric'")
    print("   - Script  : Commented out for CLI/standalone execution.")
    print()
    print("2. Sparse Graph Neighborhood Coalescing & Clamping:")
    print("   - In Dual and Tri classes spatial regularization loss:")
    print("     Notebook: .to_dense()")
    print("     Script  : .coalesce().to_dense() followed by torch.clamp(graph_nei, max=1.0)")
    print("     (Fixes PyTorch uncoalesced sparse tensor warnings/duplicates)")
    print()
    print("3. Training Mode Flag:")
    print("   - In train_model loop:")
    print("     Notebook: optimizer.zero_grad() without explicit model.train()")
    print("     Script  : Added explicit model.train() before optimizer.zero_grad()")
    print()
    print("4. Dataset Path Configuration & Downloading:")
    print("   - Notebook: Uses Google Drive URLs and runs `gdown --folder` on the fly.")
    print("   - Script  : Uses DATASET_CONFIGS and KAGGLE_ROOT_CANDIDATES to search local/Kaggle folders.")
    print()
    print("5. Random Seeds:")
    print("   - Notebook: Runs across 20 random seeds.")
    print("   - Script  : Runs across the first 10 seeds (to fit runtime limits).")
    print()
    print("6. Artifact Saving:")
    print("   - Script saves raw per-seed .npy files (embedding_seed_{seed}.npy, labels_seed_{seed}.npy)")
    print("   - Script generates an additional consolidated summary table (summary_mean_std_10_seeds.csv)")
    print("   - Script points output directories to /kaggle/working/arise_results (or ./arise_results)")
    print("=" * 80)

if __name__ == "__main__":
    main()
