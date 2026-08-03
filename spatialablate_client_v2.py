"""
SpatialAblate Benchmark Leaderboard API Client & CSV Importer (v2)
===================================================================

This module provides a Python client and CLI interface for programmatically uploading
spatial multi-omics model benchmark evaluation results to the SpatialAblate platform.

Features in v2:
---------------
- Multi-metric composite evaluation across all datasets (ARI, NMI, Silhouette, AMI, CHI, DBI, Homogeneity, V-measure)
- Automatic resolution of dataset cluster count (no_cluster):
    * a1  -> 10
    * d1  -> 11
    * e11 -> 8
    * e13 -> 12
    * e15 -> 12
    * e18 -> 14
- Identifies the BEST clustering algorithm based on composite performance across all datasets & metrics
- Submits the top-performing algorithm first as a Model Submission (/api/models/upload-csv)
- Uses the resulting base model name as reference to submit all remaining algorithm variants
  as Ablation Submissions (/api/ablation/upload-csv)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import requests
from typing import Dict, Any, Optional, List, Tuple


CLUSTER_COUNT_MAP: Dict[str, int] = {
    "a1": 10,
    "d1": 11,
    "e11": 8,
    "e13": 12,
    "e15": 12,
    "e18": 14,
}

HIGHER_IS_BETTER_METRICS = ["ARI", "NMI", "Silhouette", "AMI", "CHI", "Homogeneity", "V-measure"]
LOWER_IS_BETTER_METRICS = ["DBI"]
ALL_KNOWN_METRICS = HIGHER_IS_BETTER_METRICS + LOWER_IS_BETTER_METRICS


def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    """Parse a .env file into a dictionary of key-value pairs."""
    env_vars = {}
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    env_vars[k.strip().lower()] = v.strip().strip("'\"")
        except Exception as e:
            print(f"Warning: Failed to read .env file at {env_path}: {e}")
    return env_vars


def resolve_description(description_input: Optional[str]) -> Optional[str]:
    """
    Resolves description text. If input is a path to an existing .md or text file,
    reads and returns the file content. Otherwise returns the raw text string.
    """
    if not description_input:
        return None

    cleaned_input = str(description_input).strip().strip("'\"")
    if not cleaned_input:
        return None

    # Check if cleaned_input is a path to an existing file
    if os.path.isfile(cleaned_input):
        try:
            with open(cleaned_input, "r", encoding="utf-8") as f:
                content = f.read().strip()
                print(f"[*] Loaded description from file '{cleaned_input}' ({len(content)} chars)")
                return content
        except Exception as e:
            print(f"Warning: Failed to read description file '{cleaned_input}': {e}. Using raw input text.")
            return cleaned_input

    return cleaned_input


def get_cluster_count(dataset_name: str) -> int:
    """
    Derive cluster count K based on dataset section name matching rules:
    - a1  -> 10
    - d1  -> 11
    - e11 -> 8
    - e13 -> 12
    - e15 -> 12
    - e18 -> 14
    """
    name_lower = str(dataset_name).lower()
    keys_in_order = ["e11", "e13", "e15", "e18", "a1", "d1"]
    for key in keys_in_order:
        if key in name_lower:
            return CLUSTER_COUNT_MAP[key]
    # Default fallback if unknown section string
    print(f"Warning: Could not infer cluster count for dataset '{dataset_name}'. Defaulting to 10.")
    return 10


def find_column_case_insensitive(df: pd.DataFrame, target_col: str) -> Optional[str]:
    """Find a column in df matching target_col regardless of case or spaces."""
    target_clean = target_col.lower().replace("_", "").replace(" ", "").replace("-", "")
    for col in df.columns:
        col_clean = str(col).lower().replace("_", "").replace(" ", "").replace("-", "")
        if col_clean == target_clean:
            return col
    return None


class SpatialAblateClientV2:
    """Client interface for interacting with the SpatialAblate REST API (v2)."""

    def __init__(self, base_url: str = "https://fydp-leaderboard-nextjs.vercel.app"):
        """
        Initialize the SpatialAblate API Client v2.

        Args:
            base_url (str): Base URL of the SpatialAblate server.
        """
        self.base_url = base_url.rstrip("/")
        self.token: Optional[str] = None
        self.session = requests.Session()

    def login(self, email: str, password: str) -> str:
        """Authenticate with SpatialAblate and obtain a JWT access token."""
        url = f"{self.base_url}/api/auth/login"
        payload = {"email": email, "password": password}

        response = self.session.post(url, json=payload)
        if response.status_code != 200:
            raise requests.HTTPError(
                f"Authentication failed (HTTP {response.status_code}): {response.text}",
                response=response
            )

        data = response.json()
        self.token = data.get("token")
        if not self.token:
            raise ValueError("Token not found in login response.")

        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        })
        print(f"[OK] Successfully authenticated as '{data.get('email')}' (Role: {data.get('role')})")
        return self.token

    def upload_model_csv(
        self,
        model_name: str,
        rows: List[Dict[str, Any]],
        description: Optional[str] = None,
        github_url: Optional[str] = None,
        paper_url: Optional[str] = None,
        colab_url: Optional[str] = None,
        kaggle_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit a Model Submission via /api/models/upload-csv."""
        if not self.token:
            raise PermissionError("Client is not authenticated. Call login(email, password) first.")

        url = f"{self.base_url}/api/models/upload-csv"
        payload = {
            "model_name": model_name,
            "rows": rows
        }
        if description: payload["description"] = description
        if github_url: payload["github_url"] = github_url
        if paper_url: payload["paper_url"] = paper_url
        if colab_url: payload["colab_url"] = colab_url
        if kaggle_url: payload["kaggle_url"] = kaggle_url

        print(f"\n[Model Submission] Uploading '{model_name}' ({len(rows)} evaluation rows)...")
        response = self.session.post(url, json=payload)
        if response.status_code not in (200, 201):
            err_msg = f"Model Submission failed (HTTP {response.status_code}): {response.text}"
            print(f"[X] {err_msg}")
            raise requests.HTTPError(err_msg, response=response)

        res_data = response.json()
        print(f"[OK] Model Submission uploaded successfully! (Model: {res_data.get('modelName', model_name)})")
        return res_data

    def upload_ablation_csv(
        self,
        model_name: str,
        base_model_name: str,
        ablation_tag: str,
        rows: List[Dict[str, Any]],
        description: Optional[str] = None,
        github_url: Optional[str] = None,
        paper_url: Optional[str] = None,
        colab_url: Optional[str] = None,
        kaggle_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Submit an Ablation Submission via /api/ablation/upload-csv."""
        if not self.token:
            raise PermissionError("Client is not authenticated. Call login(email, password) first.")

        url = f"{self.base_url}/api/ablation/upload-csv"
        payload = {
            "model_name": model_name,
            "base_model_name": base_model_name,
            "ablation_tag": ablation_tag,
            "rows": rows
        }
        if description: payload["description"] = description
        if github_url: payload["github_url"] = github_url
        if paper_url: payload["paper_url"] = paper_url
        if colab_url: payload["colab_url"] = colab_url
        if kaggle_url: payload["kaggle_url"] = kaggle_url

        print(f"\n[Ablation Submission] Uploading '{model_name}' (Tag: {ablation_tag}, Base: {base_model_name})...")
        response = self.session.post(url, json=payload)
        if response.status_code not in (200, 201):
            err_msg = f"Ablation Submission failed (HTTP {response.status_code}): {response.text}"
            print(f"[X] {err_msg}")
            raise requests.HTTPError(err_msg, response=response)

        res_data = response.json()
        print(f"[OK] Ablation Submission uploaded successfully! (Variant: {model_name})")
        return res_data

    def process_and_upload_csv(
        self,
        csv_path: Optional[str] = None,
        base_model_name: Optional[str] = None,
        description: Optional[str] = None,
        github_url: Optional[str] = None,
        paper_url: Optional[str] = None,
        colab_url: Optional[str] = None,
        kaggle_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main workflow method:
        1. Reads and standardizes CSV evaluation data.
        2. Infers cluster counts for datasets.
        3. Computes multi-metric scores per algorithm across all datasets.
        4. Identifies the BEST clustering algorithm.
        5. Uploads BEST algorithm as Model Submission.
        6. Uploads all other algorithm variants as Ablation Submissions referencing the base model.
        """
        if not csv_path:
            csv_path = input("Enter CSV file path (e.g. scgpt_spatial_ablation_results.csv): ").strip().strip("'\"")

        if not csv_path:
            raise ValueError("CSV path cannot be empty.")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: '{csv_path}'")

        if not base_model_name:
            base_model_name = input("Enter base model name (e.g. scGPT-spatial CONCAT): ").strip().strip("'\"")
            if not base_model_name:
                raise ValueError("Base model name cannot be empty.")

        # Resolve description (either string or path to .md file)
        if not description:
            desc_inp = input("Enter description text OR path to Markdown (.md) file (optional): ").strip()
            description = resolve_description(desc_inp)
        else:
            description = resolve_description(description)

        # Prompt for Kaggle URL if omitted
        if not kaggle_url:
            kaggle_inp = input("Enter Kaggle URL (optional): ").strip()
            kaggle_url = kaggle_inp.strip().strip("'\"") if kaggle_inp else None
        else:
            kaggle_url = kaggle_url.strip().strip("'\"")

        df = pd.read_csv(csv_path)

        # Locate dataset column
        dataset_col = find_column_case_insensitive(df, "dataset")
        if not dataset_col:
            raise ValueError("CSV must contain a 'dataset' column.")

        # Locate cluster algorithm column
        alg_col = find_column_case_insensitive(df, "cluster alg") or find_column_case_insensitive(df, "cluster_algorithm")
        if not alg_col:
            raise ValueError("CSV must contain a 'cluster alg' or 'cluster_algorithm' column.")

        # Locate seed column if present
        seed_col = find_column_case_insensitive(df, "seed")

        # Map metric columns cleanly without duplicates
        seen_cols = set()
        present_metrics = []
        for metric in ALL_KNOWN_METRICS:
            col_match = find_column_case_insensitive(df, metric)
            if col_match and col_match not in seen_cols:
                seen_cols.add(col_match)
                df[col_match] = pd.to_numeric(df[col_match], errors="coerce")
                present_metrics.append((metric, col_match))

        if len(present_metrics) < 2:
            raise ValueError(f"CSV must contain at least 2 metric columns from: {ALL_KNOWN_METRICS}")

        # Resolve cluster count (no_cluster) per row
        no_cluster_col = find_column_case_insensitive(df, "no_cluster") or find_column_case_insensitive(df, "cluster_count")
        if not no_cluster_col:
            df["no_cluster"] = df[dataset_col].apply(get_cluster_count)
        else:
            df["no_cluster"] = df[no_cluster_col].fillna(df[dataset_col].apply(get_cluster_count)).astype(int)

        algorithms = df[alg_col].unique().tolist()
        print(f"\nDetected {len(algorithms)} clustering algorithms in CSV: {algorithms}")
        print(f"Detected metric columns: {[m[0] for m in present_metrics]}")

        # Step 1: Compute dataset-level metric means per algorithm
        metric_col_names = [m[1] for m in present_metrics]
        agg_df = df.groupby([alg_col, dataset_col])[metric_col_names].mean().reset_index()

        # Step 2: Compute overall algorithm averages across all datasets
        alg_summary = agg_df.groupby(alg_col)[metric_col_names].mean()

        print("\n==================================================")
        print("    OVERALL METRICS SUMMARY ACROSS ALL DATASETS   ")
        print("==================================================")
        print(alg_summary.to_string())
        print("==================================================\n")

        # Step 3: Multi-Metric Composite Score Normalization
        normalized_scores = pd.DataFrame(index=alg_summary.index)

        for std_metric_name, col_name in present_metrics:
            vals = alg_summary[col_name]
            if isinstance(vals, pd.DataFrame):
                vals = vals.iloc[:, 0]

            min_v = float(vals.min())
            max_v = float(vals.max())

            if np.isclose(max_v, min_v):
                normalized_scores[std_metric_name] = 1.0
            else:
                if std_metric_name in LOWER_IS_BETTER_METRICS:
                    normalized_scores[std_metric_name] = (max_v - vals) / (max_v - min_v)
                else:
                    normalized_scores[std_metric_name] = (vals - min_v) / (max_v - min_v)

        normalized_scores["Composite_Score"] = normalized_scores.mean(axis=1)
        sorted_scores = normalized_scores.sort_values(by="Composite_Score", ascending=False)

        print("==================================================")
        print("     NORMALIZED COMPOSITE RANKING ACROSS ALL METRICS")
        print("==================================================")
        print(sorted_scores.to_string())
        print("==================================================\n")

        best_alg = sorted_scores.index[0]
        print(f"[*] BEST CLUSTERING ALGORITHM IDENTIFIED: '{best_alg}' (Composite Score: {sorted_scores.loc[best_alg, 'Composite_Score']:.4f})")

        # Step 4: Prepare JSON row payloads for each algorithm
        def format_row_payload(row_dict: Dict[str, Any]) -> Dict[str, Any]:
            p = {
                "dataset": str(row_dict.get(dataset_col)),
                "no_cluster": int(row_dict.get("no_cluster", get_cluster_count(str(row_dict.get(dataset_col))))),
            }
            if seed_col and not pd.isna(row_dict.get(seed_col)):
                p["seed"] = int(row_dict[seed_col])
            else:
                p["seed"] = 42

            for std_metric, c_name in present_metrics:
                val = row_dict.get(c_name)
                if not pd.isna(val):
                    p[std_metric] = float(val)

            return p

        alg_rows_map: Dict[str, List[Dict[str, Any]]] = {}
        for alg in algorithms:
            sub_df = df[df[alg_col] == alg]
            rows_list = [format_row_payload(r) for r in sub_df.to_dict(orient="records")]
            alg_rows_map[alg] = rows_list

        # Step 5: Upload Model Submission for Best Algorithm
        best_model_submission_name = f"{base_model_name} ({best_alg})"
        best_rows = alg_rows_map[best_alg]

        model_res = self.upload_model_csv(
            model_name=best_model_submission_name,
            rows=best_rows,
            description=description or f"Top performing model variant evaluated with {best_alg} clustering.",
            github_url=github_url,
            paper_url=paper_url,
            colab_url=colab_url,
            kaggle_url=kaggle_url
        )

        # Step 6: Upload Ablation Submissions for all algorithms
        ablation_results = []
        ablation_algs = algorithms

        for alg in ablation_algs:
            variant_name = f"{base_model_name} ({alg})"
            ablation_rows = alg_rows_map[alg]
            ab_res = self.upload_ablation_csv(
                model_name=variant_name,
                base_model_name=base_model_name,
                ablation_tag=alg,
                rows=ablation_rows,
                description=description or f"Ablation variant of {base_model_name} using {alg} clustering algorithm.",
                github_url=github_url,
                paper_url=paper_url,
                colab_url=colab_url,
                kaggle_url=kaggle_url
            )
            ablation_results.append(ab_res)

        print("\n==================================================")
        print("           SUBMISSION PROCESS COMPLETE            ")
        print("==================================================")
        print(f"[OK] Model Submission:    {best_model_submission_name}")
        print(f"[OK] Ablation Submissions: {len(ablation_results)} uploaded")
        for res in ablation_results:
            print(f"   - Variant: {res.get('modelName')} (Base: {base_model_name}, Tag: {res.get('ablationTag')})")
        print("==================================================\n")

        return {
            "best_alg": best_alg,
            "model_submission": model_res,
            "ablation_submissions": ablation_results
        }


def main():
    """CLI entry point for SpatialAblate Client v2."""
    env_vars = load_env_file(".env")

    env_url = env_vars.get("url") or env_vars.get("base_url") or os.getenv("URL") or os.getenv("BASE_URL") or "https://fydp-ablation.vercel.app"
    env_email = env_vars.get("email") or os.getenv("EMAIL")
    env_password = env_vars.get("password") or os.getenv("PASSWORD")
    env_description = env_vars.get("description") or env_vars.get("desc") or os.getenv("DESCRIPTION")
    env_kaggle = env_vars.get("kaggle") or env_vars.get("kaggle_url") or os.getenv("KAGGLE") or os.getenv("KAGGLE_URL")

    parser = argparse.ArgumentParser(
        description="SpatialAblate Benchmark Leaderboard Upload Tool (v2)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default=env_url, help="SpatialAblate server base URL")
    parser.add_argument("--email", default=env_email, help="SpatialAblate account email")
    parser.add_argument("--password", default=env_password, help="SpatialAblate account password")
    
    parser.add_argument("--csv", help="Path to evaluation results CSV file (prompts interactively if omitted)")
    parser.add_argument("--model", help="Base Model name e.g. 'scGPT-spatial CONCAT' (prompts interactively if omitted)")

    parser.add_argument("--description", default=env_description, help="Description string OR path to .md file (prompts interactively if omitted)")
    parser.add_argument("--github", help="GitHub repository URL")
    parser.add_argument("--paper", help="Scientific paper URL")
    parser.add_argument("--colab", help="Google Colab notebook URL")
    parser.add_argument("--kaggle", default=env_kaggle, help="Kaggle notebook/dataset URL (prompts interactively if omitted)")

    args = parser.parse_args()

    url = args.url
    email = args.email or input("Enter email: ").strip()
    password = args.password or input("Enter password: ").strip()

    if env_vars:
        print("[*] Loaded configuration credentials from .env")

    client = SpatialAblateClientV2(base_url=url)
    try:
        client.login(email, password)
        client.process_and_upload_csv(
            csv_path=args.csv,
            base_model_name=args.model,
            description=args.description,
            github_url=args.github,
            paper_url=args.paper,
            colab_url=args.colab,
            kaggle_url=args.kaggle
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
