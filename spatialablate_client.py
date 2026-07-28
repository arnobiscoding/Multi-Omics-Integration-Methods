"""
SpatialAblate Benchmark Leaderboard API Client & CSV Importer
===========================================================

This module provides a Python client and CLI interface for programmatically uploading
spatial multi-omics model benchmark evaluation results to the SpatialAblate platform.

Features:
---------
- Automatic cluster count resolution based on dataset section names:
    * a1  -> 10
    * d1  -> 11
    * e11 -> 8
    * e13 -> 12
    * e15 -> 12
    * e18 -> 14
- Reads result CSV files (e.g., spallm_ablation_results.csv)
- Calculates and displays metric Mean and Std for all datasets (matching print.py style)
- Prompts for model name and uploads ONLY mean metrics to the SpatialAblate platform
"""

import os
import sys
import argparse
import pandas as pd
import requests
from typing import Dict, Any, Optional, List


CLUSTER_COUNT_MAP: Dict[str, int] = {
    "a1": 10,
    "d1": 11,
    "e11": 8,
    "e13": 12,
    "e15": 12,
    "e18": 14,
}


def get_cluster_count(dataset_name: str) -> int:
    """
    Derive cluster count K based on dataset name matching rules:
    - a1  -> 10
    - d1  -> 11
    - e11 -> 8
    - e13 -> 12
    - e15 -> 12
    - e18 -> 14
    """
    name_lower = dataset_name.lower()
    keys_in_order = ["e11", "e13", "e15", "e18", "a1", "d1"]
    for key in keys_in_order:
        if key in name_lower:
            return CLUSTER_COUNT_MAP[key]
    raise ValueError(
        f"Could not infer cluster count for dataset '{dataset_name}'. "
        f"Expected dataset name to contain one of: {list(CLUSTER_COUNT_MAP.keys())}"
    )


class SpatialAblateClient:
    """Client interface for interacting with the SpatialAblate REST API."""

    def __init__(self, base_url: str = "https://fydp-leaderboard-nextjs.vercel.app"):
        """
        Initialize the SpatialAblate API Client.

        Args:
            base_url (str): Base URL of the SpatialAblate server (default: https://fydp-leaderboard-nextjs.vercel.app)
        """
        self.base_url = base_url.rstrip("/")
        self.token: Optional[str] = None
        self.session = requests.Session()

    def login(self, email: str, password: str) -> str:
        """
        Authenticate with SpatialAblate and obtain a JWT access token.
        """
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
        print(f"✓ Successfully authenticated as '{data.get('email')}' (Role: {data.get('role')})")
        return self.token

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Fetch all benchmark dataset sections available in SpatialAblate."""
        url = f"{self.base_url}/api/sections"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def submit_result(
        self,
        model_name: str,
        dataset_name: str,
        cluster_count: Optional[int] = None,
        ari: Optional[float] = None,
        nmi: Optional[float] = None,
        silhouette: Optional[float] = None,
        ami: Optional[float] = None,
        homogeneity: Optional[float] = None,
        v_measure: Optional[float] = None,
        github_url: Optional[str] = None,
        paper_url: Optional[str] = None,
        colab_url: Optional[str] = None,
        kaggle_url: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit a model benchmark result for a specific dataset and cluster size.
        If cluster_count is omitted, it will be derived automatically from the dataset name.
        """
        if not self.token:
            raise PermissionError("Client is not authenticated. Please call login(email, password) first.")

        if cluster_count is None:
            cluster_count = get_cluster_count(dataset_name)

        primary_metrics = [m for m in (ari, nmi, silhouette) if m is not None]
        if len(primary_metrics) < 2:
            raise ValueError("You must provide at least two primary metrics: 'ari', 'nmi', or 'silhouette'.")

        url = f"{self.base_url}/api/models/upload-result"
        payload = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "cluster_count": cluster_count
        }

        if ari is not None: payload["ari"] = float(ari)
        if nmi is not None: payload["nmi"] = float(nmi)
        if silhouette is not None: payload["silhouette"] = float(silhouette)
        if ami is not None: payload["ami"] = float(ami)
        if homogeneity is not None: payload["homogeneity"] = float(homogeneity)
        if v_measure is not None: payload["v_measure"] = float(v_measure)

        if github_url: payload["github_url"] = github_url
        if paper_url: payload["paper_url"] = paper_url
        if colab_url: payload["colab_url"] = colab_url
        if kaggle_url: payload["kaggle_url"] = kaggle_url
        if description: payload["description"] = description

        print(f"Uploading result for '{model_name}' on dataset '{dataset_name}' (K={cluster_count})...")
        response = self.session.post(url, json=payload)
        
        if response.status_code not in (200, 201):
            err_msg = f"Submission failed (HTTP {response.status_code}): {response.text}"
            print(f"✗ {err_msg}")
            raise requests.HTTPError(err_msg, response=response)

        data = response.json()
        print("✓ Submission uploaded successfully!")
        print(f"  Submission ID:   {data.get('submissionId')}")
        print(f"  Matched Dataset: {data.get('matchedDataset')}")
        print(f"  Cluster Size:    {data.get('clusterSize')}\n")
        return data

    def process_and_upload_csv(
        self,
        csv_path: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Reads CSV file, calculates mean & std metrics like print.py for all datasets,
        asks for csv_path and model_name if not provided, and uploads ONLY the mean metrics.
        """
        if not csv_path:
            csv_path = input("Enter CSV file path (default: spallm_ablation_results.csv): ")
            
        csv_path = csv_path.strip().strip("'\"") if csv_path else ""
        if not csv_path:
            csv_path = "spallm_ablation_results.csv"

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: '{csv_path}'")

        df = pd.read_csv(csv_path)

        metric_cols = ["ARI", "NMI", "AMI", "Homogeneity", "V-measure", "Silhouette"]
        existing_metrics = [m for m in metric_cols if m in df.columns]

        if not existing_metrics:
            raise ValueError(f"CSV does not contain any expected metric columns: {metric_cols}")

        df[existing_metrics] = df[existing_metrics].apply(pd.to_numeric, errors="coerce")

        if not model_name:
            model_name = input("Enter model name (e.g. SpaLLM): ").strip().strip("'\"")
            if not model_name:
                raise ValueError("Model name cannot be empty.")

        group_cols = ["dataset"]
        if "cluster alg" in df.columns:
            group_cols.append("cluster alg")

        grouped = df.groupby(group_cols)

        print("\n==================================================")
        print("       BENCHMARK SUMMARY (Mean ± Std)")
        print("==================================================\n")

        upload_results = []
        for group_keys, group in grouped:
            if isinstance(group_keys, tuple):
                dataset = group_keys[0]
                algorithm = group_keys[1]
            else:
                dataset = group_keys
                algorithm = None

            cluster_count = get_cluster_count(dataset)

            print(f"# Dataset:   {dataset} (K={cluster_count})")
            if algorithm:
                print(f"# Algorithm: {algorithm}")
            print("metrics = {")

            means = {}
            for col in existing_metrics:
                mean_val = group[col].mean()
                std_val = group[col].std()
                means[col] = mean_val
                print(f"  {col}: mean={mean_val:.6f}, std={std_val:.6f}")
            print("}\n")

            # Construct submission model name if algorithm column exists
            curr_model_name = f"{model_name} {algorithm}" if algorithm else model_name

            # Map mean metrics to API parameters
            kwargs = {}
            if "ARI" in means and not pd.isna(means["ARI"]): kwargs["ari"] = float(means["ARI"])
            if "NMI" in means and not pd.isna(means["NMI"]): kwargs["nmi"] = float(means["NMI"])
            if "Silhouette" in means and not pd.isna(means["Silhouette"]): kwargs["silhouette"] = float(means["Silhouette"])
            if "AMI" in means and not pd.isna(means["AMI"]): kwargs["ami"] = float(means["AMI"])
            if "Homogeneity" in means and not pd.isna(means["Homogeneity"]): kwargs["homogeneity"] = float(means["Homogeneity"])
            if "V-measure" in means and not pd.isna(means["V-measure"]): kwargs["v_measure"] = float(means["V-measure"])

            result = self.submit_result(
                model_name=curr_model_name,
                dataset_name=dataset,
                cluster_count=cluster_count,
                **kwargs
            )
            upload_results.append(result)

        return upload_results


def main():
    """Command Line Interface (CLI) entry point."""
    parser = argparse.ArgumentParser(
        description="SpatialAblate Benchmark Leaderboard Upload Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--url", default="https://fydp-leaderboard-nextjs.vercel.app", help="SpatialAblate server base URL")
    parser.add_argument("--email", required=True, help="SpatialAblate account email")
    parser.add_argument("--password", required=True, help="SpatialAblate account password")
    
    parser.add_argument("--csv", help="Path to evaluation results CSV file (prompts interactively if omitted)")
    parser.add_argument("--model", help="Model/Algorithm name (prompts interactively if omitted)")
    
    # Direct single-submission arguments
    parser.add_argument("--dataset", help="Dataset section name for single upload (e.g. 'human lymph node a1')")
    parser.add_argument("--ari", type=float, help="Adjusted Rand Index score [-1.0, 1.0]")
    parser.add_argument("--nmi", type=float, help="Normalized Mutual Information score [0.0, 1.0]")
    parser.add_argument("--silhouette", type=float, help="Silhouette Coefficient score [-1.0, 1.0]")
    parser.add_argument("--ami", type=float, help="Adjusted Mutual Information score")
    parser.add_argument("--homogeneity", type=float, help="Homogeneity score")
    parser.add_argument("--vmeasure", type=float, help="V-Measure score")
    parser.add_argument("--github", help="GitHub repository URL")
    parser.add_argument("--paper", help="Scientific paper URL")
    parser.add_argument("--colab", help="Google Colab notebook URL")

    args = parser.parse_args()

    client = SpatialAblateClient(base_url=args.url)
    try:
        client.login(args.email, args.password)

        if args.dataset:
            # Single manual submission mode
            client.submit_result(
                model_name=args.model or input("Enter model name: ").strip(),
                dataset_name=args.dataset,
                ari=args.ari,
                nmi=args.nmi,
                silhouette=args.silhouette,
                ami=args.ami,
                homogeneity=args.homogeneity,
                v_measure=args.vmeasure,
                github_url=args.github,
                paper_url=args.paper,
                colab_url=args.colab
            )
        else:
            # Batch CSV processing & upload mode
            client.process_and_upload_csv(
                csv_path=args.csv,
                model_name=args.model
            )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
