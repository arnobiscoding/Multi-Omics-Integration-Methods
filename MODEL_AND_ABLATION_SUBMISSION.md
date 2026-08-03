# SpatialAblate: Comprehensive Model & Ablation Submission Guide

This guide provides an end-to-end specification of how **Model Submissions** and **Ablation Submissions** function in the **SpatialAblate** platform. It details the underlying MongoDB schemas, data validation rules, CSV formatting requirements, REST API endpoint specifications, and Python automation workflows.

---

## 📋 Table of Contents
1. [Overview & Core Concepts](#1-overview--core-concepts)
2. [Model Submissions vs. Ablation Submissions](#2-model-submissions-vs-ablation-submissions)
3. [Required & Supported Benchmark Metrics](#3-required--supported-benchmark-metrics)
4. [Dataset Resolution & Fuzzy Matching](#4-dataset-resolution--fuzzy-matching)
5. [CSV File Formats & Examples](#5-csv-file-formats--examples)
6. [API Endpoints Reference](#6-api-endpoints-reference)
7. [Python Automation & Scripting Guide](#7-python-automation--scripting-guide)
8. [Validation & Error Codes](#8-validation--error-codes)

---

## 1. Overview & Core Concepts

SpatialAblate benchmark results can be submitted through two channels:
1. **Interactive Web Interface**: Via the `/submit` page using web forms or browser-side CSV upload.
2. **Programmatic REST API**: Via HTTP requests from Python pipelines, Jupyter Notebooks, or CLI tools.

When a submission is created:
- Model metadata (description, publication links, architecture diagrams) is stored in a **`ModelProfile`**.
- Benchmarking evaluations ($K$-clusters, seeds, metrics) are aggregated into a **`ModelSubmission`** or **`AblationSubmission`**.
- Evaluations are linked to standardized biological datasets (**`DatasetSection`**).

---

## 2. Model Submissions vs. Ablation Submissions

### 🏆 Model Submissions
* **Purpose**: Tracks standalone baseline deep learning models (e.g., `SpatialGlue`, `Seurat_v4`, `STAGATE`).
* **Entity**: Stored in the `ModelSubmission` collection.
* **Profile Link**: Tied to a canonical `ModelProfile`. Updates to the model profile automatically synchronize across all user submissions sharing that profile.
* **Key Fields**:
  - `model_name` *(String, Required)*: Canonical model name.
  - `results` *(Array)*: Evaluation entries per dataset, cluster size ($K$), seed, and algorithm.

### 🔬 Ablation Submissions
* **Purpose**: Tracks structural or algorithmic variants of a base model to measure component contributions (e.g., removing a GNN encoder, disabling contrastive loss, or testing alternative decoders).
* **Entity**: Stored in the `AblationSubmission` collection.
* **Base Model Link**: References a base model profile via `baseModelName` / `baseModelProfileId`.
* **Promotion System**: Admin users can promote active ablation variants (`status: 'active'`) to full Model status (`status: 'promoted'`), making them visible in the primary leaderboard.
* **Key Fields**:
  - `model_name` *(String, Required)*: Name of the ablated model variant (e.g., `"SpatialGlue (w/o GCN)"`).
  - `base_model_name` *(String, Required)*: Name of the base model being evaluated (e.g., `"SpatialGlue"`).
  - `ablation_tag` *(String, Required)*: Short identifier for the ablated component (e.g., `"No-GCN"`, `"No-Contrastive"`).

---

## 3. Required & Supported Benchmark Metrics

Submissions evaluate spatial clustering performance across external ground-truth comparison metrics and internal clustering validation metrics.

### Primary Metrics (Rule: At least TWO are REQUIRED per evaluation entry)
| Metric | Code/Column Key | Range | Ideal Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Adjusted Rand Index** | `ARI`, `ari`, `scoreARI` | $[-1.0, 1.0]$ | $1.0$ | Measures agreement between predicted clusters and ground truth. |
| **Normalized Mutual Info** | `NMI`, `nmi`, `scoreNMI` | $[0.0, 1.0]$ | $1.0$ | Shared information between predicted and ground truth labels. |
| **Silhouette Coefficient** | `Silhouette`, `silhouette`, `scoreSilhouette` | $[-1.0, 1.0]$ | $1.0$ | Measures cluster cohesion and separation without ground truth. |

### Secondary Metrics (Optional)
| Metric | Code/Column Key | Range | Ideal Value | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Adjusted Mutual Info** | `AMI`, `ami`, `scoreAMI` | $[-1.0, 1.0]$ | $1.0$ | Adjusted mutual information score. |
| **Calinski-Harabasz Index** | `CHI`, `chi`, `scoreCHI` | $[0, \infty)$ | Higher is better | Ratio of between-clusters dispersion to within-cluster dispersion. |
| **Davies-Bouldin Index** | `DBI`, `dbi`, `scoreDBI` | $[0, \infty)$ | Lower is better | Average similarity measure of each cluster with its most similar cluster. |
| **Homogeneity** | `homogeneity`, `scoreHomogeneity` | $[0.0, 1.0]$ | $1.0$ | Checks if all clusters contain only data points of a single class. |
| **V-Measure** | `v_measure`, `vmeasure`, `scoreVMeasure` | $[0.0, 1.0]$ | $1.0$ | Harmonic mean of homogeneity and completeness. |

---

## 4. Dataset Resolution & Fuzzy Matching

The platform matches dataset names flexibly. When submitting, dataset strings are converted to lowercase and stripped of non-alphanumeric characters.

### Pre-Seeded Benchmark Datasets
1. `Human_Lymph_Node_A1`
2. `Human_Lymph_Node_D1`
3. `Mouse_Brain_ATAC`
4. `Mouse_Brain_H3K27ac`
5. `Mouse_Brain_H3K27me`
6. `Mouse_Brain_H3K4me`
7. `Mouse_Spleen`
8. `Mouse_Thymus`
9. `Mouse_Brain_E11_S1`
10. `Mouse_Brain_E13_S1`
11. `Mouse_Brain_E15_S1`
12. `Mouse_Brain_E18_S1`

#### Examples of Matching String Inputs
- `"Human Lymph Node A1"` $\rightarrow$ Resolves to `Human_Lymph_Node_A1`
- `"mouse_brain_atac"` $\rightarrow$ Resolves to `Mouse_Brain_ATAC`
- `"human-lymph-node-d1"` $\rightarrow$ Resolves to `Human_Lymph_Node_D1`

---

## 5. CSV File Formats & Examples

### Model Evaluation CSV (`model_results.csv`)

```csv
dataset,no_cluster,seed,ARI,NMI,Silhouette,CHI,DBI,cluster_algorithm
Human_Lymph_Node_A1,10,42,0.8542,0.8120,0.4531,124.5,0.82,mclust
Human_Lymph_Node_A1,10,43,0.8490,0.8095,0.4490,121.3,0.84,mclust
Mouse_Brain_ATAC,12,42,0.7812,0.7650,0.4120,98.2,0.95,kmeans
```

### Ablation Evaluation CSV (`ablation_results.csv`)

```csv
dataset,no_cluster,seed,ARI,NMI,Silhouette,CHI,DBI
Human_Lymph_Node_A1,10,42,0.8105,0.7910,0.4210,110.2,0.89
Mouse_Brain_ATAC,12,42,0.7230,0.7105,0.3850,88.4,1.05
```

---

## 6. API Endpoints Reference

### Authentication Header (Required for all POST/PUT/DELETE calls)
```http
Authorization: Bearer <YOUR_JWT_TOKEN>
Content-Type: application/json
```

---

### 1. Authenticate / Login
`POST /api/auth/login`

#### Request Payload
```json
{
  "email": "user@example.com",
  "password": "yourpassword"
}
```

#### Response (HTTP 200)
```json
{
  "_id": "66a1b2c3d4e5f67890123456",
  "name": "Researcher Name",
  "email": "user@example.com",
  "role": "member",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

---

### 2. Upload Model CSV
`POST /api/models/upload-csv`

#### Request Payload
```json
{
  "model_name": "SpatialGlue",
  "description": "SpatialGlue model evaluated across spatial omics datasets.",
  "github_url": "https://github.com/example/SpatialGlue",
  "paper_url": "https://doi.org/10.1038/example",
  "rows": [
    {
      "dataset": "Human_Lymph_Node_A1",
      "no_cluster": 10,
      "seed": 42,
      "ARI": 0.8542,
      "NMI": 0.8120,
      "Silhouette": 0.4531,
      "CHI": 124.5,
      "DBI": 0.82
    }
  ]
}
```

#### Response (HTTP 200)
```json
{
  "message": "CSV results uploaded successfully.",
  "modelName": "SpatialGlue",
  "total": 1,
  "processed": 1,
  "skipped": 0,
  "unmatchedDatasets": [],
  "errors": []
}
```

---

### 3. Upload Ablation CSV
`POST /api/ablation/upload-csv`

#### Request Payload
```json
{
  "model_name": "SpatialGlue (w/o GCN)",
  "base_model_name": "SpatialGlue",
  "ablation_tag": "No-GCN",
  "description": "Ablation study removing GCN layer from SpatialGlue.",
  "rows": [
    {
      "dataset": "Human_Lymph_Node_A1",
      "no_cluster": 10,
      "seed": 42,
      "ARI": 0.8105,
      "NMI": 0.7910,
      "Silhouette": 0.4210
    }
  ]
}
```

#### Response (HTTP 200)
```json
{
  "message": "Ablation CSV results uploaded successfully.",
  "modelName": "SpatialGlue (w/o GCN)",
  "baseModelName": "SpatialGlue",
  "ablationTag": "No-GCN",
  "total": 1,
  "processed": 1,
  "skipped": 0,
  "unmatchedDatasets": [],
  "errors": []
}
```

---

### 4. Upload Single Model Result
`POST /api/models/upload-result`

#### Request Payload
```json
{
  "model_name": "SpatialGlue",
  "dataset_name": "Human_Lymph_Node_A1",
  "cluster_count": 10,
  "seed": 42,
  "ari": 0.8542,
  "nmi": 0.8120,
  "silhouette": 0.4531,
  "github_url": "https://github.com/example/SpatialGlue"
}
```

---

## 7. Python Automation & Scripting Guide

You can integrate SpatialAblate directly into python training loops or execution scripts.

### Full Python Script (`auto_submit.py`)

```python
import pandas as pd
import requests

class SpatialAblateClient:
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip("/")
        self.token = None

    def login(self, email: str, password: str) -> str:
        url = f"{self.base_url}/api/auth/login"
        res = requests.post(url, json={"email": email, "password": password})
        res.raise_for_status()
        self.token = res.json()["token"]
        print(f"✓ Authenticated successfully. User: {res.json().get('name')}")
        return self.token

    def submit_model_csv(self, csv_filepath: str, model_name: str, description: str = "", github_url: str = ""):
        if not self.token:
            raise PermissionError("Call client.login() before submitting.")

        df = pd.read_csv(csv_filepath)
        payload = {
            "model_name": model_name,
            "description": description,
            "github_url": github_url,
            "rows": df.to_dict(orient="records")
        }
        
        headers = {"Authorization": f"Bearer {self.token}"}
        res = requests.post(f"{self.base_url}/api/models/upload-csv", json=payload, headers=headers)
        res.raise_for_status()
        return res.json()

    def submit_ablation_csv(self, csv_filepath: str, model_name: str, base_model_name: str, ablation_tag: str, description: str = ""):
        if not self.token:
            raise PermissionError("Call client.login() before submitting.")

        df = pd.read_csv(csv_filepath)
        payload = {
            "model_name": model_name,
            "base_model_name": base_model_name,
            "ablation_tag": ablation_tag,
            "description": description,
            "rows": df.to_dict(orient="records")
        }
        
        headers = {"Authorization": f"Bearer {self.token}"}
        res = requests.post(f"{self.base_url}/api/ablation/upload-csv", json=payload, headers=headers)
        res.raise_for_status()
        return res.json()

# --- Execution Example ---
if __name__ == "__main__":
    client = SpatialAblateClient("http://localhost:3000")
    client.login("admin@gmail.com", "admin")

    # Submit Baseline Model
    res_model = client.submit_model_csv(
        csv_filepath="sample_model_output_csv_file.csv",
        model_name="SpatialGlue",
        description="SpatialGlue benchmark run",
        github_url="https://github.com/example/SpatialGlue"
    )
    print("Model submission response:", res_model)

    # Submit Ablation Variant
    res_ablation = client.submit_ablation_csv(
        csv_filepath="sample_model_output_csv_file.csv",
        model_name="SpatialGlue (w/o GCN)",
        base_model_name="SpatialGlue",
        ablation_tag="No-GCN",
        description="Removed GCN feature extraction layer"
    )
    print("Ablation submission response:", res_ablation)
```

---

## 8. Validation & Error Codes

| HTTP Status | Trigger Condition | Solution |
| :--- | :--- | :--- |
| `400 Bad Request` | Fewer than 2 primary metrics (`ARI`, `NMI`, `Silhouette`) provided per row. | Ensure at least 2 primary metric columns exist with valid float values. |
| `400 Bad Request` | Missing required payload parameters (`model_name`, `base_model_name`, `ablation_tag`, or `rows`). | Provide all required body parameters in the JSON payload. |
| `400 Bad Request` | `clusterSize` / `no_cluster` is $\le 0$ or non-numeric. | Set positive integer cluster counts (e.g. 7, 10, 15). |
| `401 Unauthorized` | Missing or invalid `Authorization: Bearer <token>` header. | Login via `/api/auth/login` and pass the returned JWT token. |
| `500 Server Error` | Database connection error or unhandled internal exception. | Verify MongoDB connection and server log trace. |
