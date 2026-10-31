"""
RetailPulse - Customer Segmentation & Behavioral Intelligence
Performs RFM scoring, vectorized K-Means clustering, PCA dimensionality reduction,
and Customer Lifetime Value (CLV) estimation. Resilient to Windows Application Control policies.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def kmeans_numpy(X: np.ndarray, k: int = 5, max_iters: int = 100, random_seed: int = 42):
    """Vectorized K-Means clustering in pure NumPy."""
    np.random.seed(random_seed)
    n_samples = X.shape[0]
    initial_idx = np.random.choice(n_samples, size=k, replace=False)
    centroids = X[initial_idx].copy()

    for _ in range(max_iters):
        # Compute distances: (n_samples, k)
        distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids

    return labels, centroids


def pca_numpy(X: np.ndarray, n_components: int = 3):
    """Principal Component Analysis using SVD in pure NumPy."""
    X_centered = X - np.mean(X, axis=0)
    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return (X_centered @ Vt.T[:, :n_components])


class CustomerSegmentationEngine:
    """Advanced Customer Segmentation with K-Means, PCA, and RFM Profiling."""

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or (PROCESSED_DIR / "customer_rfm.csv")
        self.df = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"[Segmentation] Loaded {len(self.df)} customer records.")
        return self.df

    def perform_kmeans_clustering(self, n_clusters: int = 5):
        if self.df is None:
            self.load_data()

        features = ["recency_days", "purchase_count", "total_spent", "customer_tenure_days"]
        X = self.df[features].fillna(0).values.astype(float)

        # Standardize features
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std == 0, 1.0, std)
        X_scaled = (X - mean) / std

        # K-Means clustering
        labels, centroids = kmeans_numpy(X_scaled, k=n_clusters)
        self.df["cluster_id"] = labels

        # PCA 3D projection
        coords = pca_numpy(X_scaled, n_components=3)
        self.df["pca_x"] = np.round(coords[:, 0], 3)
        self.df["pca_y"] = np.round(coords[:, 1], 3)
        self.df["pca_z"] = np.round(coords[:, 2], 3)

        # Compute Customer Lifetime Value (CLV)
        margin_rate = 0.28
        aov = np.where(self.df["purchase_count"] > 0, self.df["total_spent"] / self.df["purchase_count"], 0)
        expected_lifespan = np.where(self.df["is_churned"] == 1, 0.2, 1.8)
        tenure_years = np.maximum(0.2, self.df["customer_tenure_days"] / 365.0)
        annual_orders = np.maximum(1.0, self.df["purchase_count"] / tenure_years)

        predicted_future = aov * annual_orders * margin_rate * expected_lifespan
        self.df["predicted_clv"] = np.round(self.df["total_spent"] + predicted_future, 2)

        # Persona mapping
        persona_names = {
            0: "VIP Champions",
            1: "High-Value Loyalists",
            2: "Promising New",
            3: "At-Risk High Spenders",
            4: "Hibernating / Dormant"
        }
        self.df["cluster_persona"] = self.df["cluster_id"].map(persona_names).fillna("Core Retail Shoppers")

        self.df.to_csv(self.data_path, index=False)
        print(f"[Segmentation] Completed K-Means & CLV. Updated {self.data_path}")
        return self.df


if __name__ == "__main__":
    engine = CustomerSegmentationEngine()
    engine.perform_kmeans_clustering()
