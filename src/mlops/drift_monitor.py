"""
RetailPulse - MLOps Data Drift & Model Monitoring
Monitors distribution shifts across time series and tabular customer features
using Population Stability Index (PSI) and Two-Sample Kolmogorov-Smirnov distance in pure NumPy.
Resilient to Windows Application Control policies.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def ks_2samp_numpy(data1: np.ndarray, data2: np.ndarray):
    """Computes Two-Sample Kolmogorov-Smirnov supremum distance in pure NumPy."""
    d1 = np.sort(data1[~np.isnan(data1)])
    d2 = np.sort(data2[~np.isnan(data2)])
    n1 = len(d1)
    n2 = len(d2)

    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    all_data = np.concatenate([d1, d2])
    cdf1 = np.searchsorted(d1, all_data, side="right") / n1
    cdf2 = np.searchsorted(d2, all_data, side="right") / n2

    d_stat = float(np.max(np.abs(cdf1 - cdf2)))
    # Asymptotic p-value approximation
    en = np.sqrt((n1 * n2) / (n1 + n2))
    p_val = float(np.clip(2.0 * np.exp(-2.0 * ((en * d_stat) ** 2)), 0.0, 1.0))
    return d_stat, p_val


class DriftMonitor:
    """Enterprise Data Drift and Feature Monitoring Service."""

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or (PROCESSED_DIR / "daily_sales.csv")

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) < 10 or len(actual) < 10:
            return 0.0

        percentiles = np.linspace(0, 100, num_buckets + 1)
        bins = np.percentile(expected, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
        actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(np.round(psi_val, 4))

    def evaluate_drift(self, feature: str = "units_sold", split_ratio: float = 0.70):
        df = pd.read_csv(self.data_path)
        if feature not in df.columns:
            return {"error": f"Feature {feature} not found."}

        split_idx = int(len(df) * split_ratio)
        baseline = df[feature].iloc[:split_idx].values
        current = df[feature].iloc[split_idx:].values

        ks_stat, p_value = ks_2samp_numpy(baseline, current)
        psi_score = self.calculate_psi(baseline, current)

        if psi_score >= 0.25 or p_value < 0.01:
            status = "CRITICAL DRIFT - Model Retraining Urgently Recommended"
            badge = "🔴 Critical"
        elif psi_score >= 0.10 or p_value < 0.05:
            status = "MODERATE DRIFT - Increase Monitoring Frequency"
            badge = "🟡 Warning"
        else:
            status = "STABLE - Feature Distribution In Spec"
            badge = "🟢 Stable"

        report = {
            "feature": feature,
            "baseline_samples": len(baseline),
            "current_samples": len(current),
            "baseline_mean": round(float(np.mean(baseline)), 2),
            "current_mean": round(float(np.mean(current)), 2),
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_value), 4),
            "psi_score": psi_score,
            "drift_status": status,
            "status_badge": badge
        }

        print(f"[DriftMonitor] Feature: {feature} | PSI: {psi_score} | KS p-val: {p_value:.4f} | {badge}")
        return report


if __name__ == "__main__":
    monitor = DriftMonitor()
    monitor.evaluate_drift("units_sold")
    monitor.evaluate_drift("revenue")
