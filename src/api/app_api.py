"""
RetailPulse - Real-Time Analytics & Prediction REST API
Built with FastAPI, serving endpoints for demand forecasting, customer churn scoring,
stockout alerts, and MLOps health monitoring.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
import sys
import pandas as pd
import numpy as np

app = FastAPI(
    title="RetailPulse REST API",
    description="Enterprise AI-Powered Customer Analytics & Demand Forecasting Platform",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# Pydantic Request Models
class ChurnPredictionRequest(BaseModel):
    customer_id: Optional[str] = "CUST9999"
    recency_days: int = Field(..., ge=0, description="Days since last purchase")
    purchase_count: int = Field(..., ge=0, description="Total order frequency")
    total_spent: float = Field(..., ge=0.0, description="Cumulative lifetime revenue")
    customer_tenure_days: int = Field(..., ge=0, description="Tenure with brand in days")
    avg_order_value: float = Field(..., ge=0.0, description="Average basket size")
    age: int = Field(32, ge=18, le=100)


class DemandForecastRequest(BaseModel):
    horizon_days: int = Field(7, ge=1, le=60, description="Forecast horizon in days")
    store_id: Optional[str] = "ALL"
    sku_id: Optional[str] = "ALL"


# Endpoints
@app.get("/")
def read_root():
    return {
        "platform": "RetailPulse",
        "description": "AI-Powered Customer Analytics & Demand Forecasting Platform",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs"
    }


@app.get("/health")
def healthcheck():
    data_files = ["daily_sales.csv", "baseline_predictions.csv", "customer_rfm.csv", "inventory_analysis.csv"]
    status_map = {f: (PROCESSED_DIR / f).exists() for f in data_files}
    return {
        "status": "healthy" if all(status_map.values()) else "degraded",
        "datasets_available": status_map
    }


@app.post("/customer/churn")
def predict_churn(req: ChurnPredictionRequest):
    """Predicts churn probability and assigns retention playbook."""
    # Heuristic / tree-based scoring proxy
    prob = 0.05
    if req.recency_days > 60:
        prob += 0.45
    elif req.recency_days > 30:
        prob += 0.20

    if req.purchase_count < 3:
        prob += 0.25
    elif req.purchase_count > 15:
        prob -= 0.15

    if req.total_spent > 15000:
        prob -= 0.10

    churn_prob = round(float(np.clip(prob, 0.02, 0.98)), 3)

    if churn_prob >= 0.70:
        tier = "Critical Risk"
        recommendation = "High-Touch VIP Concierge: 25% exclusive voucher + Personal Phone Call" if req.total_spent > 15000 else "Win-Back Push: 20% discount on favorite items"
    elif churn_prob >= 0.40:
        tier = "Moderate Risk"
        recommendation = "Loyalty Boost: 1.5x Reward points for next purchase within 10 days"
    else:
        tier = "Low Risk"
        recommendation = "Standard Lifecycle nurturing & monthly newsletter"

    return {
        "customer_id": req.customer_id,
        "churn_probability": churn_prob,
        "churn_risk_tier": tier,
        "recommended_retention_action": recommendation
    }


@app.post("/forecast/demand")
def forecast_demand(req: DemandForecastRequest):
    """Retrieves forecasted demand for next horizon_days."""
    pred_path = PROCESSED_DIR / "baseline_predictions.csv"
    if not pred_path.exists():
        raise HTTPException(status_code=404, detail="Forecast predictions not generated yet.")

    df_preds = pd.read_csv(pred_path)
    future = df_preds[df_preds["actual_units"].isna()].head(req.horizon_days)

    return {
        "horizon_days": req.horizon_days,
        "forecast_period_start": str(future["date"].min()) if not future.empty else None,
        "forecast_period_end": str(future["date"].max()) if not future.empty else None,
        "projected_units_total": float(future["predicted_units"].sum()) if not future.empty else 0.0,
        "projected_revenue_total": float(future["predicted_revenue"].sum()) if not future.empty else 0.0,
        "daily_breakdown": future[["date", "predicted_units", "predicted_revenue", "confidence_lower", "confidence_upper"]].to_dict(orient="records")
    }


@app.get("/inventory/alerts")
def get_inventory_alerts(min_risk_score: float = 50.0):
    """Returns products and stores facing imminent stockouts."""
    inv_path = PROCESSED_DIR / "inventory_analysis.csv"
    if not inv_path.exists():
        raise HTTPException(status_code=404, detail="Inventory analysis not available.")

    df_inv = pd.read_csv(inv_path)
    risk_col = "stockout_risk_score" if "stockout_risk_score" in df_inv.columns else "reorder_needed"
    
    if "stockout_risk_score" in df_inv.columns:
        alerts = df_inv[df_inv["stockout_risk_score"] >= min_risk_score].sort_values("stockout_risk_score", ascending=False)
    else:
        alerts = df_inv[df_inv["reorder_needed"] == 1]

    cols_to_return = [c for c in ["store_name", "product_name", "category", "current_stock", "safety_stock", "reorder_point", "stock_status", "recommended_reorder_qty", "stockout_risk_score"] if c in df_inv.columns]
    
    return {
        "total_at_risk_skus": len(alerts),
        "alerts": alerts[cols_to_return].head(50).to_dict(orient="records")
    }


@app.get("/mlops/drift")
def get_drift_status():
    """Returns data drift monitoring metrics."""
    from src.mlops.drift_monitor import DriftMonitor
    monitor = DriftMonitor()
    report = monitor.evaluate_drift("units_sold")
    return report


