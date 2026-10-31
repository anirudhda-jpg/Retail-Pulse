"""
RetailPulse - Inventory Optimization & Supply Chain Engine
Calculates dynamic safety stock, reorder points (ROP), Economic Order Quantity (EOQ),
and prioritizes automated replenishment purchase orders to eliminate stockouts.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


class InventoryOptimizer:
    """Enterprise Supply Chain & Inventory Optimization Engine."""

    def __init__(self, data_path: Path = None):
        self.data_path = data_path or (PROCESSED_DIR / "inventory_analysis.csv")
        self.df = None

    def load_data(self):
        """Loads inventory data."""
        self.df = pd.read_csv(self.data_path)
        print(f"[InventoryOptimizer] Loaded {len(self.df)} inventory records.")
        return self.df

    def optimize_inventory(self, service_level_z: float = 1.65, holding_cost_pct: float = 0.20, order_cost_fixed: float = 250.0):
        """
        Executes scientific inventory optimization calculations:
        - Safety Stock: SS = Z * std_demand * sqrt(lead_time)
        - Reorder Point: ROP = (daily_run_rate * lead_time) + SS
        - Economic Order Quantity: EOQ = sqrt((2 * annual_demand * order_cost) / unit_holding_cost)
        - Stockout Probability & Risk Score (0-100)
        """
        if self.df is None:
            self.load_data()

        df = self.df.copy()

        # Demand variance assumption (std dev ~ 35% of daily run rate)
        demand_std = np.maximum(0.5, df["daily_run_rate"] * 0.35)
        lead_time = df["lead_time_days"]

        # 1. Scientific Dynamic Safety Stock
        df["optimal_safety_stock"] = np.ceil(service_level_z * demand_std * np.sqrt(lead_time)).astype(int)

        # 2. Dynamic Reorder Point
        df["optimal_reorder_point"] = np.ceil((df["daily_run_rate"] * lead_time) + df["optimal_safety_stock"]).astype(int)

        # 3. Annual Demand & Economic Order Quantity (EOQ)
        annual_demand = np.maximum(10, df["daily_run_rate"] * 365)
        unit_holding_cost = np.maximum(5.0, df["unit_cost"] * holding_cost_pct)
        df["economic_order_qty"] = np.ceil(np.sqrt((2 * annual_demand * order_cost_fixed) / unit_holding_cost)).astype(int)

        # 4. Stockout Risk Score (0 to 100)
        # Scaled based on operational safety stock and reorder point thresholds
        rop_target = np.maximum(1, df["reorder_point"].values)
        ss_target = np.maximum(1, df["safety_stock"].values)
        curr = df["current_stock"].values

        # Dynamic continuous score: 100 at 0 stock, 80 at SS, 50 at ROP, 0 at >= 2x ROP
        risk_score = np.where(
            curr <= ss_target,
            80.0 + 20.0 * (1.0 - (curr / ss_target)),
            np.where(
                curr <= rop_target,
                50.0 + 30.0 * (1.0 - ((curr - ss_target) / np.maximum(1, rop_target - ss_target))),
                np.maximum(0.0, 50.0 * (1.0 - ((curr - rop_target) / rop_target)))
            )
        )
        df["stockout_risk_score"] = np.clip(np.round(risk_score, 1), 0.0, 100.0)

        # 5. Replenishment Urgency Priority
        def assign_priority(row):
            status = row["stock_status"]
            score = row["stockout_risk_score"]
            if "Critical" in status or score >= 80:
                return "P1 - Critical (Immediate PO Required)"
            elif "Low" in status or score >= 50:
                return "P2 - High (Order within 48h)"
            elif score >= 25:
                return "P3 - Moderate (Scheduled Batch Order)"
            else:
                return "P4 - Normal / Healthy Stock"

        df["reorder_priority"] = df.apply(assign_priority, axis=1)

        # Capital required for replenishment
        df["reorder_capital_needed"] = np.round(df["recommended_reorder_qty"] * df["unit_cost"], 2)

        self.df = df
        self.df.to_csv(self.data_path, index=False)
        print(f"[InventoryOptimizer] Completed inventory optimization. Saved to {self.data_path}")
        return self.df


if __name__ == "__main__":
    optimizer = InventoryOptimizer()
    optimizer.optimize_inventory()
