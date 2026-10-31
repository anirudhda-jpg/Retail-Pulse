"""
RetailPulse - Enterprise ETL & Feature Engineering Pipeline
Optimized for the 10-Million Transaction Synthetic Retail Dataset (Clean & Contaminated).
Performs fast streaming chunk ingestion, schema mapping, relational joins,
and multi-dimensional aggregations.
"""

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Default to the user's provided dataset
DEFAULT_ARCHIVE_CONTAMINATED = Path("C:/Users/aniru/Downloads/archive/retail_contaminated_dataset")
DEFAULT_ARCHIVE_CLEAN = Path("C:/Users/aniru/Downloads/archive/retail_clean_dataset")


class RetailETLPipeline:
    """Enterprise-Scale Streaming ETL Pipeline for 10M Row Datasets."""

    def __init__(self, source_dir: Path = None, output_dir: Path = PROCESSED_DIR):
        if source_dir is None:
            if DEFAULT_ARCHIVE_CONTAMINATED.exists():
                source_dir = DEFAULT_ARCHIVE_CONTAMINATED
            elif DEFAULT_ARCHIVE_CLEAN.exists():
                source_dir = DEFAULT_ARCHIVE_CLEAN
            else:
                source_dir = PROJECT_ROOT / "data" / "raw"
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        print(f"[ETL] Source dataset path: {self.source_dir}")

    def load_metadata_tables(self):
        """Loads store_master, sku_master, customer_master, inventory_snapshot, and flags."""
        print("[ETL] Loading metadata tables (stores, SKUs, customers, inventory, flags)...")
        
        # Store master
        stores_file = self.source_dir / "store_master.csv"
        df_stores = pd.read_csv(stores_file) if stores_file.exists() else pd.DataFrame()

        # SKU master
        skus_file = self.source_dir / "sku_master.csv"
        df_skus = pd.read_csv(skus_file) if skus_file.exists() else pd.DataFrame()

        # Customer master (maps cust_id -> customer_id)
        cust_file = self.source_dir / "customer_master.csv"
        df_customers = pd.read_csv(cust_file) if cust_file.exists() else pd.DataFrame()
        if "cust_id" in df_customers.columns:
            df_customers = df_customers.rename(columns={"cust_id": "customer_id"})

        # Inventory snapshot (maps stock_on_hand -> current_stock)
        inv_file = self.source_dir / "inventory_snapshot.csv"
        df_inv = pd.read_csv(inv_file) if inv_file.exists() else pd.DataFrame()
        if "stock_on_hand" in df_inv.columns:
            df_inv = df_inv.rename(columns={"stock_on_hand": "current_stock"})

        # Ground-truth anomaly flags (if contaminated)
        flags_file = self.source_dir / "sku_inventory_flags.csv"
        df_flags = pd.read_csv(flags_file) if flags_file.exists() else pd.DataFrame()

        return df_stores, df_skus, df_customers, df_inv, df_flags

    def process_sales_streaming(self, chunksize: int = 500000):
        """
        Streams through 10 Million transaction rows in chunks to prevent memory overhead,
        aggregating daily sales, store sales, and customer RFM metrics simultaneously.
        """
        tx_file = self.source_dir / "sales_transactions.csv"
        if not tx_file.exists():
            raise FileNotFoundError(f"Missing sales transactions file at {tx_file}")

        print(f"[ETL] Streaming {tx_file.name} in chunks of {chunksize:,} rows...")
        t0 = time.time()

        daily_acc = {}        # date -> [units, revenue, tx_count]
        store_daily_acc = {}  # (date, store_id) -> [units, revenue, tx_count]
        
        # Customer RFM accumulators
        cust_last_date = {}
        cust_first_date = {}
        cust_tx_count = {}
        cust_units = {}
        cust_spend = {}

        total_rows = 0
        chunks = pd.read_csv(
            tx_file,
            chunksize=chunksize,
            usecols=["date", "store_id", "customer_id", "quantity", "total_value"],
            dtype={"date": str, "store_id": str, "customer_id": str, "quantity": int, "total_value": float}
        )

        for i, chunk in enumerate(chunks):
            total_rows += len(chunk)
            
            # 1. Daily Aggregations in Chunk
            daily_grp = chunk.groupby("date").agg(
                u=("quantity", "sum"),
                r=("total_value", "sum"),
                c=("quantity", "count")
            )
            for d, row in daily_grp.iterrows():
                if d in daily_acc:
                    daily_acc[d][0] += row["u"]
                    daily_acc[d][1] += row["r"]
                    daily_acc[d][2] += row["c"]
                else:
                    daily_acc[d] = [row["u"], row["r"], row["c"]]

            # 2. Store Daily Aggregations in Chunk
            store_grp = chunk.groupby(["date", "store_id"]).agg(
                u=("quantity", "sum"),
                r=("total_value", "sum"),
                c=("quantity", "count")
            )
            for key, row in store_grp.iterrows():
                if key in store_daily_acc:
                    store_daily_acc[key][0] += row["u"]
                    store_daily_acc[key][1] += row["r"]
                    store_daily_acc[key][2] += row["c"]
                else:
                    store_daily_acc[key] = [row["u"], row["r"], row["c"]]

            # 3. Customer RFM Accumulations in Chunk
            cust_grp = chunk.groupby("customer_id").agg(
                last_d=("date", "max"),
                first_d=("date", "min"),
                c=("quantity", "count"),
                u=("quantity", "sum"),
                r=("total_value", "sum")
            )
            for cid, row in cust_grp.iterrows():
                # Last date
                if cid not in cust_last_date or row["last_d"] > cust_last_date[cid]:
                    cust_last_date[cid] = row["last_d"]
                # First date
                if cid not in cust_first_date or row["first_d"] < cust_first_date[cid]:
                    cust_first_date[cid] = row["first_d"]
                # Counts, units, spend
                cust_tx_count[cid] = cust_tx_count.get(cid, 0) + row["c"]
                cust_units[cid] = cust_units.get(cid, 0) + row["u"]
                cust_spend[cid] = cust_spend.get(cid, 0.0) + row["r"]

            if (i + 1) % 4 == 0:
                print(f"[ETL] Processed {total_rows:,} rows ({time.time()-t0:.1f}s)...")

        print(f"[ETL] Streaming complete! {total_rows:,} transaction rows processed in {time.time()-t0:.1f}s.")

        # Build Daily Sales DataFrame
        daily_records = [
            {"date": d, "units_sold": vals[0], "revenue": round(vals[1], 2), "transaction_count": vals[2]}
            for d, vals in daily_acc.items()
        ]
        df_daily = pd.DataFrame(daily_records)
        df_daily["date"] = pd.to_datetime(df_daily["date"])
        df_daily = df_daily.sort_values("date").reset_index(drop=True)
        df_daily["average_order_value"] = (df_daily["revenue"] / df_daily["transaction_count"]).round(2)
        df_daily["revenue_7d_ma"] = df_daily["revenue"].rolling(7, min_periods=1).mean().round(2)
        df_daily["units_7d_ma"] = df_daily["units_sold"].rolling(7, min_periods=1).mean().round(2)

        # Build Store Daily DataFrame
        store_records = [
            {"date": k[0], "store_id": k[1], "units_sold": vals[0], "revenue": round(vals[1], 2), "transaction_count": vals[2]}
            for k, vals in store_daily_acc.items()
        ]
        df_store_daily = pd.DataFrame(store_records)
        df_store_daily["date"] = pd.to_datetime(df_store_daily["date"])
        df_store_daily = df_store_daily.sort_values(["date", "store_id"]).reset_index(drop=True)

        # Build Customer RFM Summary DataFrame
        cust_records = [
            {
                "customer_id": cid,
                "last_purchase_date": cust_last_date[cid],
                "first_purchase_date": cust_first_date[cid],
                "purchase_count": cust_tx_count[cid],
                "units_bought": cust_units[cid],
                "total_spent": round(cust_spend[cid], 2)
            }
            for cid in cust_last_date
        ]
        df_cust_rfm = pd.DataFrame(cust_records)

        return df_daily, df_store_daily, df_cust_rfm

    def build_customer_analytics(self, df_cust_rfm: pd.DataFrame, df_customers: pd.DataFrame) -> pd.DataFrame:
        """Enriches customer RFM with demographics, quantiles, and personas."""
        print("[ETL] Enriching customer RFM and segmentation features...")
        merged = pd.merge(df_customers, df_cust_rfm, on="customer_id", how="left")
        
        merged["last_purchase_date"] = pd.to_datetime(merged["last_purchase_date"])
        merged["first_purchase_date"] = pd.to_datetime(merged["first_purchase_date"])
        max_date = merged["last_purchase_date"].max()

        merged["recency_days"] = (max_date - merged["last_purchase_date"]).dt.days.fillna(365).astype(int)
        merged["customer_tenure_days"] = (max_date - merged["first_purchase_date"]).dt.days.fillna(0).astype(int)
        merged["purchase_count"] = merged["purchase_count"].fillna(0).astype(int)
        merged["total_spent"] = merged["total_spent"].fillna(0.0).round(2)
        merged["avg_order_value"] = np.where(
            merged["purchase_count"] > 0,
            (merged["total_spent"] / merged["purchase_count"]).round(2),
            0.0
        )

        # Churn indicator (> 90 days since last purchase)
        merged["is_churned"] = (merged["recency_days"] > 90).astype(int)

        # Standard RFM Quantiles
        try:
            merged["r_score"] = pd.qcut(merged["recency_days"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
            merged["f_score"] = pd.qcut(merged["purchase_count"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
            merged["m_score"] = pd.qcut(merged["total_spent"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        except Exception:
            merged["r_score"] = 3
            merged["f_score"] = 3
            merged["m_score"] = 3

        def assign_segment(row):
            r, f, m = row["r_score"], row["f_score"], row["m_score"]
            if r >= 4 and f >= 4 and m >= 4:
                return "Champions"
            elif r >= 3 and f >= 3:
                return "Loyal Customers"
            elif r >= 4 and f <= 2:
                return "Promising New"
            elif r <= 2 and f >= 3:
                return "At Risk"
            elif r <= 2 and f <= 2:
                return "Hibernating"
            else:
                return "Potential Loyalists"

        merged["segment"] = merged.apply(assign_segment, axis=1)

        # Standardize loyalty tier name
        if "loyalty_segment" in merged.columns:
            merged["loyalty_tier"] = merged["loyalty_segment"]

        output_path = self.output_dir / "customer_rfm.csv"
        merged.to_csv(output_path, index=False)
        print(f"[ETL] Saved customer RFM dataset -> {output_path} ({len(merged)} customers)")
        return merged

    def build_inventory_analysis(self, df_inv: pd.DataFrame, df_skus: pd.DataFrame, df_stores: pd.DataFrame, df_flags: pd.DataFrame) -> pd.DataFrame:
        """Builds inventory risk matrix with ground truth anomaly matching."""
        print("[ETL] Building inventory analysis with ground truth flags...")
        
        # Merge SKU attributes
        inv = pd.merge(df_inv, df_skus, on="sku_id", how="left")
        if "sku_name" in inv.columns and "product_name" not in inv.columns:
            inv["product_name"] = inv["sku_name"]
        
        # Merge Store attributes
        if not df_stores.empty and "store_id" in df_stores.columns:
            inv = pd.merge(inv, df_stores[["store_id", "store_name", "city", "store_type"]], on="store_id", how="left")

        # Merge Anomaly flags if available
        if not df_flags.empty and "sku_id" in df_flags.columns:
            inv = pd.merge(inv, df_flags[["sku_id", "flag", "affected_stores", "notes"]], on="sku_id", how="left")
        else:
            inv["flag"] = np.nan
            inv["affected_stores"] = np.nan
            inv["notes"] = np.nan

        # Lead time assumption (3 to 7 days)
        inv["lead_time_days"] = np.random.choice([3, 5, 7], size=len(inv), p=[0.4, 0.4, 0.2])

        # Daily run rate proxy based on reorder point
        inv["daily_run_rate"] = np.maximum(0.5, (inv["reorder_point"] / 10.0)).round(2)
        inv["days_remaining"] = np.where(
            inv["daily_run_rate"] > 0,
            (inv["current_stock"] / inv["daily_run_rate"]).round(1),
            999.0
        )

        # Classify status
        def classify_stock(row):
            flag = str(row.get("flag", "")).strip().upper()
            curr = row["current_stock"]
            ss = row["safety_stock"]
            rop = row["reorder_point"]

            if flag == "STOCKOUT_RISK" or curr <= ss:
                return "Critical Stockout Risk"
            elif flag == "SLOW_MOVER" or curr > (rop * 3.5):
                return "Slow Mover / Dead Stock"
            elif curr <= rop:
                return "Low Stock - Reorder Needed"
            else:
                return "Adequate Stock"

        inv["stock_status"] = inv.apply(classify_stock, axis=1)
        inv["reorder_needed"] = inv["stock_status"].isin(["Critical Stockout Risk", "Low Stock - Reorder Needed"]).astype(int)

        # Recommended replenishment quantity
        inv["recommended_reorder_qty"] = np.where(
            inv["reorder_needed"] == 1,
            np.maximum(0, (inv["reorder_point"] * 2) - inv["current_stock"]),
            0
        )
        unit_cost = inv["cost_price"] if "cost_price" in inv.columns else inv["unit_price"] * 0.7
        inv["unit_cost"] = unit_cost.round(2)
        inv["reorder_capital_needed"] = (inv["recommended_reorder_qty"] * inv["unit_cost"]).round(2)

        # Continuous Stockout Risk Score (0-100)
        curr = inv["current_stock"].values
        ss = np.maximum(1, inv["safety_stock"].values)
        rop = np.maximum(1, inv["reorder_point"].values)
        
        risk = np.where(
            curr <= ss,
            80.0 + 20.0 * (1.0 - (curr / ss)),
            np.where(
                curr <= rop,
                50.0 + 30.0 * (1.0 - ((curr - ss) / np.maximum(1, rop - ss))),
                np.maximum(0.0, 50.0 * (1.0 - ((curr - rop) / rop)))
            )
        )
        inv["stockout_risk_score"] = np.clip(np.round(risk, 1), 0.0, 100.0)

        output_path = self.output_dir / "inventory_analysis.csv"
        inv.to_csv(output_path, index=False)
        print(f"[ETL] Saved inventory analysis -> {output_path} ({len(inv)} items)")
        return inv

    def run_all(self):
        """Executes full streaming ETL pipeline."""
        print("=" * 70)
        print("  ⚡ RETAILPULSE - STREAMING ETL PIPELINE (10M TRANSACTIONS)")
        print("=" * 70)
        t_start = time.time()

        df_stores, df_skus, df_customers, df_inv, df_flags = self.load_metadata_tables()
        df_daily, df_store_daily, df_cust_rfm = self.process_sales_streaming(chunksize=500000)

        # Save Daily & Store Daily
        daily_path = self.output_dir / "daily_sales.csv"
        df_daily.to_csv(daily_path, index=False)
        print(f"[ETL] Saved daily sales -> {daily_path} ({len(df_daily)} dates)")

        store_daily_path = self.output_dir / "store_daily_sales.csv"
        if not df_stores.empty and "store_id" in df_stores.columns:
            df_store_daily = pd.merge(df_store_daily, df_stores[["store_id", "store_name", "city"]], on="store_id", how="left")
        df_store_daily.to_csv(store_daily_path, index=False)
        print(f"[ETL] Saved store daily sales -> {store_daily_path} ({len(df_store_daily)} rows)")

        # Customer Analytics & RFM
        df_customer_rfm = self.build_customer_analytics(df_cust_rfm, df_customers)

        # Inventory Analysis & Anomaly Mapping
        df_inventory = self.build_inventory_analysis(df_inv, df_skus, df_stores, df_flags)

        print("\n" + "=" * 70)
        print(f"  ✅ ETL PIPELINE COMPLETED IN {time.time()-t_start:.1f} SECONDS!")
        print("=" * 70)

        return {
            "daily_sales": df_daily,
            "store_daily_sales": df_store_daily,
            "customer_rfm": df_customer_rfm,
            "inventory_analysis": df_inventory
        }


if __name__ == "__main__":
    pipeline = RetailETLPipeline()
    pipeline.run_all()
