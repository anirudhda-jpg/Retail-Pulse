# RetailPulse --- AI-Powered Customer Analytics & Demand Forecasting

RetailPulse is a data-driven retail analytics dashboard built with
**Python** and **Streamlit**. It brings together sales performance,
customer segmentation, demand forecasting, and inventory analysis in one
interactive application.

## Project Overview

RetailPulse is designed to help retail teams explore historical business
data and make more informed decisions. The dashboard presents processed
datasets through interactive analytics pages.

### Main capabilities

-   **Sales analytics** --- explore sales trends and store-level
    performance.
-   **Customer segmentation** --- inspect customer groups using the
    prepared RFM dataset.
-   **Demand forecasting** --- review baseline demand predictions
    alongside historical sales.
-   **Inventory optimization** --- explore inventory analysis and
    stock-related indicators.
-   **Project overview** --- view a consolidated summary of the project
    and its analytics.

> The dashboard uses the processed CSV files committed in
> `data/processed`. The large raw transaction dataset is intentionally
> not included in this repository.

## Repository Structure

``` text
Retail-Pulse/
├── dashboard/
│   └── app.py
├── data/
│   └── processed/
│       ├── baseline_predictions.csv
│       ├── customer_rfm.csv
│       ├── daily_sales.csv
│       ├── inventory_analysis.csv
│       ├── sku_daily_sales.csv
│       └── store_daily_sales.csv
├── src/
│   ├── api/
│   │   └── app_api.py
│   ├── etl/
│   │   └── pipeline.py
│   ├── mlops/
│   │   ├── drift_monitor.py
│   │   └── tracker.py
│   └── models/
│       ├── churn_predictor.py
│       ├── customer_segmentation.py
│       ├── demand_forecaster.py
│       └── inventory_optimizer.py
├── requirements.txt
└── README.md
```

Some optional development folders and files (for example
Docker/Kubernetes configuration, tests, and local run scripts) may also
exist in the repository. They are not required to launch the hosted
Streamlit dashboard.

## Technology Stack

-   Python
-   Streamlit
-   Pandas and NumPy
-   Scikit-learn
-   Prophet
-   Matplotlib and Seaborn
-   FastAPI and Uvicorn for the API component

The exact packages used by the project are listed in `requirements.txt`.

## Run Locally

### 1. Clone the repository

``` bash
git clone https://github.com/anirudhda-jpg/Retail-Pulse.git
cd Retail-Pulse
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

``` cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the Streamlit dashboard

Run this command from the repository root:

``` bash
streamlit run dashboard/app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`, where
the dashboard can be opened in a browser.

## Deploy on Streamlit Community Cloud

1.  Push this repository to GitHub.

2.  Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).

3.  Select **Create app** and choose the `anirudhda-jpg/Retail-Pulse`
    repository.

4.  Set the branch to the branch containing the project files (commonly
    `main`).

5.  Set the **Main file path** to:

    ``` text
    dashboard/app.py
    ```

6.  Deploy the app.

Keep `requirements.txt` in the repository root so Streamlit Community
Cloud can install the Python dependencies.

## Data Files

The dashboard reads the prepared datasets from:

``` text
data/processed/
```

The current repository includes these processed files:

  File                         Purpose
  ---------------------------- --------------------------------
  `daily_sales.csv`            Daily sales metrics
  `baseline_predictions.csv`   Baseline forecasting results
  `customer_rfm.csv`           Customer RFM/segmentation data
  `inventory_analysis.csv`     Inventory analysis
  `store_daily_sales.csv`      Store-level daily sales
  `sku_daily_sales.csv`        SKU-level daily sales

The dashboard expects the files to remain in this folder. If you move or
rename them, update the corresponding paths in `dashboard/app.py`.

## Important Notes

-   Do not commit `.venv/`, `__pycache__/`, `.pyc` files, or local
    environment folders.
-   Do not upload the very large raw `sales_transactions.csv` file to
    GitHub for this dashboard deployment. Use the processed datasets
    required by the app.
-   Keep file and folder names consistent with the paths used in the
    application.
-   The Streamlit deployment entry point is `dashboard/app.py`.
-   `run_pipeline.py`, batch scripts, Docker, Kubernetes, and tests are
    optional for the basic Streamlit dashboard deployment.

## Troubleshooting

**The app says a data file is missing** - Confirm that the required CSV
is present in `data/processed/`. - Check spelling and capitalization of
the filename. - Confirm that the repository contains the
`data/processed/` folder.

**A package cannot be imported** - Confirm that `requirements.txt` is in
the repository root. - Add any genuinely required missing package to
`requirements.txt`, then commit the change and redeploy.

**The app does not start on Streamlit Community Cloud** - Confirm the
main file path is exactly `dashboard/app.py`. - Review the deployment
logs for the first Python error. - Check that the app does not rely on a
local computer path such as `D:/...`.

## Project Status

This repository contains the Streamlit dashboard, its processed input
datasets, and supporting source modules for the API, ETL, MLOps, and
model components.

------------------------------------------------------------------------

**Repository:**
[github.com/anirudhda-jpg/Retail-Pulse](https://github.com/anirudhda-jpg/Retail-Pulse)
