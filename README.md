# RetailPulse ⚡ – AI-Powered Customer Analytics & Demand Forecasting Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)
[![FastAPI REST](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Docker & K8s Ready](https://img.shields.io/badge/Deployment-Docker%20%7C%20Kubernetes-blueviolet.svg)](https://kubernetes.io/)

RetailPulse is an enterprise-grade Data Science & Analytics platform designed to help retail enterprises make proactive, data-driven decisions using machine learning and predictive analytics. The platform integrates omni-channel sales, customer behavioral records, and supply chain inventory data to deliver **demand forecasting**, **customer segmentation**, **churn prediction**, and **scientific inventory replenishment**.

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data Layer
        RAW[Raw Enterprise Data\nTransactions, Inventory, Customers] --> ETL[Automated ETL Pipeline\nValidation & Feature Engineering]
        ETL --> PROC[Processed Datasets\nDaily Sales, RFM, Stock Status]
    end

    subgraph AI / ML Engines
        PROC --> FORECAST[Demand Forecasting Engine\nProphet | GBR | RF | Sequence Model]
        PROC --> SEGMENT[Customer Intelligence\nRFM Scoring | K-Means | CLV]
        PROC --> CHURN[Predictive Churn Engine\nRisk Scoring | SHAP Drivers | Playbooks]
        PROC --> INV[Inventory Optimizer\nSafety Stock | ROP | EOQ | POs]
    end

    subgraph MLOps & Governance
        FORECAST --> MLOPS[MLflow Experiment Tracker\nLeaderboard & Registry]
        PROC --> DRIFT[Data Drift Monitor\nPSI & KS-Test Alerts]
    end

    subgraph Serving & UI
        FORECAST --> DASHBOARD[Streamlit Executive Dashboard\nInteractive Visualizations]
        SEGMENT --> DASHBOARD
        CHURN --> DASHBOARD
        INV --> DASHBOARD
        DRIFT --> DASHBOARD
        FORECAST --> API[FastAPI Microservice\nReal-time Inference Endpoints]
        CHURN --> API
    end
```

---

## 🌟 Key Capabilities

### 1. 🔮 AI-Powered Demand Forecasting
- **Multi-Model Suite**: Incorporates Gradient Boosting Regressors, Random Forest, Recurrent Neural Sequence Forecasters, and Prophet baselines.
- **Ensemble Forecasting**: Combines tree-based non-linear dynamics with autoregressive sequential patterns.
- **Confidence Intervals**: Computes 95% confidence bands to account for holiday demand volatility and supply shocks.
- **Scenario Simulator**: Interactive What-If tool modeling price elasticity and marketing lift.
- **Evaluation**: Validated on MAE, RMSE, MAPE, and WAPE.

### 2. 👥 Customer Intelligence & RFM Studio
- **Dynamic RFM Quantiles**: Automated Recency, Frequency, and Monetary scoring.
- **K-Means Clustering**: 5 behavioral personas (VIP Champions, High-Value Loyalists, At-Risk High Spenders, Recent Starters, Hibernating).
- **Customer Lifetime Value (CLV)**: Combines historical gross margins with predictive future value horizons.

### 3. ⚠️ Churn Prediction & Retention Engine
- **Stratified Risk Scoring**: Machine learning classifier categorizing customers into Critical, Moderate, and Low churn risk tiers.
- **Explainability**: Quantifies attrition drivers (purchase cadence, recency thresholds, spending drop).
- **Automated Retention Playbooks**: Triggers targeted marketing actions (VIP concierge, personalized win-back vouchers).

### 4. 📦 Inventory Optimization & Supply Chain Cockpit
- **Scientific Safety Stock**: $SS = Z \times \sigma_d \times \sqrt{L}$ (ensuring 95%+ service levels).
- **Dynamic Reorder Point**: $ROP = (d \times L) + SS$.
- **Economic Order Quantity (EOQ)**: Calculates optimal lot sizes balancing holding and ordering costs.
- **Automated Purchase Orders**: Generates prioritized POs (P1 Critical, P2 High) with one-click CSV export.

### 5. 📈 MLOps & Production Governance
- **Experiment Tracker**: Unified MLflow-compatible logger for parameters, metrics, and models.
- **Covariate Shift & Data Drift**: Tracks Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests to alert when incoming traffic drifts out of spec ($PSI \ge 0.25$).

---

## 📁 Repository Structure

```
RetailPulse/
├── data/
│   ├── raw/                       # Raw source tables (transactions, inventory, customers)
│   ├── processed/                 # Curated tables for analytics and modeling
│   └── generate_retail_data.py    # Synthetic enterprise retail data generator
│
├── src/
│   ├── etl/
│   │   └── pipeline.py            # Data ingestion, cleaning, validation, and feature transforms
│   ├── models/
│   │   ├── demand_forecaster.py   # Multi-horizon demand forecasters & metrics
│   │   ├── customer_segmentation.py # RFM Scoring, K-Means clustering & CLV
│   │   ├── churn_predictor.py     # Churn classifier, risk tiers & playbooks
│   │   └── inventory_optimizer.py # Safety stock, ROP, EOQ & stockout risk scoring
│   ├── mlops/
│   │   ├── tracker.py             # MLflow-compatible experiment tracker
│   │   └── drift_monitor.py       # PSI & KS-test data drift detector
│   └── api/
│       └── app_api.py             # FastAPI REST endpoints for real-time serving
│
├── dashboard/
│   └── app.py                     # High-fidelity Streamlit web application
│
├── docker/
│   ├── Dockerfile                 # Production multi-service Docker image
│   └── docker-compose.yml         # Containerized Dashboard & API services
│
├── k8s/
│   ├── deployment.yaml            # Kubernetes deployment with probes and resource limits
│   └── service.yaml               # LoadBalancer service & Horizontal Pod Autoscaler (HPA)
│
├── .github/workflows/
│   └── ci.yml                     # GitHub Actions CI pipeline
│
├── tests/                         # Automated unit tests for ETL, models & inventory
├── run_pipeline.py                # One-click master orchestrator script
├── requirements.txt               # Dependencies specification
└── README.md                      # Platform documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install requirements:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Analytics Pipeline
Execute data generation, ETL, model training, and predictions:
```bash
python run_pipeline.py
```

### 3. Launch the Interactive Web Dashboard
Run the Streamlit application:
```bash
streamlit run dashboard/app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run the FastAPI REST Server
Start the inference microservice:
```bash
uvicorn src.api.app_api:app --reload --port 8000
```
Interactive Swagger documentation will be available at `http://localhost:8000/docs`.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System healthcheck and dataset availability status |
| `POST` | `/forecast/demand` | Get multi-day demand and revenue forecasts |
| `POST` | `/customer/churn` | Predict churn probability and retrieve personalized retention playbook |
| `GET` | `/inventory/alerts` | Query current stockout risks and suggested replenishment POs |
| `GET` | `/mlops/drift` | Check live feature drift scores (PSI & KS p-values) |

---

## 🐳 Containerization & Kubernetes

### Docker Compose
Run both the Streamlit Dashboard and FastAPI services together:
```bash
cd docker
docker-compose up --build
```

### Kubernetes Deployment
Deploy to a cluster:
```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```
Autoscaling scales from 2 to 10 replicas when CPU utilization exceeds 70%.

---

## 🧪 Running Unit Tests
Execute the test suite with pytest:
```bash
pytest tests/ -v
```
