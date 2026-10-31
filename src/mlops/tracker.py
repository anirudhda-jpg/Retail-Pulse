"""
RetailPulse - MLOps Experiment Tracking & Model Registry
Provides a unified MLflow-compatible experiment tracker. Works with native MLflow
if installed, or gracefully falls back to local structured JSON/SQLite experiment tracking.
"""

from pathlib import Path
from datetime import datetime
import json
import uuid

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MLRUNS_DIR = BASE_DIR / "mlruns_local"
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)


class ExperimentTracker:
    """MLOps Experiment Tracking & Model Registry Abstraction."""

    def __init__(self, experiment_name: str = "RetailPulse_Analytics"):
        self.experiment_name = experiment_name
        self.experiments_file = MLRUNS_DIR / "experiments.json"
        self.active_run = None
        self._has_mlflow = False

        try:
            import mlflow
            self._has_mlflow = True
            mlflow.set_experiment(experiment_name)
        except ImportError:
            self._has_mlflow = False

    def start_run(self, run_name: str):
        """Starts a new experiment run."""
        run_id = str(uuid.uuid4())[:8]
        self.active_run = {
            "run_id": run_id,
            "run_name": run_name,
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "status": "RUNNING",
            "parameters": {},
            "metrics": {},
            "artifacts": []
        }
        if self._has_mlflow:
            import mlflow
            mlflow.start_run(run_name=run_name)
        return run_id

    def log_param(self, key: str, value):
        """Logs a hyperparameter or configuration setting."""
        if self.active_run:
            self.active_run["parameters"][key] = value
        if self._has_mlflow:
            import mlflow
            mlflow.log_param(key, value)

    def log_params(self, params: dict):
        """Logs multiple parameters."""
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float):
        """Logs an evaluation metric."""
        val = round(float(value), 4)
        if self.active_run:
            self.active_run["metrics"][key] = val
        if self._has_mlflow:
            import mlflow
            mlflow.log_metric(key, val)

    def log_metrics(self, metrics: dict):
        """Logs multiple metrics."""
        for k, v in metrics.items():
            self.log_metric(k, v)

    def end_run(self, status: str = "FINISHED"):
        """Ends the active run and commits metadata."""
        if not self.active_run:
            return

        self.active_run["status"] = status
        self.active_run["end_time"] = datetime.now().isoformat()

        # Load existing registry
        history = []
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []

        history.append(self.active_run)
        with open(self.experiments_file, "w") as f:
            json.dump(history, f, indent=2)

        if self._has_mlflow:
            import mlflow
            mlflow.end_run()

        print(f"[MLOps] Run '{self.active_run['run_name']}' ({self.active_run['run_id']}) ended with status: {status}")
        self.active_run = None

    def get_runs_history(self):
        """Returns past experiment runs as a list of dicts."""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []


if __name__ == "__main__":
    tracker = ExperimentTracker()
    tracker.start_run("Demand_Forecasting_Ensemble_v1")
    tracker.log_params({"model_type": "Ensemble", "n_estimators": 150, "test_days": 45})
    tracker.log_metrics({"MAE": 12.4, "RMSE": 16.8, "WAPE_pct": 8.5})
    tracker.end_run()
