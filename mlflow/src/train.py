from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "avaliacoes.csv"
METRICS_PATH = PROJECT_DIR / "metrics.json"
FEATURES = ["study_hours", "attendance_pct", "previous_grade"]
TARGET = "final_grade"


def get_git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=PROJECT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "uncommitted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina um regressor e registra a execução no MLflow."
    )
    parser.add_argument("--n-estimators", type=int, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--test-size", type=float, required=True)
    parser.add_argument("--random-state", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }
    dataset_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "http://localhost:5001"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("avaliacoes-academicas")

    run_name = f"rf-n{args.n_estimators}-depth{args.max_depth}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "n_estimators": args.n_estimators,
                "max_depth": args.max_depth,
                "test_size": args.test_size,
                "random_state": args.random_state,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "dataset_sha256": dataset_sha256,
                "git_commit": get_git_commit(),
                "dataset": "avaliacoes-academicas-sintetico",
            }
        )
        mlflow.sklearn.log_model(
            model,
            name="modelo",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            input_example=X_train.head(3),
        )
        run_id = run.info.run_id

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"MLflow run: {run_id}")
    print(f"MLflow: {tracking_uri}")
    print(f"SHA-256 do CSV: {dataset_sha256}")
    print(f"Métricas: {json.dumps(metrics, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
