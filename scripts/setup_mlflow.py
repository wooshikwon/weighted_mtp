#!/usr/bin/env python3
"""MLflow Default Experiment 설정

서버에서 평가 실행 전 MLflow experiment를 초기화합니다.

Usage:
    uv run python scripts/setup_mlflow.py
"""

import mlflow


def main():
    mlflow.set_tracking_uri("file:///workspace/weighted_mtp/mlruns")
    mlflow.set_experiment("Default")

    experiment = mlflow.get_experiment_by_name("Default")
    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")
    print("MLflow setup complete")


if __name__ == "__main__":
    main()
