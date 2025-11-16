"""EC2 MLflow 서버 연결 및 동작 검증 스크립트"""

import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from rich.console import Console

console = Console()

# .env 로드
env_path = Path(__file__).parent.parent / ".env"
if not env_path.exists():
    console.print(f"[red]Error: .env 파일이 없습니다: {env_path}[/red]")
    sys.exit(1)

load_dotenv(env_path)

# 환경변수 확인
required_env_vars = [
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    console.print(f"[red]Error: 필수 환경변수 누락: {', '.join(missing_vars)}[/red]")
    sys.exit(1)

console.print("[green]✓ 환경변수 로딩 완료[/green]")

# MLflow Tracking URI (Basic Auth 주입)
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")
ec2_host = "13.50.240.176"
tracking_uri = f"http://{username}:{password}@{ec2_host}"

console.print(f"[cyan]MLflow Tracking URI: http://{username}:***@{ec2_host}[/cyan]")

# MLflow 설정
mlflow.set_tracking_uri(tracking_uri)

try:
    # 1. 서버 연결 테스트
    console.print("\n[bold]1. EC2 MLflow 서버 연결 테스트[/bold]")
    experiments = mlflow.search_experiments()
    console.print(f"[green]✓ 서버 연결 성공[/green]")
    console.print(f"  현재 experiment 개수: {len(experiments)}")

    # 2. Experiment 목록 출력
    console.print("\n[bold]2. Experiment 목록[/bold]")
    for exp in experiments:
        console.print(f"  - {exp.name} (ID: {exp.experiment_id})")

    # 3. 테스트 Experiment 생성
    console.print("\n[bold]3. 테스트 Experiment 생성[/bold]")
    test_exp_name = "weighted-mtp/connection-test"
    s3_artifact_location = "s3://wmtp/mlflow-artifacts"

    # Experiment 생성 또는 로드
    try:
        experiment = mlflow.get_experiment_by_name(test_exp_name)
        if experiment:
            console.print(f"[yellow]기존 experiment 사용: {test_exp_name}[/yellow]")
            experiment_id = experiment.experiment_id
        else:
            experiment_id = mlflow.create_experiment(
                test_exp_name, artifact_location=s3_artifact_location
            )
            console.print(f"[green]✓ Experiment 생성 완료: {test_exp_name}[/green]")
    except Exception as e:
        console.print(f"[red]Experiment 생성 실패: {e}[/red]")
        raise

    # 4. 테스트 Run 생성
    console.print("\n[bold]4. 테스트 Run 생성[/bold]")
    mlflow.set_experiment(test_exp_name)

    with mlflow.start_run(run_name="connection-verification") as run:
        console.print(f"[green]✓ Run 시작: {run.info.run_id}[/green]")

        # Parameters 로깅
        mlflow.log_param("test_param", "test_value")
        mlflow.log_param("verification_time", "2025-01-16")

        # Metrics 로깅
        mlflow.log_metric("test_metric", 1.0)
        mlflow.log_metric("connection_score", 100.0)

        console.print("[green]✓ Parameters/Metrics 로깅 완료[/green]")

    console.print(f"[green]✓ Run 종료 완료[/green]")

    # 5. S3 Artifact Location 확인
    console.print("\n[bold]5. S3 Artifact Location 확인[/bold]")
    experiment = mlflow.get_experiment_by_name(test_exp_name)
    console.print(f"  Artifact Location: {experiment.artifact_location}")

    if experiment.artifact_location.startswith("s3://wmtp/mlflow-artifacts"):
        console.print("[green]✓ S3 artifact location 올바름[/green]")
    else:
        console.print(
            f"[yellow]Warning: Artifact location이 예상과 다름: {experiment.artifact_location}[/yellow]"
        )

    # 6. 최종 검증 성공
    console.print("\n[bold green]" + "=" * 60 + "[/bold green]")
    console.print("[bold green]EC2 MLflow 서버 검증 완료 ✓[/bold green]")
    console.print("[bold green]" + "=" * 60 + "[/bold green]")
    console.print(f"\n[cyan]MLflow UI: http://{ec2_host}[/cyan]")
    console.print(f"[cyan]Username: {username}[/cyan]")
    console.print(f"[cyan]S3 Bucket: s3://wmtp/mlflow-artifacts[/cyan]")

except Exception as e:
    console.print(f"\n[bold red]검증 실패: {e}[/bold red]")
    import traceback

    console.print(f"[red]{traceback.format_exc()}[/red]")
    sys.exit(1)
