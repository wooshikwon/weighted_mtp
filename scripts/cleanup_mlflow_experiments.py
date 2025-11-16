"""MLflow EC2 서버의 불필요한 experiment 정리 스크립트"""

import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from rich.console import Console

console = Console()

# .env 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# MLflow 설정
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")
ec2_host = "13.50.240.176"
tracking_uri = f"http://{username}:{password}@{ec2_host}"

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)

console.print("[bold]MLflow Experiment 정리[/bold]\n")

# 전체 Experiment 목록
experiments = mlflow.search_experiments()
console.print(f"[cyan]현재 Experiment 개수: {len(experiments)}[/cyan]\n")

# 삭제 대상 experiment
experiments_to_delete = [
    "test-auth-experiment",
    "test-experiment",
    "weighted-mtp/connection-test",
]

# 보존 대상 experiment (삭제하지 않음)
experiments_to_keep = [
    "Default",
    "wmtp/vessl",
    "wmtp/local_distributed_test",
    "wmtp/production",
    "wmtp/local_test",
    "weighted-mtp/production",  # 새로 생성될 메인 experiment
]

console.print("[bold]Experiment 목록:[/bold]")
for exp in experiments:
    if exp.name in experiments_to_delete:
        console.print(f"  [red]❌ {exp.name} (ID: {exp.experiment_id}) - 삭제 예정[/red]")
    elif exp.name in experiments_to_keep:
        console.print(f"  [green]✓ {exp.name} (ID: {exp.experiment_id}) - 보존[/green]")
    else:
        console.print(f"  [yellow]? {exp.name} (ID: {exp.experiment_id}) - 보존 (미지정)[/yellow]")

console.print()

# 삭제 실행
for exp_name in experiments_to_delete:
    try:
        exp = client.get_experiment_by_name(exp_name)
        if exp:
            client.delete_experiment(exp.experiment_id)
            console.print(f"[green]✓ 삭제 완료: {exp_name} (ID: {exp.experiment_id})[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠ {exp_name} 삭제 실패: {e}[/yellow]")

# 최종 상태
console.print("\n[bold]최종 Experiment 목록:[/bold]")
experiments = mlflow.search_experiments()
for exp in experiments:
    console.print(f"  - {exp.name} (ID: {exp.experiment_id})")

console.print(f"\n[green]정리 완료! 남은 Experiment: {len(experiments)}개[/green]")
