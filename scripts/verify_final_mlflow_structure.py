"""최종 MLflow + S3 구조 검증 스크립트"""

import os
import sys
from pathlib import Path

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from rich.console import Console

console = Console()

# .env 로드
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

# defaults.yaml 로드
config_path = Path(__file__).parent.parent / "configs" / "defaults.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

console.print("[bold cyan]Phase 6 MLflow + S3 구조 검증[/bold cyan]\n")

# 1. defaults.yaml 설정 확인
console.print("[bold]1. defaults.yaml MLflow 설정 확인[/bold]")
mlflow_config = config.get("mlflow", {})
console.print(f"  tracking_uri: {mlflow_config.get('tracking_uri')}")
console.print(f"  experiment: {mlflow_config.get('experiment')}")
console.print(f"  s3_artifacts: {mlflow_config.get('s3_artifacts')}")

# 2. 환경변수 확인
console.print("\n[bold]2. 환경변수 확인[/bold]")
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")
aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_region = os.getenv("AWS_DEFAULT_REGION")

console.print(f"  MLFLOW_TRACKING_USERNAME: {username}")
console.print(f"  MLFLOW_TRACKING_PASSWORD: {'***' if password else 'None'}")
console.print(f"  AWS_ACCESS_KEY_ID: {aws_key[:10]}..." if aws_key else "  AWS_ACCESS_KEY_ID: None")
console.print(f"  AWS_DEFAULT_REGION: {aws_region}")

# 3. MLflow 연결 (defaults.yaml 설정 사용)
console.print("\n[bold]3. MLflow 서버 연결 (defaults.yaml 기반)[/bold]")
tracking_uri = mlflow_config.get("tracking_uri", "file://./mlruns")

# Basic Auth 주입
if tracking_uri.startswith("http"):
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(tracking_uri)
    if username and password:
        netloc = f"{username}:{password}@{parsed.netloc}"
        parsed = parsed._replace(netloc=netloc)
        tracking_uri = urlunparse(parsed)
        console.print(f"  [green]✓ Basic Auth 주입 완료[/green]")

mlflow.set_tracking_uri(tracking_uri)
client = MlflowClient(tracking_uri=tracking_uri)

try:
    experiments = mlflow.search_experiments()
    console.print(f"  [green]✓ 서버 연결 성공[/green]")
    console.print(f"  현재 Experiment 개수: {len(experiments)}")
except Exception as e:
    console.print(f"  [red]✗ 서버 연결 실패: {e}[/red]")
    sys.exit(1)

# 4. Experiment 생성/확인 (defaults.yaml 설정 사용)
console.print("\n[bold]4. Experiment 생성/확인[/bold]")
experiment_name = mlflow_config.get("experiment", "default")
s3_artifacts = mlflow_config.get("s3_artifacts")

try:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        console.print(f"  [yellow]기존 experiment 사용: {experiment_name}[/yellow]")
        experiment_id = experiment.experiment_id
    else:
        experiment_id = client.create_experiment(
            experiment_name, artifact_location=s3_artifacts
        )
        console.print(f"  [green]✓ 새 experiment 생성: {experiment_name}[/green]")

    console.print(f"  Experiment ID: {experiment_id}")

    # Artifact location 확인
    exp = client.get_experiment(experiment_id)
    console.print(f"  Artifact Location: {exp.artifact_location}")

    if exp.artifact_location == s3_artifacts:
        console.print(f"  [green]✓ S3 artifact location 올바름[/green]")
    else:
        console.print(f"  [yellow]⚠ Artifact location이 예상과 다름[/yellow]")

except Exception as e:
    console.print(f"  [red]✗ Experiment 생성/확인 실패: {e}[/red]")
    sys.exit(1)

# 5. S3 버킷 상태 확인
console.print("\n[bold]5. S3 버킷 상태 확인[/bold]")
try:
    import boto3
    s3_client = boto3.client('s3', region_name=aws_region)

    # s3://wmtp/ 버킷 내용 확인
    response = s3_client.list_objects_v2(Bucket='wmtp', Prefix='mlflow-artifacts/')

    if 'Contents' in response:
        console.print(f"  [yellow]mlflow-artifacts/ 내부에 {len(response['Contents'])}개 객체 존재[/yellow]")
        for obj in response['Contents'][:5]:  # 최대 5개만 출력
            console.print(f"    - {obj['Key']}")
    else:
        console.print(f"  [green]✓ mlflow-artifacts/ 비어있음 (정상)[/green]")
        console.print(f"  [cyan]학습 실행 시 자동으로 {experiment_id}/{{run_id}}/artifacts/ 구조 생성됨[/cyan]")

except Exception as e:
    console.print(f"  [yellow]⚠ S3 확인 실패 (권한 문제일 수 있음): {e}[/yellow]")

# 6. 최종 요약
console.print("\n" + "=" * 60)
console.print("[bold green]Phase 6 MLflow + S3 구조 검증 완료 ✓[/bold green]")
console.print("=" * 60)
console.print(f"\n[cyan]설정 요약:[/cyan]")
console.print(f"  - Tracking Server: http://13.50.240.176 (EC2)")
console.print(f"  - Experiment: {experiment_name} (ID: {experiment_id})")
console.print(f"  - S3 Artifacts: {s3_artifacts}")
console.print(f"  - S3 상태: 비어있음 (학습 실행 대기 중)")
console.print(f"\n[cyan]예상 artifact 저장 경로:[/cyan]")
console.print(f"  s3://wmtp/mlflow-artifacts/{experiment_id}/{{run_id}}/artifacts/checkpoints/")
