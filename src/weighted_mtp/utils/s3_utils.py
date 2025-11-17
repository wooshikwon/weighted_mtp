"""S3 checkpoint 관리 유틸리티

비동기 업로드 및 정리 기능 제공
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

# 전역 executor (파이프라인 초기화 시 생성)
s3_upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-upload")


def upload_to_s3_async(checkpoint_path: Path, mlflow_enabled: bool) -> None:
    """비동기로 S3에 checkpoint 업로드

    MLflow artifact store를 통해 S3에 업로드
    학습 루프를 블로킹하지 않음

    Args:
        checkpoint_path: 업로드할 checkpoint 경로
        mlflow_enabled: MLflow 사용 여부
    """
    if not mlflow_enabled:
        return

    try:
        import mlflow

        mlflow.log_artifact(str(checkpoint_path), "checkpoints")
        logger.info(f"S3 upload complete: {checkpoint_path.name}")
    except Exception as e:
        logger.error(f"S3 upload failed: {checkpoint_path.name} - {e}")


def cleanup_s3_checkpoints(
    experiment_id: str,
    run_id: str,
    save_total_limit: int,
) -> None:
    """S3에서 오래된 checkpoint 삭제

    MLflow artifact store (S3)에서 checkpoint_epoch_*.pt 파일만 정리
    checkpoint_best.pt와 checkpoint_final.pt는 유지

    Args:
        experiment_id: MLflow experiment ID
        run_id: MLflow run ID
        save_total_limit: 유지할 최대 개수
    """
    try:
        import boto3
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        s3 = boto3.client("s3")

        # S3에 있는 모든 checkpoint 목록 가져오기
        artifacts = client.list_artifacts(run_id, path="checkpoints")

        # checkpoint_epoch_*.pt 파일만 필터링
        epoch_checkpoints = [
            a for a in artifacts if a.path.startswith("checkpoints/checkpoint_epoch_")
        ]

        # 시간순 정렬 (파일명 기준)
        epoch_checkpoints.sort(key=lambda x: x.path)

        # 삭제할 파일 개수 계산
        n_to_delete = len(epoch_checkpoints) - save_total_limit

        if n_to_delete > 0:
            bucket = "wmtp"
            for artifact in epoch_checkpoints[:n_to_delete]:
                # boto3로 S3 파일 직접 삭제
                s3_key = f"mlflow-artifacts/{experiment_id}/{run_id}/artifacts/{artifact.path}"
                s3.delete_object(Bucket=bucket, Key=s3_key)
                logger.info(f"S3 checkpoint deleted: {artifact.path}")

    except Exception as e:
        logger.warning(f"S3 cleanup failed: {e}")


def shutdown_s3_executor() -> None:
    """모든 S3 업로드 완료 대기

    학습 종료 시 호출하여 모든 비동기 업로드 완료를 보장
    """
    s3_upload_executor.shutdown(wait=True)
    logger.info("All S3 uploads completed")
