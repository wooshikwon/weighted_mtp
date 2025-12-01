"""S3 스토리지 유틸리티

boto3를 사용한 직접 S3 업로드 및 동기화 기능 제공
MLflow 서버 없이 독립적으로 동작
"""

import logging
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# S3 설정
S3_BUCKET = "wmtp"

# 전역 executor (비동기 업로드용)
s3_upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-upload")


def upload_to_s3_async(checkpoint_path: Path, experiment_name: str) -> None:
    """체크포인트를 S3에 직접 업로드 (boto3 사용)

    임시 복사본을 사용하여 race condition 방지:
    - 원본 파일이 삭제되어도 업로드 성공 보장
    - cleanup_old_checkpoints와의 race condition 방지

    Args:
        checkpoint_path: 업로드할 checkpoint 경로
        experiment_name: 실험 이름 (S3 경로 구성에 사용)

    S3 저장 경로:
        s3://wmtp/checkpoints/{experiment_name}/{checkpoint_filename}
    """
    if not checkpoint_path.exists():
        logger.warning(f"S3 upload skipped (file not found): {checkpoint_path}")
        return

    tmp_dir = None
    try:
        # 임시 디렉터리에 복사 (race condition 방지)
        tmp_dir = tempfile.TemporaryDirectory(prefix="s3_upload_")
        tmp_path = Path(tmp_dir.name) / checkpoint_path.name
        shutil.copy2(checkpoint_path, tmp_path)

        # S3 업로드 (임시 복사본에서)
        s3 = boto3.client("s3")
        s3_key = f"checkpoints/{experiment_name}/{checkpoint_path.name}"

        s3.upload_file(str(tmp_path), S3_BUCKET, s3_key)
        logger.info(f"S3 upload complete: {checkpoint_path.name} -> s3://{S3_BUCKET}/{s3_key}")

    except ClientError as e:
        logger.error(f"S3 upload failed: {checkpoint_path.name} - {e}")
    except Exception as e:
        logger.error(f"S3 upload failed: {checkpoint_path.name} - {e}")
    finally:
        # 임시 디렉터리 정리
        if tmp_dir is not None:
            tmp_dir.cleanup()


def cleanup_s3_checkpoints(
    experiment_name: str,
    save_total_limit: int,
) -> None:
    """S3에서 오래된 checkpoint 삭제

    checkpoint_epoch_*.pt 파일만 정리
    checkpoint_best.pt와 checkpoint_final.pt는 유지

    Args:
        experiment_name: 실험 이름
        save_total_limit: 유지할 최대 개수
    """
    try:
        s3 = boto3.client("s3")
        prefix = f"checkpoints/{experiment_name}/"

        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)

        if "Contents" not in response:
            return

        # checkpoint_epoch_*.pt 파일만 필터링
        epoch_checkpoints = []
        for obj in response["Contents"]:
            key = obj["Key"]
            filename = key.split("/")[-1]
            if filename.startswith("checkpoint_epoch_") and filename.endswith(".pt"):
                epoch_checkpoints.append({"key": key, "last_modified": obj["LastModified"]})

        epoch_checkpoints.sort(key=lambda x: x["last_modified"])

        n_to_delete = len(epoch_checkpoints) - save_total_limit

        if n_to_delete > 0:
            for checkpoint in epoch_checkpoints[:n_to_delete]:
                s3.delete_object(Bucket=S3_BUCKET, Key=checkpoint["key"])
                filename = checkpoint["key"].split("/")[-1]
                logger.info(f"S3 checkpoint deleted: {filename}")

    except Exception as e:
        logger.warning(f"S3 cleanup failed: {e}")


def sync_mlruns_to_s3(
    mlruns_dir: Path | str = "./mlruns",
    experiment_name: str | None = None,
) -> bool:
    """mlruns 디렉터리를 S3에 동기화

    학습 종료 후 호출하여 MLflow 메트릭/파라미터를 S3에 백업
    나중에 로컬로 다운받아 mlflow ui로 확인 가능

    Args:
        mlruns_dir: mlruns 디렉터리 경로 (기본값: ./mlruns)
        experiment_name: 실험 이름 (None이면 mlruns 전체 동기화)

    Returns:
        성공 여부

    S3 저장 경로:
        s3://wmtp/mlruns/...
    """
    mlruns_path = Path(mlruns_dir)

    if not mlruns_path.exists():
        logger.warning(f"mlruns directory not found: {mlruns_path}")
        return False

    try:
        s3_dest = f"s3://{S3_BUCKET}/mlruns/"

        # aws s3 sync 명령 사용 (효율적인 증분 동기화)
        cmd = ["aws", "s3", "sync", str(mlruns_path), s3_dest, "--quiet"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            logger.info(f"mlruns synced to S3: {s3_dest}")
            return True
        else:
            logger.error(f"mlruns sync failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("mlruns sync timeout (300s)")
        return False
    except Exception as e:
        logger.error(f"mlruns sync failed: {e}")
        return False


def sync_checkpoints_to_s3(
    checkpoint_dir: Path | str,
    experiment_name: str,
) -> bool:
    """체크포인트 디렉터리를 S3에 동기화

    Args:
        checkpoint_dir: 체크포인트 디렉터리 경로
        experiment_name: 실험 이름

    Returns:
        성공 여부

    S3 저장 경로:
        s3://wmtp/checkpoints/{experiment_name}/...
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint directory not found: {checkpoint_path}")
        return False

    try:
        s3_dest = f"s3://{S3_BUCKET}/checkpoints/{experiment_name}/"

        cmd = ["aws", "s3", "sync", str(checkpoint_path), s3_dest, "--quiet"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode == 0:
            logger.info(f"Checkpoints synced to S3: {s3_dest}")
            return True
        else:
            logger.error(f"Checkpoints sync failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Checkpoints sync timeout (600s)")
        return False
    except Exception as e:
        logger.error(f"Checkpoints sync failed: {e}")
        return False


def shutdown_s3_executor() -> None:
    """모든 S3 업로드 완료 대기

    학습 종료 시 호출하여 모든 비동기 업로드 완료를 보장
    """
    s3_upload_executor.shutdown(wait=True)
    logger.info("All S3 uploads completed")


def reset_s3_executor() -> None:
    """S3 executor 재생성

    테스트 격리를 위해 shutdown 후 executor 재생성
    """
    global s3_upload_executor
    s3_upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-upload")
