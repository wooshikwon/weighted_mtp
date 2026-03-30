"""Hugging Face Hub 업로드 유틸리티

학습 중 체크포인트를 HF Hub에 비동기 업로드.
s3_utils.py와 동일한 패턴: ThreadPoolExecutor + 임시 복사본으로 race condition 방지.

Race condition 방지 전략:
- 메인 스레드에서 즉시 temp copy 수행 (cleanup_old_checkpoints 전에 완료 보장)
- executor 스레드에서 temp copy → HF 업로드 → temp 정리

환경변수:
    HF_TOKEN: HuggingFace write 토큰 (필수)

설정 (config.checkpoint):
    hf_upload: true
    hf_repo_id: wooshikwon/weighted-mtp-checkpoints
"""

import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

# 전역 executor (비동기 업로드용)
hf_upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hf-upload")


def _get_hf_api():
    """HfApi 인스턴스 생성 (HF_TOKEN 환경변수 사용)"""
    import os

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN 환경변수가 설정되지 않았습니다")
    return HfApi(token=token)


def _upload_from_tmp(
    tmp_dir_path: str,
    filename: str,
    experiment_name: str,
    repo_id: str,
) -> None:
    """임시 복사본에서 HF Hub에 업로드 (executor 스레드에서 실행)

    Args:
        tmp_dir_path: tempfile.mkdtemp()로 생성된 임시 디렉터리 경로
        filename: checkpoint 파일명
        experiment_name: 실험 이름
        repo_id: HF repo ID
    """
    tmp_path = Path(tmp_dir_path) / filename
    try:
        api = _get_hf_api()
        path_in_repo = f"{experiment_name}/{filename}"

        api.upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
        )
        logger.info(f"HF upload complete: {filename} -> {repo_id}/{path_in_repo}")

    except Exception as e:
        logger.error(f"HF upload failed: {filename} - {e}")
    finally:
        # 임시 디렉터리 정리
        shutil.rmtree(tmp_dir_path, ignore_errors=True)


def submit_hf_upload(
    checkpoint_path: Path,
    experiment_name: str,
    repo_id: str,
) -> None:
    """체크포인트를 HF Hub에 비동기 업로드 예약

    메인 스레드에서 즉시 temp copy 수행 후, 업로드를 executor에 위임.
    이렇게 하면 cleanup_old_checkpoints가 원본을 삭제해도 업로드에 영향 없음.

    Args:
        checkpoint_path: 업로드할 checkpoint 경로
        experiment_name: 실험 이름 (HF 경로 구성에 사용)
        repo_id: HF repo ID

    HF 저장 경로:
        {experiment_name}/{checkpoint_filename}
    """
    if not checkpoint_path.exists():
        logger.warning(f"HF upload skipped (file not found): {checkpoint_path}")
        return

    # 메인 스레드에서 즉시 temp copy (cleanup 전에 완료 보장)
    tmp_dir_path = tempfile.mkdtemp(prefix="hf_upload_")
    tmp_path = Path(tmp_dir_path) / checkpoint_path.name
    shutil.copy2(checkpoint_path, tmp_path)

    # executor에서 업로드 수행
    hf_upload_executor.submit(
        _upload_from_tmp, tmp_dir_path, checkpoint_path.name, experiment_name, repo_id,
    )
    logger.info(f"HF 업로드 예약: {checkpoint_path.name}")


def submit_hf_cleanup(
    experiment_name: str,
    save_total_limit: int,
    repo_id: str,
) -> None:
    """HF Hub에서 오래된 checkpoint 비동기 삭제

    training loop 블로킹 방지를 위해 executor에서 실행.

    Args:
        experiment_name: 실험 이름
        save_total_limit: 유지할 최대 개수
        repo_id: HF repo ID
    """
    hf_upload_executor.submit(
        _cleanup_hf_checkpoints, experiment_name, save_total_limit, repo_id,
    )


def _cleanup_hf_checkpoints(
    experiment_name: str,
    save_total_limit: int,
    repo_id: str,
) -> None:
    """HF Hub에서 오래된 checkpoint 삭제 (executor 스레드에서 실행)

    checkpoint_epoch_*.pt 파일만 정리.
    checkpoint_best.pt와 checkpoint_final.pt는 유지.
    """
    try:
        api = _get_hf_api()

        all_files = api.list_repo_files(repo_id=repo_id)

        prefix = f"{experiment_name}/"
        epoch_files = sorted(
            f for f in all_files
            if f.startswith(prefix)
            and f.split("/")[-1].startswith("checkpoint_epoch_")
            and f.endswith(".pt")
        )

        n_to_delete = len(epoch_files) - save_total_limit

        if n_to_delete > 0:
            for filepath in epoch_files[:n_to_delete]:
                api.delete_file(path_in_repo=filepath, repo_id=repo_id)
                filename = filepath.split("/")[-1]
                logger.info(f"HF checkpoint deleted: {filename}")

    except Exception as e:
        logger.warning(f"HF cleanup failed: {e}")


def ensure_hf_repo(repo_id: str) -> None:
    """HF repo가 존재하는지 확인하고, 없으면 생성

    학습 시작 시 한 번 호출하여 repo 존재를 보장.

    Args:
        repo_id: HF repo ID
    """
    try:
        api = _get_hf_api()
        api.create_repo(repo_id, private=True, exist_ok=True)
        logger.info(f"HF repo ready: {repo_id}")
    except Exception as e:
        logger.error(f"HF repo 생성 실패: {e}")
        raise


def shutdown_hf_executor() -> None:
    """모든 HF 업로드 완료 대기

    학습 종료 시 호출하여 모든 비동기 업로드 완료를 보장.
    """
    hf_upload_executor.shutdown(wait=True)
    logger.info("All HF uploads completed")


def reset_hf_executor() -> None:
    """HF executor 재생성

    테스트 격리를 위해 shutdown 후 executor 재생성.
    """
    global hf_upload_executor
    hf_upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hf-upload")
