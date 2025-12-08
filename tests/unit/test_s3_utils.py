"""S3 유틸리티 단위 테스트

boto3 직접 사용 방식으로 리팩토링된 s3_utils 테스트
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from weighted_mtp.utils import (
    cleanup_s3_checkpoints,
    reset_s3_executor,
    shutdown_s3_executor,
    sync_mlruns_to_s3,
    upload_to_s3_async,
)


@pytest.fixture
def temp_checkpoint(tmp_path):
    """임시 checkpoint 파일 생성"""
    checkpoint_path = tmp_path / "checkpoint_epoch_1.00.pt"
    checkpoint_path.write_text("dummy checkpoint data")
    return checkpoint_path


def test_upload_to_s3_async_file_not_found(tmp_path, caplog):
    """파일이 존재하지 않으면 업로드 스킵"""
    non_existent_path = tmp_path / "non_existent.pt"
    upload_to_s3_async(non_existent_path, experiment_name="test-experiment")
    assert "S3 upload skipped" in caplog.text


@patch("weighted_mtp.utils.s3_utils.boto3")
def test_upload_to_s3_async_success(mock_boto3, temp_checkpoint):
    """boto3 직접 사용 S3 업로드 성공 (임시 복사본 사용)"""
    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3

    upload_to_s3_async(temp_checkpoint, experiment_name="test-experiment")

    # boto3 client 생성 확인
    mock_boto3.client.assert_called_once_with("s3")

    # upload_file 호출 확인
    mock_s3.upload_file.assert_called_once()
    call_args = mock_s3.upload_file.call_args

    # 임시 복사본 경로에서 업로드됨 (s3_upload_ 접두사)
    local_path = call_args[0][0]
    assert "s3_upload_" in local_path  # 임시 디렉터리에서 업로드
    assert local_path.endswith("checkpoint_epoch_1.00.pt")  # 파일명은 동일

    assert call_args[0][1] == "wmtp"  # S3 버킷 이름
    assert call_args[0][2] == "checkpoints/test-experiment/checkpoint_epoch_1.00.pt"  # S3 키


@patch("weighted_mtp.utils.s3_utils.boto3")
def test_upload_to_s3_async_failure(mock_boto3, temp_checkpoint, caplog):
    """S3 업로드 실패 케이스"""
    from botocore.exceptions import ClientError

    mock_s3 = MagicMock()
    mock_boto3.client.return_value = mock_s3
    mock_s3.upload_file.side_effect = ClientError(
        {"Error": {"Code": "500", "Message": "Internal Server Error"}},
        "upload_file",
    )

    # 업로드 실행 (예외는 캐치되어야 함)
    upload_to_s3_async(temp_checkpoint, experiment_name="test-experiment")

    # 에러 로그 확인
    assert "S3 upload failed" in caplog.text


@patch("weighted_mtp.utils.s3_utils.boto3")
def test_cleanup_s3_checkpoints_success(mock_boto3):
    """S3 checkpoint 정리 성공 케이스"""
    mock_s3 = Mock()
    mock_boto3.client.return_value = mock_s3

    from datetime import datetime, timedelta

    base_time = datetime(2025, 1, 1)

    # S3 list_objects_v2 응답 mock (5개 checkpoint)
    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {
                "Key": "checkpoints/test-experiment/checkpoint_epoch_1.00.pt",
                "LastModified": base_time + timedelta(hours=1),
            },
            {
                "Key": "checkpoints/test-experiment/checkpoint_epoch_2.00.pt",
                "LastModified": base_time + timedelta(hours=2),
            },
            {
                "Key": "checkpoints/test-experiment/checkpoint_epoch_3.00.pt",
                "LastModified": base_time + timedelta(hours=3),
            },
            {
                "Key": "checkpoints/test-experiment/checkpoint_epoch_4.00.pt",
                "LastModified": base_time + timedelta(hours=4),
            },
            {
                "Key": "checkpoints/test-experiment/checkpoint_epoch_5.00.pt",
                "LastModified": base_time + timedelta(hours=5),
            },
        ]
    }

    # S3 정리 실행 (최대 2개 유지 -> 3개 삭제)
    cleanup_s3_checkpoints(
        experiment_name="test-experiment",
        save_total_limit=2,
    )

    # S3 삭제 호출 확인 (가장 오래된 3개)
    assert mock_s3.delete_object.call_count == 3

    # 삭제된 파일 확인
    deleted_keys = [call.kwargs["Key"] for call in mock_s3.delete_object.call_args_list]
    assert "checkpoints/test-experiment/checkpoint_epoch_1.00.pt" in deleted_keys
    assert "checkpoints/test-experiment/checkpoint_epoch_2.00.pt" in deleted_keys
    assert "checkpoints/test-experiment/checkpoint_epoch_3.00.pt" in deleted_keys


@patch("weighted_mtp.utils.s3_utils.boto3")
def test_cleanup_s3_checkpoints_preserves_special_files(mock_boto3):
    """checkpoint_best.pt와 checkpoint_final.pt는 삭제하지 않음"""
    mock_s3 = Mock()
    mock_boto3.client.return_value = mock_s3

    from datetime import datetime, timedelta

    base_time = datetime(2025, 1, 1)

    # checkpoint_best.pt, checkpoint_final.pt 포함
    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {
                "Key": "checkpoints/test-experiment/checkpoint_epoch_1.00.pt",
                "LastModified": base_time + timedelta(hours=1),
            },
            {
                "Key": "checkpoints/test-experiment/checkpoint_epoch_2.00.pt",
                "LastModified": base_time + timedelta(hours=2),
            },
            {
                "Key": "checkpoints/test-experiment/checkpoint_best.pt",
                "LastModified": base_time + timedelta(hours=3),
            },
            {
                "Key": "checkpoints/test-experiment/checkpoint_final.pt",
                "LastModified": base_time + timedelta(hours=4),
            },
        ]
    }

    # S3 정리 실행 (최대 1개 유지 -> epoch checkpoint 1개만 삭제)
    cleanup_s3_checkpoints(
        experiment_name="test-experiment",
        save_total_limit=1,
    )

    # checkpoint_epoch_*.pt만 삭제 (1개)
    assert mock_s3.delete_object.call_count == 1

    # 삭제된 파일 확인 (가장 오래된 epoch checkpoint)
    deleted_keys = [call.kwargs["Key"] for call in mock_s3.delete_object.call_args_list]
    assert "checkpoints/test-experiment/checkpoint_epoch_1.00.pt" in deleted_keys


@patch("weighted_mtp.utils.s3_utils.boto3")
def test_cleanup_s3_checkpoints_no_deletion(mock_boto3):
    """삭제할 checkpoint가 없는 경우"""
    mock_s3 = Mock()
    mock_boto3.client.return_value = mock_s3

    from datetime import datetime, timedelta

    base_time = datetime(2025, 1, 1)

    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {
                "Key": "checkpoints/test-experiment/checkpoint_epoch_4.00.pt",
                "LastModified": base_time + timedelta(hours=4),
            },
            {
                "Key": "checkpoints/test-experiment/checkpoint_epoch_5.00.pt",
                "LastModified": base_time + timedelta(hours=5),
            },
        ]
    }

    # S3 정리 실행 (최대 3개 유지 -> 삭제 없음)
    cleanup_s3_checkpoints(
        experiment_name="test-experiment",
        save_total_limit=3,
    )

    # S3 삭제 호출되지 않음
    mock_s3.delete_object.assert_not_called()


@patch("weighted_mtp.utils.s3_utils.boto3")
def test_cleanup_s3_checkpoints_empty_bucket(mock_boto3):
    """버킷이 비어있는 경우"""
    mock_s3 = Mock()
    mock_boto3.client.return_value = mock_s3

    # Contents 키가 없는 응답
    mock_s3.list_objects_v2.return_value = {}

    # S3 정리 실행 (예외 없이 완료되어야 함)
    cleanup_s3_checkpoints(
        experiment_name="test-experiment",
        save_total_limit=2,
    )

    # S3 삭제 호출되지 않음
    mock_s3.delete_object.assert_not_called()


@patch("weighted_mtp.utils.s3_utils.boto3")
def test_cleanup_s3_checkpoints_failure(mock_boto3, caplog):
    """S3 정리 실패 케이스"""
    mock_boto3.client.side_effect = Exception("S3 connection error")

    # S3 정리 실행 (예외는 캐치되어야 함)
    cleanup_s3_checkpoints(
        experiment_name="test-experiment",
        save_total_limit=2,
    )

    # 경고 로그 확인
    assert "S3 cleanup failed" in caplog.text


def test_shutdown_s3_executor():
    """S3 executor shutdown 테스트"""
    shutdown_s3_executor()
    # 테스트 격리를 위해 executor 재생성
    reset_s3_executor()


@patch("weighted_mtp.utils.s3_utils.subprocess")
def test_sync_mlruns_to_s3_success(mock_subprocess, tmp_path):
    """mlruns 동기화 성공"""
    # mlruns 디렉터리 생성
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir()

    mock_result = Mock()
    mock_result.returncode = 0
    mock_subprocess.run.return_value = mock_result

    result = sync_mlruns_to_s3(mlruns_dir=str(mlruns_dir))

    assert result is True
    mock_subprocess.run.assert_called_once()

    # aws s3 sync 명령 확인
    call_args = mock_subprocess.run.call_args[0][0]
    assert call_args[0] == "aws"
    assert call_args[1] == "s3"
    assert call_args[2] == "sync"


@patch("weighted_mtp.utils.s3_utils.subprocess")
def test_sync_mlruns_to_s3_failure(mock_subprocess, tmp_path, caplog):
    """mlruns 동기화 실패"""
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir()

    mock_result = Mock()
    mock_result.returncode = 1
    mock_result.stderr = "aws cli error"
    mock_subprocess.run.return_value = mock_result

    result = sync_mlruns_to_s3(mlruns_dir=str(mlruns_dir))

    assert result is False
    assert "mlruns sync failed" in caplog.text


def test_sync_mlruns_to_s3_directory_not_found(tmp_path, caplog):
    """mlruns 디렉터리가 없는 경우"""
    non_existent_dir = tmp_path / "non_existent_mlruns"

    result = sync_mlruns_to_s3(mlruns_dir=str(non_existent_dir))

    assert result is False
    assert "mlruns directory not found" in caplog.text
