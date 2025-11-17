"""S3 유틸리티 단위 테스트"""

import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from weighted_mtp.utils import (
    cleanup_s3_checkpoints,
    shutdown_s3_executor,
    upload_to_s3_async,
)


@pytest.fixture
def temp_checkpoint(tmp_path):
    """임시 checkpoint 파일 생성"""
    checkpoint_path = tmp_path / "checkpoint_epoch_1.00.pt"
    checkpoint_path.write_text("dummy checkpoint data")
    return checkpoint_path


def test_upload_to_s3_async_disabled(temp_checkpoint):
    """MLflow 비활성화 시 업로드 스킵"""
    # MLflow 비활성화
    upload_to_s3_async(temp_checkpoint, mlflow_enabled=False)
    # 에러 없이 완료되면 성공


@patch("weighted_mtp.utils.s3_utils.mlflow")
def test_upload_to_s3_async_success(mock_mlflow, temp_checkpoint):
    """S3 업로드 성공 케이스"""
    # MLflow mock 설정
    mock_mlflow.log_artifact = Mock()

    # 업로드 실행
    upload_to_s3_async(temp_checkpoint, mlflow_enabled=True)

    # MLflow artifact 로깅 호출 확인
    mock_mlflow.log_artifact.assert_called_once_with(str(temp_checkpoint), "checkpoints")


@patch("weighted_mtp.utils.s3_utils.mlflow")
def test_upload_to_s3_async_failure(mock_mlflow, temp_checkpoint, caplog):
    """S3 업로드 실패 케이스"""
    # MLflow artifact 로깅 실패 시뮬레이션
    mock_mlflow.log_artifact.side_effect = Exception("S3 connection error")

    # 업로드 실행 (예외는 캐치되어야 함)
    upload_to_s3_async(temp_checkpoint, mlflow_enabled=True)

    # 에러 로그 확인
    assert "S3 upload failed" in caplog.text
    assert "S3 connection error" in caplog.text


@patch("weighted_mtp.utils.s3_utils.boto3")
@patch("weighted_mtp.utils.s3_utils.MlflowClient")
def test_cleanup_s3_checkpoints_success(mock_mlflow_client_cls, mock_boto3):
    """S3 checkpoint 정리 성공 케이스"""
    # Mock 설정
    mock_client = Mock()
    mock_mlflow_client_cls.return_value = mock_client

    mock_s3 = Mock()
    mock_boto3.client.return_value = mock_s3

    # Artifact 목록 mock (5개 checkpoint, 3개 삭제 대상)
    mock_artifacts = [
        Mock(path="checkpoints/checkpoint_epoch_1.00.pt"),
        Mock(path="checkpoints/checkpoint_epoch_2.00.pt"),
        Mock(path="checkpoints/checkpoint_epoch_3.00.pt"),
        Mock(path="checkpoints/checkpoint_epoch_4.00.pt"),
        Mock(path="checkpoints/checkpoint_epoch_5.00.pt"),
    ]
    mock_client.list_artifacts.return_value = mock_artifacts

    # S3 정리 실행 (최대 2개 유지 -> 3개 삭제)
    cleanup_s3_checkpoints(
        experiment_id="1",
        run_id="abc123",
        save_total_limit=2,
    )

    # S3 삭제 호출 확인 (가장 오래된 3개)
    assert mock_s3.delete_object.call_count == 3

    # 삭제된 파일 확인
    deleted_keys = [call.kwargs["Key"] for call in mock_s3.delete_object.call_args_list]
    assert "mlflow-artifacts/1/abc123/artifacts/checkpoints/checkpoint_epoch_1.00.pt" in deleted_keys
    assert "mlflow-artifacts/1/abc123/artifacts/checkpoints/checkpoint_epoch_2.00.pt" in deleted_keys
    assert "mlflow-artifacts/1/abc123/artifacts/checkpoints/checkpoint_epoch_3.00.pt" in deleted_keys


@patch("weighted_mtp.utils.s3_utils.boto3")
@patch("weighted_mtp.utils.s3_utils.MlflowClient")
def test_cleanup_s3_checkpoints_no_deletion(mock_mlflow_client_cls, mock_boto3):
    """삭제할 checkpoint가 없는 경우"""
    # Mock 설정
    mock_client = Mock()
    mock_mlflow_client_cls.return_value = mock_client

    mock_s3 = Mock()
    mock_boto3.client.return_value = mock_s3

    # Artifact 목록 mock (2개만 존재)
    mock_artifacts = [
        Mock(path="checkpoints/checkpoint_epoch_4.00.pt"),
        Mock(path="checkpoints/checkpoint_epoch_5.00.pt"),
    ]
    mock_client.list_artifacts.return_value = mock_artifacts

    # S3 정리 실행 (최대 3개 유지 -> 삭제 없음)
    cleanup_s3_checkpoints(
        experiment_id="1",
        run_id="abc123",
        save_total_limit=3,
    )

    # S3 삭제 호출되지 않음
    mock_s3.delete_object.assert_not_called()


@patch("weighted_mtp.utils.s3_utils.boto3")
@patch("weighted_mtp.utils.s3_utils.MlflowClient")
def test_cleanup_s3_checkpoints_failure(mock_mlflow_client_cls, mock_boto3, caplog):
    """S3 정리 실패 케이스"""
    # MLflow Client 실패 시뮬레이션
    mock_mlflow_client_cls.side_effect = Exception("MLflow connection error")

    # S3 정리 실행 (예외는 캐치되어야 함)
    cleanup_s3_checkpoints(
        experiment_id="1",
        run_id="abc123",
        save_total_limit=2,
    )

    # 경고 로그 확인
    assert "S3 cleanup failed" in caplog.text
    assert "MLflow connection error" in caplog.text


def test_shutdown_s3_executor():
    """S3 executor shutdown 테스트"""
    # Executor shutdown 실행
    shutdown_s3_executor()

    # 에러 없이 완료되면 성공
    # (실제로는 전역 executor가 shutdown되지만, 테스트 격리를 위해 검증 생략)
