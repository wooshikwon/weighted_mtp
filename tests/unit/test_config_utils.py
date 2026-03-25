"""Config 검증 유틸리티 테스트"""

import pytest
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf

from weighted_mtp.utils.config_utils import (
    validate_config,
    load_and_validate_config,
    ConfigValidationError,
)


class TestValidateConfigBasic:
    """기본 config 검증 테스트"""

    @patch("pathlib.Path.exists", return_value=True)
    def test_valid_baseline_config(self, mock_exists, project_root: Path):
        """유효한 baseline config 검증 통과"""
        config_path = project_root / "configs" / "local" / "baseline_local.yaml"
        config = OmegaConf.load(config_path)

        # 검증 통과 (예외 없음)
        validate_config(config)

    @patch("pathlib.Path.exists", return_value=True)
    def test_valid_taw_config(self, mock_exists, project_root: Path):
        """유효한 TAW config 검증 통과"""
        config_path = project_root / "configs" / "production" / "taw.yaml"
        config = OmegaConf.load(config_path)

        # 검증 통과 (예외 없음)
        validate_config(config)

    def test_missing_required_field(self):
        """필수 필드 누락 시 실패"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test"},
                # stage 누락
            }
        )

        with pytest.raises(ConfigValidationError, match="필수 필드 누락"):
            validate_config(config)

    def test_invalid_stage(self):
        """잘못된 stage 값"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "invalid_stage"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="잘못된 stage"):
            validate_config(config)


class TestValidateConfigValueRanges:
    """값 범위 검증 테스트"""

    def test_invalid_learning_rate_too_high(self):
        """Learning rate가 1.0 초과"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 2.0,  # 1.0 초과
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="learning_rate 범위 오류"):
            validate_config(config)

    def test_invalid_learning_rate_negative(self):
        """Learning rate가 음수"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": -0.001,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="learning_rate 범위 오류"):
            validate_config(config)

    def test_invalid_batch_size(self):
        """Batch size가 0 이하"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 0,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="batch_size는 양수여야 함"):
            validate_config(config)

    def test_invalid_n_epochs(self):
        """n_epochs가 0 이하"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": -1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="n_epochs는 양수여야 함"):
            validate_config(config)


class TestValidateConfigLogicalConsistency:
    """논리적 일관성 검증 테스트"""

    def test_batch_size_exceeds_samples(self):
        """Batch size가 샘플 수보다 큼"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 200,
                    "learning_rate": 1e-4,
                },
                "data_sampling": {
                    "n_samples": 100,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="batch_size.*n_samples"):
            validate_config(config)


class TestValidateStageSpecific:
    """Stage별 특수 검증 테스트"""

    def test_invalid_stage_verifiable(self):
        """삭제된 verifiable stage 사용 시 실패"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "verifiable"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="잘못된 stage"):
            validate_config(config)

    def test_invalid_stage_rho1(self):
        """삭제된 rho1 stage 사용 시 실패"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "rho1"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="잘못된 stage"):
            validate_config(config)



class TestLoadAndValidateConfig:
    """load_and_validate_config() 함수 테스트"""

    @patch("pathlib.Path.exists", return_value=True)
    def test_load_valid_config(self, mock_exists, project_root: Path):
        """유효한 config 로드 및 검증"""
        config_path = project_root / "configs" / "local" / "baseline_local.yaml"
        config = load_and_validate_config(str(config_path))

        assert config.experiment.stage == "baseline"
        assert hasattr(config, "training")

    def test_load_nonexistent_config(self):
        """존재하지 않는 config 파일"""
        with pytest.raises(FileNotFoundError):
            load_and_validate_config("nonexistent_config.yaml")

    def test_load_invalid_config(self, tmp_path: Path):
        """잘못된 config 로드"""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text(
            """
experiment:
  name: test
  stage: invalid_stage
"""
        )

        with pytest.raises(ConfigValidationError):
            load_and_validate_config(str(config_file))


class TestValidateConfigPathChecks:
    """경로 존재 확인 테스트 (통합 테스트)"""

    def test_nonexistent_model_path(self):
        """존재하지 않는 모델 경로 (디렉토리)"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "/nonexistent/model/path"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(ConfigValidationError, match="모델 경로가 존재하지 않음"):
            validate_config(config)

    @patch("pathlib.Path.exists", return_value=True)
    def test_checkpoint_path_skipped(self, mock_exists):
        """Checkpoint 경로는 검증 건너뜀"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "/nonexistent/checkpoint.pt"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "storage/datasets/codecontests/processed/train.jsonl",
                    "validation": "storage/datasets/codecontests/processed/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
                "data_sampling": {
                    "use_pairwise": False,
                    "n_samples": 100,
                },
            }
        )

        # .pt 파일은 검증 건너뛰므로 "모델 경로" 에러 없음
        validate_config(config)

    def test_nonexistent_dataset_path(self):
        """존재하지 않는 데이터셋 경로"""
        config = OmegaConf.create(
            {
                "experiment": {"name": "test", "stage": "baseline"},
                "models": {"policy": {"path": "storage/models/micro-mtp"}},
                "dataset": {
                    "name": "codecontests",
                    "train": "/nonexistent/train.jsonl",
                    "validation": "/nonexistent/valid.jsonl",
                },
                "training": {
                    "n_epochs": 1,
                    "batch_size": 4,
                    "learning_rate": 1e-4,
                },
            }
        )

        with pytest.raises(
            ConfigValidationError, match="데이터셋 파일이 존재하지 않음"
        ):
            validate_config(config)
