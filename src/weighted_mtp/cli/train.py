"""학습 파이프라인 CLI 진입점

독립된 Pipeline 라우터: config의 experiment.stage 필드를 읽고 해당 pipeline을 실행
"""

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from weighted_mtp.core.env import ensure_env_loaded
from weighted_mtp.core.logging import setup_logging

# 환경변수 로드 (MLflow credentials 등)
ensure_env_loaded()

logger = setup_logging("CLI")


def main():
    parser = argparse.ArgumentParser(description="Weighted MTP 학습 파이프라인")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="실험 config 경로 (예: configs/baseline/baseline.yaml)",
    )
    parser.add_argument(
        "--run-name",
        help="MLflow run 이름 override",
    )
    parser.add_argument(
        "--device",
        help="Device override (cuda/cpu/mps)",
    )
    parser.add_argument(
        "--use-micro-model",
        action="store_true",
        help="Micro 모델 사용 (로컬 테스트)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="설정만 출력하고 종료",
    )

    args = parser.parse_args()

    # Config 파일 존재 확인
    if not args.config.exists():
        logger.error(f"Config 파일을 찾을 수 없습니다: {args.config}")
        sys.exit(1)

    # Stage 식별 (config에서 experiment.stage 읽기)
    try:
        config_preview = OmegaConf.load(args.config)
        stage = config_preview.experiment.stage
    except Exception as e:
        logger.error(f"Config 로딩 실패: {e}")
        logger.error("Config 파일에 'experiment.stage' 필드가 있는지 확인하세요.")
        sys.exit(1)

    # Override params 생성
    overrides = {}
    if args.run_name:
        overrides["experiment.name"] = args.run_name
    if args.device:
        overrides["runtime.device"] = args.device
    if args.use_micro_model:
        overrides["models.policy.path"] = "storage/models_v2/micro-mtp"

    # Dry-run
    if args.dry_run:
        logger.info(f"Pipeline: {stage}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Overrides: {overrides}")
        return

    # Pipeline 라우팅
    logger.info(f"실행 Pipeline: {stage}")
    logger.info(f"Config: {args.config}")

    if stage == "baseline":
        from weighted_mtp.pipelines.run_baseline import run_baseline_training

        run_baseline_training(config_path=str(args.config), **overrides)

    elif stage == "critic":
        from weighted_mtp.pipelines.run_critic import run_critic_training

        run_critic_training(config_path=str(args.config), **overrides)

    elif stage == "verifiable":
        from weighted_mtp.pipelines.run_verifiable import run_verifiable_training

        run_verifiable_training(config_path=str(args.config), **overrides)

    elif stage == "rho1":
        from weighted_mtp.pipelines.run_rho1 import run_rho1_training

        run_rho1_training(config_path=str(args.config), **overrides)

    else:
        logger.error(f"Unknown stage: {stage}")
        logger.error("Available stages: baseline, critic, verifiable, rho1")
        sys.exit(1)


if __name__ == "__main__":
    main()
