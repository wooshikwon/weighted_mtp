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
        "--override",
        action="append",
        dest="overrides",
        help=(
            "Config 필드 override (계층 구조 지원). "
            "형식: key=value (예: --override experiment.name=test "
            "--override training.batch_size=8). 여러 번 사용 가능."
        ),
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

    # Config 로드
    try:
        config = OmegaConf.load(args.config)
    except Exception as e:
        logger.error(f"Config 로딩 실패: {e}")
        sys.exit(1)

    # Override 적용
    if args.overrides:
        from weighted_mtp.utils.config_utils import apply_overrides

        try:
            config = apply_overrides(config, args.overrides)
            logger.info(f"Override 적용 완료: {args.overrides}")
        except ValueError as e:
            logger.error(f"Override 적용 실패: {e}")
            sys.exit(1)

    # Stage 확인
    if not hasattr(config, "experiment") or not hasattr(
        config.experiment, "stage"
    ):
        logger.error("Config에 'experiment.stage' 필드가 없습니다.")
        sys.exit(1)

    stage = config.experiment.stage

    # Dry-run
    if args.dry_run:
        logger.info(f"Pipeline: {stage}")
        logger.info(f"Config file: {args.config}")
        logger.info(f"Final config:\n{OmegaConf.to_yaml(config)}")
        return

    # Pipeline 라우팅
    logger.info(f"실행 Pipeline: {stage}")
    logger.info(f"Config: {args.config}")

    if stage == "baseline":
        from weighted_mtp.pipelines.run_baseline import run_baseline_training

        run_baseline_training(config=config)

    elif stage == "critic":
        from weighted_mtp.pipelines.run_critic import run_critic_training

        run_critic_training(config=config)

    elif stage == "verifiable":
        from weighted_mtp.pipelines.run_verifiable import run_verifiable_training

        run_verifiable_training(config=config)

    elif stage == "rho1":
        from weighted_mtp.pipelines.run_rho1 import run_rho1_training

        run_rho1_training(config=config)

    else:
        logger.error(f"Unknown stage: {stage}")
        logger.error("Available stages: baseline, critic, verifiable, rho1")
        sys.exit(1)


if __name__ == "__main__":
    main()
