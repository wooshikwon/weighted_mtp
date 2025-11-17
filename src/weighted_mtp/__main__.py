"""Weighted MTP CLI 진입점

서브커맨드:
    train                학습 파이프라인 실행
    evaluate             평가 파이프라인 실행
    validate-config      Config 파일 검증

예시:
    # 학습
    python -m weighted_mtp train --config configs/baseline/baseline.yaml

    # 평가
    python -m weighted_mtp evaluate --checkpoint storage/checkpoints/baseline/checkpoint_best.pt

    # Config 검증
    python -m weighted_mtp validate-config --config configs/baseline/baseline.yaml

자세한 정보: https://github.com/wooshikwon/weighted-mtp
"""

import sys
import argparse
from pathlib import Path


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(
        prog="weighted_mtp",
        description="Weighted Multi-Token Prediction (WMTP) 프레임워크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 학습
  python -m weighted_mtp train --config configs/baseline/baseline.yaml

  # 평가
  python -m weighted_mtp evaluate --checkpoint storage/checkpoints/baseline/checkpoint_best.pt

  # Config 검증
  python -m weighted_mtp validate-config --config configs/baseline/baseline.yaml

자세한 정보:
  각 서브커맨드의 상세 옵션은 --help로 확인하세요.
  예: python -m weighted_mtp train --help
        """,
    )

    # Version
    parser.add_argument(
        "--version",
        action="version",
        version="weighted_mtp 0.2.0",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command", help="사용 가능한 서브커맨드", required=False
    )

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train",
        help="학습 파이프라인 실행",
        description="Weighted MTP 학습 파이프라인을 실행합니다.",
        add_help=False,  # cli/train.py의 help 사용
    )

    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="평가 파이프라인 실행",
        description="학습된 checkpoint를 평가합니다.",
        add_help=False,  # cli/evaluate.py의 help 사용
    )

    # Validate-config subcommand
    validate_parser = subparsers.add_parser(
        "validate-config",
        help="Config 파일 검증",
        description="Config 파일의 필수 필드와 값을 검증합니다.",
    )
    validate_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config 파일 경로 (예: configs/baseline/baseline.yaml)",
    )

    # Parse arguments
    # sys.argv[1]만 먼저 파싱하여 서브커맨드 확인
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    # 서브커맨드 확인
    args, remaining = parser.parse_known_args()

    # Subcommand routing
    if args.command == "train":
        from weighted_mtp.cli.train import main as train_main

        # train CLI에 나머지 인자 전달
        sys.argv = [sys.argv[0]] + remaining
        train_main()

    elif args.command == "evaluate":
        from weighted_mtp.cli.evaluate import main as evaluate_main

        # evaluate CLI에 나머지 인자 전달
        sys.argv = [sys.argv[0]] + remaining
        evaluate_main()

    elif args.command == "validate-config":
        from omegaconf import OmegaConf
        from weighted_mtp.utils.config_utils import (
            validate_config,
            ConfigValidationError,
        )

        try:
            # Config 로드
            config_path = Path(args.config)
            if not config_path.exists():
                print(f"✗ Config 파일을 찾을 수 없습니다: {args.config}")
                sys.exit(1)

            config = OmegaConf.load(config_path)

            # 검증
            validate_config(config)

            print(f"✓ Config 검증 성공: {args.config}")
            print(f"  - Experiment: {config.experiment.name}")
            print(f"  - Stage: {config.experiment.stage}")
            print(f"  - Model: {config.models.policy.path}")
            print(f"  - Dataset: {config.dataset.name}")

        except ConfigValidationError as e:
            print(f"✗ Config 검증 실패:\n{e}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Config 로딩 실패: {e}")
            sys.exit(1)

    else:
        # 서브커맨드 없이 실행된 경우
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
