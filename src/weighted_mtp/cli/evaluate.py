"""평가 파이프라인 CLI 진입점

학습된 checkpoint 또는 base 모델을 로드하여 벤치마크 데이터셋에서 Pass@K 평가 수행
"""

import argparse
import sys
from pathlib import Path

import torch

from weighted_mtp.core.env import ensure_env_loaded
from weighted_mtp.core.logging import setup_logging
from weighted_mtp.pipelines.run_evaluation import load_model_for_evaluation, run_evaluation

# 환경변수 로드 (MLflow credentials 등)
ensure_env_loaded()

logger = setup_logging("EVALUATE")


def load_base_model_for_evaluation(
    model_path: str,
    device: str = "auto",
    dtype: str | None = None,
):
    """Base 모델 직접 로드 (체크포인트 없이 SFT 전 모델 평가용)

    Args:
        model_path: Base 모델 경로 (예: storage/models/meta-llama-mtp)
        device: 디바이스 ("auto", "cuda", "cpu", "mps")
        dtype: 모델 dtype ("float16", "bfloat16", None=원본유지)

    Returns:
        (model, tokenizer, metadata, device_obj)
    """
    from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
    from weighted_mtp.models.tokenizer_utils import load_tokenizer, resolve_tokenizer_path

    # Device 설정
    if device == "auto":
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device_obj = torch.device("mps")
        else:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)

    # dtype 변환
    dtype_obj = None
    if dtype == "float16":
        dtype_obj = torch.float16
    elif dtype == "bfloat16":
        dtype_obj = torch.bfloat16

    logger.info(f"Base Model: {model_path}")
    logger.info(f"Device: {device_obj}")
    if dtype_obj:
        logger.info(f"dtype: {dtype_obj}")

    # Base 모델 로드 (LoRA 없이)
    logger.info("Loading base model (no SFT)...")
    model = MetaLlamaMTPAdapter.from_pretrained(
        model_path=model_path,
        device=device_obj,
        use_lora=False,
    )

    # dtype 변환
    if dtype_obj is not None:
        model = model.to(dtype_obj)

    model.eval()

    # 토크나이저 로드
    tokenizer_path = resolve_tokenizer_path(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    # 가상 메타데이터 (checkpoint_metadata 형식 유지)
    metadata = {
        "epoch": 0.0,
        "config": {"model": {"path": model_path}},
        "val_metrics": {"val_loss": 0.0},
        "is_base_model": True,
    }

    logger.info("Base model loaded successfully")
    return model, tokenizer, metadata, device_obj


def main():
    parser = argparse.ArgumentParser(description="Weighted MTP 모델 평가")

    # Model source (checkpoint 또는 base-model 중 하나 필수)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint 경로 (예: storage/checkpoints/baseline/checkpoint_best.pt)",
    )
    model_group.add_argument(
        "--base-model",
        type=str,
        help="Base 모델 경로 (SFT 전 pure 모델 평가, 예: storage/models/meta-llama-mtp)",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="humaneval",
        choices=["humaneval", "mbpp", "gsm8k", "codecontests"],
        help="평가 데이터셋 (기본: humaneval)",
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="문제당 생성 샘플 수 (기본: codecontests=20, 나머지=100)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (0=greedy, 기본: 0.2)",
    )
    parser.add_argument(
        "--temperature-search",
        type=str,
        default=None,
        help="Temperature 탐색 범위 (예: '0.5,0.6,0.7,0.8,0.9'). 지정 시 --temperature 무시",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="최대 생성 토큰 수 (기본: 512)",
    )

    # Environment
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="디바이스 (cuda, cpu, mps, auto, 기본: auto)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16"],
        help="모델 dtype (MPS는 bfloat16 미지원, float16 권장)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="MLflow 로깅 비활성화",
    )

    # Test parameters
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="최대 평가 태스크 수 (테스트용, 기본: 전체)",
    )

    args = parser.parse_args()

    # 데이터셋별 기본 샘플 수 설정
    if args.num_samples is None:
        if args.dataset == "codecontests":
            args.num_samples = 10
        else:
            args.num_samples = 100

    # 모델 소스 확인 (checkpoint 또는 base-model)
    use_base_model = args.base_model is not None
    model_source = args.base_model if use_base_model else args.checkpoint

    if use_base_model:
        # Base 모델 디렉토리 존재 확인
        model_path = Path(args.base_model)
        if not model_path.exists():
            logger.error(f"Base 모델 경로를 찾을 수 없습니다: {args.base_model}")
            sys.exit(1)
    else:
        # Checkpoint 파일 존재 확인
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint 파일을 찾을 수 없습니다: {args.checkpoint}")
            sys.exit(1)

    # Temperature search 모드 확인
    if args.temperature_search:
        temperatures = [float(t.strip()) for t in args.temperature_search.split(",")]
        logger.info("=== Temperature Search 모드 ===")
        logger.info(f"Model: {model_source} ({'base' if use_base_model else 'checkpoint'})")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Temperatures: {temperatures}")
    else:
        temperatures = [args.temperature]
        logger.info("=== 평가 파이프라인 시작 ===")
        logger.info(f"Model: {model_source} ({'base' if use_base_model else 'checkpoint'})")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Temperature: {args.temperature}")

    logger.info(f"Samples per task: {args.num_samples}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Device: {args.device}")
    logger.info(f"dtype: {args.dtype or 'auto'}")
    logger.info(f"MLflow: {'disabled' if args.no_mlflow else 'enabled'}")

    # Run evaluation
    try:
        all_results = []

        # 모델을 한 번만 로드하고 여러 temperature에서 재사용
        if use_base_model:
            model, tokenizer, checkpoint_metadata, device_obj = load_base_model_for_evaluation(
                model_path=args.base_model,
                device=args.device,
                dtype=args.dtype,
            )
        else:
            model, tokenizer, checkpoint_metadata, device_obj = load_model_for_evaluation(
                checkpoint_path=args.checkpoint,
                device=args.device,
                dtype=args.dtype,
            )

        for temp in temperatures:
            if len(temperatures) > 1:
                logger.info(f"\n--- Temperature {temp} 평가 시작 ---")

            results = run_evaluation(
                checkpoint_path=model_source,
                dataset_name=args.dataset,
                num_samples_per_task=args.num_samples,
                temperature=temp,
                max_new_tokens=args.max_tokens,
                device=args.device,
                dtype=args.dtype,
                mlflow_enabled=not args.no_mlflow,
                max_tasks=args.max_tasks,
                model=model,
                tokenizer=tokenizer,
                checkpoint_metadata=checkpoint_metadata,
                device_obj=device_obj,
            )
            results["temperature"] = temp
            all_results.append(results)

        # Temperature Search 결과 출력
        if len(temperatures) > 1:
            print("\n" + "=" * 60)
            print("Temperature Search 결과")
            print("=" * 60)

            # 각 temperature 결과
            for r in all_results:
                temp = r["temperature"]
                pass1 = r["pass_at_k"].get("pass@1", 0)
                print(f"Temperature {temp}: pass@1 = {pass1:.2%}")

            # 최적 temperature 찾기 (pass@1 기준)
            best_result = max(all_results, key=lambda x: x["pass_at_k"].get("pass@1", 0))
            best_temp = best_result["temperature"]

            print("=" * 60)
            print(f"최적 Temperature: {best_temp}")
            print("=" * 60)
            for k, v in best_result["pass_at_k"].items():
                print(f"{k}: {v:.2%}")
            print("=" * 60)

            results = best_result
        else:
            # 단일 temperature 결과 출력
            print("\n" + "=" * 50)
            print("평가 결과 요약")
            print("=" * 50)
            for k, v in results["pass_at_k"].items():
                print(f"{k}: {v:.2%}")
            print("=" * 50)

        print(f"총 평가 태스크: {len(results['per_task'])}")
        if results['checkpoint_metadata'].get('is_base_model'):
            print("Model: Base (no SFT)")
        else:
            print(f"Checkpoint epoch: {results['checkpoint_metadata']['epoch']}")
            print(f"Checkpoint val_loss: {results['checkpoint_metadata']['val_metrics']['val_loss']:.4f}")
        print("=" * 50)

        logger.info("평가 완료")

    except Exception as e:
        logger.error(f"평가 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
