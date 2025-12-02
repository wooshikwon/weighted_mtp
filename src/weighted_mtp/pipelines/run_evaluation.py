"""평가 파이프라인

Checkpoint를 로드하여 벤치마크 데이터셋에서 Pass@K 평가 수행
"""

import json
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
from tqdm import tqdm

from weighted_mtp.core.logging import setup_logging
from weighted_mtp.data import load_evaluation_dataset
from weighted_mtp.models.tokenizer_utils import load_tokenizer, resolve_tokenizer_path
from weighted_mtp.utils import (
    evaluate_gsm8k_answer,
    evaluate_pass_at_k,
    execute_code_with_tests,
    execute_codecontests_tests,
    execute_mbpp_tests,
    generate_with_mtp,
    load_checkpoint_for_evaluation,
)


def _load_codecontests_tests(split: str = "test") -> dict[str, dict]:
    """CodeContests 테스트 케이스 로드

    Args:
        split: 데이터 split (train, valid, test)

    Returns:
        {task_name: {"public_tests": {...}, "private_tests": {...}, ...}}
    """
    tests_path = Path(f"storage/datasets/codecontests/tests/{split}_tests.json")
    if not tests_path.exists():
        raise FileNotFoundError(
            f"CodeContests 테스트 케이스 파일이 없습니다: {tests_path}\n"
            f"먼저 실행: uv run python scripts/create_storage/extract_codecontests_tests.py"
        )

    with open(tests_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(
    checkpoint_path: str,
    dataset_name: str = "humaneval",
    num_samples_per_task: int = 20,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
    device: str = "auto",
    dtype: str | None = None,
    mlflow_enabled: bool = True,
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """평가 파이프라인 실행

    Args:
        checkpoint_path: Checkpoint 경로
        dataset_name: "humaneval", "mbpp", "gsm8k", "codecontests"
        num_samples_per_task: 각 문제당 생성 개수 (Pass@K 계산용)
        temperature: Sampling temperature
        max_new_tokens: 최대 생성 토큰 수
        device: 디바이스 ("auto", "cuda", "cpu", "mps")
        dtype: 모델 dtype ("float16", "bfloat16", None=원본유지). MPS는 bfloat16 미지원으로 float16 권장
        mlflow_enabled: MLflow 로깅 여부
        max_tasks: 최대 평가 태스크 수 (테스트용, None=전체)

    Returns:
        {
            "pass_at_k": {"pass@1": 0.2, "pass@5": 0.65, ...},
            "per_task": [{"task_id": ..., "pass@1": ..., ...}, ...],
            "checkpoint_metadata": {...}
        }

    Examples:
        >>> results = run_evaluation(
        ...     checkpoint_path="storage/checkpoints/baseline/checkpoint_best.pt",
        ...     dataset_name="humaneval",
        ...     num_samples_per_task=20,
        ...     temperature=0.2,
        ... )
        >>> print(results["pass_at_k"])
        {'pass@1': 0.24, 'pass@5': 0.68, 'pass@10': 0.85}
    """
    # 1. Setup
    logger = setup_logging("EVALUATION")

    # Device 설정 (auto: cuda > mps > cpu)
    if device == "auto":
        if torch.cuda.is_available():
            device_obj = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device_obj = torch.device("mps")
        else:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)

    # dtype 변환 (문자열 → torch.dtype)
    dtype_obj = None
    if dtype == "float16":
        dtype_obj = torch.float16
    elif dtype == "bfloat16":
        dtype_obj = torch.bfloat16

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Samples per task: {num_samples_per_task}")
    logger.info(f"Device: {device_obj}")
    if dtype_obj is not None:
        logger.info(f"dtype: {dtype_obj}")

    # 2. Load checkpoint (inference_only=True: extra_heads 스킵으로 ~1.2GB 절약)
    logger.info("Loading checkpoint...")
    model, checkpoint_metadata = load_checkpoint_for_evaluation(
        checkpoint_path=Path(checkpoint_path),
        device=device_obj,
        inference_only=True,
        dtype=dtype_obj,
    )
    logger.info(f"Checkpoint epoch: {checkpoint_metadata['epoch']}")
    logger.info(f"Validation loss: {checkpoint_metadata['val_metrics']['val_loss']:.4f}")

    # 3. Load tokenizer
    tokenizer_path = resolve_tokenizer_path(checkpoint_metadata['config']['model']['path'])
    tokenizer = load_tokenizer(tokenizer_path)

    # 4. Load evaluation dataset
    logger.info(f"Loading {dataset_name} dataset...")
    dataset = load_evaluation_dataset(dataset_name, split="test")
    logger.info(f"Dataset size: {len(dataset)}")

    # 5. MLflow setup
    if mlflow_enabled:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.start_run(run_name=f"eval_{dataset_name}_{Path(checkpoint_path).stem}")
            mlflow.log_params({
                "checkpoint": checkpoint_path,
                "dataset": dataset_name,
                "num_samples_per_task": num_samples_per_task,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "checkpoint_epoch": checkpoint_metadata["epoch"],
                "checkpoint_val_loss": checkpoint_metadata["val_metrics"]["val_loss"],
            })
            logger.info(f"MLflow tracking enabled: {tracking_uri}")
            logger.info(f"MLflow Run ID: {mlflow.active_run().info.run_id}")
        else:
            logger.warning("MLflow enabled but MLFLOW_TRACKING_URI not set. Skipping MLflow.")
            mlflow_enabled = False

    # 6. CodeContests 테스트 케이스 로드 (필요 시)
    codecontests_tests = None
    if dataset_name == "codecontests":
        logger.info("Loading CodeContests test cases...")
        codecontests_tests = _load_codecontests_tests(split="test")
        logger.info(f"Loaded test cases for {len(codecontests_tests)} problems")

    # 7. Evaluation loop
    logger.info("Starting evaluation...")
    all_results = []
    generated_samples = []  # 생성 샘플 저장 (처음 10개)

    # 테스트용 제한
    eval_dataset = dataset if max_tasks is None else dataset.select(range(min(max_tasks, len(dataset))))
    logger.info(f"Evaluating {len(eval_dataset)} tasks")

    for idx, sample in enumerate(tqdm(eval_dataset, desc="Evaluating")):
        task_id = sample["task_id"]
        prompt = sample["instruction"]

        # Generate N samples
        generated_codes = generate_with_mtp(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_samples_per_task,
            device=device_obj,
        )

        # Execute and check (데이터셋별 분기)
        task_results = []
        for code in generated_codes:
            if dataset_name == "humaneval":
                # HumanEval: check(candidate) 형식
                test_code = sample["metadata"]["test"]
                entry_point = sample["metadata"]["entry_point"]
                passed = execute_code_with_tests(
                    code=code,
                    test_code=test_code,
                    entry_point=entry_point,
                    timeout=5,
                )
            elif dataset_name == "mbpp":
                # MBPP: assert 리스트 형식
                test_list = sample["metadata"]["test_list"]
                test_setup_code = sample["metadata"].get("test_setup_code", "")
                passed = execute_mbpp_tests(
                    code=code,
                    test_list=test_list,
                    test_setup_code=test_setup_code,
                    timeout=5,
                )
            elif dataset_name == "gsm8k":
                # GSM8K: exact match 형식
                ground_truth = sample["metadata"]["final_answer"]
                passed = evaluate_gsm8k_answer(
                    generated_text=code,
                    ground_truth=ground_truth,
                )
            elif dataset_name == "codecontests":
                # CodeContests: stdin/stdout 형식
                # task_id에서 problem name 추출 (예: "problem_name_correct_0" → "problem_name")
                parts = task_id.rsplit("_", 2)
                if len(parts) >= 3 and parts[-2] in ("correct", "incorrect"):
                    problem_name = "_".join(parts[:-2])
                else:
                    problem_name = task_id

                test_data = codecontests_tests.get(problem_name, {})
                public_tests = test_data.get("public_tests", {"input": [], "output": []})

                if public_tests["input"]:
                    result = execute_codecontests_tests(
                        code=code,
                        tests=public_tests,
                        timeout=10,
                    )
                    # 모든 public 테스트 통과 시 pass
                    passed = result["pass_rate"] >= 1.0
                else:
                    # 테스트 케이스가 없으면 fail 처리
                    passed = False
            else:
                raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
            task_results.append(passed)

        # Record
        all_results.append({
            "task_id": task_id,
            "results": task_results,
            "num_correct": sum(task_results),
            "num_total": len(task_results),
        })

        # 처음 10개 task의 생성 샘플 저장 (MLflow artifact용)
        if idx < 10:
            generated_samples.append({
                "task_id": task_id,
                "prompt": prompt,
                "generated_codes": generated_codes,
                "results": task_results,
            })

    # 8. Compute Pass@K (Chen et al. 2021 방식: 문제별 Pass@K 평균)
    logger.info("Computing Pass@K metrics...")

    k_values = [1, 5, 10, 20] if num_samples_per_task >= 20 else [1, 5, 10]

    # Per-task Pass@K 계산
    per_task_pass_at_k = []
    for task in all_results:
        task_metrics = evaluate_pass_at_k(
            task["results"],
            k_values=k_values,
        )
        per_task_pass_at_k.append({
            "task_id": task["task_id"],
            **task_metrics,
        })

    # 전체 Pass@K: 문제별 Pass@K의 평균 (Chen et al. 2021)
    pass_at_k_metrics = {}
    for k in k_values:
        key = f"pass@{k}"
        values = [t[key] for t in per_task_pass_at_k if key in t]
        if values:
            pass_at_k_metrics[key] = sum(values) / len(values)

    # 9. Log results
    logger.info("=== Evaluation Results ===")
    for k, v in pass_at_k_metrics.items():
        logger.info(f"{k}: {v:.2%}")

    if mlflow_enabled:
        mlflow.log_metrics(pass_at_k_metrics)

        # Save per-task results
        results_df = pd.DataFrame(per_task_pass_at_k)
        results_csv = f"results_{dataset_name}.csv"
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(results_csv)

        # Save generated samples (처음 10개)
        if generated_samples:
            import json
            samples_jsonl = f"samples_{dataset_name}.jsonl"
            with open(samples_jsonl, "w", encoding="utf-8") as f:
                for sample in generated_samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            mlflow.log_artifact(samples_jsonl)

        mlflow.end_run()

    # 10. Return results
    return {
        "pass_at_k": pass_at_k_metrics,
        "per_task": per_task_pass_at_k,
        "checkpoint_metadata": checkpoint_metadata,
    }
