"""평가 파이프라인

Checkpoint를 로드하여 벤치마크 데이터셋에서 Pass@K 평가 수행

eval_mode:
  - "evalplus": HumanEval(+)/MBPP(+) — evalplus 패키지로 평가 (학회 표준)
  - "legacy": 기존 커스텀 실행기 (CodeContests, 디버깅용)
  - GSM8K는 항상 자체 exact match 사용
"""

import json
import os
import random
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import torch
from tqdm import tqdm

from weighted_mtp.core.logging import setup_logging
from weighted_mtp.data import load_evaluation_dataset
from weighted_mtp.data.collators import apply_alpaca_template
from weighted_mtp.models.tokenizer_utils import load_tokenizer, resolve_tokenizer_path
from weighted_mtp.utils import (
    evaluate_gsm8k_answer,
    evaluate_pass_at_k,
    execute_code_with_tests,
    execute_codecontests_tests,
    execute_mbpp_tests,
    load_checkpoint_for_evaluation,
    postprocess_humaneval_completion,
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
            f"먼저 실행: uv run python scripts/create_storage/setup_datasets.py --datasets codecontests --steps tests"
        )

    with open(tests_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sample_codecontests_tests(
    test_data: dict,
    max_tests: int = 150,
    seed: int = 42,
) -> dict:
    """CodeContests 테스트 케이스 샘플링 (public + private + generated 통합)

    재현성을 위해 고정 seed 사용. public 테스트 우선, 부족 시 private/generated 추가.

    Args:
        test_data: {"public_tests": {...}, "private_tests": {...}, "generated_tests": {...}}
        max_tests: 최대 테스트 케이스 수
        seed: 랜덤 시드 (재현성 보장)

    Returns:
        {"input": [...], "output": [...]} (최대 max_tests개)
    """
    # 각 테스트 타입에서 input/output 추출
    public = test_data.get("public_tests", {"input": [], "output": []})
    private = test_data.get("private_tests", {"input": [], "output": []})
    generated = test_data.get("generated_tests", {"input": [], "output": []})

    # 모든 테스트를 (input, output) 튜플 리스트로 통합
    all_tests = []

    # public 테스트 우선 추가
    for inp, out in zip(public.get("input", []), public.get("output", [])):
        all_tests.append((inp, out))

    # private 테스트 추가
    for inp, out in zip(private.get("input", []), private.get("output", [])):
        all_tests.append((inp, out))

    # generated 테스트 추가
    for inp, out in zip(generated.get("input", []), generated.get("output", [])):
        all_tests.append((inp, out))

    # 테스트가 없으면 빈 결과 반환
    if not all_tests:
        return {"input": [], "output": []}

    # max_tests 초과 시 샘플링 (고정 seed)
    if len(all_tests) > max_tests:
        rng = random.Random(seed)
        all_tests = rng.sample(all_tests, max_tests)

    # 결과 형식으로 변환
    inputs = [t[0] for t in all_tests]
    outputs = [t[1] for t in all_tests]

    return {"input": inputs, "output": outputs}


def load_model_for_evaluation(
    checkpoint_path: str,
    device: str = "auto",
    dtype: str | None = None,
) -> tuple[Any, Any, dict[str, Any], torch.device]:
    """평가용 모델과 토크나이저 로드

    Temperature search 등에서 모델을 한 번만 로드하고 재사용할 때 사용.

    Args:
        checkpoint_path: Checkpoint 경로
        device: 디바이스 ("auto", "cuda", "cpu", "mps")
        dtype: 모델 dtype ("float16", "bfloat16", None=원본유지)

    Returns:
        (model, tokenizer, checkpoint_metadata, device_obj)
    """
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
    logger.info(f"Device: {device_obj}")
    if dtype_obj is not None:
        logger.info(f"dtype: {dtype_obj}")

    # Load checkpoint (inference_only=True: extra_heads 스킵으로 ~1.2GB 절약)
    logger.info("Loading checkpoint...")
    model, checkpoint_metadata = load_checkpoint_for_evaluation(
        checkpoint_path=Path(checkpoint_path),
        device=device_obj,
        inference_only=True,
        dtype=dtype_obj,
    )
    logger.info(f"Checkpoint epoch: {checkpoint_metadata['epoch']}")
    logger.info(f"Validation loss: {checkpoint_metadata['val_metrics']['val_loss']:.4f}")

    # LoRA 병합 (inference 최적화)
    from weighted_mtp.models.lora import merge_lora_weights
    try:
        merge_lora_weights(model)
        logger.info("LoRA weights merged for inference optimization")
    except Exception:
        logger.info("No LoRA weights to merge (or already merged)")

    # Load tokenizer — config 구조: {models: {policy: {path: "..."}}}
    config_dict = checkpoint_metadata.get('config', {})
    if 'models' in config_dict and 'policy' in config_dict['models']:
        model_path = config_dict['models']['policy']['path']
    elif 'model' in config_dict:
        model_path = config_dict['model']['path']
    else:
        model_path = "meta-llama/Meta-Llama-3-8B"
        logger.warning(f"Config에서 모델 경로를 찾을 수 없습니다. 기본값 사용: {model_path}")
    tokenizer_path = resolve_tokenizer_path(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    return model, tokenizer, checkpoint_metadata, device_obj


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
    model: Any | None = None,
    tokenizer: Any | None = None,
    checkpoint_metadata: dict | None = None,
    device_obj: torch.device | None = None,
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
        model: 외부에서 로드된 모델 (None이면 내부에서 로드)
        tokenizer: 외부에서 로드된 토크나이저 (None이면 내부에서 로드)
        checkpoint_metadata: 외부에서 로드된 메타데이터 (None이면 내부에서 로드)
        device_obj: 외부에서 설정된 device (None이면 내부에서 설정)

    Returns:
        {
            "pass_at_k": {"pass@1": 0.2, "pass@5": 0.65, ...},
            "per_task": [{"task_id": ..., "pass@1": ..., ...}, ...],
            "checkpoint_metadata": {...}
        }

    Examples:
        >>> # 단일 평가
        >>> results = run_evaluation(
        ...     checkpoint_path="storage/checkpoints/baseline/checkpoint_best.pt",
        ...     dataset_name="humaneval",
        ...     num_samples_per_task=20,
        ...     temperature=0.2,
        ... )
        >>> print(results["pass_at_k"])
        {'pass@1': 0.24, 'pass@5': 0.68, 'pass@10': 0.85}

        >>> # Temperature search (모델 재사용)
        >>> model, tokenizer, metadata, dev = load_model_for_evaluation(checkpoint_path)
        >>> for temp in [0.2, 0.8]:
        ...     results = run_evaluation(
        ...         checkpoint_path=checkpoint_path,
        ...         temperature=temp,
        ...         model=model,
        ...         tokenizer=tokenizer,
        ...         checkpoint_metadata=metadata,
        ...         device_obj=dev,
        ...     )
    """
    logger = setup_logging("EVALUATION")

    # 모델이 외부에서 주입되지 않은 경우 내부에서 로드
    if model is None or tokenizer is None or checkpoint_metadata is None:
        model, tokenizer, checkpoint_metadata, device_obj = load_model_for_evaluation(
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=dtype,
        )
    elif device_obj is None:
        # 모델은 주입되었지만 device_obj가 없는 경우
        if device == "auto":
            if torch.cuda.is_available():
                device_obj = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device_obj = torch.device("mps")
            else:
                device_obj = torch.device("cpu")
        else:
            device_obj = torch.device(device)

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Samples per task: {num_samples_per_task}")
    logger.info(f"Temperature: {temperature}")

    # 4. Load evaluation dataset
    logger.info(f"Loading {dataset_name} dataset...")
    dataset = load_evaluation_dataset(dataset_name, split="test")
    logger.info(f"Dataset size: {len(dataset)}")

    # 5. MLflow setup
    if mlflow_enabled:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("Default")
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

        # 데이터셋별 프롬프트 구성
        if dataset_name == "humaneval":
            # HumanEval: 이미 function signature + docstring 형식
            prompt = sample["instruction"]
        elif dataset_name == "codecontests":
            # CodeContests: Python 명시 + stdin/stdout 사용 안내
            instruction = sample["instruction"]
            instruction = f"Write a Python solution that reads from stdin and prints to stdout.\n\n{instruction}"
            prompt = apply_alpaca_template(
                instruction=instruction,
                input_text=sample.get("input", ""),
                output="",
                include_response_header=True,
            )
        else:
            # MBPP, GSM8K: Alpaca 템플릿
            prompt = apply_alpaca_template(
                instruction=sample["instruction"],
                input_text=sample.get("input", ""),
                output="",
                include_response_header=True,
            )

        # Generate N samples using HuggingFace model.generate() (batched)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(device_obj)
        attention_mask = inputs["attention_mask"].to(device_obj)
        prompt_len = input_ids.shape[1]

        generated_codes = []
        gen_batch_size = min(5, num_samples_per_task)  # OOM 방지용 배치 크기
        remaining = num_samples_per_task

        while remaining > 0:
            n = min(gen_batch_size, remaining)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids.expand(n, -1),
                    attention_mask=attention_mask.expand(n, -1),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=(temperature > 0),
                    top_p=0.95 if temperature > 0 else 1.0,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            # Decode each sample (exclude prompt)
            for i in range(n):
                generated_ids = output_ids[i, prompt_len:]
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                generated_codes.append(generated_text)
            remaining -= n

        # Execute and check (데이터셋별 분기)
        task_results = []
        task_partial_rates = []  # CodeContests용 partial pass rate 저장
        for sample_idx, code in enumerate(generated_codes):
            if dataset_name == "humaneval":
                # HumanEval: prompt(함수 시그니처) + 생성된 코드(함수 본문) = 완전한 함수
                # OpenAI/EvalPlus best practice: stop sequences에서 truncate, 중복 시그니처 제거
                clean_code = postprocess_humaneval_completion(code, prompt)
                full_code = prompt + clean_code
                test_code = sample["metadata"]["test"]
                entry_point = sample["metadata"]["entry_point"]
                passed = execute_code_with_tests(
                    code=full_code,
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
                # 평가용 데이터에서 task_id = problem_name (직접 사용)
                problem_name = task_id
                test_data = codecontests_tests.get(problem_name, {})

                # public + private + generated 통합, 최대 10개 샘플링 (seed=42 고정)
                sampled_tests = _sample_codecontests_tests(test_data, max_tests=10, seed=42)
                num_tests = len(sampled_tests.get("input", []))

                # 디버그: 현재 task, sample, test 개수 출력
                logger.info(f"[DEBUG] Task {idx+1}/{len(eval_dataset)}: {task_id} | Sample {sample_idx+1}/{num_samples_per_task} | Tests: {num_tests}")

                if sampled_tests["input"]:
                    result = execute_codecontests_tests(
                        code=code,
                        tests=sampled_tests,
                        timeout=5,
                    )
                    # 디버그: 테스트 결과 출력
                    logger.info(f"[DEBUG] Result: {result['passed']}/{result['total']} passed ({result['pass_rate']:.1%})")
                    # 모든 테스트 통과 시 pass
                    passed = result["pass_rate"] >= 1.0
                    # Partial pass rate 저장
                    task_partial_rates.append(result["pass_rate"])
                else:
                    # 테스트 케이스가 없으면 fail 처리
                    passed = False
                    task_partial_rates.append(0.0)
            else:
                raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
            task_results.append(passed)

        # Record
        task_record = {
            "task_id": task_id,
            "results": task_results,
            "num_correct": sum(task_results),
            "num_total": len(task_results),
        }
        # CodeContests는 partial pass rate도 저장
        if dataset_name == "codecontests" and task_partial_rates:
            task_record["partial_rates"] = task_partial_rates
            task_record["avg_partial_rate"] = sum(task_partial_rates) / len(task_partial_rates)
        all_results.append(task_record)

        # 처음 10개 task의 생성 샘플 저장 (MLflow artifact용)
        if idx < 10:
            generated_samples.append({
                "task_id": task_id,
                "prompt": prompt,
                "generated_codes": generated_codes,
                "results": task_results,
            })

        # 처음 3개 task의 생성 샘플 출력 (디버그용)
        if idx < 3 and generated_codes:
            sample_code = generated_codes[0][:300]
            logger.info(f"[SAMPLE] Task {task_id} - First 300 chars:\n{sample_code}\n{'='*50}")

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

    # CodeContests: Partial Pass Rate 로깅
    if dataset_name == "codecontests":
        partial_rates = [t.get("avg_partial_rate", 0.0) for t in all_results if "avg_partial_rate" in t]
        if partial_rates:
            avg_partial = sum(partial_rates) / len(partial_rates)
            logger.info(f"=== CodeContests Partial Pass Rate ===")
            logger.info(f"partial_pass_rate: {avg_partial:.2%}")

    if mlflow_enabled:
        # MLflow metric 이름에 @가 허용되지 않으므로 pass_at_k 형식으로 변환
        mlflow_metrics = {k.replace("@", "_at_"): v for k, v in pass_at_k_metrics.items()}
        mlflow.log_metrics(mlflow_metrics)

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


# ============================================================================
# EvalPlus 기반 평가 (HumanEval+, MBPP+)
# ============================================================================


def run_evalplus_evaluation(
    model: Any,
    tokenizer: Any,
    checkpoint_metadata: dict,
    device_obj: torch.device,
    dataset_name: str = "humaneval",
    num_samples: int = 1,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    output_dir: str = "storage/eval_results",
    max_tasks: int | None = None,
) -> dict[str, Any]:
    """EvalPlus 기반 평가 (HumanEval+/MBPP+)

    Phase 1: 솔루션 생성 → evalplus 호환 JSONL 저장
    Phase 2: evalplus evaluate 호출 → 결과 파싱

    Args:
        model: 평가 대상 모델
        tokenizer: 토크나이저
        checkpoint_metadata: 체크포인트 메타데이터
        device_obj: torch.device
        dataset_name: "humaneval" 또는 "mbpp"
        num_samples: 문제당 샘플 수 (greedy=1)
        temperature: sampling temperature (0=greedy)
        max_new_tokens: 최대 생성 토큰 수
        output_dir: 결과 저장 디렉토리
        max_tasks: 최대 문제 수 (None=전체)

    Returns:
        {"pass_at_k": {"pass@1": float, ...}, "evalplus": {"humaneval+": float, ...}, ...}
    """
    import subprocess as sp

    eval_logger = setup_logging("EVALPLUS")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # EvalPlus 데이터셋 로드
    eval_logger.info(f"Loading {dataset_name} problems from evalplus...")
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except ImportError:
        raise ImportError(
            "evalplus 패키지가 설치되지 않았습니다.\n"
            "설치: pip install evalplus"
        )

    if dataset_name == "humaneval":
        problems = get_human_eval_plus()
    elif dataset_name == "mbpp":
        problems = get_mbpp_plus()
    else:
        raise ValueError(f"evalplus는 humaneval/mbpp만 지원합니다: {dataset_name}")

    task_ids = sorted(problems.keys())
    if max_tasks:
        task_ids = task_ids[:max_tasks]
    eval_logger.info(f"Evaluating {len(task_ids)} tasks, {num_samples} samples each")

    # Phase 1: 솔루션 생성
    samples = []
    gen_batch_size = min(5, num_samples)

    for task_idx, task_id in enumerate(tqdm(task_ids, desc=f"Generating ({dataset_name})")):
        problem = problems[task_id]
        prompt = problem["prompt"]

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(device_obj)
        attention_mask = inputs["attention_mask"].to(device_obj)
        prompt_len = input_ids.shape[1]

        remaining = num_samples
        while remaining > 0:
            n = min(gen_batch_size, remaining)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids.expand(n, -1),
                    attention_mask=attention_mask.expand(n, -1),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=(temperature > 0),
                    top_p=0.95 if temperature > 0 else 1.0,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for i in range(n):
                generated_ids = output_ids[i, prompt_len:]
                completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # HumanEval 후처리: stop sequences에서 truncate
                if dataset_name == "humaneval":
                    completion = postprocess_humaneval_completion(completion, prompt)

                samples.append({
                    "task_id": task_id,
                    "completion": completion,
                })
            remaining -= n

        # 처음 3개 디버그 출력
        if task_idx < 3:
            preview = samples[-1]["completion"][:200]
            eval_logger.info(f"[SAMPLE] {task_id}:\n{preview}\n{'='*50}")

    # JSONL 저장
    samples_path = output_path / f"samples_{dataset_name}.jsonl"
    with open(samples_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    eval_logger.info(f"Saved {len(samples)} samples → {samples_path}")

    # Phase 2: evalplus evaluate 호출
    eval_logger.info("Running evalplus evaluation...")
    evalplus_dataset = "humaneval" if dataset_name == "humaneval" else "mbpp"

    cmd = [
        "evalplus.evaluate",
        "--dataset", evalplus_dataset,
        "--samples", str(samples_path),
    ]
    eval_logger.info(f"Command: {' '.join(cmd)}")

    result = sp.run(cmd, capture_output=True, text=True, timeout=600)
    eval_logger.info(f"evalplus stdout:\n{result.stdout}")
    if result.returncode != 0:
        eval_logger.warning(f"evalplus stderr:\n{result.stderr}")

    # Phase 3: 결과 파싱
    pass_at_k, evalplus_scores = _parse_evalplus_output(result.stdout, dataset_name)

    eval_logger.info("=== EvalPlus Results ===")
    for k, v in pass_at_k.items():
        eval_logger.info(f"{k}: {v:.2%}")
    for k, v in evalplus_scores.items():
        eval_logger.info(f"{k}: {v:.2%}")

    return {
        "pass_at_k": pass_at_k,
        "evalplus": evalplus_scores,
        "samples_path": str(samples_path),
        "checkpoint_metadata": checkpoint_metadata,
    }


def _parse_evalplus_output(stdout: str, dataset_name: str) -> tuple[dict, dict]:
    """evalplus CLI 출력 파싱

    Returns:
        (pass_at_k, evalplus_scores)
        pass_at_k: {"pass@1": float} (base 테스트)
        evalplus_scores: {"humaneval+_pass@1": float, ...} (augmented 테스트)
    """
    import re

    pass_at_k = {}
    evalplus_scores = {}

    # evalplus 출력 형식:
    # humaneval (base tests): pass@1: 0.7317
    # humaneval+ (base + extra tests): pass@1: 0.6585
    for line in stdout.split("\n"):
        line = line.strip()

        # Base tests
        match = re.match(r"(\w+)\s+\(base tests?\).*?pass@(\d+):\s+([\d.]+)", line, re.IGNORECASE)
        if match:
            k = int(match.group(2))
            score = float(match.group(3))
            pass_at_k[f"pass@{k}"] = score
            continue

        # Plus tests
        match = re.match(r"(\w+)\+?\s+\(base \+ extra tests?\).*?pass@(\d+):\s+([\d.]+)", line, re.IGNORECASE)
        if match:
            name = match.group(1).lower()
            k = int(match.group(2))
            score = float(match.group(3))
            evalplus_scores[f"{name}+_pass@{k}"] = score
            continue

    return pass_at_k, evalplus_scores
