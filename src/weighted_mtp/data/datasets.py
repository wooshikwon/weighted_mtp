"""JSONL → HuggingFace Dataset 로딩 및 Stage별 샘플링

핵심 기능:
- CodeContests, MBPP, HumanEval JSONL 파일 로딩
- Stage 1: is_correct 균형 샘플링 (50:50)
- Stage 2: difficulty 기반 Curriculum Learning
- 재현성을 위한 seed 고정
"""

from pathlib import Path
from typing import Literal, Optional
import logging
import random

from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset
import numpy as np

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: Literal["codecontests", "mbpp", "humaneval"],
    split: Optional[str] = None,
    stage: Optional[Literal["stage1", "stage2"]] = None,
    n_samples: Optional[int] = None,
    balance_correct: bool = False,
    correct_ratio: float = 0.5,
    difficulty_weights: Optional[dict] = None,
    difficulty_bins: Optional[dict] = None,
    seed: int = 42,
) -> Dataset | DatasetDict:
    """JSONL 파일을 HuggingFace Dataset으로 로딩하고 Stage별 샘플링 적용

    Args:
        dataset_name: 데이터셋 이름 (codecontests, mbpp, humaneval)
        split: 데이터 스플릿 (train, validation, test). None이면 모든 스플릿 로딩
        stage: 샘플링 전략 (stage1: is_correct 균형, stage2: difficulty curriculum)
        n_samples: 샘플링할 샘플 수
        balance_correct: is_correct 균형 샘플링 여부 (Stage 1 전용)
        correct_ratio: correct 샘플 비율 (기본 0.5)
        difficulty_weights: 난이도별 가중치 (Stage 2 전용)
        difficulty_bins: 난이도 구간 정의 (Stage 2 전용)
        seed: 랜덤 시드

    Returns:
        Dataset 또는 DatasetDict (split이 None인 경우)

    Examples:
        >>> # Stage 1: is_correct 균형 샘플링
        >>> dataset = load_dataset(
        ...     "codecontests",
        ...     split="train",
        ...     stage="stage1",
        ...     n_samples=50000,
        ...     balance_correct=True,
        ...     seed=42
        ... )
        >>>
        >>> # Stage 2: difficulty curriculum
        >>> dataset = load_dataset(
        ...     "codecontests",
        ...     split="train",
        ...     stage="stage2",
        ...     n_samples=200000,
        ...     difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
        ...     difficulty_bins={"low": [1,3], "medium": [4,7], "high": [8,11]},
        ...     seed=42
        ... )
    """
    # JSONL 경로 해석
    data_files = _get_dataset_paths(dataset_name)

    # HuggingFace load_dataset 호출
    logger.info(f"데이터셋 로딩 시작: {dataset_name}")

    if split is not None:
        # 단일 스플릿 로딩
        if split not in data_files:
            raise ValueError(
                f"스플릿 '{split}'이 존재하지 않습니다. "
                f"가능한 스플릿: {list(data_files.keys())}"
            )

        dataset = hf_load_dataset("json", data_files=data_files[split], split="train")
        logger.info(f"데이터셋 로딩 완료: {dataset_name}/{split} - {len(dataset)} 샘플")

        # Stage별 샘플링 적용
        if stage is not None and n_samples is not None:
            dataset = apply_stage_sampling(
                dataset=dataset,
                stage=stage,
                n_samples=n_samples,
                balance_correct=balance_correct,
                correct_ratio=correct_ratio,
                difficulty_weights=difficulty_weights,
                difficulty_bins=difficulty_bins,
                seed=seed,
            )

        return dataset
    else:
        # 모든 스플릿 로딩 (DatasetDict)
        dataset_dict = hf_load_dataset("json", data_files=data_files)
        total_samples = sum(len(ds) for ds in dataset_dict.values())
        logger.info(f"데이터셋 로딩 완료: {dataset_name} - {total_samples} 샘플")
        return dataset_dict


def apply_stage_sampling(
    dataset: Dataset,
    stage: Literal["stage1", "stage2"],
    n_samples: int,
    balance_correct: bool = False,
    correct_ratio: float = 0.5,
    difficulty_weights: Optional[dict] = None,
    difficulty_bins: Optional[dict] = None,
    seed: int = 42,
) -> Dataset:
    """Stage별 샘플링 전략 적용

    Args:
        dataset: 원본 HuggingFace Dataset
        stage: 샘플링 전략 (stage1 또는 stage2)
        n_samples: 샘플링할 샘플 수
        balance_correct: is_correct 균형 샘플링 여부
        correct_ratio: correct 샘플 비율
        difficulty_weights: 난이도별 가중치 (Stage 2)
        difficulty_bins: 난이도 구간 정의 (Stage 2)
        seed: 랜덤 시드

    Returns:
        샘플링된 Dataset
    """
    if stage == "stage1":
        return _sample_stage1(
            dataset=dataset,
            n_samples=n_samples,
            balance_correct=balance_correct,
            correct_ratio=correct_ratio,
            seed=seed,
        )
    elif stage == "stage2":
        return _sample_stage2(
            dataset=dataset,
            n_samples=n_samples,
            difficulty_weights=difficulty_weights,
            difficulty_bins=difficulty_bins,
            seed=seed,
        )
    else:
        raise ValueError(f"지원하지 않는 stage: {stage}")


def _sample_stage1(
    dataset: Dataset,
    n_samples: int,
    balance_correct: bool,
    correct_ratio: float,
    seed: int,
) -> Dataset:
    """Stage 1: is_correct 균형 샘플링

    correct와 incorrect 샘플을 지정된 비율로 균형있게 샘플링합니다.

    Args:
        dataset: 원본 Dataset
        n_samples: 샘플링할 총 샘플 수
        balance_correct: 균형 샘플링 여부
        correct_ratio: correct 샘플 비율 (0.0 ~ 1.0)
        seed: 랜덤 시드

    Returns:
        샘플링된 Dataset
    """
    random.seed(seed)
    np.random.seed(seed)

    if not balance_correct:
        # 균형 샘플링 없이 랜덤 샘플링
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        return dataset.select(indices)

    # is_correct 필드 존재 확인
    sample = dataset[0]
    if "is_correct" not in sample:
        logger.warning("is_correct 필드가 없습니다. 랜덤 샘플링으로 전환합니다.")
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        return dataset.select(indices)

    # correct/incorrect 샘플 인덱스 분리
    correct_indices = []
    incorrect_indices = []

    for idx in range(len(dataset)):
        if dataset[idx]["is_correct"]:
            correct_indices.append(idx)
        else:
            incorrect_indices.append(idx)

    logger.info(
        f"전체 샘플: {len(dataset)}, "
        f"correct: {len(correct_indices)}, "
        f"incorrect: {len(incorrect_indices)}"
    )

    # 샘플 수 계산
    n_correct = int(n_samples * correct_ratio)
    n_incorrect = n_samples - n_correct

    # 각 그룹에서 샘플링
    selected_correct = random.sample(
        correct_indices, min(n_correct, len(correct_indices))
    )
    selected_incorrect = random.sample(
        incorrect_indices, min(n_incorrect, len(incorrect_indices))
    )

    # 병합 및 섞기
    selected_indices = selected_correct + selected_incorrect
    random.shuffle(selected_indices)

    sampled_dataset = dataset.select(selected_indices)

    # 실제 비율 계산
    actual_correct_ratio = len(selected_correct) / len(selected_indices)
    logger.info(
        f"Stage 1 샘플링 완료: {len(selected_indices)} 샘플 "
        f"(correct: {len(selected_correct)}, incorrect: {len(selected_incorrect)}, "
        f"비율: {actual_correct_ratio:.2%})"
    )

    return sampled_dataset


def _sample_stage2(
    dataset: Dataset,
    n_samples: int,
    difficulty_weights: Optional[dict],
    difficulty_bins: Optional[dict],
    seed: int,
) -> Dataset:
    """Stage 2: difficulty 기반 Curriculum Learning 샘플링

    난이도 구간별로 가중치를 적용하여 샘플링합니다.

    Args:
        dataset: 원본 Dataset
        n_samples: 샘플링할 총 샘플 수
        difficulty_weights: 난이도별 가중치 (예: {"low": 0.7, "medium": 0.3, "high": 0.0})
        difficulty_bins: 난이도 구간 (예: {"low": [1,3], "medium": [4,7], "high": [8,11]})
        seed: 랜덤 시드

    Returns:
        샘플링된 Dataset
    """
    random.seed(seed)
    np.random.seed(seed)

    if difficulty_weights is None or difficulty_bins is None:
        # 가중치 없이 랜덤 샘플링
        logger.warning(
            "difficulty_weights 또는 difficulty_bins가 없습니다. "
            "랜덤 샘플링으로 전환합니다."
        )
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        return dataset.select(indices)

    # difficulty 필드 존재 확인
    sample = dataset[0]
    if "metadata" not in sample or "difficulty" not in sample["metadata"]:
        logger.warning("difficulty 필드가 없습니다. 랜덤 샘플링으로 전환합니다.")
        indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
        return dataset.select(indices)

    # 난이도 구간별 인덱스 분리
    bin_indices = {bin_name: [] for bin_name in difficulty_bins.keys()}

    for idx in range(len(dataset)):
        difficulty = dataset[idx]["metadata"]["difficulty"]

        # 해당 난이도가 어느 bin에 속하는지 판단
        for bin_name, (min_diff, max_diff) in difficulty_bins.items():
            if min_diff <= difficulty <= max_diff:
                bin_indices[bin_name].append(idx)
                break

    # 각 bin별 샘플 수 로그
    total_in_bins = sum(len(indices) for indices in bin_indices.values())
    bin_stats = ", ".join(
        [f"{name}: {len(indices)}" for name, indices in bin_indices.items()]
    )
    logger.info(f"난이도 분포: {bin_stats} (총 {total_in_bins})")

    # 가중치에 따라 각 bin에서 샘플링
    selected_indices = []

    for bin_name, weight in difficulty_weights.items():
        if weight <= 0:
            continue

        bin_n_samples = int(n_samples * weight)
        available_indices = bin_indices.get(bin_name, [])

        if len(available_indices) == 0:
            logger.warning(f"{bin_name} bin에 샘플이 없습니다.")
            continue

        # 샘플링
        sampled = random.sample(
            available_indices, min(bin_n_samples, len(available_indices))
        )
        selected_indices.extend(sampled)

        logger.info(f"{bin_name} bin에서 {len(sampled)} 샘플 선택 (목표: {bin_n_samples})")

    # 샘플 수가 부족하면 랜덤으로 추가
    if len(selected_indices) < n_samples:
        remaining = n_samples - len(selected_indices)
        all_indices = list(range(len(dataset)))
        available = [idx for idx in all_indices if idx not in selected_indices]

        if available:
            additional = random.sample(available, min(remaining, len(available)))
            selected_indices.extend(additional)
            logger.warning(f"샘플 부족으로 {len(additional)} 샘플 추가 (랜덤)")

    # 섞기
    random.shuffle(selected_indices)

    # 정확히 n_samples 개수 맞추기
    selected_indices = selected_indices[:n_samples]

    sampled_dataset = dataset.select(selected_indices)

    logger.info(f"Stage 2 샘플링 완료: {len(selected_indices)} 샘플")

    return sampled_dataset


def _get_dataset_paths(dataset_name: str) -> dict[str, str]:
    """데이터셋 이름으로 JSONL 파일 경로 해석

    Args:
        dataset_name: 데이터셋 이름

    Returns:
        스플릿별 JSONL 파일 경로 딕셔너리
    """
    base_dir = Path("storage/datasets_v2")
    dataset_dir = base_dir / dataset_name / "processed"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"데이터셋 디렉터리가 존재하지 않습니다: {dataset_dir}")

    # 표준 스플릿 이름 매핑
    split_mappings = {
        "train": ["train.jsonl"],
        "validation": ["valid.jsonl", "validation.jsonl"],
        "test": ["test.jsonl"],
    }

    data_files = {}

    for split_name, candidates in split_mappings.items():
        for candidate in candidates:
            file_path = dataset_dir / candidate
            if file_path.exists():
                data_files[split_name] = str(file_path)
                break

    if not data_files:
        raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {dataset_dir}")

    return data_files
