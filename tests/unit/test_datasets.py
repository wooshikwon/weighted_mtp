"""datasets.py 기능 검증 테스트

핵심 검증 항목:
- 데이터셋 로딩 (codecontests, mbpp, humaneval)
- Stage 1 샘플링 (is_correct 균형)
- Stage 2 샘플링 (difficulty curriculum)
- 재현성 (seed 고정)
"""

import pytest
from datasets import Dataset

from weighted_mtp.data import load_dataset, apply_stage_sampling


class TestLoadDataset:
    """데이터셋 로딩 기본 기능 테스트"""

    def test_load_codecontests_train(self):
        """CodeContests train split 로딩"""
        dataset = load_dataset("codecontests", split="train")

        assert isinstance(dataset, Dataset)
        assert len(dataset) > 0

        # 필수 필드 검증
        sample = dataset[0]
        assert "instruction" in sample
        assert "input" in sample
        assert "output" in sample
        assert "task_id" in sample
        assert "is_correct" in sample
        assert "metadata" in sample
        assert "difficulty" in sample["metadata"]

    def test_load_codecontests_valid(self):
        """CodeContests validation split 로딩"""
        dataset = load_dataset("codecontests", split="validation")

        assert isinstance(dataset, Dataset)
        assert len(dataset) > 0

    def test_load_without_sampling(self):
        """샘플링 없이 전체 데이터셋 로딩"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage=None,
            n_samples=None,
        )

        # 전체 데이터셋이 로딩되어야 함
        assert len(dataset) > 100000

    def test_difficulty_field_parsing(self):
        """difficulty 필드가 정수로 파싱되는지 확인"""
        dataset = load_dataset("codecontests", split="train")

        for i in range(min(100, len(dataset))):
            sample = dataset[i]
            difficulty = sample["metadata"]["difficulty"]

            assert isinstance(difficulty, int)
            assert 1 <= difficulty <= 11


class TestStage1Sampling:
    """Stage 1: is_correct 균형 샘플링 테스트"""

    def test_stage1_basic_sampling(self):
        """Stage 1 기본 샘플링"""
        n_samples = 1000
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=n_samples,
            balance_correct=True,
            correct_ratio=0.5,
            seed=42,
        )

        assert len(dataset) == n_samples

    def test_stage1_correct_ratio(self):
        """Stage 1 is_correct 비율 검증 (50:50)"""
        n_samples = 1000
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=n_samples,
            balance_correct=True,
            correct_ratio=0.5,
            seed=42,
        )

        # is_correct 비율 계산
        correct_count = sum(1 for sample in dataset if sample["is_correct"])
        incorrect_count = len(dataset) - correct_count
        actual_ratio = correct_count / len(dataset)

        # 40-60% 범위 허용 (±10%)
        assert 0.4 <= actual_ratio <= 0.6, (
            f"is_correct 비율이 범위 밖: {actual_ratio:.2%} "
            f"(correct: {correct_count}, incorrect: {incorrect_count})"
        )

    def test_stage1_reproducibility(self):
        """Stage 1 재현성 검증 (seed 고정)"""
        # 같은 seed로 두 번 샘플링
        dataset1 = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=100,
            balance_correct=True,
            seed=42,
        )

        dataset2 = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=100,
            balance_correct=True,
            seed=42,
        )

        # task_id 비교 (동일한 샘플 선택)
        ids1 = [sample["task_id"] for sample in dataset1]
        ids2 = [sample["task_id"] for sample in dataset2]

        assert ids1 == ids2, "동일한 seed에도 다른 샘플이 선택됨"

    def test_stage1_without_balance(self):
        """Stage 1 균형 샘플링 없이 랜덤 샘플링"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=1000,
            balance_correct=False,
            seed=42,
        )

        assert len(dataset) == 1000


class TestStage2Sampling:
    """Stage 2: difficulty 기반 Curriculum Learning 샘플링 테스트"""

    def test_stage2_basic_sampling(self):
        """Stage 2 기본 샘플링"""
        n_samples = 1000
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=n_samples,
            difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        assert len(dataset) == n_samples

    def test_stage2_difficulty_distribution(self):
        """Stage 2 difficulty 분포 검증"""
        n_samples = 1000
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=n_samples,
            difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        # difficulty 분포 계산
        low_count = 0
        medium_count = 0
        high_count = 0

        for sample in dataset:
            diff = sample["metadata"]["difficulty"]
            if 1 <= diff <= 3:
                low_count += 1
            elif 4 <= diff <= 7:
                medium_count += 1
            elif 8 <= diff <= 11:
                high_count += 1

        low_ratio = low_count / len(dataset)
        medium_ratio = medium_count / len(dataset)

        # 가중치 ±15% 범위 허용
        assert 0.55 <= low_ratio <= 0.85, (
            f"low 비율이 범위 밖: {low_ratio:.2%} (목표: 70% ±15%)"
        )
        assert 0.15 <= medium_ratio <= 0.45, (
            f"medium 비율이 범위 밖: {medium_ratio:.2%} (목표: 30% ±15%)"
        )

    def test_stage2_reproducibility(self):
        """Stage 2 재현성 검증"""
        dataset1 = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=100,
            difficulty_weights={"low": 0.5, "medium": 0.5, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        dataset2 = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=100,
            difficulty_weights={"low": 0.5, "medium": 0.5, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        ids1 = [sample["task_id"] for sample in dataset1]
        ids2 = [sample["task_id"] for sample in dataset2]

        assert ids1 == ids2

    def test_stage2_high_difficulty_only(self):
        """Stage 2 고난이도만 샘플링"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=100,
            difficulty_weights={"low": 0.0, "medium": 0.0, "high": 1.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        # 모든 샘플이 고난이도여야 함
        for sample in dataset:
            diff = sample["metadata"]["difficulty"]
            assert 8 <= diff <= 11, f"고난이도가 아닌 샘플: difficulty={diff}"


class TestApplyStageSampling:
    """apply_stage_sampling() 함수 직접 테스트"""

    def test_apply_stage1_to_dataset(self):
        """Dataset에 Stage 1 샘플링 직접 적용"""
        # 전체 데이터셋 로딩
        full_dataset = load_dataset("codecontests", split="train")

        # Stage 1 샘플링 적용
        sampled = apply_stage_sampling(
            dataset=full_dataset,
            stage="stage1",
            n_samples=500,
            balance_correct=True,
            correct_ratio=0.5,
            seed=42,
        )

        assert len(sampled) == 500

    def test_apply_stage2_to_dataset(self):
        """Dataset에 Stage 2 샘플링 직접 적용"""
        full_dataset = load_dataset("codecontests", split="train")

        sampled = apply_stage_sampling(
            dataset=full_dataset,
            stage="stage2",
            n_samples=500,
            difficulty_weights={"low": 0.6, "medium": 0.4, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        assert len(sampled) == 500
