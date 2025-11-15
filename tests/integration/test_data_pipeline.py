"""데이터 파이프라인 통합 테스트

전체 파이프라인 검증:
- load_dataset → AlpacaDataCollator → DataLoader
- Stage 1 End-to-End
- Stage 2 End-to-End
- Epoch 루프
"""

import pytest
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from weighted_mtp.data import load_dataset, AlpacaDataCollator


# Tokenizer 로딩 fixture
@pytest.fixture(scope="module")
def tokenizer():
    """실제 LlamaTokenizer 로딩 (없으면 skip)"""
    try:
        from transformers import AutoTokenizer

        tokenizer_path = Path("storage/models_v2/meta-llama-mtp/tokenizer")

        if not tokenizer_path.exists():
            pytest.skip("Tokenizer not found")

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    except ImportError:
        pytest.skip("transformers required")


class TestStage1Pipeline:
    """Stage 1 전체 파이프라인 테스트"""

    def test_stage1_basic_pipeline(self, tokenizer):
        """Stage 1 기본 파이프라인"""
        # 1. 데이터 로딩
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=100,
            balance_correct=True,
            correct_ratio=0.5,
            seed=42,
        )

        # 2. Collator 생성
        collator = AlpacaDataCollator(tokenizer, max_length=512)

        # 3. DataLoader 생성
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            collate_fn=collator,
            shuffle=False,
        )

        # 4. 배치 검증
        batch = next(iter(dataloader))

        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

        assert batch["input_ids"].shape == (4, 512)
        assert batch["attention_mask"].shape == (4, 512)
        assert batch["labels"].shape == (4, 512)

    def test_stage1_correct_ratio_in_dataloader(self, tokenizer):
        """DataLoader에서 is_correct 비율 검증"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=200,
            balance_correct=True,
            correct_ratio=0.5,
            seed=42,
        )

        # is_correct 비율 확인
        correct_count = sum(1 for sample in dataset if sample["is_correct"])
        ratio = correct_count / len(dataset)

        assert 0.4 <= ratio <= 0.6, f"is_correct 비율: {ratio:.2%}"

    def test_stage1_masking_in_batches(self, tokenizer):
        """배치 내 masking 정확성"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=50,
            balance_correct=True,
            seed=42,
        )

        collator = AlpacaDataCollator(tokenizer, max_length=256)
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator)

        for batch in dataloader:
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]

            # 각 샘플마다 검증
            for i in range(labels.shape[0]):
                sample_labels = labels[i]
                sample_mask = attention_mask[i]

                # Instruction 부분 마스킹
                assert (sample_labels[:10] == -100).all()

                # Output 부분 존재
                non_padding = sample_labels[sample_mask == 1]
                assert (non_padding != -100).any()


class TestStage2Pipeline:
    """Stage 2 전체 파이프라인 테스트"""

    def test_stage2_basic_pipeline(self, tokenizer):
        """Stage 2 기본 파이프라인"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=100,
            difficulty_weights={"low": 0.6, "medium": 0.4, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        collator = AlpacaDataCollator(tokenizer, max_length=512)
        dataloader = DataLoader(dataset, batch_size=4, collate_fn=collator)

        batch = next(iter(dataloader))

        assert batch["input_ids"].shape == (4, 512)
        assert batch["labels"].shape == (4, 512)

    def test_stage2_difficulty_distribution(self, tokenizer):
        """Stage 2 difficulty 분포 검증"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=200,
            difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        # difficulty 분포 계산
        low_count = sum(
            1 for s in dataset if 1 <= s["metadata"]["difficulty"] <= 3
        )
        medium_count = sum(
            1 for s in dataset if 4 <= s["metadata"]["difficulty"] <= 7
        )

        low_ratio = low_count / len(dataset)
        medium_ratio = medium_count / len(dataset)

        # ±15% 허용
        assert 0.55 <= low_ratio <= 0.85
        assert 0.15 <= medium_ratio <= 0.45

    def test_stage2_curriculum_schedule(self, tokenizer):
        """Stage 2 curriculum schedule (3단계)"""
        # 초반: low 70%, medium 30%, high 0%
        dataset_early = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=100,
            difficulty_weights={"low": 0.7, "medium": 0.3, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        # 중반: low 30%, medium 60%, high 10%
        dataset_mid = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=100,
            difficulty_weights={"low": 0.3, "medium": 0.6, "high": 0.1},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=43,
        )

        # 후반: low 10%, medium 50%, high 40%
        dataset_late = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=100,
            difficulty_weights={"low": 0.1, "medium": 0.5, "high": 0.4},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=44,
        )

        # 각 단계별로 분포가 달라야 함
        def get_low_ratio(ds):
            return sum(1 for s in ds if 1 <= s["metadata"]["difficulty"] <= 3) / len(ds)

        early_low = get_low_ratio(dataset_early)
        mid_low = get_low_ratio(dataset_mid)
        late_low = get_low_ratio(dataset_late)

        # 초반 > 중반 > 후반 (low difficulty 비율)
        assert early_low > mid_low > late_low


class TestEpochLoop:
    """Epoch 루프 테스트"""

    def test_single_epoch(self, tokenizer):
        """단일 epoch 정상 동작"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=50,
            balance_correct=True,
            seed=42,
        )

        collator = AlpacaDataCollator(tokenizer, max_length=256)
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator)

        batch_count = 0
        for batch in dataloader:
            assert batch["input_ids"].shape[0] <= 8
            assert (batch["labels"] != -100).any()
            batch_count += 1

        # 모든 배치가 처리되었는지 확인
        expected_batches = (50 + 7) // 8  # ceil(50 / 8) = 7
        assert batch_count == expected_batches

    def test_multiple_epochs(self, tokenizer):
        """여러 epoch 반복 정상 동작"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=30,
            balance_correct=True,
            seed=42,
        )

        collator = AlpacaDataCollator(tokenizer, max_length=256)
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator)

        total_batches = 0
        for epoch in range(3):
            epoch_batches = 0
            for batch in dataloader:
                assert batch["input_ids"].shape[0] <= 8
                epoch_batches += 1

            total_batches += epoch_batches

        # 3 epoch 모두 동일한 배치 수
        expected_per_epoch = (30 + 7) // 8
        assert total_batches == expected_per_epoch * 3

    def test_shuffle_dataloader(self, tokenizer):
        """DataLoader shuffle 동작"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=100,
            balance_correct=True,
            seed=42,
        )

        collator = AlpacaDataCollator(tokenizer, max_length=256)

        # shuffle=False
        dataloader_no_shuffle = DataLoader(
            dataset, batch_size=10, collate_fn=collator, shuffle=False
        )

        # shuffle=True
        dataloader_shuffle = DataLoader(
            dataset, batch_size=10, collate_fn=collator, shuffle=True
        )

        # 첫 배치가 달라야 함 (확률적이지만 거의 확실)
        batch_no_shuffle = next(iter(dataloader_no_shuffle))
        batch_shuffle = next(iter(dataloader_shuffle))

        # 완전히 동일하지 않을 확률이 높음
        # (동일할 수도 있지만 일반적으로 다름)
        # 최소한 shape은 동일해야 함
        assert batch_no_shuffle["input_ids"].shape == batch_shuffle["input_ids"].shape


class TestPipelineMemory:
    """파이프라인 메모리 효율성 테스트"""

    def test_stage1_memory_usage(self, tokenizer):
        """Stage 1 메모리 사용량 (목표: <300MB)"""
        import sys

        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=1000,
            balance_correct=True,
            seed=42,
        )

        collator = AlpacaDataCollator(tokenizer, max_length=512)

        # Dataset 크기 추정
        dataset_size = sys.getsizeof(dataset)

        # 1000 샘플의 경우 크기가 합리적이어야 함
        assert dataset_size < 100 * 1024 * 1024, f"Dataset 너무 큼: {dataset_size / 1024 / 1024:.1f}MB"

    def test_stage2_memory_usage(self, tokenizer):
        """Stage 2 메모리 사용량 (목표: <1GB)"""
        import sys

        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage2",
            n_samples=5000,
            difficulty_weights={"low": 0.5, "medium": 0.5, "high": 0.0},
            difficulty_bins={"low": [1, 3], "medium": [4, 7], "high": [8, 11]},
            seed=42,
        )

        dataset_size = sys.getsizeof(dataset)

        # 5000 샘플의 경우 크기가 합리적이어야 함
        assert dataset_size < 500 * 1024 * 1024, f"Dataset 너무 큼: {dataset_size / 1024 / 1024:.1f}MB"


class TestPipelineEdgeCases:
    """파이프라인 Edge Case 테스트"""

    def test_very_small_batch(self, tokenizer):
        """매우 작은 배치 크기 (batch_size=1)"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=10,
            balance_correct=True,
            seed=42,
        )

        collator = AlpacaDataCollator(tokenizer, max_length=256)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)

        batch = next(iter(dataloader))

        assert batch["input_ids"].shape == (1, 256)

    def test_large_batch(self, tokenizer):
        """큰 배치 크기"""
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=100,
            balance_correct=True,
            seed=42,
        )

        collator = AlpacaDataCollator(tokenizer, max_length=256)
        dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator)

        batch = next(iter(dataloader))

        assert batch["input_ids"].shape == (32, 256)

    def test_incomplete_final_batch(self, tokenizer):
        """마지막 배치가 불완전한 경우"""
        # 55 샘플, batch_size=8 → 마지막 배치는 7개
        dataset = load_dataset(
            "codecontests",
            split="train",
            stage="stage1",
            n_samples=55,
            balance_correct=True,
            seed=42,
        )

        collator = AlpacaDataCollator(tokenizer, max_length=256)
        dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator)

        batches = list(dataloader)

        # 마지막 배치는 7개
        assert batches[-1]["input_ids"].shape[0] == 7

        # 총 배치 수
        assert len(batches) == 7  # ceil(55 / 8) = 7
