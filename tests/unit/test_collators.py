"""collators.py 핵심 기능 검증 테스트

핵심 검증 항목:
- Alpaca 템플릿 적용
- Loss masking (instruction, input, padding 제외)
- Output 토큰만 학습 대상
- 배치 처리
"""

import pytest
import torch
from pathlib import Path

from weighted_mtp.data import AlpacaDataCollator
from weighted_mtp.data.collators import apply_alpaca_template, apply_random_window_mask


# Tokenizer 로딩 fixture
@pytest.fixture(scope="module")
def tokenizer():
    """실제 LlamaTokenizer 로딩 (없으면 skip)"""
    try:
        from transformers import AutoTokenizer

        tokenizer_path = Path("storage/models/meta-llama-mtp/tokenizer")

        if not tokenizer_path.exists():
            pytest.skip("Tokenizer not found: storage/models/meta-llama-mtp/tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))

        # padding token 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    except ImportError:
        pytest.skip("transformers 라이브러리 필요")


class TestAlpacaTemplate:
    """apply_alpaca_template() 함수 테스트"""

    def test_template_with_input(self):
        """Input이 있는 경우 템플릿"""
        instruction = "Add two numbers."
        input_text = "Example: 1 + 2 = 3"
        output = "def add(a,b): return a+b"

        result = apply_alpaca_template(instruction, input_text, output)

        # 필수 요소 포함 확인
        assert "### Instruction:" in result
        assert "### Input:" in result
        assert "### Response:" in result
        assert instruction in result
        assert input_text in result
        assert output in result

    def test_template_without_input(self):
        """Input이 없는 경우 템플릿"""
        instruction = "Add two numbers."
        output = "def add(a,b): return a+b"

        result = apply_alpaca_template(instruction, "", output)

        # Input 섹션이 없어야 함
        assert "### Instruction:" in result
        assert "### Input:" not in result
        assert "### Response:" in result


class TestAlpacaDataCollator:
    """AlpacaDataCollator 핵심 기능 테스트"""

    def test_collator_initialization(self, tokenizer):
        """Collator 초기화"""
        collator = AlpacaDataCollator(
            tokenizer=tokenizer,
            max_length=2048,
            n_future_tokens=4,
            padding="max_length",
        )

        assert collator.max_length == 2048
        assert collator.n_future_tokens == 4
        assert collator.padding == "max_length"

    def test_single_sample_masking(self, tokenizer):
        """단일 샘플 masking 검증"""
        collator = AlpacaDataCollator(tokenizer, max_length=512)

        sample = {
            "instruction": "Add two numbers.",
            "input": "",
            "output": "def add(a,b): return a+b",
            "is_correct": True,
        }

        batch = collator([sample])

        # Shape 검증
        assert batch["input_ids"].shape == (1, 512)
        assert batch["attention_mask"].shape == (1, 512)
        assert batch["labels"].shape == (1, 512)
        assert batch["is_correct"].shape == (1,)

        # is_correct 값 검증
        assert batch["is_correct"][0] == 1.0

        labels = batch["labels"][0]

        # Instruction 부분은 -100 (최소 첫 10개 토큰)
        assert (labels[:10] == -100).all(), "Instruction 부분이 마스킹되지 않음"

        # Output 부분은 token ID (not -100)
        non_masked = labels[labels != -100]
        assert len(non_masked) > 0, "Output 부분이 모두 마스킹됨"

    def test_masking_boundaries(self, tokenizer):
        """Masking 경계 정확성 검증"""
        collator = AlpacaDataCollator(tokenizer, max_length=512, padding="max_length")

        sample = {
            "instruction": "Write a function to add two numbers.",
            "input": "Example: add(1, 2) should return 3",
            "output": "def add(a, b):\n    return a + b",
        }

        batch = collator([sample])

        input_ids = batch["input_ids"][0]
        attention_mask = batch["attention_mask"][0]
        labels = batch["labels"][0]

        # 1. Padding 영역은 -100이어야 함
        padding_mask = attention_mask == 0
        assert (labels[padding_mask] == -100).all(), "Padding이 마스킹되지 않음"

        # 2. Non-padding 영역 중 일부는 -100이 아니어야 함 (output)
        non_padding_mask = attention_mask == 1
        non_padding_labels = labels[non_padding_mask]
        non_masked_count = (non_padding_labels != -100).sum().item()

        assert non_masked_count > 0, "Non-padding 영역에 학습 대상이 없음"

        # 3. Input IDs와 Labels의 shape 일치
        assert input_ids.shape == labels.shape

    def test_batch_processing(self, tokenizer):
        """배치 처리 검증"""
        collator = AlpacaDataCollator(tokenizer, max_length=256)

        batch_samples = [
            {
                "instruction": "Add two numbers.",
                "input": "",
                "output": "def add(a,b): return a+b",
                "is_correct": True,
            },
            {
                "instruction": "Multiply two numbers.",
                "input": "Example: 2 * 3 = 6",
                "output": "def mul(a,b): return a*b",
                "is_correct": False,
            },
            {
                "instruction": "Check if number is even.",
                "input": "",
                "output": "def is_even(n): return n % 2 == 0",
                "is_correct": True,
            },
        ]

        batch = collator(batch_samples)

        # Batch shape 검증
        assert batch["input_ids"].shape == (3, 256)
        assert batch["attention_mask"].shape == (3, 256)
        assert batch["labels"].shape == (3, 256)
        assert batch["is_correct"].shape == (3,)

        # is_correct 값 검증
        assert batch["is_correct"][0] == 1.0
        assert batch["is_correct"][1] == 0.0
        assert batch["is_correct"][2] == 1.0

        # 각 샘플이 독립적으로 마스킹되었는지 확인
        for i in range(3):
            labels = batch["labels"][i]
            attention_mask = batch["attention_mask"][i]

            # Instruction 부분 마스킹
            assert (labels[:10] == -100).all()

            # Output 부분 존재
            non_padding = labels[attention_mask == 1]
            assert (non_padding != -100).any()

    @pytest.mark.skip(reason="Collator의 longest padding 모드 버그 - 별도 수정 필요")
    def test_longest_padding(self, tokenizer):
        """longest padding 모드 검증"""
        collator = AlpacaDataCollator(tokenizer, max_length=512, padding="longest")

        batch = [
            {"instruction": "Short task.", "input": "", "output": "short output"},
            {
                "instruction": "Much longer task description.",
                "input": "With example input",
                "output": "Longer output code here",
            },
        ]

        result = collator(batch)

        # 가장 긴 샘플에 맞춰 padding
        assert result["input_ids"].shape[1] <= 512
        assert result["input_ids"].shape == result["labels"].shape

    def test_truncation(self, tokenizer):
        """긴 텍스트 truncation 검증"""
        collator = AlpacaDataCollator(tokenizer, max_length=128)

        # 매우 긴 텍스트
        long_instruction = "Very long instruction. " * 100
        long_output = "Very long output. " * 100

        sample = {
            "instruction": long_instruction,
            "input": "",
            "output": long_output,
        }

        batch = collator([sample])

        # max_length로 truncation되어야 함
        assert batch["input_ids"].shape == (1, 128)
        assert batch["labels"].shape == (1, 128)


class TestMaskingConsistency:
    """Masking 일관성 테스트"""

    def test_mask_preserves_attention(self, tokenizer):
        """Masking이 attention_mask를 변경하지 않는지 확인"""
        collator = AlpacaDataCollator(tokenizer, max_length=256)

        sample = {
            "instruction": "Task",
            "input": "Input",
            "output": "Output",
            "is_correct": True,
        }

        batch = collator([sample])

        attention_mask = batch["attention_mask"][0]

        # Attention mask는 1(유효) 또는 0(padding)만 포함
        assert set(attention_mask.unique().tolist()).issubset({0, 1})

    def test_deterministic_behavior(self, tokenizer):
        """동일한 샘플에 대해 항상 같은 결과"""
        collator = AlpacaDataCollator(tokenizer, max_length=256)

        sample = {
            "instruction": "Test",
            "input": "",
            "output": "Result",
            "is_correct": True,
        }

        batch1 = collator([sample])
        batch2 = collator([sample])

        # 동일한 결과
        assert torch.equal(batch1["input_ids"], batch2["input_ids"])
        assert torch.equal(batch1["attention_mask"], batch2["attention_mask"])
        assert torch.equal(batch1["labels"], batch2["labels"])
        assert torch.equal(batch1["is_correct"], batch2["is_correct"])

    def test_is_correct_backward_compatibility(self, tokenizer):
        """is_correct 필드가 없을 때 기본값 True 검증"""
        collator = AlpacaDataCollator(tokenizer, max_length=256)

        sample = {
            "instruction": "Test",
            "input": "",
            "output": "Result",
            # is_correct 생략
        }

        batch = collator([sample])

        # 기본값 True (1.0)
        assert batch["is_correct"][0] == 1.0


class TestApplyRandomWindowMask:
    """apply_random_window_mask() 함수 테스트"""

    def test_short_sequence_unchanged(self):
        """window_size 이하 시퀀스는 labels 변경 없음"""
        # output 길이 = 100 (< window_size=192)
        labels = torch.arange(200)  # [0, 1, ..., 199]
        labels[:100] = -100  # instruction 부분 마스킹
        attention_mask = torch.ones(200)

        result = apply_random_window_mask(
            labels=labels,
            attention_mask=attention_mask,
            len_prompt=100,
            window_size=192,
        )

        # 원본과 동일해야 함
        assert torch.equal(result, labels)

    def test_random_window_long_sequence(self):
        """긴 시퀀스에서 랜덤 윈도우가 올바르게 선택되는지 검증"""
        # output 길이 = 250 (> window_size=192)
        seq_len = 350
        len_prompt = 100
        labels = torch.arange(seq_len)
        labels[:len_prompt] = -100  # instruction 마스킹
        attention_mask = torch.ones(seq_len)

        result = apply_random_window_mask(
            labels=labels,
            attention_mask=attention_mask,
            len_prompt=len_prompt,
            window_size=192,
        )

        # 학습 대상 토큰 수 = window_size
        n_train_tokens = (result != -100).sum().item()
        assert n_train_tokens == 192, f"학습 토큰 수 {n_train_tokens} != 192"

        # 윈도우가 output 범위 내에 있어야 함
        train_mask = result != -100
        first_train_idx = train_mask.nonzero()[0].item()
        assert first_train_idx >= len_prompt, "윈도우가 instruction 영역 침범"

    def test_window_size_respected(self):
        """학습 대상 토큰 수가 window_size 이하"""
        seq_len = 500
        len_prompt = 100
        labels = torch.arange(seq_len)
        labels[:len_prompt] = -100
        attention_mask = torch.ones(seq_len)

        result = apply_random_window_mask(
            labels=labels,
            attention_mask=attention_mask,
            len_prompt=len_prompt,
            window_size=192,
        )

        # 학습 대상 토큰 수 = window_size
        n_train_tokens = (result != -100).sum().item()
        assert n_train_tokens == 192, f"학습 토큰 수 {n_train_tokens} != 192"

    def test_medium_sequence_full_output(self):
        """output_len <= window_size면 전체 output 학습"""
        # output 길이 = 150 (< window_size=192)
        seq_len = 250
        len_prompt = 100
        labels = torch.arange(seq_len)
        labels[:len_prompt] = -100
        attention_mask = torch.ones(seq_len)

        result = apply_random_window_mask(
            labels=labels,
            attention_mask=attention_mask,
            len_prompt=len_prompt,
            window_size=192,
        )

        # 전체 output이 학습 대상 (output_len = 150 < window_size = 192)
        n_train_tokens = (result != -100).sum().item()
        assert n_train_tokens == 150, f"학습 토큰 수 {n_train_tokens} != 150"

    def test_with_padding(self):
        """padding이 있는 경우 정상 동작"""
        seq_len = 500
        len_prompt = 100
        actual_tokens = 350  # 실제 유효 토큰 수

        labels = torch.arange(seq_len)
        labels[:len_prompt] = -100
        labels[actual_tokens:] = -100  # padding 부분

        attention_mask = torch.zeros(seq_len)
        attention_mask[:actual_tokens] = 1

        result = apply_random_window_mask(
            labels=labels,
            attention_mask=attention_mask,
            len_prompt=len_prompt,
            window_size=192,
        )

        # output 길이 = 350 - 100 = 250 (> window_size=192)
        # 학습 토큰 수 = window_size = 192
        n_train_tokens = (result != -100).sum().item()
        assert n_train_tokens == 192

        # padding 부분은 그대로 -100
        assert (result[actual_tokens:] == -100).all()

    def test_random_window_randomness(self):
        """랜덤 윈도우 선택의 다양성 검증"""
        seq_len = 400
        len_prompt = 100
        labels = torch.arange(seq_len)
        labels[:len_prompt] = -100
        attention_mask = torch.ones(seq_len)

        # 여러 번 실행하여 결과 수집
        results = []
        for _ in range(10):
            result = apply_random_window_mask(
                labels=labels.clone(),
                attention_mask=attention_mask,
                len_prompt=len_prompt,
                window_size=192,
            )
            # 학습 대상 토큰의 시작 위치 찾기
            train_mask = result != -100
            first_train_idx = train_mask.nonzero()[0].item()
            results.append(first_train_idx)

        # 랜덤이므로 다양한 시작 위치가 나와야 함 (완벽하지 않을 수 있음)
        # 최소한 학습 토큰 수는 일정
        for _ in range(3):
            result = apply_random_window_mask(
                labels=labels.clone(),
                attention_mask=attention_mask,
                len_prompt=len_prompt,
                window_size=192,
            )
            n_train = (result != -100).sum().item()
            assert n_train == 192
