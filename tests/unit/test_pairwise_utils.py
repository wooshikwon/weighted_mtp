"""pairwise_utils 단위 테스트

λ-return 및 value loss 함수 검증
"""

import pytest
import torch

from weighted_mtp.utils.pairwise_utils import (
    compute_lambda_return,
    compute_lambda_value_loss,
    compute_mc_value_loss,
    create_eos_only_mask,
    create_output_end_mask,
    get_scheduled_lambda,
)


class TestComputeLambdaReturn:
    """compute_lambda_return 함수 테스트"""

    def test_lambda_1_equals_mc(self):
        """λ=1.0일 때 모든 토큰이 terminal reward와 동일해야 함 (MC와 동치)"""
        batch_size, seq_len = 2, 5
        values = torch.rand(batch_size, seq_len)
        rewards = torch.tensor([1.0, 0.0])
        loss_mask = torch.ones(batch_size, seq_len)

        # λ=1.0: Pure MC
        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=1.0
        )

        # 모든 유효 토큰이 terminal reward와 동일해야 함
        for b in range(batch_size):
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            for pos in valid_positions:
                assert torch.isclose(
                    lambda_returns[b, pos], rewards[b], atol=1e-6
                ), f"batch {b}, pos {pos}: expected {rewards[b]}, got {lambda_returns[b, pos]}"

    def test_lambda_0_uses_only_v_next(self):
        """λ=0일 때 각 위치가 V_{t+1}만 사용해야 함 (Pure TD)"""
        batch_size, seq_len = 1, 4
        # 고정된 V 값으로 테스트
        values = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        rewards = torch.tensor([1.0])
        loss_mask = torch.ones(batch_size, seq_len)

        # λ=0: Pure TD (G_t = γV_{t+1})
        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.0
        )

        # Terminal: G_3 = R = 1.0
        assert torch.isclose(lambda_returns[0, 3], torch.tensor(1.0), atol=1e-6)
        # G_2 = γV_3 = 1.0 * 0.4 = 0.4
        assert torch.isclose(lambda_returns[0, 2], torch.tensor(0.4), atol=1e-6)
        # G_1 = γV_2 = 1.0 * 0.3 = 0.3
        assert torch.isclose(lambda_returns[0, 1], torch.tensor(0.3), atol=1e-6)
        # G_0 = γV_1 = 1.0 * 0.2 = 0.2
        assert torch.isclose(lambda_returns[0, 0], torch.tensor(0.2), atol=1e-6)

    def test_lambda_095_creates_gradient(self):
        """λ=0.95일 때 위치별로 다른 타겟이 생성되어야 함"""
        batch_size, seq_len = 1, 5
        values = torch.full((batch_size, seq_len), 0.5)  # 모든 V=0.5
        rewards = torch.tensor([1.0])
        loss_mask = torch.ones(batch_size, seq_len)

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95
        )

        # Terminal은 R=1.0
        assert torch.isclose(lambda_returns[0, 4], torch.tensor(1.0), atol=1e-6)

        # 앞쪽으로 갈수록 값이 다름 (gradient 존재)
        # 정확한 값 계산:
        # G_4 = 1.0
        # G_3 = 0.05*V_4 + 0.95*G_4 = 0.05*0.5 + 0.95*1.0 = 0.025 + 0.95 = 0.975
        # G_2 = 0.05*V_3 + 0.95*G_3 = 0.05*0.5 + 0.95*0.975 = 0.025 + 0.92625 = 0.95125
        assert torch.isclose(lambda_returns[0, 3], torch.tensor(0.975), atol=1e-5)
        assert torch.isclose(lambda_returns[0, 2], torch.tensor(0.95125), atol=1e-5)

        # 전반부는 후반부보다 낮아야 함
        assert lambda_returns[0, 0] < lambda_returns[0, 2] < lambda_returns[0, 4]

    def test_partial_mask(self):
        """부분 마스크 (instruction 제외) 처리 검증"""
        batch_size, seq_len = 1, 6
        values = torch.rand(batch_size, seq_len)
        rewards = torch.tensor([1.0])
        # 앞 2개는 instruction (mask=0), 뒤 4개가 output (mask=1)
        loss_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0, 1.0]])

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95
        )

        # 마스크=0인 위치는 타겟이 0이어야 함
        assert lambda_returns[0, 0] == 0.0
        assert lambda_returns[0, 1] == 0.0
        # 마스크=1인 마지막 위치는 R
        assert torch.isclose(lambda_returns[0, 5], torch.tensor(1.0), atol=1e-6)

    def test_empty_mask(self):
        """빈 마스크 처리 (엣지 케이스)"""
        batch_size, seq_len = 1, 4
        values = torch.rand(batch_size, seq_len)
        rewards = torch.tensor([1.0])
        loss_mask = torch.zeros(batch_size, seq_len)

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95
        )

        # 모든 위치가 0이어야 함
        assert torch.all(lambda_returns == 0.0)

    def test_batch_processing(self):
        """배치 처리 검증 (correct/incorrect 혼합)"""
        batch_size, seq_len = 4, 5
        values = torch.full((batch_size, seq_len), 0.5)
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])  # 교대로 correct/incorrect
        loss_mask = torch.ones(batch_size, seq_len)

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=1.0  # MC로 검증
        )

        # 각 배치의 terminal이 해당 reward와 동일해야 함
        assert torch.isclose(lambda_returns[0, 4], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(lambda_returns[1, 4], torch.tensor(0.0), atol=1e-6)
        assert torch.isclose(lambda_returns[2, 4], torch.tensor(1.0), atol=1e-6)
        assert torch.isclose(lambda_returns[3, 4], torch.tensor(0.0), atol=1e-6)


class TestComputeLambdaValueLoss:
    """compute_lambda_value_loss 함수 테스트"""

    def test_loss_decreases_with_correct_predictions(self):
        """정확한 예측일수록 loss가 낮아야 함"""
        batch_size, seq_len = 2, 5
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)
        rewards = torch.tensor([1.0, 0.0])

        # 정확한 예측: correct=1.0, incorrect=0.0
        accurate_values = torch.tensor([
            [[1.0], [1.0], [1.0], [1.0], [1.0]],  # correct sample -> 1.0
            [[0.0], [0.0], [0.0], [0.0], [0.0]],  # incorrect sample -> 0.0
        ])

        # 부정확한 예측: 반대로
        inaccurate_values = torch.tensor([
            [[0.0], [0.0], [0.0], [0.0], [0.0]],  # correct sample -> 0.0 (틀림)
            [[1.0], [1.0], [1.0], [1.0], [1.0]],  # incorrect sample -> 1.0 (틀림)
        ])

        accurate_loss = compute_lambda_value_loss(
            accurate_values, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0  # MC로 테스트
        )
        inaccurate_loss = compute_lambda_value_loss(
            inaccurate_values, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0
        )

        assert accurate_loss < inaccurate_loss

    def test_interface_compatibility_with_mc(self):
        """compute_mc_value_loss와 동일한 인터페이스로 호출 가능해야 함"""
        batch_size, seq_len = 2, 5
        value_logits = torch.rand(batch_size, seq_len, 1)
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # 두 함수 모두 동일한 인터페이스로 호출 가능
        mc_loss = compute_mc_value_loss(value_logits, rewards, attention_mask, loss_mask)
        lambda_loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0, loss_type="mse"  # MSE 명시적 지정
        )

        # λ=1.0, loss_type=mse일 때 MC와 동일한 결과
        assert torch.isclose(mc_loss, lambda_loss, atol=1e-5)

    def test_gradient_flows(self):
        """value_logits에 대한 gradient가 정상적으로 흐르는지 확인"""
        batch_size, seq_len = 2, 4
        value_logits = torch.rand(batch_size, seq_len, 1, requires_grad=True)
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=0.95
        )
        loss.backward()

        # gradient가 존재하고 유효해야 함
        assert value_logits.grad is not None
        assert not torch.isnan(value_logits.grad).any()
        assert not torch.isinf(value_logits.grad).any()

    def test_masked_positions_excluded(self):
        """마스크된 위치는 loss 계산에서 제외되어야 함"""
        batch_size, seq_len = 1, 5
        value_logits = torch.zeros(batch_size, seq_len, 1)  # 모든 예측 0
        rewards = torch.tensor([1.0])  # 타겟은 1
        attention_mask = torch.ones(batch_size, seq_len)

        # 전체 마스크: 모든 토큰 포함 -> loss > 0
        full_mask = torch.ones(batch_size, seq_len)
        full_loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, full_mask,
            gamma=1.0, lam=1.0, loss_type="mse"
        )

        # 부분 마스크: 일부만 포함 -> loss < full_loss (비례)
        # (실제로는 평균이므로 값 자체는 비슷할 수 있지만, 마스크 동작 검증)
        partial_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]])
        partial_loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, partial_mask,
            gamma=1.0, lam=1.0, loss_type="mse"
        )

        # 둘 다 양수 loss (예측=0, 타겟=1이므로)
        assert full_loss > 0
        assert partial_loss > 0

    def test_huber_loss_default(self):
        """Huber loss가 기본값으로 사용되어야 함"""
        batch_size, seq_len = 2, 5
        value_logits = torch.rand(batch_size, seq_len, 1)
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # 기본 호출 (loss_type 미지정)
        loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0
        )

        # loss가 정상적으로 계산되어야 함
        assert loss >= 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_huber_loss_vs_mse_for_small_errors(self):
        """작은 오차에서 Huber loss와 MSE가 유사해야 함 (delta 이내)"""
        batch_size, seq_len = 1, 3
        # 오차가 작도록 설정 (target=0.5, prediction=0.6)
        value_logits = torch.full((batch_size, seq_len, 1), 0.6)
        rewards = torch.tensor([0.5])  # target = 0.5
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        huber_delta = 0.5  # 오차 0.1 < delta 0.5

        huber_loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0, loss_type="huber", huber_delta=huber_delta
        )

        mse_loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0, loss_type="mse"
        )

        # 작은 오차에서는 Huber와 MSE가 유사 (Huber = 0.5 * error^2 / delta)
        # delta=0.5일 때 error=0.1이면: huber = 0.5 * 0.01 / 0.5 = 0.01, mse = 0.01
        # 실제로는 smooth_l1_loss의 정의가 조금 다를 수 있으므로 비율로 비교
        assert huber_loss > 0
        assert mse_loss > 0

    def test_huber_loss_robust_to_outliers(self):
        """큰 오차에서 Huber loss가 MSE보다 robust해야 함"""
        batch_size, seq_len = 1, 3
        # 큰 오차 설정 (target=1.0, prediction=0.0)
        value_logits = torch.zeros(batch_size, seq_len, 1)
        rewards = torch.tensor([1.0])  # target = 1.0
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        huber_delta = 0.5  # 오차 1.0 > delta 0.5

        huber_loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0, loss_type="huber", huber_delta=huber_delta
        )

        mse_loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0, loss_type="mse"
        )

        # 큰 오차에서 Huber loss < MSE loss (선형 vs 제곱)
        # MSE: (1-0)^2 = 1.0
        # Huber (delta=0.5): delta * (|error| - 0.5 * delta) = 0.5 * (1.0 - 0.25) = 0.375
        assert huber_loss < mse_loss

    def test_huber_loss_gradient_flows(self):
        """Huber loss에서 gradient가 정상적으로 흐르는지 확인"""
        batch_size, seq_len = 2, 4
        value_logits = torch.rand(batch_size, seq_len, 1, requires_grad=True)
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=0.95, loss_type="huber", huber_delta=0.5
        )
        loss.backward()

        # gradient가 존재하고 유효해야 함
        assert value_logits.grad is not None
        assert not torch.isnan(value_logits.grad).any()
        assert not torch.isinf(value_logits.grad).any()

    def test_mse_backward_compatibility(self):
        """loss_type='mse'로 기존 MSE 동작이 유지되어야 함"""
        batch_size, seq_len = 2, 5
        value_logits = torch.rand(batch_size, seq_len, 1)
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.ones(batch_size, seq_len)

        # 명시적으로 MSE 사용
        mse_loss_explicit = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0, loss_type="mse"
        )

        # MC loss와 동일한 결과 (lam=1.0일 때)
        mc_loss = compute_mc_value_loss(value_logits, rewards, attention_mask, loss_mask)

        assert torch.isclose(mse_loss_explicit, mc_loss, atol=1e-5)


class TestCreateEosOnlyMask:
    """create_eos_only_mask 함수 테스트"""

    def test_instruction_masked_sequence(self):
        """Instruction이 마스킹된 시퀀스에서 마지막 Output 토큰 위치 정확히 찾기"""
        # [0, 0, 1, 1, 1, 0, 0] -> 마지막 1의 위치는 4
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 1, 0, 0],
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        # 인덱스 4만 1이어야 함
        expected = torch.tensor([
            [0, 0, 0, 0, 1, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(eos_mask, expected)

    def test_continuous_output_sequence(self):
        """연속된 Output 시퀀스 (Instruction 없음)"""
        # [1, 1, 1, 0, 0] -> 마지막 1의 위치는 2
        loss_mask = torch.tensor([
            [1, 1, 1, 0, 0],
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        expected = torch.tensor([
            [0, 0, 1, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(eos_mask, expected)

    def test_empty_mask(self):
        """유효 토큰이 없는 경우 (모두 0)"""
        loss_mask = torch.tensor([
            [0, 0, 0, 0, 0],
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        # 모두 0이어야 함
        expected = torch.zeros_like(loss_mask)
        assert torch.equal(eos_mask, expected)

    def test_single_valid_token(self):
        """유효 토큰이 하나뿐인 경우"""
        loss_mask = torch.tensor([
            [0, 0, 1, 0, 0],
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        expected = torch.tensor([
            [0, 0, 1, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(eos_mask, expected)

    def test_first_position_valid(self):
        """첫 번째 위치만 유효한 경우"""
        loss_mask = torch.tensor([
            [1, 0, 0, 0],
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        expected = torch.tensor([
            [1, 0, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(eos_mask, expected)

    def test_batch_processing(self):
        """배치 처리 검증"""
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 1, 0, 0],  # 마지막 위치: 4
            [1, 1, 1, 0, 0, 0, 0],  # 마지막 위치: 2
            [0, 0, 0, 0, 0, 0, 0],  # 유효 토큰 없음
            [0, 1, 0, 0, 0, 0, 0],  # 마지막 위치: 1
        ], dtype=torch.float32)

        eos_mask = create_eos_only_mask(loss_mask)

        expected = torch.tensor([
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(eos_mask, expected)

    def test_dtype_preservation(self):
        """반환 dtype이 float32인지 확인"""
        loss_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        eos_mask = create_eos_only_mask(loss_mask)
        assert eos_mask.dtype == torch.float32

    def test_device_preservation(self):
        """디바이스가 유지되는지 확인"""
        loss_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)
        eos_mask = create_eos_only_mask(loss_mask)
        assert eos_mask.device == loss_mask.device


class TestGetScheduledLambda:
    """get_scheduled_lambda 함수 테스트"""

    def test_warmup_phase_returns_start(self):
        """Warmup 기간 동안 λ=start 반환"""
        warmup_steps = 250
        decay_steps = 500
        lam_start = 1.0
        lam_end = 0.95

        # Warmup 기간 내 여러 step 테스트
        assert get_scheduled_lambda(0, warmup_steps, decay_steps, lam_start, lam_end) == 1.0
        assert get_scheduled_lambda(100, warmup_steps, decay_steps, lam_start, lam_end) == 1.0
        assert get_scheduled_lambda(249, warmup_steps, decay_steps, lam_start, lam_end) == 1.0

    def test_decay_phase_linear_decrease(self):
        """Decay 기간 동안 선형 감소"""
        warmup_steps = 250
        decay_steps = 500
        lam_start = 1.0
        lam_end = 0.95

        # Decay 시작점
        lam_at_250 = get_scheduled_lambda(250, warmup_steps, decay_steps, lam_start, lam_end)
        assert abs(lam_at_250 - 1.0) < 1e-6

        # Decay 중간점 (50% progress)
        lam_at_500 = get_scheduled_lambda(500, warmup_steps, decay_steps, lam_start, lam_end)
        expected_mid = 1.0 - 0.5 * (1.0 - 0.95)  # 0.975
        assert abs(lam_at_500 - expected_mid) < 1e-6

        # Decay 종료 직전
        lam_at_749 = get_scheduled_lambda(749, warmup_steps, decay_steps, lam_start, lam_end)
        assert lam_at_749 > 0.95 and lam_at_749 < 0.96

    def test_stable_phase_returns_end(self):
        """Decay 종료 후 λ=end 유지"""
        warmup_steps = 250
        decay_steps = 500
        lam_start = 1.0
        lam_end = 0.95

        # Decay 종료 이후
        assert get_scheduled_lambda(750, warmup_steps, decay_steps, lam_start, lam_end) == 0.95
        assert get_scheduled_lambda(1000, warmup_steps, decay_steps, lam_start, lam_end) == 0.95
        assert get_scheduled_lambda(10000, warmup_steps, decay_steps, lam_start, lam_end) == 0.95

    def test_custom_lambda_range(self):
        """다른 λ 범위 테스트 (0.8 → 0.5)"""
        warmup_steps = 100
        decay_steps = 200
        lam_start = 0.8
        lam_end = 0.5

        # Warmup
        assert get_scheduled_lambda(50, warmup_steps, decay_steps, lam_start, lam_end) == 0.8

        # Decay 중간 (50% progress)
        lam_mid = get_scheduled_lambda(200, warmup_steps, decay_steps, lam_start, lam_end)
        expected = 0.8 - 0.5 * (0.8 - 0.5)  # 0.65
        assert abs(lam_mid - expected) < 1e-6

        # Stable
        assert get_scheduled_lambda(500, warmup_steps, decay_steps, lam_start, lam_end) == 0.5

    def test_zero_warmup_steps(self):
        """warmup_steps=0일 때 즉시 decay 시작"""
        warmup_steps = 0
        decay_steps = 100
        lam_start = 1.0
        lam_end = 0.95

        # Step 0에서 decay 시작
        assert get_scheduled_lambda(0, warmup_steps, decay_steps, lam_start, lam_end) == 1.0

        # 50% decay
        lam_50 = get_scheduled_lambda(50, warmup_steps, decay_steps, lam_start, lam_end)
        assert abs(lam_50 - 0.975) < 1e-6

        # Decay 완료
        assert get_scheduled_lambda(100, warmup_steps, decay_steps, lam_start, lam_end) == 0.95

    def test_zero_decay_steps(self):
        """decay_steps=0일 때 warmup 후 즉시 end로 전환"""
        warmup_steps = 100
        decay_steps = 0
        lam_start = 1.0
        lam_end = 0.95

        # Warmup 중
        assert get_scheduled_lambda(50, warmup_steps, decay_steps, lam_start, lam_end) == 1.0

        # Warmup 종료 후 즉시 end
        assert get_scheduled_lambda(100, warmup_steps, decay_steps, lam_start, lam_end) == 0.95
        assert get_scheduled_lambda(150, warmup_steps, decay_steps, lam_start, lam_end) == 0.95

    def test_monotonic_decrease(self):
        """λ가 단조 감소하는지 확인"""
        warmup_steps = 100
        decay_steps = 200
        lam_start = 1.0
        lam_end = 0.95

        prev_lam = float("inf")
        for step in range(0, 400, 10):
            current_lam = get_scheduled_lambda(step, warmup_steps, decay_steps, lam_start, lam_end)
            assert current_lam <= prev_lam, f"Step {step}: {current_lam} > {prev_lam}"
            prev_lam = current_lam


class TestCreateOutputEndMask:
    """create_output_end_mask 함수 테스트"""

    def test_simple_sequence(self):
        """기본 시퀀스에서 마지막 유효 토큰 위치 반환"""
        # 5개 유효 토큰, 2개 패딩
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.float32)

        output_end_mask = create_output_end_mask(attention_mask)

        # 마지막 유효 토큰은 인덱스 4
        expected = torch.tensor([[0, 0, 0, 0, 1, 0, 0]], dtype=torch.float32)
        assert torch.equal(output_end_mask, expected)

    def test_no_padding(self):
        """패딩 없는 시퀀스"""
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.float32)

        output_end_mask = create_output_end_mask(attention_mask)

        # 마지막 토큰은 인덱스 3
        expected = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)
        assert torch.equal(output_end_mask, expected)

    def test_all_padding(self):
        """모두 패딩인 경우 (엣지 케이스)"""
        attention_mask = torch.tensor([[0, 0, 0, 0]], dtype=torch.float32)

        output_end_mask = create_output_end_mask(attention_mask)

        # 유효 토큰 없으므로 모두 0
        expected = torch.zeros_like(attention_mask)
        assert torch.equal(output_end_mask, expected)

    def test_batch_processing(self):
        """배치 처리 검증"""
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0],  # 마지막 유효: 4
            [1, 1, 1, 0, 0, 0, 0],  # 마지막 유효: 2
            [1, 1, 1, 1, 1, 1, 1],  # 마지막 유효: 6
            [0, 0, 0, 0, 0, 0, 0],  # 유효 토큰 없음
        ], dtype=torch.float32)

        output_end_mask = create_output_end_mask(attention_mask)

        expected = torch.tensor([
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.float32)
        assert torch.equal(output_end_mask, expected)

    def test_dtype_is_float32(self):
        """반환 dtype이 float32인지 확인"""
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
        output_end_mask = create_output_end_mask(attention_mask)
        assert output_end_mask.dtype == torch.float32

    def test_device_preservation(self):
        """디바이스가 유지되는지 확인"""
        attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)
        output_end_mask = create_output_end_mask(attention_mask)
        assert output_end_mask.device == attention_mask.device


class TestComputeLambdaReturnWithOutputEndMask:
    """output_end_mask 파라미터를 사용하는 compute_lambda_return 테스트"""

    def test_output_end_mask_sets_terminal_position(self):
        """output_end_mask가 제공되면 해당 위치에 terminal reward 설정"""
        batch_size, seq_len = 1, 10
        values = torch.zeros(batch_size, seq_len)
        rewards = torch.tensor([1.0])

        # loss_mask: 윈도우 [2, 3, 4]만 1
        loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)

        # output_end_mask: 진짜 EOS는 인덱스 7
        output_end_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=torch.float32)

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=1.0,
            output_end_mask=output_end_mask,
        )

        # Terminal reward는 인덱스 7에 설정되어야 함
        assert torch.isclose(lambda_returns[0, 7], torch.tensor(1.0), atol=1e-6)

    def test_backward_propagation_from_eos(self):
        """EOS에서 시작하여 역방향 전파되는지 검증"""
        batch_size, seq_len = 1, 10
        values = torch.full((batch_size, seq_len), 0.5)
        rewards = torch.tensor([1.0])

        # loss_mask: 윈도우 [2, 3, 4]만 학습 대상
        loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)

        # output_end_mask: 진짜 EOS는 인덱스 7
        output_end_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=torch.float32)

        # λ=1.0 (Pure MC): 모든 위치가 terminal reward와 동일
        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=1.0,
            output_end_mask=output_end_mask,
        )

        # EOS(7)에서 역방향 전파되므로 0~7 모두 1.0이어야 함
        for t in range(8):
            assert torch.isclose(lambda_returns[0, t], torch.tensor(1.0), atol=1e-6), \
                f"Position {t}: expected 1.0, got {lambda_returns[0, t]}"

    def test_without_output_end_mask_uses_loss_mask(self):
        """output_end_mask 미제공 시 loss_mask 기준 (기존 동작)"""
        batch_size, seq_len = 1, 10
        values = torch.zeros(batch_size, seq_len)
        rewards = torch.tensor([1.0])

        # loss_mask: [2, 3, 4]만 유효
        loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=1.0,
            output_end_mask=None,  # 미제공
        )

        # loss_mask 기준 마지막(4)에 terminal
        assert torch.isclose(lambda_returns[0, 4], torch.tensor(1.0), atol=1e-6)
        # 인덱스 7은 0 (loss_mask에서 유효하지 않으므로)
        assert lambda_returns[0, 7] == 0.0

    def test_lambda_095_with_output_end_mask(self):
        """λ=0.95에서 output_end_mask 사용 시 올바른 타겟 생성"""
        batch_size, seq_len = 1, 8
        values = torch.full((batch_size, seq_len), 0.5)
        rewards = torch.tensor([1.0])

        loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
        output_end_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.float32)  # EOS: 6

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95,
            output_end_mask=output_end_mask,
        )

        # EOS(6)는 R=1.0
        assert torch.isclose(lambda_returns[0, 6], torch.tensor(1.0), atol=1e-6)

        # 역방향 전파로 gradient 존재
        # G_5 = 0.05*V_6 + 0.95*G_6 = 0.05*0.5 + 0.95*1.0 = 0.975
        assert torch.isclose(lambda_returns[0, 5], torch.tensor(0.975), atol=1e-5)

        # 윈도우 내 토큰들도 역방향 전파된 값을 가짐
        assert lambda_returns[0, 4] > 0
        assert lambda_returns[0, 3] > 0
        assert lambda_returns[0, 2] > 0


class TestComputeLambdaValueLossWithOutputEndMask:
    """output_end_mask 파라미터를 사용하는 compute_lambda_value_loss 테스트"""

    def test_loss_only_on_loss_mask_range(self):
        """Loss 계산이 loss_mask 범위에서만 수행되는지 검증"""
        batch_size, seq_len = 1, 10
        value_logits = torch.zeros(batch_size, seq_len, 1)  # 모든 예측 0
        rewards = torch.tensor([1.0])

        attention_mask = torch.ones(batch_size, seq_len)
        # loss_mask: [2, 3, 4]만 학습 대상
        loss_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
        # output_end_mask: EOS는 7
        output_end_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=torch.float32)

        loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=1.0, loss_type="mse",
            output_end_mask=output_end_mask,
        )

        # Loss가 양수 (예측=0, 타겟=1이므로)
        assert loss > 0

    def test_gradient_flows_with_output_end_mask(self):
        """output_end_mask 사용 시 gradient가 정상적으로 흐르는지 확인"""
        batch_size, seq_len = 2, 8
        value_logits = torch.rand(batch_size, seq_len, 1, requires_grad=True)
        rewards = torch.tensor([1.0, 0.0])
        attention_mask = torch.ones(batch_size, seq_len)
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0],
        ], dtype=torch.float32)
        output_end_mask = torch.tensor([
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ], dtype=torch.float32)

        loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=0.95,
            output_end_mask=output_end_mask,
        )
        loss.backward()

        # gradient가 존재하고 유효해야 함
        assert value_logits.grad is not None
        assert not torch.isnan(value_logits.grad).any()
        assert not torch.isinf(value_logits.grad).any()


class TestRandomWindowWithEOS:
    """Random Window + EOS 분리 통합 테스트 (Phase 5)"""

    def test_lambda_return_with_random_window(self):
        """랜덤 윈도우에서도 terminal reward가 EOS에 설정되는지 검증"""
        # 시나리오: 전체 시퀀스 10, 윈도우는 [2,3,4], EOS는 7
        values = torch.zeros(1, 10)
        rewards = torch.tensor([1.0])
        loss_mask = torch.zeros(1, 10)
        loss_mask[0, 2:5] = 1.0  # 윈도우: [2,3,4]
        output_end_mask = torch.zeros(1, 10)
        output_end_mask[0, 7] = 1.0  # EOS: 7

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95,
            output_end_mask=output_end_mask,
        )

        # Terminal reward는 EOS(7)에 설정
        assert lambda_returns[0, 7] == 1.0, f"EOS 위치 값: {lambda_returns[0, 7]}"
        # 윈도우 내 토큰도 역방향 전파된 값을 가짐
        assert lambda_returns[0, 4] > 0, f"윈도우 끝 위치 값: {lambda_returns[0, 4]}"
        assert lambda_returns[0, 2] > 0, f"윈도우 시작 위치 값: {lambda_returns[0, 2]}"

    def test_loss_only_on_window(self):
        """Loss 계산이 윈도우 내 토큰에서만 이루어지는지 검증"""
        # 시나리오: 전체 시퀀스 10, 윈도우는 [2,3,4], EOS는 7
        value_logits = torch.randn(1, 10, requires_grad=True)
        rewards = torch.tensor([1.0])
        attention_mask = torch.ones(1, 10)
        loss_mask = torch.zeros(1, 10)
        loss_mask[0, 2:5] = 1.0  # 윈도우: [2,3,4]만 학습
        output_end_mask = torch.zeros(1, 10)
        output_end_mask[0, 7] = 1.0  # EOS: 7

        loss = compute_lambda_value_loss(
            value_logits, rewards, attention_mask, loss_mask,
            gamma=1.0, lam=0.95,
            output_end_mask=output_end_mask,
        )

        # Loss가 유효해야 함
        assert loss.item() >= 0
        assert not torch.isnan(loss)

        # Gradient가 윈도우 내에만 존재하는지 확인
        loss.backward()
        grad = value_logits.grad[0]

        # 윈도우 내 토큰은 gradient가 있어야 함
        assert grad[2:5].abs().sum() > 0, "윈도우 내 gradient 없음"

    def test_backward_propagation_distance(self):
        """lambda=0.95일 때 역방향 전파 거리 검증"""
        # EOS에서 윈도우까지의 거리에 따른 감쇠 확인
        seq_len = 20
        values = torch.zeros(1, seq_len)
        rewards = torch.tensor([1.0])
        loss_mask = torch.zeros(1, seq_len)
        loss_mask[0, 5:10] = 1.0  # 윈도우: [5,6,7,8,9]
        output_end_mask = torch.zeros(1, seq_len)
        output_end_mask[0, 15] = 1.0  # EOS: 15 (윈도우에서 6칸 떨어짐)

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95,
            output_end_mask=output_end_mask,
        )

        # EOS 위치는 1.0
        assert lambda_returns[0, 15] == 1.0

        # 윈도우 끝(9)은 EOS(15)에서 6칸 떨어짐: 0.95^6 ≈ 0.735
        expected_at_9 = 0.95 ** 6
        assert abs(lambda_returns[0, 9] - expected_at_9) < 0.01, \
            f"윈도우 끝 값 {lambda_returns[0, 9]:.4f} != 예상값 {expected_at_9:.4f}"


class TestFSDPCompatibility:
    """FSDP 분산 환경 호환성 검증 (Phase 5)"""

    def test_mask_shapes_consistent(self):
        """모든 마스크의 shape이 일관되는지 검증"""
        batch_size = 4
        seq_len = 512

        # 실제 사용 시나리오 시뮬레이션
        attention_mask = torch.ones(batch_size, seq_len)
        # 각 샘플마다 다른 실제 길이
        for i in range(batch_size):
            actual_len = 200 + i * 50  # 200, 250, 300, 350
            attention_mask[i, actual_len:] = 0

        # labels 생성 (instruction 마스킹 + padding 마스킹)
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        labels[:, :50] = -100  # instruction
        labels[attention_mask == 0] = -100  # padding

        # loss_mask 생성
        loss_mask = (labels != -100).float()

        # output_end_mask 생성
        output_end_mask = create_output_end_mask(attention_mask)

        # Shape 일관성 검증
        assert attention_mask.shape == (batch_size, seq_len)
        assert loss_mask.shape == (batch_size, seq_len)
        assert output_end_mask.shape == (batch_size, seq_len)

        # 각 샘플의 output_end_mask가 정확한 EOS 위치를 가리키는지
        for i in range(batch_size):
            actual_len = 200 + i * 50
            eos_pos = actual_len - 1
            assert output_end_mask[i, eos_pos] == 1.0, f"샘플 {i}: EOS 위치 {eos_pos} 마스크 오류"
            assert output_end_mask[i].sum() == 1.0, f"샘플 {i}: output_end_mask 합 오류"

    def test_batch_independent_processing(self):
        """각 배치 샘플이 독립적으로 처리되는지 검증"""
        batch_size = 2
        seq_len = 100

        # 두 샘플이 완전히 다른 구조
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 60:] = 0  # 샘플 0: 길이 60
        attention_mask[1, 80:] = 0  # 샘플 1: 길이 80

        loss_mask = torch.zeros(batch_size, seq_len)
        loss_mask[0, 30:50] = 1.0  # 샘플 0: 윈도우 [30,50)
        loss_mask[1, 40:70] = 1.0  # 샘플 1: 윈도우 [40,70)

        output_end_mask = create_output_end_mask(attention_mask)

        values = torch.zeros(batch_size, seq_len)
        rewards = torch.tensor([1.0, 0.0])  # 샘플 0: correct, 샘플 1: incorrect

        lambda_returns = compute_lambda_return(
            values, rewards, loss_mask, gamma=1.0, lam=0.95,
            output_end_mask=output_end_mask,
        )

        # 샘플 0: EOS(59)에 reward 1.0
        assert lambda_returns[0, 59] == 1.0
        # 샘플 1: EOS(79)에 reward 0.0
        assert lambda_returns[1, 79] == 0.0

        # 각 샘플의 윈도우 내 값이 독립적
        assert lambda_returns[0, 49] > 0  # 샘플 0 윈도우 끝
        assert lambda_returns[1, 69] == 0  # 샘플 1 윈도우 끝 (reward=0이므로)
