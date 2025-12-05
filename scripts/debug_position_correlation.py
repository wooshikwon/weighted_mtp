#!/usr/bin/env python3
"""Position Correlation 디버깅 스크립트

λ=1.0 (MC)에서 타겟이 정말 동일한지,
그리고 학습 전/후 position correlation이 어떤지 확인.
"""

import torch
import sys
sys.path.insert(0, "src")

from weighted_mtp.utils.pairwise_utils import (
    compute_lambda_return,
    compute_position_correlation,
)


def test_lambda_return_with_mc():
    """λ=1.0일 때 타겟이 모든 토큰에서 동일한지 확인"""
    print("=" * 60)
    print("테스트 1: λ=1.0 (MC)에서 타겟 검증")
    print("=" * 60)

    batch_size = 2
    seq_len = 100

    # 가상 데이터
    values = torch.randn(batch_size, seq_len) * 0.1 + 0.5  # 초기 V ≈ 0.5
    rewards = torch.tensor([1.0, 0.0])  # 정답, 오답
    loss_mask = torch.zeros(batch_size, seq_len)
    loss_mask[:, 20:80] = 1  # output 영역: 20~79

    # λ=1.0, γ=1.0
    lambda_returns = compute_lambda_return(
        values, rewards, loss_mask,
        gamma=1.0, lam=1.0,
        output_end_mask=None
    )

    print("\n정답 샘플 (R=1.0):")
    pos_targets = lambda_returns[0, loss_mask[0].bool()]
    print(f"  타겟 범위: [{pos_targets.min():.4f}, {pos_targets.max():.4f}]")
    print(f"  타겟 평균: {pos_targets.mean():.4f}")
    print(f"  타겟 std:  {pos_targets.std():.6f}")
    print(f"  기대값:    1.0 (모든 토큰 동일)")

    print("\n오답 샘플 (R=0.0):")
    neg_targets = lambda_returns[1, loss_mask[1].bool()]
    print(f"  타겟 범위: [{neg_targets.min():.4f}, {neg_targets.max():.4f}]")
    print(f"  타겟 평균: {neg_targets.mean():.4f}")
    print(f"  타겟 std:  {neg_targets.std():.6f}")
    print(f"  기대값:    0.0 (모든 토큰 동일)")

    # 검증
    is_correct = pos_targets.std() < 0.001 and neg_targets.std() < 0.001
    print(f"\n✓ λ=1.0에서 타겟이 동일: {is_correct}")

    return is_correct


def test_position_correlation_with_uniform_values():
    """동일한 V 예측에서 position correlation이 0인지 확인"""
    print("\n" + "=" * 60)
    print("테스트 2: 동일한 V에서 position correlation 검증")
    print("=" * 60)

    batch_size = 4
    seq_len = 100

    # 모든 토큰에서 동일한 V
    values = torch.ones(batch_size, seq_len, 1) * 0.7
    mask = torch.zeros(batch_size, seq_len)
    mask[:, 20:80] = 1  # output 영역

    corr = compute_position_correlation(values, mask)
    print(f"\n동일한 V=0.7에서 position correlation: {corr:.6f}")
    print(f"기대값: ~0 (또는 NaN, 분산 0이므로)")

    # 약간의 랜덤 노이즈 추가
    values_noisy = values + torch.randn_like(values) * 0.01
    corr_noisy = compute_position_correlation(values_noisy, mask)
    print(f"\n약간의 노이즈 추가 후: {corr_noisy:.6f}")
    print(f"기대값: ~0 (랜덤 노이즈는 position과 무관)")


def test_position_correlation_with_position_dependent_values():
    """Position-dependent V에서 position correlation 확인"""
    print("\n" + "=" * 60)
    print("테스트 3: Position-dependent V에서 correlation 검증")
    print("=" * 60)

    batch_size = 4
    seq_len = 100

    mask = torch.zeros(batch_size, seq_len)
    mask[:, 20:80] = 1  # output 영역: 20~79

    # Case A: 뒤로 갈수록 V 증가 (기대: 양수 correlation)
    positions = torch.arange(seq_len).float()
    values_increasing = positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
    values_increasing = values_increasing / seq_len  # 0~1로 정규화

    corr_inc = compute_position_correlation(values_increasing, mask)
    print(f"\nV가 position에 따라 증가: correlation = {corr_inc:.4f}")
    print(f"기대값: ~1.0 (강한 양수)")

    # Case B: 뒤로 갈수록 V 감소 (기대: 음수 correlation)
    values_decreasing = 1.0 - values_increasing
    corr_dec = compute_position_correlation(values_decreasing, mask)
    print(f"\nV가 position에 따라 감소: correlation = {corr_dec:.4f}")
    print(f"기대값: ~-1.0 (강한 음수)")


def simulate_learning_dynamics():
    """MC 학습 시 position correlation 변화 시뮬레이션"""
    print("\n" + "=" * 60)
    print("테스트 4: MC 학습 dynamics 시뮬레이션")
    print("=" * 60)

    batch_size = 4
    seq_len = 100
    output_start, output_end = 20, 80

    mask = torch.zeros(batch_size, seq_len)
    mask[:, output_start:output_end] = 1

    # 초기 V: position-dependent (backbone bias 시뮬레이션)
    # 가정: backbone hidden state가 뒤로 갈수록 "낮은 값" 방향
    positions = torch.arange(seq_len).float()
    initial_bias = -0.002 * (positions - output_start)  # 뒤로 갈수록 감소

    V = 0.5 + initial_bias.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)

    print("\n초기 상태 (backbone bias 있음):")
    print(f"  V[20] = {V[0, 20, 0]:.4f}")
    print(f"  V[79] = {V[0, 79, 0]:.4f}")
    corr_init = compute_position_correlation(V, mask)
    print(f"  Position correlation: {corr_init:.4f}")

    # MC 학습 시뮬레이션 (타겟 = 1.0 for correct)
    # 단순화: V = V - lr * (V - target)
    target = 1.0
    lr = 0.1

    for step in range(10):
        V = V - lr * (V - target)

    print(f"\n학습 후 (10 steps, target=1.0):")
    print(f"  V[20] = {V[0, 20, 0]:.4f}")
    print(f"  V[79] = {V[0, 79, 0]:.4f}")
    corr_after = compute_position_correlation(V, mask)
    print(f"  Position correlation: {corr_after:.4f}")

    print(f"\n분석:")
    print(f"  초기 correlation: {corr_init:.4f}")
    print(f"  학습 후 correlation: {corr_after:.4f}")
    print(f"  변화: {corr_after - corr_init:.4f}")

    if abs(corr_after) < abs(corr_init):
        print("  → 학습이 position bias를 줄이고 있음 ✓")
    else:
        print("  → 학습이 position bias를 줄이지 못함 ✗")


def check_hidden_state_position_dependency():
    """실제 모델의 hidden state position 의존성 확인"""
    print("\n" + "=" * 60)
    print("테스트 5: 실제 모델 hidden state 분석 (옵션)")
    print("=" * 60)
    print("이 테스트는 실제 모델 로딩이 필요합니다.")
    print("필요시 별도 스크립트로 실행하세요.")


if __name__ == "__main__":
    print("Position Correlation 디버깅")
    print("=" * 60)

    test_lambda_return_with_mc()
    test_position_correlation_with_uniform_values()
    test_position_correlation_with_position_dependent_values()
    simulate_learning_dynamics()

    print("\n" + "=" * 60)
    print("결론")
    print("=" * 60)
    print("""
λ=1.0 (MC)에서 position correlation이 음수인 이유:

1. λ=1.0에서 타겟은 모든 토큰에서 동일 (R) ✓
2. 하지만 backbone hidden state가 position-dependent
3. Value head가 이 hidden state를 처리할 때:
   - 동일한 gradient signal이 주어져도
   - hidden state가 다르면 weight update 효과가 다름
4. 결과: 학습 후에도 V 예측이 position-dependent

해결책:
- 학습률 높이기 (backbone bias 극복)
- 더 많은 학습 step
- Value head 구조 변경 (position-invariant)
- 또는: position correlation은 무시하고 mean 수렴에 집중
""")
