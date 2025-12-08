"""Value Loss 유틸리티

Pairwise Ranking Loss 및 MC MSE Loss 계산
run_critic, run_verifiable에서 공통 사용
"""

import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    loss_mask_pos: torch.Tensor,
    loss_mask_neg: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry Pairwise Ranking Loss

    P(pos > neg) = sigmoid(V_pos - V_neg)
    Loss = -log(sigmoid(V_pos - V_neg))

    Output 토큰만 사용하여 시퀀스 평균 비교 (Instruction 제외)

    Args:
        v_pos: [batch, seq, 1] positive sample values
        v_neg: [batch, seq, 1] negative sample values
        loss_mask_pos: [batch, seq] valid token mask for positive (labels != -100)
        loss_mask_neg: [batch, seq] valid token mask for negative (labels != -100)

    Returns:
        Scalar loss
    """
    # 시퀀스 평균 value 계산 (Output 토큰만)
    v_pos_mean = (v_pos.squeeze(-1) * loss_mask_pos).sum(dim=1) / (loss_mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg.squeeze(-1) * loss_mask_neg).sum(dim=1) / (loss_mask_neg.sum(dim=1) + 1e-8)

    # Pairwise ranking loss: -log(sigmoid(v_pos - v_neg))
    return -F.logsigmoid(v_pos_mean - v_neg_mean).mean()


def compute_pairwise_accuracy(
    v_pos: torch.Tensor,
    v_neg: torch.Tensor,
    loss_mask_pos: torch.Tensor,
    loss_mask_neg: torch.Tensor,
) -> dict[str, float]:
    """Pairwise Accuracy 및 관련 메트릭 계산

    Args:
        v_pos: [batch, seq, 1] positive sample values
        v_neg: [batch, seq, 1] negative sample values
        loss_mask_pos: [batch, seq] valid token mask for positive
        loss_mask_neg: [batch, seq] valid token mask for negative

    Returns:
        {pairwise_accuracy, mean_pos, mean_neg, margin, correct_pairs, total_pairs}
    """
    # 시퀀스 평균 value 계산
    v_pos_mean = (v_pos.squeeze(-1) * loss_mask_pos).sum(dim=1) / (loss_mask_pos.sum(dim=1) + 1e-8)
    v_neg_mean = (v_neg.squeeze(-1) * loss_mask_neg).sum(dim=1) / (loss_mask_neg.sum(dim=1) + 1e-8)

    # V(correct) > V(incorrect)인 쌍의 비율
    correct_pairs = (v_pos_mean > v_neg_mean).float().sum()
    total_pairs = v_pos_mean.size(0)
    pairwise_accuracy = (correct_pairs / total_pairs).item()

    # 평균 값
    mean_pos = v_pos_mean.mean().item()
    mean_neg = v_neg_mean.mean().item()
    margin = mean_pos - mean_neg

    return {
        "pairwise_accuracy": pairwise_accuracy,
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "margin": margin,
        "correct_pairs": correct_pairs.item(),
        "total_pairs": total_pairs,
    }


def compute_token_variance(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """시퀀스 내 토큰별 Value Variance 계산

    모델이 토큰별 차별화를 학습하는지 진단하는 메트릭.
    Variance ≈ 0: 모든 토큰에 동일한 값 예측 (problem-level 학습)
    Variance > 0: 토큰별 차별화 (token-level 학습)

    Args:
        values: [batch, seq, 1] Value head 출력
        mask: [batch, seq] 유효 토큰 마스크 (labels != -100)

    Returns:
        배치 평균 variance (scalar)
    """
    values = values.squeeze(-1)  # [batch, seq]

    # 시퀀스별 평균 (masked)
    seq_mean = (values * mask).sum(dim=1, keepdim=True) / (mask.sum(dim=1, keepdim=True) + 1e-8)

    # 시퀀스별 variance (masked)
    sq_diff = ((values - seq_mean) ** 2) * mask
    seq_variance = sq_diff.sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    return seq_variance.mean().item()


def compute_td_error_stats(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """TD Error 스파이크 분석 (좋은 Variance vs 나쁜 Variance 구분)

    좋은 학습: 특정 토큰에서 TD error 스파이크 발생 (spikiness 높음)
    나쁜 학습: 모든 토큰에서 밋밋한 TD error (spikiness 낮음, 길이 편향)

    Args:
        values: [batch, seq, 1] Value head 출력
        mask: [batch, seq] 유효 토큰 마스크 (labels != -100)

    Returns:
        dict with:
            mean_abs_td: 평균 |δ_t| (TD 크기)
            max_td: 최대 |δ_t| (스파이크 크기)
            spikiness: max_td / mean_abs_td (클수록 좋음, >3이면 건강)
    """
    values = values.squeeze(-1)  # [batch, seq]

    # TD error: δ_t = V_{t+1} - V_t
    td_errors = values[:, 1:] - values[:, :-1]  # [batch, seq-1]
    td_mask = mask[:, 1:] * mask[:, :-1]  # 둘 다 valid해야 함

    abs_td = torch.abs(td_errors) * td_mask

    # 시퀀스별 통계
    seq_lengths = td_mask.sum(dim=1).clamp(min=1)
    mean_abs_td = abs_td.sum(dim=1) / seq_lengths

    # Masked max: invalid 위치는 -inf로 처리
    masked_abs_td = abs_td.clone()
    masked_abs_td[td_mask == 0] = -float('inf')
    max_td = masked_abs_td.max(dim=1).values
    max_td = torch.where(max_td == -float('inf'), torch.zeros_like(max_td), max_td)

    spikiness = max_td / (mean_abs_td + 1e-8)

    return {
        "mean_abs_td": mean_abs_td.mean().item(),
        "max_td": max_td.mean().item(),
        "spikiness": spikiness.mean().item(),
    }


def compute_position_correlation(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """V(t)와 position의 Pearson 상관관계 (길이 편향 감지)

    높은 상관관계: "뒤로 갈수록 V 상승" = 길이 편향 (암기)
    낮은 상관관계: 위치와 무관하게 V 결정 = 논리적 학습

    Args:
        values: [batch, seq, 1] Value head 출력
        mask: [batch, seq] 유효 토큰 마스크 (labels != -100)

    Returns:
        correlation: -1~1 (>0.7이면 길이 편향 의심)
    """
    values = values.squeeze(-1)  # [batch, seq]
    batch_size, seq_len = values.shape
    device = values.device

    # Position index (0, 1, 2, ...)
    positions = torch.arange(seq_len, device=device, dtype=values.dtype)
    positions = positions.unsqueeze(0).expand(batch_size, -1)

    correlations = []
    for b in range(batch_size):
        valid_idx = mask[b].bool()
        n_valid = valid_idx.sum().item()
        if n_valid < 3:  # 최소 3개 토큰 필요
            continue

        v = values[b, valid_idx]
        p = positions[b, valid_idx]

        # Pearson correlation
        v_mean = v.mean()
        p_mean = p.mean()
        v_centered = v - v_mean
        p_centered = p - p_mean

        numerator = (v_centered * p_centered).sum()
        denominator = v_centered.norm() * p_centered.norm() + 1e-8

        corr = (numerator / denominator).item()
        correlations.append(corr)

    return sum(correlations) / len(correlations) if correlations else 0.0


def get_scheduled_lambda(
    current_step: int,
    warmup_steps: int = 250,
    decay_steps: int = 500,
    lam_start: float = 1.0,
    lam_end: float = 0.95,
    schedule_type: str = "linear",
) -> float:
    """Lambda scheduling (Hold & Decay)

    EOS-only warmup 대신 λ 값을 동적으로 조절하여 학습 안정성 확보.
    초기에 Pure MC (λ=1.0)로 방향성을 잡고, 점진적으로 TD mixture로 전환.

    Phase 1 (Hold): warmup_steps 동안 λ=start 유지
        - 모델이 terminal reward 패턴을 확실히 학습
        - V=0.5 편향에서 벗어나도록 강제

    Phase 2 (Decay): decay_steps 동안 start → end 감소
        - linear: 선형 감소
        - cosine: 코사인 감소 (초반/후반 느리게, 중반 빠르게)

    Phase 3 (Stable): 이후 λ=end 유지
        - 토큰별 차별화된 타겟으로 일반화 성능 향상

    Args:
        current_step: 현재 global step
        warmup_steps: Hold 기간 (λ=start 유지)
        decay_steps: Decay 기간 (start → end)
        lam_start: 초기 λ 값 (기본값 1.0, Pure MC)
        lam_end: 최종 λ 값 (기본값 0.95)
        schedule_type: "linear" 또는 "cosine"

    Returns:
        현재 step의 λ 값
    """
    import math

    # constant 타입: 학습 전체에 걸쳐 동일한 λ 사용
    if schedule_type == "constant":
        return lam_start

    if current_step < warmup_steps:
        return lam_start
    elif current_step < warmup_steps + decay_steps:
        progress = (current_step - warmup_steps) / decay_steps

        if schedule_type == "cosine":
            # Cosine decay: 초반/후반 느리게, 중반 빠르게
            decay_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return lam_end + (lam_start - lam_end) * decay_factor
        else:
            # Linear decay
            return lam_start - progress * (lam_start - lam_end)
    else:
        return lam_end


def create_eos_only_mask(loss_mask: torch.Tensor) -> torch.Tensor:
    """마지막 유효 토큰(Output 끝)만 1인 마스크 생성

    Warmup 학습에서 terminal position만 학습하기 위한 마스크.
    loss_mask가 [0,0,1,1,1,0,0] 형태(Instruction 마스킹)일 때
    마지막 1의 위치를 정확히 찾음.

    Args:
        loss_mask: [batch, seq] Output 토큰 마스크 (labels != -100)

    Returns:
        eos_mask: [batch, seq] 마지막 유효 토큰만 1
    """
    batch_size, seq_len = loss_mask.shape
    device = loss_mask.device

    # 인덱스 [1, 2, ..., seq_len] 생성 (0이 아닌 1부터 시작하여 0값과 구분)
    seq_indices = torch.arange(1, seq_len + 1, device=device).unsqueeze(0)

    # 유효 위치만 인덱스 값 유지, 나머지는 0
    valid_indices = seq_indices * loss_mask.float()

    # 각 배치의 최대 인덱스 = 마지막 유효 토큰 위치
    last_valid_pos = valid_indices.argmax(dim=1)

    # EOS 마스크 생성
    eos_mask = torch.zeros_like(loss_mask, dtype=torch.float32)
    batch_indices = torch.arange(batch_size, device=device)

    # 유효 토큰이 있는 배치만 마킹
    has_valid_tokens = loss_mask.sum(dim=1) > 0
    eos_mask[batch_indices[has_valid_tokens], last_valid_pos[has_valid_tokens]] = 1.0

    return eos_mask


def create_output_end_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Output 영역의 마지막 토큰(EOS) 위치 마스크 생성

    Random Window 적용 여부와 무관하게 진짜 output 끝 위치를 식별.
    Lambda Return의 terminal position (G_T = R) 설정에 사용.

    create_eos_only_mask와의 차이:
    - create_eos_only_mask: loss_mask 기준 (윈도우 마스킹 후에는 윈도우 끝)
    - create_output_end_mask: attention_mask 기준 (진짜 EOS, 윈도우 무관)

    Args:
        attention_mask: [batch, seq] Padding 제외 마스크 (1: 유효, 0: padding)

    Returns:
        output_end_mask: [batch, seq] output 끝 위치만 1
    """
    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device

    # 각 시퀀스의 마지막 유효 토큰 위치 계산
    seq_lengths = attention_mask.sum(dim=1)  # [batch]
    last_valid_indices = (seq_lengths - 1).clamp(min=0).long()  # [batch]

    # EOS 마스크 생성
    output_end_mask = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)
    batch_indices = torch.arange(batch_size, device=device)

    # 유효 토큰이 있는 시퀀스만 마킹
    has_valid = seq_lengths > 0
    output_end_mask[batch_indices[has_valid], last_valid_indices[has_valid]] = 1.0

    return output_end_mask


def compute_mc_value_loss(
    value_logits: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Monte Carlo Value Loss (Tokenwise MSE)

    Terminal reward를 모든 토큰에 전파하여 MSE loss 계산
    V(s_t) → R (correct: 1.0, incorrect: 0.0)

    Args:
        value_logits: [batch, seq, 1] Value head 출력
        rewards: [batch] Binary reward (0.0: incorrect, 1.0: correct)
        attention_mask: [batch, seq] 유효 토큰 마스크 (padding 제외)
        loss_mask: [batch, seq] 학습 대상 토큰 마스크 (labels != -100)

    Returns:
        mc_loss: scalar tensor
    """
    _, seq_len, _ = value_logits.shape
    dtype = value_logits.dtype

    # MC targets: 모든 토큰에 동일한 terminal reward 할당
    mc_targets = rewards.view(-1, 1).expand(-1, seq_len).to(dtype)

    # 학습 대상 토큰만으로 MSE 계산 (Instruction 제외)
    combined_mask = attention_mask * loss_mask

    # Tokenwise MSE
    values = value_logits.squeeze(-1)
    mse = (values - mc_targets) ** 2

    # 시퀀스별 평균 후 배치 평균 (길이 편향 방지)
    seq_lengths = combined_mask.sum(dim=1)
    valid_seq_mask = seq_lengths > 0
    n_valid = valid_seq_mask.sum()

    if n_valid == 0:
        return torch.tensor(0.0, device=mse.device, dtype=mse.dtype)

    seq_mse = (mse * combined_mask).sum(dim=1) / (seq_lengths + 1e-8)
    # 유효 시퀀스만 평균 (0-length 시퀀스 제외)
    masked_mse = seq_mse[valid_seq_mask].sum() / n_valid

    return masked_mse


def compute_lambda_return(
    values: torch.Tensor,
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    output_end_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fitted λ-Return 타겟 계산 (Offline TD)

    GAE 스타일의 λ-return을 offline 환경에 적용.
    Terminal reward만 있을 때 역방향으로 타겟 전파.

    수식:
        G_t^λ = (1-λ)γV_{t+1} + λγG_{t+1}^λ  (r_t=0 for t<T)
        G_T = R (terminal)

    γ=1.0, λ=0.95 (InstructGPT 설정):
        - 후반 토큰: R에 가까운 타겟
        - 전반 토큰: V 예측에 의존하는 타겟

    Args:
        values: [batch, seq] Value model 예측 (detached 상태로 전달)
        rewards: [batch] Terminal reward (1.0: correct, 0.0: incorrect)
        loss_mask: [batch, seq] 학습 대상 토큰 마스크 (labels != -100)
        gamma: Discount factor (기본값 1.0)
        lam: GAE smoothing factor (기본값 0.95)
        output_end_mask: [batch, seq] 진짜 EOS 위치 마스크 (옵션)
            - 제공되면: EOS 위치에서 G_T = R 설정 후 전체 시퀀스 역방향 전파
            - 미제공: loss_mask 기준 마지막 위치 사용 (기존 동작)

    Returns:
        lambda_returns: [batch, seq] 위치별 λ-return 타겟
    """
    batch_size, seq_len = values.shape
    device = values.device
    dtype = values.dtype

    lambda_returns = torch.zeros(batch_size, seq_len, device=device, dtype=dtype)

    for b in range(batch_size):
        # 유효한 토큰 위치 추출 (loss_mask 기준)
        valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
        if len(valid_positions) == 0:
            continue

        # Terminal position 결정
        if output_end_mask is not None:
            # 진짜 EOS 위치 사용 (Random Window 적용 시에도 정확한 terminal)
            eos_positions = output_end_mask[b].nonzero(as_tuple=True)[0]
            if len(eos_positions) == 0:
                continue
            terminal_pos = eos_positions[0].item()

            # G_T = R (terminal position)
            lambda_returns[b, terminal_pos] = rewards[b]

            # 역방향 전파: terminal_pos-1부터 0까지 모든 위치
            # Loss 계산 시 loss_mask로 필터링되므로 전체 시퀀스에 계산해도 무방
            G_next = rewards[b]
            for t in range(terminal_pos - 1, -1, -1):
                V_next = values[b, t + 1]
                td_component = (1 - lam) * gamma * V_next
                mc_component = lam * gamma * G_next
                G_t = td_component + mc_component

                lambda_returns[b, t] = G_t
                G_next = G_t
        else:
            # 기존 동작: loss_mask 범위 내에서만 역방향 전파
            terminal_pos = valid_positions[-1].item()
            lambda_returns[b, terminal_pos] = rewards[b]

            G_next = rewards[b]
            for i in range(len(valid_positions) - 2, -1, -1):
                t = valid_positions[i].item()
                t_next = valid_positions[i + 1].item()

                V_next = values[b, t_next]
                td_component = (1 - lam) * gamma * V_next
                mc_component = lam * gamma * G_next
                G_t = td_component + mc_component

                lambda_returns[b, t] = G_t
                G_next = G_t

    return lambda_returns


def compute_lambda_value_loss(
    value_logits: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    loss_type: str = "huber",
    huber_delta: float = 0.5,
    output_end_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """λ-Return 기반 Value Loss

    MC MSE를 대체하는 Fitted TD 방식의 value loss.
    타겟 계산 시 현재 V 예측을 활용하여 위치별 차별화된 타겟 생성.

    Args:
        value_logits: [batch, seq, 1] Value head 출력
        rewards: [batch] Terminal reward (1.0: correct, 0.0: incorrect)
        attention_mask: [batch, seq] Padding 마스크
        loss_mask: [batch, seq] 학습 대상 토큰 마스크 (labels != -100, 윈도우 적용됨)
        gamma: Discount factor (기본값 1.0)
        lam: GAE smoothing factor (기본값 0.95)
        loss_type: "huber" 또는 "mse" (기본값 huber)
        huber_delta: Huber loss delta (기본값 0.5, 0/1 종단 보상에 최적)
        output_end_mask: [batch, seq] 진짜 EOS 위치 마스크 (옵션)
            - 제공되면: EOS 위치에서 terminal reward 설정 (윈도우 무관)
            - 미제공: loss_mask 기준 마지막 위치 사용 (기존 동작)

    Returns:
        loss: Scalar tensor
    """
    values = value_logits.squeeze(-1)  # [batch, seq]

    # λ-return 타겟 계산 (gradient 차단)
    with torch.no_grad():
        lambda_targets = compute_lambda_return(
            values.detach(), rewards, loss_mask, gamma, lam,
            output_end_mask=output_end_mask,
        )

    combined_mask = attention_mask * loss_mask

    if loss_type == "huber":
        # Huber loss: outlier에 강건 (delta 이내 MSE, 초과 시 선형)
        loss = F.smooth_l1_loss(
            values, lambda_targets,
            reduction='none',
            beta=huber_delta
        )
    else:
        # MSE loss (기존 동작)
        loss = (values - lambda_targets) ** 2

    # 시퀀스별 평균 후 배치 평균 (길이 편향 방지)
    seq_lengths = combined_mask.sum(dim=1)
    valid_seq_mask = seq_lengths > 0
    n_valid = valid_seq_mask.sum()

    if n_valid == 0:
        return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

    seq_loss = (loss * combined_mask).sum(dim=1) / (seq_lengths + 1e-8)
    # 유효 시퀀스만 평균 (0-length 시퀀스 제외)
    masked_loss = seq_loss[valid_seq_mask].sum() / n_valid
    return masked_loss
