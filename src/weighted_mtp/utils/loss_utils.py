"""NTP Loss 계산 유틸리티

Next-Token Prediction Cross-Entropy Loss 계산 (uniform / weighted)
run_baseline에서 사용
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_weighted_ntp_loss(
    logits: Tensor,
    labels: Tensor,
    attention_mask: Tensor,
    weights: Tensor | None = None,
) -> dict[str, Tensor]:
    """Next-Token Prediction Cross-Entropy Loss 계산 (uniform / weighted).

    표준 shifted CE를 사용하여 logits[:, :-1]과 labels[:, 1:]을 비교한다.
    weights가 주어지면 토큰별 가중 평균, 없으면 균등 평균(Baseline)을 반환한다.

    Args:
        logits: [batch, seq, vocab] — 모델 출력 logits (3D)
        labels: [batch, seq] — 타겟 토큰 ID (-100: ignore)
        attention_mask: [batch, seq] — 유효 토큰 마스크 (1: 유효, 0: padding)
        weights: [batch, seq] 또는 None — 토큰별 가중치.
            None이면 균등 가중치 1.0 (Baseline).
            [batch, seq]이면 TAW, Random-Matched, Shuffled 등 실험용 per-token weight.

    Returns:
        dict with:
            "weighted_ce_loss": scalar tensor — 가중 CE loss (backprop용 메인 loss).
                weights=None이면 unweighted_ce_loss와 동일.
            "unweighted_ce_loss": scalar tensor — 균등 CE loss (모니터링/비교용).
    """
    vocab_size = logits.size(-1)

    # Standard shifted CE: predict next token
    shift_logits = logits[:, :-1, :].contiguous()   # [batch, seq-1, vocab]
    shift_labels = labels[:, 1:].contiguous()        # [batch, seq-1]
    shift_mask = attention_mask[:, 1:].contiguous()  # [batch, seq-1]

    # Per-token cross-entropy
    ce_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )  # [batch * (seq-1)]

    # Valid token mask: labels != -100 AND attention_mask == 1
    valid_mask = (shift_labels != -100).float() * shift_mask.float()  # [batch, seq-1]
    valid_mask_flat = valid_mask.view(-1)  # [batch * (seq-1)]
    valid_count = valid_mask.sum()

    # Unweighted CE loss
    if valid_count > 0:
        unweighted_ce_loss = (ce_loss * valid_mask_flat).sum() / valid_count
    else:
        unweighted_ce_loss = (ce_loss * valid_mask_flat).sum()

    # Weighted CE loss
    if weights is not None:
        shift_weights = weights[:, 1:].contiguous()  # [batch, seq-1]
        shift_weights_flat = shift_weights.view(-1)   # [batch * (seq-1)]
        if valid_count > 0:
            weighted_ce_loss = (ce_loss * shift_weights_flat * valid_mask_flat).sum() / valid_count
        else:
            weighted_ce_loss = (ce_loss * shift_weights_flat * valid_mask_flat).sum()
    else:
        weighted_ce_loss = unweighted_ce_loss

    return {
        "weighted_ce_loss": weighted_ce_loss,
        "unweighted_ce_loss": unweighted_ce_loss,
    }
