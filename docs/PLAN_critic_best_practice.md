# Critic Best Practice 개선 계획서

Stage-1 Value Model 학습에 RL Best Practice 적용을 위한 개선 계획.

---

## 데이터 특성

- **학습 대상**: Output 토큰만 (Instruction 마스킹)
- **평균 답변 길이**: ~200 토큰
- **보상 구조**: 종단 0/1 이진 보상 (is_correct)
- **유효 전파 길이**: 1/(1-λ) ≈ 100~200 토큰 필요

---

## 현재 상태 분석

### 기존 구현

| 항목 | 현재 구현 | 위치 |
|------|----------|------|
| Loss | MSE | `pairwise_utils.py:336-337` |
| Target Network | 없음 (detach만) | `pairwise_utils.py:329-332` |
| γ (gamma) | 1.0 | `critic_mlp.yaml:92` |
| λ (lambda) | 1.0→0.995 스케줄링 | `critic_mlp.yaml:98-103` |
| Value Head LR | 5e-4 | `critic_mlp.yaml:86` |
| LoRA LR | 1e-4 | `critic_mlp.yaml:79` |
| Warmup | 1% | `critic_mlp.yaml:108` |
| Grad clip | 1.0 | `critic_mlp.yaml:59` |

### 최종 권장사항 (데이터 특성 반영)

| 항목 | 권장값 | 근거 |
|------|--------|------|
| Loss | **Huber (δ=0.5)** | 0/1 종단 보상에서 기울기 제어 강화 |
| Target Network | **Polyak EMA (τ=0.997)** | 부트스트랩 발산 방지, step마다 갱신 |
| γ (gamma) | 1.0 | InstructGPT 표준, 할인 불필요 |
| λ (lambda) | **0.995 고정** | 평균 길이 200 → 유효 전파 200토큰 |
| Value Head LR | 2e-4 | - |
| LoRA LR | 5e-5 | Head의 1/4 |
| Warmup | 5% | - |
| min_lr_ratio | 0.0 | - |
| Grad clip | 1.0 | 표준 |
| **검증 셋** | val 500 고정 + 2k hold-out | 과적합/캘리브레이션 추적 |

### λ 선택 근거

```
유효 전파 길이 ≈ 1/(1-λ)

λ=0.95  → 전파 길이 ~20 토큰  (평균 200에 부족)
λ=0.99  → 전파 길이 ~100 토큰
λ=0.995 → 전파 길이 ~200 토큰 (평균 답변 길이와 일치)
```

---

## 개선 대상 파일

```
src/weighted_mtp/
├── utils/
│   ├── pairwise_utils.py          # Huber Loss 도입, Brier Score 추가
│   └── logging_utils.py           # Brier Score 로깅 함수 추가
├── value_weighting/
│   ├── target_network_ema.py      # 신규: Target Network EMA (FSDP 호환)
│   └── __init__.py                # Export 추가
├── pipelines/
│   └── run_critic.py              # Target Network 통합, 검증 지표 확장
└── models/
    └── value_model.py             # (변경 없음, 참조용)

configs/production/
└── critic_mlp.yaml                # 설정값 수정 (λ, δ, 검증 셋)
```

---

## Phase 1: Huber Loss 도입

### 목표
MSE Loss를 Huber Loss로 대체하여 outlier에 강건한 학습 구현.

### 변경 파일
- `src/weighted_mtp/utils/pairwise_utils.py`

### 상세 설계

**기존 코드** (`pairwise_utils.py:334-338`):
```python
# MSE loss (masked)
combined_mask = attention_mask * loss_mask
mse = (values - lambda_targets) ** 2
masked_mse = (mse * combined_mask).sum() / (combined_mask.sum() + 1e-8)
return masked_mse
```

**변경 코드**:
```python
def compute_lambda_value_loss(
    value_logits: torch.Tensor,
    rewards: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.995,
    loss_type: str = "huber",  # 신규 파라미터
    huber_delta: float = 0.5,   # 신규 파라미터 (0/1 보상에 최적)
) -> torch.Tensor:
    """λ-Return 기반 Value Loss

    Args:
        ...
        loss_type: "mse" 또는 "huber" (기본값 huber)
        huber_delta: Huber loss delta (기본값 0.5, 0/1 종단 보상에 최적)
    """
    values = value_logits.squeeze(-1)

    with torch.no_grad():
        lambda_targets = compute_lambda_return(
            values.detach(), rewards, loss_mask, gamma, lam
        )

    combined_mask = attention_mask * loss_mask

    if loss_type == "huber":
        # Huber loss: outlier에 강건
        loss = F.smooth_l1_loss(
            values, lambda_targets,
            reduction='none',
            beta=huber_delta
        )
    else:
        # MSE loss (기존 동작)
        loss = (values - lambda_targets) ** 2

    masked_loss = (loss * combined_mask).sum() / (combined_mask.sum() + 1e-8)
    return masked_loss
```

### 호환성
- `loss_type` 기본값을 "huber"로 설정하여 기존 호출부 수정 최소화
- 필요 시 "mse"로 fallback 가능

---

## Phase 2: Target Network EMA 구현 (FSDP 호환)

### 목표
LoRA + Value Head 파라미터의 Polyak EMA를 추적하여 부트스트랩 안정화.
FSDP 분산학습 환경에서 정상 동작하도록 `summon_full_params` API 활용.

### 신규 파일
- `src/weighted_mtp/value_weighting/target_network_ema.py`

### 설계 원칙

1. **메모리 효율**: 전체 2.7B backbone이 아닌 LoRA + Value Head만 EMA 추적
   - LoRA: ~8M params (rank=32 기준)
   - Value Head (MLP): ~1.3M params
   - 총 ~10M params → ~40MB 추가 메모리

2. **FSDP 호환**: `summon_full_params` API 활용
   - FSDP FULL_SHARD에서 각 rank는 파라미터의 일부(shard)만 보유
   - 직접 `param.data` 접근 시 sharded 상태 파괴 위험
   - `summon_full_params`로 안전하게 full tensor 접근

3. **기존 패턴 준수**: `checkpoint_utils.py:661-676`의 패턴 참고
   - `summon_full_params` API 활용
   - 기존 코드베이스와 일관된 FSDP 처리 방식

4. **랭크 동기화**: 모든 rank가 동일한 EMA 유지
   - `rank0_only=False`로 모든 rank에서 full params 접근
   - broadcast 없이 각 rank가 독립적으로 동일한 EMA 계산
   - 추가 메모리: ~40MB × 3 ranks = ~120MB (H200에서 무시 가능)

### FSDP 호환성 분석

| 문제점 | 원인 | 해결책 |
|--------|------|--------|
| 초기화 시 불완전 파라미터 | 각 rank가 shard만 보유 | `summon_full_params`로 full tensor 접근 |
| EMA 업데이트 불일치 | rank마다 독립 업데이트 | `rank0_only=False`로 모든 rank 동일 EMA |
| forward 시 FSDP 상태 파괴 | `param.data.copy_()` 직접 호출 | `summon_full_params(writeback=True)` 사용 |

### 상세 설계

```python
# src/weighted_mtp/value_weighting/target_network_ema.py

"""Target Network EMA (Polyak Averaging) - FSDP 호환

LoRA + Value Head 파라미터의 EMA를 유지하여 부트스트랩 안정화.
TD 타겟 계산 시 online network 대신 target network 사용.

수식: θᵗᵃʳᵍ ← τ·θᵗᵃʳᵍ + (1−τ)·θ  (τ=0.997)

FSDP 호환 전략:
- EMA params는 CPU에서 full tensor로 별도 관리 (rank 0만 유효)
- summon_full_params로 FSDP sharded 모델에 안전하게 접근
- checkpoint_utils.py의 기존 패턴과 일관성 유지

메모리 효율:
- 전체 backbone (2.7B)가 아닌 학습 가능 파라미터만 EMA
- LoRA (~8M) + Value Head (~1.3M) ≈ 10M params (~40MB)
"""

import logging
import torch
from torch import nn
from typing import Optional

logger = logging.getLogger(__name__)


class TargetNetworkEMA:
    """LoRA + Value Head Target Network EMA (FSDP 호환)

    학습 가능 파라미터(LoRA + Value Head)만 EMA 추적하여
    메모리 효율적인 target network 구현.

    FSDP 환경에서 summon_full_params API를 활용하여
    sharded 파라미터에 안전하게 접근.

    Usage:
        1. FSDP wrapping 후 인스턴스 생성 (중요!)
        2. 매 optimizer step 후 update() 호출
        3. λ-return 타겟 계산 시 get_target_values() 사용
        4. checkpoint 저장 시 state_dict() 포함

    Examples:
        >>> # FSDP wrapping 후 생성
        >>> value_model = wrap_model_fsdp(value_model, device, ...)
        >>> target_ema = TargetNetworkEMA(value_model, tau=0.997)
        >>>
        >>> for batch in batches:
        ...     # Forward with online network
        ...     value_logits = value_model(input_ids, attention_mask)
        ...     # Target values for bootstrapping
        ...     target_values = target_ema.get_target_values(
        ...         input_ids, attention_mask
        ...     )
        ...     # Compute loss with target_values
        ...     loss.backward()
        ...     optimizer.step()
        ...     target_ema.update()
    """

    def __init__(
        self,
        value_model: nn.Module,
        tau: float = 0.997,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            value_model: ValueModel 인스턴스 (FSDP-wrapped 가능)
            tau: EMA decay factor (0.997 = 99.7% 이전 + 0.3% 현재)
            device: 디바이스 (None이면 모델에서 추론)
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        self.tau = tau
        self.value_model = value_model
        self.is_fsdp = isinstance(value_model, FSDP)

        # 디바이스 설정 (EMA는 CPU에서 관리)
        if device is not None:
            self.device = device
        elif self.is_fsdp:
            # FSDP 모델은 summon_full_params 후에만 파라미터 접근 가능
            self.device = torch.device("cpu")
        else:
            self.device = next(value_model.parameters()).device

        # EMA 파라미터 저장소 (CPU full tensor, rank 0만 유효)
        self.ema_params = {}
        self._init_ema_params()

        logger.info(
            f"TargetNetworkEMA 초기화: tau={tau}, "
            f"is_fsdp={self.is_fsdp}, params={self.param_count:,}"
        )

    def _init_ema_params(self):
        """학습 가능 파라미터의 EMA 초기화 (FSDP 호환)"""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if self.is_fsdp:
            # FSDP: summon_full_params로 전체 파라미터 접근
            # rank0_only=False로 모든 rank가 동일한 EMA 유지
            with FSDP.summon_full_params(
                self.value_model,
                writeback=False,
                offload_to_cpu=True,
            ):
                for name, param in self.value_model.named_parameters():
                    if param.requires_grad:
                        # CPU로 복사하여 EMA 저장
                        self.ema_params[name] = param.data.cpu().clone()
        else:
            # Non-FSDP: 직접 접근
            for name, param in self.value_model.named_parameters():
                if param.requires_grad:
                    self.ema_params[name] = param.data.cpu().clone()

    @torch.no_grad()
    def update(self):
        """EMA 업데이트: θᵗᵃʳᵍ ← τ·θᵗᵃʳᵍ + (1−τ)·θ (FSDP 호환)"""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if self.is_fsdp:
            # FSDP: summon_full_params로 전체 파라미터 접근
            # rank0_only=False로 모든 rank가 동일한 EMA 유지
            with FSDP.summon_full_params(
                self.value_model,
                writeback=False,
                offload_to_cpu=True,
            ):
                for name, param in self.value_model.named_parameters():
                    if name in self.ema_params:
                        self.ema_params[name].mul_(self.tau).add_(
                            param.data.cpu(), alpha=1 - self.tau
                        )
        else:
            # Non-FSDP: 직접 접근
            for name, param in self.value_model.named_parameters():
                if name in self.ema_params:
                    self.ema_params[name].mul_(self.tau).add_(
                        param.data.cpu(), alpha=1 - self.tau
                    )

    @torch.no_grad()
    def get_target_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Target network로 value 계산 (FSDP 호환)

        EMA 파라미터를 임시로 적용하여 forward 수행.
        FSDP 환경에서는 summon_full_params(writeback=True)로
        안전하게 파라미터 교체 후 복원.

        Args:
            input_ids: [batch, seq] 입력 토큰 ID
            attention_mask: [batch, seq] 어텐션 마스크

        Returns:
            target_values: [batch, seq, 1] Target value 예측
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        was_training = self.value_model.training
        self.value_model.eval()

        if self.is_fsdp:
            # FSDP: summon_full_params + writeback=True로 임시 파라미터 교체
            with FSDP.summon_full_params(
                self.value_model,
                writeback=True,  # 수정사항을 shard로 다시 분산
                offload_to_cpu=False,  # GPU에서 forward 수행
            ):
                # 현재 파라미터 백업
                backup = {}
                for name, param in self.value_model.named_parameters():
                    if name in self.ema_params:
                        backup[name] = param.data.clone()
                        param.data.copy_(self.ema_params[name].to(param.device))

                # Target forward
                target_values = self.value_model(input_ids, attention_mask)

                # 원본 파라미터 복원
                for name, param in self.value_model.named_parameters():
                    if name in backup:
                        param.data.copy_(backup[name])
        else:
            # Non-FSDP: 직접 파라미터 교체
            backup = {}
            for name, param in self.value_model.named_parameters():
                if name in self.ema_params:
                    backup[name] = param.data.clone()
                    param.data.copy_(self.ema_params[name].to(param.device))

            target_values = self.value_model(input_ids, attention_mask)

            # 원본 복원
            for name, param in self.value_model.named_parameters():
                if name in backup:
                    param.data.copy_(backup[name])

        # 원래 모드 복원
        if was_training:
            self.value_model.train()

        return target_values

    def state_dict(self) -> dict:
        """Checkpoint 저장용 state dict

        모든 rank가 동일한 ema_params 보유.
        checkpoint 저장은 rank 0에서만 수행됨 (save_value_model_checkpoint 참고).
        """
        return {
            "tau": self.tau,
            "ema_params": {k: v.cpu() for k, v in self.ema_params.items()},
        }

    def load_state_dict(self, state_dict: dict):
        """Checkpoint에서 복원

        학습 재개 시 EMA 상태 복원.
        모든 rank에서 동일하게 로드됨.
        """
        self.tau = state_dict.get("tau", self.tau)
        for name, param in state_dict.get("ema_params", {}).items():
            if name in self.ema_params:
                self.ema_params[name] = param.cpu().clone()

    @property
    def param_count(self) -> int:
        """EMA 추적 중인 파라미터 수"""
        return sum(p.numel() for p in self.ema_params.values())
```

### Export 추가

`src/weighted_mtp/value_weighting/__init__.py`:
```python
from weighted_mtp.value_weighting.target_network_ema import TargetNetworkEMA

__all__ = [
    ...
    "TargetNetworkEMA",
]
```

---

## Phase 3: run_critic.py Target Network 통합

### 목표
Target Network EMA를 critic 학습 파이프라인에 통합.
FSDP 환경에서 올바른 순서로 초기화 및 업데이트 수행.

### 변경 파일
- `src/weighted_mtp/pipelines/run_critic.py`
- `src/weighted_mtp/utils/pairwise_utils.py` (compute_lambda_return 시그니처 확장)
- `src/weighted_mtp/utils/checkpoint_utils.py` (target_ema_state 파라미터 추가)

### FSDP 통합 시 주의사항

1. **초기화 순서**: FSDP wrapping 후 TargetNetworkEMA 생성
   - FSDP wrapping 전에는 파라미터가 sharded 상태가 아님
   - wrapping 후 생성해야 올바른 `is_fsdp` 감지

2. **업데이트 위치**: optimizer.step() 직후
   - gradient accumulation 완료 후 업데이트
   - scheduler.step()과 같은 시점

3. **Checkpoint 저장**: rank 0에서만 유효
   - `save_value_model_checkpoint()`는 이미 rank 0에서만 저장
   - target_ema_state도 동일하게 처리

### 상세 설계

**1. compute_lambda_return() 확장** (`pairwise_utils.py`):

```python
def compute_lambda_return(
    values: torch.Tensor,
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.995,
    target_values: Optional[torch.Tensor] = None,  # 신규: Target network values
) -> torch.Tensor:
    """Fitted λ-Return 타겟 계산

    Args:
        ...
        target_values: [batch, seq] Target network의 value 예측
            None이면 values를 target으로 사용 (기존 동작)
    """
    # target_values가 주어지면 bootstrapping에 사용
    bootstrap_values = target_values if target_values is not None else values

    # 역방향 전파 시 bootstrap_values 사용
    for i in range(len(valid_positions) - 2, -1, -1):
        ...
        V_next = bootstrap_values[b, t_next]  # target network value
        ...
```

**2. run_critic.py 통합**:

```python
from weighted_mtp.value_weighting import TargetNetworkEMA

def run_critic_training(config: DictConfig):
    ...

    # 6. Value Model 로드
    value_model = load_value_model(config, device)

    # ... (bias 초기화, LoRA 설정 등)

    # FSDP wrapping
    value_model = wrap_model_fsdp(
        value_model,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        ...
    )

    # Target Network EMA 초기화 (FSDP wrapping 후 - 중요!)
    use_target_network = config.training.value_loss.get("use_target_network", False)
    target_tau = config.training.value_loss.get("target_tau", 0.997)

    target_ema = None
    if use_target_network:
        target_ema = TargetNetworkEMA(value_model, tau=target_tau, device=device)
        logger.info(
            f"Target Network EMA 활성화: tau={target_tau}, "
            f"params={target_ema.param_count:,}, is_fsdp={target_ema.is_fsdp}"
        )

    ...

    # Training loop 내부
    for _ in range(batches_this_period):
        ...

        # Loss 계산 시 target network 사용
        if loss_type == "lambda_return":
            # Target values 계산 (target network 사용 시)
            pos_target_values = None
            neg_target_values = None

            if target_ema is not None:
                # Target forward (FSDP 호환)
                combined_target_values = target_ema.get_target_values(
                    combined_input_ids, combined_attention_mask
                )
                pos_target_values = combined_target_values[:batch_size].squeeze(-1)
                neg_target_values = combined_target_values[batch_size:].squeeze(-1)

            pos_lambda_loss = compute_lambda_value_loss(
                pos_value_logits, pos_rewards, pos_attention_mask, pos_loss_mask.float(),
                gamma=lambda_gamma, lam=current_lam,
                loss_type=loss_fn_type,
                huber_delta=huber_delta,
                target_values=pos_target_values,  # Target network values
            )
            neg_lambda_loss = compute_lambda_value_loss(
                neg_value_logits, neg_rewards, neg_attention_mask, neg_loss_mask.float(),
                gamma=lambda_gamma, lam=current_lam,
                loss_type=loss_fn_type,
                huber_delta=huber_delta,
                target_values=neg_target_values,
            )
            ...

        # Optimizer step 후 EMA 업데이트
        if accumulation_counter >= gradient_accumulation_steps:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            # Target Network EMA 업데이트 (optimizer.step() 직후)
            if target_ema is not None:
                target_ema.update()

            global_step += 1
            accumulation_counter = 0

    ...

    # Checkpoint 저장 시 EMA state 포함
    save_value_model_checkpoint(
        value_model=value_model,
        optimizer=optimizer,
        epoch=current_epoch,
        train_metrics={"train_loss": train_loss_avg},
        val_metrics=aggregated_val_metrics,
        checkpoint_path=checkpoint_path,
        config=config,
        s3_upload=use_s3_upload,
        experiment_name=config.experiment.name,
        target_ema_state=target_ema.state_dict() if target_ema else None,
    )
```

**3. checkpoint_utils.py 확장**:

`save_value_model_checkpoint()`에 `target_ema_state` 파라미터 추가:
```python
def save_value_model_checkpoint(
    value_model,
    optimizer: torch.optim.Optimizer,
    epoch: float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    checkpoint_path: Path | str,
    config: Any = None,
    s3_upload: bool = False,
    experiment_name: str | None = None,
    target_ema_state: dict | None = None,  # 신규
) -> None:
    ...

    checkpoint = {
        ...
        "target_ema_state": target_ema_state,  # 신규
    }
```

---

## Phase 4: Config 설정 수정

### 목표
데이터 특성(평균 답변 길이 200, 0/1 종단 보상)에 맞춘 하이퍼파라미터 조정.

### 변경 파일
- `configs/production/critic_mlp.yaml`

### 변경 내용

```yaml
# 데이터 샘플링 (검증 셋 고정)
data_sampling:
  seed: 84
  val_n_samples: 500          # 고정 검증 셋 (캘리브레이션 추적)
  holdout_n_samples: 2000     # 신규: 별도 hold-out (과적합 감지)
  use_pairwise: true
  n_samples: 60000
  max_pairs_per_problem: 10

# 학습 설정 (H200 3-GPU 최적화)
training:
  n_epochs: 2.0
  batch_size: 24
  gradient_accumulation_steps: 1

  max_grad_norm: 1.0  # 유지
  log_interval: 5

  backbone_frozen: true

  # LoRA 설정 (LR 조정)
  use_lora: true
  lora:
    rank: 32
    alpha: 64.0
    dropout: 0.1
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
    learning_rate: 5.0e-5     # 1e-4 → 5e-5 (Head의 1/4)
    weight_decay: 0.01

  # Value Head 설정 (LR 조정)
  value_head:
    type: mlp
    dropout: 0.2              # 0.3 → 0.2
    learning_rate: 2.0e-4     # 5e-4 → 2e-4
    weight_decay: 0.01

  # Value loss 설정 (핵심 변경)
  value_loss:
    type: "lambda_return"
    gamma: 1.0
    coef: 1.0
    bias_init: 0.5

    # Loss function 설정 (신규)
    loss_fn: "huber"          # "mse" → "huber"
    huber_delta: 0.5          # 0/1 종단 보상에 최적 (기울기 제어 강화)

    # Target Network 설정 (신규)
    use_target_network: true  # 신규
    target_tau: 0.997         # Polyak EMA, step마다 갱신

    # Lambda 설정 (고정값으로 변경)
    lambda_schedule:
      type: constant          # linear → constant
      start: 0.995            # 평균 답변 길이 200 → 유효 전파 200토큰
      end: 0.995

  # Learning rate scheduler (warmup 조정)
  lr_scheduler:
    type: cosine
    warmup_ratio: 0.05        # 0.01 → 0.05
    min_lr_ratio: 0.0         # 0.05 → 0.0
```

### 검증 지표 (로깅 필수)

| 지표 | 설명 | 목적 |
|------|------|------|
| Huber Loss | 학습 loss | 수렴 추적 |
| MSE | 보조 지표 | Huber와 비교 |
| MAE | 보조 지표 | 평균 오차 크기 |
| **Brier Score** | (V - R)² 평균 | 캘리브레이션 품질 |
| Pairwise Accuracy | V(correct) > V(incorrect) | 순위 정확도 |
| Value Mean (pos/neg) | 정답/오답 평균 | 분리도 확인 |

---

## 구현 순서 및 의존성

```
Phase 1 (독립)
    │
    ▼
Phase 2 (독립)
    │
    ▼
Phase 3 (Phase 1, 2에 의존)
    │
    ▼
Phase 4 (Phase 3 완료 후)
```

### 예상 작업량

| Phase | 파일 수 | 예상 난이도 |
|-------|--------|------------|
| 1 | 1 | 낮음 (함수 수정) |
| 2 | 2 | 중간 (신규 클래스, FSDP 호환) |
| 3 | 3 | 높음 (파이프라인 통합) |
| 4 | 1 | 낮음 (config 수정) |

---

## FSDP 호환성 요약

### 핵심 설계 결정

| 항목 | 설계 | 근거 |
|------|------|------|
| EMA 저장 위치 | CPU full tensor | FSDP shard와 독립적으로 관리 |
| 파라미터 접근 | `summon_full_params` | FSDP 내부 상태 보존 |
| rank 전략 | `rank0_only=False` | 모든 rank 동일 EMA, broadcast 불필요 |
| 초기화 시점 | FSDP wrapping 후 | 올바른 `is_fsdp` 감지 |
| 업데이트 시점 | optimizer.step() 직후 | gradient accumulation 완료 후 |

### 통신 오버헤드 분석

| 연산 | all-gather 대상 | 데이터량 | 예상 시간 |
|------|----------------|----------|-----------|
| `_init_ema_params()` | LoRA + Value Head | ~20MB (bf16) | 1회, <0.1s |
| `update()` | LoRA + Value Head | ~20MB | step마다, <0.1s |
| `get_target_values()` | LoRA + Value Head | ~20MB | step마다, <0.1s |

3-GPU H200 NVLink 환경에서 **통신 오버헤드 무시 가능 수준**.

---

## 검증 계획

### Phase 1 검증
- 단위 테스트: Huber loss 출력값 검증 (δ=0.5 경계 동작)
- δ 범위 내(|error|<0.5): MSE와 유사
- δ 범위 외(|error|≥0.5): Linear로 전환
- 기존 테스트 통과 확인

### Phase 2 검증
- 단위 테스트: EMA 업데이트 정확성 (τ=0.997)
- **FSDP 호환성 테스트**: 분산환경에서 EMA 일관성 검증
- 메모리 사용량 측정 (예상: ~40MB 추가)
- step마다 갱신 확인

### Phase 3 검증
- 통합 테스트: critic 학습 파이프라인 정상 동작
- **FSDP 분산학습 테스트**: 3-GPU 환경에서 정상 동작
- 로깅 확인: target_ema 관련 메트릭
- checkpoint 저장/로드 테스트

### Phase 4 검증 (핵심)
- **검증 셋 구성**: val 500 고정 + 2k hold-out 분리
- **로깅 지표**:
  - Huber/MSE/MAE loss (학습/검증)
  - **Brier Score**: (V - R)² 평균 (캘리브레이션 품질)
  - Pairwise Accuracy (순위 정확도)
  - Value 평균/표준편차 (pos/neg 분리도)
- **과적합 감지**: hold-out 성능 추적
- **캘리브레이션 드리프트**: Brier Score 변화 모니터링

---

## 롤백 계획

각 Phase는 config 플래그로 비활성화 가능:

```yaml
value_loss:
  loss_fn: "mse"              # huber → mse (Phase 1 롤백)
  huber_delta: 1.0            # 0.5 → 1.0 (덜 공격적인 설정)
  use_target_network: false   # true → false (Phase 2, 3 롤백)
  lambda_schedule:
    start: 1.0                # 0.995 → 1.0 (Pure MC 롤백)
```

---

## 참고문헌

- **GAE**: Schulman et al. (2015). High-Dimensional Continuous Control Using GAE.
- **Huber Loss**: Mnih et al. (2015). Human level control through deep reinforcement learning.
- **Target Network (Polyak EMA)**: OpenAI Spinning Up - SAC documentation.
- **RLHF GAE**: Secrets of RLHF in Large Language Models Part I: PPO.
- **FSDP summon_full_params**: PyTorch FSDP Documentation.
