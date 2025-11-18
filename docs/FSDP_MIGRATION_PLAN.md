# FSDP 전환 계획서 (DDP → FSDP Migration)

## 문서 정보
- **작성일**: 2025-11-18
- **대상**: A100 80GB GPU 4-way 분산 학습
- **목적**: DDP의 메모리 한계 극복 및 batch size 최적화
- **개발원칙 준수**: 기존 구조 존중, 점진적 전환, 철저한 테스트

---

## Executive Summary

### 현재 상황
- **분산 방식**: DDP (DistributedDataParallel)
- **메모리 사용**: 80-90GB per GPU (6.7B 모델 full finetuning)
- **문제**: A100 80GB 한계 초과 → **batch_size=4 불가능**

### FSDP 전환 효과
- **메모리 절감**: 75% (80GB → 20GB per GPU)
- **Batch size 증가**: 4 (현재 목표) → 8 (가능)
- **학습 속도**: 6-10% 느려질 수 있으나, OOM 해결이 우선

### 전환 범위
- **즉시 전환 필요**: Verifiable, Baseline, Rho-1 (full finetuning)
- **유지**: Critic (Value head만 학습, 메모리 여유 충분)

---

## I. 현재 코드베이스 분석

### 1. 분산 학습 아키텍처

#### 핵심 설계 원칙 (ARCHITECTURE.md 기반)

```
분산학습 구조:
1. Torchrun 기반 프로세스 생성 (RANK, WORLD_SIZE 환경변수)
2. DDP wrapping (runtime/ddp.py)
3. Rank-aware data sampling (datasets.py)
4. Rank 0 중심 MLflow 로깅 및 checkpoint 저장
```

#### 현재 DDP 구현 (runtime/ddp.py)

**핵심 기능:**
```python
def wrap_model_ddp(model, device, find_unused_parameters=False):
    """DDP로 모델 래핑

    특징:
    - 분산 환경에서만 적용 (is_distributed() 체크)
    - MPS/CPU local test는 skip
    - device_ids 자동 설정 (CUDA만)
    - Gradient all-reduce 자동화
    """
    if not is_distributed():
        return model  # 단일 장치 환경

    return DDP(model, device_ids=[device.index])
```

**설계 의도:**
1. **환경 독립성**: 로컬(MPS/CPU) 테스트와 VESSL GPU 학습 모두 지원
2. **단순성**: DDP wrapper 적용만으로 분산 학습 가능
3. **테스트 용이성**: `is_distributed()` 플래그로 단위 테스트 가능

#### Rank-aware Data Sampling (datasets.py)

**핵심 로직:**
```python
# 메타데이터 기반 인덱스 계산 (모든 rank 동일)
all_indices = _compute_sampling_indices_from_metadata(...)

# Rank별 분할
if world_size > 1:
    rank_indices = all_indices[rank::world_size]

# Rank 0: samples[0::4]
# Rank 1: samples[1::4]
# ...
```

**설계 의도:**
1. **메모리 효율**: 각 rank가 전체의 1/world_size만 로드 (75% 절감)
2. **재현성**: 모든 rank가 동일한 인덱스 계산 로직 사용
3. **커리큘럼 학습 지원**: 메타데이터 기반 난이도 샘플링

**장점:**
- DistributedSampler 불필요 (더 유연한 샘플링 가능)
- 메타데이터만으로 전체 분포 파악 가능
- Curriculum learning과 통합 용이

### 2. 테스트 커버리지 분석

#### test_ddp.py (DDP 유틸리티 테스트)

**검증 항목:**
1. ✅ Single-device 환경에서 DDP wrapping 하지 않음
2. ✅ `unwrap_model()`로 원본 모델 복원
3. ✅ `all_reduce_scalar()`로 metric 집계
4. ✅ Forward pass 정상 동작
5. ✅ State dict 일관성

**걱정한 부분:**
- DDP wrapper 제거 후 checkpoint 저장 가능 여부
- Device 타입별 (CPU/MPS/CUDA) 호환성
- Single-device와 multi-device 코드 경로 통일

#### test_distributed_loading.py (분산 데이터 로딩 테스트)

**검증 항목:**
1. ✅ Rank별 샘플 수 균등 분배
2. ✅ 전체 rank 합치면 100% 커버 (중복 없음)
3. ✅ 재현성 (같은 seed → 같은 결과)
4. ✅ 2/4/8 GPU 환경 모두 지원

**걱정한 부분:**
- Rank간 데이터 중복/누락 없는지
- 불균등 샘플 수 (20개를 4-way로 나누면 5, 5, 5, 5)
- Deterministic 샘플링 보장

### 3. 기존 코드의 강점

**강점 1: 환경 독립성**
```python
# is_distributed() 플래그로 local/distributed 자동 전환
if not is_distributed():
    return model  # MPS/CPU local test
else:
    return DDP(model, ...)  # VESSL GPU
```

**강점 2: Rank-aware Sampling의 유연성**
- DistributedSampler 없이 커스텀 샘플링
- Metadata 기반 난이도/정답률 제어
- 메모리 효율적 (각 rank가 1/N만 로드)

**강점 3: Pure PyTorch 구현**
- `nn.Embedding`, `nn.Linear` 사용 (fairscale 제거)
- FSDP 완전 호환 가능 (ARCHITECTURE.md에 명시)
- Safetensors 저장/로딩 지원

**강점 4: 이미 FSDP 인프라 준비됨**
```python
# distributed.py에 이미 setup_fsdp_config() 존재
def setup_fsdp_config(
    sharding_strategy="FULL_SHARD",
    cpu_offload=False,
    mixed_precision="bf16",
    activation_checkpointing=True,
):
    """FSDP 설정 딕셔너리 생성 (Phase 6에서 사용)"""
```

**주석에 명시**: "실제 FSDP wrapping은 Phase 6에서 구현됩니다"
→ **이미 FSDP 전환을 계획했었음!**

### 4. 제약사항 및 고려사항

**제약 1: DDP와 FSDP의 API 차이**
```python
# DDP: 단순 wrapper
model = DDP(model, device_ids=[rank])

# FSDP: 설정 복잡
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=...,
    auto_wrap_policy=...,
    device_id=rank,
)
```

**제약 2: Checkpoint 저장/로딩**
- DDP: `unwrap_model().state_dict()` 단순
- FSDP: Full state dict gathering 필요 (`FullStateDictConfig`)

**제약 3: Rank 0 checkpoint 저장 방식**
- 현재: Rank 0만 checkpoint 저장
- FSDP: 모든 rank가 shard 저장 또는 Rank 0이 full gather

**제약 4: find_unused_parameters**
- DDP에서는 옵션으로 제공
- FSDP에서는 자동 처리 (불필요)

**고려사항 1: Gradient Accumulation 호환성**
- DDP: 자동 호환
- FSDP: `no_sync()` context manager 사용 권장

**고려사항 2: Activation Checkpointing**
- 현재: 미사용
- FSDP: `activation_checkpointing=True` 옵션으로 간단히 활성화

---

## II. FSDP 전환 Phase 계획

### 개발 원칙

1. **[원칙 1]** 기존 DDP 구조를 존중하여 점진적으로 전환
2. **[원칙 2]** 중복 없이 `wrap_model_ddp` → `wrap_model_fsdp` 깔끔하게 대체
3. **[원칙 3]** 기존 잘못된 구조 없음 (DDP는 정상 작동, 메모리만 문제)
4. **[원칙 4]** 하위 호환성 고려 불필요 (DDP는 완전히 FSDP로 대체)
5. **[원칙 5]** 각 Phase 완료 후 테스트 및 검증
6. **[원칙 6]** 의존성/환경 문제는 코드가 아니라 환경으로 해결

---

### Phase 1: FSDP 래퍼 함수 구현 (runtime/fsdp.py 생성)

**목표**: DDP와 동일한 인터페이스의 FSDP wrapper 구현

**파일 생성**: `src/weighted_mtp/runtime/fsdp.py`

**구현 내용**:
```python
"""FSDP 유틸리티

FullyShardedDataParallel model wrapping, unwrapping, checkpoint 관리
"""

from typing import Optional
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, FullStateDictConfig

from weighted_mtp.runtime.distributed import is_distributed, is_main_process


def wrap_model_fsdp(
    model: torch.nn.Module,
    device: torch.device,
    sharding_strategy: str = "FULL_SHARD",
    mixed_precision: bool = True,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
) -> torch.nn.Module:
    """FSDP로 모델 래핑

    DDP wrap_model_ddp()와 동일한 인터페이스 유지.
    분산 환경에서만 FSDP 적용, MPS/CPU local test는 skip.

    Args:
        model: 원본 모델
        device: torch.device (cuda:rank, mps, cpu)
        sharding_strategy: "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"
        mixed_precision: FP16 mixed precision 사용 여부
        cpu_offload: CPU 오프로드 (메모리 부족 시)
        activation_checkpointing: Gradient checkpointing

    Returns:
        FSDP-wrapped model (또는 원본 model if not distributed)
    """
    if not is_distributed():
        # MPS/CPU local test - no wrapping
        return model

    # Sharding strategy 변환
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    strategy = strategy_map.get(sharding_strategy, ShardingStrategy.FULL_SHARD)

    # Mixed precision 설정
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # CPU offload 설정
    cpu_offload_config = CPUOffload(offload_params=True) if cpu_offload else None

    # FSDP wrapping
    wrapped_model = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload_config,
        device_id=device.index if device.type == "cuda" else None,
        sync_module_states=True,  # 모든 rank에서 동일한 초기 weights
    )

    if is_main_process():
        logger.info(
            f"FSDP wrapping 완료: strategy={sharding_strategy}, "
            f"mixed_precision={mixed_precision}, "
            f"cpu_offload={cpu_offload}, "
            f"activation_checkpointing={activation_checkpointing}"
        )

    return wrapped_model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """FSDP wrapper 제거하여 원본 모델 추출

    DDP unwrap_model()과 동일한 인터페이스.
    Checkpoint 저장 시 사용.

    Args:
        model: FSDP-wrapped model (또는 원본 model)

    Returns:
        Original model
    """
    if isinstance(model, FSDP):
        return model.module
    return model


def save_fsdp_checkpoint(
    model: FSDP,
    save_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """FSDP 모델 checkpoint 저장 (Rank 0만)

    Full state dict를 Rank 0에 모아서 저장.

    Args:
        model: FSDP-wrapped model
        save_path: Checkpoint 저장 경로
        optimizer: Optimizer (선택)
    """
    if not is_main_process():
        return  # Rank 0만 저장

    # Full state dict 설정
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()

    # Checkpoint 저장 (Rank 0만)
    checkpoint = {"model": state_dict}

    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()

    torch.save(checkpoint, save_path)
    logger.info(f"FSDP checkpoint 저장 완료: {save_path}")


def load_fsdp_checkpoint(
    model: FSDP,
    load_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """FSDP 모델 checkpoint 로드

    Full state dict를 load하여 각 rank에 shard 분배.

    Args:
        model: FSDP-wrapped model
        load_path: Checkpoint 경로
        optimizer: Optimizer (선택)
    """
    checkpoint = torch.load(load_path, map_location="cpu")

    # Full state dict 설정
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        model.load_state_dict(checkpoint["model"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    logger.info(f"FSDP checkpoint 로드 완료: {load_path}")
```

**테스트**:
- `tests/unit/test_fsdp.py` 생성 (test_ddp.py 복사 후 수정)
- Single-device 환경에서 FSDP wrapping 하지 않는지 확인
- `unwrap_model()` 동작 확인

**검증 기준**:
- ✅ test_fsdp.py 모든 테스트 통과
- ✅ `wrap_model_fsdp()` 인터페이스가 `wrap_model_ddp()`와 일치

---

### Phase 2: Runtime 모듈 통합 (runtime/__init__.py 수정)

**목표**: FSDP 함수를 runtime 모듈에 노출

**파일 수정**: `src/weighted_mtp/runtime/__init__.py`

**수정 내용**:
```python
from weighted_mtp.runtime.ddp import (
    wrap_model_ddp,
    unwrap_model as unwrap_model_ddp,
    all_reduce_scalar,
)
from weighted_mtp.runtime.fsdp import (
    wrap_model_fsdp,
    unwrap_model as unwrap_model_fsdp,
    save_fsdp_checkpoint,
    load_fsdp_checkpoint,
)

# 기본 export (FSDP 우선)
__all__ = [
    # Distributed
    "init_distributed",
    "is_distributed",
    "is_main_process",
    "cleanup_distributed",
    # FSDP (권장)
    "wrap_model_fsdp",
    "save_fsdp_checkpoint",
    "load_fsdp_checkpoint",
    # DDP (legacy, 호환용)
    "wrap_model_ddp",
    # Common
    "unwrap_model",  # FSDP/DDP 공통 인터페이스
    "all_reduce_scalar",
]

# unwrap_model은 FSDP/DDP 모두 지원
def unwrap_model(model):
    """FSDP 또는 DDP wrapper 제거

    자동으로 FSDP/DDP 감지하여 unwrap.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.nn.parallel import DistributedDataParallel as DDP

    if isinstance(model, FSDP):
        return unwrap_model_fsdp(model)
    elif isinstance(model, DDP):
        return unwrap_model_ddp(model)
    return model
```

**검증 기준**:
- ✅ `from weighted_mtp.runtime import wrap_model_fsdp` 가능
- ✅ 기존 DDP import도 정상 동작 (하위 호환)

---

### Phase 3: Verifiable 파이프라인 FSDP 전환

**목표**: 첫 번째 full finetuning 파이프라인을 FSDP로 전환

**파일 수정**: `src/weighted_mtp/pipelines/run_verifiable.py`

**수정 내용**:
```python
# 기존 DDP import 삭제
from weighted_mtp.runtime import (
    # wrap_model_ddp,  # 삭제
    wrap_model_fsdp,  # 추가
    unwrap_model,
    all_reduce_scalar,
)

# Line 326 수정
# 기존:
# adapter = wrap_model_ddp(adapter, device)

# 변경:
adapter = wrap_model_fsdp(
    adapter,
    device,
    sharding_strategy="FULL_SHARD",
    mixed_precision=True,
    cpu_offload=False,
    activation_checkpointing=False,  # Phase 4에서 활성화
)
```

**Checkpoint 저장 수정**:
```python
# Line ~450 (checkpoint 저장 부분)
# 기존:
if is_main_process():
    unwrapped = unwrap_model(adapter)
    torch.save({
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        ...
    }, checkpoint_path)

# 변경:
if is_main_process():
    from weighted_mtp.runtime.fsdp import save_fsdp_checkpoint
    save_fsdp_checkpoint(adapter, checkpoint_path, optimizer)
```

**Config 추가**: `configs/verifiable/verifiable.yaml`
```yaml
# 분산 학습 설정
distributed:
  strategy: fsdp  # "ddp" 또는 "fsdp"
  fsdp:
    sharding_strategy: FULL_SHARD
    mixed_precision: true
    cpu_offload: false
    activation_checkpointing: false
```

**검증 기준**:
- ✅ Integration test 통과: `test_pipeline_verifiable.py`
- ✅ 1.5 epoch 학습 정상 완료
- ✅ Checkpoint 저장/로드 정상
- ✅ MLflow 로깅 정상

---

### Phase 4: Baseline 및 Rho-1 파이프라인 FSDP 전환

**목표**: 나머지 full finetuning 파이프라인들도 FSDP 전환

**파일 수정**:
- `src/weighted_mtp/pipelines/run_baseline.py`
- `src/weighted_mtp/pipelines/run_rho1.py`

**수정 내용**: Phase 3과 동일한 패턴 적용

**Rho-1 특이사항**: Reference model은 frozen이므로 FSDP wrapping 불필요
```python
# Reference model은 DDP로 유지 가능 (frozen이므로 메모리 부담 적음)
ref_model = load_reference_model(...)
for param in ref_model.parameters():
    param.requires_grad = False

# Policy model만 FSDP
adapter = wrap_model_fsdp(adapter, device, ...)
```

**검증 기준**:
- ✅ test_pipeline_baseline.py 통과
- ✅ test_pipeline_rho1.py 통과
- ✅ 3개 파이프라인 모두 FSDP로 전환 완료

---

### Phase 5: Gradient Checkpointing 추가 (선택)

**목표**: Activation 메모리 50% 절감으로 batch size 증가 가능

**파일 수정**: `src/weighted_mtp/models/meta_mtp/transformer.py`

**구현 내용**:
```python
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.use_gradient_checkpointing = False  # 기본 False
        ...

    def _forward(self, x, freqs_cis, mask):
        """실제 forward 로직"""
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def forward(self, x, freqs_cis, mask):
        """Gradient checkpointing wrapper"""
        if self.training and self.use_gradient_checkpointing:
            return checkpoint(self._forward, x, freqs_cis, mask, use_reentrant=False)
        return self._forward(x, freqs_cis, mask)
```

**Config 추가**:
```yaml
# configs/verifiable/verifiable.yaml
distributed:
  fsdp:
    activation_checkpointing: true  # 활성화
```

**효과**:
- Activation 메모리: 10GB → 5GB
- Batch size 증가 가능: 4 → 8

**검증 기준**:
- ✅ Batch size=8로 학습 가능
- ✅ 학습 속도 5-10% 저하 (trade-off 허용)
- ✅ Loss 수렴 정상

---

### Phase 6: DDP 코드 정리 (선택)

**목표**: DDP 관련 레거시 코드 제거 (필수는 아님)

**수행 여부**: 사용자 결정
- **유지**: DDP도 계속 지원 (config로 전환 가능)
- **삭제**: FSDP만 사용, DDP 완전 제거

**삭제 시**:
1. `src/weighted_mtp/runtime/ddp.py` 삭제
2. `tests/unit/test_ddp.py` 삭제
3. `runtime/__init__.py`에서 DDP import 제거

**유지 시**:
- Config에서 `strategy: ddp` 또는 `strategy: fsdp` 선택 가능
- Critic 파이프라인은 DDP 유지 (메모리 충분)

---

## III. 테스트 계획

### Unit Tests

**test_fsdp.py** (Phase 1)
```python
def test_wrap_model_fsdp_single_device():
    """단일 장치 환경에서는 FSDP wrapping 하지 않음"""

def test_unwrap_model_fsdp():
    """FSDP wrapper 제거 확인"""

def test_save_load_fsdp_checkpoint():
    """FSDP checkpoint 저장/로드 확인"""
```

### Integration Tests

**test_pipeline_verifiable.py** (Phase 3)
- 기존 테스트 그대로 실행
- n_epochs=1.5로 FSDP 학습 검증

**test_pipeline_baseline.py** (Phase 4)
**test_pipeline_rho1.py** (Phase 4)
- 동일하게 FSDP 적용 후 테스트

### GPU 메모리 모니터링

**메모리 측정 스크립트**:
```bash
# VESSL에서 실행
nvidia-smi --query-gpu=memory.used --format=csv -l 1 > memory_log.csv
```

**검증 목표**:
- Verifiable (bs=4): ~30GB per GPU
- Verifiable (bs=8, checkpointing): ~45GB per GPU
- Baseline (bs=4): ~30GB per GPU
- Rho-1 (bs=4): ~35GB per GPU (ref model 포함)

---

## IV. Rollback 계획

각 Phase는 독립적이므로 이전 Phase로 되돌리기 쉬움.

### Phase 1-2 실패 시
→ FSDP 코드 삭제, DDP 유지 (영향 없음)

### Phase 3 실패 시
→ `run_verifiable.py`만 DDP로 복원

### Phase 4 실패 시
→ 개별 파이프라인별 DDP 복원

### Git Branch 전략
```bash
# 각 Phase마다 branch 생성
git checkout -b fsdp/phase1-wrapper
git checkout -b fsdp/phase2-runtime
git checkout -b fsdp/phase3-verifiable
git checkout -b fsdp/phase4-all-pipelines
git checkout -b fsdp/phase5-checkpointing
```

---

## V. 성공 기준

### Phase별 성공 기준

| Phase | 성공 기준 |
|-------|----------|
| Phase 1 | test_fsdp.py 모든 테스트 통과 |
| Phase 2 | Import 오류 없음 |
| Phase 3 | test_pipeline_verifiable.py 통과, 메모리 ~30GB |
| Phase 4 | 모든 pipeline integration test 통과 |
| Phase 5 | batch_size=8 학습 가능 |

### 최종 성공 기준

1. ✅ **메모리**: A100 80GB에서 batch_size=4 정상 작동 (목표: ~30GB)
2. ✅ **학습**: Loss 수렴 정상 (DDP 대비 결과 동일)
3. ✅ **테스트**: 모든 integration test 통과
4. ✅ **속도**: 학습 시간 10% 이내 증가 허용
5. ✅ **안정성**: OOM 없이 3 epoch 완주 가능

---

## VI. 예상 일정

| Phase | 작업 시간 | 누적 시간 |
|-------|----------|----------|
| Phase 1 | 2-3 시간 | 2-3 시간 |
| Phase 2 | 30분 | 3-4 시간 |
| Phase 3 | 1-2 시간 | 5-6 시간 |
| Phase 4 | 1 시간 | 6-7 시간 |
| Phase 5 | 1 시간 | 7-8 시간 |
| **Total** | **7-8 시간** | |

**검증 시간**: 각 Phase마다 +1 시간 (테스트 및 디버깅)

---

## VII. 리스크 및 대응

### 리스크 1: FSDP API 변경
**확률**: 낮음
**영향**: 중간
**대응**: PyTorch 2.5.1 문서 참고, 예제 코드 확인

### 리스크 2: Checkpoint 호환성
**확률**: 중간
**영향**: 높음
**대응**: Phase 3에서 저장/로드 철저히 테스트

### 리스크 3: Gradient Accumulation 동작 차이
**확률**: 낮음
**영향**: 중간
**대응**: `no_sync()` context manager 사용

### 리스크 4: 학습 속도 저하
**확률**: 높음 (예상됨)
**영향**: 낮음 (허용 가능)
**대응**: All-gather/reduce-scatter overhead는 NVLink로 최소화

---

## VIII. 참고 자료

### PyTorch FSDP 공식 문서
- https://pytorch.org/docs/stable/fsdp.html
- https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

### Meta 논문
- "Better & Faster Large Language Models via Multi-token Prediction" (2024)
- Multi-token prediction with FSDP 사용 언급

### 외부 벤치마크
- Hugging Face Transformers FSDP 가이드
- PyTorch Lightning FSDP 통합

---

## IX. 결론

### 왜 FSDP 전환이 필요한가?

1. **메모리 한계**: DDP는 A100 80GB로 batch_size=4 불가능 (80-90GB 필요)
2. **확장성**: FSDP는 75% 메모리 절감 (~20GB)으로 batch_size=8까지 가능
3. **미래 대비**: 더 큰 모델(13B, 70B)로 확장 시 필수

### 왜 이 계획이 안전한가?

1. **점진적 전환**: Phase별로 독립 실행, rollback 가능
2. **기존 구조 존중**: DDP 인터페이스와 동일하게 설계
3. **테스트 보장**: 각 Phase마다 unit/integration test
4. **이미 준비됨**: `setup_fsdp_config()` 함수가 이미 존재

### 즉시 시작 가능한 이유

1. ✅ Pure PyTorch 구현 (FSDP 호환)
2. ✅ Safetensors 지원
3. ✅ FSDP config 함수 준비됨
4. ✅ 테스트 인프라 완비

**다음 단계**: Phase 1 시작 승인 요청
