"""Checkpoint 저장 및 로드 유틸리티

Stage 분리 파이프라인을 위한 checkpoint handoff 지원
MLflow artifact URI 및 local path 모두 지원
"""

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    adapter,
    optimizer: torch.optim.Optimizer,
    epoch: int | float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    checkpoint_path: Path | str,
    config: dict | None = None,
    s3_upload: bool = False,
    experiment_name: str | None = None,
    td_ema_state: dict | None = None,
) -> None:
    """Checkpoint 저장 (FSDP 지원, S3 업로드 옵션)

    FSDP 환경에서는 모든 rank가 state_dict gathering에 참여해야 하며,
    실제 파일 저장 및 S3 업로드는 rank 0만 수행합니다.

    Args:
        adapter: MetaLlamaMTPAdapter (FSDP-wrapped 또는 일반 모델)
        optimizer: torch.optim.Optimizer
        epoch: 현재 epoch (fractional epoch 지원)
        train_metrics: Training metrics
        val_metrics: Validation metrics
        checkpoint_path: 저장 경로
        config: 학습 설정 정보 (모델 경로 등, 평가 시 필요)
        s3_upload: S3 업로드 여부
        experiment_name: 실험 이름 (S3 경로 구성에 사용)
        td_ema_state: TD EMA 통계 state dict (TDStatsEMA.state_dict())

    Saved checkpoint format:
        {
            "epoch": float,
            "adapter_state_dict": dict,
            "optimizer_state_dict": dict,
            "train_metrics": dict,
            "val_metrics": dict,
            "config": dict,
            "td_ema_state": dict,  # TD EMA 통계 (선택적)
        }
    """
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        StateDictType,
        FullStateDictConfig,
    )

    checkpoint_path = Path(checkpoint_path)

    # FSDP Full state dict gathering (모든 rank가 참여해야 함)
    if isinstance(adapter, FSDP):
        with FSDP.state_dict_type(
            adapter,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            adapter_state_dict = adapter.state_dict()

        # rank 0만 실제 저장 수행
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    else:
        # 일반 모델 (single-device 환경)
        adapter_state_dict = adapter.state_dict()

    # 이하 저장 로직 (rank 0만 실행)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "adapter_state_dict": adapter_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": config,
        "td_ema_state": td_ema_state,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint 저장 완료: {checkpoint_path}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Val loss: {val_metrics.get('val_loss', 'N/A')}")

    # S3 업로드 (비동기, boto3 직접 사용)
    if s3_upload and experiment_name:
        from weighted_mtp.utils.s3_utils import s3_upload_executor, upload_to_s3_async
        s3_upload_executor.submit(upload_to_s3_async, checkpoint_path, experiment_name)
        logger.info(f"S3 업로드 예약: {checkpoint_path.name}")
    elif s3_upload and not experiment_name:
        logger.warning(f"S3 업로드 건너뜀 (experiment_name 없음): {checkpoint_path.name}")


def load_checkpoint_for_evaluation(
    checkpoint_path: Path,
    device: torch.device,
    inference_only: bool = True,
    dtype: torch.dtype | None = None,
):
    """평가용 checkpoint 로드 (full 또는 LoRA checkpoint 자동 감지)

    checkpoint_type에 따라 적절한 방식으로 모델을 로드합니다.
    - "full": 전체 adapter_state_dict 로드
    - "lora": base model 로드 후 LoRA + extra_heads 적용

    Args:
        checkpoint_path: Checkpoint 파일 경로
        device: torch.device
        inference_only: True면 extra_heads, value_head 로드 스킵 (메모리 절약 ~1.2GB)
                       평가/추론 시에는 head 0만 사용하므로 True 권장
        dtype: 모델 dtype 지정 (None이면 원본 유지, MPS는 bfloat16 미지원으로 float16 권장)

    Returns:
        (model, checkpoint_metadata)
        - model: MetaLlamaMTPAdapter (eval 모드)
        - checkpoint_metadata: {
            "epoch": float,
            "config": dict,  # 학습 설정 (모델 경로 등)
            "val_metrics": dict,
          }

    Raises:
        FileNotFoundError: Checkpoint 파일이 존재하지 않음
        KeyError: checkpoint에 필수 키가 없음

    Examples:
        >>> # 기본 사용 (inference_only=True, 메모리 최적화)
        >>> model, metadata = load_checkpoint_for_evaluation(
        ...     checkpoint_path=Path("storage/checkpoints/baseline/checkpoint_best.pt"),
        ...     device=torch.device("cpu"),
        ... )
        >>> # MPS (Apple Silicon) 사용 시
        >>> model, metadata = load_checkpoint_for_evaluation(
        ...     checkpoint_path=Path("checkpoint.pt"),
        ...     device=torch.device("mps"),
        ...     dtype=torch.float16,  # MPS는 bfloat16 미지원
        ... )
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 파일이 존재하지 않습니다: {checkpoint_path}")

    # Checkpoint 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f"Checkpoint 로드 완료: {checkpoint_path}")

    # checkpoint_type 자동 감지 (기본값: "full")
    checkpoint_type = checkpoint.get("checkpoint_type", "full")
    logger.info(f"Checkpoint 타입: {checkpoint_type}")

    # Config 정보 추출
    config_info = checkpoint.get("config", {})
    if not config_info:
        config_info = {"model": {"path": "storage/models/meta-llama-mtp"}}
        logger.warning(
            f"Checkpoint에 config 정보가 없습니다. 기본값 사용: {config_info['model']['path']}"
        )

    from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter

    if checkpoint_type == "lora":
        # LoRA checkpoint: base model 로드 후 학습된 파라미터 적용
        lora_config = checkpoint.get("lora_config")

        model = MetaLlamaMTPAdapter.from_pretrained(
            model_path=config_info["model"]["path"],
            device=device,
            use_lora=True,
            lora_config=lora_config,
        )

        # LoRA 파라미터 적용 (필수)
        current_state_dict = model.state_dict()

        for key, value in checkpoint.get("lora_state_dict", {}).items():
            if key in current_state_dict:
                current_state_dict[key] = value

        # extra_heads, value_head 적용 (inference_only=False일 때만)
        skipped_params = 0
        if not inference_only:
            for key, value in checkpoint.get("extra_heads_state_dict", {}).items():
                if key in current_state_dict:
                    current_state_dict[key] = value

            for key, value in checkpoint.get("value_head_state_dict", {}).items():
                if key in current_state_dict:
                    current_state_dict[key] = value
        else:
            # 스킵된 파라미터 수 계산 (로깅용)
            extra_heads_params = sum(
                v.numel() for v in checkpoint.get("extra_heads_state_dict", {}).values()
            )
            value_head_params = sum(
                v.numel() for v in checkpoint.get("value_head_state_dict", {}).values()
            )
            skipped_params = extra_heads_params + value_head_params
            if skipped_params > 0:
                skipped_mb = skipped_params * 2 / 1024 / 1024  # bfloat16 기준
                logger.info(f"inference_only=True: extra_heads/value_head 스킵 ({skipped_params:,} params, ~{skipped_mb:.0f}MB 절약)")

        model.load_state_dict(current_state_dict)
        logger.info("LoRA checkpoint 적용 완료")

    else:
        # Full checkpoint: 전체 adapter_state_dict 로드
        required_keys = ["adapter_state_dict", "epoch", "val_metrics"]
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise KeyError(
                f"Checkpoint에 필수 키가 없습니다: {missing_keys}\n"
                f"사용 가능한 키: {list(checkpoint.keys())}"
            )

        model = MetaLlamaMTPAdapter.from_pretrained(
            model_path=config_info["model"]["path"],
            device=device,
        )
        model.load_state_dict(checkpoint["adapter_state_dict"])
        logger.info("Full checkpoint 적용 완료")

    model.eval()

    # dtype 변환 (MPS는 bfloat16 미지원, float16 사용 권장)
    if dtype is not None:
        model = model.to(dtype=dtype)
        logger.info(f"모델 dtype 변환: {dtype}")

    # Metadata 구성
    checkpoint_metadata = {
        "epoch": checkpoint.get("epoch"),
        "config": config_info,
        "val_metrics": checkpoint.get("val_metrics", {}),
    }

    logger.info("평가용 모델 로드 성공")
    logger.info(f"  Epoch: {checkpoint_metadata['epoch']}")
    logger.info(f"  Val loss: {checkpoint_metadata['val_metrics'].get('val_loss', 'N/A')}")

    return model, checkpoint_metadata


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    save_total_limit: int,
) -> None:
    """오래된 중간 checkpoint 삭제

    checkpoint_best.pt와 checkpoint_final.pt는 절대 삭제하지 않음
    checkpoint_epoch_*.pt만 save_total_limit 개수만큼 유지

    Args:
        checkpoint_dir: Checkpoint 디렉터리
        save_total_limit: 유지할 최대 개수
    """
    if not checkpoint_dir.exists():
        return

    # 중간 checkpoint 파일만 수집 (checkpoint_epoch_*.pt)
    epoch_checkpoints = sorted(
        [f for f in checkpoint_dir.glob("checkpoint_epoch_*.pt")],
        key=lambda x: x.stat().st_mtime,
    )

    # 삭제할 파일 개수 계산
    n_to_delete = len(epoch_checkpoints) - save_total_limit

    if n_to_delete > 0:
        for checkpoint_path in epoch_checkpoints[:n_to_delete]:
            logger.info(f"오래된 checkpoint 삭제: {checkpoint_path.name}")
            checkpoint_path.unlink()


def save_lora_checkpoint(
    adapter,
    optimizer: torch.optim.Optimizer,
    epoch: int | float,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    checkpoint_path: Path | str,
    config: dict | None = None,
    s3_upload: bool = False,
    experiment_name: str | None = None,
    save_value_head: bool = True,
    td_ema_state: dict | None = None,
) -> None:
    """LoRA checkpoint 저장

    학습 가능한 파라미터만 저장하여 checkpoint 크기를 대폭 줄임.
    - LoRA 파라미터 (trunk layers + extra_heads 모두 포함)
    - value_head 파라미터 (선택적)

    FSDP 환경에서 메모리 효율적 저장:
    - summon_full_params()를 사용해 학습 가능 파라미터만 gather
    - 전체 모델을 rank 0에 모으지 않아 OOM 방지

    Args:
        adapter: MetaLlamaMTPAdapter (FSDP-wrapped 또는 일반 모델, LoRA 적용됨)
        optimizer: torch.optim.Optimizer
        epoch: 현재 epoch (fractional epoch 지원)
        train_metrics: Training metrics
        val_metrics: Validation metrics
        checkpoint_path: 저장 경로
        config: 학습 설정 정보 (base model 경로 등)
        s3_upload: S3 업로드 여부
        experiment_name: 실험 이름 (S3 경로 구성에 사용)
        save_value_head: Value head도 함께 저장할지 여부
        td_ema_state: TD EMA 통계 state dict (TDStatsEMA.state_dict())

    Saved checkpoint format:
        {
            "epoch": float,
            "lora_state_dict": dict,  # LoRA 파라미터 (trunk + extra_heads)
            "extra_heads_state_dict": dict,  # 빈 dict (하위 호환성)
            "value_head_state_dict": dict,  # Value head (선택적)
            "optimizer_state_dict": dict,
            "train_metrics": dict,
            "val_metrics": dict,
            "config": dict,
            "lora_config": dict,  # LoRA 설정 (rank, alpha 등)
            "td_ema_state": dict,  # TD EMA 통계 (선택적)
            "checkpoint_type": "lora",
        }
    """
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    checkpoint_path = Path(checkpoint_path)
    is_fsdp = isinstance(adapter, FSDP)

    if is_fsdp:
        # 메모리 효율적 저장: 학습 가능 파라미터만 추출
        lora_state_dict, extra_heads_state_dict, value_head_state_dict, lora_config = \
            _extract_trainable_params_fsdp(adapter, save_value_head)

        # rank 0만 실제 저장 수행
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    else:
        # 일반 모델 (single-device 환경)
        full_state_dict = adapter.state_dict()
        unwrapped = adapter

        # LoRA 파라미터 추출 (trunk + extra_heads 모두 포함)
        lora_state_dict = {
            k: v for k, v in full_state_dict.items()
            if "lora_A" in k or "lora_B" in k
        }

        # extra_heads는 LoRA가 적용되어 있으므로 별도 저장 불필요
        # 하위 호환성을 위해 빈 dict 유지
        extra_heads_state_dict = {}

        # Value head 파라미터 추출
        value_head_state_dict = {}
        if save_value_head:
            value_head_state_dict = {
                k: v for k, v in full_state_dict.items()
                if "value_head" in k
            }

        # LoRA config 추출
        lora_config = None
        if hasattr(unwrapped, "lora_enabled") and unwrapped.lora_enabled:
            for _, module in unwrapped.transformer.named_modules():
                if hasattr(module, "rank") and hasattr(module, "alpha"):
                    lora_config = {"rank": module.rank, "alpha": module.alpha}
                    break

    # 크기 로깅
    lora_size = sum(v.numel() for v in lora_state_dict.values())
    extra_heads_size = sum(v.numel() for v in extra_heads_state_dict.values())
    vh_size = sum(v.numel() for v in value_head_state_dict.values())
    saved_size = lora_size + extra_heads_size + vh_size

    logger.info(f"LoRA checkpoint 저장:")
    logger.info(f"  LoRA: {lora_size:,} params")
    logger.info(f"  extra_heads: {extra_heads_size:,} params")
    if save_value_head and vh_size > 0:
        logger.info(f"  Value head: {vh_size:,} params")
    logger.info(f"  총 저장 크기: {saved_size:,} params")

    # 저장
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "lora_state_dict": lora_state_dict,
        "extra_heads_state_dict": extra_heads_state_dict,
        "value_head_state_dict": value_head_state_dict if save_value_head else {},
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "config": config,
        "lora_config": lora_config,
        "checkpoint_type": "lora",
        "td_ema_state": td_ema_state,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"LoRA checkpoint 저장 완료: {checkpoint_path}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Val loss: {val_metrics.get('val_loss', 'N/A')}")

    # S3 업로드 (비동기, boto3 직접 사용)
    if s3_upload and experiment_name:
        from weighted_mtp.utils.s3_utils import s3_upload_executor, upload_to_s3_async
        s3_upload_executor.submit(upload_to_s3_async, checkpoint_path, experiment_name)
        logger.info(f"S3 업로드 예약: {checkpoint_path.name}")
    elif s3_upload and not experiment_name:
        logger.warning(f"S3 업로드 건너뜀 (experiment_name 없음): {checkpoint_path.name}")


def _extract_trainable_params_fsdp(
    adapter,
    save_value_head: bool = True,
) -> tuple[dict, dict, dict, dict | None]:
    """FSDP 환경에서 학습 가능 파라미터만 메모리 효율적으로 추출

    FSDP summon_full_params()를 사용해 trainable=True인 파라미터만 gather.
    전체 모델을 rank 0에 모으지 않아 OOM 방지.

    Args:
        adapter: FSDP-wrapped MetaLlamaMTPAdapter
        save_value_head: Value head 포함 여부

    Returns:
        (lora_state_dict, extra_heads_state_dict, value_head_state_dict, lora_config)
        extra_heads_state_dict는 빈 dict (LoRA가 적용되어 lora_state_dict에 포함됨)
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    unwrapped = adapter.module
    lora_state_dict = {}
    extra_heads_state_dict = {}  # 빈 dict (하위 호환성)
    value_head_state_dict = {}
    lora_config = None

    # summon_full_params: 학습 가능 파라미터만 gather (writeback=False로 읽기 전용)
    with FSDP.summon_full_params(adapter, writeback=False, offload_to_cpu=True, rank0_only=True):
        # 학습 가능 파라미터 순회하며 추출
        for name, param in adapter.named_parameters():
            if not param.requires_grad:
                continue

            # CPU로 복사 (rank 0에서만 유효)
            param_cpu = param.detach().cpu().clone()

            # LoRA 파라미터 (trunk + extra_heads 모두 포함)
            if "lora_A" in name or "lora_B" in name:
                lora_state_dict[name] = param_cpu
            # Value head 파라미터
            elif save_value_head and "value_head" in name:
                value_head_state_dict[name] = param_cpu

        # LoRA config 추출
        if hasattr(unwrapped, "lora_enabled") and unwrapped.lora_enabled:
            for _, module in unwrapped.transformer.named_modules():
                if hasattr(module, "rank") and hasattr(module, "alpha"):
                    lora_config = {"rank": module.rank, "alpha": module.alpha}
                    break

    return lora_state_dict, extra_heads_state_dict, value_head_state_dict, lora_config


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
) -> None:
    """Value Model checkpoint 저장 (FSDP 지원, 메모리 효율적)

    ValueModel (backbone + value_head 구조)의 checkpoint를 저장합니다.
    LoRA 모드에서는 LoRA weights + value head만 저장하고,
    Full 모드에서는 전체 backbone + value head를 저장합니다.

    FSDP 환경에서 메모리 효율적 저장:
    - LoRA 모드: summon_full_params()로 학습 가능 파라미터만 gather (OOM 방지)
    - Full 모드: FULL_STATE_DICT 사용 (전체 모델 저장 필요)

    Args:
        value_model: ValueModel 인스턴스 (FSDP-wrapped 가능)
        optimizer: torch.optim.Optimizer
        epoch: 현재 epoch
        train_metrics: Training metrics
        val_metrics: Validation metrics
        checkpoint_path: 저장 경로
        config: 학습 설정 (OmegaConf DictConfig 또는 dict)
        s3_upload: S3 업로드 여부
        experiment_name: 실험 이름 (S3 경로 구성에 사용)

    Saved checkpoint format (LoRA mode):
        {
            "checkpoint_type": "hf_lora",
            "lora_state_dict": dict,
            "value_head_state_dict": dict,
            "optimizer_state_dict": dict,  # Resume 학습 지원
            "lora_config": dict,
            "base_model_path": str,
            "epoch": float,
            "train_metrics": dict,
            "val_metrics": dict,
            "config": dict,
        }

    Saved checkpoint format (Full mode):
        {
            "checkpoint_type": "full",
            "backbone_state_dict": dict,
            "value_head_state_dict": dict,
            "optimizer_state_dict": dict,
            "epoch": float,
            "train_metrics": dict,
            "val_metrics": dict,
            "config": dict,
        }
    """
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        StateDictType,
        FullStateDictConfig,
    )

    checkpoint_path = Path(checkpoint_path)
    is_fsdp = isinstance(value_model, FSDP)

    # LoRA 모드 확인
    if is_fsdp:
        unwrapped = value_model.module
    else:
        unwrapped = value_model
    use_lora = getattr(unwrapped, "lora_enabled", False)
    lora_config = getattr(unwrapped, "lora_config", None)

    if is_fsdp:
        if use_lora:
            # LoRA 모드: 메모리 효율적 저장 (summon_full_params 사용)
            lora_state_dict, value_head_state_dict = \
                _extract_value_model_trainable_params_fsdp(value_model)

            # rank 0만 실제 저장 수행
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        else:
            # Full 모드: 전체 모델 저장 필요
            with FSDP.state_dict_type(
                value_model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                full_state_dict = value_model.state_dict()

            if dist.is_initialized() and dist.get_rank() != 0:
                return

            # backbone 파라미터 추출 (prefix 제거)
            backbone_state_dict = {
                k.replace("backbone.", ""): v
                for k, v in full_state_dict.items()
                if "backbone" in k and "lora" not in k
            }
            # value_head 파라미터 추출 (prefix 제거)
            value_head_state_dict = {
                k.replace("value_head.", ""): v
                for k, v in full_state_dict.items()
                if "value_head" in k
            }
            lora_state_dict = {}
    else:
        # Non-FSDP: 단일 디바이스 환경
        if use_lora:
            from weighted_mtp.models.lora import get_hf_lora_state_dict
            lora_state_dict = get_hf_lora_state_dict(unwrapped.backbone)
        else:
            lora_state_dict = {}
            backbone_state_dict = unwrapped.backbone.state_dict()

        value_head_state_dict = unwrapped.value_head.state_dict()

    # Config 변환 (OmegaConf → dict)
    config_dict = None
    if config is not None:
        try:
            from omegaconf import OmegaConf
            if hasattr(config, "_metadata"):  # OmegaConf DictConfig
                config_dict = OmegaConf.to_container(config, resolve=True)
            else:
                config_dict = dict(config) if not isinstance(config, dict) else config
        except ImportError:
            config_dict = dict(config) if not isinstance(config, dict) else config

    # base_model_path 추출
    base_model_path = None
    if config is not None:
        if hasattr(config, "models") and hasattr(config.models, "value_model"):
            base_model_path = config.models.value_model.path
        elif isinstance(config, dict) and "models" in config:
            base_model_path = config.get("models", {}).get("value_model", {}).get("path")

    # Checkpoint 구성
    if use_lora:
        checkpoint = {
            "checkpoint_type": "hf_lora",
            "lora_state_dict": lora_state_dict,
            "value_head_state_dict": value_head_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "lora_config": lora_config,
            "base_model_path": base_model_path,
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": config_dict,
        }
    else:
        checkpoint = {
            "checkpoint_type": "full",
            "backbone_state_dict": backbone_state_dict,
            "value_head_state_dict": value_head_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": config_dict,
        }

    # 저장
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Value Model checkpoint 저장 완료: {checkpoint_path}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Val loss: {val_metrics.get('val_loss', 'N/A')}")
    logger.info(f"  Mode: {'LoRA' if use_lora else 'Full'}")

    # S3 업로드 (비동기, boto3 직접 사용)
    if s3_upload and experiment_name:
        from weighted_mtp.utils.s3_utils import s3_upload_executor, upload_to_s3_async
        s3_upload_executor.submit(upload_to_s3_async, checkpoint_path, experiment_name)
        logger.info(f"S3 업로드 예약: {checkpoint_path.name}")
    elif s3_upload and not experiment_name:
        logger.warning(f"S3 업로드 건너뜀 (experiment_name 없음): {checkpoint_path.name}")


def _extract_value_model_trainable_params_fsdp(
    value_model,
) -> tuple[dict, dict]:
    """FSDP 환경에서 ValueModel의 학습 가능 파라미터만 메모리 효율적으로 추출

    summon_full_params()를 사용해 trainable=True인 파라미터만 gather.
    전체 2.7B 모델을 rank 0에 모으지 않아 OOM 방지.

    Args:
        value_model: FSDP-wrapped ValueModel

    Returns:
        (lora_state_dict, value_head_state_dict)
        - lora_state_dict: backbone의 LoRA 파라미터 (backbone. prefix 포함)
        - value_head_state_dict: value head 파라미터 (prefix 제거됨)
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    lora_state_dict = {}
    value_head_state_dict = {}

    with FSDP.summon_full_params(value_model, writeback=False, offload_to_cpu=True, rank0_only=True):
        for name, param in value_model.named_parameters():
            if not param.requires_grad:
                continue

            param_cpu = param.detach().cpu().clone()

            # LoRA 파라미터 (backbone. prefix 유지)
            if "lora_A" in name or "lora_B" in name:
                lora_state_dict[name] = param_cpu
            # Value head 파라미터 (prefix 제거)
            elif "value_head" in name:
                key = name.replace("value_head.", "")
                value_head_state_dict[key] = param_cpu

    return lora_state_dict, value_head_state_dict


def load_lora_checkpoint(
    adapter,
    checkpoint_path: Path | str,
    device: torch.device,
    load_value_head: bool = True,
    strict: bool = True,
) -> dict:
    """LoRA checkpoint 로드 (LoRA + extra_heads + value_head를 기존 adapter에 적용)

    Base model이 이미 로드된 adapter에 학습된 파라미터를 로드합니다.
    - LoRA 파라미터 (trunk layers)
    - extra_heads 파라미터 (MTP heads)
    - value_head 파라미터 (선택적)

    Args:
        adapter: MetaLlamaMTPAdapter (LoRA가 적용된 상태)
        checkpoint_path: LoRA checkpoint 경로
        device: torch.device
        load_value_head: Value head도 함께 로드할지 여부
        strict: 누락된 키가 있으면 에러 발생 여부

    Returns:
        checkpoint metadata (epoch, config, val_metrics 등)

    Raises:
        FileNotFoundError: Checkpoint 파일이 존재하지 않음
        ValueError: Checkpoint이 LoRA 타입이 아님
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint 파일이 존재하지 않습니다: {checkpoint_path}")

    # Checkpoint 로드
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f"LoRA checkpoint 로드: {checkpoint_path}")

    # LoRA checkpoint 타입 확인
    checkpoint_type = checkpoint.get("checkpoint_type", "full")
    if checkpoint_type != "lora":
        raise ValueError(
            f"LoRA checkpoint가 아닙니다. checkpoint_type={checkpoint_type}\n"
            f"일반 checkpoint는 load_state_dict()를 사용하세요."
        )

    # LoRA state dict 로드
    lora_state_dict = checkpoint.get("lora_state_dict", {})
    if not lora_state_dict:
        logger.warning("LoRA state_dict가 비어있습니다.")

    # extra_heads state dict 로드
    extra_heads_state_dict = checkpoint.get("extra_heads_state_dict", {})

    # Value head state dict 로드
    value_head_state_dict = checkpoint.get("value_head_state_dict", {})

    # 현재 adapter의 state_dict 가져오기
    current_state_dict = adapter.state_dict()

    # LoRA 가중치 업데이트
    for key, value in lora_state_dict.items():
        if key in current_state_dict:
            current_state_dict[key] = value
        elif strict:
            raise KeyError(f"LoRA key not found in model: {key}")
        else:
            logger.warning(f"LoRA key not found, skipping: {key}")

    # extra_heads 가중치 업데이트
    for key, value in extra_heads_state_dict.items():
        if key in current_state_dict:
            current_state_dict[key] = value
        elif strict:
            raise KeyError(f"extra_heads key not found in model: {key}")
        else:
            logger.warning(f"extra_heads key not found, skipping: {key}")

    # Value head 가중치 업데이트
    if load_value_head and value_head_state_dict:
        for key, value in value_head_state_dict.items():
            if key in current_state_dict:
                current_state_dict[key] = value
            elif strict:
                raise KeyError(f"Value head key not found in model: {key}")
            else:
                logger.warning(f"Value head key not found, skipping: {key}")

    # State dict 적용
    adapter.load_state_dict(current_state_dict)

    logger.info(f"LoRA checkpoint 로드 완료:")
    logger.info(f"  LoRA params: {len(lora_state_dict)} keys")
    logger.info(f"  extra_heads params: {len(extra_heads_state_dict)} keys")
    if load_value_head and value_head_state_dict:
        logger.info(f"  Value head params: {len(value_head_state_dict)} keys")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")

    # Metadata 반환
    return {
        "epoch": checkpoint.get("epoch"),
        "config": checkpoint.get("config"),
        "val_metrics": checkpoint.get("val_metrics", {}),
        "train_metrics": checkpoint.get("train_metrics", {}),
        "lora_config": checkpoint.get("lora_config"),
    }


def save_hf_checkpoint(
    model,
    tokenizer,
    save_dir: Path | str,
    epoch: float,
    val_metrics: dict[str, float],
) -> None:
    """HuggingFace 형식 checkpoint 저장

    AutoModelForCausalLM.from_pretrained()로 로드 가능한 형식으로 저장합니다.
    FSDP 환경에서는 모든 rank가 state_dict gathering에 참여하고,
    실제 저장은 rank 0만 수행합니다.

    Args:
        model: HuggingFace 모델 (FSDP-wrapped 또는 일반 모델)
        tokenizer: HuggingFace 토크나이저
        save_dir: 저장 디렉터리 경로
        epoch: 현재 epoch
        val_metrics: Validation metrics

    저장되는 파일:
        save_dir/
        ├── config.json
        ├── model.safetensors (또는 pytorch_model.bin)
        ├── tokenizer.json
        ├── tokenizer_config.json
        └── training_state.json  # epoch, val_metrics 등
    """
    import json
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        StateDictType,
        FullStateDictConfig,
    )

    save_dir = Path(save_dir)

    # FSDP Full state dict gathering (모든 rank가 참여해야 함)
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()

        # rank 0만 실제 저장 수행
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        # FSDP unwrap된 원본 모델 구조 필요
        unwrapped_model = model.module
    else:
        # 일반 모델 (single-device 환경)
        state_dict = model.state_dict()
        unwrapped_model = model

    # 저장 디렉터리 생성
    save_dir.mkdir(parents=True, exist_ok=True)

    # HuggingFace 형식으로 저장 (FSDP 안전: state_dict를 직접 저장)
    # 모델 config 저장
    unwrapped_model.config.save_pretrained(save_dir)

    # state_dict를 safetensors로 저장
    from safetensors.torch import save_file
    save_file(state_dict, save_dir / "model.safetensors")

    # 토크나이저 저장
    tokenizer.save_pretrained(save_dir)

    # 학습 상태 저장 (별도 파일)
    training_state = {
        "epoch": epoch,
        "val_metrics": val_metrics,
    }
    with open(save_dir / "training_state.json", "w") as f:
        json.dump(training_state, f, indent=2)

    logger.info(f"HuggingFace checkpoint 저장 완료: {save_dir}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Val loss: {val_metrics.get('val_loss', 'N/A')}")


def save_hf_lora_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path | str,
    config: dict,
    epoch: float,
    val_metrics: dict[str, float],
    value_head_state_dict: dict | None = None,
) -> None:
    """HuggingFace 모델용 LoRA checkpoint 저장

    LlamaForCausalLM 또는 LlamaModel에 적용된 LoRA 파라미터를 저장합니다.
    ref_tuning, critic 등 HuggingFace 기반 파이프라인에서 사용.

    FSDP 환경에서 메모리 효율적 저장:
    - summon_full_params()를 사용해 학습 가능 파라미터만 gather
    - 전체 모델을 rank 0에 모으지 않아 OOM 방지

    Args:
        model: LoRA가 적용된 HuggingFace 모델 (FSDP-wrapped 가능)
        optimizer: torch.optim.Optimizer (Resume 학습용)
        checkpoint_path: 저장 경로 (.pt 파일)
        config: 학습 설정 (base_model_path, lora config 등)
        epoch: 현재 epoch
        val_metrics: Validation metrics
        value_head_state_dict: Value head state dict (critic용, 없으면 빈 dict)

    Saved checkpoint format:
        {
            "checkpoint_type": "hf_lora",
            "lora_state_dict": dict,
            "value_head_state_dict": dict,
            "optimizer_state_dict": dict,  # Resume 학습 지원
            "lora_config": dict,
            "base_model_path": str,
            "epoch": float,
            "val_metrics": dict,
            "config": dict,
        }
    """
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    checkpoint_path = Path(checkpoint_path)

    # FSDP 환경: 학습 가능 파라미터만 메모리 효율적으로 추출
    if isinstance(model, FSDP):
        lora_state_dict = _extract_hf_lora_params_fsdp(model)

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    else:
        full_state_dict = model.state_dict()
        # LoRA 파라미터 추출
        lora_state_dict = {
            k: v for k, v in full_state_dict.items()
            if "lora_A" in k or "lora_B" in k
        }

    # LoRA config 추출
    lora_config = None
    if hasattr(config, "training") and hasattr(config.training, "lora"):
        from omegaconf import OmegaConf
        lora_config = OmegaConf.to_container(config.training.lora, resolve=True)

    # base_model_path 추출
    base_model_path = None
    if hasattr(config, "models"):
        if hasattr(config.models, "policy"):
            base_model_path = config.models.policy.path
        elif hasattr(config.models, "value_model"):
            base_model_path = config.models.value_model.path

    # 저장
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    from omegaconf import OmegaConf
    checkpoint = {
        "checkpoint_type": "hf_lora",
        "lora_state_dict": lora_state_dict,
        "value_head_state_dict": value_head_state_dict or {},
        "optimizer_state_dict": optimizer.state_dict(),
        "lora_config": lora_config,
        "base_model_path": base_model_path,
        "epoch": epoch,
        "val_metrics": val_metrics,
        "config": OmegaConf.to_container(config, resolve=True) if hasattr(config, "_metadata") else config,
    }

    torch.save(checkpoint, checkpoint_path)

    # 크기 로깅
    lora_size = sum(v.numel() for v in lora_state_dict.values())
    vh_size = sum(v.numel() for v in (value_head_state_dict or {}).values())

    logger.info(f"HF LoRA checkpoint 저장 완료: {checkpoint_path}")
    logger.info(f"  Epoch: {epoch}")
    logger.info(f"  Val loss: {val_metrics.get('val_loss', 'N/A')}")
    logger.info(f"  LoRA: {lora_size:,} params")
    if vh_size > 0:
        logger.info(f"  Value head: {vh_size:,} params")


def _extract_hf_lora_params_fsdp(model) -> dict:
    """FSDP 환경에서 HuggingFace 모델의 LoRA 파라미터만 메모리 효율적으로 추출

    summon_full_params()를 사용해 trainable=True인 파라미터만 gather.
    전체 모델을 rank 0에 모으지 않아 OOM 방지.

    Args:
        model: FSDP-wrapped HuggingFace 모델 (LoRA 적용됨)

    Returns:
        lora_state_dict: LoRA 파라미터 (lora_A, lora_B)
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    lora_state_dict = {}

    with FSDP.summon_full_params(model, writeback=False, offload_to_cpu=True, rank0_only=True):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # LoRA 파라미터만 추출
            if "lora_A" in name or "lora_B" in name:
                param_cpu = param.detach().cpu().clone()
                lora_state_dict[name] = param_cpu

    return lora_state_dict
