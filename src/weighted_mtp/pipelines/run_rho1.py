"""Rho-1 WMTP Runner

독립 실행:
    python -m weighted_mtp.pipelines.run_rho1 --config configs/rho1/rho1.yaml
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoModelForCausalLM

from weighted_mtp.core.env import ensure_env_loaded
from weighted_mtp.core.logging import setup_logging
from weighted_mtp.data.dataloader import create_dataloader
from weighted_mtp.models.meta_mtp.adapter import MetaLlamaMTPAdapter
from weighted_mtp.models.tokenizer_utils import load_tokenizer_from_config
from weighted_mtp.utils import (
    GPUMonitor,
    ThroughputTracker,
    cleanup_old_checkpoints,
    cleanup_s3_checkpoints,
    compute_gradient_clip_stats,
    compute_mtp_ce_loss,
    compute_weight_statistics,
    create_scheduler,
    get_model_dtype,
    get_model_size,
    get_system_info,
    save_checkpoint,
    save_lora_checkpoint,
    shutdown_s3_executor,
    sync_mlruns_to_s3,
)
from weighted_mtp.runtime import (
    init_distributed,
    setup_environment,
    is_main_process,
    wrap_model_fsdp,
    unwrap_model,
    all_reduce_scalars,
)
from weighted_mtp.value_weighting.rho1_weighting import compute_mtp_selective_weights


def load_adapter(
    config: dict,
    device: torch.device,
    use_lora: bool = False,
    lora_config: dict | None = None,
) -> MetaLlamaMTPAdapter:
    """Adapter 로드

    Args:
        config: 모델 설정
        device: 디바이스
        use_lora: LoRA 사용 여부
        lora_config: LoRA 설정 (rank, alpha, dropout, target_modules)

    Returns:
        MetaLlamaMTPAdapter 인스턴스
    """
    adapter = MetaLlamaMTPAdapter.from_pretrained(
        model_path=config.models.policy.path,
        device=device,
        dtype=config.models.policy.dtype,
        use_lora=use_lora,
        lora_config=lora_config,
    )
    return adapter


def load_reference_model(config: dict, device: torch.device) -> nn.Module:
    """Reference model 로드 (HuggingFace LlamaForCausalLM)

    Rho-1에서 Reference 모델은 NTP loss 계산용으로만 사용되므로
    HuggingFace 모델을 직접 로드합니다.

    checkpoint_path 형식에 따른 로드 방식:
    - .pt 파일: LoRA checkpoint (base_model_path + LoRA merge)
    - 디렉토리: HuggingFace 모델 (그대로 로드)

    Args:
        config: 모델 설정
        device: 디바이스

    Returns:
        Reference model (eval mode, HuggingFace LlamaForCausalLM)
    """
    from pathlib import Path as PathlibPath
    from weighted_mtp.models.lora import (
        apply_lora_to_hf_model,
        load_hf_lora_state_dict,
        merge_lora_weights,
    )

    # dtype 변환
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map.get(config.models.reference.dtype, torch.bfloat16)

    # checkpoint_path 확인 (필수)
    checkpoint_path = getattr(config.models.reference, "checkpoint_path", None)
    if checkpoint_path is None:
        raise ValueError("config.models.reference.checkpoint_path가 필요합니다.")

    checkpoint_path_obj = PathlibPath(checkpoint_path)

    # Config에서 base_model_path 읽기 (명시적 지정)
    base_model_path_config = getattr(config.models.reference, "base_model_path", None)

    if str(checkpoint_path).endswith(".pt"):
        # LoRA checkpoint에서 로드
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if checkpoint.get("checkpoint_type") == "hf_lora":
            # Base model 경로: config 우선, 없으면 checkpoint에서 추출
            base_model_path = base_model_path_config or checkpoint.get("base_model_path")
            if base_model_path is None:
                raise ValueError(
                    "base_model_path가 필요합니다. "
                    "config에서 base_model_path를 지정하거나, checkpoint에 base_model_path가 포함되어야 합니다."
                )

            ref_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            ).to(device)

            # LoRA 적용
            lora_config = checkpoint.get("lora_config", {})
            apply_lora_to_hf_model(ref_model, lora_config)

            # LoRA weights 로드
            lora_state_dict = checkpoint.get("lora_state_dict", {})
            if lora_state_dict:
                load_hf_lora_state_dict(ref_model, lora_state_dict)

            # LoRA merge (inference 최적화)
            merge_lora_weights(ref_model)
        else:
            raise ValueError(
                f"지원하지 않는 checkpoint 타입입니다: {checkpoint.get('checkpoint_type')}. "
                "hf_lora 타입만 지원합니다."
            )
    else:
        # 디렉토리인 경우: HuggingFace 모델로 직접 로드
        ref_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)

    # Eval mode (gradient 불필요)
    ref_model.eval()

    # Gradient 계산 비활성화
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


def validate_rho1(
    adapter: MetaLlamaMTPAdapter,
    ref_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    k_percent: float,
    rho1_mode: str = "signed",
) -> dict[str, float]:
    """Validation 수행 (Rho-1)

    Args:
        adapter: Adapter (FSDP-wrapped 가능)
        ref_model: Reference model (HuggingFace LlamaForCausalLM, eval mode)
        dataloader: Validation DataLoader
        device: 디바이스
        k_percent: Top-k selection ratio (0~1)
        rho1_mode: "signed" (Policy > Ref만) or "absolute" (차이 큰 모든 토큰)

    Returns:
        Validation metrics (FSDP 환경에서는 all-reduce 적용됨)
    """
    adapter.eval()
    ref_model.eval()

    total_weighted_ce_loss = 0.0
    total_unweighted_ce_loss = 0.0
    total_head0_ce_loss = 0.0
    total_excess_loss = 0.0
    n_batches = 0

    # 모델 dtype 감지
    model_dtype = get_model_dtype(adapter)

    with torch.no_grad():
        for batch in dataloader:
            # 1. Batch를 device로 이동
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # 2. Loss mask 생성 (labels != -100)
            loss_mask = (labels != -100)

            # 3. Reference forward (HuggingFace 모델)
            ref_outputs = ref_model(input_ids)
            ref_logits = ref_outputs.logits  # [batch, seq, vocab]

            # 4. Policy forward (MTP만)
            policy_logits = adapter(input_ids)

            batch_size, seq_len, n_future, vocab_size = policy_logits.shape

            # 5. MTP selective weights (per-head binary selection)
            weights, selection_stats = compute_mtp_selective_weights(
                policy_logits=policy_logits,
                ref_logits=ref_logits,
                labels=labels,
                loss_mask=loss_mask,
                k_percent=k_percent,
                mode=rho1_mode,
            )

            # 6. Weighted CE loss (per-head) 및 Head0 CE loss, Unweighted CE loss
            batch_weighted_ce_loss = 0.0
            batch_unweighted_ce_loss = 0.0
            batch_head0_ce_loss = 0.0
            valid_heads = 0

            for k in range(1, n_future + 1):
                valid_len = seq_len - k

                if valid_len <= 0:
                    continue

                policy_logits_k = policy_logits[:, :valid_len, k - 1, :]
                labels_k = labels[:, k : k + valid_len]
                weights_k = weights[:, :valid_len, k - 1]  # Per-head weights
                loss_mask_k = loss_mask[:, k : k + valid_len]

                ce_loss_k = F.cross_entropy(
                    policy_logits_k.reshape(-1, vocab_size),
                    labels_k.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                )

                # loss_mask_k는 이미 labels != -100 조건 포함
                loss_mask_k_float = loss_mask_k.to(model_dtype)

                # Head0 (k=1): 순수 CE loss 평균 (weight 없이)
                if k == 1:
                    valid_count = loss_mask_k_float.sum()
                    if valid_count > 0:
                        batch_head0_ce_loss = (ce_loss_k * loss_mask_k_float.reshape(-1)).sum() / valid_count

                # Weighted CE (선택된 토큰만)
                weighted_ce_k = ce_loss_k * weights_k.reshape(-1) * loss_mask_k_float.reshape(-1)
                # Unweighted CE (모든 valid 토큰)
                unweighted_ce_k = ce_loss_k * loss_mask_k_float.reshape(-1)

                # 선택된 토큰 수로 나눠서 정확한 평균 계산
                selected_sum_k = (weights_k.reshape(-1) * loss_mask_k_float.reshape(-1)).sum()
                mask_sum_k = loss_mask_k_float.sum()

                if selected_sum_k > 0:
                    batch_weighted_ce_loss += weighted_ce_k.sum() / selected_sum_k
                    valid_heads += 1
                if mask_sum_k > 0:
                    batch_unweighted_ce_loss += unweighted_ce_k.sum() / mask_sum_k

                # 메모리 최적화: 중간 텐서 즉시 삭제
                del policy_logits_k, labels_k, weights_k, loss_mask_k
                del ce_loss_k, loss_mask_k_float
                del weighted_ce_k, unweighted_ce_k

            weighted_ce_loss = batch_weighted_ce_loss / max(valid_heads, 1)
            unweighted_ce_loss = batch_unweighted_ce_loss / n_future

            # 6. Metrics 수집
            total_weighted_ce_loss += weighted_ce_loss.item()
            total_unweighted_ce_loss += unweighted_ce_loss.item()
            total_head0_ce_loss += batch_head0_ce_loss.item() if isinstance(batch_head0_ce_loss, torch.Tensor) else batch_head0_ce_loss
            total_excess_loss += selection_stats.get('head_1_excess_mean', 0.0)
            n_batches += 1

    # 평균 metrics 계산
    avg_weighted_ce_loss = total_weighted_ce_loss / n_batches
    avg_unweighted_ce_loss = total_unweighted_ce_loss / n_batches
    avg_head0_ce_loss = total_head0_ce_loss / n_batches
    avg_excess_loss = total_excess_loss / n_batches

    # Validation metrics aggregation (DDP) - 1회 통신
    reduced_val = all_reduce_scalars({
        "weighted_ce_loss": avg_weighted_ce_loss,
        "unweighted_ce_loss": avg_unweighted_ce_loss,
        "head0_ce_loss": avg_head0_ce_loss,
        "excess_loss": avg_excess_loss,
    })
    avg_weighted_ce_loss = reduced_val["weighted_ce_loss"]
    avg_unweighted_ce_loss = reduced_val["unweighted_ce_loss"]
    avg_head0_ce_loss = reduced_val["head0_ce_loss"]
    avg_excess_loss = reduced_val["excess_loss"]

    metrics = {
        "val_weighted_ce_loss": avg_weighted_ce_loss,
        "val_unweighted_ce_loss": avg_unweighted_ce_loss,
        "val_head0_ce_loss": avg_head0_ce_loss,
        "val_excess_loss": avg_excess_loss,
        "val_loss": avg_unweighted_ce_loss,  # Best checkpoint 기준: 전체 head unweighted CE
    }

    return metrics


def run_rho1_training(config: DictConfig) -> tuple[dict[str, float], str]:
    """Rho-1 WMTP 실행

    Args:
        config: 완전한 config 객체 (OmegaConf DictConfig)

    Returns:
        (final_metrics, best_checkpoint_path)
    """
    # 0. 환경변수 로드 (MLflow credentials 등)
    ensure_env_loaded()

    # 2. Distributed 초기화 (torchrun 환경인 경우)
    if "RANK" in os.environ:
        rank, world_size = init_distributed()
    else:
        rank, world_size = 0, 1

    # 3. 로깅 설정 (rank 정보 포함)
    logger = setup_logging("RHO1", level=config.logging.level, rank=rank)

    logger.info("=== Rho-1 WMTP (Reference-based Weighting) ===")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")

    if "RANK" in os.environ:
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
    else:
        logger.info("Local training (single device)")

    # 4. Environment setup (seed + device)
    actual_seed, device = setup_environment(config.runtime.seed)
    logger.info(f"Device: {device}, Seed: {actual_seed}")

    # 5. MLflow 초기화 (Rank 0만)
    use_mlflow = bool(config.mlflow.experiment)
    use_s3_upload = config.checkpoint.get("s3_upload", True) and use_mlflow
    mlflow_run_id = None  # S3 업로드 시 스레드 안전을 위해 명시적 run_id 저장
    if is_main_process() and use_mlflow:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment)
        mlflow.start_run(
            run_name=config.experiment.name,
            tags={tag: "true" for tag in config.experiment.tags},
        )
        mlflow.log_params(OmegaConf.to_container(config, resolve=True))
        mlflow_run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow Run ID: {mlflow_run_id}")

    # 6. LoRA 설정 파싱
    use_lora = getattr(config.training, "use_lora", False)
    lora_config = None
    if use_lora:
        lora_section = config.training.get("lora", {})
        lora_config = {
            "rank": lora_section.get("rank", 8),
            "alpha": lora_section.get("alpha", 16.0),
            "dropout": lora_section.get("dropout", 0.0),
            "target_modules": lora_section.get(
                "target_modules", ["wq", "wk", "wv", "wo"]
            ),
        }
        logger.info(f"LoRA enabled: rank={lora_config['rank']}, alpha={lora_config['alpha']}")

    # 7. Resource 로딩
    logger.info(f"Loading reference model: {config.models.reference.name}")
    adapter = load_adapter(config, device, use_lora=use_lora, lora_config=lora_config)
    ref_model = load_reference_model(config, device)
    tokenizer = load_tokenizer_from_config(config)
    logger.info("Reference model loaded successfully")

    # 7. DDP wrapping (adapter만 - reference는 frozen inference용)
    adapter = wrap_model_fsdp(
        adapter,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        mixed_precision=config.distributed.fsdp.mixed_precision,
        cpu_offload=config.distributed.fsdp.cpu_offload,
        activation_checkpointing=config.distributed.fsdp.get("activation_checkpointing", False),
    )

    # Model size + System info 로깅 (Rank 0만)
    if is_main_process() and use_mlflow:
        model_size_local = get_model_size(unwrap_model(adapter))

        # FSDP FULL_SHARD 시 world_size를 곱해 실제 전체 파라미터 수 계산
        sharding_strategy = config.distributed.fsdp.sharding_strategy
        if sharding_strategy == "FULL_SHARD" and world_size > 1:
            model_size = {
                "total_params": model_size_local["total_params"] * world_size,
                "trainable_params": model_size_local["trainable_params"] * world_size,
            }
        else:
            model_size = model_size_local

        mlflow.log_params(
            {
                "model_total_params": model_size["total_params"],
                "model_trainable_params": model_size["trainable_params"],
            }
        )
        system_info = get_system_info()
        mlflow.log_params(
            {
                "system_cpu_count": system_info["cpu_count"],
                "system_ram_total_gb": round(system_info["ram_total_gb"], 2),
            }
        )

    # GPU monitor 초기화
    gpu_monitor = GPUMonitor(device)
    throughput_tracker = ThroughputTracker()

    # 8. Dataset & DataLoader 생성
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Train: {config.dataset.train}")
    logger.info(f"Validation: {config.dataset.validation}")

    # sampling_config를 dict로 변환
    sampling_config = OmegaConf.to_container(config.data_sampling, resolve=True)

    train_loader = create_dataloader(
        dataset_path=config.dataset.train,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        sampling_config=sampling_config,
        seed=config.data_sampling.seed,
        shuffle=True,
    )

    # Validation용 sampling_config (n_samples를 val_n_samples로 변경)
    val_sampling_config = sampling_config.copy()
    val_sampling_config["n_samples"] = config.data_sampling.val_n_samples

    val_loader = create_dataloader(
        dataset_path=config.dataset.validation,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        sampling_config=val_sampling_config,
        seed=config.data_sampling.seed,
        shuffle=False,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Dataset statistics 로깅
    if is_main_process() and use_mlflow:
        mlflow.log_params(
            {
                "dataset_train_samples": len(train_loader.dataset),
                "dataset_val_samples": len(val_loader.dataset),
                "dataset_train_batches": len(train_loader),
                "dataset_val_batches": len(val_loader),
            }
        )

    # Optimizer 설정
    # FSDP-wrapped 모델에서 직접 파라미터 접근 (표준 패턴)
    if use_lora:
        trainable_params = [p for p in adapter.parameters() if p.requires_grad]
        logger.info(f"LoRA optimizer: {sum(p.numel() for p in trainable_params):,} trainable params")
    else:
        trainable_params = list(adapter.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # 7. Training setup
    best_val_loss = float("inf")
    global_step = 0

    checkpoint_dir = Path(config.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = config.training.n_epochs
    save_checkpoint_every = config.checkpoint.save_checkpoint_every

    # Fractional epoch 처리
    total_batches = len(train_loader)
    batches_to_run = int(total_batches * n_epochs)

    # Gradient accumulation 초기화
    accumulation_counter = 0
    gradient_accumulation_steps = config.training.gradient_accumulation_steps

    # Optimization steps 계산
    total_optimization_steps = (batches_to_run + gradient_accumulation_steps - 1) // gradient_accumulation_steps

    logger.info(f"Total epochs: {n_epochs}")
    logger.info(f"Total batches to run: {batches_to_run}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {total_optimization_steps}")
    logger.info(f"Validation & Checkpoint every: {save_checkpoint_every} epochs")
    logger.info(f"Top-k selection ratio: {config.training.k_percent}")

    # Learning rate scheduler 생성
    lr_scheduler_config = config.training.get("lr_scheduler", {})
    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_optimization_steps,
        scheduler_type=lr_scheduler_config.get("type", "constant"),
        warmup_ratio=lr_scheduler_config.get("warmup_ratio", 0.05),
        min_lr_ratio=lr_scheduler_config.get("min_lr_ratio", 0.0),
    )

    current_epoch = 0.0
    batch_count = 0
    next_checkpoint_epoch = save_checkpoint_every

    # 모델 dtype 감지
    model_dtype = get_model_dtype(adapter)

    # 8. Training loop
    optimizer.zero_grad()

    while batch_count < batches_to_run:
        # Checkpoint 경계까지 훈련
        target_epoch = min(next_checkpoint_epoch, n_epochs)
        target_batches = int(target_epoch * total_batches)
        batches_this_period = target_batches - batch_count

        logger.info(f"--- Training to epoch {target_epoch:.2f} ---")

        # Throughput 추적 시작
        throughput_tracker.start_epoch()

        # DataLoader에서 필요한 만큼만 사용
        epoch_train_loader = iter(train_loader)
        period_metrics_sum = {"weighted_ce_loss": 0.0, "unweighted_ce_loss": 0.0, "excess_loss": 0.0}
        period_batches = 0

        for _ in range(batches_this_period):
            try:
                batch = next(epoch_train_loader)
            except StopIteration:
                # DataLoader 재시작
                epoch_train_loader = iter(train_loader)
                batch = next(epoch_train_loader)

            # 1 batch 훈련 (Rho-1 로직)
            adapter.train()
            ref_model.eval()  # Reference는 항상 eval

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Loss mask 생성 (labels != -100)
            loss_mask = (labels != -100)

            # Reference forward (no grad, HuggingFace 모델)
            with torch.no_grad():
                ref_outputs = ref_model(input_ids)
                ref_logits = ref_outputs.logits  # [batch, seq, vocab]

            # Pass 1: Policy forward (no_grad) - weights 계산용
            with torch.no_grad():
                policy_logits_for_weights = adapter(input_ids)

                # MTP selective weights (per-head binary selection)
                weights, selection_stats = compute_mtp_selective_weights(
                    policy_logits=policy_logits_for_weights,
                    ref_logits=ref_logits,
                    labels=labels,
                    loss_mask=loss_mask,
                    k_percent=config.training.k_percent,
                    mode=config.training.rho1_mode,
                )

                del policy_logits_for_weights  # 메모리 즉시 해제

            # Pass 2: Forward + Weighted Loss
            logits = adapter(input_ids)  # [batch, seq, n_future, vocab]

            # Weighted Loss 계산 (3D weights)
            loss_dict = compute_mtp_ce_loss(
                logits=logits,
                labels=labels,
                attention_mask=attention_mask,
                weights=weights,  # [batch, seq, n_future] 3D weights
            )
            weighted_ce_loss = loss_dict["weighted_ce_loss"]
            unweighted_ce_loss = loss_dict["unweighted_ce_loss"]

            # Backward (외부에서 명시적 호출 - FSDP 호환)
            scaled_loss = weighted_ce_loss / gradient_accumulation_steps
            scaled_loss.backward()

            accumulation_counter += 1
            batch_count += 1
            period_batches += 1

            # Throughput 추적
            batch_size_actual = input_ids.size(0)
            n_tokens = (attention_mask.sum()).item()
            throughput_tracker.update(batch_size_actual, int(n_tokens))

            # Period metrics 누적 (batch 단위)
            period_metrics_sum["weighted_ce_loss"] += weighted_ce_loss.detach().item()
            period_metrics_sum["unweighted_ce_loss"] += unweighted_ce_loss.detach().item()
            period_metrics_sum["excess_loss"] += selection_stats.get('head_1_excess_mean', 0.0)

            # Optimizer step (accumulation 완료 시에만)
            if accumulation_counter >= gradient_accumulation_steps:
                # Gradient clipping (누적된 gradient에 적용)
                if config.training.max_grad_norm > 0:
                    grad_clip_stats = compute_gradient_clip_stats(
                        adapter,
                        config.training.max_grad_norm,
                    )
                else:
                    from weighted_mtp.utils.metrics_utils import compute_gradient_norm

                    grad_norm_dict = compute_gradient_norm(adapter)
                    grad_clip_stats = {
                        "grad_norm_pre_clip": grad_norm_dict["grad_norm"],
                        "grad_norm_post_clip": grad_norm_dict["grad_norm"],
                        "grad_clip_ratio": 1.0,
                    }

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                accumulation_counter = 0

            # Step-level 로깅 (optimizer step 시에만)
            if global_step % config.training.log_interval == 0 and accumulation_counter == 0:
                # GPU metrics
                gpu_metrics = gpu_monitor.get_metrics()

                # Weight distribution statistics (response 토큰만)
                weight_dist_stats = compute_weight_statistics(weights, attention_mask, labels)

                # Metric aggregation (분산 환경)
                # grad_clip_stats는 clip_grad_norm_이 이미 전역 값을 반환하므로 all_reduce 불필요
                avg_grad_norm_post = grad_clip_stats["grad_norm_post_clip"]
                avg_grad_norm_pre = grad_clip_stats["grad_norm_pre_clip"]
                avg_grad_clip_ratio = grad_clip_stats["grad_clip_ratio"]

                # 모든 로컬 메트릭 배치 all_reduce (1회 통신)
                reduced = all_reduce_scalars({
                    "weighted_ce": weighted_ce_loss.item(),
                    "unweighted_ce": unweighted_ce_loss.item(),
                    "selection_ratio": selection_stats['selection_ratio'],
                    "avg_heads_per_pos": selection_stats.get('avg_heads_per_position', 0.0),
                    "weight_mean": weight_dist_stats["weight_mean"],
                    "weight_std": weight_dist_stats["weight_std"],
                    "weight_min": weight_dist_stats["weight_min"],
                    "weight_max": weight_dist_stats["weight_max"],
                    "weight_entropy": weight_dist_stats["weight_entropy"],
                })
                avg_weighted_ce = reduced["weighted_ce"]
                avg_unweighted_ce = reduced["unweighted_ce"]
                avg_selection_ratio = reduced["selection_ratio"]

                if is_main_process():
                    if use_mlflow:
                        mlflow.log_metrics(
                            {
                                "train/weighted_ce_loss": avg_weighted_ce,
                                "train/unweighted_ce_loss": avg_unweighted_ce,
                                "train/selection_ratio": avg_selection_ratio,
                                "train/avg_heads_per_pos": reduced["avg_heads_per_pos"],
                                "train/grad_norm": avg_grad_norm_post,
                                "train/grad_norm_pre_clip": avg_grad_norm_pre,
                                "train/grad_clip_ratio": avg_grad_clip_ratio,
                                "train/learning_rate": optimizer.param_groups[0]["lr"],
                                "weight/mean": reduced["weight_mean"],
                                "weight/std": reduced["weight_std"],
                                "weight/min": reduced["weight_min"],
                                "weight/max": reduced["weight_max"],
                                "weight/entropy": reduced["weight_entropy"],
                                "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                                "system/gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
                            },
                            step=global_step,
                        )
                    logger.info(
                        f"Step {global_step}/{total_optimization_steps}, "
                        f"Loss: {avg_weighted_ce:.4f}, "
                        f"Grad Norm: {avg_grad_norm_post:.4f} (Clip: {avg_grad_clip_ratio:.2f}), "
                        f"Selection: {avg_selection_ratio:.1%}"
                    )

        # Period loop 종료

        # Incomplete accumulation 처리 (validation 전)
        if accumulation_counter > 0:
            logger.info(f"Processing incomplete accumulation ({accumulation_counter} batches before validation)")

            # Gradient clipping
            if config.training.max_grad_norm > 0:
                grad_clip_stats = compute_gradient_clip_stats(
                    adapter,
                    config.training.max_grad_norm,
                )

            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            accumulation_counter = 0

        # Epoch 경계 도달
        current_epoch = batch_count / total_batches

        # Period-level metrics 계산
        train_weighted_ce_avg = period_metrics_sum["weighted_ce_loss"] / period_batches
        train_unweighted_ce_avg = period_metrics_sum["unweighted_ce_loss"] / period_batches
        train_excess_avg = period_metrics_sum["excess_loss"] / period_batches

        logger.info(
            f"Epoch {current_epoch:.2f} 도달 - "
            f"Train Weighted CE: {train_weighted_ce_avg:.4f}"
        )

        # Validation 실행 (epoch 경계에서)
        logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

        val_metrics = validate_rho1(
            adapter=adapter,
            ref_model=ref_model,
            dataloader=val_loader,
            device=device,
            k_percent=config.training.k_percent,
            rho1_mode=config.training.rho1_mode,
        )

        # Epoch-level 로깅 (Rank 0만)
        if is_main_process():
            # Throughput 및 GPU 메트릭 수집
            throughput_metrics = throughput_tracker.get_epoch_metrics()
            gpu_metrics_epoch = gpu_monitor.get_metrics()

            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "train/epoch_weighted_ce_loss": train_weighted_ce_avg,
                        "train/epoch_unweighted_ce_loss": train_unweighted_ce_avg,
                        "train/epoch_excess_loss": train_excess_avg,
                        "val/weighted_ce_loss": val_metrics["val_weighted_ce_loss"],
                        "val/unweighted_ce_loss": val_metrics["val_unweighted_ce_loss"],
                        "val/head0_ce_loss": val_metrics["val_head0_ce_loss"],
                        "val/excess_loss": val_metrics["val_excess_loss"],
                        "perf/epoch_time_sec": throughput_metrics["epoch_time_sec"],
                        "perf/samples_per_sec": throughput_metrics["samples_per_sec"],
                        "perf/tokens_per_sec": throughput_metrics["tokens_per_sec"],
                        "system/gpu_memory_reserved_gb": gpu_metrics_epoch["gpu_memory_reserved_gb"],
                    },
                    step=global_step,
                )

            logger.info(
                f"Validation - Unweighted CE: {val_metrics['val_unweighted_ce_loss']:.4f}, "
                f"Weighted CE: {val_metrics['val_weighted_ce_loss']:.4f}, "
                f"Head0 CE: {val_metrics['val_head0_ce_loss']:.4f}"
            )

        # Checkpoint 저장 (validation loss 개선 시만)
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}.pt"

            save_lora_only = config.checkpoint.get("save_lora_only", False) and use_lora

            # 모든 rank가 참여 (FSDP gathering), 실제 저장은 함수 내부에서 rank 0만 수행
            if save_lora_only:
                save_lora_checkpoint(
                    adapter=adapter,
                    optimizer=optimizer,
                    epoch=current_epoch,
                    train_metrics={
                        "train_weighted_ce_loss": train_weighted_ce_avg,
                        "train_excess_loss": train_excess_avg,
                    },
                    val_metrics=val_metrics,
                    checkpoint_path=checkpoint_path,
                    config={"model": {"path": config.models.policy.path}},
                    s3_upload=use_s3_upload,
                    experiment_name=config.experiment.name,
                )
            else:
                save_checkpoint(
                    adapter=adapter,
                    optimizer=optimizer,
                    epoch=current_epoch,
                    train_metrics={
                        "train_weighted_ce_loss": train_weighted_ce_avg,
                        "train_excess_loss": train_excess_avg,
                    },
                    val_metrics=val_metrics,
                    checkpoint_path=checkpoint_path,
                    config={"model": {"path": config.models.policy.path}},
                    s3_upload=use_s3_upload,
                    experiment_name=config.experiment.name,
                )

            # 로깅 및 cleanup은 rank 0만 수행
            if is_main_process():
                logger.info(f"Checkpoint saved: {checkpoint_path.name} (val_unweighted_ce: {best_val_loss:.4f})")

                if config.checkpoint.get("save_total_limit"):
                    cleanup_old_checkpoints(
                        checkpoint_dir=checkpoint_dir,
                        save_total_limit=config.checkpoint.save_total_limit,
                    )

                    if use_s3_upload:
                        cleanup_s3_checkpoints(
                            experiment_name=config.experiment.name,
                            save_total_limit=config.checkpoint.save_total_limit,
                        )
        else:
            logger.info(f"Validation unweighted CE did not improve ({val_metrics['val_loss']:.4f} >= {best_val_loss:.4f}), skipping checkpoint save")

        # 다음 checkpoint 경계 설정
        next_checkpoint_epoch += save_checkpoint_every

    # 9. Final checkpoint
    if config.checkpoint.save_final:
        final_path = checkpoint_dir / "checkpoint_final.pt"

        # 최종 validation 실행
        logger.info("--- Final Validation ---")
        final_val_metrics = validate_rho1(
            adapter=adapter,
            ref_model=ref_model,
            dataloader=val_loader,
            device=device,
            k_percent=config.training.k_percent,
            rho1_mode=config.training.rho1_mode,
        )

        save_lora_only = config.checkpoint.get("save_lora_only", False) and use_lora

        # 모든 rank가 참여 (FSDP gathering), 실제 저장은 함수 내부에서 rank 0만 수행
        if save_lora_only:
            save_lora_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={
                    "train_weighted_ce_loss": train_weighted_ce_avg,
                    "train_excess_loss": train_excess_avg,
                },
                val_metrics=final_val_metrics,
                checkpoint_path=final_path,
                config={"model": {"path": config.models.policy.path}},
                s3_upload=use_s3_upload,
                experiment_name=config.experiment.name,
            )
        else:
            save_checkpoint(
                adapter=adapter,
                optimizer=optimizer,
                epoch=current_epoch,
                train_metrics={
                    "train_weighted_ce_loss": train_weighted_ce_avg,
                    "train_excess_loss": train_excess_avg,
                },
                val_metrics=final_val_metrics,
                checkpoint_path=final_path,
                config={"model": {"path": config.models.policy.path}},
                s3_upload=use_s3_upload,
                experiment_name=config.experiment.name,
            )

        if is_main_process():
            logger.info(f"Final checkpoint saved: {final_path.name}")

    # 10. 모든 S3 업로드 완료 대기 및 MLflow 종료
    shutdown_s3_executor()
    if is_main_process() and use_mlflow:
        mlflow.end_run()

        # MLflow 메트릭/파라미터 S3 백업
        if use_s3_upload:
            sync_mlruns_to_s3()

    # 최신 checkpoint 경로 반환
    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    latest_checkpoint_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    logger.info(f"Rho-1 WMTP 완료! Latest checkpoint: {latest_checkpoint_path}")

    return final_val_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rho-1 WMTP (Reference-based Weighting)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/rho1/rho1.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        help="Config override (e.g., --override experiment.name=test)",
    )
    args = parser.parse_args()

    # Config 로드
    config = OmegaConf.load(args.config)

    # Override 적용
    if args.overrides:
        from weighted_mtp.utils.config_utils import apply_overrides

        config = apply_overrides(config, args.overrides)

    run_rho1_training(config)
