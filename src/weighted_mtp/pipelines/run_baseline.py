"""Unified NTP Training Pipeline (4 Weight Modes)

HuggingFace LlamaForCausalLM 기반 Next-Token Prediction 학습.
weight_mode 설정으로 4가지 가중치 전략을 통합 지원.

Weight modes:
  - "uniform": 표준 NTP (가중치 없음, baseline)
  - "critic": TAW - Token Advantage Weighting (핵심 연구 기여)
  - "random": Random-Matched (LogNormal 분포, 대조군)
  - "shuffled": Shuffled (Critic 가중치 위치 셔플, 대조군)

독립 실행:
    python -m weighted_mtp.pipelines.run_baseline --config configs/baseline/baseline.yaml
"""

import argparse
import os
from pathlib import Path

import mlflow
import torch
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

from weighted_mtp.core.env import ensure_env_loaded
from weighted_mtp.core.logging import setup_logging
from weighted_mtp.data.dataloader import create_dataloader
from weighted_mtp.models.lora import apply_lora_to_hf_model
from weighted_mtp.models.tokenizer_utils import load_tokenizer_from_config
from weighted_mtp.utils import (
    GPUMonitor,
    ThroughputTracker,
    cleanup_old_checkpoints,
    cleanup_s3_checkpoints,
    compute_gradient_clip_stats,
    create_scheduler,
    get_model_size,
    get_system_info,
    save_hf_checkpoint,
    save_hf_lora_checkpoint,
    shutdown_s3_executor,
    sync_mlruns_to_s3,
)
from weighted_mtp.utils.loss_utils import compute_weighted_ntp_loss
from weighted_mtp.runtime import (
    init_distributed,
    setup_environment,
    is_main_process,
    wrap_model_fsdp,
    unwrap_model,
    all_reduce_scalars,
    barrier,
)

# Value weighting (critic/shuffled modes)
from weighted_mtp.models.value_model import ValueModel
from weighted_mtp.value_weighting.td_weighting import compute_gae_advantage, build_weights
from weighted_mtp.value_weighting.td_stats_ema import TDStatsEMA

# Control weights (random/shuffled modes)
from weighted_mtp.value_weighting.control_weights import (
    generate_random_matched_weights,
    shuffle_weights_within_sequence,
)

VALID_WEIGHT_MODES = {"uniform", "critic", "random", "shuffled"}


# ---------------------------------------------------------------------------
# Model / Tokenizer loading
# ---------------------------------------------------------------------------

def load_hf_model(
    config: DictConfig,
    device: torch.device,
    use_lora: bool = False,
    lora_config: dict | None = None,
) -> AutoModelForCausalLM:
    """HuggingFace LlamaForCausalLM 로드

    Args:
        config: 전체 config (models.policy 하위 사용)
        device: 디바이스
        use_lora: LoRA 적용 여부
        lora_config: LoRA 설정 dict

    Returns:
        LlamaForCausalLM 인스턴스
    """
    model_path = config.models.policy.path
    dtype_str = getattr(config.models.policy, "dtype", "bfloat16")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # 분산학습 감지: FSDP 사용 시 CPU에 로드
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )

    # 디바이스 이동: 분산학습 시 CPU 유지 (FSDP가 GPU 배치 담당)
    if not is_distributed and device.type != "cpu":
        model = model.to(device)

    # LoRA 적용
    if use_lora and lora_config is not None:
        apply_lora_to_hf_model(model, lora_config)

    return model


def load_hf_tokenizer(config: DictConfig) -> AutoTokenizer:
    """HuggingFace 토크나이저 로드 (config 기반)

    Args:
        config: 전체 config

    Returns:
        AutoTokenizer 인스턴스
    """
    return load_tokenizer_from_config(config)


# ---------------------------------------------------------------------------
# Critic weight computation
# ---------------------------------------------------------------------------

def compute_critic_weights(
    value_model: ValueModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    rewards: torch.Tensor,
    td_stats_ema: TDStatsEMA,
    config: DictConfig,
) -> torch.Tensor:
    """Critic GAE 가중치 계산

    Steps:
        1. value_model forward -> value_logits [batch, seq, 1]
        2. loss_mask = (labels != -100)
        3. compute_gae_advantage(value_logits, rewards, loss_mask, gamma=1.0, lam)
        4. td_stats_ema.update(advantages, loss_mask) -> get EMA mean/std
        5. build_weights(advantages, loss_mask, beta, clip, external_mean, external_std)

    Args:
        value_model: 학습된 frozen ValueModel
        input_ids: [batch, seq]
        attention_mask: [batch, seq]
        labels: [batch, seq] (-100 = ignore)
        rewards: [batch] (NTP TAW에서는 모두 1.0)
        td_stats_ema: EMA 통계 추적기
        config: 전체 config (training 하위 사용)

    Returns:
        weights: [batch, seq] 토큰별 가중치
    """
    with torch.no_grad():
        value_logits = value_model(input_ids, attention_mask=attention_mask)
        # value_logits: [batch, seq, 1]

    loss_mask = (labels != -100).float()

    # GAE Advantage 계산
    td_lambda = config.training.get("td_lambda", 0.95)
    advantages = compute_gae_advantage(
        value_logits=value_logits,
        rewards=rewards,
        loss_mask=loss_mask,
        gamma=1.0,
        lam=td_lambda,
    )
    # advantages: [batch, seq]

    # EMA 통계로 가중치 빌드 (get_stats -> build_weights -> update 순서)
    ema_mean, ema_std = td_stats_ema.get_stats()

    beta = config.training.get("beta", 1.0)
    weight_clip_min = config.training.get("weight_clip_min", 0.1)
    weight_clip_max = config.training.get("weight_clip_max", 3.0)

    weights = build_weights(
        td_errors=advantages,
        loss_mask=loss_mask,
        beta=beta,
        min_weight=weight_clip_min,
        max_weight=weight_clip_max,
        external_mean=ema_mean,
        external_std=ema_std,
    )
    # weights: [batch, seq]

    # EMA 업데이트 (다음 batch에 반영)
    td_stats_ema.update(advantages, loss_mask)

    return weights


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_ntp(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    weight_mode: str = "uniform",
    value_model: ValueModel | None = None,
    td_stats_ema: TDStatsEMA | None = None,
    config: DictConfig | None = None,
) -> dict[str, float]:
    """Validation 수행 (모든 weight mode 지원)

    Args:
        model: HuggingFace CausalLM (FSDP-wrapped 가능)
        dataloader: Validation DataLoader
        device: 디바이스
        weight_mode: 가중치 모드
        value_model: ValueModel (critic/shuffled에서 필요)
        td_stats_ema: EMA 추적기 (critic/shuffled에서 필요)
        config: 전체 config (critic/shuffled에서 필요)

    Returns:
        Validation metrics dict
    """
    model.eval()
    if value_model is not None:
        value_model.eval()

    total_weighted_loss = 0.0
    total_unweighted_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits  # [batch, seq, vocab]

            # Weight computation
            if weight_mode == "uniform":
                weights = None
            elif weight_mode == "critic":
                batch_size = input_ids.size(0)
                rewards = torch.ones(batch_size, device=device)
                weights = compute_critic_weights(
                    value_model, input_ids, attention_mask, labels,
                    rewards, td_stats_ema, config,
                )
            elif weight_mode == "random":
                weights = generate_random_matched_weights(
                    logits.shape[:2], device,
                )
            elif weight_mode == "shuffled":
                batch_size = input_ids.size(0)
                rewards = torch.ones(batch_size, device=device)
                weights = compute_critic_weights(
                    value_model, input_ids, attention_mask, labels,
                    rewards, td_stats_ema, config,
                )
                weights = shuffle_weights_within_sequence(weights, attention_mask)
            else:
                raise ValueError(f"Unknown weight_mode: {weight_mode}")

            # Loss
            loss_dict = compute_weighted_ntp_loss(logits, labels, attention_mask, weights)

            total_weighted_loss += loss_dict["weighted_ce_loss"].item()
            total_unweighted_loss += loss_dict["unweighted_ce_loss"].item()
            n_batches += 1

    avg_weighted_loss = total_weighted_loss / max(n_batches, 1)
    avg_unweighted_loss = total_unweighted_loss / max(n_batches, 1)

    return {
        "val_loss": avg_unweighted_loss,
        "val_weighted_loss": avg_weighted_loss,
        "val_unweighted_loss": avg_unweighted_loss,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def run_baseline_training(config: DictConfig) -> tuple[dict[str, float], str]:
    """Unified NTP Training (4 weight modes)

    Args:
        config: 완전한 config 객체 (OmegaConf DictConfig)

    Returns:
        (final_metrics, best_checkpoint_path)
    """
    # 0. 환경변수 로드
    ensure_env_loaded()

    # 1. Weight mode 결정
    weight_mode = config.training.get("weight_mode", "uniform")
    assert weight_mode in VALID_WEIGHT_MODES, (
        f"Invalid weight_mode '{weight_mode}'. Must be one of {VALID_WEIGHT_MODES}"
    )

    # 2. Distributed 초기화
    if "RANK" in os.environ:
        rank, world_size = init_distributed()
    else:
        rank, world_size = 0, 1

    # 3. 로깅 설정
    logger = setup_logging("NTP", level=config.logging.level, rank=rank)

    if "RANK" in os.environ:
        logger.info(f"Distributed training: rank={rank}, world_size={world_size}")
    else:
        logger.info("Local training (single device)")

    mode_descriptions = {
        "uniform": "Uniform (Baseline NTP)",
        "critic": "TAW - Token Advantage Weighting (Critic)",
        "random": "Random-Matched (LogNormal Control)",
        "shuffled": "Shuffled (Position-Destroyed Control)",
    }
    logger.info(f"=== NTP Training — {mode_descriptions[weight_mode]} ===")
    logger.info(f"Experiment: {config.experiment.name}")
    logger.info(f"Description: {config.experiment.description}")
    logger.info(f"Weight mode: {weight_mode}")

    # 4. Environment setup (seed + device)
    actual_seed, device = setup_environment(config.runtime.seed)
    logger.info(f"Device: {device}, Seed: {actual_seed}")

    # 5. MLflow 초기화 (Rank 0만)
    use_mlflow = bool(config.mlflow.experiment)
    use_s3_upload = config.checkpoint.get("s3_upload", True) and use_mlflow
    mlflow_run_id = None
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

    # 6. LoRA 설정
    use_lora = getattr(config.training, "use_lora", False)
    lora_config = None
    if use_lora:
        lora_section = config.training.get("lora", {})
        lora_config = {
            "rank": lora_section.get("rank", 8),
            "alpha": lora_section.get("alpha", 16.0),
            "dropout": lora_section.get("dropout", 0.0),
            "target_modules": lora_section.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
        }
        logger.info(f"LoRA enabled: rank={lora_config['rank']}, alpha={lora_config['alpha']}")

    # 7. Policy model 로드
    model = load_hf_model(config, device, use_lora=use_lora, lora_config=lora_config)
    model = wrap_model_fsdp(
        model,
        device,
        sharding_strategy=config.distributed.fsdp.sharding_strategy,
        mixed_precision=config.distributed.fsdp.mixed_precision,
        cpu_offload=config.distributed.fsdp.cpu_offload,
        activation_checkpointing=config.distributed.fsdp.get("activation_checkpointing", False),
    )
    tokenizer = load_hf_tokenizer(config)

    # 8. Value model 로드 (critic/shuffled 전용)
    value_model = None
    td_stats_ema = None

    if weight_mode in ("critic", "shuffled"):
        if not hasattr(config.models, "value_model") or not hasattr(config.models.value_model, "checkpoint_path"):
            raise ValueError(
                f"weight_mode='{weight_mode}' requires config.models.value_model.checkpoint_path, "
                "but it is not defined in the config."
            )
        value_checkpoint_path = config.models.value_model.checkpoint_path
        logger.info(f"Loading Value Model from: {value_checkpoint_path}")

        base_model_override = getattr(config.models.value_model, "base_model_path", None)
        value_model = ValueModel.from_checkpoint(
            checkpoint_path=value_checkpoint_path,
            device=str(device),
            base_model_path_override=base_model_override,
        )
        value_model.eval_mode()  # 전체 frozen + eval

        # FSDP wrapping (LlamaDecoderLayer 기반)
        value_model = wrap_model_fsdp(
            value_model,
            device,
            sharding_strategy=config.distributed.fsdp.sharding_strategy,
            mixed_precision=config.distributed.fsdp.mixed_precision,
            cpu_offload=config.distributed.fsdp.cpu_offload,
            activation_checkpointing=False,  # frozen 모델은 activation checkpointing 불필요
        )
        logger.info("Value Model loaded and FSDP-wrapped (frozen)")

        # TDStatsEMA 초기화
        td_ema_momentum = config.training.get("td_ema_momentum", 0.1)
        td_ema_warmup_steps = config.training.get("td_ema_warmup_steps", 10)
        td_stats_ema = TDStatsEMA(
            device=device,
            momentum=td_ema_momentum,
            warmup_steps=td_ema_warmup_steps,
        )
        logger.info(
            f"TDStatsEMA initialized: momentum={td_ema_momentum}, "
            f"warmup_steps={td_ema_warmup_steps}"
        )

    # 9. Model size 로깅
    model_size_local = get_model_size(unwrap_model(model))

    sharding_strategy = config.distributed.fsdp.sharding_strategy
    if sharding_strategy == "FULL_SHARD" and world_size > 1:
        model_size = {
            "total_params": model_size_local["total_params"] * world_size,
            "trainable_params": model_size_local["trainable_params"] * world_size,
            "non_trainable_params": model_size_local["non_trainable_params"] * world_size,
        }
    else:
        model_size = model_size_local

    if is_main_process():
        if use_mlflow:
            mlflow.log_params(
                {
                    "model_total_params": model_size["total_params"],
                    "model_trainable_params": model_size["trainable_params"],
                    "model_non_trainable_params": model_size["non_trainable_params"],
                }
            )
        logger.info(
            f"Model size: {model_size['trainable_params']:,} trainable / "
            f"{model_size['total_params']:,} total params"
        )

        system_info = get_system_info()
        if use_mlflow:
            mlflow.log_params(
                {
                    "system_cpu_count": system_info["cpu_count"],
                    "system_ram_total_gb": round(system_info["ram_total_gb"], 2),
                }
            )

    # GPU monitor
    gpu_monitor = GPUMonitor(device)

    # 10. Dataset & DataLoader
    logger.info(f"Dataset: {config.dataset.name}")
    logger.info(f"Train: {config.dataset.train}")
    logger.info(f"Validation: {config.dataset.validation}")

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

    if is_main_process() and use_mlflow:
        mlflow.log_params(
            {
                "dataset_train_samples": len(train_loader.dataset),
                "dataset_val_samples": len(val_loader.dataset),
                "dataset_train_batches": len(train_loader),
                "dataset_val_batches": len(val_loader),
            }
        )

    # 11. Optimizer 설정
    if use_lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"LoRA optimizer: {sum(p.numel() for p in trainable_params):,} trainable params")
    else:
        trainable_params = list(model.parameters())

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # 12. Training setup
    best_val_loss = float("inf")
    global_step = 0

    checkpoint_dir = Path(config.checkpoint.save_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    n_epochs = config.training.n_epochs
    save_checkpoint_every = config.checkpoint.save_checkpoint_every

    # Fractional epoch 처리
    total_batches = len(train_loader)
    batches_to_run = max(1, int(total_batches * n_epochs))

    # Gradient accumulation
    accumulation_counter = 0
    grad_clip_stats = {
        "grad_norm_pre_clip": 0.0,
        "grad_norm_post_clip": 0.0,
        "grad_clip_ratio": 1.0,
    }
    gradient_accumulation_steps = config.training.gradient_accumulation_steps

    total_optimization_steps = (batches_to_run + gradient_accumulation_steps - 1) // gradient_accumulation_steps

    logger.info(f"Total epochs: {n_epochs}")
    logger.info(f"Total batches to run: {batches_to_run}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {total_optimization_steps}")
    logger.info(f"Validation & Checkpoint every: {save_checkpoint_every} epochs")

    # LR scheduler
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
    train_loss_avg = 0.0

    throughput_tracker = ThroughputTracker()

    # 13. Training loop
    optimizer.zero_grad()

    if "RANK" in os.environ:
        logger.info(f"[Rank {rank}] Reaching barrier before training loop")
        barrier()
        logger.info(f"[Rank {rank}] Barrier passed, entering training loop")

    while batch_count < batches_to_run:
        target_epoch = min(next_checkpoint_epoch, n_epochs)
        target_batches = int(target_epoch * total_batches)
        batches_this_period = target_batches - batch_count

        logger.info(f"--- Training to epoch {target_epoch:.2f} ---")

        epoch_train_loader = iter(train_loader)
        period_loss_sum = 0.0
        period_unweighted_loss_sum = 0.0
        period_batches = 0

        for _ in range(batches_this_period):
            try:
                batch = next(epoch_train_loader)
            except StopIteration:
                epoch_train_loader = iter(train_loader)
                batch = next(epoch_train_loader)

            model.train()

            if batch_count == 0:
                logger.info(f"[Rank {rank}] Starting first batch — moving data to device")

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward
            if batch_count == 0:
                logger.info(f"[Rank {rank}] Data moved to device, starting forward")

            outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits  # [batch, seq, vocab]

            # Weight computation based on weight_mode
            if weight_mode == "uniform":
                weights = None
            elif weight_mode == "critic":
                batch_size = input_ids.size(0)
                rewards = torch.ones(batch_size, device=device)
                weights = compute_critic_weights(
                    value_model, input_ids, attention_mask, labels,
                    rewards, td_stats_ema, config,
                )
            elif weight_mode == "random":
                weights = generate_random_matched_weights(
                    logits.shape[:2], device,
                )
            elif weight_mode == "shuffled":
                batch_size = input_ids.size(0)
                rewards = torch.ones(batch_size, device=device)
                weights = compute_critic_weights(
                    value_model, input_ids, attention_mask, labels,
                    rewards, td_stats_ema, config,
                )
                weights = shuffle_weights_within_sequence(weights, attention_mask)
            else:
                raise ValueError(f"Unknown weight_mode: {weight_mode}")

            # Loss
            loss_dict = compute_weighted_ntp_loss(logits, labels, attention_mask, weights)
            weighted_loss = loss_dict["weighted_ce_loss"]
            unweighted_loss = loss_dict["unweighted_ce_loss"]

            # Backward
            scaled_loss = weighted_loss / gradient_accumulation_steps
            scaled_loss.backward()

            if batch_count == 0:
                logger.info(f"[Rank {rank}] Forward and backward completed")

            accumulation_counter += 1
            batch_count += 1
            period_batches += 1

            # Throughput tracking
            batch_size_actual = input_ids.size(0)
            n_tokens = attention_mask.sum().item()
            throughput_tracker.update(batch_size_actual, int(n_tokens))

            # Period metrics 누적
            period_loss_sum += weighted_loss.detach().item()
            period_unweighted_loss_sum += unweighted_loss.detach().item()

            # Optimizer step
            if accumulation_counter >= gradient_accumulation_steps:
                if config.training.max_grad_norm > 0:
                    grad_clip_stats = compute_gradient_clip_stats(
                        model,
                        config.training.max_grad_norm,
                    )
                else:
                    from weighted_mtp.utils.metrics_utils import compute_gradient_norm

                    grad_norm_dict = compute_gradient_norm(model)
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

            # Step-level 로깅
            if global_step % config.training.log_interval == 0 and accumulation_counter == 0:
                gpu_metrics = gpu_monitor.get_metrics()

                avg_grad_norm_pre = grad_clip_stats["grad_norm_pre_clip"]
                avg_grad_norm_post = grad_clip_stats["grad_norm_post_clip"]
                avg_grad_clip_ratio = grad_clip_stats["grad_clip_ratio"]

                reduced = all_reduce_scalars({
                    "weighted_loss": weighted_loss.detach().item(),
                    "unweighted_loss": unweighted_loss.detach().item(),
                })
                avg_weighted_loss = reduced["weighted_loss"]
                avg_unweighted_loss = reduced["unweighted_loss"]

                if is_main_process():
                    step_metrics = {
                        "train/loss": avg_unweighted_loss,
                        "train/weighted_loss": avg_weighted_loss,
                        "train/unweighted_loss": avg_unweighted_loss,
                        "train/grad_norm": avg_grad_norm_post,
                        "train/grad_norm_pre_clip": avg_grad_norm_pre,
                        "train/grad_clip_ratio": avg_grad_clip_ratio,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "system/gpu_memory_allocated_gb": gpu_metrics["gpu_memory_allocated_gb"],
                        "system/gpu_utilization_pct": gpu_metrics["gpu_utilization_pct"],
                    }

                    # TD EMA stats (critic/shuffled only)
                    if td_stats_ema is not None:
                        ema_debug = td_stats_ema.get_debug_stats()
                        step_metrics["train/td_ema_mean"] = ema_debug["ema_mean"]
                        step_metrics["train/td_ema_std"] = ema_debug["ema_std"]
                        step_metrics["train/td_ema_step"] = ema_debug["step_count"]

                    if use_mlflow:
                        mlflow.log_metrics(step_metrics, step=global_step)

                    logger.info(
                        f"Step {global_step}/{total_optimization_steps}, "
                        f"Loss: {avg_unweighted_loss:.4f}, "
                        f"Weighted Loss: {avg_weighted_loss:.4f}, "
                        f"Grad Norm: {avg_grad_norm_post:.4f} (Clip: {avg_grad_clip_ratio:.2f})"
                    )

        # Period loop 종료

        # Incomplete accumulation 처리
        if accumulation_counter > 0:
            logger.info(f"Processing incomplete accumulation ({accumulation_counter} batches before validation)")

            if config.training.max_grad_norm > 0:
                grad_clip_stats = compute_gradient_clip_stats(
                    model,
                    config.training.max_grad_norm,
                )
            else:
                from weighted_mtp.utils.metrics_utils import compute_gradient_norm

                grad_norm_dict = compute_gradient_norm(model)
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

        # Epoch 경계 도달
        current_epoch = batch_count / total_batches
        if period_batches > 0:
            train_loss_avg = period_loss_sum / period_batches
            train_unweighted_loss_avg = period_unweighted_loss_sum / period_batches
            reduced_period = all_reduce_scalars({
                "train_loss": train_loss_avg,
                "train_unweighted_loss": train_unweighted_loss_avg,
            })
            train_loss_avg = reduced_period["train_loss"]
            train_unweighted_loss_avg = reduced_period["train_unweighted_loss"]

        logger.info(
            f"Epoch {current_epoch:.2f} — "
            f"Weighted Loss: {train_loss_avg:.4f}, "
            f"Unweighted Loss: {train_unweighted_loss_avg:.4f}"
        )

        # Validation
        logger.info(f"--- Validation at epoch {current_epoch:.2f} ---")

        val_metrics = validate_ntp(
            model=model,
            dataloader=val_loader,
            device=device,
            weight_mode=weight_mode,
            value_model=value_model,
            td_stats_ema=td_stats_ema,
            config=config,
        )

        reduced_val = all_reduce_scalars({
            "val_loss": val_metrics["val_loss"],
            "val_weighted_loss": val_metrics["val_weighted_loss"],
        })
        avg_val_loss = reduced_val["val_loss"]
        avg_val_weighted_loss = reduced_val["val_weighted_loss"]

        gpu_metrics_epoch = gpu_monitor.get_metrics()
        throughput_metrics = throughput_tracker.get_epoch_metrics()

        if is_main_process():
            epoch_metrics = {
                "train/epoch_loss": train_loss_avg,
                "train/epoch_unweighted_loss": train_unweighted_loss_avg,
                "val/loss": avg_val_loss,
                "val/weighted_loss": avg_val_weighted_loss,
                "perf/epoch_time_sec": throughput_metrics["epoch_time_sec"],
                "perf/samples_per_sec": throughput_metrics["samples_per_sec"],
                "perf/tokens_per_sec": throughput_metrics["tokens_per_sec"],
                "system/gpu_memory_reserved_gb": gpu_metrics_epoch["gpu_memory_reserved_gb"],
            }
            if use_mlflow:
                mlflow.log_metrics(epoch_metrics, step=global_step)

        logger.info(
            f"Validation — Loss: {avg_val_loss:.4f}, "
            f"Weighted Loss: {avg_val_weighted_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")

        # Checkpoint 저장
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}.pt"

        save_lora_only = config.checkpoint.get("save_lora_only", False) and use_lora

        if save_lora_only:
            # TD EMA state 포함 (critic/shuffled)
            extra_state = {}
            if td_stats_ema is not None:
                extra_state["td_ema_state"] = td_stats_ema.state_dict()

            save_hf_lora_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=checkpoint_path,
                config=config,
                epoch=current_epoch,
                val_metrics=val_metrics,
            )

            # TD EMA state를 별도로 checkpoint에 추가 (rank 0만)
            if is_main_process() and extra_state and checkpoint_path.exists():
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                ckpt.update(extra_state)
                torch.save(ckpt, checkpoint_path)
        else:
            save_hf_checkpoint(
                model=model,
                tokenizer=tokenizer,
                save_dir=checkpoint_dir / f"epoch_{current_epoch:.2f}",
                epoch=current_epoch,
                val_metrics=val_metrics,
            )

        if is_main_process():
            logger.info(f"Checkpoint saved: {checkpoint_path.name} (val_loss: {avg_val_loss:.4f})")

            if config.checkpoint.save_total_limit:
                cleanup_old_checkpoints(
                    checkpoint_dir=checkpoint_dir,
                    save_total_limit=config.checkpoint.save_total_limit,
                )

                if use_s3_upload:
                    cleanup_s3_checkpoints(
                        experiment_name=config.experiment.name,
                        save_total_limit=config.checkpoint.save_total_limit,
                    )

        next_checkpoint_epoch += save_checkpoint_every
        throughput_tracker.start_epoch()

    # 14. Final checkpoint
    if config.checkpoint.save_final:
        final_path = checkpoint_dir / "checkpoint_final.pt"

        logger.info("--- Final Validation ---")
        final_val_metrics = validate_ntp(
            model=model,
            dataloader=val_loader,
            device=device,
            weight_mode=weight_mode,
            value_model=value_model,
            td_stats_ema=td_stats_ema,
            config=config,
        )

        save_lora_only = config.checkpoint.get("save_lora_only", False) and use_lora

        if save_lora_only:
            save_hf_lora_checkpoint(
                model=model,
                optimizer=optimizer,
                checkpoint_path=final_path,
                config=config,
                epoch=current_epoch,
                val_metrics=final_val_metrics,
            )

            # TD EMA state 추가 (rank 0만)
            if is_main_process() and td_stats_ema is not None and final_path.exists():
                ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
                ckpt["td_ema_state"] = td_stats_ema.state_dict()
                torch.save(ckpt, final_path)
        else:
            save_hf_checkpoint(
                model=model,
                tokenizer=tokenizer,
                save_dir=checkpoint_dir / "final",
                epoch=current_epoch,
                val_metrics=final_val_metrics,
            )

        if is_main_process():
            logger.info(f"Final checkpoint saved: {final_path.name}")

    # 15. Cleanup
    shutdown_s3_executor()
    if is_main_process() and use_mlflow:
        mlflow.end_run()

        if use_s3_upload:
            sync_mlruns_to_s3()

    # 최신 checkpoint 경로 반환
    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    latest_checkpoint_path = str(epoch_checkpoints[-1]) if epoch_checkpoints else None

    logger.info(
        f"NTP Training ({weight_mode}) 완료! Latest checkpoint: {latest_checkpoint_path}"
    )

    final_metrics = final_val_metrics if config.checkpoint.save_final else val_metrics
    return final_metrics, latest_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified NTP Training (4 Weight Modes)")
    parser.add_argument(
        "--config",
        required=True,
        help="Config path (e.g., configs/baseline/baseline.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        dest="overrides",
        help="Config override (e.g., --override training.weight_mode=critic)",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    if args.overrides:
        from weighted_mtp.utils.config_utils import apply_overrides
        config = apply_overrides(config, args.overrides)

    run_baseline_training(config)
