"""독립 Value Model

HuggingFace LlamaModel 기반의 완전 독립된 Value Model.
Policy Model(MTP)과 완전히 분리되어 별도 backbone 사용.

Critic 파이프라인에서 학습, Verifiable에서 eval only로 사용.
"""

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import LlamaModel, LlamaConfig

from .value_head import create_value_head, ValueHeadType


class ValueModel(nn.Module):
    """독립 Value Model

    HuggingFace LlamaModel + Value Head 구조.
    Policy Model과 완전히 독립된 별도 모델.
    LoRA를 통한 효율적인 fine-tuning 지원.

    Args:
        backbone: HuggingFace LlamaModel
        value_head: Value Head (Linear, MLP 등)
        config: LlamaConfig
    """

    def __init__(
        self,
        backbone: LlamaModel,
        value_head: ValueHeadType,
        config: LlamaConfig,
    ):
        super().__init__()
        self.backbone = backbone
        self.value_head = value_head
        self.config = config

        # LoRA 상태
        self.lora_enabled = False
        self.lora_config = None

    def apply_lora(self, lora_config: Optional[dict] = None) -> None:
        """Backbone에 LoRA 적용

        Args:
            lora_config: LoRA 설정 (None이면 기본값 사용)
                - rank: LoRA rank (기본 64)
                - alpha: scaling factor (기본 128.0)
                - dropout: dropout 확률 (기본 0.05)
                - target_modules: 적용 대상 레이어 이름 리스트
        """
        from weighted_mtp.models.lora import (
            apply_lora_to_hf_model,
            DEFAULT_HF_LORA_CONFIG,
        )

        config = {**DEFAULT_HF_LORA_CONFIG}
        if lora_config:
            config.update(lora_config)

        apply_lora_to_hf_model(self.backbone, config)

        self.lora_enabled = True
        self.lora_config = config

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        value_head_type: str = "mlp",
        dropout: float = 0.0,
        device: str = "cuda",
        dtype: str = "bfloat16",
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
    ) -> "ValueModel":
        """HuggingFace pretrained 모델에서 Value Model 생성

        Args:
            model_path: HuggingFace 모델 경로 (예: storage/models/ref-sheared-llama-2.7b/raw)
            value_head_type: "linear", "sigmoid", 또는 "mlp"
            dropout: MLP dropout (mlp 타입에만 적용)
            device: 디바이스 ("cuda", "cpu" 등)
            dtype: 데이터 타입 ("float16", "bfloat16", "float32")
            use_lora: LoRA 적용 여부
            lora_config: LoRA 설정 (use_lora=True일 때 사용)

        Returns:
            ValueModel 인스턴스
        """
        # dtype 변환
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # HuggingFace LlamaModel 로드 (SDPA 활성화)
        config = LlamaConfig.from_pretrained(model_path)
        config._attn_implementation = "sdpa"  # Config에 명시적 설정

        # 분산학습 감지: FSDP 사용 시 CPU에 로드 (FSDP가 GPU 배치 담당)
        import os
        is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

        backbone = LlamaModel.from_pretrained(
            model_path,
            config=config,  # 수정된 config 전달
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )

        # SDPA 활성화 검증 로그
        import logging
        logger = logging.getLogger(__name__)
        actual_impl = getattr(config, "_attn_implementation", "unknown")
        logger.info(f"LlamaModel loaded with attn_implementation={actual_impl}")

        # 디바이스 이동: 분산학습 시 CPU 유지 (FSDP가 GPU 배치 담당)
        # 단일 GPU에서만 직접 GPU로 이동
        if not is_distributed and device != "cpu":
            backbone = backbone.to(device)
            logger.info(f"Single GPU mode: backbone moved to {device}")
        else:
            logger.info("Distributed mode: backbone stays on CPU for FSDP sharding")

        # Value Head 생성 (분산학습 시에도 CPU에서 시작)
        hidden_size = config.hidden_size
        value_head = create_value_head(hidden_size, value_head_type, dropout)
        if not is_distributed and device != "cpu":
            value_head = value_head.to(device=device, dtype=torch_dtype)
        else:
            value_head = value_head.to(dtype=torch_dtype)  # dtype만 설정, device는 CPU

        model = cls(backbone, value_head, config)

        # LoRA 적용
        if use_lora:
            model.apply_lora(lora_config)

        return model
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = "cuda",
        base_model_path_override: Optional[str] = None,
    ) -> "ValueModel":
        """Critic checkpoint에서 Value Model 로드

        hf_lora 타입: Base model + LoRA weights + value head 로드
        full 타입: 전체 backbone + value head 로드 (하위 호환)

        Args:
            checkpoint_path: checkpoint 파일 경로
            device: 디바이스
            base_model_path_override: base 모델 경로 (지정 시 checkpoint 내 경로 대신 사용)

        Returns:
            ValueModel 인스턴스 (학습된 weights 로드됨)
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        checkpoint_type = checkpoint.get("checkpoint_type", "full")

        if checkpoint_type == "hf_lora":
            return cls._load_from_lora_checkpoint(
                checkpoint, device, base_model_path_override
            )
        else:
            return cls._load_from_full_checkpoint(checkpoint, device)

    @classmethod
    def _load_from_lora_checkpoint(
        cls,
        checkpoint: dict,
        device: str,
        base_model_path_override: Optional[str] = None,
    ) -> "ValueModel":
        """hf_lora 타입 checkpoint에서 로드

        Base model + LoRA 적용 후 학습된 weights 로드

        Args:
            checkpoint: 로드된 checkpoint dict
            device: 디바이스
            base_model_path_override: base 모델 경로 (지정 시 checkpoint 내 경로 대신 사용)
        """
        from weighted_mtp.models.lora import load_hf_lora_state_dict

        # Base model 경로: override 우선, 없으면 checkpoint에서 추출
        base_model_path = base_model_path_override or checkpoint.get("base_model_path")
        if base_model_path is None:
            raise ValueError(
                "base_model_path가 필요합니다. "
                "config에서 base_model_path를 지정하거나, checkpoint에 base_model_path가 포함되어야 합니다."
            )

        lora_config = checkpoint.get("lora_config", {})

        # Config에서 value head 설정 추출
        config_dict = checkpoint.get("config", {})
        training_config = config_dict.get("training", {})
        value_head_type = training_config.get("value_head_type", "mlp")
        dropout = training_config.get("dropout", 0.0)

        models_config = config_dict.get("models", {})
        value_model_config = models_config.get("value_model", {})
        dtype = value_model_config.get("dtype", "bfloat16")

        # Base model + LoRA 적용
        model = cls.from_pretrained(
            model_path=base_model_path,
            value_head_type=value_head_type,
            dropout=dropout,
            device=device,
            dtype=dtype,
            use_lora=True,
            lora_config=lora_config,
        )

        # LoRA weights 로드
        lora_state_dict = checkpoint.get("lora_state_dict", {})
        if lora_state_dict:
            load_hf_lora_state_dict(model.backbone, lora_state_dict)

        # Value head 로드
        value_head_state_dict = checkpoint.get("value_head_state_dict", {})
        if value_head_state_dict:
            model.value_head.load_state_dict(value_head_state_dict)

        return model

    @classmethod
    def _load_from_full_checkpoint(
        cls,
        checkpoint: dict,
        device: str,
    ) -> "ValueModel":
        """full 타입 checkpoint에서 로드 (하위 호환)

        전체 backbone + value head 로드
        """
        # Config에서 모델 설정 추출
        config_dict = checkpoint.get("config", {})

        models_config = config_dict.get("models", {})
        value_model_config = models_config.get("value_model", {})
        model_path = value_model_config.get("path")

        if model_path is None:
            raise ValueError(
                "Checkpoint에 모델 경로가 없습니다. "
                "config.models.value_model.path가 필요합니다."
            )

        # 학습 설정 추출
        training_config = config_dict.get("training", {})
        value_head_type = training_config.get("value_head_type", "mlp")
        dropout = training_config.get("dropout", 0.0)
        dtype = value_model_config.get("dtype", "bfloat16")

        # 모델 생성
        model = cls.from_pretrained(
            model_path=model_path,
            value_head_type=value_head_type,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )

        # State dict 로드
        if "backbone_state_dict" in checkpoint:
            model.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        if "value_head_state_dict" in checkpoint:
            model.value_head.load_state_dict(checkpoint["value_head_state_dict"])

        return model
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass
        
        Args:
            input_ids: [batch, seq] 입력 토큰 ID
            attention_mask: [batch, seq] 어텐션 마스크 (optional)
        
        Returns:
            value_logits: [batch, seq, 1] 토큰별 value 예측
        """
        # Backbone forward (use_cache=False로 activation checkpointing 호환성 확보)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]
        
        # Value head
        value_logits = self.value_head(hidden_states)  # [batch, seq, 1]
        
        return value_logits
    
    def freeze_backbone(self) -> None:
        """Backbone frozen 설정 (value head만 학습)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Backbone 학습 가능 설정"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def eval_mode(self) -> None:
        """Eval only 모드 (전체 frozen, eval 상태)"""
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def get_trainable_parameters(self) -> list:
        """학습 가능한 파라미터 반환"""
        return [p for p in self.parameters() if p.requires_grad]
    
    @property
    def hidden_size(self) -> int:
        """Hidden dimension 반환"""
        return self.config.hidden_size
    
    @property
    def num_layers(self) -> int:
        """Transformer layer 수 반환"""
        return self.config.num_hidden_layers

