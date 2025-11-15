"""Meta LLaMA MTP Adapter

순수 PyTorch 구현:
- transformer.py: Meta 구조 참고, fairscale 제거, FSDP 호환
- checkpoints.py: safetensors 로딩
- value_head.py: Unbounded linear value head
- adapter.py: trunk/full forward wrapper
"""

from .adapter import MetaLlamaMTPAdapter
from .checkpoints import load_meta_mtp_model
from .transformer import ModelArgs, Transformer
from .value_head import ValueHead

__all__ = [
    "MetaLlamaMTPAdapter",
    "load_meta_mtp_model",
    "ModelArgs",
    "Transformer",
    "ValueHead",
]
