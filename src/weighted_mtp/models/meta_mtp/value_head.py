"""Value Head 구현"""

from pathlib import Path

import torch
from torch import nn


class ValueHead(nn.Module):
    """Unbounded linear value head

    RLHF 표준: 활성화 함수 없이 Linear layer만 사용
    표현력 유지 위해 unbounded 설계

    Args:
        hidden_size: Transformer hidden dimension
        bias: Linear layer bias (기본값 False, RLHF 표준)
    """

    def __init__(self, hidden_size: int, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            hidden_states: [batch, seq, hidden_size] (Transformer norm 적용 후)

        Returns:
            value: [batch, seq, 1]
        """
        return self.linear(hidden_states)

    def save_checkpoint(self, path: Path):
        """Value head checkpoint 저장

        Args:
            path: 저장 경로 (예: "value_head.pt")
        """
        torch.save({
            "state_dict": self.state_dict(),
            "hidden_size": self.hidden_size,
        }, path)

    @classmethod
    def load_checkpoint(cls, path: Path, device: torch.device) -> "ValueHead":
        """Value head checkpoint 로드

        Args:
            path: checkpoint 경로
            device: 로딩할 device

        Returns:
            ValueHead 인스턴스
        """
        ckpt = torch.load(path, map_location=device)
        value_head = cls(hidden_size=ckpt["hidden_size"])
        value_head.load_state_dict(ckpt["state_dict"])
        return value_head.to(device)
