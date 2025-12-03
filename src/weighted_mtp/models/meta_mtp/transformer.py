"""순수 PyTorch Meta LLaMA MTP Transformer 구현

Meta 레퍼런스 구조를 참고하되 다음을 개선:
- fairscale 의존성 제거 (순수 PyTorch)
- FSDP 호환성 확보
- device-agnostic 설계
- 학습 가능 (gradient 계산 가능)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ModelArgs:
    """Meta LLaMA MTP 모델 설정

    Meta params.json과 호환되는 구조
    """
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    n_future_tokens: int = 1
    rope_theta: float = 10000.0
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(nn.Module):
    """RMS Normalization"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """RoPE 주파수 사전 계산"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """주파수 텐서를 broadcasting을 위해 reshape"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """RoPE 적용"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA를 위한 KV heads 반복"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """Multi-head attention with GQA support"""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # KV cache (setup_caches()로 초기화)
        self.cache_k: Optional[torch.Tensor] = None
        self.cache_v: Optional[torch.Tensor] = None
        self._cache_batch_size: int = 0
        self._cache_seq_len: int = 0

    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """KV cache 사전 할당 (inference 시작 전 1회 호출)

        Args:
            max_batch_size: 최대 batch 크기 (inference 시 보통 1 또는 num_return_sequences)
            max_seq_len: 최대 시퀀스 길이
            dtype: 텐서 dtype
            device: 텐서 device
        """
        self.cache_k = torch.zeros(
            (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self.cache_v = torch.zeros(
            (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim),
            dtype=dtype,
            device=device,
        )
        self._cache_batch_size = max_batch_size
        self._cache_seq_len = max_seq_len

    def reset_caches(self) -> None:
        """KV cache 초기화 (새 시퀀스 생성 전 호출)"""
        if self.cache_k is not None:
            self.cache_k.zero_()
            self.cache_v.zero_()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Training mode: Flash Attention (KV cache 미사용)
        if self.training:
            keys = repeat_kv(xk, self.n_rep)
            values = repeat_kv(xv, self.n_rep)

            xq = xq.transpose(1, 2)
            keys = keys.transpose(1, 2)
            values = values.transpose(1, 2)

            output = F.scaled_dot_product_attention(
                xq, keys, values,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Inference mode: KV cache 유무에 따라 분기
        else:
            use_cache = self.cache_k is not None

            # Prefill (start_pos=0, seqlen>1): Flash Attention 사용
            if start_pos == 0 and seqlen > 1:
                # Cache가 setup되어 있으면 업데이트 (generation용)
                if use_cache:
                    self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
                    self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

                # Flash Attention (현재 xk, xv로 직접 계산)
                keys = repeat_kv(xk, self.n_rep)
                values = repeat_kv(xv, self.n_rep)

                xq = xq.transpose(1, 2)
                keys = keys.transpose(1, 2)
                values = values.transpose(1, 2)

                output = F.scaled_dot_product_attention(
                    xq, keys, values,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                )
                output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

            # Decode (start_pos>0 또는 seqlen=1): KV cache 필수
            else:
                # Cache가 없으면 동적 생성
                if not use_cache:
                    self.setup_caches(
                        max_batch_size=bsz,
                        max_seq_len=2048,
                        dtype=x.dtype,
                        device=x.device,
                    )

                # KV cache 업데이트
                self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
                self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

                # 캐시에서 필요한 범위 읽기
                keys = self.cache_k[:bsz, : start_pos + seqlen]
                values = self.cache_v[:bsz, : start_pos + seqlen]

                keys = repeat_kv(keys, self.n_rep)
                values = repeat_kv(values, self.n_rep)

                xq = xq.transpose(1, 2)
                keys = keys.transpose(1, 2)
                values = values.transpose(1, 2)

                scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
                if mask is not None:
                    scores = scores + mask
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                output = torch.matmul(scores, values)

                output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network"""

    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    """Transformer Block"""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    """Meta LLaMA MTP Transformer (순수 PyTorch 구현)

    Meta 구조를 참고하되 다음을 개선:
    - fairscale 제거 (순수 PyTorch nn.Linear/Embedding)
    - @torch.inference_mode() 제거 (학습 가능)
    - device-agnostic 설계
    - FSDP 호환성
    """

    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.n_future_tokens = params.n_future_tokens

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)

        # Trunk layers (n_layers - n_future_tokens + 1개)
        self.layers = nn.ModuleList()
        for layer_id in range(params.n_layers - self.n_future_tokens + 1):
            self.layers.append(TransformerBlock(layer_id, params))

        # Extra heads for MTP (n_future_tokens - 1개)
        self.extra_heads = nn.ModuleList()
        for layer_id in range(self.n_layers - self.n_future_tokens + 1, self.n_layers):
            self.extra_heads.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # RoPE freqs_cis 사전 계산
        # register_buffer 대신 일반 속성으로 저장 (safetensors 호환성)
        # complex64 타입은 safetensors 미지원, state_dict에 포함 불필요
        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            theta=params.rope_theta
        )

    def setup_caches(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """모든 Attention layer의 KV cache 사전 할당

        Inference 시작 전 호출하여 메모리를 효율적으로 관리.

        Args:
            max_batch_size: 최대 batch 크기 (batch generation 시 num_return_sequences)
            max_seq_len: 최대 시퀀스 길이 (prompt + max_new_tokens)
            dtype: 텐서 dtype
            device: 텐서 device

        Example:
            >>> model.setup_caches(
            ...     max_batch_size=20,  # Pass@20
            ...     max_seq_len=2048,
            ...     dtype=torch.float16,
            ...     device=torch.device("cuda"),
            ... )
        """
        for layer in self.layers:
            layer.attention.setup_caches(max_batch_size, max_seq_len, dtype, device)
        for layer in self.extra_heads:
            layer.attention.setup_caches(max_batch_size, max_seq_len, dtype, device)

    def reset_caches(self) -> None:
        """모든 Attention layer의 KV cache 값 초기화 (새 시퀀스 생성 전 호출)"""
        for layer in self.layers:
            layer.attention.reset_caches()
        for layer in self.extra_heads:
            layer.attention.reset_caches()

    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int = 0,
        return_all_heads: bool = False,
        return_hidden_states: bool = False,
    ):
        """Forward pass

        Args:
            tokens: [batch, seq] 입력 토큰
            start_pos: KV cache 시작 위치 (학습 시 0)
            return_all_heads: True면 모든 MTP heads 반환
            return_hidden_states: True면 hidden_states도 함께 반환 (Value Head용)

        Returns:
            return_hidden_states=True:
                (logits, hidden_states) tuple
            기본:
                logits: [batch, seq, n_future_tokens, vocab] or [batch, seq, vocab]
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)

        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(tokens.device)

        # Causal mask 생성
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        # Trunk forward (마지막 layer 제외)
        for layer in self.layers[:-1]:
            h = layer(h, start_pos, freqs_cis, mask)
        h_trunk = h

        # MTP heads forward: 모든 head의 logits를 한 번에 생성
        latents = []
        n_heads_to_use = self.n_future_tokens if return_all_heads else 1
        prediction_heads = [self.layers[-1]] + list(self.extra_heads)
        for layer in prediction_heads[:n_heads_to_use]:
            h = layer(h_trunk, start_pos, freqs_cis, mask)
            latents.append(h)

        h = torch.stack(latents, dim=-2)  # [batch, seq, n_heads_to_use, dim]
        h = self.norm(h)
        output = self.output(h)  # [batch, seq, n_heads_to_use, vocab]

        if return_hidden_states:
            hidden_states = self.norm(h_trunk.unsqueeze(-2)).squeeze(-2)
            return output, hidden_states

        return output
