"""MTP Generation 유틸리티

MTP 모델의 autoregressive text generation 지원 (KV cache 활용)

최적화된 LLM inference:
1. Batch Generation: N개 샘플을 병렬로 생성 (Pass@K 평가 속도 N배 향상)
2. Flash Attention: Prefill 시 scaled_dot_product_attention 사용
3. 동적 KV cache: 필요한 만큼만 메모리 할당
"""

from typing import Any

import torch
import torch.nn.functional as F

from weighted_mtp.models.meta_mtp import MetaLlamaMTPAdapter


def _sample_next_token_batch(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Batch 토큰 샘플링 (temperature scaling + nucleus sampling)

    Args:
        logits: [batch, vocab] 형태의 logits
        temperature: sampling temperature (0=greedy)
        top_p: nucleus sampling threshold

    Returns:
        next_tokens: [batch, 1] 형태의 토큰 IDs
    """
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Top-p filtering
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = False

    # Batch-wise masking (vectorized)
    logits_masked = logits.clone()
    for batch_idx in range(logits.shape[0]):
        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
        logits_masked[batch_idx, indices_to_remove] = float('-inf')

    probs = F.softmax(logits_masked, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_with_mtp(
    model: MetaLlamaMTPAdapter,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 0.95,
    num_return_sequences: int = 1,
    device: torch.device = torch.device("cpu"),
) -> list[str]:
    """MTP 모델로 batch autoregressive generation (KV cache 활용)

    최적화된 batch generation:
    1. N개 샘플을 동시에 처리 (KV cache를 batch 차원으로 확장)
    2. Prefill: Flash Attention으로 빠른 prompt 처리
    3. Decode: 각 샘플별 EOS 추적, 완료된 샘플은 pad 토큰으로 처리

    Pass@K 평가에서 num_return_sequences=K로 설정하면 ~K배 속도 향상.

    Args:
        model: MetaLlamaMTPAdapter (eval mode 권장)
        tokenizer: HuggingFace AutoTokenizer
        prompt: 생성 프롬프트
        max_new_tokens: 최대 생성 토큰 수
        temperature: Sampling temperature (0=greedy, >0=sampling)
        top_p: Nucleus sampling threshold
        num_return_sequences: 생성할 시퀀스 개수 (Pass@K용, batch 처리)
        device: 디바이스

    Returns:
        생성된 텍스트 리스트 (길이 num_return_sequences)
    """
    model.eval()
    batch_size = num_return_sequences

    # Tokenize prompt
    input_ids_single = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_len = input_ids_single.shape[1]

    # Batch로 확장: [1, seq] -> [batch_size, seq]
    input_ids = input_ids_single.expand(batch_size, -1).contiguous()

    # KV cache 설정 (batch_size에 맞게)
    max_seq_len = prompt_len + max_new_tokens
    sample_tensor = next(model.parameters())
    model.transformer.setup_caches(
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        dtype=sample_tensor.dtype,
        device=device,
    )
    model.transformer.reset_caches()

    # EOS 토큰 ID 및 완료 추적
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # 1. Prefill: 전체 prompt 처리 (batch)
    with torch.no_grad():
        logits = model.transformer(input_ids, start_pos=0, return_all_heads=False)
        logits = logits.squeeze(-2)  # [batch, seq, vocab]

    # 마지막 토큰의 logits로 첫 번째 새 토큰 생성
    next_token_logits = logits[:, -1, :]
    next_tokens = _sample_next_token_batch(next_token_logits, temperature, top_p)  # [batch, 1]

    # 생성된 토큰 저장
    generated_tokens = [next_tokens]
    current_pos = prompt_len

    # EOS 체크
    finished = finished | (next_tokens.squeeze(-1) == eos_token_id)

    # 2. Decode: 토큰별 batch 생성, KV cache 활용
    for _ in range(max_new_tokens - 1):
        if finished.all():
            break

        with torch.no_grad():
            # 완료된 샘플은 pad 토큰으로 처리 (KV cache 일관성 유지)
            tokens_to_process = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_tokens, pad_token_id),
                next_tokens,
            )

            # Batch forward (seqlen=1), KV cache 활용
            logits = model.transformer(tokens_to_process, start_pos=current_pos, return_all_heads=False)
            logits = logits.squeeze(-2)  # [batch, 1, vocab]

        next_token_logits = logits[:, -1, :]
        next_tokens = _sample_next_token_batch(next_token_logits, temperature, top_p)

        generated_tokens.append(next_tokens)
        current_pos += 1

        # EOS 체크 (아직 완료되지 않은 샘플만)
        finished = finished | (next_tokens.squeeze(-1) == eos_token_id)

    # 3. Decode: 각 샘플별로 텍스트 생성
    all_generated_tokens = torch.cat(generated_tokens, dim=-1)  # [batch, num_generated]
    all_tokens = torch.cat([input_ids, all_generated_tokens], dim=-1)  # [batch, total_len]

    generated_texts = []
    for batch_idx in range(batch_size):
        # EOS까지만 디코드 (EOS 포함)
        tokens = all_tokens[batch_idx]
        eos_positions = (tokens == eos_token_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            # prompt 이후의 첫 EOS 찾기
            eos_after_prompt = eos_positions[eos_positions >= prompt_len]
            if len(eos_after_prompt) > 0:
                tokens = tokens[:eos_after_prompt[0] + 1]

        generated_text = tokenizer.decode(tokens[prompt_len:], skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts


# Legacy 호환성을 위한 별칭
def _reset_kv_cache(model: MetaLlamaMTPAdapter) -> None:
    """KV cache 초기화 (Legacy 호환성)

    Deprecated: model.transformer.reset_caches() 사용 권장
    """
    model.transformer.reset_caches()
