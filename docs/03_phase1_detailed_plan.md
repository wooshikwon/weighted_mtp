# Phase 1: Storage 자산 변환 실행 계획

본 문서는 Phase 1에서 수행할 모델·데이터 자산 준비 작업을 **Step-by-Step**으로 정리한 실행 시나리오다. 목표는 `docs/00_ideal_structure.md`와 `docs/01_storage_preparation_plan.md`에서 정의한 구조를 그대로 구현하여, WMTP 제안서에서 요구하는 세 실험(Baseline, Verifiable Critic, Rho-1)이 재현 가능한 기반을 마련하는 것이다. **기존 자산은 모두 삭제하고, Meta 공식 모델과 데이터셋을 재다운로드하여 완전한 v2.0.0 스토리지를 구축**한다.

---

## Step 0. 사전 준비
- **목표**: 기존 다운로드 잔여물을 제거하고 깨끗한 상태에서 새 자산을 받는다.
- **작업**
  1. `uv sync` 실행, `ruff`, `black`, `pytest` 등 개발 도구 최신화.
  2. `storage/` 내 기존 모델·데이터 폴더 정리
     ```bash
     rm -rf storage/models_v2 storage/datasets_v2 storage/datasets_local_small
     rm -rf storage/models storage/datasets  # legacy 구조 사용 시
     ```
     > 이미 다운로드 중인 `storage/models_v2/meta-llama-mtp/raw/7B_1T_4` 및 `.cache` 폴더는 그대로 둔다.
  3. `scripts/verify_mtp_model.py`, `scripts/convert_*` 등 유틸리티가 최신인지 확인 (`uv run ruff check scripts/`).
- **검증**: `storage/`에 필수 골격만 남아 있는지 확인 (`tree storage -L 2`).

---

## Step 1. 디렉터리 구조 생성
- **목표**: v2 표준 레이아웃(모델/데이터/스몰셋)을 미리 만들어 후속 작업을 단순화한다.
- **작업**
  ```bash
  mkdir -p storage/models_v2/{meta-llama-mtp,ref-sheared-llama-2.7b,micro-mtp}/{raw,safetensors,configs,tokenizer}
  mkdir -p storage/datasets_v2/{codecontests,mbpp,humaneval}/{raw,processed,stats}
  mkdir -p storage/datasets_local_small/{codecontests_small,mbpp_small,humaneval_small}
  touch storage/models_v2/meta-llama-mtp/safetensors/SHA256SUMS
  ```
- **산출물**: 빈 디렉터리 구조.
- **검증**: `tree storage -L 3`가 문서와 동일한 레이아웃을 보여야 한다.

---

## Step 2. 모델 원본 다운로드 (Meta 7B_1T_4, Sheared LLaMA 2.7B)
- **목표**: Hugging Face에서 최신 모델 번들을 받아 `raw/`에 저장한다. Reward 모델은 Phase 1에서 사용하지 않는다.
- **작업**
  1. Meta MTP `7B_1T_4`:
     ```bash
     hf download facebook/multi-token-prediction 7B_1T_4/consolidated.pth \
       --local-dir storage/models_v2/meta-llama-mtp/raw
     hf download facebook/multi-token-prediction 7B_1T_4/params.json \
       --local-dir storage/models_v2/meta-llama-mtp/raw
     hf download facebook/multi-token-prediction tokenizer.model \
       --local-dir storage/models_v2/meta-llama-mtp/tokenizer
     ```
  2. Sheared LLaMA 2.7B (Rho-1 base model):
     ```bash
     hf download princeton-nlp/Sheared-LLaMA-2.7B \
       --local-dir storage/models_v2/ref-sheared-llama-2.7b/raw
     ```
     Note: princeton-nlp/Sheared-LLaMA-2.7B는 Rho-1의 base 모델이며, Meta LLaMA MTP와 동일한 tokenizer(SHA256: 9e556afd...)를 사용합니다.
  3. Reward 모델(N/A): Phase 1에서는 다운로드하지 않는다.
  4. Micro 모델: Stage 5에서 Base safetensors 생성 후 스크립트로 변환할 예정이므로 지금은 비워 둔다.
- **검증**: 각 `raw/` 디렉터리에 다운로드한 파일이 존재하는지 확인 (`ls -lh`), SHA256 계산 시작.

---

## Step 3. Meta LLaMA MTP 파생 자산 생성 (Pure PyTorch)
- **목표**: Pure PyTorch Transformer를 생성하고 safetensors로 저장한다. Meta vendor 코드는 참고용으로만 사용하며, 실제 학습에는 Pure PyTorch 재구현을 사용한다.
- **작업**
  1. **Pure PyTorch Transformer 생성 및 safetensors 저장**
     ```bash
     python - <<'PY'
     import json
     import torch
     from safetensors.torch import save_file
     from pathlib import Path
     from weighted_mtp.models.meta_mtp import ModelArgs, Transformer

     # params.json 읽기
     params_path = Path("storage/models_v2/meta-llama-mtp/raw/7B_1T_4/params.json")
     with open(params_path) as f:
         params = json.load(f)

     # ModelArgs 생성
     model_args = ModelArgs(
         dim=params["dim"],
         n_layers=params["n_layers"],
         n_heads=params["n_heads"],
         n_kv_heads=params.get("n_kv_heads"),
         vocab_size=params["vocab_size"],
         n_future_tokens=params.get("n_future_tokens", 4),
         rope_theta=params.get("rope_theta", 10000.0),
         max_seq_len=params.get("max_position_embeddings", 2048),
         norm_eps=params.get("rms_norm_eps", 1e-5),
     )

     # Pure PyTorch Transformer 생성
     transformer = Transformer(model_args)

     # Meta 원본 weights 로드 (선택적)
     raw_path = Path("storage/models_v2/meta-llama-mtp/raw/7B_1T_4/consolidated.pth")
     if raw_path.exists():
         state_dict = torch.load(raw_path, map_location="cpu")
         transformer.load_state_dict(state_dict, strict=True)

     # Safetensors 저장 (freqs_cis는 자동 제외됨)
     out_path = Path("storage/models_v2/meta-llama-mtp/safetensors/model.safetensors")
     save_file(
         transformer.state_dict(),
         str(out_path),
         metadata={"dtype": "float16", "implementation": "pure_pytorch"}
     )
     print(f"✅ Safetensors 저장 완료: {out_path}")
     print(f"  - freqs_cis는 자동 제외됨 (runtime 계산)")
     PY

     # SHA256 기록
     sha256sum storage/models_v2/meta-llama-mtp/safetensors/model.safetensors \
       > storage/models_v2/meta-llama-mtp/safetensors/SHA256SUMS
     ```

     **핵심 구현 원칙**:
     - **Pure PyTorch 구조**: `nn.Embedding`, `nn.Linear` 사용 (fairscale 제거)
     - **Gradient 계산 가능**: `@torch.inference_mode()` 제거
     - **Device-agnostic**: cuda/mps/cpu 자동 지원
     - **RoPE freqs_cis 처리**: 일반 속성으로 저장 (state_dict 미포함), forward 시 runtime 계산
     - **Safetensors 호환성**: complex64 타입 제외로 완전한 호환성 확보

  2. **Config 파일 생성**
     - `params.json` 복사 → `configs/params.json`
     - `configs/meta_adapter.yaml` 작성 (dim=4096, num_layers=32, n_future_tokens=4, intermediate_size=11008, rope_theta=10000.0, dtype=float16)
     - `tokenizer_config.json` 생성 (`model_type: sentencepiece`, `vocab_size: 32000`)

  3. **metadata.json 작성**
     ```json
     {
       "model_name": "meta-llama-mtp",
       "version": "v2.0.0",
       "implementation": "pure_pytorch",
       "n_params": 6.7e9,
       "format": "safetensors",
       "dtype": "float16",
       "source": {
         "architecture": "Meta LLaMA MTP (Pure PyTorch reimplementation)",
         "original_repo": "facebook/multi-token-prediction",
         "original_revision": "7B_1T_4",
         "note": "vendor/meta_llama/ 참고용, 실제 구현은 src/models/meta_mtp/"
       },
       "safetensors": {
         "freqs_cis_handling": "runtime_computed",
         "note": "freqs_cis는 complex64 타입(safetensors 미지원)으로 state_dict 미포함"
       },
       "distributed": {
         "fsdp_compatible": true
       }
     }
     ```

- **검증**
  - Pure PyTorch Transformer 생성 성공 (fairscale 의존성 없음)
  - Forward pass shape 정확: `[batch, seq, n_future_tokens, vocab]`
  - Gradient 계산 가능 확인
  - Safetensors 저장/로딩 정상 (freqs_cis 자동 제외)
  - Device 이동 정상 (cuda/mps/cpu)
  - FSDP wrapping 가능
  - dtype=float16, rope_theta=10000.0 확인
  - `meta_adapter.yaml` ↔ `params.json` ↔ `metadata.json` 파라미터 일치
  - Unit tests 통과

---

## Step 4. Sheared LLaMA 2.7B 파생 자산 생성
- **목표**: Sheared-LLaMA 2.7B 모델을 safetensors로 변환하고 토크나이저 공유 정보를 기록한다.
- **작업**
  1. 병합 스크립트 실행:
     ```bash
     uv run python scripts/convert_sharded_to_safetensors.py \
       storage/models_v2/ref-sheared-llama-2.7b/raw \
       storage/models_v2/ref-sheared-llama-2.7b/safetensors/model.safetensors
     ```
  2. SHA256 계산, `safetensors/SHA256SUMS`에 기록.
  3. `configs/config.json` 복사 (일관된 디렉토리 구조 유지):
     ```bash
     # setup_models.py의 sync_config가 자동으로 수행
     # 또는 수동: cp raw/reference_model/config.json configs/config.json
     ```
  4. `metadata.json` 작성 (`tokenizer_shared_with: "meta-llama-mtp"`, dtype=float16, SHA256, tokenizer SHA256 기록).
- **검증**
  - safetensors 로딩 성공.
  - `tokenizer.model` SHA256이 Meta MTP와 완전히 동일 (9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347).
  - metadata에 토크나이저 공유 여부 및 SHA256 기록.

---

## Step 5. Micro 모델 생성 (Pure PyTorch, Policy & Reference)
- **목표**: 로컬 M3 환경에서 Pure PyTorch 구조로 경량 모델을 생성하여 빠르게 검증한다.
- **작업**
  1. **Micro Policy (Pure PyTorch 경량 모델)**
     ```bash
     # Pure PyTorch 구조로 Micro 모델 재생성
     uv run python scripts/regenerate_micro_model.py

     # 또는 수동 생성:
     python - <<'PY'
     import json
     from pathlib import Path
     import torch
     from safetensors.torch import save_file
     from weighted_mtp.models.meta_mtp import ModelArgs, Transformer

     # Config 읽기
     config_path = Path("storage/models_v2/micro-mtp/configs/config.json")
     with open(config_path) as f:
         config = json.load(f)

     # ModelArgs 생성
     model_args = ModelArgs(
         dim=config["hidden_size"],
         n_layers=config["num_hidden_layers"],
         n_heads=config["num_attention_heads"],
         n_kv_heads=config["num_key_value_heads"],
         vocab_size=config["vocab_size"],
         n_future_tokens=config["n_future_tokens"],
         rope_theta=config["rope_theta"],
         max_seq_len=config["max_position_embeddings"],
         norm_eps=config["rms_norm_eps"],
     )

     # Pure PyTorch Transformer 생성 (랜덤 초기화)
     transformer = Transformer(model_args)

     # Safetensors 저장 (freqs_cis 자동 제외)
     out_path = Path("storage/models_v2/micro-mtp/safetensors/model.safetensors")
     save_file(transformer.state_dict(), str(out_path))
     print(f"✅ Micro 모델 생성 완료: {out_path}")
     PY

     # SHA256 기록
     sha256sum storage/models_v2/micro-mtp/safetensors/model.safetensors \
       > storage/models_v2/micro-mtp/safetensors/SHA256SUMS
     ```

     **Micro 모델 특징**:
     - **Pure PyTorch 구조**: Trunk + Extra heads 분리 (n_layers=4, n_future_tokens=4 → layers 1개 + extra_heads 3개)
     - **경량화**: 4-layer/512-dim, vocab 32000
     - **Safetensors 호환**: freqs_cis 자동 제외, runtime 계산
     - **Config**: `config.json`, `tokenizer/`, `metadata.json(target_device: "mps")`

  2. **Micro Reference (선택적)**
     - Sheared-LLaMA 2.7B 모델을 동일 방식으로 축소
     - Base와 동일 토크나이저 공유 (`metadata.json.tokenizer_shared_with="meta-llama-mtp"`)

- **산출물**
  - `storage/models_v2/micro-mtp/` (필수)
    - `safetensors/model.safetensors` (freqs_cis 제외)
    - `configs/config.json`
    - `tokenizer/tokenizer.model`
    - `metadata.json`
    - `safetensors/SHA256SUMS`
  - `storage/models_v2/micro-ref/` (선택적)

- **검증**
  - Pure PyTorch Transformer 생성 성공
  - Trunk + Extra heads 구조 정확 (layers 1개 + extra_heads 3개)
  - Forward pass shape: `[batch, seq, n_future_tokens, vocab]`
  - Safetensors 저장/로딩 정상 (freqs_cis runtime 계산)
  - 파일 크기 < 50MB, dtype=float16 유지
  - **Unit tests 11/11 통과**: `uv run pytest tests/unit/test_adapter.py -k micro`
  - Device 이동 정상 (mps/cpu)

---

## Step 6-8. 데이터셋 다운로드 및 전처리 (통합)
- **목표**: HuggingFace datasets 라이브러리를 사용하여 데이터셋을 다운로드하고, correct/incorrect solutions를 모두 포함한 Alpaca 형식으로 변환한다.
- **실제 구현**: `scripts/setup_datasets.py`에 다운로드, 변환, 통계 생성을 통합 구현
- **주요 특징**:
  - ✓ HuggingFace 데이터셋을 Parquet 형식으로 직접 로드
  - ✓ CodeContests validation split은 "valid"로 명명됨 (HuggingFace 원본 split 이름 사용)
  - ✓ **Correct와 incorrect solutions를 모두 처리하여 단일 JSONL에 통합**
  - ✓ **Top-level `is_correct` 필드로 솔루션 정답 여부 표시**
  - ✓ 토큰 길이 필터링 (max_tokens=2048, instruction+input+output 합산)

- **작업**
  1. **전체 데이터셋 처리** (다운로드 + 변환 + small + stats):
     ```bash
     # 전체 데이터셋 일괄 처리
     uv run python scripts/setup_datasets.py --datasets all --steps all

     # 개별 데이터셋 처리
     uv run python scripts/setup_datasets.py --datasets codecontests --steps all
     uv run python scripts/setup_datasets.py --datasets mbpp --steps all
     uv run python scripts/setup_datasets.py --datasets humaneval --steps all
     ```

  2. **데이터 변환 로직** (`scripts/setup_datasets.py`):
     - **CodeContests** (핵심 데이터셋):
       - HF 원본: `description` → `instruction`, `public_tests` → `input` (최대 2개)
       - **Correct solutions**: `solutions` 필드의 Python/Python3 솔루션 추출
         - task_id: `"{name}_correct_{idx}"` (예: `"brcktsrm_correct_0"`)
         - is_correct: `true`
       - **Incorrect solutions**: `incorrect_solutions` 필드의 Python/Python3 솔루션 추출
         - task_id: `"{name}_incorrect_{idx}"` (예: `"brcktsrm_incorrect_0"`)
         - is_correct: `false`
       - 필터링: Python only (언어 코드 1 또는 3), 토큰 길이 ≤2048
       - **실제 샘플 수** (2025-11-14, 토큰 필터링 후):
         - Train: **3,691,981 samples** (correct: 1,754,404 / incorrect: 1,937,577)
         - Valid: **14,725 samples** (correct: 8,184 / incorrect: 6,541)
         - Test: **14,851 samples** (correct: 8,038 / incorrect: 6,813)
     - **MBPP**:
       - Parquet → Alpaca (text → instruction, code → output)
       - 374 train + 90 validation + 500 test
       - is_correct 필드 없음 (모두 정답 코드)
     - **HumanEval**:
       - Parquet → Alpaca (prompt → instruction, canonical_solution → output)
       - 164 test samples
       - is_correct 필드 없음 (모두 정답 코드)

  3. **자동 생성 산출물**:
     - `processed/*.jsonl`: Alpaca 형식 JSONL
       - **필수 필드**: `instruction`, `input`, `output`, `task_id`, `is_correct` (CodeContests만)
       - **선택 필드**: `metadata` (source, difficulty, has_tests)
     - `processed/schema.json`: 데이터셋 스키마 (is_correct를 required_fields에 포함)
     - `stats/YYYY-MM-DD_summary.json`: 샘플 수, 토큰 길이 통계, is_correct 분포
     - `datasets_local_small/*_small/*.jsonl`: Small 버전 (train≤100, val/test≤32)

- **검증**:
  - ✓ Top-level `is_correct` 필드 존재 (CodeContests)
  - ✓ task_id에 `_correct_` / `_incorrect_` 접미사 포함
  - ✓ Correct와 incorrect solutions가 단일 JSONL에 통합 저장
  - ✓ 토큰 길이 필터링 적용 (2048 토큰 초과 샘플 제외)
  - ✓ Schema.json에 is_correct 필드 명시
  - ✓ Stats에 is_correct 분포 포함

---

## Step 9. 자산 무결성 검증
- **목표**: 모델과 데이터 모두에 대해 dtype/해시/스키마 검증을 완료하고 체크리스트를 채운다.
- **작업**
  1. `scripts/verify_mtp_model.py` 실행 → Meta 모델 검증.  
  2. 자체 스크립트 또는 수동으로 `models_v2/ref-sheared-llama-2.7b` SHA256, dtype 확인.  
  3. `scripts/validate_datasets.py` 결과 검토, stats 보고서 확인.  
  4. `storage/README.md`에 최신 버전, SHA256 검증 방법, 다운로드 링크 기록.
- **산출물**: 체크리스트 업데이트, README 갱신.  
- **검증**: 모든 항목 ✔️, 실패 시 원인·재작업 기록.

---

## Step 10. 문서 및 리포트 업데이트
- **목표**: Phase 1 결과를 문서화하고 Phase 2에 필요한 산출물을 정리한다.
- **작업**
  - `docs/phase1_asset_inventory.md`: 최종 자산 목록 작성.  
  - `docs/migration_notes.md`: 예상과 다른 점, 리스크, 후속 작업 기록.  
  - `docs/phase1_completion_report.md`: 일정, 산출물, 이슈, 다음 단계 요약.  
  - `storage/README.md`: 검증 명령, 버전 히스토리, FAQ 갱신.
- **검증**: 문서가 실제 자산 구조와 일치, reviewer 승인.

---

## Step 11. 완료 체크리스트 & 승인
- **목표**: Phase 1이 종료되었음을 명확히 하고 다음 Phase 착수 조건을 확인한다.
- **작업**
  - 체크리스트 항목(모델, 데이터, 스크립트, 문서) 모두 확인.  
  - 미해결 이슈는 `docs/migration_notes.md`에 Action Item으로 기록.  
  - 리뷰어/PO 승인 획득, Phase 2 착수 조건 정리.
- **검증**: 체크리스트 전항목 ✔️, 승인 서명, 백업 및 태그 완료.

---

## 병행 전략
- **모델 변환**(Step 2~5)과 **데이터 전처리 스크립트 작성**(Step 7)은 병행 가능하나, processed 실행(Step 8)은 스크립트 완료 후 진행.
- 검증 단계(Step 9)는 모델·데이터가 모두 준비된 뒤 일괄 수행.
- 문서 업데이트(Step 10)는 자산 검증 이후 착수.

---

## 위험 요소 및 대응
| 위험 | 영향 | 대응 전략 |
|------|------|-----------|
| Hugging Face 다운로드 중단 | 일정 지연 | 토큰/권한 확인, 미러링 계획 수립 |
| safetensors 변환 실패 | 모델 로딩 불가 | 변환 전 raw 백업, 변환 스크립트 재시도 |
| dtype 불일치 | 학습 오류 | 변환 즉시 dtype 검사(`scripts/verify_mtp_model.py`) |
| 데이터 분할 오염 | 평가 왜곡 | `task_id` 기준 분리, seed 고정 |
| 문서 미갱신 | 이후 Phase 혼선 | Step 10에서 최신 상태 문서화 필수 |

---

## Step 완료 체크리스트 (요약)
- [x] Step 0: 기존 자산 삭제, 환경 정비 ✓
- [x] Step 1: v2 디렉터리 구조 생성 ✓
- [x] Step 2: Meta 7B_1T_4 & Sheared 2.7B raw 다운로드 ✓
- [x] Step 3: Meta 모델 safetensors/metadata 구성, 검증 통과 ✓
- [x] Step 4: Reference 모델 변환 & 검증 ✓
- [x] Step 5: Micro 모델 생성 & 테스트 통과 ✓
- [x] Step 6-8 (통합): 데이터셋 다운로드 및 전처리 완료 ✓
  - [x] HumanEval: 164 test samples ✓
  - [x] MBPP: 374 train + 90 validation + 500 test ✓
  - [x] CodeContests: **3.7M samples** (train 3.69M + valid 14.7K + test 14.8K) ✓
  - [x] processed/ JSONL, schema.json, stats/ 생성 완료 ✓
  - [x] datasets_local_small/ 생성 완료 (train≤100, val/test≤32) ✓
- [x] Step 9: 모델·데이터 무결성 검증 완료 ✓
- [x] Step 10: 문서(README, reports) 업데이트 ✓
- [x] Step 11: 체크리스트 및 승인 완료 ✓

**Phase 1 완료** (2025-11-14):
- **모델**: 5개 모델 safetensors 변환 완료 (meta-llama-mtp, ref-sheared-llama-2.7b, starling-rm-7b, micro-mtp, micro-ref)
- **데이터**: CodeContests 3.7M (correct+incorrect 완전 통합), MBPP 964, HumanEval 164
- **Split**: train/valid/test (HuggingFace 원본 "valid" split 사용)
- **검증**: SHA256 기록, metadata.json 완비, 토큰 필터링 적용
# Phase 1: storage 자산 변환 상세 실행 계획

본 문서는 `implementation_plan.md`의 Phase 1을 step별로 세분화한 실행 계획이다. 각 step은 **목표 → 선행조건 → 작업 항목 → 산출물 → 검증 기준**으로 구성되며, 순차적으로 수행하되 병렬 가능한 작업은 명시한다.
