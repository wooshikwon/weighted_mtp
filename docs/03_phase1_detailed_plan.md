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
  2. Sheared LLaMA 2.7B (Rho-1 reference):
     ```bash
     hf download microsoft/rho-1 reference_model/pytorch_model.bin.index.json \
       --local-dir storage/models_v2/ref-sheared-llama-2.7b/raw
     hf download microsoft/rho-1 reference_model/pytorch_model-00001-of-00002.bin \
       --local-dir storage/models_v2/ref-sheared-llama-2.7b/raw
     hf download microsoft/rho-1 reference_model/pytorch_model-00002-of-00002.bin \
       --local-dir storage/models_v2/ref-sheared-llama-2.7b/raw
     hf download microsoft/rho-1 tokenizer/tokenizer.model \
       --local-dir storage/models_v2/ref-sheared-llama-2.7b/tokenizer
     hf download microsoft/rho-1 tokenizer/tokenizer.json \
       --local-dir storage/models_v2/ref-sheared-llama-2.7b/tokenizer
     ```
  3. Reward 모델(N/A): Phase 1에서는 다운로드하지 않는다.
  4. Micro 모델: Stage 5에서 Base safetensors 생성 후 스크립트로 변환할 예정이므로 지금은 비워 둔다.
- **검증**: 각 `raw/` 디렉터리에 다운로드한 파일이 존재하는지 확인 (`ls -lh`), SHA256 계산 시작.

---

## Step 3. Meta LLaMA MTP 파생 자산 생성
- **목표**: `consolidated.pth`를 프로젝트 표준 구조로 변환하고 검증한다.
- **작업**
  1. safetensors 변환 및 SHA256 기록
     ```bash
     python - <<'PY'
     import torch
     from safetensors.torch import save_file, safe_open
     from pathlib import Path

     raw = Path("storage/models_v2/meta-llama-mtp/raw/7B_1T_4/consolidated.pth")
     out = Path("storage/models_v2/meta-llama-mtp/safetensors/model.safetensors")
     state_dict = torch.load(raw, map_location="cpu")
     save_file(state_dict, out, metadata={"dtype": "float16"})
     PY
     sha256sum storage/models_v2/meta-llama-mtp/safetensors/model.safetensors \
       > storage/models_v2/meta-llama-mtp/safetensors/SHA256SUMS
     ```
  2. `params.json` 복사 → `configs/params.json` (필요 시 추가 필드 포함).
  3. `configs/meta_adapter.yaml` 작성 (dim=4096, num_layers=32, n_future_tokens=4, intermediate_size=11008, rope_theta=10000.0, dtype=float16).
  4. `tokenizer_config.json` 생성 (`model_type: sentencepiece`, `vocab_size: 32000` 등).
  5. `metadata.json` 작성 (dtype, SHA256, 원본 repo, 생성 일자 기록).
- **검증**
  - `scripts/verify_mtp_model.py` 실행 → 모든 체크 통과.  
  - dtype=float16, rope_theta=10000.0 확인.  
  - `meta_adapter.yaml` ↔ `params.json` ↔ `metadata.json` 파라미터가 모두 일치.

---

## Step 4. Sheared LLaMA 2.7B 파생 자산 생성
- **목표**: Rho-1 reference 모델을 safetensors로 변환하고 토크나이저 공유 정보를 기록한다.
- **작업**
  1. 병합 스크립트 실행:
     ```bash
     uv run python scripts/convert_sharded_to_safetensors.py \
       storage/models_v2/ref-sheared-llama-2.7b/raw \
       storage/models_v2/ref-sheared-llama-2.7b/safetensors/model.safetensors
     ```
  2. SHA256 계산, `safetensors/SHA256SUMS`에 기록.  
  3. `configs/config.json` 복사 (원본에서 사용).  
  4. `metadata.json` 작성 (`tokenizer_shared_with: "meta-llama-mtp"`, dtype=float16, SHA256 등).
- **검증**
  - safetensors 로딩 성공.  
  - `tokenizer.model` vocab_size가 Base와 동일(32000).  
  - metadata에 토크나이저 공유 여부 기록.

---

## Step 5. Micro 모델 생성 (Policy & Reference)
- **목표**: 로컬 M3 환경에서 Baseline과 Rho-1 실험을 빠르게 검증할 수 있도록 정책/레퍼런스용 경량 safetensors를 준비한다.
- **작업**
  1. **Micro Policy (Meta MTP 축소판)**
     ```bash
     uv run python scripts/prepare_local_small_model.py \
       --source storage/models_v2/meta-llama-mtp/safetensors/model.safetensors \
       --target storage/models_v2/micro-mtp
     sha256sum storage/models_v2/micro-mtp/safetensors/model.safetensors \
       > storage/models_v2/micro-mtp/safetensors/SHA256SUMS
     ```
     - 출력: 4-layer/512-dim, vocab 8000 모델.  
     - 구성 파일: `micro-mtp/configs/config.json`, `tokenizer/tokenizer.model`, `metadata.json(target_device: "mps", dtype: float16)`.
  2. **Micro Reference (선택)**
     - Rho-1 비교를 로컬에서 실험하기 위해 Sheared 2.7B 모델을 동일 방식으로 축소한다.  
     - 스크립트 예시(향후 `prepare_micro_reference.py`로 분리 가능):
       ```bash
       uv run python scripts/prepare_micro_reference.py \
         --source storage/models_v2/ref-sheared-llama-2.7b/safetensors/model.safetensors \
         --target storage/models_v2/micro-ref
       sha256sum storage/models_v2/micro-ref/safetensors/model.safetensors \
         > storage/models_v2/micro-ref/safetensors/SHA256SUMS
       ```
       - 최소 4-layer, hidden_size 512 수준으로 축소하고, Base와 동일 토크나이저를 공유하도록 `metadata.json.tokenizer_shared_with="meta-llama-mtp"` 기록.
- **산출물**
  - `storage/models_v2/micro-mtp/` (필수)  
  - `storage/models_v2/micro-ref/` (필요 시)  
  - 각 디렉터리의 safetensors/config/tokenizer/metadata/SHA256SUMS.
- **검증**
  - 파일 크기 < 50MB, dtype=float16 유지.  
  - `uv run pytest tests/unit/test_adapter.py -k micro` 통과.  
  - (Micro reference 사용 시) Rho-1 비교 스크립트와 토크나이저 호환성 확인.

---

## Step 6. 데이터셋 원본 다운로드 및 정리
- **목표**: CodeContests/MBPP/HumanEval raw JSONL을 최신 버전으로 받아 `datasets_v2/.../raw/`에 저장한다.
- **작업**
  ```bash
  # 예시: CodeContests (누락 시 wget/rsync 등 이용)
  hf download deepmind/code_contests code_contests.jsonl \
    --local-dir storage/datasets_v2/codecontests/raw
  hf download google-research-datasets/mbpp data/mbpp.jsonl \
    --local-dir storage/datasets_v2/mbpp/raw
  hf download openai/humaneval data/HumanEval.jsonl \
    --local-dir storage/datasets_v2/humaneval/raw
  ```
  - 기존 로컬 파일이 있다면 삭제/백업 후 새로 다운로드한다.
  - 다운로드 후 SHA256 기록, `stats/raw_checksums.txt` 생성.
- **검증**: raw JSONL 존재, SHA256 기록 완료.

---

## Step 7. 데이터 전처리 스크립트 구현
- **목표**: raw 데이터를 Alpaca 스타일 processed 데이터로 변환하는 파이프라인을 완성한다.
- **작업**
  - `src/data/prepare.py`:  
    ```
    raw JSONL -> prompt/response/metadata 생성 (Instruction/Input/Evaluation Notes 템플릿)
                -> train/validation/test 분할
                -> schema.json, stats JSON 저장
    ```
    - prompt는 CodeContests 설명+입력 요약, response는 정답 코드+EOS.  
    - metadata는 `{"task_id": ..., "source": ..., "is_correct": true/false, "has_tests": true/false}`.  
    - 2048 토큰 초과 시 설명 축약/코드 truncate 후 재검증.  
    - `seed` 고정으로 분할 재현성 보장.
  - `scripts/validate_datasets.py`: schema 검증, 토큰 길이 체크, SHA256 비교, 분할 중복 검사.
  - `src/data/analyze_dataset.py`(옵션): 통계 수집.
- **산출물**: 전처리/검증 스크립트, README 참고 섹션.
- **검증**: `uv run python -m weighted_mtp.data.prepare ...` dry-run 성공, lint/test 통과.

---

## Step 8. 데이터 전처리 실행 & stats 생성
- **목표**: Step 7 스크립트를 이용해 실제 processed 데이터를 생성한다.
- **작업**
  1. CodeContests 변환
     ```bash
     uv run python -m weighted_mtp.data.prepare \
       --dataset codecontests \
       --source-dir storage/datasets_v2/codecontests/raw \
       --output-dir storage/datasets_v2/codecontests/processed \
       --max-length 2048 --seed 42
     ```
  2. MBPP, HumanEval 동일 실행.  
  3. `scripts/validate_datasets.py`로 모든 processed JSONL 검증.  
  4. `datasets_local_small` 생성 (train_small≤100, validation_small≤32).
- **산출물**: `processed/*.jsonl`, `schema.json`, `stats/YYYY-MM-DD_summary.json`, `datasets_local_small/`.  
- **검증**: 2048 토큰 초과 없음, `is_correct` 존재, Schema 일치, 스몰셋 조건 충족, **train/validation/test 분할이 `task_id` 기준으로 상호 배타적임을 확인**.

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
- [ ] Step 0: 기존 자산 삭제, 환경 정비
- [ ] Step 1: v2 디렉터리 구조 생성
- [ ] Step 2: Meta 7B_1T_4 & Sheared 2.7B raw 다운로드
- [ ] Step 3: Meta 모델 safetensors/metadata 구성, 검증 통과
- [ ] Step 4: Reference 모델 변환 & 검증
- [ ] Step 5: Micro 모델 생성 & 테스트 통과
- [ ] Step 6: CodeContests/MBPP/HumanEval raw 최신화
- [ ] Step 7: 전처리/검증 스크립트 구현
- [ ] Step 8: processed 데이터 생성, stats·스몰셋 완료
- [ ] Step 9: 모델·데이터 무결성 검증 완료
- [ ] Step 10: 문서(README, reports) 업데이트
- [ ] Step 11: 체크리스트 및 승인 완료

Phase 1 완료 시 `storage/`는 v2.0.0 품질 기준을 만족하며, Phase 2(코드 스켈레톤) 작업에 즉시 착수할 수 있다.
# Phase 1: storage 자산 변환 상세 실행 계획

본 문서는 `implementation_plan.md`의 Phase 1을 step별로 세분화한 실행 계획이다. 각 step은 **목표 → 선행조건 → 작업 항목 → 산출물 → 검증 기준**으로 구성되며, 순차적으로 수행하되 병렬 가능한 작업은 명시한다.
