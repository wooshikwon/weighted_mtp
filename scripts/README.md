# Phase1 Setup Scripts

Phase1 자산 준비를 위한 통합 스크립트 (9개 → 3개로 통합)

## 📁 스크립트 구성

### 1. `setup_models.py` - 모델 설정
모델 다운로드, 변환, Micro 생성, 검증을 통합 처리

**기능:**
- HuggingFace에서 모델 다운로드
- PyTorch → SafeTensors 변환 (단일/Sharded 자동 감지)
- Micro 모델 생성 (Meta/HuggingFace 아키텍처 지원)
- MTP Config 동기화 (params.json → meta_adapter.yaml)
- 모델 무결성 검증 (SHA256, Config, SafeTensors)

**사용 예:**
```bash
# Meta MTP 전체 설정 (다운로드 → 변환 → config → micro → 검증)
uv run python scripts/setup_models.py --model meta-llama-mtp --steps all --create-micro

# Sheared-LLaMA 변환만
uv run python scripts/setup_models.py --model ref-sheared-llama --steps convert,verify

# 검증만
uv run python scripts/setup_models.py --model meta-llama-mtp --steps verify
```

**지원 모델:**
- `meta-llama-mtp`: Meta LLaMA MTP 7B_1T_4
- `ref-sheared-llama`: Sheared-LLaMA 2.7B (Rho-1 reference)
- `starling-rm`: Starling-RM-7B-alpha (선택)

---

### 2. `setup_datasets.py` - 데이터셋 설정
데이터셋 다운로드, Small 버전 생성, 검증, 통계 생성을 통합 처리

**기능:**
- HuggingFace에서 데이터셋 다운로드
- Small 버전 생성 (train_small, validation_small, test_small)
- 데이터셋 검증 (schema, 길이, 필수 필드)
- 통계 생성 (샘플 수, 평균 길이 등)

**사용 예:**
```bash
# 전체 데이터셋 설정
uv run python scripts/setup_datasets.py --datasets all --steps all

# MBPP만 small 생성
uv run python scripts/setup_datasets.py --datasets mbpp --steps small

# 검증만
uv run python scripts/setup_datasets.py --datasets all --steps validate

# 커스텀 크기 지정
uv run python scripts/setup_datasets.py --datasets codecontests --steps small --train-size 50 --val-size 16
```

**지원 데이터셋:**
- `codecontests`: DeepMind CodeContests
- `mbpp`: MBPP (Mostly Basic Python Problems)
- `humaneval`: OpenAI HumanEval

---

### 3. `verify_storage.py` - 무결성 검증
전체 storage 디렉터리 검증 및 Phase1 체크리스트 생성

**기능:**
- 모든 모델 무결성 검증 (SHA256, Config, SafeTensors)
- 모든 데이터셋 검증 (파일 존재, Schema)
- Phase1 체크리스트 자동 생성
- 검증 리포트 생성 (JSON)

**사용 예:**
```bash
# 전체 검증
uv run python scripts/verify_storage.py --check all

# 모델만 검증
uv run python scripts/verify_storage.py --check models

# Phase1 체크리스트 생성
uv run python scripts/verify_storage.py --check all --phase1-checklist

# 리포트 생성
uv run python scripts/verify_storage.py --check all --generate-report
```

---

## 🚀 Phase1 원클릭 실행

전체 Phase1 설정을 한 번에 실행:

```bash
# 1. Meta MTP 모델 설정
uv run python scripts/setup_models.py \
  --model meta-llama-mtp \
  --steps all \
  --create-micro

# 2. Sheared-LLaMA 모델 설정
uv run python scripts/setup_models.py \
  --model ref-sheared-llama \
  --steps all \
  --create-micro \
  --micro-type reference

# 3. 데이터셋 설정
uv run python scripts/setup_datasets.py \
  --datasets all \
  --steps all

# 4. 전체 검증 및 체크리스트 생성
uv run python scripts/verify_storage.py \
  --check all \
  --phase1-checklist \
  --generate-report
```

---

## 📋 통합 효과

| 구분 | Before | After | 개선 |
|------|--------|-------|------|
| 스크립트 개수 | 9개 | 3개 | 67% 감소 |
| 중복 코드 | SHA256 4회, 검증 4회 | 각 1회 | 중복 제거 |
| 누락 기능 | 다운로드 없음 | 모두 구현 | Phase1 완성 |
| 유지보수 | 분산된 로직 | 통합된 구조 | 유지보수 용이 |

---

## ⚠️ 주의사항

1. **HuggingFace 토큰**: 다운로드 전에 `HF_TOKEN` 환경변수 설정 필요
   ```bash
   export HF_TOKEN=hf_...
   ```

2. **디스크 용량**: 모델 다운로드 시 약 50GB 필요

3. **실행 순서**: 모델 → 데이터셋 → 검증 순서 권장

4. **재실행**: 모든 스크립트는 멱등성(idempotent) 보장
   - 이미 존재하는 파일은 건너뛰거나 덮어쓰기

---

## 🗑️ 삭제된 스크립트 (참고용)

다음 9개 스크립트가 위 3개 통합 스크립트로 대체되었습니다:

1. `setup_mtp_model.sh` → `setup_models.py`
2. `convert_mtp_to_safetensors.py` → `setup_models.py`
3. `convert_sharded_to_safetensors.py` → `setup_models.py`
4. `convert_pytorch_to_safetensors.py` → `setup_models.py`
5. `sync_mtp_config.py` → `setup_models.py`
6. `prepare_local_small_model.py` → `setup_models.py`
7. `prepare_micro_reference.py` → `setup_models.py`
8. `prepare_dataset.py` → `setup_datasets.py`
9. `verify_mtp_model.py` → `verify_storage.py`

---

## 📖 추가 문서

- Phase1 상세 계획: `docs/03_phase1_detailed_plan.md`
- Storage 구조: `docs/01_storage_preparation_plan.md`
- 이상적 구조: `docs/00_ideal_structure.md`
