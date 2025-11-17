# WMTP Implementation Roadmap

본 문서는 `docs/00_ideal_structure.md`와 `docs/01_storage_preparation_plan.md`에서 합의한 목표 구조를 실제 구현 순서로 풀어낸 실행 계획이다. 목적은 `docs/wmtp_research_proposal.md`에 정의된 세 가지 WMTP 실험(Baseline, Verifiable Critic, Rho-1 Weighted)이 재현 가능한 형태로 완성되도록 단계별 수행 항목, 산출물, 검증 조건을 명확히 하는 데 있다. 각 Phase는 선행 조건 충족 시 착수하며, 산출물이 승인되지 않으면 다음 단계로 넘어가지 않는다.

---

## 전체 Phase 개요

| Phase | 명칭 | 핵심 목표 | 주요 선행 조건 | 대표 산출물 |
|-------|------|------------|----------------|-------------|
| P0 | 프로젝트 킥오프 & 환경 정비 | 문서 정합성·용어·역할 정리 | 없음 | 용어 사전, 일정, 책임 매트릭스 |
| P1 | 모델·데이터 자산 확보 | `storage/` 표준 구조로 원본 수집·변환 | P0 | `models_v2/`, `datasets_v2/` v2.0.0 |
| P2 | 코드 스켈레톤 & 벤더 정리 | `vendor/meta_llama/` 및 `src/` 골격 구축 | P1 | 스켈레톤 코드, `pyproject.toml` 갱신 |
| P3 | 데이터 파이프라인 구현 | Raw → Alpaca 변환, 검증 유틸 완성 | P1 | `src/data/prepare.py`, `scripts/validate_datasets.py` |
| P4 | Meta Adapter 통합 | safetensors 로딩/forward 경로 확립 | P2 | `MetaLlamaMTPAdapter`, unit test |
| P5 | Value Weighting 모듈 | TD error, weight builder, metrics 구현 | P2 | `src/value_weighting/*`, 테스트 |
| P6 | 학습 파이프라인 Stage 0~3 | 환경 초기화·Stage1/2·MLflow 로깅 | P3, P4, P5 | `pipelines/training.py`, 통합 테스트 |
| P7 | 평가·분석 파이프라인 | Rho-1 비교 및 인퍼런스 루틴 마련 | P6 | `pipelines/evaluation.py`, Pass@K 리포트 |
| P8 | Config & CLI 체계 | defaults/recipes, CLI preset 완성 | P2, P6 | `core/config.py`, `cli/train.py`, recipes |
| P9 | 테스트 & 품질 게이트 | unit/integration/CI 파이프라인 구축 | P4~P8 | pytest suite, CI 워크플로우 |
| P10 | 배포 & 실험 운영 | VESSL 연동, MLflow/스토리지 업로드 | P1, P6, P8 | `scripts/sync_to_vessl_storage.py`, 배포 가이드 |
| P11 | 문서화 & 인수 | 문서 업데이트, 최종 체크리스트 | 전체 | 최신 문서/체인지로그/릴리즈 노트 |

---

## Phase 상세

### P0. 프로젝트 킥오프 & 환경 정비
- **목표**  
  - 모든 참여자가 설계 문서(00·01·제안서)를 이해하고 용어 정의를 통일한다.
  - 일정, 역할 분담, 브랜치 전략, 코드 리뷰 정책을 확립한다.
- **주요 활동**  
  - 문서 리뷰 & 이슈 목록화, `docs/glossary.md` 작성.  
  - 개발 환경 세팅(uv/ruff/pre-commit), 비밀키 관리 정책 확정.  
  - 작업 추적 보드 구성(Jira/Linear 등).
- **산출물**: 용어 사전, 일정표, 책임 매트릭스, 환경 세팅 가이드.  
- **검증 기준**: 결정 사항을 문서화하고 P1 착수 체크리스트 승인.

### P1. 모델·데이터 자산 확보 (`storage/` v2.0.0)
- **목표**
  - Meta `7B_1T_4` 번들, Rho-1 reference, 소형 테스트 모델을 `storage/models_v2/` 표준 구조로 정리한다.
  - CodeContests/MBPP/HumanEval raw 자산을 `storage/datasets_v2/`에 수집하고 SHA256 기록을 남긴다.
  - **메타데이터 추출**: 메모리 효율적 학습을 위해 `is_correct`, `difficulty` 정보만 별도 파일로 추출한다.
- **주요 활동**
  1. Hugging Face에서 `consolidated.pth`, `params.json`, `tokenizer.model` 다운로드 → `raw/` 저장.
  2. safetensors 변환, `configs/params.json` 복사, `meta_adapter.yaml` 작성, `metadata.json` 갱신.
  3. CodeContests correct/incorrect solutions 통합 JSONL 생성 (`instruction/input/output/task_id/is_correct/metadata` 포함), SHA256 계산.
  4. **메타데이터 추출**: `scripts/extract_metadata.py`로 각 데이터셋의 `*_metadata.json` 생성 (is_correct, difficulty만 포함)
  5. `storage/README.md` 업데이트.
- **실제 성과** (2025-11-14):
  - **모델**: 5개 모델 (meta-llama-mtp 6.7B, ref-sheared-llama-2.7b, starling-rm-7b 13.3B, micro-mtp, micro-ref)
  - **데이터**: CodeContests **3.7M samples** (train 3.69M, valid 14.7K, test 14.8K), MBPP 964, HumanEval 164
  - **Split**: train/valid/test (HuggingFace 원본 "valid" split 사용)
  - **메타데이터**: 10개 메타데이터 파일 생성 (codecontests, mbpp, humaneval - train/valid/test)
    - 크기: 전체 데이터(~15GB) 대비 ~217MB (99% 메모리 절감)
    - 구조: `{"metadata": [{"is_correct": bool, "difficulty": int}, ...], "stats": {...}}`
- **산출물**: `models_v2/` 및 `datasets_v2/` 디렉터리, `*_metadata.json` 파일들, SHA256 로그, 업데이트된 README.
- **검증 기준**: dtype(float16) 유지, 토크나이저 공유 여부 기재, split 누락 없음, 메타데이터 파일 검증 완료, 체크리스트 서명.

### P2. 코드 스켈레톤 & 벤더 정리
- **목표**  
  - `vendor/meta_llama/`에 Meta 레퍼런스 코드( `llama/*.py` )를 옮기고, `src/` 모듈 골격을 생성한다.
- **주요 활동**  
  - vendor 패키지 초기화(`__init__.py`), mypy-friendly type stub 정리.
  - `src/` 하위 디렉터리/`__init__.py` 생성, 인터페이스 스텁 작성.
  - `pyproject.toml`, `ruff.toml`, pre-commit 훅 구성.
  - `configs/defaults.yaml`, recipe 초안 마련(값은 placeholder 가능).
- **실제 성과** (2025-11-14):
  - HuggingFace에서 Meta 레퍼런스 코드 다운로드 → `vendor/meta_llama/` 배치
  - 8개 src 모듈 스켈레톤, CLI --dry-run 동작, 7개 unit test 통과
- **산출물**: 스켈레톤 코드, vendor 패키지, 기본 설정 파일.
- **검증 기준**: `uv run python -c "from vendor.meta_llama import Transformer"` 성공, lint/format 통과.

### P3. 데이터 파이프라인 구현
- **목표**
  - 전처리된 JSONL을 학습용 PyTorch Dataset으로 **메타데이터 기반 효율적 로딩**한다.
  - **Loss masking collator 구현**: instruction/input 토큰은 학습 제외, output 토큰만 학습 대상
  - **Stage별 샘플링 전략**: Stage 1/2에 맞는 데이터 효율적 로딩 (메모리 99% 절감)
  - **분산학습 런타임 모듈**: A100 4-GPU 환경을 위한 분산학습 초기화 및 환경 설정
- **주요 활동**
  - `src/data/datasets.py`: **메타데이터 기반 JSONL → HuggingFace Dataset 로딩** + **Stage별 샘플링**
    - **핵심 혁신 - 메타데이터 기반 로딩** (99% 메모리 절감):
      1. `_load_metadata()`: `*_metadata.json` 로드 (is_correct, difficulty만 포함, ~217MB)
      2. `_compute_sampling_indices_from_metadata()`: Config 기반으로 샘플링 인덱스 계산 (Stage별 전략 적용)
      3. `_read_jsonl_by_indices()`: JSONL 파일에서 계산된 인덱스의 라인만 선택적으로 읽기
      4. HuggingFace Dataset으로 변환
    - 기존 함수 제거: `_sample_stage1()`, `_sample_stage2()`, `apply_stage_sampling()`, `use_small` 파라미터
    - **Stage 1 샘플링**: `is_correct` 균형 (50:50), 전체 난이도, n_samples=10K~50K
    - **Stage 2 샘플링**: Curriculum Learning (difficulty 기반 점진적 증가), n_samples=100K~500K
      - 초반 epoch (0~30%): low (1-3) 70%, medium (4-7) 30%, high (8-11) 0%
      - 중반 epoch (30~70%): low 30%, medium 60%, high 10%
      - 후반 epoch (70~100%): low 10%, medium 50%, high 40%
    - DatasetDict 구성
  - `src/data/collators.py`: **Instruction/Input masking collator** 구현
    - Alpaca 형식 (instruction/input/output) 파싱
    - Tokenizer로 instruction 길이 추적 → labels에 -100 설정
    - Input 길이 추적 → labels에 -100 설정
    - Output만 실제 token ID 유지 (loss 계산 대상)
    - attention_mask는 모든 토큰 포함 (전체 context 활용)
    - n_future_tokens 대응 (MTP 헤드용)
  - `src/runtime/distributed.py`: **분산학습 초기화 및 유틸리티**
    - torch.distributed 초기화 (NCCL backend)
    - Rank/World size 조회 함수 (get_rank, get_world_size, get_local_rank)
    - 분산 환경 확인 (is_distributed, is_main_process)
    - DistributedSampler 생성 헬퍼 (create_distributed_sampler)
    - 동기화 및 정리 (barrier, cleanup_distributed)
    - FSDP 설정 헬퍼 (setup_fsdp_config, Phase 6에서 사용)
  - `src/runtime/environment.py`: **Rank-aware 환경 설정**
    - Rank별 독립 seed 설정 (base_seed + rank)
    - GPU 디바이스 할당 (cuda:{rank}, mps, cpu)
    - PyTorch backends 최적화 (cuDNN, TF32)
    - 통합 환경 설정 함수 (setup_environment)
    - GPU 메모리 모니터링 (get_gpu_memory_info)
- **데이터셋 규모** (실제):
  - CodeContests: train 3.69M, valid 14.7K, test 14.8K (correct + incorrect 통합)
  - **Difficulty 분포**: diff=7 (86.7%), diff=2 (6.4%), diff=1 (4.4%), diff=11 (2.1%), diff=6 (0.4%)
  - MBPP: train 374, validation 90, test 500
  - HumanEval: test 164
- **실제 성과** (2025-11-14):
  - **datasets.py 완전 재작성**: 893 lines → 557 lines (38% 코드 감소)
  - **메모리 효율**:
    - Stage 1 (50K): 메타데이터(~217MB) + 샘플(~200MB) = **~417MB** (기존 15GB 대비 97% 절감)
    - Stage 2 (200K): 메타데이터(~217MB) + 샘플(~800MB) = **~1GB** (기존 15GB 대비 93% 절감)
  - **메타데이터 기반 로딩 함수**:
    - `_load_metadata()`: 메타데이터 파일 로드
    - `_compute_sampling_indices_from_metadata()`: 샘플링 인덱스 계산 (Stage별 전략)
    - `_read_jsonl_by_indices()`: JSONL에서 해당 라인만 선택적 읽기
  - **테스트 통과**: 33 passed, 3 skipped (호환성 100%)
- **산출물**:
  - src/data/ 모듈 (datasets.py, collators.py)
  - src/runtime/ 모듈 (distributed.py, environment.py, __init__.py)
  - unit tests (test_datasets.py, test_collators.py)
  - integration tests (test_data_pipeline.py - DistributedSampler 사용 예시 포함)
- **검증 기준**
  - 메타데이터 기반 JSONL 로딩 및 DatasetDict 생성 성공
  - Collator가 instruction/input을 -100으로 마스킹 (unit test)
  - Output 토큰만 loss 계산 확인
  - Stage별 샘플링 분포 검증 (integration test)
  - 분산학습 모듈 import 성공 (로컬/분산 환경 자동 감지)
  - DistributedSampler가 로컬에서는 None 반환, 분산 환경에서는 데이터 자동 분할
  - 메모리 사용량 목표 달성 (<1GB for Stage 2)

### P4. Meta Adapter 통합
- **목표**
  - `MetaLlamaMTPAdapter`가 safetensors/params/json 조합을 로딩해 trunk/full forward를 제공하도록 구현한다.
  - **`from_pretrained()` Classmethod 구현**: 모델 로딩을 통합하고 Stage별 Value Head 초기화를 제어한다.
- **주요 활동**
  - `src/models/meta_mtp/adapter.py`:
    - `from_pretrained(model_path, device, dtype, initialize_value_head)` classmethod 구현
    - Transformer 로딩: `checkpoints.load_meta_mtp_model()` 호출
    - ModelArgs 파싱: params.json 또는 config.json 자동 감지
    - **Value Head 선택적 초기화**:
      - `initialize_value_head=True`: Critic/Verifiable Stage용 (기본값)
      - `initialize_value_head=False`: Rho-1 Stage용 (Value head 불필요)
    - trunk_forward/full_forward 메서드 구현
  - `src/models/meta_mtp/checkpoints.py`: safetensors 로딩, dtype 변환, 장치 선택.
  - micro 모델을 사용한 unit test (`tests/unit/test_adapter.py`).
  - 오류/로그 정책 정리.
- **산출물**: Adapter 모듈 (from_pretrained() 포함), 체크포인트 유틸, unit test.
- **검증 기준**:
  - micro 모델 trunk_forward < 2s, dtype & shape 검증
  - `initialize_value_head=True` 시 adapter.value_head 존재 확인
  - `initialize_value_head=False` 시 adapter.value_head is None 확인
  - `pytest -k adapter` 통과

### P5. Value Weighting 모듈
- **목표**  
  - Verifiable Critic/Rho-1 Weighted 실험에 필요한 TD error 계산, weight builder, metrics를 모듈화한다.
- **주요 활동**  
  - `td_error.py`: advantage 계산 (GAE, Z-score), dtype 안전성 확보.  
  - `weight_builder.py`: temperature softmax, clipping, entropy floor.  
  - `metrics.py`: TD error mean/std, weight entropy, KL 모니터링.  
  - (옵션) `regularizers.py`: critic drift 방지 로직.  
  - 단위 테스트: zero reward, extreme reward, masking 케이스.
- **산출물**: value_weighting 패키지.  
- **검증 기준**: 모든 unit test 통과, 수치 비교(참조 스크립트)에서 오차 허용 범위 내.

### P6. 독립 Pipeline 구현 및 CLI 연결
- **목표**
  - 4개 독립 Pipeline 구현: Baseline, Critic, Verifiable, Rho-1
  - CLI가 config의 `experiment.stage` 필드로 pipeline 라우팅
  - **A100 4-GPU 분산학습**: DDP/FSDP 기반 모델 분산, DistributedSampler 기반 데이터 분산
  - **Verifiable Critic Continual Learning**: Value loss를 auxiliary loss로 추가하여 policy 학습 중 critic도 지속 학습
  - **Reference 모델 전략**: Rho-1 Pipeline에서 HuggingFace `AutoModelForCausalLM` 직접 사용 (Custom wrapper 불필요)
- **주요 활동**
  - **독립 Pipeline 구현**:
    - `run_baseline.py`: 균등 가중치 MTP (Value head 없음, `initialize_value_head=False`)
    - `run_critic.py`: Value Head 사전학습 (trunk frozen, `is_correct` 균형 샘플링)
    - `run_verifiable.py`: TD error 기반 WMTP (critic checkpoint 의존, curriculum learning)
    - `run_rho1.py`: Reference 모델 기반 weighting (Value head 없음)
  - **공통 내부 흐름** (모든 Pipeline):
    1. Config 로딩 (`defaults.yaml` + recipe 병합)
    2. Distributed Init (`runtime.distributed.init_distributed()`)
    3. Environment Setup (`runtime.environment.setup_environment()`: seed, device)
    4. Model 로딩 (`MetaLlamaMTPAdapter.from_pretrained()` → DDP/FSDP wrapping)
    5. Dataset 로딩 (메타데이터 기반 샘플링 → `DistributedSampler`)
    6. Training Loop (pipeline별 loss 로직, gradient sync, validation, checkpoint)
    7. MLflow Logging (Rank 0 전용)
  - **CLI 라우터 구현** (`cli/train.py`):
    - `experiment.stage` 필드 읽기 (baseline/critic/verifiable/rho1)
    - 해당 pipeline 함수 import 및 실행
    - Override params 전달 (`--run-name`, `--device`, `--use-micro-model`)
  - **Verifiable Pipeline Loss 구조**:
    - `total_loss = weighted_ce_loss + value_coef * value_loss`
    - TD error 계산 (`compute_td_errors()`) → Weight 산출 (`build_weights()`)
    - Value coefficient: 0.5 (기본) 또는 1.0 (recipe 설정)
    - Gradient clipping: max_grad_norm=0.5~1.0
  - `runtime/mlflow.py`: 실험 생성, 메트릭/아티팩트 기록 (Rank 0 전용)
  - 통합 테스트: micro 모델 + small 데이터로 각 pipeline별 smoke test
- **실제 성과** (2025-11-14):
  - ✅ 4개 독립 Pipeline 구현 완료 (`run_baseline.py`, `run_critic.py`, `run_verifiable.py`, `run_rho1.py`)
  - ✅ 각 Pipeline별 config 파일 구성 (`baseline.yaml`, `critic.yaml`, `verifiable.yaml`, `rho1.yaml`)
  - ✅ Runtime 모듈 구현 (`distributed.py`, `environment.py`, `ddp.py`)
  - ✅ Integration tests 3개 작성 (`test_pipeline_baseline.py`, `test_pipeline_critic.py`, `test_pipeline_verifiable.py`)
  - ✅ CLI 단순화 및 라우팅 로직 구현 (2025-11-17)
- **산출물**:
  - 4개 Pipeline 모듈 (`pipelines/run_*.py`)
  - CLI 라우터 (`cli/train.py`)
  - Runtime 모듈 (`runtime/distributed.py`, `runtime/environment.py`)
  - MLflow 모듈 (`runtime/mlflow.py`)
  - 통합 테스트 (`tests/integration/test_pipeline_*.py`)
- **검증 기준**:
  - CLI dry-run 동작 확인 (`--config configs/baseline/baseline.yaml --dry-run`)
  - 로컬 smoke test 성공 (micro 모델, runtime 모듈 자동 감지)
  - VESSL A100 4-GPU 환경에서 torchrun 실행 성공
  - DistributedSampler가 데이터를 4개 GPU로 중복 없이 분할 확인
  - DDP/FSDP가 모델 파라미터를 GPU로 분산 확인
  - MLflow에 핵심 메트릭 기록 (Rank 0만)
  - Verifiable Pipeline에서 value loss가 auxiliary loss로 추가되어 critic 지속 학습 확인

### P7. 평가·분석 파이프라인
- **목표**  
  - 학습된 모델의 Pass@K, Exact Match, Rho-1 비교 분석을 자동화한다.
- **주요 활동**  
  - `pipelines/evaluation.py`: inference 루틴, beam/nucleus 옵션, Rho-1 reference와의 loss 비교.  
  - `scripts/generate_metrics_report.py`: Pass@1/5/10, 평균 길이, 예시 코드 저장.  
  - MBPP/HumanEval 평가 스크립트 통합(채점 모듈).  
  - 결과를 MLflow/JSON으로 기록.
- **산출물**: 평가 파이프라인, 리포트 생성 스크립트.  
- **검증 기준**: baseline 모델용 평가 스크립트 문제없이 동작, 리포트 파일 생성.

### P8. Config & CLI 체계
- **목표**  
  - 사용자 진입점(`cli/train.py`)과 설정(`core/config.py`, recipes)을 완성해 실험 실행을 단순화한다.
- **주요 활동**  
  - Pydantic 기반 Config/Recipe 정의, defaults/recipes/ 환경변수 로딩.  
  - CLI 옵션(`--config`, `--recipe`, `--preset`, `--use-micro-model`).  
  - 세 실험 레시피(Baseline, Verifiable, Rho-1)의 파라미터 테이블 정리.  
  - dry-run 모드, 로깅 포맷 통일.
- **산출물**: `core/config.py`, `cli/train.py`, `configs/*.yaml`.  
- **검증 기준**: `uv run python -m weighted_mtp.cli.train --dry-run` 성공, recipe별 파이프라인 파라미터 주입 확인.

### P9. 테스트 & 품질 게이트
- **목표**  
  - 회귀 방지 체계를 구축하고 GitHub Actions 등 CI를 통해 지속적으로 검증한다.
- **주요 활동**  
  - unit/integration 테스트 확장, coverage 목표 설정.  
  - `scripts/run_smoke_tests.sh` 작성, pre-commit 훅 구성.  
  - `.github/workflows/ci.yaml`(lint, format, tests) 구축.  
  - 실패 시 triage 가이드 작성.
- **산출물**: pytest suite, smoke script, CI 워크플로우.  
- **검증 기준**: CI 100% 통과, 커버리지 목표(예: 핵심 모듈 80%+) 충족.

### P10. 배포 & 실험 운영
- **목표**  
  - VESSL/클라우드 환경에서 실험을 실행할 준비를 갖추고, 스토리지·MLflow 연동을 완성한다.
- **주요 활동**  
  - `scripts/sync_to_vessl_storage.py`: 모델/데이터 업로드 자동화.  
  - Dockerfile 또는 VESSL 기본 이미지 설정, requirements 잠금(`uv pip compile`).  
  - MLflow endpoint 검증, 자격 증명 관리.  
  - 배포 가이드, 예제 명령 작성.
- **산출물**: 동기화 스크립트, 배포 가이드, dry-run 로그.  
- **검증 기준**: `vessl run create ... --dry-run` 성공, MLflow artifact 업로드 확인.

### P11. 문서화 & 인수
- **목표**  
  - 모든 문서/체인지로그를 최신 상태로 맞추고, 최종 인수 체크리스트를 완료한다.
- **주요 활동**  
  - `docs/00_ideal_structure.md`, `docs/01_storage_preparation_plan.md` 차이 반영.  
  - `docs/migration_notes.md` 및 CHANGELOG 업데이트.  
  - 운영 가이드, Known Issues 작성.  
  - 인수 체크리스트(모델/데이터/코드/배포/문서) 전항목 확인 후 승인.
- **산출물**: 최신 문서 세트, 체인지로그, 릴리즈 노트, 인수 보고서.  
- **검증 기준**: 모든 체크 항목 ✔️, PO 승인 기록.

---

## 의존성 & 병행 전략
- **P0 → P1 → P2 → P3 → P4/P5 → P6 → P7** 순으로 핵심 경로를 이행한다.  
- P4(Value weighting)과 P5(파이프라인) 일부는 설계 확정 후 병행 가능하지만, 테스트는 Stage 통합 후에만 유효하다.  
- P8~P10은 학습 파이프라인이 안정화(P6 완료)된 뒤 바로 착수 가능하며, P9(CI)는 주요 모듈이 준비되는 시점부터 점진적으로 구축한다.
- 모든 Phase 종료 시 `migration_notes.md`에 결과와 차이를 기록해 다음 단계 착수 조건으로 활용한다.

---

## 완료 기준 요약
- **실험 실행**: CLI recipe로 Baseline/Verifiable/Rho-1 실험이 동일 파이프라인 위에서 실행되고, 평가 리포트가 자동 생성된다.
- **자산 관리**: `storage/` 구조와 metadata가 문서와 일치하며, SHA256/dtype/토크나이저 정보가 기록되어 재현성을 보장한다.
- **품질 보증**: unit/integration/CI가 모두 통과하고, MLflow/배포 스크립트가 정상 동작한다.
- **문서화**: 설계·준비·구현·운영 문서가 최신 상태로 유지되며, 제안서에 명시된 세 실험 비교 보고가 가능하다.

---

## 구현 완료 사항 (2025-11-17)

### ✅ Rho-1 Pipeline 개선 완료
**참조**: `docs/07_rho1_refactoring_plan.md`

- Rho-1 논문 방식 정확 구현 (signed difference, top-k binary selection)
- Per-head binary weights 도입, MTP 확장 전략 적용
- Config: `temperature` → `k_percent` 변경
- Integration test 통과: 8 tests, 88.11s

### ✅ S3 Checkpoint 최적화 완료
**참조**: `docs/checkpoint_s3_optimization.md`

- `utils/s3_utils.py` 생성: 비동기 S3 업로드 및 자동 정리
- 4개 파이프라인 적용 (Baseline, Critic, Verifiable, Rho1)
- MLflow 조건부 처리 통일
- 학습 중 실시간 Best checkpoint S3 백업

### 🔧 추가 개선
- Dummy Critic checkpoint 생성 (`scripts/create_dummy_critic_checkpoint.py`)
- Logger 구조 일관화 (module-level 제거, `setup_logging()` 통일)
- Verifiable pipeline MLflow 버그 수정

**상세**: `docs/00_ideal_structure.md` 끝부분 "구현 완료 사항" 섹션
