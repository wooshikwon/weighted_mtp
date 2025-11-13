# VESSL 실행 가이드 (리팩토링 버전)

`docs/00_ideal_structure.md`와 `docs/02_implementation_plan.md`에서 정의한 파이프라인/스토리지 구조에 맞추어, 기존 WMTP VESSL 운용 경험을 그대로 살리면서 리팩토링된 `weighted_mtp/` 프로젝트를 VESSL GPU 클러스터에서 실행하는 방법을 정리하였다.

---

## 1. 개요

- **지원 실험**: Baseline MTP, Verifiable Critic WMTP, Rho-1 Weighted (3개 핵심 레시피)
- **재사용 인프라**: 기존 EC2 MLflow Tracking Server + S3 Artifact Store
- **리팩토링 포인트**
  - storage v2 구조(`storage/models_v2`, `storage/datasets_v2`)
  - 새 CLI 엔트리 (`weighted_mtp.cli.train`)
  - configs/defaults + recipe 분리
  - `scripts/sync_to_vessl_storage.py`를 통한 자산 업로드

---

## 2. Phase별 준비 사항 정리

| Phase | 준비 내용 |
|-------|-----------|
| P1 | `01_storage_preparation_plan.md`에 따라 모델/데이터 자산 정리 후 VESSL Storage 업로드 |
| P6 | CLI & Config 완료 (`weighted_mtp/cli/train.py`, `configs/recipe.*.yaml`) |
| P8 | `scripts/sync_to_vessl_storage.py` 및 Docker 이미지 확인 |
| P9 | 문서/체크리스트 업데이트 |

*이 문서는 P8 착수 시점을 기준으로 활용한다.*

---

## 3. 사전 준비

### 3.1 자격증명
- **VESSL Secrets**
  - `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
  - `HF_TOKEN` (필요 시)
  - (선택) `STORAGE_MODELS_VERSION`, `STORAGE_DATASETS_VERSION`
- **로컬 `.env.vessl` 예시**
  ```bash
  MLFLOW_TRACKING_URI=http://13.50.240.176
  MLFLOW_TRACKING_USERNAME=<USER>
  MLFLOW_TRACKING_PASSWORD=<PASS>
  AWS_ACCESS_KEY_ID=<KEY>
  AWS_SECRET_ACCESS_KEY=<SECRET>
  AWS_DEFAULT_REGION=eu-north-1
  STORAGE_MODELS_VERSION=v2
  STORAGE_DATASETS_VERSION=v2
  ```

### 3.2 Docker 이미지
- `ghcr.io/wooshikwon/weighted-mtp:latest` (리팩토링 빌드)
- 필요 시 Phase P8에서 Dockerfile 업데이트 및 CI 빌드 파이프라인 적용

### 3.3 자산 업로드
```bash
uv run python scripts/sync_to_vessl_storage.py \
  --models-version v2 \
  --datasets-version v2
```
- 업로드 위치 예: `vessl://weighted-mtp/models_v2/meta-llama-mtp`
- 성공 후 MLflow run 태그에 `storage_sync=True` 기록

### 3.4 MLflow 서버 접속 정보
- **Tracking URI**: `http://13.50.240.176` (nginx reverse proxy, 80/tcp)
- **인증**: Basic Auth — `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`
- **헬스 체크**
  ```bash
  curl -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD" \
    http://13.50.240.176/health
  ```
- **웹 UI**: 브라우저에서 `http://13.50.240.176` 접속 후 위 계정으로 로그인
- **SSH (서버 점검 시)**
  ```bash
  ssh -i <path_to_mlflow_key.pem> ubuntu@13.50.240.176
  ```
  - 서버 내부에서 MLflow는 5000/tcp, nginx는 80/tcp로 서비스됨
- **보안 그룹**: 22/tcp는 허용 IP만, 80/tcp는 Basic Auth 전제로 공개

---

## 4. Config 및 CLI 구조

### 4.1 `configs/defaults.yaml`
```yaml
mlflow:
  tracking_uri: ${env:MLFLOW_TRACKING_URI}
  experiment_prefix: "weighted_mtp"

paths:
  storage_root: "/mnt/workspace/storage"  # VESSL volume mount 시 경로
  models_version: ${env:STORAGE_MODELS_VERSION}
  datasets_version: ${env:STORAGE_DATASETS_VERSION}

runtime:
  preset: "production"     # local-light, production 등
```

### 4.2 레시피 (요약)

| 파일 | 목적 | 특징 |
|------|------|------|
| `recipe.baseline.yaml` | Baseline MTP | Stage2 only, ref/RM 불필요 |
| `recipe.verifiable.yaml` | Verifiable Critic | Stage1 trunk pretrain + Stage2, 라벨 사용 |
| `recipe.rho1_weighted.yaml` | Rho-1 Weighted | Reference 모델 필요, Stage1 생략 |

### 4.3 CLI
```bash
uv run python -m weighted_mtp.cli.train \
  --config configs/defaults.yaml \
  --recipe configs/recipe.verifiable.yaml \
  --run-name vessl_verifiable_001 \
  --preset production
```

---

## 5. VESSL 실행 명령 예시

### 5.1 공통 템플릿
```bash
vessl run create \
  --cluster vessl-gcp-oregon \
  --resource a100-1gpu \
  --image ghcr.io/wooshikwon/weighted-mtp:latest \
  --name <run-name> \
  --env-file .env.vessl \
  --volume weighted-mtp-storage:/mnt/workspace/storage:rw \
  --command "
      uv run python -m weighted_mtp.cli.train \
        --config configs/defaults.yaml \
        --recipe configs/<recipe-file>.yaml \
        --run-name <run-name> \
        --preset production \
        --tags vessl,<recipe-base>
  "
```

#### Volume 전략
- `weighted-mtp-storage` VESSL volume을 `/mnt/workspace/storage`에 마운트하여 `storage/models_v2`, `storage/datasets_v2` 접근.
- 최초 실행 전에 `vessl storage upload` 또는 커스텀 sync 스크립트로 업로드.

### 5.2 실험별 샘플

| 실험 | GPU | 명령 |
|------|-----|------|
| Baseline | v100-1gpu | `--recipe configs/recipe.baseline.yaml --tags vessl,baseline` |
| Verifiable Critic | a100-1gpu | `--recipe configs/recipe.verifiable.yaml --tags vessl,verifiable` |
| Rho-1 Weighted | v100-1gpu | `--recipe configs/recipe.rho1_weighted.yaml --tags vessl,rho1-weighted` |

### 5.3 로컬 `.env`와 병용
- 로컬에서 `.env`를 작성한 뒤 `env-file .env.vessl`로 전달 (파일 내부는 Secrets 키만 포함).
- 민감 정보는 VESSL Secrets로 등록한 뒤 `--env-file` 없이 실행 가능.

---

## 6. 실행 모니터링

- **CLI**: `vessl run logs <run-id> --follow`, `vessl run get <run-id>`
- **MLflow**: 기존 EC2 UI에서 `weighted_mtp/<experiment>` 확인
- **Artifacts**: `vessl run artifact list <run-id>` 또는 MLflow UI → Artifacts
- **모니터링 포인트**
- Stage1: `metrics/value_head_loss`, `metrics/value_norm`
- Stage2: `metrics/td_error_mean`, `metrics/weight_entropy`, `metrics/wmtp_loss`
- 최종: `tags.model_version`, `params.dataset_version`

---

## 7. 트러블슈팅 요약

| 이슈 | 확인 항목 | 해결 |
|------|----------|------|
| MLflow 연결 실패 | `MLFLOW_TRACKING_URI`, 보안그룹, nginx | `curl -u "$MLFLOW_TRACKING_USERNAME:$MLFLOW_TRACKING_PASSWORD" http://13.50.240.176/health` |
| 모델 로드 실패 | volume 마운트, `STORAGE_MODELS_VERSION` | `vessl storage ls weighted-mtp/models_v2` |
| Dataset 누락 | `STORAGE_DATASETS_VERSION`, 전처리 로그 | `scripts/sync_to_vessl_storage.py --verify` |
| OOM | GPU 리소스 변경, recipe 배치 축소 | `data.train.batch_size` 조정 |
| Docker pull 실패 | 이미지 태그 확인 | `docker pull ghcr.io/wooshikwon/weighted-mtp:latest` |

자세한 내용은 `legacy_docs/WMTP_VESSL_실행_가이드.md`의 트러블슈팅 섹션 참고 (명령어만 최신 체계로 교체).

---

## 8. 비용 및 운영 팁

- `--preemptible` 옵션으로 Spot 사용 시 비용 절감 (중단 대비 체크포인트 저장 필수)
- `recipes`에서 `max_steps`, `eval_interval` 조정으로 벽시계 시간 관리
- MLflow에서 `compute/tokens_seen` 모니터링으로 GPU 활용도 분석

---

## 9. 체크리스트

1. [ ] `storage/models_v2`, `storage/datasets_v2`가 VESSL volume에 존재
2. [ ] `.env.vessl` 또는 Secrets에 MLflow/S3 credentials 설정
3. [ ] `configs/defaults.yaml`, recipe 파일 최신화
4. [ ] `uv run python -m weighted_mtp.cli.train --dry-run` 로컬 통과
5. [ ] `vessl run create ... --dry-run` 성공
6. [ ] MLflow에서 실험/메트릭/Artifacts 확인

---

## 10. 문서 업데이트 지침

- VESSL 명령 템플릿 또는 storage 버전이 변경되면 본 문서를 즉시 수정.
- Phase 8 완료 후 `docs/migration_notes.md`에 VESSL 연동 결과 기록.
- 세부 레시피/하이퍼파라미터 설명은 `docs/00_ideal_structure.md` 및 각 recipe 주석으로 관리.

---

이 문서는 리팩토링된 `weighted_mtp/` 환경에서 VESSL 클러스터를 효과적으로 활용하기 위한 표준 가이드로, 기존 가이드(`legacy_docs/WMTP_VESSL_실행_가이드.md`)의 노하우를 최신 구조에 맞춰 압축·재구성하였다.

