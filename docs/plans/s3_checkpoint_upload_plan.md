# MLflow S3 Checkpoint Upload 구현 계획

## 현황 분석

### 기존 구현 상태

| 파일 | 상태 | 문제점 |
|------|------|--------|
| `s3_utils.py` | 구현됨 | `upload_to_s3_async`, `cleanup_s3_checkpoints` 함수 존재 |
| `run_critic.py` | **미완성** | `use_s3_upload` 변수만 정의, 실제 호출 없음 |
| `run_baseline.py` | 미확인 | 동일 패턴 예상 |
| `run_verifiable.py` | 미확인 | 동일 패턴 예상 |

### 기존 S3 Utils 구조

```python
# s3_utils.py
s3_upload_executor = ThreadPoolExecutor(max_workers=2)

def upload_to_s3_async(checkpoint_path: Path, run_id: str):
    """MlflowClient로 artifact 업로드"""
    client = MlflowClient()
    client.log_artifact(run_id, str(checkpoint), artifact_path="checkpoints")

def cleanup_s3_checkpoints(experiment_id, run_id, save_total_limit):
    """S3에서 오래된 checkpoint 삭제"""
    ...

def shutdown_s3_executor():
    """업로드 완료 대기"""
    s3_upload_executor.shutdown(wait=True)
```

### S3 저장 경로

```
s3://wmtp/mlflow-artifacts/{run_id}/artifacts/checkpoints/
├── checkpoint_epoch_0.20.pt
├── checkpoint_epoch_0.40.pt
└── checkpoint_final.pt
```

---

## Phase 1: run_critic.py S3 업로드 구현

### 목표
- `checkpoint.s3_upload: true` 설정 시 실제 S3 업로드 활성화
- 비동기 업로드로 학습 속도 영향 최소화

### 수정 범위

#### 1.1 Import 추가

```python
# 현재
from weighted_mtp.utils import (
    ...
    shutdown_s3_executor,
)

# 변경
from weighted_mtp.utils import (
    ...
    shutdown_s3_executor,
    upload_to_s3_async,
    s3_upload_executor,
    cleanup_s3_checkpoints,
)
```

#### 1.2 Checkpoint 저장 후 S3 업로드 호출

위치: `save_value_model_checkpoint` 호출 후 (약 line 990)

```python
# Checkpoint 저장
save_value_model_checkpoint(...)

barrier()

if is_main_process():
    logger.info(f"Checkpoint saved: {checkpoint_path.name}")

    # S3 비동기 업로드 추가
    if use_s3_upload and mlflow_run_id:
        s3_upload_executor.submit(
            upload_to_s3_async, checkpoint_path, mlflow_run_id
        )
        logger.info(f"S3 upload queued: {checkpoint_path.name}")

    # 로컬 cleanup
    if config.checkpoint.save_total_limit:
        cleanup_old_checkpoints(...)

        # S3 cleanup 추가
        if use_s3_upload and mlflow_run_id:
            cleanup_s3_checkpoints(
                experiment_id=None,
                run_id=mlflow_run_id,
                save_total_limit=config.checkpoint.save_total_limit,
            )
```

#### 1.3 Final Checkpoint S3 업로드

위치: `checkpoint_final.pt` 저장 후 (약 line 1055)

```python
# Final checkpoint 저장 후
if is_main_process():
    logger.info(f"Final checkpoint saved: {final_path.name}")

    if use_s3_upload and mlflow_run_id:
        s3_upload_executor.submit(
            upload_to_s3_async, final_path, mlflow_run_id
        )
```

### 검증 항목

- [ ] `checkpoint.s3_upload: false` 시 S3 업로드 안됨
- [ ] `checkpoint.s3_upload: true` 시 S3 업로드 정상 작동
- [ ] MLflow 없을 때 (`mlflow.experiment: ""`) S3 업로드 스킵
- [ ] 비동기 업로드로 학습 블로킹 없음
- [ ] `shutdown_s3_executor()` 호출로 종료 전 업로드 완료 보장

---

## Phase 2: 다른 파이프라인 동기화

### 대상 파일

1. `run_baseline.py`
2. `run_verifiable.py`
3. `run_rho1.py`

### 작업 내용

각 파이프라인에서:
1. S3 관련 함수 import 확인/추가
2. `use_s3_upload` 변수 정의 확인
3. checkpoint 저장 위치에 S3 업로드 호출 추가
4. cleanup 로직 추가

### 공통 패턴 추출 고려

반복 코드가 많으면 헬퍼 함수 고려:

```python
def save_checkpoint_with_s3(
    checkpoint_path: Path,
    use_s3_upload: bool,
    mlflow_run_id: Optional[str],
    save_total_limit: Optional[int] = None,
):
    """Checkpoint 저장 후 S3 업로드 및 cleanup 처리"""
    ...
```

---

## Phase 3: Config 검증 및 문서화

### 3.1 Config 스키마 정리

```yaml
checkpoint:
  save_dir: storage/checkpoints/critic/${experiment.name}
  save_checkpoint_every: 0.2
  save_best: true
  save_final: true
  save_total_limit: 5
  s3_upload: true  # MLflow artifact store (S3)에 업로드

mlflow:
  tracking_uri: "http://13.50.240.176:5000"  # MLflow 서버
  experiment: "weighted-mtp/production"
```

### 3.2 환경 변수 요구사항

S3 업로드를 위해 필요한 환경 변수:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=eu-north-1
export MLFLOW_TRACKING_URI=http://13.50.240.176:5000
```

### 3.3 의존성 확인

```toml
# pyproject.toml
[project.dependencies]
boto3 = "..."
mlflow = "..."
```

---

## Phase 4: 에러 핸들링 강화

### 4.1 S3 업로드 실패 시 동작

현재 `s3_utils.py`에서 예외 catch 후 로깅만 수행:

```python
except Exception as e:
    logger.error(f"S3 upload failed: {checkpoint_path.name} - {e}")
```

**개선 고려사항**:
- 재시도 로직 (exponential backoff)
- 실패 시 로컬 백업 경로 유지 알림
- MLflow run에 실패 태그 추가

### 4.2 네트워크 타임아웃 설정

```python
s3 = boto3.client(
    "s3",
    config=Config(
        connect_timeout=10,
        read_timeout=30,
        retries={"max_attempts": 3}
    )
)
```

---

## 구현 우선순위

| Phase | 작업 | 우선순위 | 예상 소요 |
|-------|------|----------|----------|
| 1 | run_critic.py S3 업로드 | **높음** | 30분 |
| 2 | 다른 파이프라인 동기화 | 중간 | 1시간 |
| 3 | Config 검증/문서화 | 낮음 | 30분 |
| 4 | 에러 핸들링 강화 | 낮음 | 1시간 |

---

## 테스트 계획

### 단위 테스트

```python
def test_s3_upload_disabled():
    """s3_upload=false 시 업로드 스킵 확인"""
    ...

def test_s3_upload_without_mlflow():
    """MLflow 없을 때 업로드 스킵 확인"""
    ...

def test_s3_cleanup():
    """save_total_limit 초과 시 오래된 checkpoint 삭제"""
    ...
```

### 통합 테스트

```bash
# 로컬 테스트 (S3 업로드 비활성화)
python -m weighted_mtp.pipelines.run_critic \
    --config configs/local/critic_local.yaml \
    --override checkpoint.s3_upload=false

# 프로덕션 테스트 (S3 업로드 활성화)
torchrun --nproc_per_node=3 \
    -m weighted_mtp.pipelines.run_critic \
    --config configs/production/critic_mlp.yaml
```
