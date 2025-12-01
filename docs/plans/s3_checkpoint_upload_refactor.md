# S3 Checkpoint Upload 리팩토링 계획

## 현황 분석 (원칙 1)

### 기존 구조의 문제점
| 파일 | 문제점 |
|------|--------|
| `s3_utils.py` | MlflowClient 사용 → MLflow 서버 필요 (현재 서버 없음) |
| `checkpoint_utils.py` | `mlflow_run_id` 파라미터 → MLflow 서버 의존 |
| `run_verifiable.py` | S3 파라미터 누락 (이전 작업에서 추가 완료) |

### 이미 완료된 작업
1. `s3_utils.py` boto3 직접 사용으로 전면 교체
2. `sync_mlruns_to_s3`, `sync_checkpoints_to_s3` 함수 추가
3. `utils/__init__.py` export 업데이트
4. `run_verifiable.py` S3 파라미터 추가
5. `baseline.yaml` tracking_uri 로컬 저장으로 변경

---

## 수정 범위 (원칙 2, 4)

### 1. checkpoint_utils.py 파라미터 변경

**변경 대상 함수:**
- `save_checkpoint`
- `save_lora_checkpoint`
- `save_value_model_checkpoint`

**변경 내용:**
```python
# Before
def save_checkpoint(
    ...
    s3_upload: bool = False,
    mlflow_run_id: str | None = None,  # MLflow 서버 의존
)

# After
def save_checkpoint(
    ...
    s3_upload: bool = False,
    experiment_name: str | None = None,  # boto3 직접 업로드용
)
```

**S3 업로드 로직 변경:**
```python
# Before (MlflowClient 사용)
if s3_upload and mlflow_run_id:
    s3_upload_executor.submit(upload_to_s3_async, checkpoint_path, mlflow_run_id)

# After (boto3 직접 사용)
if s3_upload and experiment_name:
    s3_upload_executor.submit(upload_to_s3_async, checkpoint_path, experiment_name)
```

### 2. 각 파이프라인 호출부 수정

**변경 대상:**
- `run_baseline.py`
- `run_critic.py`
- `run_verifiable.py`
- `run_rho1.py`

**변경 내용:**
```python
# Before
save_checkpoint(
    ...
    s3_upload=use_s3_upload,
    mlflow_run_id=mlflow_run_id,
)

# After
save_checkpoint(
    ...
    s3_upload=use_s3_upload,
    experiment_name=config.experiment.name,
)
```

### 3. 파이프라인 종료 시 mlruns 동기화 추가

**위치:** 각 파이프라인의 `shutdown_s3_executor()` 후

```python
# 기존
shutdown_s3_executor()
if is_main_process() and use_mlflow:
    mlflow.end_run()

# 추가
shutdown_s3_executor()
if is_main_process() and use_mlflow:
    mlflow.end_run()

    # MLflow 메트릭/파라미터 S3 백업
    if use_s3_upload:
        sync_mlruns_to_s3()
```

### 4. cleanup_s3_checkpoints 호출부 수정

**변경 내용:**
```python
# Before
cleanup_s3_checkpoints(
    experiment_id="",
    run_id=mlflow_run_id,
    save_total_limit=config.checkpoint.save_total_limit,
)

# After
cleanup_s3_checkpoints(
    experiment_name=config.experiment.name,
    save_total_limit=config.checkpoint.save_total_limit,
)
```

---

## S3 저장 구조

```
s3://wmtp/
├── checkpoints/{experiment_name}/     # 개별 체크포인트 (async 업로드)
│   ├── checkpoint_epoch_0.20.pt
│   ├── checkpoint_epoch_0.40.pt
│   └── checkpoint_final.pt
└── mlruns/                            # MLflow 메트릭/파라미터 (종료 시 동기화)
    └── {experiment_id}/
        └── {run_id}/
            ├── metrics/
            ├── params/
            └── tags/
```

---

## 구현 순서

| 단계 | 작업 | 파일 |
|:---:|------|------|
| 1 | checkpoint_utils.py 파라미터 변경 | `checkpoint_utils.py` |
| 2 | run_baseline.py 호출부 수정 | `run_baseline.py` |
| 3 | run_critic.py 호출부 수정 | `run_critic.py` |
| 4 | run_verifiable.py 호출부 수정 | `run_verifiable.py` |
| 5 | run_rho1.py 호출부 수정 | `run_rho1.py` |
| 6 | 구현 결과 검토 | - |

---

## 검증 항목

- [ ] `s3_upload: false` 시 S3 업로드 안됨
- [ ] `s3_upload: true` 시 체크포인트 S3 업로드 정상
- [ ] 학습 중단 시에도 이미 저장된 체크포인트는 S3에 존재
- [ ] 학습 완료 시 mlruns 디렉터리 S3 동기화
- [ ] `save_total_limit` 초과 시 S3에서 오래된 체크포인트 삭제

---

## 로컬 확인 방법

```bash
# S3에서 mlruns 다운로드
aws s3 sync s3://wmtp/mlruns/ ./mlruns/

# MLflow UI 실행
mlflow ui --backend-store-uri ./mlruns --port 5000

# 브라우저에서 확인
open http://localhost:5000
```
