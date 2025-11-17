# Checkpoint S3 저장 최적화 계획

**작성일**: 2025-11-17
**목적**: 학습 중 checkpoint의 효율적인 S3 백업 및 관리

---

## 현재 구현 분석

### Checkpoint 저장 흐름

**학습 중:**
```python
# run_baseline.py:563-579
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}.pt"

    save_checkpoint(...)  # 로컬 디스크에 동기 저장 (torch.save)

    if config.checkpoint.save_total_limit:
        cleanup_old_checkpoints(...)  # 로컬 디스크에서 오래된 파일 삭제
```

**학습 종료 후:**
```python
# run_baseline.py:612-620
if is_main_process() and use_mlflow:
    epoch_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if epoch_checkpoints:
        latest_checkpoint = epoch_checkpoints[-1]
        mlflow.log_artifact(str(latest_checkpoint), "checkpoints")  # S3 업로드 (동기)
```

---

## 문제점

### ❌ 문제 1: 학습 중 S3 백업 없음

**현재 동작:**
1. 학습 중 checkpoint → 로컬 디스크만 저장
2. 학습 완료 후 → 단 1개만 S3 업로드 (latest checkpoint)
3. Best checkpoint는 S3에 업로드되지 않음

**리스크:**
- VESSL run이 중단되면 checkpoint 손실
- Best model이 S3에 백업되지 않음
- `save_total_limit=3`으로 로컬에서 삭제된 checkpoint는 영구 손실

---

### ❌ 문제 2: `mlflow.log_artifact()`는 동기 (Blocking)

**검색 결과 (2025-01-17):**
- `log_artifact()`는 동기 함수
- 비동기 지원은 `log_metric()`, `log_image()`만 가능
- Feature Request (mlflow/mlflow#14153) 진행 중이지만 미구현

**현재 영향:**
- 학습 종료 후 호출이므로 학습 자체는 블로킹 안됨
- 하지만 학습 완료 → artifact 업로드 완료까지 시간 소요

---

### ❌ 문제 3: S3 삭제 로직 없음

**현재 동작:**
- 로컬: `cleanup_old_checkpoints()`로 `save_total_limit=3` 유지
- S3: 삭제 로직 없음 → 무한 누적

**리스크:**
- S3 스토리지 비용 증가
- 불필요한 checkpoint가 계속 쌓임

---

## 개선 계획

### ✅ 해결책 1: 학습 중 비동기 S3 업로드

**방법**: ThreadPoolExecutor 사용

**구현 위치**: `src/weighted_mtp/pipelines/s3_utils.py` (신규 생성)

```python
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# 전역 executor (파이프라인 초기화 시 생성)
s3_upload_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-upload")

def upload_to_s3_async(checkpoint_path: Path, mlflow_enabled: bool):
    """비동기로 S3에 checkpoint 업로드

    Args:
        checkpoint_path: 업로드할 checkpoint 경로
        mlflow_enabled: MLflow 사용 여부
    """
    if not mlflow_enabled:
        return

    try:
        import mlflow
        mlflow.log_artifact(str(checkpoint_path), "checkpoints")
        logger.info(f"✓ S3 upload complete: {checkpoint_path.name}")
    except Exception as e:
        logger.error(f"✗ S3 upload failed: {checkpoint_path.name} - {e}")

def shutdown_s3_executor():
    """모든 S3 업로드 완료 대기"""
    s3_upload_executor.shutdown(wait=True)
    logger.info("All S3 uploads completed")
```

**파이프라인 수정**: `run_baseline.py` (및 다른 파이프라인들)

```python
from weighted_mtp.pipelines.s3_utils import upload_to_s3_async, shutdown_s3_executor

# Checkpoint 저장 후
if avg_val_loss < best_val_loss:
    best_val_loss = avg_val_loss
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{current_epoch:.2f}.pt"

    # 1. 로컬 저장 (동기 - 빠름)
    save_checkpoint(...)

    # 2. S3 업로드 (비동기 - 학습 방해 안함) ✅
    if is_main_process() and use_mlflow:
        s3_upload_executor.submit(upload_to_s3_async, checkpoint_path, use_mlflow)

    # 3. 로컬 정리 (동기 - 빠름)
    if config.checkpoint.save_total_limit:
        cleanup_old_checkpoints(...)

# 학습 종료 시
shutdown_s3_executor()  # 모든 업로드 완료 대기
if is_main_process() and use_mlflow:
    mlflow.end_run()
```

**장점:**
1. 학습 루프가 S3 업로드를 기다리지 않음 (non-blocking)
2. Best checkpoint가 실시간으로 S3에 백업됨
3. VESSL run 중단 시에도 이미 업로드된 checkpoint 보존

---

### ✅ 해결책 2: S3 삭제 로직 추가

**구현 위치**: `src/weighted_mtp/pipelines/s3_utils.py`

```python
def cleanup_s3_checkpoints(
    experiment_id: str,
    run_id: str,
    save_total_limit: int,
) -> None:
    """S3에서 오래된 checkpoint 삭제 (MLflow artifact store)

    Args:
        experiment_id: MLflow experiment ID
        run_id: MLflow run ID
        save_total_limit: 유지할 최대 개수
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    import boto3

    client = MlflowClient()
    s3 = boto3.client('s3')

    try:
        # S3에 있는 모든 checkpoint 목록 가져오기
        artifacts = client.list_artifacts(run_id, path="checkpoints")

        # checkpoint_epoch_*.pt 파일만 필터링
        epoch_checkpoints = [
            a for a in artifacts
            if a.path.startswith("checkpoints/checkpoint_epoch_")
        ]

        # 시간순 정렬 (파일명 기준)
        epoch_checkpoints.sort(key=lambda x: x.path)

        # 삭제할 파일 개수
        n_to_delete = len(epoch_checkpoints) - save_total_limit

        if n_to_delete > 0:
            bucket = 'wmtp'
            for artifact in epoch_checkpoints[:n_to_delete]:
                # boto3로 S3 파일 직접 삭제
                s3_key = f"mlflow-artifacts/{experiment_id}/{run_id}/artifacts/{artifact.path}"
                s3.delete_object(Bucket=bucket, Key=s3_key)
                logger.info(f"S3 checkpoint deleted: {artifact.path}")

    except Exception as e:
        logger.warning(f"S3 cleanup failed: {e}")
```

**파이프라인 적용:**

```python
# 로컬 정리 후 S3도 정리
if config.checkpoint.save_total_limit:
    cleanup_old_checkpoints(checkpoint_dir, save_total_limit)

    # S3 정리 (비동기)
    if is_main_process() and use_mlflow:
        s3_upload_executor.submit(
            cleanup_s3_checkpoints,
            experiment_id=mlflow.active_run().info.experiment_id,
            run_id=mlflow.active_run().info.run_id,
            save_total_limit=save_total_limit,
        )
```

---

### ✅ 해결책 3: S3 업로드 속도 최적화

**환경변수 설정**: `.env` 또는 파이프라인 초기화 시

```python
import os
import json

# S3 멀티파트 업로드 최적화
os.environ['MLFLOW_S3_UPLOAD_EXTRA_ARGS'] = json.dumps({
    'ServerSideEncryption': 'AES256',     # 선택적
    'StorageClass': 'STANDARD_IA',        # 비용 절감 (자주 접근 안하는 경우)
})
```

**boto3 transfer 설정** (선택적):

```python
from boto3.s3.transfer import TransferConfig

transfer_config = TransferConfig(
    multipart_threshold=8 * 1024 * 1024,  # 8MB 이상 멀티파트
    max_concurrency=10,                    # 동시 업로드 스레드
    multipart_chunksize=8 * 1024 * 1024,  # 청크 크기
    use_threads=True,
)
```

**파일 압축** (선택적, 대용량 checkpoint의 경우):

```python
import gzip
import shutil

def compress_checkpoint(checkpoint_path: Path) -> Path:
    """Checkpoint를 gzip으로 압축 (S3 전송 속도 향상)

    Args:
        checkpoint_path: 원본 checkpoint 경로

    Returns:
        압축된 파일 경로
    """
    compressed_path = checkpoint_path.with_suffix('.pt.gz')

    with open(checkpoint_path, 'rb') as f_in:
        with gzip.open(compressed_path, 'wb', compresslevel=6) as f_out:
            shutil.copyfileobj(f_in, f_out)

    return compressed_path
```

---

### ✅ 해결책 4: Best Model 개수 2개로 축소

**Config 수정**: `configs/defaults.yaml`

```yaml
checkpoint:
  save_checkpoint_every: 1.0
  save_best: true
  save_final: true
  save_total_limit: 2  # 3 → 2로 변경 (로컬 저장 부담 감소)
```

**효과:**
- 로컬 디스크 사용량 33% 감소
- S3 스토리지 비용 33% 감소
- Checkpoint 관리 단순화

---

## 최종 구조

### 학습 루프 흐름

```python
# Validation 후 checkpoint 저장
if avg_val_loss < best_val_loss:
    # 1. 로컬 저장 (동기 - 빠름)
    save_checkpoint(
        adapter=unwrap_model(adapter),
        optimizer=optimizer,
        epoch=current_epoch,
        train_metrics={"train_loss": train_loss_avg},
        val_metrics=val_metrics,
        checkpoint_path=checkpoint_path,
    )

    # 2. S3 업로드 (비동기 - 학습 방해 안함) ✅
    if is_main_process() and use_mlflow:
        s3_upload_executor.submit(
            upload_to_s3_async,
            checkpoint_path=checkpoint_path,
            mlflow_enabled=use_mlflow,
        )

    # 3. 로컬 정리 (동기 - 빠름)
    if config.checkpoint.save_total_limit:
        cleanup_old_checkpoints(
            checkpoint_dir=checkpoint_dir,
            save_total_limit=config.checkpoint.save_total_limit,
        )

    # 4. S3 정리 (비동기 - 학습 방해 안함) ✅
    if is_main_process() and use_mlflow:
        s3_upload_executor.submit(
            cleanup_s3_checkpoints,
            experiment_id=mlflow.active_run().info.experiment_id,
            run_id=mlflow.active_run().info.run_id,
            save_total_limit=config.checkpoint.save_total_limit,
        )

# 학습 종료 시
shutdown_s3_executor()  # 모든 비동기 작업 완료 대기
if is_main_process() and use_mlflow:
    mlflow.end_run()
```

---

## 비교표

| 항목 | 현재 상태 | 문제점 | 개선 후 |
|------|----------|--------|---------|
| **S3 업로드 시점** | 학습 종료 후 1번 | Best checkpoint 백업 안됨 | 학습 중 실시간 업로드 ✅ |
| **Blocking 여부** | 학습 종료 후라 무관 | 종료 지연 | 비동기 (non-blocking) ✅ |
| **S3 정리** | 없음 | 무한 누적 | boto3로 자동 삭제 ✅ |
| **로컬 정리** | ✅ 정상 작동 (3개 유지) | - | ✅ 2개로 축소 |
| **업로드 속도** | 기본 설정 | 최적화 여지 | S3 transfer config 튜닝 ✅ |
| **Best model 개수** | 3개 | 저장 공간 부담 | 2개로 축소 ✅ |

---

## 구현 우선순위

1. **P0 (필수)**:
   - `src/weighted_mtp/pipelines/s3_utils.py` 생성
   - 비동기 S3 업로드 구현
   - `save_total_limit: 2`로 변경

2. **P1 (중요)**:
   - S3 삭제 로직 구현
   - 4개 파이프라인에 적용 (baseline, critic, verifiable, rho1)

3. **P2 (선택)**:
   - S3 업로드 속도 최적화 (transfer config)
   - Checkpoint 압축 (대용량 모델의 경우)

---

## 참고 자료

- MLflow Issue #14153: [FR] add asynchronous option to client log_artifact
- MLflow Docs: [Environment Variables](https://mlflow.org/docs/latest/python_api/mlflow.environment_variables.html)
- boto3 S3 Transfer: [TransferConfig](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/customizations/s3.html#boto3.s3.transfer.TransferConfig)

---

## 업데이트 이력

- **2025-11-17**: 초안 작성 및 분석 완료
