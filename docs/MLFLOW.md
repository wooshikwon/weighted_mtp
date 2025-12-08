# MLflow 실험 추적

프로젝트의 학습 실험을 추적하고 시각화하기 위한 MLflow 사용 가이드.

## 실험 기록 확인

프로젝트에는 `weighted_mtp` 실험의 27개 학습 기록이 SQLite 데이터베이스(`mlflow.db`)에 저장되어 있습니다.

### UI 실행

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

브라우저에서 http://127.0.0.1:5000 접속 후 `weighted_mtp` 실험을 클릭하면 학습 기록을 확인할 수 있습니다.

### 포트 변경

기본 포트(5000)가 사용 중인 경우:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

## 학습 시 로깅

### Config 기반 설정

로컬 config 파일에 MLflow 설정이 포함되어 있습니다:

```yaml
# configs/local/*.yaml
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment: "weighted_mtp"
```

파이프라인 실행 시 자동으로 MLflow에 기록됩니다:

```bash
PYTHONPATH=src python src/weighted_mtp/pipelines/run_baseline.py \
  --config configs/local/baseline_local.yaml
```

### 기록되는 항목

- **Parameters**: 학습 설정 (learning_rate, batch_size, epochs 등)
- **Metrics**: 학습/검증 손실, 정확도 등 (전체 히스토리 포함)
- **Artifacts**: 체크포인트, 설정 파일

### MLflow 비활성화

로깅을 끄려면 config에서 experiment를 빈 문자열로 설정:

```yaml
mlflow:
  tracking_uri: ""
  experiment: ""
```

## Python API

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = mlflow.tracking.MlflowClient()

# 실험 조회
experiment = client.get_experiment_by_name("weighted_mtp")

# Run 목록 조회
runs = client.search_runs(experiment.experiment_id)
for run in runs:
    print(f"{run.info.run_name}: {run.data.metrics}")
```

## 원격 서버

원격 MLflow 서버를 사용하는 경우 config 또는 환경변수로 설정:

```yaml
# configs/production/*.yaml
mlflow:
  tracking_uri: "http://<server-ip>:5000"
  experiment: "weighted_mtp"
```

또는:

```bash
export MLFLOW_TRACKING_URI=http://<server-ip>:5000
export MLFLOW_TRACKING_USERNAME=<username>
export MLFLOW_TRACKING_PASSWORD=<password>
```

## 참고

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- 로컬 백업: `mlruns_backup/` (gitignore 적용)
