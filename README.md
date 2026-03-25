# Weighted Multi-Token Prediction (WMTP)

코드 생성 모델의 성능 향상을 위한 **토큰별 가중치 학습** 프레임워크.

## 핵심 아이디어

모든 토큰을 동일하게 학습하는 대신, **올바른 코드 생성에 중요한 토큰에 더 높은 가중치**를 부여합니다.

```
V(t) → TD Error → Advantage Whitening → exp(A/β) → Weighted CE Loss
```

- **NTP + Weight Modes**: HuggingFace LlamaForCausalLM 기반 Next-Token Prediction에 4가지 가중치 전략 적용
- **Value Model**: 토큰별 성공 확률 예측 (Pairwise Ranking 학습)
- **TD Weighting**: GAE 기반 토큰 중요도 계산

## 빠른 시작

```bash
# 의존성 설치
uv sync --dev

# 환경변수 설정
echo "HF_TOKEN=<your-token>" > .env

# 데이터셋 준비
uv run python scripts/create_storage/setup_datasets.py --datasets all --steps process,small

# Baseline 학습 (uniform weight mode)
python -m weighted_mtp train --config configs/production/baseline.yaml

# 분산 학습 (4-GPU)
torchrun --nproc_per_node=4 -m weighted_mtp.pipelines.run_baseline --config configs/production/baseline.yaml
```

## 학습 순서

```
1. Critic 학습       →  run_critic.py       (Value Model 학습)
2. Baseline NTP      →  run_baseline.py     (4가지 weight mode 지원)
3. 평가              →  run_evaluation.py   (HumanEval, MBPP, CodeContests)
```

| 파이프라인 | 설명 | Config 예시 |
|-----------|------|-------------|
| `run_baseline.py` | NTP 학습 (4 weight modes: uniform/critic/random/shuffled) | `configs/production/baseline.yaml` |
| `run_baseline.py` | TAW - critic weight mode | `configs/production/taw.yaml` |
| `run_baseline.py` | Random-Matched 대조군 | `configs/production/random_matched.yaml` |
| `run_baseline.py` | Shuffled 대조군 | `configs/production/shuffled.yaml` |
| `run_critic.py` | Pairwise Ranking Value Model | `configs/production/critic_mlp.yaml` |
| `run_evaluation.py` | Pass@K 평가 | CLI arguments |

## 프로젝트 구조

```
weighted_mtp/
├── src/weighted_mtp/
│   ├── pipelines/        # 학습/평가 파이프라인 (run_baseline, run_critic, run_evaluation)
│   ├── models/           # Policy Model (LlamaForCausalLM), Value Model (LlamaModel), LoRA
│   ├── value_weighting/  # TD error, control weights
│   ├── data/             # 데이터셋, Collators
│   └── runtime/          # FSDP, 분산학습
├── configs/              # YAML 설정 (local/, production/)
├── storage/              # 모델, 데이터셋, 체크포인트
└── tests/                # 테스트
```

## 문서

| 문서 | 내용 |
|------|------|
| [SETUP.md](docs/SETUP.md) | 환경 설정, 모델/데이터 준비 |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | 코드베이스 구조, 핵심 구현 |
| [RESEARCH.md](docs/RESEARCH.md) | 연구 배경, 이론, 실험 설계 |
| [MLFLOW.md](docs/MLFLOW.md) | 실험 추적, MLflow UI 사용법 |

## 라이선스

MIT License
