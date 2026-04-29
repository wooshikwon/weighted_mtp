### (0) Front Matter

- **Title (확정)**
  - *Token-Weighted MTP via GAE-based Advantage and Pairwise Value Learning for Code Generation*

- **Keywords**
  - Weighted SFT, Multi-Token Prediction, Advantage Weighting, Pairwise Ranking, Verifiable Reward, Offline RL for LLMs, Code Generation

- **Contributions (5개)**
  1. **GAE-weighted CE의 LLM 코드 생성 적용 탐색**: 로봇 제어 도메인에서 주로 쓰인 GAE 기반 advantage를 코드 생성 토큰 가중치로 전이하여 out-of-domain에서 긍정적 신호 관찰
  2. **Pairwise Ranking 기반 Value 학습**: pointwise regression/MSE 대신 정답/오답 쌍 비교(Bradley-Terry)로 value를 학습해 코드 도메인에 적합한 안정적 신호 제공
  3. **Sparse reward 안정화 레시피**: whitening + EMA 정규화 + exp-weight(\(w_t=\exp(\hat A_t/\beta)\)) + clipping으로 학습 불안정 제어
  4. **Out-of-domain 전이 관찰**: 제한된 학습 데이터(50K/3.6M, 1.4%)에서 HumanEval/MBPP/GSM8K 개선 관찰 (인과관계 미확정)
  5. **한계의 정직한 공개**: MTP 자체의 필요성(NTP 대비) 미검증, critic 정확도(63%) 한계 및 in-domain 하락 분석

---

### (1) Abstract

- **1문단 구성 (150-250 words)**
  - **문제**: 표준 MTP는 미래 토큰을 균등 학습 → 중요한 토큰과 비중요 토큰을 구분 못함
  - **방법**: 독립 critic으로 토큰 value \(V_t\) 추정 → GAE로 advantage \(A_t\) 계산 → AWR 형태의 exp-weight로 MTP CE 가중화
  - **특징**: offline weighted SFT 성격(학습 안정성), inference-time overhead 없음
  - **결과(숫자 명시)**: HumanEval +2.86%, MBPP +3.00%, GSM8K +1.07% (CodeContests -1.05%도 함께 정직하게)
  - **해석**: out-of-domain 전이와 데이터 효율(1.4% 샘플로 개선) 강조, in-domain 저하 원인/향후 과제 한 줄

---

### (2) Introduction

#### 2.1 배경: 코드 생성과 학습 신호의 문제

- **코드 생성의 특성**
  - 긴 시퀀스, sparse/episodic 보상(정답/오답), 결정적 토큰(함수명/조건/반환 등)의 영향이 큼
- **기존 접근의 한계**
  - MTP는 구조적으로 빠르고 좋지만, **학습 손실이 토큰 중요도 차이를 반영하지 못함(균등 CE)**

#### 2.2 핵심 가설 및 연구 질문

- **가설**: "Not All Tokens Are What You Need" — 중요 토큰에 더 큰 학습 자원을 배분하면 동일 FLOPs에서 성능/수렴이 개선된다

- **Research Questions (Conditional 형태)**
  - **RQ1**: GAE-weighted CE는 **어떤 조건**에서 코드 생성 성능을 개선하는가?
    - RQ1a: Out-of-domain (짧은 함수 단위, HumanEval/MBPP)에서 효과는?
    - RQ1b: In-domain (긴 경쟁 프로그래밍, CodeContests)에서 효과는?
  - **RQ2**: Token-level value를 어떻게 안정적으로 학습할 수 있는가?
    - Pairwise ranking vs MSE 비교
    - Length-balanced sampling 효과
  - **RQ3**: Critic 품질이 downstream 성능에 미치는 영향은? (탐색적, 미완결)

#### 2.3 제안 방법 한눈에 보기 (Figure 1 유도)

- **2-phase 파이프라인**
  - Phase 1: Critic 학습 (정답/오답 pairwise ranking)
  - Phase 2: Critic frozen + GAE/EMA/whitening → exp-weighted MTP loss로 policy 학습

#### 2.4 주요 기여 요약

- Abstract와 동일하되, "training-time only", "decoupled critic", "pairwise ranking" 강조
- MTP는 **구현 선택**이며, NTP 대비 필요성은 별도 검증 필요함을 명시

---

### (3) Related Work

#### 3.1 Advantage-weighted 계열 (AWR/APA)

- **AWR (Peng et al., 2019)**: \(\exp(A/\beta)\) 형태의 exp-weighted supervised objective
- **APA**: squared error 기반 안정화 (online)
- **WMTP 차별점 (오해 방지 핵심 문장)**:
  > WMTP는 **AWR 전체를 재현하는 것이 아니라** *\(\exp(A/\beta)\) 가중치 형태만 차용*하고, **advantage 계산은 GAE**, **value 학습은 pairwise ranking**으로 대체한다.

- **Table 1a: AWR vs WMTP 비교**

  | 구성요소 | AWR (Peng et al., 2019) | WMTP (Ours) |
  |----------|-------------------------|-------------|
  | 도메인 | 로봇 제어 | LLM 코드 생성 |
  | Advantage 계산 | TD(λ) return 기반 | **GAE (TD-error 누적)** |
  | Value 학습 | Regression (MSE) | **Pairwise Ranking** |
  | 가중치 형태 | exp(A/β) | exp(A/β) (동일 형태만 차용) |

#### 3.2 Implicit Q / Offline RL (Q-SFT, IQL/ILQL, OREO)

- **Q-SFT**: 확률을 Q로 해석, bellman target로 가중
- **ILQL**: inference-time logit perturbation 필요 (추론 비용 증가)
- **OREO**: policy gradient/odds ratio 성격 (불안정 가능)
- **WMTP 위치**: **weighted SFT (안정)** + **training-time 가중 (추론비용 0)** + **explicit value (해석/디버깅 용이)**

#### 3.3 Multi-Token Prediction

- Meta MTP (독립 head로 미래 토큰 예측, 손실은 균등 평균)
- **WMTP 기여**: "미래 토큰 예측 head" 구조는 유지하면서 **토큰 학습 신호를 가치 기반으로 재분배**

- **Table 1b: Related Work 비교표 (권장)**

  | Method | Loss | Weighting | Value Learning | Extra Model | MTP Support | Inference Overhead |
  |--------|------|-----------|----------------|-------------|-------------|-------------------|
  | Standard MTP | CE | Uniform | - | - | Yes | 0 |
  | AWR | Weighted CE | exp(A/β) | Regression | Critic | No | 0 |
  | ILQL | Modified CE | Q-based | Implicit | - | No | Yes |
  | **WMTP** | **Weighted CE** | **exp(A/β)** | **Pairwise** | **Critic** | **Yes** | **0** |

---

### (4) Method (WMTP)

#### 4.1 Problem Setup & Notation

- 시퀀스 \(y_{1:T}\), 입력 \(x\), policy \(\pi_\theta\)
- MTP head \(k \in \{1..K\}\), logits \(z_{t,k}\)
- Loss mask: Instruction/Input 마스킹, Output만 학습

#### 4.2 Multi-Token Prediction Objective (Baseline)

- **Baseline 손실**
  \[L_{\text{MTP}} = \frac{1}{K}\sum_{k=1}^{K}\sum_t \mathrm{CE}(z_{t,k}, y_{t+k})\]
- Config 참조: `configs/production/baseline.yaml`

#### 4.3 Phase 1: Independent Critic via Pairwise Ranking

- **데이터 구성**
  - 같은 problem_id 내 correct/incorrect 솔루션 쌍
  - Length-balanced stratified sampling (17 bins, 길이 편향 제거)
  - 총 200K pairs (400K samples)

- **모델**
  - Backbone: Sheared-LLaMA 2.7B (HF LlamaModel 계열)
  - Value head: MLP (hidden_dim × 4 → 1)
  - 출력: \(V_t \in \mathbb{R}\) (토큰별)

- **Pairwise ranking loss (Bradley-Terry)**
  - 시퀀스 평균 (출력 토큰만): \(\bar V^{+}, \bar V^{-}\)
  - \[L_{\text{pair}} = -\log \sigma(\bar V^{+} - \bar V^{-})\]

- Config 참조: `configs/production/critic_mlp_pairwise.yaml`

- **Figure 2 (권장)**: Pairwise ranking 학습 도식 (correct/incorrect → critic → V → loss)

#### 4.4 Phase 2: Token Advantage Estimation with GAE

- **목표**: Token-level 중요도 (credit assignment)를 \(V_t\) 변화로 추정

- **표준 GAE 정의**
  \[\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)\]
  \[A_t = \sum_{l \geq 0} (\gamma\lambda)^l \delta_{t+l}\]

- **본 구현의 단순화**
  - 코드 생성은 episodic task로 **중간 reward \(r_t = 0\)** (모든 \(t < T\))
  - Terminal reward \(r_T\)도 별도 사용하지 않음 (value 자체가 정답/오답 신호 반영)
  - 실제 구현:
    ```
    δ_t = γ V(s_{t+1}) - V(s_t)  # reward 항 없음, 순수 value 변화
    ```

- **설정**
  - \(\gamma = 1.0\) (episodic task, no discounting)
  - \(\lambda = 0.95\) (GAE 표준 권장값)

- **해석**
  - Value의 **변화량**이 토큰의 기여도를 나타냄
  - \(V\) 증가 → 정답 확률 상승에 기여한 토큰
  - \(V\) 감소 → 정답 확률 하락에 기여한 토큰

#### 4.5 AWR-Style Weighting + Stabilization

- **Whitening / 정규화**
  \[\hat A_t = \frac{A_t - \mu}{\sigma + \epsilon}\]
  - EMA running stats (배치간 스케일 안정화)
  - Warmup: 10 steps

- **Exp weighting + Clipping**
  \[w_t = \exp(\hat A_t / \beta)\]
  \[w_t \leftarrow \mathrm{clip}(w_t, w_{\min}, w_{\max})\]
  - 설정: \(\beta = 1.0\), \([w_{\min}, w_{\max}] = [0.1, 3.0]\)

- **분산 학습 동기화**: All-reduce로 rank 간 통계 동기화

#### 4.6 Final WMTP Objective (Weighted MTP CE)

\[L_{\text{WMTP}} = \frac{1}{K}\sum_{k=1}^{K}\sum_t w_t \cdot \mathrm{CE}(z_{t,k}, y_{t+k})\]

- Config 참조: `configs/production/verifiable.yaml`

#### 4.7 Implementation Details

- **데이터 파이프라인**
  - Alpaca template + labels=-100 마스킹 (Instruction/Input 제외)
  - Max length: 2048

- **학습 인프라**
  - Hardware: H100 80GB × 4
  - LoRA: rank=64, alpha=128, dropout=0.05 (Policy)
  - FSDP FULL_SHARD, mixed precision (bf16), activation checkpointing

- **추론 비용**
  - Phase 2 학습 시 critic forward 필요 (학습 비용 증가)
  - **Inference 시 추가 비용 없음** (critic 사용 안 함)

---

### (5) Experimental Setup

#### 5.1 Datasets

**CodeContests:**
| 항목 | 값 |
|------|-----|
| 전체 가용 데이터 | ~3.6M samples |
| **실제 사용량** | **50K samples (1.4%)** |
| Critic 학습 | 200K pairs (정답/오답 쌍) |
| Policy 학습 | 200K samples (정답 only) |
| Max length | 2048 |

**평가 데이터셋:**
| Dataset | 목적 | 평가 지표 |
|---------|------|----------|
| HumanEval | Out-of-domain (짧은 함수) | Pass@k |
| MBPP | Out-of-domain (짧은 함수) | Pass@k |
| GSM8K | Out-of-domain (수학 추론) | Accuracy |
| CodeContests | In-domain (경쟁 프로그래밍) | Pass@k |

#### 5.2 Models

**Policy Model (Meta-LLaMA MTP 7B):**
| 항목 | 값 |
|------|-----|
| Base parameters | 7B |
| MTP heads | K=4 |
| **LoRA parameters** | **190.8M (2.73%)** |
| LoRA config | rank=64, alpha=128, dropout=0.05 |
| Target modules | wq, wk, wv, wo, w1, w2, w3 |

**Critic Model (Sheared-LLaMA 2.7B):**
| 항목 | 값 |
|------|-----|
| Base parameters | 2.7B |
| **LoRA parameters** | **119.5M (4.43%)** |
| LoRA config | rank=32, alpha=64, dropout=0.1 |
| Value head | MLP (hidden × 4 → 1) |
| **Final Val Accuracy** | **63.1%** |

#### 5.3 Training Details

**Table 4: 3-Config 하이퍼파라미터 비교**

| Parameter | Baseline | Critic | Verifiable (WMTP) |
|-----------|----------|--------|-------------------|
| Batch size (per GPU) | 12 | 8 | 12 |
| Gradient accumulation | 2 | 2 | 2 |
| Learning rate | 1e-4 | 5e-5 (LoRA), 1e-4 (head) | 1e-4 |
| Epochs | 1.0 | 1.0 | 1.0 |
| LR scheduler | Cosine | Cosine | Cosine |
| Warmup ratio | 0.03 | 0.03 | 0.03 |
| td_lambda | - | - | 0.95 |
| beta | - | - | 1.0 |
| weight_clip | - | - | [0.1, 3.0] |
| EMA momentum | - | - | 0.1 |

#### 5.4 Evaluation Protocol

- **Pass@K**: k ∈ {1, 10, 100}
- **Generation**: temperature=0.8, num_samples 문제당 200
- **Execution**: 샌드박스 환경에서 테스트 케이스 실행

---

### (6) Results

#### 6.1 Main Results

**Table 5: Baseline vs WMTP 성능 비교**

| Benchmark | Baseline | WMTP | Δ | 특성 |
|-----------|----------|------|---|------|
| HumanEval | (base) | +2.86% | ↑ | Out-of-domain |
| MBPP | (base) | +3.00% | ↑ | Out-of-domain |
| GSM8K | (base) | +1.07% | ↑ | Out-of-domain |
| CodeContests | (base) | -1.05% | ↓ | In-domain |

**해석:**
- **Out-of-domain 일관된 개선**: 짧은 함수 단위에서 토큰 가중화가 효과적으로 작동했을 가능성
- **In-domain 하락**: 긴 시퀀스에서 critic 품질/GAE credit assignment 한계

#### 6.2 Training Diagnostics

**Weight 통계 (last-final-verifiable):**

| Metric | Value | 해석 |
|--------|-------|------|
| weight_mean | 1.328 | 평균적으로 upweighting |
| weight_std | 0.981 | 적절한 분산 |
| weight_min | 0.100 | Clipping 하한 |
| weight_max | 3.000 | Clipping 상한 |
| **clipping_ratio** | **17.9%** | 극단값 제어 작동 |

**Figure 3 (권장)**: 학습 스텝에 따른 weight_mean / clip_ratio / val_loss 변화

**Figure 4 (권장)**: 토큰 위치별 평균 가중치 분포 (앞/중간/뒤)

---

### (7) Analysis & Discussion

#### 7.1 Out-of-domain 개선 관찰

**관찰 사실:**
- HumanEval / MBPP / GSM8K에서 일관된 개선 (+1.07% ~ +3.00%)
- 1.4% 학습 데이터만으로 전이 효과 관찰

**가설 (미검증):**
- 짧은 함수 단위에서 value 변화가 토큰 중요도를 잘 포착했을 가능성
- Weighted CE가 "결정적 토큰"에 더 큰 gradient 제공

**검증 필요:**
- 토큰별 가중치 분포 분석 (코드 토큰 타입별: 함수명, 조건문, return 등)
- Attention / gradient 분석으로 "중요 토큰" 가설 확인
- Random weight baseline 비교

#### 7.2 In-domain (CodeContests) 하락 원인

**관찰 사실:**
- CodeContests에서 -1.05% 하락

**가설 (우선순위순):**
1. **Critic 품질 한계**: Val accuracy 63%로 긴 시퀀스에서 신뢰도 저하
2. **GAE credit assignment 한계**: 긴 시퀀스 (>1000 tokens)에서 TD-error 누적 불안정
3. **데이터 coverage 부족**: 1.4% 샘플로 경쟁 프로그래밍 난이도 분포 커버 한계
4. **Critic backbone 용량**: 2.7B 모델로 7B policy의 value를 추정하는 구조적 한계

#### 7.3 Critic 품질과 최종 성능의 관계 (미검증)

**현재 상태 (사실만):**
- Critic Val Accuracy: 63.1%
- 이 수준에서도 out-of-domain 개선 관찰됨

**미해결 질문:**
1. 63% 정확도의 critic이 "유효한 토큰 가중치"를 제공한다고 말할 수 있는가?
2. Random / shuffled weight 대비 얼마나 나은가? (정보 효과 vs 가중치 효과 분리)
3. Critic accuracy가 70%, 80%로 올라가면 downstream 성능이 단조 증가하는가?

**검증 필요 실험 (Future Work):**
- Random weight / Shuffled advantage vs Value-based weight A/B 테스트
- Critic checkpoint sweep (accuracy 단계별) → 동일 policy 학습으로 효과 비교

---

### (8) Ablations

#### 8.1 수행 완료

| Ablation | 비교 | 결과 | 근거 |
|----------|------|------|------|
| **Value loss type** | MSE vs Pairwise | Pairwise 우세 (Val Acc 55% → 63%) | MLflow 실험 기록 |
| **TD vs GAE** | TD-error 직접 사용 vs GAE | GAE가 weight 분산 제어에 유리 | 학습 안정성 관찰 |

**Table 6: Ablation 결과 요약**

| Setting | Critic Val Acc | Weight Stability | 비고 |
|---------|----------------|------------------|------|
| MSE loss | 55% | 불안정 | Pointwise regression 한계 |
| **Pairwise loss** | **63%** | **안정** | 상대 비교가 코드에 적합 |
| TD-error only | - | 불안정 (high variance) | 스케일 변동 심함 |
| **GAE (λ=0.95)** | - | **안정** | Bias-variance trade-off |

#### 8.2 미수행 (Future Work)

| Ablation | 목적 | 중요도 |
|----------|------|--------|
| **Random weight baseline** | 정보 효과 vs 가중치 효과 분리 | **높음** |
| **NTP vs MTP** | MTP 필요성 검증 | **높음** |
| λ sweep (0.9, 0.95, 0.99) | GAE λ 민감도 | 중간 |
| β sweep (0.5, 1.0, 2.0) | 가중치 스케일 민감도 | 중간 |
| Critic accuracy sweep | 품질-성능 상관관계 | 중간 |
| EMA on/off | 정규화 필요성 | 낮음 |

---

### (9) Limitations

1. **MTP 필요성 미검증**
   - GAE-weighted CE가 NTP에서도 동일 효과인지 별도 실험 필요
   - 현재 결과만으로는 "MTP이기 때문에" 효과가 있다고 주장 불가

2. **Critic 성능 한계**
   - Val accuracy 63% 수준
   - 긴 코드에서 신뢰도 문제 (in-domain 하락 원인 중 하나)

3. **Random baseline 부재**
   - 가중치의 "정보적 가치" vs "가중치 자체의 효과" 분리 미검증
   - 향후 random weight 비교 필수

4. **계산 비용**
   - Phase 2에서 critic forward 비용 발생 (학습 비용 ~1.3x 증가 추정)
   - 단, inference overhead는 0

5. **일반화 범위**
   - 코드 도메인 중심 검증
   - 다른 생성 태스크 (요약, 대화, 번역) 확장 가능성은 미확정

---

### (10) Conclusion

**요약:**
- "독립 critic + pairwise value + GAE advantage + exp-weighted CE (AWR-style 형태)" 조합으로 **토큰 가중화 학습을 코드 생성에 적용**하고 효과/한계를 보고

**핵심 성과:**
- Out-of-domain 벤치마크에서 일관된 개선 (HumanEval +2.86%, MBPP +3.00%, GSM8K +1.07%)
- 1.4% 학습 데이터로 전이 효과 관찰

**핵심 한계:**
- In-domain (CodeContests) -1.05% 하락
- MTP 필요성 미검증, Random baseline 부재

**향후 과제:**
- NTP vs MTP 비교 실험
- Random weight baseline 추가
- Critic accuracy sweep으로 품질-성능 상관관계 검증

---

### (11) Reproducibility / Appendix

#### Appendix A: Config-to-Paper 매핑표

| 실험 | Config 파일 | 핵심 파라미터 |
|------|-------------|---------------|
| Baseline | `configs/production/baseline.yaml` | batch=12, lr=1e-4, epochs=1.0 |
| Critic | `configs/production/critic_mlp_pairwise.yaml` | pairwise loss, lr=5e-5/1e-4 |
| Verifiable | `configs/production/verifiable.yaml` | td_lambda=0.95, beta=1.0, clip=[0.1,3.0] |

#### Appendix B: 수식 모음

1. **Pairwise Ranking Loss**: \(L_{\text{pair}} = -\log \sigma(\bar V^{+} - \bar V^{-})\)
2. **GAE**: \(A_t = \sum_{l \geq 0} (\gamma\lambda)^l \delta_{t+l}\), where \(\delta_t = \gamma V(s_{t+1}) - V(s_t)\)
3. **Whitening**: \(\hat A_t = (A_t - \mu) / (\sigma + \epsilon)\)
4. **Exp-weight**: \(w_t = \mathrm{clip}(\exp(\hat A_t / \beta), w_{\min}, w_{\max})\)
5. **WMTP Objective**: \(L_{\text{WMTP}} = \frac{1}{K}\sum_{k}\sum_t w_t \cdot \mathrm{CE}(z_{t,k}, y_{t+k})\)

#### Appendix C: 구현 디테일

- **Alpaca Template**: Instruction/Input은 labels=-100으로 마스킹, Output만 학습
- **분산 통계 동기화**: All-reduce로 rank 간 EMA 통계 동기화
- **FSDP 설정**: FULL_SHARD (ZeRO-3), activation checkpointing, bf16 mixed precision

#### Appendix D: Experiment Journey (문제 해결 과정)

| 문제 | 원인 분석 | 해결책 | 효과 |
|------|----------|--------|------|
| Random sampling 과적합 | 특정 problem 반복 노출 | max_pairs_per_problem=60 | 다양성 확보 |
| 토큰 길이 불균형 | 짧은 코드 과대표집 | Length-balanced stratified sampling (17 bins) | 길이 분포 균형 |
| MSE loss 실패 | Pointwise value 학습 불안정 | Pairwise ranking loss | Val Acc 55%→63% |
| TD error 불안정 | 스케일 변동 심함 | GAE + EMA + whitening | Weight std 제어 |

---

### (12) 필수 그림/표 체크리스트

| 번호 | 유형 | 내용 | 필수도 |
|------|------|------|--------|
| Figure 1 | 도식 | 전체 파이프라인 (Phase1 critic / Phase2 weighted policy) | **필수** |
| Figure 2 | 도식 | Pairwise ranking 학습 다이어그램 | 권장 |
| Figure 3 | 그래프 | 학습 중 weight_mean / clip_ratio / val_loss dynamics | 권장 |
| Figure 4 | 그래프 | 토큰 위치별 평균 가중치 분포 | 선택 |
| Table 1a | 비교표 | AWR vs WMTP 구성요소 비교 | **필수** |
| Table 1b | 비교표 | Related work 포지셔닝 | 권장 |
| Table 4 | 설정표 | 3-config 하이퍼파라미터 비교 | **필수** |
| Table 5 | 결과표 | 메인 결과 (benchmark 성능) | **필수** |
| Table 6 | 결과표 | Ablation 결과 요약 | 권장 |
