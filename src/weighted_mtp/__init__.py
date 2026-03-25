"""
Weighted Next-Token Prediction (WNTP) 프로젝트

HuggingFace LlamaForCausalLM 기반 NTP 파이프라인에 토큰별 가중치 학습을 적용.
4가지 weight mode를 지원한다:
- uniform: 표준 NTP (baseline)
- critic: TAW - Token Advantage Weighting (핵심 연구 기여)
- random: Random-Matched (LogNormal 분포, 대조군)
- shuffled: Shuffled (Critic 가중치 위치 셔플, 대조군)

주요 모듈:
- cli: 사용자 진입점
- core: 설정, 로깅, 레지스트리
- data: 데이터 로딩 및 전처리
- models: HuggingFace LlamaForCausalLM (Policy), LlamaModel (Value/Critic)
- value_weighting: TD error 기반 토큰 가중치 계산
- pipelines: 학습/평가 파이프라인
- runtime: 환경 초기화, MLflow 연동
- utils: 공통 유틸리티
"""

__version__ = "0.2.0"
