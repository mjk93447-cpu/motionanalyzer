# 로드맵 시나리오 및 준비도

**목적**: 로드맵 진행 시 발생 가능한 작업 시나리오별 준비도 평가 및 도구 매핑  
**목표**: 상위 4% (96점 이상) 시나리오 기반 세팅 대비

---

## 시나리오 정의 (로드맵 기반)

| ID | 시나리오 | 로드맵 연계 | 핵심 과제 |
|----|----------|-------------|-----------|
| S01 | Temporal 모델 개선 | 목표1, Phase B.3 | ROC AUC 0.25→개선, 시계열 구조 보존 |
| S02 | CPD 정확도 향상 | 목표1, Phase B.5 | 크랙 발생 시점(프레임) 정확도 |
| S03 | 충격파·진동 감지 강화 | 목표1 | 합성 데이터·시계열 특징 정교화 |
| S04 | 고급 특징 과적합 관리 | 목표2 | DREAM+Advanced, 특징 선택 |
| S05 | 전체 파이프라인 실행 | Phase A/B | 100k 생성→ML 분석→리포트 |
| S06 | 논문/리포트 작성 | 배포 | DOCX/PPT 생성 |
| S07 | EXE 빌드·배포 | Phase A.1 | 경량/ML 포함 빌드, 검증 |
| S08 | QA 게이트 검증 | 사전 평가 | 데이터 품질, DREAM 검증 |
| S09 | 앙상블 가중치 최적화 | 목표2 | Ensemble 성능 향상 |
| S10 | Goal1/Goal2 ML 평가 | 목표 연계 | evaluate_goal1_ml, evaluate_goal2_ml |
| S11 | 배치 분석 (Phase D) | Phase D | 다수 데이터셋 일괄 분석 |
| S12 | Phase C 준비 | Phase C | Few-shot, Contrastive Learning 설계 |

---

## 시나리오별 필수 도구

| ID | 필수 스크립트 | 필수 데이터 | 필수 문서 |
|----|---------------|-------------|-----------|
| S01 | validate_temporal_synthetic, benchmark_phase_b | ml_dataset 또는 합성 | PHASE_B_INSIGHTS |
| S02 | validate_cpd_optimization, evaluate_goal1_cpd | crack 합성 | CHANGEPOINT_DETECTION |
| S03 | validate_enhanced_synthetic | crack 시나리오 | SYNTHETIC_DATA_SPEC |
| S04 | analyze_advanced_features_overfitting, validate_advanced_features | ml_dataset | PHASE_B_INSIGHTS |
| S05 | run_full_pipeline, generate_ml_dataset, analyze_crack_detection | (생성) | PIPELINE_SETUP_COMPLETE |
| S06 | generate_final_report_docx, generate_final_report_ppt | reports/ | RELEASE_NOTES_TEMPLATE |
| S07 | build_exe, test_build_exe | - | EXE_LOCAL_TEST_GUIDE |
| S08 | evaluate_synthetic_dataset_quality, validate_enhanced_dream | (생성) | - |
| S09 | analyze_crack_detection (ensemble), benchmark | ml_dataset | PHASE_B_INSIGHTS |
| S10 | evaluate_goal1_ml, evaluate_goal2_ml, evaluate_goals_summary | ml_dataset | PROJECT_GOALS |
| S11 | (배치 스크립트) | 다수 데이터셋 | ANALYSIS_SCENARIOS |
| S12 | (설계 문서) | - | - |

---

## 준비도 평가 기준 (0-100)

- **도구 존재** (40): 필수 스크립트 존재 여부
- **실행 가능** (30): venv, 의존성, 데이터 준비
- **문서 연계** (20): 시나리오별 가이드 존재
- **원클릭 실행** (10): `scripts/agent_tools/run_scenario_Sxx.ps1`

---

## 시나리오별 원클릭 실행

| ID | 명령 |
|----|------|
| S01 | `.\scripts\agent_tools\run_scenario_S01.ps1` |
| S02 | `.\scripts\agent_tools\run_scenario_S02.ps1` |
| S03 | `.\scripts\agent_tools\run_scenario_S03.ps1` |
| S04 | `.\scripts\agent_tools\run_scenario_S04.ps1` |
| S05 | `.\scripts\agent_tools\run_scenario_S05.ps1` |
| S06 | `.\scripts\agent_tools\run_scenario_S06.ps1` |
| S07 | `.\scripts\agent_tools\run_scenario_S07.ps1` |
| S08 | `.\scripts\agent_tools\run_scenario_S08.ps1` |
| S09 | `.\scripts\agent_tools\run_scenario_S09.ps1` |
| S10 | `.\scripts\agent_tools\run_scenario_S10.ps1` |
| S11 | `.\scripts\agent_tools\run_scenario_S11.ps1` |
| S12 | `.\scripts\agent_tools\run_scenario_S12.ps1` |
| S13 | `.\scripts\agent_tools\run_scenario_S13.ps1` |
| S14 | `.\scripts\agent_tools\run_scenario_S14.ps1` |
| S15 | `.\scripts\agent_tools\run_scenario_S15.ps1` |
