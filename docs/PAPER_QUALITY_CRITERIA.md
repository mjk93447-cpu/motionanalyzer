# 논문 완성도 평가 기준 (문헌 조사)

IMRaD, 재현성, QuOCCA 등 문헌 기반 논문 품질 평가 기준.

---

## 1. IMRaD 구조 평가 (Standard Quality Assessment)

**출처**: IMRaD model (Introduction, Methods, Results, and Discussion), STEM 표준.

| 섹션 | 평가 질문 | 기준 |
|------|-----------|------|
| **Introduction** | Why was this study done? | 문헌 검토, 기존 지식 간극, 연구가 해결하는 문제 명시 |
| **Methods** | How was this study done? | 연구의 conceptual epicenter; 재현 가능한 절차 |
| **Results** | What was found? | 객관적 결과 제시 |
| **Discussion** | What does it mean? | 해석, 한계, 시사점 |

---

## 2. 연구 품질 일반 기준 (Qualitative/Quantitative)

**출처**: Standard Quality Assessment Criteria for Evaluating Primary Research Papers.

| 차원 | 내용 |
|------|------|
| **Reliability** | 방법·결과의 일관성, 재현 가능성 |
| **Validity** | 측정·추론의 타당성 |
| **Feasibility** | 실행 가능성, 자원·시간 고려 |
| **Utility** | 실무·연구에의 활용가치 |

---

## 3. 재현성 체크리스트 (ML Research)

**출처**: NeurIPS 2019 Reproducibility Program, JMLR 2020, arXiv 2511.21354.

| 항목 | 설명 |
|------|------|
| 코드 공개 | 학습·평가 스크립트 제공 |
| 데이터·전처리 | 데이터셋, split, 전처리 공개 또는 명시 |
| 하이퍼파라미터 | 시드, epoch, batch, threshold 등 명시 |
| 평가 프레임워크 | 표준 harness, 메트릭 정의 |
| Overfitting 모니터링 | LOR, COS 등 (선택) |

**대규모 검증**: 소규모(fp_focused 851)뿐 아니라 100k 수준 교차 검증으로 일반화·신뢰성 확보.

---

## 4. QuOCCA (Quality Output Checklist and Content Assessment)

**출처**: BMJ Open 2022, QuOCCA tool.

| 카테고리 | 평가 내용 |
|----------|-----------|
| Research quality | 방법론, 설계, 분석의 적절성 |
| Reproducibility | 코드·데이터·절차의 재현 가능성 |
| Reporting | 결과 보고의 투명성·완전성 |

---

## 5. 논문 평가 루브릭 (Cornell 등)

| 차원 | 내용 |
|------|------|
| Knowledge integration | 문헌·개념 통합 |
| Topic focus | 명확한 주제·가설 |
| Depth of discussion | 충분한 논의·해석 |
| Cohesiveness | 섹션 간 논리적 연결 |

---

## 6. FPCB 논문 적용 체크리스트

| # | 기준 | 상태 |
|---|------|:----:|
| 1 | Introduction: 기존 연구 간극·목표 명시 | - |
| 2 | Methods: 데이터·모델·임계값 선택 절차 명시 | - |
| 3 | Results: fp_focused + 100k CM·메트릭 제시 | - |
| 4 | Discussion: 한계·합성→실제 전이 시사점 | - |
| 5 | 재현성: 스크립트·설정·시드 문서화 | - |
| 6 | 대규모 검증: 100k 교차 검증 결과 포함 | - |

---

**문헌**:
- IMRaD: Sollaci & Pereira (2004), Int J Ed Psych; Springer "Improving the writing of research papers"
- QuOCCA: BMJ Open 2022, e060976
- NeurIPS 2019 Reproducibility: arXiv 2003.12206
- ML Experimentation: arXiv 2511.21354
