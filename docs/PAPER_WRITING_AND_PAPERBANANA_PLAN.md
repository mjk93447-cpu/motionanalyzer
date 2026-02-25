# 논문 작성 및 Paper Banana 통합 계획

**목적**: 진척도 점검, 분석/평가 진행도 정리, Paper Banana 실제 사용법 반영, 완성도 높은 최종 논문을 한 번에 정리하기 위한 실행 계획.

---

## Part 1. 프로젝트 진척도 요약

### 1.1 개발 목표 대비 현황

| 목표 | 상태 | 비고 |
|------|:----:|------|
| Normal FP Rate ≤ 0.1% | ✅ 달성 | normal_fp_max=0 → FP=0, 0.00% |
| Precision 99%+ (단일 목표) | ✅ | Ensemble 100% |
| fp_focused 데이터셋 검증 | ✅ | 20k, train/val/test |
| 100k 교차 평가 (inference-only) | 🔄 | 스크립트·문서 완료, 실행 시 결과 반영 |
| Phase 3 (추가 개선 반복) | ⏳ | 필요 시 margin/percentile 등 |

### 1.2 주요 산출물 위치

| 산출물 | 경로 | 비고 |
|--------|------|------|
| fp_focused 분석 결과 | `reports/crack_detection_analysis/analysis.json` | TN/FP/FN/TP, 임계값 |
| 인사이트 | `reports/crack_detection_analysis/insights.md` | 모델별 요약 |
| 100k inference 결과 | `reports/crack_detection_analysis/analysis_100k_inference.json` | 실행 후 생성 |
| Confusion matrix PNG | `reports/crack_detection_analysis/confusion_matrix_*.png` | analyze 또는 evaluate 스크립트 실행 시 |
| Vector map PNG | `reports/crack_detection_analysis/vector_map_*.png` | analyze 스크립트 실행 시 |
| 최종 보고서(Word) | `reports/deliverables/FPCB_Crack_Detection_Final_Report.docx` | generate_final_report_docx.py |
| PPT | `reports/deliverables/FPCB_Crack_Detection_Final_Report.pptx` | generate_final_report_ppt.py |

### 1.3 분석/평가 진행도

| 단계 | 스크립트 | 입력 | 출력 | 상태 |
|------|----------|------|------|:----:|
| fp_focused 학습·평가 | `analyze_crack_detection.py` | fp_focused base_dir | analysis.json, confusion PNG, insights | ✅ |
| 100k inference-only | `evaluate_100k_inference_only.py` | fp_focused norm + 100k eval | analysis_100k_inference.json, 100k confusion PNG | 🔄 |
| Word 논문 생성 | `generate_final_report_docx.py` | analysis.json + analysis_dir PNG | .docx | ✅ |
| PDF 변환 | docx2pdf 또는 Word 수동 | .docx | .pdf | 수동/스크립트 |
| PPT 생성 | `generate_final_report_ppt.py` | .docx 또는 동일 소스 | .pptx | ✅ |

**갭**: (1) 논문용 **방법론/아키텍처 다이어그램**이 수동 또는 웹만 안내되어 있음 → Paper Banana CLI/API로 자동화 필요. (2) **이미지·분석·인사이트가 최종 논문에 한 번에 반영**되도록 단일 파이프라인 정리 필요.

---

## Part 2. Paper Banana 실제 사용 방법

### 2.1 공식 도구 (pip + CLI)

- **패키지**: [PyPI paperbanana](https://pypi.org/project/paperbanana/) (v0.1.2)
- **저장소**: https://github.com/llmsresearch/paperbanana
- **필요 조건**: Python 3.10+, **Google Gemini API 키** (무료 발급: [Google AI Studio](https://makersuite.google.com/app/apikey))

### 2.2 설치 및 설정

```powershell
# 설치
pip install paperbanana

# 최초 1회: API 키 설정 (대화형)
paperbanana setup
# → 브라우저에서 Gemini API 키 발급 후 .env에 저장

# 또는 수동
# 프로젝트 루트 또는 사용자 디렉터리에 .env 생성
# GOOGLE_API_KEY=your-gemini-api-key-here
```

### 2.3 방법론 다이어그램 생성 (CLI)

```powershell
# 방법론 텍스트 파일 준비 후
paperbanana generate ^
  --input docs/paperbanana_inputs/fpcb_methodology.txt ^
  --caption "FPCB bending crack detection pipeline: feature extraction, DREAM and PatchCore, ensemble" ^
  --output reports/deliverables/figures/fig_methodology.png
```

**옵션**: `--iterations 3` (기본 3회 refinement), `--config path/to/config.yaml`

### 2.4 통계 플롯 생성 (CLI)

```powershell
# CSV/JSON 결과로 차트 생성
paperbanana plot ^
  --data reports/crack_detection_analysis/analysis.json ^
  --intent "Bar chart comparing DREAM, PatchCore, Ensemble precision and recall" ^
  --output reports/deliverables/figures/fig_model_comparison.png
```

(참고: plot은 CSV/JSON 구조에 따라 intent를 조정해야 할 수 있음.)

### 2.5 Python API (스크립트 내 자동화)

```python
import asyncio
from paperbanana import PaperBananaPipeline, GenerationInput, DiagramType
from paperbanana.core.config import Settings

settings = Settings(
    vlm_provider="gemini",
    image_provider="google_imagen",
    refinement_iterations=3,
)
pipeline = PaperBananaPipeline(settings=settings)

with open("docs/paperbanana_inputs/fpcb_methodology.txt") as f:
    text = f.read()

result = asyncio.run(pipeline.generate(GenerationInput(
    source_context=text,
    communicative_intent="Overview of FPCB crack detection pipeline.",
    diagram_type=DiagramType.METHODOLOGY,
)))
# result.image_path → 저장 경로
```

### 2.6 FPCB용 방법론 텍스트 (입력 파일 예시)

`docs/paperbanana_inputs/fpcb_methodology.txt` 로 저장해 두고 `--input` 으로 사용:

```
FPCB bending crack detection pipeline:
1) Input: contour trajectory frames from bending process
2) Feature extraction: velocity, acceleration, curvature, strain surrogate per frame
3) Two anomaly detectors in parallel: DREAM (reconstruction + discriminator) and PatchCore (memory bank + k-NN)
4) Ensemble: logical AND - predict crack only when both models agree
5) Output: binary label (normal / crack)
```

시스템 아키텍처용 별도 파일 예: `fpcb_architecture.txt`

```
Two-branch anomaly detection architecture:
Left branch: DREAM model with autoencoder reconstruction and discriminative head
Right branch: PatchCore with normal feature memory bank and distance-based scoring
Both branches feed into AND gate for final crack prediction
```

### 2.7 MCP 연동 (선택)

Cursor/Claude Code에서 Paper Banana MCP 서버를 쓰면 `generate_diagram`, `generate_plot`, `evaluate_diagram` 도구로 호출 가능. 설정 예:

```json
{
  "mcpServers": {
    "paperbanana": {
      "command": "uvx",
      "args": ["--from", "paperbanana[mcp]", "paperbanana-mcp"],
      "env": { "GOOGLE_API_KEY": "your-google-api-key" }
    }
  }
}
```

---

## Part 3. 논문에 이미지·분석·인사이트를 한 번에 반영하는 완결 계획

### 3.1 데이터 흐름 (단일 소스 → 논문)

```
[분석 실행]
  analyze_crack_detection.py (fp_focused)     → analysis.json, insights.md, PNG들
  evaluate_100k_inference_only.py (선택)     → analysis_100k_inference.json, 100k PNG들

[논문용 도표 생성]
  Paper Banana (CLI 또는 Python)
  --input fpcb_methodology.txt
  --output reports/deliverables/figures/fig_*.png

[Word 논문 생성]
  generate_final_report_docx.py
  - 읽음: analysis_dir/analysis.json, analysis_dir/*.png
  - (선택) 100k 결과 반영 시: 스크립트 확장으로 analysis_100k_inference.json 또는 insights_100k 병합
  - 출력: deliverables/FPCB_Crack_Detection_Final_Report.docx

[PDF/PPT]
  docx2pdf 또는 Word 수동 저장 → .pdf
  generate_final_report_ppt.py → .pptx
```

### 3.2 실행 순서 체크리스트 (한 번에 논문까지)

| 순서 | 작업 | 명령/조치 | 산출물 |
|------|------|-----------|--------|
| 1 | 분석 이미지·JSON 확보 | `python scripts/analyze_crack_detection.py --base-dir data/synthetic/ml_dataset_fp_focused --dataset-level-eval --normal-fp-max 0` | analysis.json, confusion_matrix_*.png, vector_map_*.png, insights_summary.png, insights.md |
| 2 | (선택) 100k 평가 | `python scripts/evaluate_100k_inference_only.py` | analysis_100k_inference.json, confusion_matrix_100k_*.png |
| 3 | Paper Banana 설정 | `pip install paperbanana` 후 `paperbanana setup` | .env (GOOGLE_API_KEY) |
| 4 | 방법론/아키텍처 도표 생성 | `paperbanana generate -i docs/paperbanana_inputs/fpcb_methodology.txt -c "..." -o reports/deliverables/figures/fig_methodology.png` | fig_methodology.png 등 |
| 5 | Word 논문 생성 | `python scripts/generate_final_report_docx.py` | FPCB_Crack_Detection_Final_Report.docx |
| 6 | PDF 생성 | `docx2pdf` 또는 Word에서 저장 | FPCB_Crack_Detection_Final_Report.pdf |
| 7 | PPT 생성 | `python scripts/generate_final_report_ppt.py` | FPCB_Crack_Detection_Final_Report.pptx |

### 3.3 Word 스크립트가 참조하는 이미지 (현재)

- `analysis_dir/vector_map_normal.png`
- `analysis_dir/vector_map_crack.png`
- `analysis_dir/confusion_matrix_dream.png`
- `analysis_dir/confusion_matrix_patchcore.png`
- `analysis_dir/confusion_matrix_ensemble.png`
- `analysis_dir/insights_summary.png`

→ **1번** 분석 실행을 한 번 하면 위 파일이 채워짐. Paper Banana로 만든 그림은 `reports/deliverables/figures/` 에 두고, 필요 시 `generate_final_report_docx.py` 에 해당 경로를 추가해 Methods 등에 삽입하면 됨.

### 3.4 100k 결과를 논문에 반영하려면

- `generate_final_report_docx.py` 에서 `analysis_100k_inference.json` 을 읽는 분기 추가.
- 또는 "Evaluation on 100k" 절을 추가하고, 해당 절의 표/수치는 `analysis_100k_inference.json` 의 `models` 필드에서 가져오도록 수정.
- 100k confusion figure: `confusion_matrix_100k_ensemble.png` 등을 같은 방식으로 경로만 추가해 삽입.

---

## Part 4. 실행 방법 요약 (복사해서 사용)

### 4.1 최소 실행 (fp_focused 기준 논문)

```powershell
cd c:\motionanalyzer

# 1) 분석 (이미지 + JSON)
python scripts/analyze_crack_detection.py --base-dir data/synthetic/ml_dataset_fp_focused --dataset-level-eval --normal-fp-max 0

# 2) 논문 Word 생성 (analysis.json + PNG 사용)
pip install python-docx
python scripts/generate_final_report_docx.py

# 3) PDF (선택)
pip install docx2pdf
python -c "from docx2pdf import convert; convert('reports/deliverables/FPCB_Crack_Detection_Final_Report.docx', 'reports/deliverables/FPCB_Crack_Detection_Final_Report.pdf')"

# 4) PPT
python scripts/generate_final_report_ppt.py
```

### 4.2 Paper Banana 포함 (방법론 도표까지)

```powershell
pip install paperbanana
paperbanana setup
# .env에 GOOGLE_API_KEY 설정 후:

mkdir -p reports/deliverables/figures
# fpcb_methodology.txt 내용은 docs/paperbanana_inputs/ 에 저장해 둠
paperbanana generate -i docs/paperbanana_inputs/fpcb_methodology.txt -c "FPCB crack detection pipeline" -o reports/deliverables/figures/fig_methodology.png
```

이후 Word 스크립트에 `reports/deliverables/figures/fig_methodology.png` 삽입 로직을 추가하면, 논문에 방법론 그림이 한 번에 포함됨.

### 4.3 100k 평가 포함

```powershell
python scripts/evaluate_100k_inference_only.py
# → analysis_100k_inference.json, confusion_matrix_100k_*.png 생성
# 논문에 반영하려면 generate_final_report_docx.py 확장 필요 (위 3.4 참고)
```

---

## Part 5. 논문 작성 전 사전 조치

| 조치 | 설명 |
|------|------|
| **레거시 아카이빙** | 기존 docx/pptx/pdf/이미지/영상을 `reports/archive/legacy_deliverables/` 로 이동. `.\scripts\archive_legacy_deliverables.ps1` 실행. |
| **수치 출처** | 모든 테스트 결과·데이터셋 숫자는 `analysis.json`, `analysis_100k_inference.json` 에서만 사용. 아카이브 수치 절대 혼용 금지. |
| **참고 논문** | DRAEM, PatchCore, 앙상블·FP 개선 관련 3+ 편 정리: [REFERENCE_PAPERS_FOR_PAPER_WRITING.md](REFERENCE_PAPERS_FOR_PAPER_WRITING.md) |

---

## Part 6. 추후 계획 (우선순위)

| 우선순위 | 항목 | 내용 |
|----------|------|------|
| P0 | Paper Banana 입력 파일 고정 | `docs/paperbanana_inputs/fpcb_methodology.txt`, `fpcb_architecture.txt` 생성 및 버전 관리 |
| P0 | 한 번에 실행 스크립트 | `scripts/run_paper_pipeline.ps1` 또는 `run_paper_pipeline.py`: 분석 → (100k 선택) → Paper Banana → docx → (pdf/ppt) |
| P1 | Word 스크립트에 Paper Banana 그림 경로 추가 | Methods 절에 `fig_methodology.png` 등 삽입 |
| P1 | 100k 결과를 Word에 선택 반영 | analysis_100k_inference.json 읽어 표/그림 추가 |
| P2 | Paper Banana plot 연동 | analysis.json 또는 CSV 요약으로 `paperbanana plot` 호출해 비교 차트 자동 생성 |
| P2 | MCP 연동 (선택) | Cursor에서 MCP로 diagram/plot 생성 워크플로 정리 |

---

**문서 버전**: 1.0  
**최종 갱신**: 2026-02-23  
**참조**: [PAPERBANANA_FIGURE_GUIDE.md](PAPERBANANA_FIGURE_GUIDE.md), [FINAL_REPORT_EVALUATION_AND_STRATEGY.md](../reports/deliverables/FINAL_REPORT_EVALUATION_AND_STRATEGY.md), [PyPI paperbanana](https://pypi.org/project/paperbanana/), [GitHub llmsresearch/paperbanana](https://github.com/llmsresearch/paperbanana)
