# 오피스 PC 워크플로우 (RTX 2070 Super)

**대상**: 보안 네트워크, Cursor 불가. exe로 분석 후 리포트 이메일 전송 → Cursor Composer 검토.

---

## 1. exe 및 프로젝트 준비

1. GitHub 저장소 → **Actions** → 최신 `build-windows-exe` 워크플로우 실행
2. **Artifacts** → `motionanalyzer-windows-exe` 다운로드
3. 압축 해제 → `motionanalyzer-gui-ml.exe` (ML 포함 버전)
4. **프로젝트 폴더**를 오피스 PC로 복사 (scripts/, data/, reports/ 포함)
5. exe를 프로젝트 루트에 두고, **해당 폴더에서 exe 실행** (cwd = 프로젝트 루트)

---

## 2. 오피스 PC에서 실행

1. `motionanalyzer-gui-ml.exe` 실행
2. **Synthetic & Goals** 탭 이동
3. **Generate ML Dataset**: Small set 체크 후 생성 (또는 전체 1120건)
4. **Run Goal 1 (CPD)**, **Run Goal 1 (ML)**, **Run Goal 2 (ML)** 순서로 실행
5. **Summary** 클릭 → `goal_achievement_summary.md` 내용 확인

---

## 3. 리포트 수집

exe 실행 디렉터리 기준 `reports/` 폴더 (또는 exe와 동일 경로):

| 파일 | 용도 |
|------|------|
| `goal_achievement_summary.md` | 종합 요약 |
| `goal1_cpd_evaluation.json` | CPD 정확도 |
| `goal1_ml_evaluation.json` | DREAM·PatchCore PR |
| `goal2_ml_evaluation.json` | 목표 2 ML |

---

## 4. 이메일 전송

1. `reports/` 폴더를 ZIP으로 압축
2. 또는 `goal_achievement_summary.md` 내용을 이메일 본문에 붙여넣기
3. Cursor Composer 검토 담당자에게 전송

---

## 5. Cursor Composer 1.5 검토

**입력**: 이메일로 받은 리포트

**검토 항목**:
- Precision, Recall, F1, PR AUC 수치
- CPD mean_error_frames
- 추가 개발 필요 사항 (특징 강화, 모델 튜닝, 시나리오 확장 등)

**출력**: 추가 개발 TODO, 전략 수정 제안
