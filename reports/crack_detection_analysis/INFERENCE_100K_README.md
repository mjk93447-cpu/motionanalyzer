# 100k Inference-Only 평가

## 요약

- **모델**: fp_focused 학습으로 0% Normal FP 달성한 DREAM, PatchCore (추가 학습 없음)
- **임계값**: fp_focused `analysis.json` 기반 고정값 사용
- **평가 데이터**: 100k 데이터셋 전체 (train+val+test, 비디오 단위)
- **산출물**: TN+FP+FN+TP ≈ 100k, DREAM / PatchCore / Ensemble Confusion Matrix

## 모델 저장 경로 (fp_focused 학습 시)

| 모델 | 기본 경로 |
|------|-----------|
| DREAM | `%APPDATA%\motionanalyzer\models\dream_model.pt` |
| PatchCore | `%APPDATA%\motionanalyzer\models\patchcore_model.npz` |

`analyze_crack_detection.py`는 `model_save_dir`을 지정하지 않으므로 위 기본 경로에 저장됩니다.

## 임계값 (fp_focused analysis.json)

| 모델 | 임계값 |
|------|--------|
| DREAM | 130.395751953125 |
| PatchCore | 81.40042114257812 |
| Ensemble | both_agree (DREAM ∧ PatchCore) |

## 실행 방법

```powershell
python scripts/evaluate_100k_inference_only.py
```

커스텀 경로/임계값:

```powershell
python scripts/evaluate_100k_inference_only.py `
  --dream-model "C:\path\to\dream_model.pt" `
  --patchcore-model "C:\path\to\patchcore_model.npz" `
  --dream-threshold 130.396 `
  --patchcore-threshold 81.4
```

## 출력

- `reports/crack_detection_analysis/analysis_100k_inference.json`
- `confusion_matrix_100k_dream.png`
- `confusion_matrix_100k_patchcore.png`
- `confusion_matrix_100k_ensemble.png`
