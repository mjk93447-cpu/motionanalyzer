# Goal Achievement Summary

Target: **Precision-Recall maximization** for bending-in-process crack detection.

## Goal 1: Bending-in-process crack — CPD (change point)

- **Metric**: CPD accuracy (detected vs crack_frame)
- **n_evaluated**: 80
- **mean_error_frames**: 1.09
- **within_5_frames_pct**: 100.0%

## Goal 1: Bending-in-process crack — ML (DREAM/PatchCore, PR maximization)

- **n_train**: 7320, **n_test**: 3660
- **n_crack_test**: 610, **n_normal_test**: 3050

- **DREAM**: Precision=1.0, Recall=1.0, F1=1.0, PR AUC=1.0
- **PatchCore**: Precision=1.0, Recall=1.0, F1=1.0, PR AUC=1.0

## Goal 2: Already-cracked panel detection (ML)

- **n_train**: 6600, **n_test**: 3180

- **DREAM**: ROC AUC=0.843, PR AUC=0.5702
- **PatchCore**: ROC AUC=0.8421, PR AUC=0.6428
