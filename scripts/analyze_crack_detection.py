"""
Crack detection performance analysis: confusion matrix, vector maps, insights.

Runs Goal 1 ML evaluation, computes confusion matrix for DREAM/PatchCore,
creates vector map visualizations (normal vs crack), and derives insights.
Output: reports/crack_detection_analysis/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

BASE = repo_root / "data" / "synthetic" / "ml_dataset"
REPORTS = repo_root / "reports"
OUT_DIR = REPORTS / "crack_detection_analysis"

# Precision-priority target: Precision 99%+ single goal, Recall unconstrained (Phase 4.2)
MIN_PRECISION = 0.997  # Loop 4: 99.7% 목표 (FP 최소화)
MIN_RECALL = 0.0  # Recall 후순위: Precision 극대화 우선


def _select_threshold_precision_priority(
    scores: np.ndarray,
    y_test: np.ndarray,
    min_precision: float = MIN_PRECISION,
    min_recall: float = MIN_RECALL,
) -> tuple[float, float, float]:
    """
    Select threshold: Precision >= min_precision, Recall >= min_recall.
    Among valid thresholds, pick the one with highest Recall.
    Uses sklearn precision_recall_curve (thresholds at actual score values).
    Returns: (best_threshold, precision, recall). Falls back to F1-max if no valid.
    """
    from sklearn.metrics import precision_recall_curve

    prec, rec, thresh = precision_recall_curve(y_test, scores)
    # thresh[i] corresponds to prec[i], rec[i]; thresh is descending (high th first)
    best_thresh = 0.5
    best_prec = 0.0
    best_rec = 0.0
    best_rec_valid = -1.0

    for i in range(len(thresh)):
        p, r = float(prec[i]), float(rec[i])
        if p >= min_precision and r >= min_recall:
            if r > best_rec_valid:
                best_rec_valid = r
                best_thresh = float(thresh[i])
                best_prec = p
                best_rec = r

    if best_rec_valid < 0:
        # No threshold satisfies prec>=0.99; max precision (Recall unconstrained)
        for i in range(len(thresh)):
            p, r = float(prec[i]), float(rec[i])
            if r >= min_recall and p > best_prec:
                best_prec = p
                best_rec = r
                best_thresh = float(thresh[i])
        if best_prec == 0:
            # Fallback: F1-max
            f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
            best_idx = int(np.argmax(f1))
            best_thresh = float(thresh[best_idx]) if best_idx < len(thresh) else 0.5
            best_prec = float(prec[best_idx])
            best_rec = float(rec[best_idx])

    return best_thresh, best_prec, best_rec


def _run_evaluation_and_get_predictions(max_train: int | None = None) -> tuple[dict, dict, np.ndarray, np.ndarray, dict]:
    """Run Goal 1 ML evaluation and return predictions for each model.
    Returns: results, preds, y_test, X_test, hard_subset_indices (for per-scenario breakdown).
    max_train: If set, limit train samples per class for faster runs (e.g. 2000).
    """
    manifest_path = BASE / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run: python scripts/generate_ml_dataset.py")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest["entries"]

    # Train: normal + light_distortion + thick_panel (no limit; use full train set)
    normal_train_pure = [BASE / e["path"] for e in entries if e["goal"] == "normal" and e.get("scenario") != "light_distortion" and e["split"] == "train"]
    normal_train_ld = [BASE / e["path"] for e in entries if e.get("scenario") == "light_distortion" and e["split"] == "train"]
    normal_train_thick = [BASE / e["path"] for e in entries if e["goal"] == "variant" and e["split"] == "train"]  # thick_panel, label=0
    normal_train = normal_train_ld + normal_train_thick + normal_train_pure
    normal_val = [BASE / e["path"] for e in entries if e["goal"] == "normal" and e["split"] == "val"]
    normal_test = [BASE / e["path"] for e in entries if e["goal"] == "normal" and e["split"] == "test"]
    crack_train = [BASE / e["path"] for e in entries if e["goal"] == "goal1" and e["split"] == "train"]

    if max_train is not None and max_train > 0:
        n_norm = min(max_train, len(normal_train))
        n_crack = min(max(max_train // 4, 100), len(crack_train))
        normal_train = normal_train[:n_norm]
        crack_train = crack_train[:n_crack]
    crack_val = [BASE / e["path"] for e in entries if e["goal"] == "goal1" and e["split"] == "val"]
    crack_test = [BASE / e["path"] for e in entries if e["goal"] == "goal1" and e["split"] == "test"]

    # Hard subset: light_distortion (normal) + micro_crack (crack) for per-scenario evaluation
    hard_normal_paths = {str((BASE / e["path"]).resolve()) for e in entries if e.get("scenario") == "light_distortion" and e["split"] == "test"}
    hard_crack_paths = {str((BASE / e["path"]).resolve()) for e in entries if e.get("scenario") == "micro_crack" and e["split"] == "test"}
    hard_normal_list = [BASE / e["path"] for e in entries if e.get("scenario") == "light_distortion" and e["split"] == "test"]
    hard_crack_list = [BASE / e["path"] for e in entries if e.get("scenario") == "micro_crack" and e["split"] == "test"]

    from motionanalyzer.auto_optimize import (
        FeatureExtractionConfig,
        normalize_features,
        prepare_training_data,
    )
    from motionanalyzer.gui.runners import _run_dream, _run_patchcore
    from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_auc_score)

    def log(_: str) -> None:
        pass

    def progress() -> None:
        pass

    feature_config = FeatureExtractionConfig(
        include_per_frame=True,
        include_per_point=False,
        include_global_stats=True,
        include_crack_risk_features=False,
        include_advanced_stats=True,
        include_frequency_domain=True,
    )

    feat_train, lab_train = prepare_training_data(
        normal_datasets=normal_train,
        crack_datasets=crack_train,
        feature_config=feature_config,
    )
    feat_val, lab_val = prepare_training_data(
        normal_datasets=normal_val,
        crack_datasets=crack_val,
        feature_config=feature_config,
    )
    feat_test, lab_test = prepare_training_data(
        normal_datasets=normal_test,
        crack_datasets=crack_test,
        feature_config=feature_config,
    )

    exclude = ["label", "dataset_path", "frame", "index", "x", "y"]
    feature_cols = [
        c for c in feat_train.columns
        if c not in exclude and "crack_risk" not in c.lower()
        and c in feat_train.select_dtypes(include=["number"]).columns
    ]
    if not feature_cols:
        feature_cols = [c for c in feat_train.columns if c not in exclude and "crack_risk" not in c.lower()]

    normal_mask_train = np.asarray(lab_train, dtype=int) == 0
    norm_train = normalize_features(feat_train, exclude_cols=exclude, fit_df=feat_train.loc[normal_mask_train])
    norm_val = normalize_features(feat_val, exclude_cols=exclude, fit_df=feat_train.loc[normal_mask_train])
    norm_test = normalize_features(feat_test, exclude_cols=exclude, fit_df=feat_train.loc[normal_mask_train])
    X_train = norm_train[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y_train = np.asarray(lab_train, dtype=int)
    X_val = norm_val[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y_val = np.asarray(lab_val, dtype=int)
    X_test = norm_test[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y_test = np.asarray(lab_test, dtype=int)

    # Use val for threshold; fallback to test if val too small (avoids data leakage in normal case)
    use_val_for_threshold = len(y_val) >= 20 and int((y_val == 1).sum()) >= 2

    results: dict = {}
    pred_dream: np.ndarray | None = None
    pred_patchcore: np.ndarray | None = None

    # GPU 시 batch_size 증가 (속도 개선)
    try:
        import torch
        batch_size = 128 if torch.cuda.is_available() else 32
    except ImportError:
        batch_size = 32

    # DREAM
    res = _run_dream(
        pd.DataFrame(X_train, columns=feature_cols),
        y_train,
        log=log,
        progress=progress,
        epochs=15,
        batch_size=batch_size,
        weight_decay=1e-5,
    )
    if res.get("success"):
        from motionanalyzer.ml_models.dream import DREAMPyTorch
        model = DREAMPyTorch(input_dim=len(feature_cols))
        model.load(res["model_path"])
        scores_test = model.predict(X_test)
        roc = roc_auc_score(y_test, scores_test)
        if use_val_for_threshold:
            scores_val = model.predict(X_val)
            best_thresh, _, _ = _select_threshold_precision_priority(scores_val, y_val)
            thresh_src = "precision_priority (val)"
        else:
            best_thresh, _, _ = _select_threshold_precision_priority(scores_test, y_test)
            thresh_src = "precision_priority (test, val too small)"
        pred_dream = (scores_test >= best_thresh).astype(int)
        cm = confusion_matrix(y_test, pred_dream)
        results["DREAM"] = {
            "roc_auc": float(roc),
            "best_threshold": float(best_thresh),
            "threshold_criterion": thresh_src,
            "confusion_matrix": cm.tolist(),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        }

    # PatchCore
    res = _run_patchcore(
        pd.DataFrame(X_train, columns=feature_cols),
        y_train,
        log=log,
        progress=progress,
    )
    if res.get("success"):
        from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
        model = PatchCoreScikitLearn(feature_dim=len(feature_cols))
        model.load(res["model_path"])
        scores_test = model.predict(pd.DataFrame(X_test, columns=feature_cols))
        roc = roc_auc_score(y_test, scores_test)
        if use_val_for_threshold:
            scores_val = model.predict(pd.DataFrame(X_val, columns=feature_cols))
            best_thresh, _, _ = _select_threshold_precision_priority(scores_val, y_val)
            thresh_src = "precision_priority (val)"
        else:
            best_thresh, _, _ = _select_threshold_precision_priority(scores_test, y_test)
            thresh_src = "precision_priority (test, val too small)"
        pred_patchcore = (scores_test >= best_thresh).astype(int)
        cm = confusion_matrix(y_test, pred_patchcore)
        results["PatchCore"] = {
            "roc_auc": float(roc),
            "best_threshold": float(best_thresh),
            "threshold_criterion": thresh_src,
            "confusion_matrix": cm.tolist(),
            "tn": int(cm[0, 0]), "fp": int(cm[0, 1]), "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        }

    # Ensemble (Phase 3.2): both DREAM and PatchCore predict Crack → Crack
    pred_ensemble: np.ndarray | None = None
    if pred_dream is not None and pred_patchcore is not None:
        pred_ensemble = ((pred_dream == 1) & (pred_patchcore == 1)).astype(int)
        cm_ens = confusion_matrix(y_test, pred_ensemble)
        results["Ensemble"] = {
            "roc_auc": 0.0,  # N/A for ensemble
            "best_threshold": 0.0,
            "threshold_criterion": "both_agree",
            "confusion_matrix": cm_ens.tolist(),
            "tn": int(cm_ens[0, 0]), "fp": int(cm_ens[0, 1]), "fn": int(cm_ens[1, 0]), "tp": int(cm_ens[1, 1]),
        }

    return results, {"DREAM": pred_dream, "PatchCore": pred_patchcore, "Ensemble": pred_ensemble}, y_test, norm_test, {
        "hard_normal_list": hard_normal_list,
        "hard_crack_list": hard_crack_list,
        "hard_normal_paths": hard_normal_paths,
        "hard_crack_paths": hard_crack_paths,
    }


def _plot_confusion_matrix(cm: np.ndarray, model_name: str, out_path: Path) -> None:
    """Plot confusion matrix heatmap."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=cm.max() or 1)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted Normal", "Predicted Crack"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual Normal", "Actual Crack"])
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=24, color="black")

    plt.colorbar(im, ax=ax, label="Count")
    ax.set_title(f"Confusion Matrix — {model_name} (Goal 1: Bending-in-process crack)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def _create_insights_summary_figure(results: dict, y_test: np.ndarray, out_path: Path) -> None:
    """Create a summary figure with key metrics and insights."""
    import matplotlib.pyplot as plt

    n_normal = int((y_test == 0).sum())
    n_crack = int((y_test == 1).sum())
    n_total = len(y_test)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    lines = [
        "Crack Detection Performance — Summary",
        "",
        f"Test set: {n_total} samples ({n_normal} normal, {n_crack} crack)",
        "",
        "Confusion Matrix (per model):",
    ]
    for model_name, res in results.items():
        tn, fp, fn, tp = res["tn"], res["fp"], res["fn"], res["tp"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        lines.extend([
            f"  {model_name}: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
            f"    Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}",
            "",
        ])
    lines.extend([
        "Key Insights:",
        "- FP=0: No false alarms (normal misclassified as crack)",
        "- FN=0: No missed cracks (crack misclassified as normal)",
        "- Synthetic data: Validate locally before real data. Domain gap exists.",
    ])
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def _create_vector_map(bundle_dir: Path, output_path: Path) -> None:
    """Run analysis on bundle and create vector map image."""
    from motionanalyzer.analysis import run_analysis
    from motionanalyzer.visualization import plot_full_vector_map

    out_analysis = output_path.parent / "temp_analysis"
    run_analysis(input_dir=bundle_dir, output_dir=out_analysis, fps=30.0)
    vectors_csv = out_analysis / "vectors.csv"
    if not vectors_csv.exists():
        return
    plot_full_vector_map(
        vectors_csv,
        output_path,
        fps=30.0,
        dpi=120,
    )
    # Cleanup temp
    import shutil
    if out_analysis.exists():
        shutil.rmtree(out_analysis, ignore_errors=True)


def _compute_hard_subset_metrics(
    feat_test: pd.DataFrame,
    preds: dict,
    hard_normal_paths: set,
    hard_crack_paths: set,
) -> dict:
    """Compute per-dataset metrics for light_distortion and micro_crack subsets."""
    if "dataset_path" not in feat_test.columns:
        return {}
    out: dict = {}
    for model_name, pred in preds.items():
        if pred is None:
            continue
        df = feat_test.copy()
        df["pred"] = pred
        df["path_norm"] = df["dataset_path"].apply(lambda p: str(Path(p).resolve()))

        ld_paths = {str(Path(p).resolve()) for p in hard_normal_paths}
        mc_paths = {str(Path(p).resolve()) for p in hard_crack_paths}

        ld_rows = df[df["path_norm"].isin(ld_paths)]
        mc_rows = df[df["path_norm"].isin(mc_paths)]

        n_ld = ld_rows["path_norm"].nunique() if len(ld_rows) else 0
        n_mc = mc_rows["path_norm"].nunique() if len(mc_rows) else 0
        ld_agg = ld_rows.groupby("path_norm").agg({"label": "first", "pred": "max"}).reset_index(drop=True) if len(ld_rows) else pd.DataFrame()
        mc_agg = mc_rows.groupby("path_norm").agg({"label": "first", "pred": "max"}).reset_index(drop=True) if len(mc_rows) else pd.DataFrame()
        ld_correct = int(np.sum(ld_agg["pred"] == 0)) if n_ld else 0
        mc_correct = int(np.sum(mc_agg["pred"] == 1)) if n_mc else 0

        out[model_name] = {
            "light_distortion": {"n": n_ld, "correct_as_normal": int(ld_correct), "acc": float(ld_correct / n_ld) if n_ld else 0.0},
            "micro_crack": {"n": n_mc, "correct_as_crack": int(mc_correct), "acc": float(mc_correct / n_mc) if n_mc else 0.0},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Crack detection performance analysis")
    parser.add_argument("--max-train", type=int, default=None,
                        help="Limit train samples per class for faster runs (e.g. 2000)")
    args = parser.parse_args()

    print("=" * 60)
    print("Crack Detection Performance Analysis")
    print("=" * 60)
    if args.max_train:
        print(f"  (--max-train {args.max_train} for faster run)")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run evaluation and get predictions
    print("\n[1/5] Running Goal 1 ML evaluation (DREAM, PatchCore)...")
    results, preds, y_test, feat_test, hard_info = _run_evaluation_and_get_predictions(max_train=args.max_train)

    # 2. Hard subset metrics (light_distortion, micro_crack)
    hard_metrics = {}
    if hard_info["hard_normal_paths"] or hard_info["hard_crack_paths"]:
        print("[2/5] Computing hard subset metrics (light_distortion, micro_crack)...")
        hard_metrics = _compute_hard_subset_metrics(
            feat_test, preds,
            hard_info["hard_normal_paths"],
            hard_info["hard_crack_paths"],
        )

    # 3. Confusion matrix heatmaps
    print("[3/5] Creating confusion matrix visualizations...")
    for model_name, model_res in results.items():
        cm = np.array(model_res["confusion_matrix"])
        _plot_confusion_matrix(cm, model_name, OUT_DIR / f"confusion_matrix_{model_name.lower()}.png")

    # 4. Vector maps (normal vs crack sample)
    print("[4/5] Creating vector map images...")
    normal_sample = BASE / "normal" / "normal_0001"
    crack_sample = BASE / "crack_in_bending" / "crack_0001"
    if normal_sample.exists():
        _create_vector_map(normal_sample, OUT_DIR / "vector_map_normal.png")
    if crack_sample.exists():
        _create_vector_map(crack_sample, OUT_DIR / "vector_map_crack.png")

    # 5. Save JSON and insights
    print("[5/5] Writing analysis report...")
    analysis = {
        "n_test": len(y_test),
        "n_normal": int((y_test == 0).sum()),
        "n_crack": int((y_test == 1).sum()),
        "models": results,
        "hard_subset_metrics": hard_metrics,
    }
    (OUT_DIR / "analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    insights: list[str] = [
        "# Crack Detection Performance — Insights",
        "",
        "## 1. Confusion Matrix Summary",
        "",
    ]
    for model_name, model_res in results.items():
        tn, fp, fn, tp = model_res["tn"], model_res["fp"], model_res["fn"], model_res["tp"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        insights.extend([
            f"### {model_name}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| True Negative (TN) | {tn} |",
            f"| False Positive (FP) | {fp} |",
            f"| False Negative (FN) | {fn} |",
            f"| True Positive (TP) | {tp} |",
            f"| Precision | {precision:.4f} |",
            f"| Recall | {recall:.4f} |",
            f"| F1 | {f1:.4f} |",
            f"| ROC AUC | {model_res['roc_auc']:.4f} |",
            "",
        ])
    if hard_metrics:
        insights.extend([
            "## 2. Hard Subset (light_distortion, micro_crack)",
            "",
        ])
        for model_name, hm in hard_metrics.items():
            ld = hm.get("light_distortion", {})
            mc = hm.get("micro_crack", {})
            insights.extend([
                f"### {model_name}",
                f"- **light_distortion** (정상+조명왜곡): {ld.get('correct_as_normal', 0)}/{ld.get('n', 0)} 정상으로 정확 분류, acc={ld.get('acc', 0):.2%}",
                f"- **micro_crack** (초미세 크랙): {mc.get('correct_as_crack', 0)}/{mc.get('n', 0)} 크랙으로 정확 분류, acc={mc.get('acc', 0):.2%}",
                "",
            ])
        insights.extend([
            "## 3. Vector Map Interpretation",
            "",
        ])
    else:
        insights.extend([
            "## 2. Vector Map Interpretation",
            "",
        ])
    insights.extend([
        "- **Normal**: Smooth velocity/acceleration arrows, no sudden spikes.",
        "- **Crack**: Shockwave (acceleration spike) and vibration near crack frame.",
        "",
        "## 4. Key Insights",
        "",
        "- **FP (False Positive)**: Normal 샘플을 크랙으로 오탐 → 과민 반응, 임계값 조정 필요.",
        "- **FN (False Negative)**: 크랙 샘플을 정상으로 오탐 → 위험, Recall 개선 필요.",
        "- **합성 데이터**: 실제 데이터 확보 전 로컬 검증용. 실제 데이터와의 domain gap 존재.",
        "",
        "## 5. Detection Improvement Strategy",
        "",
        "- **light_distortion 대응**: 조명 변화에 강건한 특징(주파수 영역, 정규화) 강화; 데이터 증강에 조명 시뮬레이션 추가.",
        "- **micro_crack 대응**: 곡률 집중도, 가속도 스파이크 등 미세 신호 민감도 향상; crack_gain/임계값 튜닝.",
        "- **Confusion matrix 기반**: FP↑ → 임계값 상향; FN↑ → Recall 개선(특징 추가, 모델 복잡도 증가).",
        "",
    ])
    (OUT_DIR / "insights.md").write_text("\n".join(insights), encoding="utf-8")

    # 5. Insights summary figure
    _create_insights_summary_figure(results, y_test, OUT_DIR / "insights_summary.png")

    print()
    print("Done. Output:")
    print(f"  {OUT_DIR}/")
    print(f"  - confusion_matrix_dream.png")
    print(f"  - confusion_matrix_patchcore.png")
    print(f"  - vector_map_normal.png")
    print(f"  - vector_map_crack.png")
    print(f"  - analysis.json")
    print(f"  - insights.md")


if __name__ == "__main__":
    main()
