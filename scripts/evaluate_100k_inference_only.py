"""
Inference-only evaluation on 100k dataset.

Uses pre-trained DREAM and PatchCore (fp_focused, 0% Normal FP) without re-training.
Fixed thresholds from fp_focused analysis.json.
Evaluates on ALL 100k videos (train+val+test) for TN+FP+FN+TP = 100k.
Video-level (dataset-level) aggregation.

Usage:
  python scripts/evaluate_100k_inference_only.py
  python scripts/evaluate_100k_inference_only.py --dream-model path/to/dream.pt --patchcore-model path/to/patchcore.npz
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

# fp_focused 분석 결과 (analysis.json)에서 도출된 임계값 (0% Normal FP 달성)
DREAM_THRESHOLD = 130.395751953125
PATCHCORE_THRESHOLD = 81.40042114257812

REPORTS = repo_root / "reports"
OUT_DIR = REPORTS / "crack_detection_analysis"
FP_FOCUSED_BASE = repo_root / "data" / "synthetic" / "ml_dataset_fp_focused"
DEFAULT_100K_BASE = repo_root / "data" / "synthetic" / "ml_dataset_100k_v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference-only 100k evaluation (DREAM, PatchCore, Ensemble)"
    )
    parser.add_argument(
        "--norm-base-dir",
        type=str,
        default=str(FP_FOCUSED_BASE),
        help="Dataset for normalization fit (fp_focused train)",
    )
    parser.add_argument(
        "--eval-base-dir",
        type=str,
        default=str(DEFAULT_100K_BASE),
        help="100k dataset for evaluation (all entries)",
    )
    parser.add_argument(
        "--dream-model",
        type=str,
        default=None,
        help="DREAM model path (default: %%APPDATA%%/motionanalyzer/models/dream_model.pt)",
    )
    parser.add_argument(
        "--patchcore-model",
        type=str,
        default=None,
        help="PatchCore model path (default: %%APPDATA%%/motionanalyzer/models/patchcore_model.npz)",
    )
    parser.add_argument(
        "--dream-threshold",
        type=float,
        default=DREAM_THRESHOLD,
        help=f"DREAM score threshold (default: {DREAM_THRESHOLD})",
    )
    parser.add_argument(
        "--patchcore-threshold",
        type=float,
        default=PATCHCORE_THRESHOLD,
        help=f"PatchCore score threshold (default: {PATCHCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--from-analysis",
        action="store_true",
        help="Read DREAM/PatchCore thresholds from analysis.json (fp_focused)",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=None,
        help="Limit eval entries for faster iteration (proportional sample). Omit for full 100k.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers for feature extraction (default 1). Use 4-8 to speed up 5k/100k.",
    )
    args = parser.parse_args()

    from motionanalyzer.paths import get_default_dream_model_path, get_default_patchcore_model_path
    from motionanalyzer.auto_optimize import (
        FeatureExtractionConfig,
        normalize_features,
        prepare_training_data,
    )
    from sklearn.metrics import confusion_matrix, roc_auc_score

    norm_base = Path(args.norm_base_dir)
    eval_base = Path(args.eval_base_dir)
    dream_path = Path(args.dream_model) if args.dream_model else get_default_dream_model_path()
    patchcore_path = Path(args.patchcore_model) if args.patchcore_model else get_default_patchcore_model_path()

    dream_thr = args.dream_threshold
    patchcore_thr = args.patchcore_threshold
    if args.from_analysis:
        a_path = OUT_DIR / "analysis.json"
        if a_path.exists():
            a = json.loads(a_path.read_text(encoding="utf-8"))
            m = a.get("models", {})
            dream_thr = m.get("DREAM", {}).get("best_threshold", dream_thr)
            patchcore_thr = m.get("PatchCore", {}).get("best_threshold", patchcore_thr)
            print(f"[from-analysis] DREAM threshold={dream_thr}, PatchCore threshold={patchcore_thr}")

    print("=" * 60)
    print("100k Inference-Only Evaluation")
    print("=" * 60)
    print(f"  Norm fit:  {norm_base}")
    print(f"  Eval:      {eval_base} (all entries)")
    print(f"  DREAM:     {dream_path} (threshold={dream_thr})")
    print(f"  PatchCore: {patchcore_path} (threshold={patchcore_thr})")
    print()

    if not dream_path.exists():
        print(f"ERROR: DREAM model not found: {dream_path}")
        print("  Train first: python scripts/analyze_crack_detection.py --base-dir data/synthetic/ml_dataset_fp_focused --dataset-level-eval --normal-fp-max 0")
        sys.exit(1)
    if not patchcore_path.exists():
        print(f"ERROR: PatchCore model not found: {patchcore_path}")
        sys.exit(1)

    # 1. Load fp_focused train for normalization fit
    norm_mf = norm_base / "manifest.json"
    if not norm_mf.exists():
        print(f"ERROR: Normalization manifest not found: {norm_mf}")
        sys.exit(1)
    norm_entries = json.loads(norm_mf.read_text(encoding="utf-8"))["entries"]
    normal_train = [norm_base / e["path"] for e in norm_entries if e.get("label", 0) == 0 and e.get("split") == "train"]
    crack_train = [norm_base / e["path"] for e in norm_entries if e.get("label", 1) == 1 and e.get("split") == "train"]
    if not normal_train or not crack_train:
        # Fallback: use goal-based
        normal_train = [norm_base / e["path"] for e in norm_entries if e.get("goal") in ("normal", "variant") and e.get("split") == "train"]
        crack_train = [norm_base / e["path"] for e in norm_entries if e.get("goal") == "goal1" and e.get("split") == "train"]

    feature_config = FeatureExtractionConfig(
        include_per_frame=True,
        include_per_point=False,
        include_global_stats=True,
        include_crack_risk_features=False,
        include_advanced_stats=True,
        include_frequency_domain=True,
    )

    print(f"[1/5] Loading fp_focused train for normalization fit (n_jobs={args.n_jobs})...")
    feat_norm, _ = prepare_training_data(
        normal_datasets=normal_train,
        crack_datasets=crack_train,
        feature_config=feature_config,
        n_jobs=args.n_jobs,
    )

    # 2. Load 100k ALL entries (train+val+test)
    eval_mf = eval_base / "manifest.json"
    if not eval_mf.exists():
        print(f"ERROR: Eval manifest not found: {eval_mf}")
        sys.exit(1)
    eval_entries = json.loads(eval_mf.read_text(encoding="utf-8"))["entries"]
    normal_eval = [eval_base / e["path"] for e in eval_entries if e.get("label", 0) == 0]
    crack_eval = [eval_base / e["path"] for e in eval_entries if e.get("label", 1) == 1]
    if not normal_eval and not crack_eval:
        normal_eval = [eval_base / e["path"] for e in eval_entries if e.get("goal") in ("normal", "variant")]
        crack_eval = [eval_base / e["path"] for e in eval_entries if e.get("goal") in ("goal1", "goal2")]

    if args.max_eval and args.max_eval < len(normal_eval) + len(crack_eval):
        total = len(normal_eval) + len(crack_eval)
        n_norm = min(len(normal_eval), max(1, int(args.max_eval * len(normal_eval) / total)))
        n_crack = min(len(crack_eval), args.max_eval - n_norm)
        normal_eval = normal_eval[:n_norm]
        crack_eval = crack_eval[:n_crack]
        print(f"[max-eval] Sampled {n_norm} normal + {n_crack} crack = {n_norm + n_crack} (from {total})")

    print(f"[2/5] Extracting features ({len(normal_eval)} normal + {len(crack_eval)} crack, n_jobs={args.n_jobs})...")
    feat_eval, lab_eval = prepare_training_data(
        normal_datasets=normal_eval,
        crack_datasets=crack_eval,
        feature_config=feature_config,
        n_jobs=args.n_jobs,
    )

    exclude = ["label", "dataset_path", "frame", "index", "x", "y"]
    feature_cols = [
        c for c in feat_norm.columns
        if c not in exclude and "crack_risk" not in c.lower()
        and c in feat_norm.select_dtypes(include=["number"]).columns
    ]
    if not feature_cols:
        feature_cols = [c for c in feat_norm.columns if c not in exclude and "crack_risk" not in c.lower()]

    # Fit normalization on fp_focused normal train
    fit_df = feat_norm.loc[feat_norm["label"] == 0] if "label" in feat_norm.columns else feat_norm
    norm_eval = normalize_features(feat_eval, exclude_cols=exclude, fit_df=fit_df)
    X_eval = norm_eval[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y_eval = np.asarray(lab_eval, dtype=int)

    # 3. Dataset-level aggregation (video-level)
    def aggregate_by_dataset(scores: np.ndarray, y: np.ndarray, paths: pd.Series) -> tuple[np.ndarray, np.ndarray, list]:
        df = pd.DataFrame({"path": paths.astype(str).values, "score": scores, "y": y})
        agg = df.groupby("path", sort=False).agg({"score": "max", "y": "max"})
        return agg["score"].values.astype(np.float32), agg["y"].values.astype(int), agg.index.tolist()

    if "dataset_path" not in norm_eval.columns:
        norm_eval = norm_eval.copy()
        norm_eval["dataset_path"] = [str(i) for i in range(len(norm_eval))]

    scores_dream = np.zeros(len(X_eval), dtype=np.float32)
    scores_patchcore = np.zeros(len(X_eval), dtype=np.float32)

    # 4. Load DREAM and predict
    print("[3/5] Loading DREAM and predicting...")
    from motionanalyzer.ml_models.dream import DREAMPyTorch
    dream = DREAMPyTorch(input_dim=len(feature_cols))
    dream.load(dream_path)
    scores_dream[:] = dream.predict(X_eval)

    # 5. Load PatchCore and predict
    print("[4/5] Loading PatchCore and predicting...")
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
    pc = PatchCoreScikitLearn(feature_dim=len(feature_cols))
    pc.load(patchcore_path)
    scores_patchcore[:] = pc.predict(pd.DataFrame(X_eval, columns=feature_cols))

    # 6. Aggregate by video
    paths_ser = norm_eval["dataset_path"] if "dataset_path" in norm_eval.columns else pd.Series([str(i) for i in range(len(X_eval))])
    scores_dream_agg, y_agg, paths_agg = aggregate_by_dataset(scores_dream, y_eval, paths_ser)
    scores_pc_agg, _, _ = aggregate_by_dataset(scores_patchcore, y_eval, paths_ser)

    # 7. Apply fixed thresholds
    pred_dream = (scores_dream_agg >= dream_thr).astype(int)
    pred_pc = (scores_pc_agg >= patchcore_thr).astype(int)
    pred_ens = ((pred_dream == 1) & (pred_pc == 1)).astype(int)

    # 8. Confusion matrices
    n_total = len(y_agg)
    n_normal = int((y_agg == 0).sum())
    n_crack = int((y_agg == 1).sum())
    print(f"\n[5/5] Computing confusion matrices (n_videos={n_total})...")
    print(f"       TN+FP+FN+TP = {n_total} (normal={n_normal}, crack={n_crack})")

    results = {}
    for name, pred in [("DREAM", pred_dream), ("PatchCore", pred_pc), ("Ensemble", pred_ens)]:
        cm = confusion_matrix(y_agg, pred)
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        fp_rate = fp / n_normal if n_normal > 0 else 0
        roc = roc_auc_score(y_agg, scores_dream_agg) if name == "DREAM" else (roc_auc_score(y_agg, scores_pc_agg) if name == "PatchCore" else 0)
        results[name] = {
            "confusion_matrix": cm.tolist(),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "precision": prec, "recall": rec, "normal_fp_rate": fp_rate,
            "roc_auc": float(roc) if name != "Ensemble" else 0,
            "threshold": dream_thr if name == "DREAM" else (patchcore_thr if name == "PatchCore" else "both_agree"),
        }
        print(f"  {name}: TN={tn} FP={fp} FN={fn} TP={tp} | Prec={prec:.4f} Rec={rec:.4f} NormalFP={fp_rate:.4%}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis = {
        "n_test": n_total,
        "n_normal": int(n_normal),
        "n_crack": int(n_crack),
        "eval_mode": "inference_only_100k" + (f"_max{args.max_eval}" if args.max_eval else ""),
        "norm_fit": str(norm_base),
        "eval_data": str(eval_base),
        **({"max_eval": args.max_eval} if args.max_eval else {}),
        "dream_model": str(dream_path),
        "patchcore_model": str(patchcore_path),
        "dream_threshold": dream_thr,
        "patchcore_threshold": patchcore_thr,
        "models": results,
    }
    (OUT_DIR / "analysis_100k_inference.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    # Plot confusion matrices
    import matplotlib.pyplot as plt
    for name, res in results.items():
        cm = np.array(res["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4.5))
        vmax = cm.max() or 1
        ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. Normal", "Pred. Crack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Normal", "Actual Crack"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=18, fontweight="bold")
        ax.set_title(f"{name} Confusion Matrix (100k)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_matrix_100k_{name.lower()}.png", dpi=300, bbox_inches="tight")
        plt.close()

    print()
    print("Done. Output:")
    print(f"  {OUT_DIR}/analysis_100k_inference.json")
    print(f"  {OUT_DIR}/confusion_matrix_100k_dream.png")
    print(f"  {OUT_DIR}/confusion_matrix_100k_patchcore.png")
    print(f"  {OUT_DIR}/confusion_matrix_100k_ensemble.png")


if __name__ == "__main__":
    main()
