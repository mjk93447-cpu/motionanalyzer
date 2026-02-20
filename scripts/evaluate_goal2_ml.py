"""
Evaluate Goal 2 (already-cracked panel) via DREAM/PatchCore.

Uses normal vs pre_damaged from ML dataset. Train on train split, evaluate on test.
Output: reports/goal2_ml_evaluation.json

Usage:
  python scripts/evaluate_goal2_ml.py         # full dataset
  python scripts/evaluate_goal2_ml.py --small # cap normal to 100 train, 50 test (faster)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="Use smaller subset for faster testing")
    args = ap.parse_args()

    manifest_path = BASE / "manifest.json"
    if not manifest_path.exists():
        print("Run: python scripts/generate_ml_dataset.py")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normal_train = [BASE / e["path"] for e in manifest["entries"] if e["goal"] == "normal" and e["split"] == "train"]
    normal_test = [BASE / e["path"] for e in manifest["entries"] if e["goal"] == "normal" and e["split"] == "test"]
    predam_train = [BASE / e["path"] for e in manifest["entries"] if e["goal"] == "goal2" and e["split"] == "train"]
    predam_test = [BASE / e["path"] for e in manifest["entries"] if e["goal"] == "goal2" and e["split"] == "test"]

    if args.small:
        normal_train = normal_train[:100]
        normal_test = normal_test[:50]
        predam_train = predam_train[:10]
        predam_test = predam_test[:5]
        print("Using --small: normal_train=100, normal_test=50, predam_train=10, predam_test=5")

    if not predam_train or not predam_test:
        print("Insufficient goal2 (pre_damaged) data. Use full dataset: python scripts/generate_ml_dataset.py")
        sys.exit(1)

    try:
        from motionanalyzer.auto_optimize import (
            FeatureExtractionConfig,
            normalize_features,
            prepare_training_data,
        )
        from motionanalyzer.gui.runners import _run_dream, _run_patchcore
        from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
    except ImportError as e:
        print(f"ML dependencies required: pip install -e '.[ml]'\n{e}")
        sys.exit(1)

    def log(_: str) -> None:
        pass

    def progress() -> None:
        pass

    # Prepare train and test separately
    feat_train, lab_train = prepare_training_data(
        normal_datasets=normal_train,
        crack_datasets=predam_train,
        feature_config=FeatureExtractionConfig(
            include_per_frame=True,
            include_per_point=False,
            include_global_stats=False,
            include_crack_risk_features=False,
            include_advanced_stats=False,
            include_frequency_domain=False,
        ),
    )
    feat_test, lab_test = prepare_training_data(
        normal_datasets=normal_test,
        crack_datasets=predam_test,
        feature_config=FeatureExtractionConfig(
            include_per_frame=True,
            include_per_point=False,
            include_global_stats=False,
            include_crack_risk_features=False,
            include_advanced_stats=False,
            include_frequency_domain=False,
        ),
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
    norm_test = normalize_features(feat_test, exclude_cols=exclude, fit_df=feat_train.loc[normal_mask_train])
    X_train = norm_train[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y_train = np.asarray(lab_train, dtype=int)
    X_test = norm_test[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y_test = np.asarray(lab_test, dtype=int)

    results: dict = {}

    # DREAM (use fewer epochs when --small for faster iteration)
    dream_epochs = 15 if args.small else 50
    try:
        res = _run_dream(
            pd.DataFrame(X_train, columns=feature_cols),
            y_train,
            log=log,
            progress=progress,
            epochs=dream_epochs,
        )
        if res.get("success"):
            from motionanalyzer.ml_models.dream import DREAMPyTorch
            model = DREAMPyTorch(input_dim=len(feature_cols))
            model.load(res["model_path"])
            scores = model.predict(X_test)
            roc = roc_auc_score(y_test, scores)
            prec, rec, _ = precision_recall_curve(y_test, scores)
            pr_auc = auc(rec, prec)
            results["DREAM"] = {"roc_auc": round(float(roc), 4), "pr_auc": round(float(pr_auc), 4)}
    except Exception as e:
        results["DREAM"] = {"error": str(e)}

    # PatchCore
    try:
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
            scores = model.predict(pd.DataFrame(X_test, columns=feature_cols))
            roc = roc_auc_score(y_test, scores)
            prec, rec, _ = precision_recall_curve(y_test, scores)
            pr_auc = auc(rec, prec)
            results["PatchCore"] = {"roc_auc": round(float(roc), 4), "pr_auc": round(float(pr_auc), 4)}
    except Exception as e:
        results["PatchCore"] = {"error": str(e)}

    out = {
        "goal": "goal2",
        "metric": "ML_anomaly_detection",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "models": results,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "goal2_ml_evaluation.json").write_text(
        json.dumps(out, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print("Goal 2 ML evaluation:")
    for k, v in results.items():
        print(f"  {k}: {v}")
    print(f"  Saved: {REPORTS / 'goal2_ml_evaluation.json'}")


if __name__ == "__main__":
    main()
