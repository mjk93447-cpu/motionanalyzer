#!/usr/bin/env python3
"""
Balanced synthetic pretrain: 7:2:1 split, NG diversity, P~0.9 / R~0.7 threshold tuning.

Usage:
  python scripts/train_balanced_pretrain.py --generate
  python scripts/train_balanced_pretrain.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parent.parent
src = repo_root / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from motionanalyzer.auto_optimize import (  # noqa: E402
    PAPER_FEATURE_CONFIG,
    apply_norm_stats,
    compute_norm_stats,
    get_default_n_jobs,
    prepare_training_data,
    select_feature_columns,
)
from motionanalyzer.ml_thresholds import (  # noqa: E402
    aggregate_dataset_scores,
    metrics_at_threshold,
    select_threshold_target_metrics,
)
from motionanalyzer.paths import get_artifacts_cache_dir  # noqa: E402
from motionanalyzer.services.ml_training import run_draem_training, run_patchcore_training  # noqa: E402

DATASET_DIR = repo_root / "data" / "synthetic" / "ml_pretrain_balanced_3k_60f"
OUT_DIR = repo_root / "release" / "models"


def _manifest_paths(base: Path, split: str, *, goal1_only: bool = True) -> tuple[list[Path], list[Path]]:
    entries = json.loads((base / "manifest.json").read_text(encoding="utf-8"))["entries"]
    normal: list[Path] = []
    crack: list[Path] = []
    for e in entries:
        if e.get("split") != split:
            continue
        p = base / e["path"]
        if e.get("label", 0) == 0:
            normal.append(p)
        elif goal1_only and e.get("goal") == "goal1":
            crack.append(p)
        elif not goal1_only and e.get("label", 0) == 1:
            crack.append(p)
    return normal, crack


def _cap_paths(normal: list[Path], crack: list[Path], max_normal: int, max_crack: int) -> tuple[list[Path], list[Path]]:
    return normal[:max_normal], crack[:max_crack]


def _prepare_split(
    base: Path,
    split: str,
    *,
    max_normal: int | None,
    max_crack: int | None,
    cache: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    n_paths, c_paths = _manifest_paths(base, split)
    if max_normal:
        n_paths, c_paths = _cap_paths(n_paths, c_paths, max_normal, max_crack or max_normal // 5)
    feat, lab = prepare_training_data(
        normal_datasets=n_paths,
        crack_datasets=c_paths,
        feature_config=PAPER_FEATURE_CONFIG,
        n_jobs=get_default_n_jobs(),
        cache_dir=cache,
        cache_key=f"{base.name}_{split}",
    )
    return feat, lab


def _normalize_df(combined: pd.DataFrame, labels: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    normal_mask = labels == 0
    feature_cols = select_feature_columns(combined)
    normed = combined.copy()
    transformed = apply_norm_stats(combined, compute_norm_stats(combined.loc[normal_mask]), feature_cols)
    for col in feature_cols:
        normed[col] = transformed[col]
    return normed.fillna(0.0), feature_cols


def _subsample_rows(df: pd.DataFrame, labels: np.ndarray, max_rows: int, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    if len(df) <= max_rows:
        return df, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=max_rows, replace=False)
    return df.iloc[idx].reset_index(drop=True), labels[idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Balanced synthetic pretrain (P~0.9, R~0.7)")
    ap.add_argument("--generate", action="store_true", help="Generate ml_pretrain_balanced_3k_60f (7:2:1)")
    ap.add_argument("--base-dir", type=Path, default=DATASET_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--target-precision", type=float, default=0.9)
    ap.add_argument("--target-recall", type=float, default=0.7)
    ap.add_argument("--max-train-normal", type=int, default=1100, help="Cap train bundles (~30min)")
    ap.add_argument("--max-val-normal", type=int, default=320)
    ap.add_argument("--max-train-rows", type=int, default=14000)
    ap.add_argument("--draem-epochs", type=int, default=28)
    ap.add_argument("--coreset-size", type=int, default=900)
    args = ap.parse_args()

    base = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = get_artifacts_cache_dir()

    if args.generate or not (base / "manifest.json").exists():
        print("Generating balanced synthetic dataset (7:2:1)...")
        subprocess.check_call(
            [
                sys.executable,
                str(repo_root / "scripts" / "generate_ml_dataset.py"),
                "--scale",
                "pretrain_balanced",
                "--split-train",
                "0.7",
                "--split-val",
                "0.2",
                "--split-test",
                "0.1",
                "--workers",
                "4",
            ],
            cwd=str(repo_root),
        )

    t0 = time.time()
    print(f"Loading train split from {base.name}...")
    feat_train, lab_train = _prepare_split(
        base,
        "train",
        max_normal=args.max_train_normal,
        max_crack=args.max_train_normal // 6,
        cache=cache,
    )
    print(f"Loading val split...")
    feat_val, lab_val = _prepare_split(
        base,
        "val",
        max_normal=args.max_val_normal,
        max_crack=args.max_val_normal // 6,
        cache=cache,
    )
    print(f"Loading test split...")
    feat_test, lab_test = _prepare_split(
        base,
        "test",
        max_normal=None,
        max_crack=None,
        cache=cache,
    )

    combined_train, labels_train = _subsample_rows(feat_train, lab_train, args.max_train_rows)
    normed_train, _ = _normalize_df(combined_train, labels_train)

    print(f"Train rows: {len(normed_train)} (normal={int((labels_train==0).sum())}, crack={int((labels_train==1).sum())})")

    def log(msg: str) -> None:
        print(msg, flush=True)

    log("Training DRAEM (scratch, balanced targets on val)...")
    draem_res = run_draem_training(
        normed_train,
        labels_train,
        log=log,
        progress=lambda: None,
        epochs=args.draem_epochs,
        model_save_dir=out_dir,
        feature_config=PAPER_FEATURE_CONFIG,
        training_dataset_id=base.name,
        batch_size=48,
        use_discriminative=True,
        optimize_threshold=False,
        threshold_percentile=95.0,
    )
    if not draem_res.get("success"):
        print(f"DRAEM failed: {draem_res.get('message')}")
        sys.exit(1)

    log("Training PatchCore...")
    pc_res = run_patchcore_training(
        normed_train,
        labels_train,
        log=log,
        progress=lambda: None,
        coreset_size=args.coreset_size,
        model_save_dir=out_dir,
        train_mode="scratch",
        feature_config=PAPER_FEATURE_CONFIG,
        training_dataset_id=base.name,
        threshold_percentile=95.0,
    )
    if not pc_res.get("success"):
        print(f"PatchCore failed: {pc_res.get('message')}")
        sys.exit(1)

    from motionanalyzer.ml_models.draem import DRAEMPyTorch
    from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn

    feature_cols = select_feature_columns(combined_train)
    norm_stats = compute_norm_stats(combined_train.loc[labels_train == 0])
    val_normed = feat_val.copy()
    tr = apply_norm_stats(feat_val, norm_stats, feature_cols)
    for col in feature_cols:
        val_normed[col] = tr[col]
    val_normed = val_normed.fillna(0.0)

    draem = DRAEMPyTorch(input_dim=len(feature_cols))
    draem.load(out_dir / "draem_model.pt")
    pc = PatchCoreScikitLearn(feature_dim=len(feature_cols))
    pc.load(out_dir / "patchcore_model.npz")

    X_val = val_normed[feature_cols].fillna(0.0)
    d_scores = draem.predict(X_val.to_numpy(dtype=np.float32))
    p_scores = pc.predict(X_val)

    d_ds_scores, d_ds_labels = aggregate_dataset_scores(feat_val, d_scores)
    p_ds_scores, p_ds_labels = aggregate_dataset_scores(feat_val, p_scores)

    d_thr, d_p, d_r, d_cm = select_threshold_target_metrics(
        d_ds_scores,
        d_ds_labels,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
    )
    p_thr, p_p, p_r, p_cm = select_threshold_target_metrics(
        p_ds_scores,
        p_ds_labels,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
    )
    log(f"Val DRAEM: thr={d_thr:.4f} P={d_p:.3f} R={d_r:.3f} {d_cm}")
    log(f"Val PatchCore: thr={p_thr:.4f} P={p_p:.3f} R={p_r:.3f} {p_cm}")

    draem.reconstruction_error_threshold = d_thr
    draem.save(out_dir / "draem_model.pt")
    pc.anomaly_threshold = p_thr
    pc.save(out_dir / "patchcore_model.npz")

    raw_all = pd.concat([combined_train, feat_val], ignore_index=True)
    labels_all = np.concatenate([labels_train, lab_val])
    from motionanalyzer.ml_bundle import save_model_bundle

    save_model_bundle(
        out_dir,
        feature_config=PAPER_FEATURE_CONFIG,
        features_df=raw_all,
        labels=labels_all,
        draem_threshold=d_thr,
        patchcore_threshold=p_thr,
        ensemble_strategy="both_agree",
        training_dataset_id=base.name,
        patchcore_n_bank_samples=pc.n_bank_samples,
        patchcore_training_sources=list(pc.training_sources),
    )

    test_normed = feat_test.copy()
    tr_t = apply_norm_stats(feat_test, norm_stats, feature_cols)
    for col in feature_cols:
        test_normed[col] = tr_t[col]
    test_normed = test_normed.fillna(0.0)
    X_test = test_normed[feature_cols].fillna(0.0)
    td = draem.predict(X_test.to_numpy(dtype=np.float32))
    tp = pc.predict(X_test)
    td_ds, y_test = aggregate_dataset_scores(feat_test, td)
    tp_ds, _ = aggregate_dataset_scores(feat_test, tp)
    ens_pred = ((td_ds >= d_thr) & (tp_ds >= p_thr)).astype(int)
    y_test_arr = np.asarray(y_test, dtype=int)
    cm = __import__("sklearn.metrics", fromlist=["confusion_matrix"]).confusion_matrix(y_test_arr, ens_pred, labels=[0, 1])
    tp_c, fp_c, fn_c, tn_c = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0]), int(cm[0, 0])
    ens_m = {
        "precision": tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0,
        "recall": tp_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0,
        "tn": tn_c,
        "fp": fp_c,
        "fn": fn_c,
        "tp": tp_c,
    }

    report = {
        "dataset": str(base),
        "split_ratios": [0.7, 0.2, 0.1],
        "target_precision": args.target_precision,
        "target_recall": args.target_recall,
        "val": {
            "draem": {"threshold": d_thr, "precision": d_p, "recall": d_r, **d_cm},
            "patchcore": {"threshold": p_thr, "precision": p_p, "recall": p_r, **p_cm},
        },
        "test": {
            "draem": metrics_at_threshold(td_ds, y_test, d_thr),
            "patchcore": metrics_at_threshold(tp_ds, y_test, p_thr),
            "ensemble_both_agree": ens_m,
        },
        "elapsed_min": round((time.time() - t0) / 60.0, 1),
    }
    rep_path = repo_root / "reports" / "balanced_pretrain_metrics.json"
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Test ensemble: P={ens_m['precision']:.3f} R={ens_m['recall']:.3f} FP={ens_m['fp']} FN={ens_m['fn']}")
    log(f"Wrote {rep_path}")
    log(f"Models: {out_dir} ({report['elapsed_min']} min)")


if __name__ == "__main__":
    main()
