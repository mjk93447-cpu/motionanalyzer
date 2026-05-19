#!/usr/bin/env python3
"""Train DRAEM + PatchCore + bundle_manifest for release / CI artifacts."""

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

from motionanalyzer.auto_optimize import (  # noqa: E402
    PAPER_FEATURE_CONFIG,
    apply_norm_stats,
    compute_norm_stats,
    get_default_n_jobs,
    prepare_training_data,
    select_feature_columns,
)
from motionanalyzer.paths import (  # noqa: E402
    get_artifacts_cache_dir,
    get_default_ml_dataset_dir,
    get_fp_focused_dataset_dir,
    get_user_models_dir,
)
from motionanalyzer.services.ml_training import run_draem_training, run_patchcore_training  # noqa: E402


def _load_manifest_paths(base: Path, max_train: int | None) -> dict[str, list[Path]]:
    mf = base / "manifest.json"
    if not mf.exists():
        raise FileNotFoundError(f"manifest not found: {mf}\nRun: python scripts/generate_ml_dataset.py --scale default")
    entries = json.loads(mf.read_text(encoding="utf-8"))["entries"]

    normal_train_pure = [
        base / e["path"]
        for e in entries
        if e["goal"] == "normal" and e.get("scenario") != "light_distortion" and e["split"] == "train"
    ]
    normal_train_ld = [
        base / e["path"] for e in entries if e.get("scenario") == "light_distortion" and e["split"] == "train"
    ]
    normal_train_thick = [
        base / e["path"] for e in entries if e["goal"] == "variant" and e["split"] == "train"
    ]
    normal_train = normal_train_ld + normal_train_thick + normal_train_pure
    crack_train = [base / e["path"] for e in entries if e["goal"] == "goal1" and e["split"] == "train"]
    normal_val = [base / e["path"] for e in entries if e["goal"] == "normal" and e["split"] == "val"]
    crack_val = [base / e["path"] for e in entries if e["goal"] == "goal1" and e["split"] == "val"]
    normal_test = [base / e["path"] for e in entries if e["goal"] == "normal" and e["split"] == "test"]
    crack_test = [base / e["path"] for e in entries if e["goal"] == "goal1" and e["split"] == "test"]

    if max_train:
        normal_train = normal_train[: max_train]
        crack_train = crack_train[: max(max_train // 4, 50)]
        normal_val = normal_val[: min(200, len(normal_val))]
        crack_val = crack_val[: min(50, len(crack_val))]
        normal_test = normal_test[: min(400, len(normal_test))]
        crack_test = crack_test[: min(80, len(crack_test))]

    return {
        "normal_train": normal_train,
        "crack_train": crack_train,
        "normal_val": normal_val,
        "crack_val": crack_val,
        "normal_test": normal_test,
        "crack_test": crack_test,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Train release model bundle")
    ap.add_argument("--base-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: release/models or APPDATA models")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--ci-fast", action="store_true", help="Small epochs and max-train=500")
    ap.add_argument("--draem-epochs", type=int, default=None)
    ap.add_argument("--generate-dataset", action="store_true", help="Run generate_ml_dataset --scale default if missing")
    args = ap.parse_args()

    draem_extra: dict = {}
    if args.ci_fast:
        args.max_train = args.max_train or 500
        draem_epochs = args.draem_epochs or 8
        coreset = 200
        draem_extra = {"use_discriminative": False, "batch_size": 64}
    else:
        draem_epochs = args.draem_epochs or 30
        coreset = 1000

    base = args.base_dir or (get_default_ml_dataset_dir() if args.ci_fast else get_fp_focused_dataset_dir())
    if args.generate_dataset and not (base / "manifest.json").exists():
        import subprocess

        scale = "default" if args.ci_fast else "fp_focused"
        subprocess.check_call(
            [sys.executable, str(repo_root / "scripts" / "generate_ml_dataset.py"), "--scale", scale],
            cwd=str(repo_root),
        )

    out_dir = args.out_dir or (repo_root / "release" / "models")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = _load_manifest_paths(base, args.max_train)
    n_jobs = get_default_n_jobs()
    cache = get_artifacts_cache_dir()
    suffix = f"_{args.max_train}" if args.max_train else ""

    print(f"Loading train from {base.name}...")
    feat_train, lab_train = prepare_training_data(
        normal_datasets=paths["normal_train"],
        crack_datasets=paths["crack_train"],
        feature_config=PAPER_FEATURE_CONFIG,
        n_jobs=n_jobs,
        cache_dir=cache,
        cache_key=f"{base.name}_release_train{suffix}",
    )
    feat_val, lab_val = prepare_training_data(
        normal_datasets=paths["normal_val"],
        crack_datasets=paths["crack_val"],
        feature_config=PAPER_FEATURE_CONFIG,
        n_jobs=n_jobs,
        cache_dir=cache,
        cache_key=f"{base.name}_release_val{suffix}",
    )
    combined = pd.concat([feat_train, feat_val], ignore_index=True)
    labels = np.concatenate([lab_train, lab_val])
    if args.ci_fast and len(combined) > 8000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(combined), size=8000, replace=False)
        combined = combined.iloc[idx].reset_index(drop=True)
        labels = labels[idx]
    normal_mask = labels == 0
    feature_cols = select_feature_columns(combined)
    norm_stats = compute_norm_stats(combined.loc[normal_mask])
    normed = combined.copy()
    transformed = apply_norm_stats(combined, norm_stats, feature_cols)
    for col in feature_cols:
        normed[col] = transformed[col]
    normed = normed.fillna(0.0)

    def log(msg: str) -> None:
        print(msg, flush=True)

    thresh_pct = 99.5 if args.ci_fast else 95.0

    log("Training DRAEM...")
    draem_res = run_draem_training(
        normed,
        labels,
        log=log,
        progress=lambda: None,
        epochs=draem_epochs,
        model_save_dir=out_dir,
        feature_config=PAPER_FEATURE_CONFIG,
        training_dataset_id=base.name,
        batch_size=draem_extra.get("batch_size", 32),
        use_discriminative=draem_extra.get("use_discriminative", True),
        threshold_percentile=thresh_pct if args.ci_fast else 95.0,
    )
    if not draem_res.get("success"):
        print(f"DRAEM failed: {draem_res.get('message')}")
        sys.exit(1)

    log("Training PatchCore...")
    pc_res = run_patchcore_training(
        normed,
        labels,
        log=log,
        progress=lambda: None,
        coreset_size=coreset,
        model_save_dir=out_dir,
        feature_config=PAPER_FEATURE_CONFIG,
        train_mode="scratch",
        training_dataset_id=base.name,
        threshold_percentile=thresh_pct,
    )
    if not pc_res.get("success"):
        print(f"PatchCore failed: {pc_res.get('message')}")
        sys.exit(1)

    from motionanalyzer.ml_bundle import save_model_bundle  # noqa: E402

    import json

    mf_path = out_dir / "bundle_manifest.json"
    draem_thr = pc_thr = None
    if mf_path.exists():
        existing = json.loads(mf_path.read_text(encoding="utf-8"))
        draem_thr = existing.get("draem_threshold")
        pc_thr = existing.get("patchcore_threshold")
    save_model_bundle(
        out_dir,
        feature_config=PAPER_FEATURE_CONFIG,
        features_df=combined,
        labels=labels,
        draem_threshold=float(draem_thr) if draem_thr is not None else None,
        patchcore_threshold=float(pc_thr) if pc_thr is not None else None,
        ensemble_strategy="both_agree",
        training_dataset_id=base.name,
        patchcore_n_bank_samples=pc_res.get("patchcore_n_bank_samples"),
        patchcore_training_sources=pc_res.get("patchcore_training_sources"),
    )
    log("Rewrote bundle_manifest.json from raw features (inference-aligned).")

    from motionanalyzer.services.ml_inference import predict_manifest  # noqa: E402

    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_out: dict[str, object] = {"base_dir": str(base), "out_dir": str(out_dir)}
    for mode in ("draem", "patchcore", "ensemble"):
        rep = predict_manifest(
            base / "manifest.json",
            mode,  # type: ignore[arg-type]
            models_dir=out_dir,
            split="test",
            max_bundles=80 if args.ci_fast else None,
        )
        metrics_out[mode] = rep["metrics"]
        log(f"  {mode}: P={rep['metrics']['precision']:.3f} R={rep['metrics']['recall']:.3f} FP={rep['metrics']['fp']}")

    out_json = reports_dir / "release_bundle_metrics.json"
    out_json.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")
    log(f"Wrote {out_json}")
    log(f"Models in {out_dir}")

    if args.ci_fast:
        pc_m = metrics_out.get("patchcore", {})  # type: ignore[union-attr]
        pc_p = float(pc_m.get("precision", 0))
        pc_fp = int(pc_m.get("fp", 999))
        # Small default-1k test split: require reasonable precision and bounded FP
        if pc_p < 0.50 or pc_fp > 35:
            print(f"CI gate: PatchCore precision={pc_p:.3f} FP={pc_fp} (need P>=0.50, FP<=35)")
            sys.exit(1)


if __name__ == "__main__":
    main()
