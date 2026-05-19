from __future__ import annotations

import os
from pathlib import Path

# Repo root (scripts and src may have different depth)
def _repo_root() -> Path:
    p = Path(__file__).resolve().parent.parent.parent
    return p


def get_user_app_dir() -> Path:
    """
    User-specific application directory.

    - Windows: %APPDATA%/motionanalyzer
    - Others:  ~/.config/motionanalyzer
    - Override: MOTIONANALYZER_APP_DIR
    """
    override = os.getenv("MOTIONANALYZER_APP_DIR")
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        appdata = os.getenv("APPDATA") or os.path.expanduser("~")
        return Path(appdata) / "motionanalyzer"
    return Path.home() / ".config" / "motionanalyzer"


def get_user_models_dir() -> Path:
    """User-specific model directory; override with MOTIONANALYZER_MODELS_DIR."""
    override = os.getenv("MOTIONANALYZER_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    return get_user_app_dir() / "models"


def get_default_draem_model_path() -> Path:
    return get_user_models_dir() / "draem_model.pt"


def resolve_draem_model_path(path: Path | None = None) -> Path:
    """
    Resolve DRAEM checkpoint path.

    Prefer explicit path, then ``draem_model.pt``, then legacy ``draem_model.pt``.
    """
    if path is not None:
        p = Path(path)
        if p.exists():
            return p
    primary = get_default_draem_model_path()
    if primary.exists():
        return primary
    legacy = get_user_models_dir() / "draem_model.pt"
    if legacy.exists():
        return legacy
    return primary if path is None else Path(path)


def get_default_patchcore_model_path() -> Path:
    return get_user_models_dir() / "patchcore_model.npz"


def get_default_temporal_model_path() -> Path:
    return get_user_models_dir() / "temporal_model.pt"


def get_bundle_manifest_path(models_dir: Path | None = None) -> Path:
    return (models_dir or get_user_models_dir()) / "bundle_manifest.json"


def resolve_all_model_paths(models_dir: Path | None = None) -> dict[str, Path]:
    """Canonical paths for DRAEM, PatchCore, and bundle manifest."""
    root = Path(models_dir or get_user_models_dir())
    return {
        "models_dir": root,
        "draem": resolve_draem_model_path(root / "draem_model.pt"),
        "patchcore": root / "patchcore_model.npz",
        "manifest": get_bundle_manifest_path(root),
        "ensemble_config": root / "ensemble_config.json",
    }


def get_bundled_models_dir() -> Path | None:
    """PyInstaller _MEIPASS/models or repo release/models (dev)."""
    import sys

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        p = Path(meipass) / "models"
        if p.is_dir():
            return p
    rel = _repo_root() / "release" / "models"
    return rel if rel.is_dir() else None


def ensure_user_models_from_bundle() -> Path:
    """Copy bundled pretrained weights to user models dir if missing."""
    user_dir = get_user_models_dir()
    user_dir.mkdir(parents=True, exist_ok=True)
    bundled = get_bundled_models_dir()
    if bundled is None:
        return user_dir
    import shutil

    for name in ("draem_model.pt", "patchcore_model.npz", "bundle_manifest.json", "ensemble_config.json"):
        src = bundled / name
        dst = user_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return user_dir


# Canonical dataset dirs (docs/DATASET_FOLDER_STRUCTURE.md)
def get_synthetic_data_root() -> Path:
    return _repo_root() / "data" / "synthetic"


def get_fp_focused_dataset_dir() -> Path:
    """Baseline/Loop1/paper: fp_focused 20k, 60f."""
    return get_synthetic_data_root() / "ml_fp_focused_20k_60f"


def get_100k_dataset_dir() -> Path:
    """100k inference-only evaluation."""
    return get_synthetic_data_root() / "ml_100k_60f"


def get_default_ml_dataset_dir() -> Path:
    """Default small ML dataset (1k)."""
    return get_synthetic_data_root() / "ml_default_1k_60f"


def get_pretrain_balanced_dataset_dir() -> Path:
    """Turnkey GUI pretrain: balanced 3k, 7:2:1 (docs/BALANCED_PRETRAIN.md)."""
    return get_synthetic_data_root() / "ml_pretrain_balanced_3k_60f"


def get_artifacts_cache_dir() -> Path:
    """Feature cache directory (CPU optimization, .gitignore)."""
    return _repo_root() / "artifacts" / "cache"

