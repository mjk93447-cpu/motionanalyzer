from __future__ import annotations

import os
from pathlib import Path


def get_user_app_dir() -> Path:
    """
    User-specific application directory.

    - Windows: %APPDATA%/motionanalyzer
    - Others:  ~/.config/motionanalyzer
    """
    if os.name == "nt":
        appdata = os.getenv("APPDATA") or os.path.expanduser("~")
        return Path(appdata) / "motionanalyzer"
    return Path.home() / ".config" / "motionanalyzer"


def get_user_models_dir() -> Path:
    """User-specific model directory (%APPDATA%/motionanalyzer/models on Windows)."""
    return get_user_app_dir() / "models"


def get_default_dream_model_path() -> Path:
    return get_user_models_dir() / "dream_model.pt"


def get_default_patchcore_model_path() -> Path:
    return get_user_models_dir() / "patchcore_model.npz"


def get_default_temporal_model_path() -> Path:
    return get_user_models_dir() / "temporal_model.pt"

