"""Portable model path resolution via environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def test_motionanalyzer_models_dir_override(tmp_path: Path, monkeypatch) -> None:
    custom = tmp_path / "my_models"
    custom.mkdir()
    monkeypatch.setenv("MOTIONANALYZER_MODELS_DIR", str(custom))
    from motionanalyzer.paths import get_user_models_dir, resolve_all_model_paths

    assert get_user_models_dir() == custom.resolve()
    paths = resolve_all_model_paths()
    assert paths["models_dir"] == custom.resolve()
    assert paths["manifest"] == custom / "bundle_manifest.json"


def test_motionanalyzer_app_dir_override(tmp_path: Path, monkeypatch) -> None:
    custom = tmp_path / "app"
    monkeypatch.setenv("MOTIONANALYZER_APP_DIR", str(custom))
    from motionanalyzer.paths import get_user_app_dir, get_user_models_dir

    assert get_user_app_dir() == custom.resolve()
    assert get_user_models_dir() == (custom / "models").resolve()
