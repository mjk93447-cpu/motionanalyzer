"""One-shot text migration: DREAM -> DRAEM across repo (Phase 1 rename)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Skip binary / generated / venv
SKIP_DIRS = {
    ".git",
    ".venv",
    ".venv-gpu",
    "__pycache__",
    "dist",
    "build",
    "node_modules",
    ".cursor",
    "artifacts",
    "data",
}

SKIP_FILES = {
    "migrate_dream_to_draem.py",
    "backup-manifest.json",
}

REPLACEMENTS: list[tuple[str, str]] = [
    ("DREAMAnomalyDetector", "DRAEMAnomalyDetector"),
    ("DREAMPyTorch", "DRAEMPyTorch"),
    ("get_default_dream_model_path", "get_default_draem_model_path"),
    ("dream_model_path", "draem_model_path"),
    ("dream_model.pt", "draem_model.pt"),
    ("dream_model", "draem_model"),
    ("dream_threshold", "draem_threshold"),
    ("run_dream_training", "run_draem_training"),
    ("_run_dream", "_run_draem"),
    ("MODE_DREAM", "MODE_DRAEM"),
    ("validate_enhanced_dream", "validate_enhanced_draem"),
    ("validate_dream_synthetic", "validate_draem_synthetic"),
    ("validate_dream", "validate_draem"),
    ("test_dream_draem", "test_draem"),
    ("ml_models.dream_temporal", "ml_models.draem_temporal"),
    ("ml_models.dream", "ml_models.draem"),
    ("dream_temporal.py", "draem_temporal.py"),
    ("dream.py", "draem.py"),
    ("dream_weight", "draem_weight"),
    ("dream_anomaly", "draem_anomaly"),
    ("confusion_matrix_dream", "confusion_matrix_draem"),
    ("confusion_matrix_100k_dream", "confusion_matrix_100k_draem"),
    ("confusion_matrix_hard_10k_dream", "confusion_matrix_hard_10k_draem"),
    ("DREAM_THRESHOLD", "DRAEM_THRESHOLD"),
    ("dream_path", "draem_path"),
    ("dream_thr", "draem_thr"),
    ("scores_dream", "scores_draem"),
    ("pred_dream", "pred_draem"),
    ("dream_epochs", "draem_epochs"),
    ("dream_img", "draem_img"),
    ("dream_prec", "draem_prec"),
    ("dream_fp", "draem_fp"),
    ("dream_rec", "draem_rec"),
    ("dream_auc", "draem_auc"),
    ("dream_result", "draem_result"),
    ("dream_adv_result", "draem_adv_result"),
    ("dream_adv", "draem_adv"),
    ("dream_roc", "draem_roc"),
    ("roc_dream", "roc_draem"),
    ("pr_auc_dream", "pr_auc_draem"),
    ("dream_scores", "draem_scores"),
    ("dream_scores_norm", "draem_scores_norm"),
    ("dream_normal", "draem_normal"),
    ("dream_normal_norm", "draem_normal_norm"),
    ("test_dream_", "test_draem_"),
    ("DREAM_DRAEM_REFERENCE", "DRAEM_REFERENCE"),
    ("DREAM_CRACK_LIKE_ANOMALY", "DRAEM_CRACK_LIKE_ANOMALY"),
    ("DREAM_FEWSHOT_REAL_STRATEGY", "DRAEM_FEWSHOT_REAL_STRATEGY"),
    ('"DREAM"', '"DRAEM"'),
    ("'DREAM'", "'DRAEM'"),
    ('"dream"', '"draem"'),
    ("'dream'", "'draem'"),
]

TEXT_EXTENSIONS = {
    ".py", ".md", ".mdc", ".json", ".csv", ".txt", ".ps1", ".sh", ".yml", ".yaml",
    ".ipynb", ".toml", ".ini", ".bat",
}


def should_process(path: Path) -> bool:
    if path.name in SKIP_FILES:
        return False
    for part in path.parts:
        if part in SKIP_DIRS:
            return False
    return path.suffix.lower() in TEXT_EXTENSIONS


def transform(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    # Standalone DREAM word (docs)
    text = re.sub(r"\bDREAM\b", "DRAEM", text)
    text = re.sub(r"\bdream\b", "draem", text)
    return text


def copy_ml_modules() -> None:
    ml = REPO / "src" / "motionanalyzer" / "ml_models"
    for name in ("dream.py", "dream_temporal.py"):
        src = ml / name
        if not src.exists():
            continue
        dst = ml / name.replace("dream", "draem")
        dst.write_text(transform(src.read_text(encoding="utf-8")), encoding="utf-8")
        print(f"created {dst.relative_to(REPO)}")


PROCESS_ROOTS = [
    REPO / "src",
    REPO / "tests",
    REPO / "scripts",
    REPO / "docs",
    REPO / "reports",
    REPO / ".github",
]


def process_tree() -> int:
    changed = 0
    roots = list(PROCESS_ROOTS) + [REPO / f for f in ("AGENTS.md", "README.md", "CHANGELOG.md", "MIGRATION_GUIDE.md", "RELEASE_NOTES_v0.2.0.md", "GITHUB_SETUP.md", "backup-manifest.json")]
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            paths.append(root)
        elif root.is_dir():
            paths.extend(p for p in root.rglob("*") if p.is_file())
    for path in paths:
        if not should_process(path):
            continue
        if path.suffix == ".py" and "ml_models" in path.parts:
            if path.name in ("dream.py", "dream_temporal.py"):
                continue
        original = path.read_text(encoding="utf-8", errors="replace")
        updated = transform(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
            print(f"updated {path.relative_to(REPO)}")
    return changed


def rename_files() -> None:
    renames = [
        (REPO / "src/motionanalyzer/ml_models/dream.py", REPO / "src/motionanalyzer/ml_models/draem.py"),
        (REPO / "src/motionanalyzer/ml_models/dream_temporal.py", REPO / "src/motionanalyzer/ml_models/draem_temporal.py"),
        (REPO / "tests/test_dream_draem.py", REPO / "tests/test_draem.py"),
        (REPO / "scripts/validate_dream_synthetic.py", REPO / "scripts/validate_draem_synthetic.py"),
        (REPO / "scripts/validate_enhanced_dream.py", REPO / "scripts/validate_enhanced_draem.py"),
        (REPO / "docs/DREAM_DRAEM_REFERENCE.md", REPO / "docs/DRAEM_REFERENCE.md"),
        (REPO / "docs/DREAM_CRACK_LIKE_ANOMALY.md", REPO / "docs/DRAEM_CRACK_LIKE_ANOMALY.md"),
        (REPO / "docs/DREAM_FEWSHOT_REAL_STRATEGY.md", REPO / "docs/DRAEM_FEWSHOT_REAL_STRATEGY.md"),
    ]
    for src, dst in renames:
        if src.exists() and not dst.exists():
            src.rename(dst)
            print(f"renamed {src.relative_to(REPO)} -> {dst.relative_to(REPO)}")
        elif src.exists() and dst.exists():
            src.unlink()
            print(f"removed duplicate {src.relative_to(REPO)}")


def main() -> int:
    copy_ml_modules()
    n = process_tree()
    rename_files()
    # Remove old ml modules if draem exists
    for old in ("dream.py", "dream_temporal.py"):
        p = REPO / "src/motionanalyzer/ml_models" / old
        if p.exists() and (REPO / "src/motionanalyzer/ml_models" / old.replace("dream", "draem")).exists():
            p.unlink()
            print(f"removed {p.relative_to(REPO)}")
    print(f"done: {n} files updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
