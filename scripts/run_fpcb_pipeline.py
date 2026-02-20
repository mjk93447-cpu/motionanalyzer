"""Compatibility wrapper for CLI `run-fpcb-pipeline`.

Execute from repo root:
  python scripts/run_fpcb_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parent.parent
if str(_repo / "src") not in sys.path:
    sys.path.insert(0, str(_repo / "src"))

from motionanalyzer.cli import run_fpcb_pipeline


def main() -> None:
    run_fpcb_pipeline()


if __name__ == "__main__":
    main()
