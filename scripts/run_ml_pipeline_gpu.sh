#!/bin/bash
# VM GPU 환경에서 ML 파이프라인 일괄 실행
# 사용: ./scripts/run_ml_pipeline_gpu.sh
# 또는: bash scripts/run_ml_pipeline_gpu.sh

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# venv 활성화 (있는 경우)
if [ -d ".venv-gpu" ]; then
    source .venv-gpu/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# MA_WORKERS, MA_DEVICE: A.5, A.3/A.4 적용 후 사용
WORKERS=${MA_WORKERS:-4}
DEVICE=${MA_DEVICE:-cuda}

echo "=== ML Pipeline (GPU) ==="
echo "  REPO_ROOT=$REPO_ROOT"
echo ""

echo "[1/4] Generating synthetic data..."
# TODO: --workers 지원 시: python scripts/generate_ml_dataset.py --workers $WORKERS
python scripts/generate_ml_dataset.py

echo "[2/4] Goal 1 ML evaluation (DREAM/PatchCore)..."
# TODO: --device 지원 시: python scripts/evaluate_goal1_ml.py --device $DEVICE
python scripts/evaluate_goal1_ml.py

echo "[3/4] Goal 2 ML evaluation..."
# TODO: --device 지원 시: python scripts/evaluate_goal2_ml.py --device $DEVICE
python scripts/evaluate_goal2_ml.py

echo "[4/4] Goal achievement summary..."
python scripts/evaluate_goals_summary.py

echo ""
echo "Done. Reports: $REPO_ROOT/reports/"
echo "  - goal1_ml_evaluation.json"
echo "  - goal2_ml_evaluation.json"
echo "  - goal_achievement_summary.md"
