"""
Summarize goal achievement from evaluation reports.

Reads: goal1_cpd_evaluation.json, goal1_ml_evaluation.json, goal2_ml_evaluation.json.
Output: reports/goal_achievement_summary.md

Target: Maximize Precision-Recall for bending-in-process crack (Goal 1).
"""

from __future__ import annotations

import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
REPORTS = repo_root / "reports"


def main() -> None:
    lines: list[str] = [
        "# Goal Achievement Summary",
        "",
        "Target: **Precision-Recall maximization** for bending-in-process crack detection.",
        "",
    ]

    # Goal 1 CPD
    g1_cpd_path = REPORTS / "goal1_cpd_evaluation.json"
    if g1_cpd_path.exists():
        g1 = json.loads(g1_cpd_path.read_text(encoding="utf-8"))
        lines.extend([
            "## Goal 1: Bending-in-process crack — CPD (change point)",
            "",
            f"- **Metric**: CPD accuracy (detected vs crack_frame)",
            f"- **n_evaluated**: {g1.get('n_evaluated', 0)}",
            f"- **mean_error_frames**: {g1.get('mean_error_frames', 'N/A')}",
            f"- **within_5_frames_pct**: {g1.get('within_5_frames_pct', 'N/A')}%",
            "",
        ])
    else:
        lines.extend(["## Goal 1 CPD: (run evaluate_goal1_cpd.py first)", "", ""])

    # Goal 1 ML (DREAM/PatchCore) — primary metric
    g1_ml_path = REPORTS / "goal1_ml_evaluation.json"
    if g1_ml_path.exists():
        g1_ml = json.loads(g1_ml_path.read_text(encoding="utf-8"))
        lines.extend([
            "## Goal 1: Bending-in-process crack — ML (DREAM/PatchCore, PR maximization)",
            "",
            f"- **n_train**: {g1_ml.get('n_train', 0)}, **n_test**: {g1_ml.get('n_test', 0)}",
            f"- **n_crack_test**: {g1_ml.get('n_crack_test', 'N/A')}, **n_normal_test**: {g1_ml.get('n_normal_test', 'N/A')}",
            "",
        ])
        for model, metrics in g1_ml.get("models", {}).items():
            if "error" in metrics:
                lines.append(f"- **{model}**: error - {metrics['error']}")
            else:
                p, r, f1 = metrics.get("precision", "N/A"), metrics.get("recall", "N/A"), metrics.get("f1", "N/A")
                pr_auc = metrics.get("pr_auc", "N/A")
                lines.append(f"- **{model}**: Precision={p}, Recall={r}, F1={f1}, PR AUC={pr_auc}")
        lines.append("")
    else:
        lines.extend(["## Goal 1 ML: (run evaluate_goal1_ml.py first)", "", ""])

    # Goal 2
    g2_path = REPORTS / "goal2_ml_evaluation.json"
    if g2_path.exists():
        g2 = json.loads(g2_path.read_text(encoding="utf-8"))
        lines.extend([
            "## Goal 2: Already-cracked panel detection (ML)",
            "",
            f"- **n_train**: {g2.get('n_train', 0)}, **n_test**: {g2.get('n_test', 0)}",
            "",
        ])
        for model, metrics in g2.get("models", {}).items():
            if "error" in metrics:
                lines.append(f"- **{model}**: error - {metrics['error']}")
            else:
                lines.append(f"- **{model}**: ROC AUC={metrics.get('roc_auc', 'N/A')}, PR AUC={metrics.get('pr_auc', 'N/A')}")
        lines.append("")
    else:
        lines.extend(["## Goal 2: (run evaluate_goal2_ml.py first)", "", ""])

    out_path = REPORTS / "goal_achievement_summary.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
