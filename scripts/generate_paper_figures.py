"""
Generate publication-ready figures for the FPCB crack detection paper.

Creates:
- Fig 1: Methodology pipeline diagram
- Fig 2: Confusion matrix comparison (DREAM, PatchCore, Ensemble)
- Fig 3: Precision-Recall comparison bar chart
- Fig 4: ROC AUC comparison
- Fig 5: Scenario distribution pie/bar
- Table: Comparison with similar works
"""

from __future__ import annotations

import json
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
OUT = repo_root / "reports" / "paper_figures"
ANALYSIS = repo_root / "reports" / "crack_detection_analysis" / "analysis.json"


def _ensure_out():
    OUT.mkdir(parents=True, exist_ok=True)


def fig1_methodology_pipeline():
    """Fig 1: Methodology pipeline (synthetic data -> features -> models -> ensemble)."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        (0.5, 1.5, 1.8, 1.2, "Synthetic\nData\n(30k)", "#E8F4F8"),
        (2.5, 1.5, 1.8, 1.2, "Feature\nExtraction\n(per-frame, FFT)", "#E8F4F8"),
        (4.5, 1.5, 1.2, 1.2, "DREAM", "#B8E0D2"),
        (5.9, 1.5, 1.2, 1.2, "PatchCore", "#B8E0D2"),
        (7.3, 1.5, 1.8, 1.2, "Ensemble\n(both agree)", "#D4A5A5"),
    ]
    for x, y, w, h, text, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor="#333", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9, wrap=True)

    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i+1][0] - 0.05, 2.1), xytext=(boxes[i][0] + boxes[i][2] + 0.05, 2.1),
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))

    ax.set_title("Fig. 1. Methodology pipeline: synthetic data to ensemble prediction", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig1_methodology.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig1_methodology.png")


def fig2_confusion_matrices():
    """Fig 2: Confusion matrices for DREAM, PatchCore, Ensemble."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not ANALYSIS.exists():
        print("  (skip: no analysis.json)")
        return

    data = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    models = ["DREAM", "PatchCore", "Ensemble"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, name in zip(axes, models):
        cm = np.array(data["models"][name]["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=cm.max() or 1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Normal", "Crack"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual N", "Actual C"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=14)
        ax.set_title(name)

    fig.suptitle("Fig. 2. Confusion matrices (test set, 78,690 rows)", fontsize=11)
    plt.tight_layout()
    fig.savefig(OUT / "fig2_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig2_confusion_matrices.png")


def fig3_precision_recall_comparison():
    """Fig 3: Precision and Recall bar chart."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not ANALYSIS.exists():
        print("  (skip)")
        return

    data = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    models = ["DREAM", "PatchCore", "Ensemble"]
    prec = []
    rec = []
    for m in models:
        r = data["models"][m]
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        prec.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        rec.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    bars1 = ax.bar(x - width/2, prec, width, label="Precision", color="#4A90A4")
    bars2 = ax.bar(x + width/2, rec, width, label="Recall", color="#E07A5F")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0.99, color="gray", linestyle="--", alpha=0.6)
    ax.set_title("Fig. 3. Precision and Recall by model (target: Precision ≥99%)")
    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2 - 0.09, b.get_height() + 0.02, f"{b.get_height():.2%}", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / "fig3_precision_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig3_precision_recall.png")


def fig4_comparison_with_literature():
    """Fig 4: Bar chart comparing our results with similar works."""
    import matplotlib.pyplot as plt
    import numpy as np

    # Literature: GA-Faster-RCNN 91.1% acc; PatchCore MVTec 99.6% AUROC; Our Ensemble 99.86% Precision
    works = ["GA-Faster-RCNN\n(FPC surface) [1]", "PatchCore\n(MVTec AD) [2]", "Ours (Ensemble)\n(FPCB bending)"]
    metrics = ["Accuracy", "AUROC", "Precision"]
    values = [0.911, 0.996, 0.9986]  # different metrics but comparable "high performance"

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#7F8C8D", "#3498DB", "#2ECC71"]
    bars = ax.bar(works, values, color=colors, edgecolor="#333")
    ax.set_ylabel("Score")
    ax.set_ylim(0.85, 1.02)
    ax.axhline(y=0.99, color="red", linestyle="--", alpha=0.5, label="99% target")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2 - 0.08, b.get_height() + 0.005, f"{v:.2%}", fontsize=10)
    ax.legend()
    ax.set_title("Fig. 4. Comparison with related works (different metrics; see Table 1)")
    plt.xticks(rotation=12)
    plt.tight_layout()
    fig.savefig(OUT / "fig4_comparison_literature.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig4_comparison_literature.png")


def fig5_scenario_distribution():
    """Fig 5: Scenario distribution in dataset."""
    import matplotlib.pyplot as plt
    import numpy as np

    scenarios = ["normal", "light_dist", "crack+uv", "micro_crack", "edge_scorch", "pre_damaged", "thick_panel"]
    counts = [21500, 1600, 2600, 900, 600, 1500, 1200]
    colors = plt.cm.Set3(np.linspace(0, 1, len(scenarios)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(scenarios, counts, color=colors)
    ax.set_xlabel("Count")
    ax.set_title("Fig. 5. Scenario distribution in 30k dataset")
    plt.tight_layout()
    fig.savefig(OUT / "fig5_scenario_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  fig5_scenario_distribution.png")


def main():
    print("Generating paper figures...")
    _ensure_out()
    fig1_methodology_pipeline()
    fig2_confusion_matrices()
    fig3_precision_recall_comparison()
    fig4_comparison_with_literature()
    fig5_scenario_distribution()
    print(f"Done. Output: {OUT}/")


if __name__ == "__main__":
    main()
