from __future__ import annotations

import os
import platform
import subprocess
import sys
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
import pandas as pd

from motionanalyzer.analysis import (
    AnalysisSummary,
    compare_summaries,
    load_summary,
    run_analysis,
)
from motionanalyzer.crack_model import (
    CrackModelParams,
    get_user_params_path,
    load_params,
    save_params,
)
from motionanalyzer.paths import get_default_dream_model_path, get_default_patchcore_model_path
from motionanalyzer.time_series.changepoint import ChangePointResult
from motionanalyzer.visualization import create_full_vector_map_figure

DEFAULT_INPUT_DIR = "data/synthetic/normal_case"
DEFAULT_OUTPUT_DIR = "exports/vectors/normal_case"
DEFAULT_BASE_SUMMARY = "exports/vectors/normal_case/summary.json"
DEFAULT_CANDIDATE_SUMMARY = "exports/vectors/crack_case/summary.json"


def _project_root() -> Path:
    """Project root (cwd or repo containing src/motionanalyzer). Avoids Korean-path issues."""
    try:
        # From src/motionanalyzer/desktop_gui.py -> repo root
        root = Path(__file__).resolve().parent.parent.parent
        if (root / "src" / "motionanalyzer").exists() or (root / "docs").exists():
            return root
    except Exception:
        pass
    return Path.cwd()


def _resolve_path(p: str) -> Path:
    """Resolve path; relative paths are resolved against project root (avoids encoding issues)."""
    path = Path(p).expanduser()
    if not path.is_absolute():
        path = _project_root() / path
    return path


class MotionAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("motionanalyzer - FPCB bending analysis (offline Windows GUI)")
        # Let user resize freely; start in a reasonable size.
        self.geometry("1280x800")

        self._build_menu()
        self._build_widgets()

    # --------------------------------------------------------------------- #
    # UI construction
    # --------------------------------------------------------------------- #
    def _build_menu(self) -> None:
        """Build menu bar with Help menu."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)

        help_menu.add_command(label="Quick Start Guide", command=self._show_quick_start)
        help_menu.add_command(label="User Guide", command=self._show_user_guide)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)

    def _show_quick_start(self) -> None:
        """Show quick start guide in a message box."""
        guide = """Quick Start Guide

1. Analyze Tab:
   - Select input dataset folder (contains frame_*.txt and fps.txt)
   - Choose output directory
   - Select analysis mode (Physics/DREAM/PatchCore/Ensemble/Temporal)
   - Click "Run Analysis"

2. Compare Tab:
   - Select base summary.json (normal case)
   - Select candidate summary.json (test case)
   - Click "Compare" to see differences

3. Time Series Analysis Tab:
   - Select dataset folder
   - Choose detection method (CUSUM/Window-based/PELT)
   - Select feature (e.g., acceleration_max)
   - Enable auto-tuning for optimal parameters
   - Click "Detect Change Points"

4. ML & Optimization Tab:
   - Select normal datasets (for training)
   - Select crack datasets (for validation)
   - Choose model type and options
   - Click "Train Model"

For detailed documentation, see docs/USER_GUIDE.md"""
        messagebox.showinfo("Quick Start Guide", guide)

    def _show_user_guide(self) -> None:
        """Show user guide information."""
        # src/motionanalyzer/desktop_gui.py -> repo_root = parent.parent.parent
        guide_path = Path(__file__).resolve().parent.parent.parent / "docs" / "USER_GUIDE.md"
        if guide_path.exists():
            try:
                import webbrowser
                import platform
                # Check if we're in a CI environment (no display)
                # On Windows, DISPLAY env var is not used, so only check CI
                is_ci = os.environ.get("CI") == "true"
                is_windows = platform.system() == "Windows"
                if is_ci or (not is_windows and not os.environ.get("DISPLAY")):
                    raise OSError("No display available")
                webbrowser.open(guide_path.as_uri())
            except (OSError, ImportError):
                messagebox.showinfo(
                    "User Guide",
                    f"User guide is available at:\n{guide_path}\n\n"
                    "Please open it with a text editor or markdown viewer.",
                )
        else:
            messagebox.showinfo(
                "User Guide",
                "User guide documentation:\n\n"
                "See README.md for basic usage.\n"
                "See docs/USER_GUIDE.md for detailed instructions.\n\n"
                "Key features:\n"
                "- Analyze: Vector analysis and visualization\n"
                "- Compare: Compare two analysis results\n"
                "- ML & Optimization: Train anomaly detection models\n"
                "- Time Series Analysis: Change point detection",
            )

    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """motionanalyzer v0.2.0

FPCB Bending Analysis Tool
Physics-based time-series motion analyzer

Features:
- Vector analysis (position, velocity, acceleration)
- Synthetic data generation (5 scenarios)
- ML-based anomaly detection (DREAM, PatchCore, Ensemble, Temporal)
- Change Point Detection (CUSUM, Window-based, PELT)
- Advanced feature engineering

For more information:
- User Guide: docs/USER_GUIDE.md
- Development Roadmap: docs/DEVELOPMENT_ROADMAP_FINAL.md
- Project Status: docs/PROJECT_READINESS_ASSESSMENT.md

© 2026 motionanalyzer contributors"""
        messagebox.showinfo("About motionanalyzer", about_text)

    def _build_widgets(self) -> None:
        notebook = ttk.Notebook(self)
        frame_analyze = ttk.Frame(notebook)
        frame_compare = ttk.Frame(notebook)
        frame_tuning = ttk.Frame(notebook)
        frame_auto_opt = ttk.Frame(notebook)
        frame_timeseries = ttk.Frame(notebook)
        frame_synthetic_goals = ttk.Frame(notebook)
        notebook.add(frame_analyze, text="Analyze")
        notebook.add(frame_compare, text="Compare")
        notebook.add(frame_tuning, text="Crack Model Tuning")
        notebook.add(frame_auto_opt, text="ML & Optimization")
        notebook.add(frame_timeseries, text="Time Series Analysis")
        notebook.add(frame_synthetic_goals, text="Synthetic & Goals")
        notebook.pack(fill=tk.BOTH, expand=True)

        self._build_analyze_tab(frame_analyze)
        self._build_compare_tab(frame_compare)
        self._build_tuning_tab(frame_tuning)
        self._build_ml_optimization_tab(frame_auto_opt)
        self._build_timeseries_tab(frame_timeseries)
        self._build_synthetic_goals_tab(frame_synthetic_goals)

    def _build_analyze_tab(self, parent: tk.Widget) -> None:
        input_frame = ttk.LabelFrame(parent, text="Input / Output")
        input_frame.pack(fill=tk.X, padx=8, pady=4)

        self.input_dir_var = tk.StringVar(value=DEFAULT_INPUT_DIR)
        self.output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)
        self.fps_var = tk.StringVar(value="30.0")
        self.analysis_mode_var = tk.StringVar(value="physics")
        self.dream_model_path_var = tk.StringVar(value=str(get_default_dream_model_path()))
        self.patchcore_model_path_var = tk.StringVar(value=str(get_default_patchcore_model_path()))

        # Input dir
        row = ttk.Frame(input_frame)
        row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row, text="Input bundle path:").pack(side=tk.LEFT)
        entry_in = ttk.Entry(row, textvariable=self.input_dir_var, width=80)
        entry_in.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse...", command=self._browse_input_dir).pack(side=tk.LEFT)

        # Output dir
        row2 = ttk.Frame(input_frame)
        row2.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row2, text="Output analysis path:").pack(side=tk.LEFT)
        entry_out = ttk.Entry(row2, textvariable=self.output_dir_var, width=80)
        entry_out.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(row2, text="Browse...", command=self._browse_output_dir).pack(side=tk.LEFT)

        # FPS
        row3 = ttk.Frame(input_frame)
        row3.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row3, text="FPS (frames per second):").pack(side=tk.LEFT)
        entry_fps = ttk.Entry(row3, textvariable=self.fps_var, width=10)
        entry_fps.pack(side=tk.LEFT, padx=4)

        # Scale (mm/px) for SI units: 1 px = this many mm → m/px = value * 1e-3
        self.scale_mm_per_px_var = tk.StringVar(value="")
        row_scale = ttk.Frame(input_frame)
        row_scale.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row_scale, text="Scale (mm/px, optional):").pack(side=tk.LEFT)
        ttk.Entry(row_scale, textvariable=self.scale_mm_per_px_var, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Label(row_scale, text="1 px = this many mm. Empty = use metadata or px units.").pack(side=tk.LEFT, padx=4)

        # Analysis mode
        row4 = ttk.Frame(input_frame)
        row4.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row4, text="Analysis mode:").pack(side=tk.LEFT)
        mode_combo = ttk.Combobox(
            row4,
            textvariable=self.analysis_mode_var,
            values=["physics", "dream", "patchcore", "ensemble"],
            state="readonly",
            width=18,
        )
        mode_combo.pack(side=tk.LEFT, padx=4)

        # Model paths (used for DREAM/PatchCore inference)
        model_frame = ttk.LabelFrame(parent, text="Models (for DREAM/PatchCore)")
        model_frame.pack(fill=tk.X, padx=8, pady=4)

        row_m1 = ttk.Frame(model_frame)
        row_m1.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row_m1, text="DREAM model path:").pack(side=tk.LEFT)
        ttk.Entry(row_m1, textvariable=self.dream_model_path_var, width=80).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True
        )
        ttk.Button(row_m1, text="Browse...", command=self._browse_dream_model).pack(side=tk.LEFT)

        row_m2 = ttk.Frame(model_frame)
        row_m2.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row_m2, text="PatchCore model path:").pack(side=tk.LEFT)
        ttk.Entry(row_m2, textvariable=self.patchcore_model_path_var, width=80).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True
        )
        ttk.Button(row_m2, text="Browse...", command=self._browse_patchcore_model).pack(side=tk.LEFT)

        # Run button
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(
            btn_frame,
            text="Run Analysis",
            command=self._on_run_analysis,
        ).pack(side=tk.LEFT)

        # Summary display
        summary_frame = ttk.LabelFrame(parent, text="Summary")
        summary_frame.pack(fill=tk.X, padx=8, pady=4)
        self.summary_text = tk.Text(summary_frame, height=8, wrap=tk.NONE)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Anomaly scores display (DREAM/PatchCore)
        self.anomaly_frame = ttk.LabelFrame(parent, text="Anomaly Scores (DREAM/PatchCore)")
        self.anomaly_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._anomaly_canvas: Any = None
        self._anomaly_toolbar: Any = None
        self._anomaly_fig: Any = None

        # Vector map display (zoom/pan enabled)
        self.vector_map_frame = ttk.LabelFrame(parent, text="Vector Map (zoom/pan with toolbar)")
        self.vector_map_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._vector_map_canvas: Any = None
        self._vector_map_toolbar: Any = None
        self._vector_map_fig: Any = None

        # Log area
        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.log_text = tk.Text(log_frame, height=6, wrap=tk.NONE)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def _build_compare_tab(self, parent: tk.Widget) -> None:
        paths_frame = ttk.LabelFrame(parent, text="Summary paths")
        paths_frame.pack(fill=tk.X, padx=8, pady=4)

        self.base_summary_var = tk.StringVar(value=DEFAULT_BASE_SUMMARY)
        self.cand_summary_var = tk.StringVar(value=DEFAULT_CANDIDATE_SUMMARY)

        # Base summary
        row = ttk.Frame(paths_frame)
        row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row, text="Base summary.json:").pack(side=tk.LEFT)
        entry_base = ttk.Entry(row, textvariable=self.base_summary_var, width=80)
        entry_base.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse...", command=self._browse_base_summary).pack(side=tk.LEFT)

        # Candidate summary
        row2 = ttk.Frame(paths_frame)
        row2.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row2, text="Candidate summary.json:").pack(side=tk.LEFT)
        entry_cand = ttk.Entry(row2, textvariable=self.cand_summary_var, width=80)
        entry_cand.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(row2, text="Browse...", command=self._browse_cand_summary).pack(side=tk.LEFT)

        # Compare button
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(
            btn_frame,
            text="Compare Results",
            command=self._on_compare_summaries,
        ).pack(side=tk.LEFT)

        # Result area
        result_frame = ttk.LabelFrame(parent, text="Delta (candidate - base)")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.compare_text = tk.Text(result_frame, height=12, wrap=tk.NONE)
        self.compare_text.pack(fill=tk.BOTH, expand=True)

    # --------------------------------------------------------------------- #
    # Browse helpers
    # --------------------------------------------------------------------- #
    def _browse_input_dir(self) -> None:
        path = filedialog.askdirectory(
            title="Select input bundle directory",
            initialdir=str(_project_root()),
        )
        if path:
            self.input_dir_var.set(path)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(
            title="Select output directory",
            initialdir=str(_project_root()),
        )
        if path:
            self.output_dir_var.set(path)

    def _browse_dream_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select DREAM model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
            initialdir=str(_project_root()),
        )
        if path:
            self.dream_model_path_var.set(path)

    def _browse_patchcore_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select PatchCore model",
            filetypes=[("PatchCore model", "*.npz"), ("All files", "*.*")],
            initialdir=str(_project_root()),
        )
        if path:
            self.patchcore_model_path_var.set(path)

    def _browse_base_summary(self) -> None:
        path = filedialog.askopenfilename(
            title="Select base summary.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(_project_root()),
        )
        if path:
            self.base_summary_var.set(path)

    def _browse_cand_summary(self) -> None:
        path = filedialog.askopenfilename(
            title="Select candidate summary.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(_project_root()),
        )
        if path:
            self.cand_summary_var.set(path)

    # --------------------------------------------------------------------- #
    # Analyze tab handlers
    # --------------------------------------------------------------------- #
    def _append_log(self, msg: str) -> None:
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _clear_vector_map(self) -> None:
        for w in self.vector_map_frame.winfo_children():
            w.destroy()
        if self._vector_map_fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._vector_map_fig)
            self._vector_map_fig = None
        self._vector_map_canvas = None
        self._vector_map_toolbar = None

    def _show_vector_map(
        self,
        vectors_csv: Path,
        fps: float,
        *,
        meters_per_pixel: float | None = None,
    ) -> None:
        self._clear_vector_map()
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            fig = create_full_vector_map_figure(
                vectors_csv, fps, meters_per_pixel=meters_per_pixel
            )
            self._vector_map_fig = fig
            canvas = FigureCanvasTkAgg(fig, master=self.vector_map_frame)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, self.vector_map_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._vector_map_canvas = canvas
            self._vector_map_toolbar = toolbar
        except Exception as exc:  # pragma: no cover
            self._append_log(f"Vector map display failed: {exc}")

    def _clear_anomaly_plot(self) -> None:
        for w in self.anomaly_frame.winfo_children():
            w.destroy()
        if self._anomaly_fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._anomaly_fig)
            self._anomaly_fig = None
        self._anomaly_canvas = None
        self._anomaly_toolbar = None

    def _show_anomaly_plot(self, frames: np.ndarray, scores: np.ndarray, out_path: Path) -> None:
        self._clear_anomaly_plot()
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5), constrained_layout=True)
            ax1.plot(frames, scores, lw=1.5)
            ax1.set_title("Anomaly score over frames")
            ax1.set_xlabel("frame")
            ax1.set_ylabel("score")

            ax2.hist(scores, bins=40)
            ax2.set_title("Anomaly score histogram")
            ax2.set_xlabel("score")
            ax2.set_ylabel("count")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=160)

            self._anomaly_fig = fig
            canvas = FigureCanvasTkAgg(fig, master=self.anomaly_frame)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, self.anomaly_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._anomaly_canvas = canvas
            self._anomaly_toolbar = toolbar
        except Exception as exc:  # pragma: no cover
            self._append_log(f"Anomaly plot display failed: {exc}")

    def _build_inference_features(self, vectors_csv: Path, frame_metrics_csv: Path) -> pd.DataFrame:
        vectors = pd.read_csv(vectors_csv)
        agg_spec: dict[str, list[str]] = {}
        for base in [
            "strain_surrogate",
            "stress_surrogate",
            "impact_surrogate",
            "curvature_like",
            "acceleration",
            "speed",
        ]:
            if base in vectors.columns:
                agg_spec[base] = ["mean", "max", "std"]
        if not agg_spec:
            raise ValueError(f"No supported feature columns found in {vectors_csv}")

        per_frame = vectors.groupby("frame").agg(agg_spec).reset_index()
        per_frame.columns = ["frame"] + [f"{c[0]}_{c[1]}" for c in per_frame.columns[1:]]

        if frame_metrics_csv.exists():
            fm = pd.read_csv(frame_metrics_csv)
            cols = [
                c
                for c in ["frame", "bend_angle_deg", "curvature_concentration", "est_max_strain"]
                if c in fm.columns
            ]
            if "frame" in cols and len(cols) > 1:
                per_frame = per_frame.merge(fm[cols], on="frame", how="left")

        per_frame = per_frame.sort_values("frame").reset_index(drop=True)
        return per_frame

    def _load_dream_model(self) -> Any:
        model_path = _resolve_path(self.dream_model_path_var.get())
        if not model_path.exists():
            raise FileNotFoundError(
                f"DREAM model not found: {model_path}\n"
                "Train a model first in 'ML & Optimization' tab, or use Physics mode."
            )
        try:
            from motionanalyzer.ml_models.dream import DREAMPyTorch
        except ImportError as e:
            raise ImportError(
                "PyTorch not available. Install with: pip install -e '.[ml]'\n"
                "Or use Physics mode instead."
            ) from e

        model = DREAMPyTorch(input_dim=1)
        model.load(model_path)
        return model

    def _load_patchcore_model(self) -> Any:
        model_path = _resolve_path(self.patchcore_model_path_var.get())
        if not model_path.exists():
            raise FileNotFoundError(
                f"PatchCore model not found: {model_path}\n"
                "Train a model first in 'ML & Optimization' tab, or use Physics mode."
            )
        try:
            from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
        except ImportError as e:
            raise ImportError(
                "scikit-learn not available. Install with: pip install -e '.[ml]'\n"
                "Or use Physics mode instead."
            ) from e

        model = PatchCoreScikitLearn(feature_dim=1)
        model.load(model_path)
        return model

    def _load_ensemble_model(self) -> Any:
        dream_model_path = _resolve_path(self.dream_model_path_var.get())
        patchcore_model_path = _resolve_path(self.patchcore_model_path_var.get())
        if not dream_model_path.exists():
            raise FileNotFoundError(
                f"DREAM model not found: {dream_model_path}\n"
                "Train a DREAM model first in 'ML & Optimization' tab."
            )
        if not patchcore_model_path.exists():
            raise FileNotFoundError(
                f"PatchCore model not found: {patchcore_model_path}\n"
                "Train a PatchCore model first in 'ML & Optimization' tab."
            )
        try:
            from motionanalyzer.ml_models.hybrid import EnsembleAnomalyDetector, EnsembleStrategy
            from motionanalyzer.ml_models.dream import DREAMPyTorch
            from motionanalyzer.ml_models.patchcore import PatchCoreScikitLearn
        except ImportError as e:
            raise ImportError(
                "ML dependencies not available. Install with: pip install -e '.[ml]'\n"
                "Or use Physics mode instead."
            ) from e

        dream_model = DREAMPyTorch(input_dim=1)
        dream_model.load(dream_model_path)
        patchcore_model = PatchCoreScikitLearn(feature_dim=1)
        patchcore_model.load(patchcore_model_path)

        ensemble = EnsembleAnomalyDetector(
            dream_model=dream_model,
            patchcore_model=patchcore_model,
            strategy=EnsembleStrategy.WEIGHTED_AVERAGE,
            dream_weight=0.5,
            patchcore_weight=0.5,
        )

        # Try to load ensemble config if exists
        from motionanalyzer.paths import get_user_models_dir
        ensemble_config_path = get_user_models_dir() / "ensemble_config.json"
        if ensemble_config_path.exists():
            ensemble.load(ensemble_config_path)
        else:
            # Set default threshold from normal data if available
            # This is a placeholder; in practice, threshold should be set during training
            pass

        return ensemble

    def _clear_summary(self) -> None:
        self.summary_text.delete("1.0", tk.END)

    def _render_summary(self, summary: AnalysisSummary) -> None:
        self._clear_summary()
        data: dict[str, Any] = {
            "fps": summary.fps,
            "frame_count": summary.frame_count,
            "point_count_per_frame_min": summary.point_count_per_frame_min,
            "point_count_per_frame_max": summary.point_count_per_frame_max,
            "unique_index_count": summary.unique_index_count,
            "mean_speed": summary.mean_speed,
            "max_speed": summary.max_speed,
            "mean_acceleration": summary.mean_acceleration,
            "max_acceleration": summary.max_acceleration,
            "mean_curvature_like": summary.mean_curvature_like,
            "p95_curvature_like": summary.p95_curvature_like,
            "max_curvature_like": summary.max_curvature_like,
        }
        for k, v in data.items():
            self.summary_text.insert(tk.END, f"{k}: {v}\n")
        self.summary_text.see(tk.END)

    def _on_run_analysis(self) -> None:
        input_dir = _resolve_path(self.input_dir_var.get())
        output_dir = _resolve_path(self.output_dir_var.get())
        mode = self.analysis_mode_var.get().strip().lower()

        if not input_dir.exists():
            messagebox.showerror("Error", f"Input path not found:\n{input_dir}")
            return

        # FPS from user input (required for GUI; fps.txt is optional/ignored here)
        try:
            fps_val = float(self.fps_var.get())
        except ValueError:
            messagebox.showerror("Error", "FPS must be a positive number.")
            return
        if fps_val <= 0:
            messagebox.showerror("Error", "FPS must be a positive number.")
            return

        # Optional scale (mm/px) for SI: m/px = mm_per_px * 1e-3
        meters_per_pixel_override = None
        scale_str = (self.scale_mm_per_px_var.get() or "").strip()
        if scale_str:
            try:
                mm_per_px = float(scale_str)
                if mm_per_px > 0:
                    meters_per_pixel_override = mm_per_px * 1e-3
                    self._append_log(f"Using scale: {mm_per_px} mm/px → {meters_per_pixel_override:.2e} m/px (SI units).")
            except ValueError:
                messagebox.showwarning("Scale", "Scale (mm/px) must be a positive number; ignoring.")
        try:
            self._append_log(f"Running analysis...\n  input={input_dir}\n  output={output_dir}")
            summary = run_analysis(
                input_dir=input_dir,
                output_dir=output_dir,
                fps=fps_val,
                meters_per_pixel_override=meters_per_pixel_override,
            )
            self._append_log("Analysis complete.")
            self._render_summary(summary)
            vectors_csv = output_dir / "vectors.csv"
            if vectors_csv.exists():
                self._show_vector_map(
                    vectors_csv, fps_val, meters_per_pixel=summary.meters_per_pixel
                )

            # Optional: ML inference for anomaly scores
            self._clear_anomaly_plot()
            if mode in {"dream", "patchcore", "ensemble"} and vectors_csv.exists():
                self._append_log(f"Running {mode} inference (requires a saved model)...")
                features_df = self._build_inference_features(vectors_csv, input_dir / "frame_metrics.csv")
                feature_cols = [c for c in features_df.columns if c != "frame" and "crack_risk" not in c.lower()]
                X_df = features_df[feature_cols].fillna(0.0)

                if mode == "dream":
                    model = self._load_dream_model()
                    X = X_df.to_numpy(dtype=np.float32)
                    scores = model.predict(X)
                    preds = model.predict_binary(X)
                elif mode == "patchcore":
                    model = self._load_patchcore_model()
                    scores = model.predict(X_df)
                    preds = model.predict_binary(X_df)
                else:  # ensemble
                    ensemble = self._load_ensemble_model()
                    scores = ensemble.predict(X_df)
                    preds = ensemble.predict_binary(X_df)

                frames = features_df["frame"].to_numpy(dtype=int)
                scores = np.asarray(scores, dtype=float)
                preds = np.asarray(preds, dtype=int)

                scores_out = pd.DataFrame({"frame": frames, "anomaly_score": scores, "is_anomaly": preds})
                scores_csv = output_dir / f"{mode}_anomaly_scores.csv"
                scores_png = output_dir / f"{mode}_anomaly_scores.png"
                scores_out.to_csv(scores_csv, index=False, encoding="utf-8")
                self._show_anomaly_plot(frames=frames, scores=scores, out_path=scores_png)
                self._append_log(
                    f"{mode} inference complete.\n"
                    f"  anomaly_rate={float(preds.mean()):.3f}\n"
                    f"  scores_csv={scores_csv}\n"
                    f"  plot_png={scores_png}"
                )

            messagebox.showinfo(
                "Success",
                f"Analysis complete.\n\nOutput directory:\n{output_dir}\n"
                f"Includes: vectors.csv, vector_map.png, summary.json",
            )
        except Exception as exc:  # pragma: no cover - defensive, surfaced via GUI
            tb = traceback.format_exc()
            self._append_log(f"ERROR: {exc}\n{tb}")
            messagebox.showerror(
                "Error",
                f"Analysis failed:\n{exc}\n\nDetails:\n{tb}",
            )

    # --------------------------------------------------------------------- #
    # Compare tab handlers
    # --------------------------------------------------------------------- #
    def _on_compare_summaries(self) -> None:
        base_path = _resolve_path(self.base_summary_var.get())
        cand_path = _resolve_path(self.cand_summary_var.get())

        missing: list[str] = []
        if not base_path.exists():
            missing.append(str(base_path))
        if not cand_path.exists():
            missing.append(str(cand_path))
        if missing:
            messagebox.showerror("Error", "Missing file(s):\n" + "\n".join(missing))
            return

        try:
            base = load_summary(base_path)
            cand = load_summary(cand_path)
            delta = compare_summaries(base_summary=base, candidate_summary=cand)
            self.compare_text.delete("1.0", tk.END)
            for k, v in delta.items():
                self.compare_text.insert(tk.END, f"{k}: {v:.6f}\n")
            self.compare_text.see(tk.END)
        except Exception as exc:  # pragma: no cover - defensive, surfaced via GUI
            tb = traceback.format_exc()
            messagebox.showerror(
                "Error",
                f"Compare failed:\n{exc}\n\nDetails:\n{tb}",
            )

    # --------------------------------------------------------------------- #
    # Crack Model Tuning tab
    # --------------------------------------------------------------------- #
    def _build_tuning_tab(self, parent: tk.Widget) -> None:
        """Build Crack Model Tuning tab for parameter adjustment."""
        # Load current params (user settings or default)
        self.crack_params = load_params(get_user_params_path())

        # Top: Load/Save buttons
        file_frame = ttk.LabelFrame(parent, text="Parameter File")
        file_frame.pack(fill=tk.X, padx=8, pady=4)
        btn_file = ttk.Frame(file_frame)
        btn_file.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(btn_file, text="Load from File...", command=self._load_params_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_file, text="Save to File...", command=self._save_params_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_file, text="Reset to Default", command=self._reset_params).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_file, text="Save to User Config", command=self._save_user_params).pack(side=tk.LEFT, padx=4)

        # Scrollable frame for parameters
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Parameter variables and widgets
        self.param_vars: dict[str, tk.DoubleVar] = {}
        self.param_widgets: dict[str, tuple[ttk.Scale | ttk.Entry, ttk.Label]] = {}

        # Caps section
        caps_frame = ttk.LabelFrame(scrollable_frame, text="Normalization Caps")
        caps_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add_param_slider(caps_frame, "strain_cap", 0.0, 0.1, self.crack_params.strain_cap, "Strain cap")
        self._add_param_slider(caps_frame, "curvature_concentration_cap", 1.0, 20.0, self.crack_params.curvature_concentration_cap, "Curvature concentration cap")
        self._add_param_slider(caps_frame, "bend_angle_cap_deg", 90.0, 360.0, self.crack_params.bend_angle_cap_deg, "Bend angle cap (deg)")
        self._add_param_slider(caps_frame, "impact_cap_px_s2", 1000.0, 20000.0, self.crack_params.impact_cap_px_s2, "Impact cap (px/s²)")

        # Weights section
        weights_frame = ttk.LabelFrame(scrollable_frame, text="Weights")
        weights_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add_param_slider(weights_frame, "w_strain", 0.0, 1.0, self.crack_params.w_strain, "Weight: Strain")
        self._add_param_slider(weights_frame, "w_stress", 0.0, 1.0, self.crack_params.w_stress, "Weight: Stress")
        self._add_param_slider(weights_frame, "w_curvature_concentration", 0.0, 1.0, self.crack_params.w_curvature_concentration, "Weight: Curvature concentration")
        self._add_param_slider(weights_frame, "w_bend_angle", 0.0, 1.0, self.crack_params.w_bend_angle, "Weight: Bend angle")
        self._add_param_slider(weights_frame, "w_impact", 0.0, 1.0, self.crack_params.w_impact, "Weight: Impact")

        # Sigmoid section
        sigmoid_frame = ttk.LabelFrame(scrollable_frame, text="Sigmoid Parameters")
        sigmoid_frame.pack(fill=tk.X, padx=8, pady=4)
        self._add_param_slider(sigmoid_frame, "sigmoid_steepness", 1.0, 20.0, self.crack_params.sigmoid_steepness, "Sigmoid steepness")
        self._add_param_slider(sigmoid_frame, "sigmoid_center", 0.0, 1.0, self.crack_params.sigmoid_center, "Sigmoid center")

        # Preview section (optional: show crack_risk distribution for selected dataset)
        preview_frame = ttk.LabelFrame(parent, text="Preview (select dataset to test parameters)")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        preview_inner = ttk.Frame(preview_frame)
        preview_inner.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(preview_inner, text="Test dataset:").pack(side=tk.LEFT)
        self.preview_dataset_var = tk.StringVar()
        ttk.Entry(preview_inner, textvariable=self.preview_dataset_var, width=60).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(preview_inner, text="Browse...", command=self._browse_preview_dataset).pack(side=tk.LEFT, padx=4)
        ttk.Button(preview_inner, text="Preview", command=self._preview_params).pack(side=tk.LEFT, padx=4)
        self.preview_text = tk.Text(preview_frame, height=6, wrap=tk.NONE)
        self.preview_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _add_param_slider(self, parent: tk.Widget, param_name: str, min_val: float, max_val: float, initial: float, label: str) -> None:
        """Add a parameter slider with label and value display."""
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text=label, width=30).pack(side=tk.LEFT)
        var = tk.DoubleVar(value=initial)
        self.param_vars[param_name] = var
        slider = ttk.Scale(row, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL, length=300)
        slider.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        value_label = ttk.Label(row, text=f"{initial:.4f}", width=10)
        value_label.pack(side=tk.LEFT, padx=4)
        var.trace_add("write", lambda *args, p=param_name, l=value_label: l.config(text=f"{var.get():.4f}"))
        self.param_widgets[param_name] = (slider, value_label)

    def _load_params_file(self) -> None:
        """Load parameters from JSON file."""
        path = filedialog.askopenfilename(
            title="Load Crack Model Parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(_project_root()),
        )
        if path:
            try:
                self.crack_params = load_params(Path(path))
                self._update_ui_from_params()
                messagebox.showinfo("Success", f"Loaded parameters from {path}")
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to load parameters:\n{exc}")

    def _save_params_file(self) -> None:
        """Save parameters to JSON file."""
        path = filedialog.asksaveasfilename(
            title="Save Crack Model Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(_project_root()),
        )
        if path:
            try:
                params = self._get_params_from_ui()
                save_params(params, Path(path))
                self.crack_params = params
                messagebox.showinfo("Success", f"Saved parameters to {path}")
            except Exception as exc:
                messagebox.showerror("Error", f"Failed to save parameters:\n{exc}")

    def _reset_params(self) -> None:
        """Reset to default parameters."""
        self.crack_params = CrackModelParams()
        self._update_ui_from_params()
        messagebox.showinfo("Success", "Reset to default parameters")

    def _save_user_params(self) -> None:
        """Save parameters to user config directory."""
        try:
            params = self._get_params_from_ui()
            save_params(params, get_user_params_path())
            self.crack_params = params
            messagebox.showinfo("Success", f"Saved to user config:\n{get_user_params_path()}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to save user config:\n{exc}")

    def _update_ui_from_params(self) -> None:
        """Update UI widgets from current params."""
        for param_name, var in self.param_vars.items():
            value = getattr(self.crack_params, param_name)
            var.set(value)

    def _get_params_from_ui(self) -> CrackModelParams:
        """Get CrackModelParams from UI widgets."""
        return CrackModelParams(
            strain_cap=self.param_vars["strain_cap"].get(),
            curvature_concentration_cap=self.param_vars["curvature_concentration_cap"].get(),
            bend_angle_cap_deg=self.param_vars["bend_angle_cap_deg"].get(),
            impact_cap_px_s2=self.param_vars["impact_cap_px_s2"].get(),
            w_strain=self.param_vars["w_strain"].get(),
            w_stress=self.param_vars["w_stress"].get(),
            w_curvature_concentration=self.param_vars["w_curvature_concentration"].get(),
            w_bend_angle=self.param_vars["w_bend_angle"].get(),
            w_impact=self.param_vars["w_impact"].get(),
            sigmoid_steepness=self.param_vars["sigmoid_steepness"].get(),
            sigmoid_center=self.param_vars["sigmoid_center"].get(),
        )

    def _browse_preview_dataset(self) -> None:
        """Browse for dataset to preview parameters."""
        path = filedialog.askdirectory(
            title="Select dataset directory",
            initialdir=str(_project_root()),
        )
        if path:
            self.preview_dataset_var.set(path)

    def _preview_params(self) -> None:
        """Preview crack risk with current parameters on selected dataset."""
        dataset_path = _resolve_path(self.preview_dataset_var.get())
        if not dataset_path.exists():
            messagebox.showerror("Error", f"Dataset path not found:\n{dataset_path}")
            return
        try:
            from motionanalyzer.analysis import load_bundle, compute_vectors
            from motionanalyzer.crack_model import compute_crack_risk, load_frame_metrics

            params = self._get_params_from_ui()
            df, fps_val, meters_per_pixel = load_bundle(input_dir=dataset_path)
            vectors = compute_vectors(df=df, fps=fps_val, meters_per_pixel=meters_per_pixel)
            dt_s = 1.0 / fps_val
            frame_metrics = load_frame_metrics(dataset_path / "frame_metrics.csv")
            vectors = compute_crack_risk(vectors, frame_metrics, dt_s, meters_per_pixel=meters_per_pixel, params=params)

            max_risk = float(vectors["crack_risk"].max())
            mean_risk = float(vectors["crack_risk"].mean())
            p95_risk = float(vectors["crack_risk"].quantile(0.95))

            self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert(tk.END, f"Dataset: {dataset_path}\n")
            self.preview_text.insert(tk.END, f"Frames: {vectors['frame'].nunique()}, Points: {vectors['index'].nunique()}\n")
            self.preview_text.insert(tk.END, f"Max crack risk: {max_risk:.6f}\n")
            self.preview_text.insert(tk.END, f"Mean crack risk: {mean_risk:.6f}\n")
            self.preview_text.insert(tk.END, f"P95 crack risk: {p95_risk:.6f}\n")
            self.preview_text.see(tk.END)
        except Exception as exc:
            tb = traceback.format_exc()
            self.preview_text.delete("1.0", tk.END)
            self.preview_text.insert(tk.END, f"Preview failed:\n{exc}\n\n{tb}")
            self.preview_text.see(tk.END)

    # --------------------------------------------------------------------- #
    # ML & Optimization tab (model modes: physics, dream, patchcore, grid_search, bayesian)
    # --------------------------------------------------------------------- #
    def _build_ml_optimization_tab(self, parent: tk.Widget) -> None:
        """Build ML & Optimization tab: select mode then train or optimize (runners dispatch)."""
        # Dataset selection
        datasets_frame = ttk.LabelFrame(parent, text="Dataset Selection")
        datasets_frame.pack(fill=tk.X, padx=8, pady=4)

        # Normal datasets
        normal_frame = ttk.Frame(datasets_frame)
        normal_frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(normal_frame, text="Normal datasets (folders):").pack(side=tk.LEFT)
        self.normal_datasets_var = tk.StringVar()
        ttk.Entry(normal_frame, textvariable=self.normal_datasets_var, width=60).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(normal_frame, text="Add...", command=self._add_normal_dataset).pack(side=tk.LEFT, padx=4)
        self.normal_datasets_listbox = tk.Listbox(normal_frame, height=3)
        self.normal_datasets_listbox.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(normal_frame, text="Remove", command=self._remove_normal_dataset).pack(side=tk.LEFT, padx=4)

        # Crack datasets
        crack_frame = ttk.Frame(datasets_frame)
        crack_frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(crack_frame, text="Crack datasets (folders):").pack(side=tk.LEFT)
        self.crack_datasets_var = tk.StringVar()
        ttk.Entry(crack_frame, textvariable=self.crack_datasets_var, width=60).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(crack_frame, text="Add...", command=self._add_crack_dataset).pack(side=tk.LEFT, padx=4)
        self.crack_datasets_listbox = tk.Listbox(crack_frame, height=3)
        self.crack_datasets_listbox.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(crack_frame, text="Remove", command=self._remove_crack_dataset).pack(side=tk.LEFT, padx=4)

        # Feature extraction options
        feature_frame = ttk.LabelFrame(parent, text="Feature Extraction")
        feature_frame.pack(fill=tk.X, padx=8, pady=4)
        self.include_per_frame_var = tk.BooleanVar(value=True)
        self.include_per_point_var = tk.BooleanVar(value=False)
        self.include_global_stats_var = tk.BooleanVar(value=True)
        self.include_advanced_stats_var = tk.BooleanVar(value=False)
        self.include_frequency_domain_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(feature_frame, text="Per-frame features", variable=self.include_per_frame_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(feature_frame, text="Per-point features", variable=self.include_per_point_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(feature_frame, text="Global statistics", variable=self.include_global_stats_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(feature_frame, text="Advanced stats (skew/kurt/autocorr)", variable=self.include_advanced_stats_var).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(feature_frame, text="Frequency domain (FFT)", variable=self.include_frequency_domain_var).pack(side=tk.LEFT, padx=8)

        # Model / Optimization mode (single place; runners dispatch by mode)
        opt_frame = ttk.LabelFrame(parent, text="Model / Optimization mode")
        opt_frame.pack(fill=tk.X, padx=8, pady=4)
        opt_inner = ttk.Frame(opt_frame)
        opt_inner.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(opt_inner, text="Mode:").pack(side=tk.LEFT)
        self.opt_method_var = tk.StringVar(value="dream")
        ttk.Radiobutton(opt_inner, text="DREAM (anomaly)", variable=self.opt_method_var, value="dream").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(opt_inner, text="PatchCore (anomaly)", variable=self.opt_method_var, value="patchcore").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(opt_inner, text="Ensemble (DREAM+PatchCore)", variable=self.opt_method_var, value="ensemble").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(opt_inner, text="Temporal (LSTM/GRU)", variable=self.opt_method_var, value="temporal").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(opt_inner, text="Grid Search (params)", variable=self.opt_method_var, value="grid_search").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(opt_inner, text="Bayesian (params)", variable=self.opt_method_var, value="bayesian").pack(side=tk.LEFT, padx=4)

        # Action buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(btn_frame, text="Prepare Data", command=self._on_prepare_data).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Run (Train or Optimize)", command=self._on_start_ml_or_optimization).pack(side=tk.LEFT, padx=4)

        # Progress and results
        result_frame = ttk.LabelFrame(parent, text="Progress & Results")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.auto_opt_text = tk.Text(result_frame, height=15, wrap=tk.NONE)
        self.auto_opt_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        scrollbar_result = ttk.Scrollbar(result_frame, orient="vertical", command=self.auto_opt_text.yview)
        self.auto_opt_text.configure(yscrollcommand=scrollbar_result.set)

    def _add_normal_dataset(self) -> None:
        """Add normal dataset directory."""
        path = filedialog.askdirectory(
            title="Select normal dataset directory",
            initialdir=str(_project_root()),
        )
        if path:
            self.normal_datasets_listbox.insert(tk.END, path)

    def _remove_normal_dataset(self) -> None:
        """Remove selected normal dataset."""
        selection = self.normal_datasets_listbox.curselection()
        for idx in reversed(selection):
            self.normal_datasets_listbox.delete(idx)

    def _add_crack_dataset(self) -> None:
        """Add crack dataset directory."""
        path = filedialog.askdirectory(
            title="Select crack dataset directory",
            initialdir=str(_project_root()),
        )
        if path:
            self.crack_datasets_listbox.insert(tk.END, path)

    def _remove_crack_dataset(self) -> None:
        """Remove selected crack dataset."""
        selection = self.crack_datasets_listbox.curselection()
        for idx in reversed(selection):
            self.crack_datasets_listbox.delete(idx)

    def _on_prepare_data(self) -> None:
        """Prepare training data from selected datasets."""
        normal_paths = [self.normal_datasets_listbox.get(i) for i in range(self.normal_datasets_listbox.size())]
        crack_paths = [self.crack_datasets_listbox.get(i) for i in range(self.crack_datasets_listbox.size())]

        if not normal_paths and not crack_paths:
            messagebox.showerror("Error", "Please add at least one normal or crack dataset.")
            return

        try:
            from motionanalyzer.auto_optimize import (
                prepare_training_data,
                FeatureExtractionConfig,
                normalize_features,
            )
            from motionanalyzer.crack_model import get_user_params_path, load_params

            self.auto_opt_text.delete("1.0", tk.END)
            self.auto_opt_text.insert(tk.END, "Preparing training data...\n")
            self.auto_opt_text.see(tk.END)
            self.update()

            normal_paths_obj = [Path(p) for p in normal_paths]
            crack_paths_obj = [Path(p) for p in crack_paths]

            # Load user params if available
            try:
                crack_params = load_params(get_user_params_path())
            except (ValueError, FileNotFoundError):
                crack_params = None

            feature_config = FeatureExtractionConfig(
                include_per_frame=self.include_per_frame_var.get(),
                include_per_point=self.include_per_point_var.get(),
                include_global_stats=self.include_global_stats_var.get(),
                include_advanced_stats=self.include_advanced_stats_var.get(),
                include_frequency_domain=self.include_frequency_domain_var.get(),
                include_crack_risk_features=False,  # ML validation: exclude to avoid leakage
            )

            features_df, labels = prepare_training_data(
                normal_datasets=normal_paths_obj,
                crack_datasets=crack_paths_obj,
                crack_params=crack_params,
                feature_config=feature_config,
            )

            # Normalize features (fit on normal-only to avoid leakage)
            normal_mask = labels == 0
            fit_df = features_df.loc[normal_mask]
            features_normalized = normalize_features(features_df, fit_df=fit_df)

            self.auto_opt_text.insert(tk.END, f"Data preparation complete.\n")
            self.auto_opt_text.insert(tk.END, f"Total samples: {len(features_df)}\n")
            self.auto_opt_text.insert(tk.END, f"Normal samples: {np.sum(labels == 0)}\n")
            self.auto_opt_text.insert(tk.END, f"Crack samples: {np.sum(labels == 1)}\n")
            self.auto_opt_text.insert(tk.END, f"Feature columns: {len([c for c in features_df.columns if c not in ['label', 'dataset_path', 'frame', 'index']])}\n")
            self.auto_opt_text.see(tk.END)

            # Store for optimization (and dataset paths for grid/bayesian)
            self.training_features = features_normalized
            self.training_labels = labels
            self.normal_dataset_paths = normal_paths_obj
            self.crack_dataset_paths = crack_paths_obj

            messagebox.showinfo("Success", f"Training data prepared:\n{len(features_df)} samples\n{np.sum(labels == 0)} normal, {np.sum(labels == 1)} crack")
        except Exception as exc:
            tb = traceback.format_exc()
            self.auto_opt_text.insert(tk.END, f"Error:\n{exc}\n\n{tb}\n")
            self.auto_opt_text.see(tk.END)
            messagebox.showerror("Error", f"Data preparation failed:\n{exc}")

    def _on_start_ml_or_optimization(self) -> None:
        """Dispatch to runners by selected mode (dream, patchcore, grid_search, bayesian)."""
        if not hasattr(self, "training_features") or not hasattr(self, "training_labels"):
            messagebox.showerror("Error", "Please prepare training data first.")
            return

        mode = self.opt_method_var.get()
        self.auto_opt_text.insert(tk.END, f"\n--- Mode: {mode} ---\n")
        self.auto_opt_text.see(tk.END)
        self.update()

        def log(msg: str) -> None:
            self.auto_opt_text.insert(tk.END, msg + "\n")
            self.auto_opt_text.see(tk.END)

        def progress() -> None:
            self.update()

        try:
            from motionanalyzer.gui.runners import run_training_or_optimization

            opts = {
                "log_callback": log,
                "progress_callback": progress,
                "epochs": 50,
                "batch_size": 32,
            }
            if mode == "ensemble":
                # Ensemble requires pre-trained DREAM and PatchCore models
                opts["dream_model_path"] = self.dream_model_path_var.get()
                opts["patchcore_model_path"] = self.patchcore_model_path_var.get()
                opts["strategy"] = "weighted_average"  # Can be made configurable
                opts["optimize_weights"] = True
            if mode in ("grid_search", "bayesian") and hasattr(self, "normal_dataset_paths") and hasattr(self, "crack_dataset_paths"):
                opts["normal_dataset_paths"] = getattr(self, "normal_dataset_paths", [])
                opts["crack_dataset_paths"] = getattr(self, "crack_dataset_paths", [])
            if mode == "bayesian":
                opts["n_trials"] = 20
            result = run_training_or_optimization(
                mode,
                self.training_features,
                self.training_labels,
                **opts,
            )
            log(result.get("message", ""))
            if result.get("success"):
                if result.get("model_path"):
                    log(f"Model path: {result['model_path']}")
                messagebox.showinfo("Success", result.get("message", "Done."))
            else:
                messagebox.showwarning("Note", result.get("message", "Not implemented or failed."))
        except Exception as exc:
            tb = traceback.format_exc()
            self.auto_opt_text.insert(tk.END, f"Error:\n{exc}\n\n{tb}\n")
            self.auto_opt_text.see(tk.END)
            messagebox.showerror("Error", f"Run failed:\n{exc}")

    # --------------------------------------------------------------------- #
    # Time Series Analysis tab (Change Point Detection)
    # --------------------------------------------------------------------- #
    def _build_timeseries_tab(self, parent: tk.Widget) -> None:
        """Build Time Series Analysis tab for Change Point Detection."""
        # Input section
        input_frame = ttk.LabelFrame(parent, text="Input")
        input_frame.pack(fill=tk.X, padx=8, pady=4)

        self.timeseries_input_dir_var = tk.StringVar(value=DEFAULT_INPUT_DIR)
        row = ttk.Frame(input_frame)
        row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row, text="Dataset path:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.timeseries_input_dir_var, width=80).pack(
            side=tk.LEFT, padx=4, fill=tk.X, expand=True
        )
        ttk.Button(row, text="Browse...", command=self._browse_timeseries_input_dir).pack(side=tk.LEFT)

        # Detection method selection
        method_frame = ttk.LabelFrame(parent, text="Detection Method")
        method_frame.pack(fill=tk.X, padx=8, pady=4)

        self.cpd_method_var = tk.StringVar(value="cusum")
        method_inner = ttk.Frame(method_frame)
        method_inner.pack(fill=tk.X, padx=8, pady=4)
        ttk.Radiobutton(method_inner, text="CUSUM", variable=self.cpd_method_var, value="cusum").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(method_inner, text="Window-based", variable=self.cpd_method_var, value="window").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(method_inner, text="PELT", variable=self.cpd_method_var, value="pelt").pack(side=tk.LEFT, padx=4)

        # Feature selection (single or multiple)
        feature_frame = ttk.LabelFrame(parent, text="Time Series Feature")
        feature_frame.pack(fill=tk.X, padx=8, pady=4)

        self.cpd_multi_feature_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            feature_frame, text="Use multiple features", variable=self.cpd_multi_feature_var
        ).pack(side=tk.LEFT, padx=8, pady=4)

        self.cpd_feature_var = tk.StringVar(value="acceleration_max")
        feature_inner = ttk.Frame(feature_frame)
        feature_inner.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(feature_inner, text="Feature:").pack(side=tk.LEFT)
        feature_combo = ttk.Combobox(
            feature_inner,
            textvariable=self.cpd_feature_var,
            values=[
                "acceleration_max",
                "acceleration_mean",
                "curvature_concentration",
                "strain_surrogate_max",
                "impact_surrogate_max",
            ],
            state="readonly",
            width=30,
        )
        feature_combo.pack(side=tk.LEFT, padx=4)

        # Multi-feature selection (checkboxes)
        self.cpd_multi_features_frame = ttk.Frame(feature_frame)
        self.cpd_multi_features_frame.pack(fill=tk.X, padx=8, pady=4)
        self.cpd_feature_vars: dict[str, tk.BooleanVar] = {}
        available_features = [
            "acceleration_max",
            "acceleration_mean",
            "curvature_concentration",
            "strain_surrogate_max",
            "impact_surrogate_max",
        ]
        for feat in available_features:
            var = tk.BooleanVar(value=(feat == "acceleration_max"))
            self.cpd_feature_vars[feat] = var
            ttk.Checkbutton(
                self.cpd_multi_features_frame, text=feat, variable=var
            ).pack(side=tk.LEFT, padx=4)

        # Parameters (collapsible)
        param_frame = ttk.LabelFrame(parent, text="Parameters")
        param_frame.pack(fill=tk.X, padx=8, pady=4)

        # CUSUM parameters
        cusum_params = ttk.Frame(param_frame)
        cusum_params.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(cusum_params, text="CUSUM threshold:").pack(side=tk.LEFT)
        self.cusum_threshold_var = tk.StringVar(value="2.0")
        ttk.Entry(cusum_params, textvariable=self.cusum_threshold_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(cusum_params, text="min_size:").pack(side=tk.LEFT, padx=(8, 0))
        self.cusum_min_size_var = tk.StringVar(value="5")
        ttk.Entry(cusum_params, textvariable=self.cusum_min_size_var, width=10).pack(side=tk.LEFT, padx=4)

        # Window-based parameters
        window_params = ttk.Frame(param_frame)
        window_params.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(window_params, text="Window size:").pack(side=tk.LEFT)
        self.window_size_var = tk.StringVar(value="10")
        ttk.Entry(window_params, textvariable=self.window_size_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(window_params, text="threshold_ratio:").pack(side=tk.LEFT, padx=(8, 0))
        self.window_threshold_var = tk.StringVar(value="1.5")
        ttk.Entry(window_params, textvariable=self.window_threshold_var, width=10).pack(side=tk.LEFT, padx=4)

        # PELT parameters
        pelt_params = ttk.Frame(param_frame)
        pelt_params.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(pelt_params, text="PELT penalty:").pack(side=tk.LEFT)
        self.pelt_pen_var = tk.StringVar(value="3.0")
        ttk.Entry(pelt_params, textvariable=self.pelt_pen_var, width=10).pack(side=tk.LEFT, padx=4)
        ttk.Label(pelt_params, text="min_size:").pack(side=tk.LEFT, padx=(8, 0))
        self.pelt_min_size_var = tk.StringVar(value="5")
        ttk.Entry(pelt_params, textvariable=self.pelt_min_size_var, width=10).pack(side=tk.LEFT, padx=4)

        # Auto-tuning options
        tuning_frame = ttk.LabelFrame(parent, text="Parameter Auto-Tuning")
        tuning_frame.pack(fill=tk.X, padx=8, pady=4)

        self.cpd_auto_tune_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            tuning_frame, text="Enable auto-tuning", variable=self.cpd_auto_tune_var
        ).pack(side=tk.LEFT, padx=8, pady=4)

        ttk.Label(tuning_frame, text="Method:").pack(side=tk.LEFT, padx=(16, 4))
        self.cpd_tuning_method_var = tk.StringVar(value="grid")
        ttk.Radiobutton(tuning_frame, text="Grid Search", variable=self.cpd_tuning_method_var, value="grid").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(tuning_frame, text="Bayesian", variable=self.cpd_tuning_method_var, value="bayesian").pack(side=tk.LEFT, padx=4)

        ttk.Label(tuning_frame, text="Trials:").pack(side=tk.LEFT, padx=(16, 4))
        self.cpd_n_trials_var = tk.StringVar(value="20")
        ttk.Entry(tuning_frame, textvariable=self.cpd_n_trials_var, width=8).pack(side=tk.LEFT, padx=4)

        # Ensemble options
        ensemble_frame = ttk.LabelFrame(parent, text="Ensemble Detection")
        ensemble_frame.pack(fill=tk.X, padx=8, pady=4)

        self.cpd_ensemble_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            ensemble_frame, text="Use ensemble (multiple methods)", variable=self.cpd_ensemble_var
        ).pack(side=tk.LEFT, padx=8, pady=4)

        ttk.Label(ensemble_frame, text="Combine strategy:").pack(side=tk.LEFT, padx=(16, 4))
        self.cpd_combine_strategy_var = tk.StringVar(value="union")
        ttk.Radiobutton(ensemble_frame, text="Union", variable=self.cpd_combine_strategy_var, value="union").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(ensemble_frame, text="Intersection", variable=self.cpd_combine_strategy_var, value="intersection").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(ensemble_frame, text="Majority", variable=self.cpd_combine_strategy_var, value="majority").pack(side=tk.LEFT, padx=4)

        # Run button
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(btn_frame, text="Detect Change Points", command=self._on_run_changepoint_detection).pack(side=tk.LEFT)

        # Results display
        result_frame = ttk.LabelFrame(parent, text="Results")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.cpd_result_text = tk.Text(result_frame, height=8, wrap=tk.NONE)
        self.cpd_result_text.pack(fill=tk.BOTH, expand=True)

        # Visualization
        self.cpd_plot_frame = ttk.LabelFrame(parent, text="Change Point Visualization")
        self.cpd_plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self._cpd_plot_canvas: Any = None
        self._cpd_plot_toolbar: Any = None
        self._cpd_plot_fig: Any = None

    def _browse_timeseries_input_dir(self) -> None:
        path = filedialog.askdirectory(
            title="Select dataset directory",
            initialdir=str(_project_root()),
        )
        if path:
            self.timeseries_input_dir_var.set(path)

    def _on_run_changepoint_detection(self) -> None:
        """Run change point detection on selected dataset with optional auto-tuning and multi-feature support."""
        input_dir = _resolve_path(self.timeseries_input_dir_var.get())
        if not input_dir.exists():
            messagebox.showerror("Error", f"Input path not found:\n{input_dir}")
            return

        method = self.cpd_method_var.get()
        use_multi_feature = self.cpd_multi_feature_var.get()
        use_auto_tune = self.cpd_auto_tune_var.get()
        use_ensemble = self.cpd_ensemble_var.get()

        try:
            self.cpd_result_text.delete("1.0", tk.END)
            self.cpd_result_text.insert(tk.END, f"Running change point detection...\n")
            self.cpd_result_text.insert(tk.END, f"  Method: {method}\n")
            self.cpd_result_text.see(tk.END)
            self.update()

            # Run analysis to get vectors
            output_dir = input_dir.parent / f"{input_dir.name}_cpd_analysis"
            run_analysis(input_dir=input_dir, output_dir=output_dir, fps=30.0)

            vectors_csv = output_dir / "vectors.csv"
            if not vectors_csv.exists():
                raise FileNotFoundError(f"Vectors file not found: {vectors_csv}")

            vectors = pd.read_csv(vectors_csv)
            frame_metrics_csv = input_dir / "frame_metrics.csv"

            # Build features DataFrame
            features_list: list[dict[str, float]] = []
            frame_features_df = None

            if use_multi_feature:
                # Multi-feature mode: collect all selected features
                selected_features = [feat for feat, var in self.cpd_feature_vars.items() if var.get()]
                if not selected_features:
                    messagebox.showerror("Error", "Please select at least one feature for multi-feature detection.")
                    return

                self.cpd_result_text.insert(tk.END, f"  Features: {', '.join(selected_features)}\n")
                self.cpd_result_text.see(tk.END)
                self.update()

                # Build features DataFrame
                feature_dict: dict[str, list[float]] = {"frame": []}
                for feat_name in selected_features:
                    feature_dict[feat_name] = []

                # Get frame range
                frames = sorted(vectors["frame"].unique())
                feature_dict["frame"] = frames

                for feat_name in selected_features:
                    if feat_name == "curvature_concentration" and frame_metrics_csv.exists():
                        fm = pd.read_csv(frame_metrics_csv)
                        if "curvature_concentration" in fm.columns:
                            feature_dict[feat_name] = fm["curvature_concentration"].tolist()
                        else:
                            raise ValueError(f"Feature '{feat_name}' not found in frame_metrics.csv")
                    else:
                        if feat_name.startswith("acceleration_"):
                            base = "acceleration"
                            agg = feat_name.split("_")[1]
                        elif feat_name.startswith("strain_"):
                            base = "strain_surrogate"
                            agg = feat_name.split("_")[-1]
                        elif feat_name.startswith("impact_"):
                            base = "impact_surrogate"
                            agg = feat_name.split("_")[-1]
                        else:
                            base = feat_name
                            agg = "mean"

                        if base not in vectors.columns:
                            raise ValueError(f"Feature '{base}' not found in vectors.csv")
                        frame_features = vectors.groupby("frame")[base].agg(agg).reset_index()
                        feature_dict[feat_name] = frame_features[base].tolist()

                frame_features_df = pd.DataFrame(feature_dict)
            else:
                # Single feature mode
                feature_name = self.cpd_feature_var.get()
                self.cpd_result_text.insert(tk.END, f"  Feature: {feature_name}\n")
                self.cpd_result_text.see(tk.END)
                self.update()

                if feature_name == "curvature_concentration" and frame_metrics_csv.exists():
                    fm = pd.read_csv(frame_metrics_csv)
                    if "curvature_concentration" in fm.columns:
                        signal = fm["curvature_concentration"].values
                        frames = fm.index.values if "frame" not in fm.columns else fm["frame"].values
                    else:
                        raise ValueError(f"Feature '{feature_name}' not found in frame_metrics.csv")
                else:
                    if feature_name.startswith("acceleration_"):
                        base = "acceleration"
                        agg = feature_name.split("_")[1]
                    elif feature_name.startswith("strain_"):
                        base = "strain_surrogate"
                        agg = feature_name.split("_")[-1]
                    elif feature_name.startswith("impact_"):
                        base = "impact_surrogate"
                        agg = feature_name.split("_")[-1]
                    else:
                        base = feature_name
                        agg = "mean"

                    if base not in vectors.columns:
                        raise ValueError(f"Feature '{base}' not found in vectors.csv")
                    frame_features = vectors.groupby("frame")[base].agg(agg).reset_index()
                    signal = frame_features[base].values
                    frames = frame_features["frame"].values

                frame_features_df = pd.DataFrame({"frame": frames, feature_name: signal})

            # Run detection
            if use_ensemble:
                # Ensemble detection with multiple methods
                from motionanalyzer.time_series.changepoint_optimizer import ensemble_change_point_detection

                methods = ["cusum", "window"]  # Can add "pelt" if available
                combine_strategy = self.cpd_combine_strategy_var.get()
                selected_feature_names = (
                    [feat for feat, var in self.cpd_feature_vars.items() if var.get()]
                    if use_multi_feature
                    else [self.cpd_feature_var.get()]
                )

                result = ensemble_change_point_detection(
                    frame_features_df,
                    selected_feature_names,
                    methods=methods,
                    combine_strategy=combine_strategy,
                )
            elif use_multi_feature:
                # Multi-feature detection
                from motionanalyzer.time_series.changepoint_optimizer import detect_change_points_multi_feature

                selected_feature_names = [feat for feat, var in self.cpd_feature_vars.items() if var.get()]
                combine_strategy = self.cpd_combine_strategy_var.get()

                method_kwargs = {}
                if method == "cusum":
                    method_kwargs["threshold"] = float(self.cusum_threshold_var.get())
                    method_kwargs["min_size"] = int(self.cusum_min_size_var.get())
                elif method == "window":
                    method_kwargs["window_size"] = int(self.window_size_var.get())
                    method_kwargs["threshold_ratio"] = float(self.window_threshold_var.get())
                    method_kwargs["min_size"] = int(self.cusum_min_size_var.get())

                result = detect_change_points_multi_feature(
                    frame_features_df,
                    selected_feature_names,
                    method=method,
                    combine_strategy=combine_strategy,
                    **method_kwargs,
                )
            else:
                # Single feature detection
                signal = frame_features_df.iloc[:, 1].values  # Second column is the feature

                if use_auto_tune:
                    # Auto-tuning mode
                    from motionanalyzer.time_series.changepoint_optimizer import (
                        optimize_cusum_parameters,
                        optimize_window_parameters,
                    )

                    tuning_method = self.cpd_tuning_method_var.get()
                    n_trials = int(self.cpd_n_trials_var.get())

                    self.cpd_result_text.insert(tk.END, f"  Auto-tuning ({tuning_method})...\n")
                    self.cpd_result_text.see(tk.END)
                    self.update()

                    # Expected change range for crack detection (frame 30-45 is typical)
                    expected_range = (30, 45)

                    if method == "cusum":
                        opt_result = optimize_cusum_parameters(
                            signal,
                            expected_change_range=expected_range,
                            n_trials=n_trials,
                            optimization_method=tuning_method,
                        )
                        threshold = opt_result.best_params["threshold"]
                        sensitivity = opt_result.best_params.get("sensitivity")
                        from motionanalyzer.time_series.changepoint import CUSUMDetector

                        detector = CUSUMDetector(threshold=threshold, sensitivity=sensitivity)
                        result = detector.detect(signal)
                        result.method = f"{result.method} (auto-tuned, score={opt_result.best_score:.3f})"
                    elif method == "window":
                        opt_result = optimize_window_parameters(
                            signal,
                            expected_change_range=expected_range,
                            n_trials=n_trials,
                            optimization_method=tuning_method,
                        )
                        window_size = opt_result.best_params["window_size"]
                        threshold_ratio = opt_result.best_params["threshold_ratio"]
                        from motionanalyzer.time_series.changepoint import WindowBasedDetector

                        detector = WindowBasedDetector(window_size=window_size, threshold_ratio=threshold_ratio)
                        result = detector.detect(signal)
                        result.method = f"{result.method} (auto-tuned, score={opt_result.best_score:.3f})"
                    else:
                        # PELT doesn't support auto-tuning yet
                        pen = float(self.pelt_pen_var.get())
                        min_size = int(self.pelt_min_size_var.get())
                        from motionanalyzer.time_series.changepoint import detect_change_points_pelt

                        result = detect_change_points_pelt(signal, min_size=min_size, pen=pen)
                else:
                    # Manual parameters
                    if method == "cusum":
                        threshold = float(self.cusum_threshold_var.get())
                        min_size = int(self.cusum_min_size_var.get())
                        from motionanalyzer.time_series.changepoint import CUSUMDetector

                        detector = CUSUMDetector(threshold=threshold, min_size=min_size)
                        result = detector.detect(signal)
                    elif method == "window":
                        window_size = int(self.window_size_var.get())
                        threshold_ratio = float(self.window_threshold_var.get())
                        min_size = int(self.cusum_min_size_var.get())
                        from motionanalyzer.time_series.changepoint import WindowBasedDetector

                        detector = WindowBasedDetector(window_size=window_size, threshold_ratio=threshold_ratio, min_size=min_size)
                        result = detector.detect(signal)
                    else:  # pelt
                        pen = float(self.pelt_pen_var.get())
                        min_size = int(self.pelt_min_size_var.get())
                        from motionanalyzer.time_series.changepoint import detect_change_points_pelt

                        result = detect_change_points_pelt(signal, min_size=min_size, pen=pen)

            # Display results
            self.cpd_result_text.delete("1.0", tk.END)
            self.cpd_result_text.insert(tk.END, f"Change Point Detection Results\n")
            self.cpd_result_text.insert(tk.END, f"{'=' * 60}\n")
            self.cpd_result_text.insert(tk.END, f"Method: {result.method}\n")
            if use_multi_feature:
                self.cpd_result_text.insert(tk.END, f"Features: {', '.join(selected_feature_names)}\n")
            else:
                self.cpd_result_text.insert(tk.END, f"Feature: {self.cpd_feature_var.get()}\n")
            self.cpd_result_text.insert(tk.END, f"Change points detected: {len(result.change_points)}\n")
            if result.change_points:
                self.cpd_result_text.insert(tk.END, f"Frames: {result.change_points}\n")
            else:
                self.cpd_result_text.insert(tk.END, "No change points detected.\n")
            self.cpd_result_text.insert(tk.END, f"\nNote: For crack detection, change points around frame 30-45 are expected.\n")
            self.cpd_result_text.see(tk.END)

            # Visualize (use first feature for visualization)
            if use_multi_feature:
                viz_feature = selected_feature_names[0]
                viz_signal = frame_features_df[viz_feature].values
            else:
                viz_feature = self.cpd_feature_var.get()
                viz_signal = frame_features_df.iloc[:, 1].values

            self._show_changepoint_plot(viz_signal, result, output_dir / "changepoint_detection.png")

            messagebox.showinfo("Success", f"Change point detection complete.\n\nDetected {len(result.change_points)} change point(s).")
        except ImportError as e:
            if "ruptures" in str(e):
                messagebox.showerror(
                    "Error",
                    "PELT detection requires 'ruptures' library.\n"
                    "Install with: pip install ruptures\n"
                    "Or use CUSUM or Window-based method instead."
                )
            else:
                messagebox.showerror("Error", f"Import error:\n{e}")
        except Exception as exc:
            tb = traceback.format_exc()
            self.cpd_result_text.insert(tk.END, f"\nERROR:\n{exc}\n{tb}\n")
            self.cpd_result_text.see(tk.END)
            messagebox.showerror("Error", f"Change point detection failed:\n{exc}")

    def _clear_changepoint_plot(self) -> None:
        for w in self.cpd_plot_frame.winfo_children():
            w.destroy()
        if self._cpd_plot_fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._cpd_plot_fig)
            self._cpd_plot_fig = None
        self._cpd_plot_canvas = None
        self._cpd_plot_toolbar = None

    def _show_changepoint_plot(self, signal: np.ndarray, result: ChangePointResult, out_path: Path) -> None:
        """Visualize change points on time series signal."""
        self._clear_changepoint_plot()
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

            fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
            frames = np.arange(len(signal))
            ax.plot(frames, signal, lw=1.5, label="Signal", color="blue")
            for cp in result.change_points:
                ax.axvline(x=cp, color="red", linestyle="--", alpha=0.7, label="Change point" if cp == result.change_points[0] else "")
            ax.set_title(f"Change Point Detection ({result.method})")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Feature Value")
            ax.legend()
            ax.grid(True, alpha=0.3)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=160)

            self._cpd_plot_fig = fig
            canvas = FigureCanvasTkAgg(fig, master=self.cpd_plot_frame)
            canvas.draw()
            toolbar = NavigationToolbar2Tk(canvas, self.cpd_plot_frame)
            toolbar.update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._cpd_plot_canvas = canvas
            self._cpd_plot_toolbar = toolbar
        except Exception as exc:
            self.cpd_result_text.insert(tk.END, f"\nPlot display failed: {exc}\n")
            self.cpd_result_text.see(tk.END)

    def _build_synthetic_goals_tab(self, parent: tk.Widget) -> None:
        """Build Synthetic Data & Goals tab: generate ML dataset, run goal evaluations."""
        root = _project_root()
        base_dir = root / "data" / "synthetic" / "ml_dataset"
        reports_dir = root / "reports"

        # Synthetic data generation
        synth_frame = ttk.LabelFrame(parent, text="Synthetic Data Generation")
        synth_frame.pack(fill=tk.X, padx=8, pady=4)
        row1 = ttk.Frame(synth_frame)
        row1.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row1, text="Output: data/synthetic/ml_dataset/").pack(side=tk.LEFT)
        row2 = ttk.Frame(synth_frame)
        row2.pack(fill=tk.X, padx=8, pady=4)
        self.synthetic_small_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Small set (115)", variable=self.synthetic_small_var).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="Generate ML Dataset", command=self._on_generate_synthetic).pack(side=tk.LEFT, padx=8)

        # Goal evaluation
        goals_frame = ttk.LabelFrame(parent, text="Goal Evaluation")
        goals_frame.pack(fill=tk.X, padx=8, pady=4)
        row3 = ttk.Frame(goals_frame)
        row3.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(row3, text="Run Goal 1 (CPD)", command=self._on_run_goal1_cpd).pack(side=tk.LEFT, padx=4)
        ttk.Button(row3, text="Run Goal 1 (ML)", command=self._on_run_goal1_ml).pack(side=tk.LEFT, padx=4)
        ttk.Button(row3, text="Run Goal 2 (ML)", command=self._on_run_goal2_ml).pack(side=tk.LEFT, padx=4)
        ttk.Button(row3, text="Summary", command=self._on_run_goals_summary).pack(side=tk.LEFT, padx=4)
        row4 = ttk.Frame(goals_frame)
        row4.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row4, text="Reports:").pack(side=tk.LEFT)
        self.reports_dir_var = tk.StringVar(value=str(reports_dir))
        ttk.Entry(row4, textvariable=self.reports_dir_var, width=50).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        # Results
        result_frame = ttk.LabelFrame(parent, text="Output")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.synthetic_goals_text = tk.Text(result_frame, height=12, wrap=tk.NONE)
        self.synthetic_goals_text.pack(fill=tk.BOTH, expand=True)

    def _append_synthetic_log(self, msg: str) -> None:
        self.synthetic_goals_text.insert(tk.END, msg + "\n")
        self.synthetic_goals_text.see(tk.END)
        self.update()

    def _on_generate_synthetic(self) -> None:
        """Run generate_ml_dataset.py from GUI."""
        small = self.synthetic_small_var.get()
        script = _project_root() / "scripts" / "generate_ml_dataset.py"
        if not script.exists():
            messagebox.showerror("Error", f"Script not found: {script}")
            return
        cmd = [sys.executable, str(script)]
        if small:
            cmd.append("--small")
        self.synthetic_goals_text.delete("1.0", tk.END)
        self._append_synthetic_log(f"Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_project_root()), timeout=600)
            self._append_synthetic_log(result.stdout or "")
            if result.stderr:
                self._append_synthetic_log(result.stderr)
            if result.returncode == 0:
                base_dir = _project_root() / "data" / "synthetic" / "ml_dataset"
                messagebox.showinfo("Done", f"ML dataset generated at:\n{base_dir}")
            else:
                messagebox.showerror("Error", "Synthetic data generation failed. Check output.")
        except subprocess.TimeoutExpired:
            messagebox.showerror("Error", "Generation timed out (10 min).")
        except Exception as exc:
            self._append_synthetic_log(str(exc))
            messagebox.showerror("Error", str(exc))

    def _on_run_goal1_cpd(self) -> None:
        self._run_script("evaluate_goal1_cpd", "Goal 1 CPD")

    def _on_run_goal1_ml(self) -> None:
        self._run_script("evaluate_goal1_ml", "Goal 1 ML")

    def _on_run_goal2_ml(self) -> None:
        self._run_script("evaluate_goal2_ml", "Goal 2 ML")

    def _on_run_goals_summary(self) -> None:
        self._run_script("evaluate_goals_summary", "Goals Summary")

    def _run_script(self, name: str, label: str) -> None:
        """Run evaluation script by name."""
        script = _project_root() / "scripts" / f"{name}.py"
        if not script.exists():
            messagebox.showerror("Error", f"Script not found: {script}")
            return
        self.synthetic_goals_text.delete("1.0", tk.END)
        self._append_synthetic_log(f"Running {label}...")
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                cwd=str(_project_root()),
                timeout=600,
            )
            self._append_synthetic_log(result.stdout or "")
            if result.stderr:
                self._append_synthetic_log(result.stderr)
            if result.returncode == 0:
                self._append_synthetic_log("Done.")
                summary_path = _project_root() / "reports" / "goal_achievement_summary.md"
                if summary_path.exists():
                    self._append_synthetic_log("\n--- Summary ---\n")
                    self._append_synthetic_log(summary_path.read_text(encoding="utf-8"))
            else:
                messagebox.showerror("Error", f"{label} failed. Check output.")
        except subprocess.TimeoutExpired:
            messagebox.showerror("Error", f"{label} timed out.")
        except Exception as exc:
            self._append_synthetic_log(str(exc))
            messagebox.showerror("Error", str(exc))


def main() -> None:
    app = MotionAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
