from __future__ import annotations

import tkinter as tk
import traceback
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from motionanalyzer.analysis import (
    AnalysisSummary,
    compare_summaries,
    load_summary,
    run_analysis,
)

DEFAULT_INPUT_DIR = "data/synthetic/normal_case"
DEFAULT_OUTPUT_DIR = "exports/vectors/normal_case"
DEFAULT_BASE_SUMMARY = "exports/vectors/normal_case/summary.json"
DEFAULT_CANDIDATE_SUMMARY = "exports/vectors/crack_case/summary.json"


class MotionAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("motionanalyzer - FPCB bending analysis (offline Windows GUI)")
        # Let user resize freely; start in a reasonable size.
        self.geometry("1280x800")

        self._build_widgets()

    # --------------------------------------------------------------------- #
    # UI construction
    # --------------------------------------------------------------------- #
    def _build_widgets(self) -> None:
        notebook = ttk.Notebook(self)
        frame_analyze = ttk.Frame(notebook)
        frame_compare = ttk.Frame(notebook)
        notebook.add(frame_analyze, text="Analyze")
        notebook.add(frame_compare, text="Compare")
        notebook.pack(fill=tk.BOTH, expand=True)

        self._build_analyze_tab(frame_analyze)
        self._build_compare_tab(frame_compare)

    def _build_analyze_tab(self, parent: tk.Widget) -> None:
        input_frame = ttk.LabelFrame(parent, text="Input / Output")
        input_frame.pack(fill=tk.X, padx=8, pady=4)

        self.input_dir_var = tk.StringVar(value=DEFAULT_INPUT_DIR)
        self.output_dir_var = tk.StringVar(value=DEFAULT_OUTPUT_DIR)

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
        self.summary_text = tk.Text(summary_frame, height=10, wrap=tk.NONE)
        self.summary_text.pack(fill=tk.BOTH, expand=True)

        # Log area
        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.NONE)
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
        path = filedialog.askdirectory(title="Select input bundle directory")
        if path:
            self.input_dir_var.set(path)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select output directory")
        if path:
            self.output_dir_var.set(path)

    def _browse_base_summary(self) -> None:
        path = filedialog.askopenfilename(
            title="Select base summary.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.base_summary_var.set(path)

    def _browse_cand_summary(self) -> None:
        path = filedialog.askopenfilename(
            title="Select candidate summary.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.cand_summary_var.set(path)

    # --------------------------------------------------------------------- #
    # Analyze tab handlers
    # --------------------------------------------------------------------- #
    def _append_log(self, msg: str) -> None:
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

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
        input_dir = Path(self.input_dir_var.get()).expanduser()
        output_dir = Path(self.output_dir_var.get()).expanduser()

        # Basic validations to avoid FileNotFoundError from analysis.load_bundle
        if not input_dir.exists():
            messagebox.showerror("Error", f"Input path not found:\n{input_dir}")
            return

        fps_file = input_dir / "fps.txt"
        if not fps_file.exists():
            messagebox.showerror(
                "Error",
                f"fps.txt not found in:\n{input_dir}\n\n"
                "Please make sure the bundle contains fps.txt with a single fps value.",
            )
            return

        frame_files = list(input_dir.glob("frame_*.txt"))
        if not frame_files:
            messagebox.showerror(
                "Error",
                f"No frame_*.txt files found in:\n{input_dir}\n\n"
                "Expected frame_0001.txt style files for each frame.",
            )
            return

        try:
            self._append_log(f"Running analysis...\n  input={input_dir}\n  output={output_dir}")
            summary = run_analysis(input_dir=input_dir, output_dir=output_dir)
            self._append_log("Analysis complete.")
            self._render_summary(summary)
            messagebox.showinfo(
                "Success",
                f"Analysis complete.\n\nOutput directory:\n{output_dir}",
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
        base_path = Path(self.base_summary_var.get()).expanduser()
        cand_path = Path(self.cand_summary_var.get()).expanduser()

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


def main() -> None:
    app = MotionAnalyzerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
