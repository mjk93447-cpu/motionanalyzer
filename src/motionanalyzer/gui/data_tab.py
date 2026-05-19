"""Data tab: inbox scan, preflight batch, ingest/manifest hooks."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from motionanalyzer.preflight import PreflightConfig, preflight_realdata_bundle


LogFn = Callable[[str], None]


def scan_bundles_under(root: Path) -> list[dict[str, Any]]:
    """Find directories containing frame_*.txt (bundle candidates)."""
    root = Path(root)
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    for frame_file in sorted(root.rglob("frame_*.txt")):
        bundle = frame_file.parent
        key = str(bundle.resolve())
        if any(e["path"] == key for e in found):
            continue
        n_frames = len(list(bundle.glob("frame_*.txt")))
        found.append({"path": key, "name": bundle.name, "parent": bundle.parent.name, "n_frames": n_frames})
    return sorted(found, key=lambda x: x["path"])


class DataTabController:
    """Builds Data tab widgets; callbacks wired from MotionAnalyzerApp."""

    def __init__(
        self,
        parent: Any,
        *,
        log_fn: LogFn,
        on_open_in_analyze: Callable[[str], None],
        project_root: Path,
        on_open_patchcore_refine: Callable[[str | None], None] | None = None,
    ) -> None:
        self.log = log_fn
        self.on_open_in_analyze = on_open_in_analyze
        self.on_open_patchcore_refine = on_open_patchcore_refine
        self.project_root = project_root
        self.inbox_var: Any = None
        self.tree: Any = None
        self._build(parent)

    def _build(self, parent: Any) -> None:
        path_frame = ttk.LabelFrame(parent, text="Raw / bundle inbox")
        path_frame.pack(fill=tk.X, padx=8, pady=4)
        row = ttk.Frame(path_frame)
        row.pack(fill=tk.X, padx=8, pady=4)
        self.inbox_var = __import__("tkinter").StringVar(value=str(self.project_root / "data" / "raw"))
        ttk.Label(row, text="Inbox root:").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.inbox_var, width=70).pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        ttk.Button(row, text="Browse...", command=self._browse_inbox).pack(side=tk.LEFT)
        ttk.Button(row, text="Scan bundles", command=self._scan).pack(side=tk.LEFT, padx=4)

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(btn_row, text="Preflight selected", command=self._preflight_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Open in Analyze", command=self._open_analyze).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Run ingest script...", command=self._run_ingest).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="Build manifest...", command=self._run_build_manifest).pack(side=tk.LEFT, padx=4)
        if self.on_open_patchcore_refine is not None:
            ttk.Button(btn_row, text="PatchCore refine (ML tab)", command=self._patchcore_refine).pack(
                side=tk.LEFT, padx=4
            )

        tree_frame = ttk.LabelFrame(parent, text="Bundles")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        cols = ("parent", "name", "frames", "path")
        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (120, 140, 60, 400)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w)
        self.tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def _browse_inbox(self) -> None:
        path = filedialog.askdirectory(initialdir=str(self.project_root))
        if path:
            self.inbox_var.set(path)

    def _scan(self) -> None:
        root = Path(self.inbox_var.get())
        for item in self.tree.get_children():
            self.tree.delete(item)
        bundles = scan_bundles_under(root)
        for b in bundles:
            self.tree.insert("", "end", values=(b["parent"], b["name"], b["n_frames"], b["path"]))
        self.log(f"Scan complete: {len(bundles)} bundle(s) under {root}")

    def _selected_paths(self) -> list[str]:
        out: list[str] = []
        for iid in self.tree.selection():
            vals = self.tree.item(iid, "values")
            if len(vals) >= 4:
                out.append(str(vals[3]))
        return out

    def _preflight_selected(self) -> None:
        paths = self._selected_paths()
        if not paths:
            messagebox.showwarning("Preflight", "Select one or more bundles in the table.")
            return
        cfg = PreflightConfig()
        for p in paths:
            summary, errors = preflight_realdata_bundle(Path(p), cfg)
            status = "PASS" if summary.passed else "FAIL"
            self.log(f"[{status}] {p} frames={summary.frame_count} errors={len(errors)}")
            if errors:
                for e in errors[:5]:
                    self.log(f"  - {e}")

    def _open_analyze(self) -> None:
        paths = self._selected_paths()
        if not paths:
            messagebox.showwarning("Analyze", "Select a bundle row first.")
            return
        self.on_open_in_analyze(paths[0])

    def _run_ingest(self) -> None:
        script = self.project_root / "scripts" / "ingest_edge_points.py"
        if not script.exists():
            messagebox.showerror("Ingest", f"Script not found:\n{script}")
            return
        subprocess.Popen([sys.executable, str(script), "--help"], cwd=str(self.project_root))

    def _patchcore_refine(self) -> None:
        if self.on_open_patchcore_refine is None:
            return
        out = self.project_root / "data" / "real" / f"ml_{Path(self.inbox_var.get()).name}"
        mf = out / "manifest.json"
        if mf.exists():
            self.on_open_patchcore_refine(str(mf))
        else:
            messagebox.showinfo(
                "PatchCore refine",
                f"Build manifest first (expected):\n{mf}\n\nOr use ML tab → Load manifest normals.",
            )

    def _run_build_manifest(self) -> None:
        script = self.project_root / "scripts" / "build_manifest_from_bundles.py"
        if not script.exists():
            messagebox.showerror("Manifest", f"Script not found:\n{script}")
            return
        root = Path(self.inbox_var.get())
        out = self.project_root / "data" / "real" / f"ml_{root.name}"
        subprocess.Popen(
            [sys.executable, str(script), "--root", str(root), "--output", str(out)],
            cwd=str(self.project_root),
        )
        self.log(f"Building manifest → {out / 'manifest.json'}")
