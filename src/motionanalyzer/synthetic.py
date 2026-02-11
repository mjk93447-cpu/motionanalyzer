from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

ScenarioName = Literal["normal", "crack", "pre_damage", "thick_panel", "uv_overcured"]


@dataclass(frozen=True)
class SyntheticConfig:
    frames: int = 120
    points_per_frame: int = 230
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    panel_length_px: float = 230.0
    panel_thickness_um: float = 90.0
    pixels_per_mm: float = 10.0
    noise_std: float = 0.8
    seed: int = 42
    scenario: ScenarioName = "normal"


@dataclass(frozen=True)
class ScenarioParams:
    final_angle_ratio: float
    response_alpha: float
    damping: float
    crack_gain: float
    crack_center_ratio: float
    crack_width_ratio: float
    uv_delay_ratio: float
    uv_snap_gain: float
    pre_damage_skew: float


def _scenario_params(name: ScenarioName) -> ScenarioParams:
    table: dict[ScenarioName, ScenarioParams] = {
        "normal": ScenarioParams(
            final_angle_ratio=1.0,
            response_alpha=0.2,
            damping=0.96,
            crack_gain=0.0,
            crack_center_ratio=0.65,
            crack_width_ratio=0.05,
            uv_delay_ratio=0.0,
            uv_snap_gain=0.0,
            pre_damage_skew=0.0,
        ),
        "crack": ScenarioParams(
            final_angle_ratio=0.95,
            response_alpha=0.2,
            damping=0.95,
            crack_gain=16.0,
            crack_center_ratio=0.72,
            crack_width_ratio=0.009,
            uv_delay_ratio=0.0,
            uv_snap_gain=0.0,
            pre_damage_skew=0.05,
        ),
        "pre_damage": ScenarioParams(
            final_angle_ratio=0.9,
            response_alpha=0.16,
            damping=0.96,
            crack_gain=1.9,
            crack_center_ratio=0.68,
            crack_width_ratio=0.03,
            uv_delay_ratio=0.0,
            uv_snap_gain=0.0,
            pre_damage_skew=0.08,
        ),
        "thick_panel": ScenarioParams(
            final_angle_ratio=0.72,
            response_alpha=0.1,
            damping=0.975,
            crack_gain=0.0,
            crack_center_ratio=0.65,
            crack_width_ratio=0.05,
            uv_delay_ratio=0.0,
            uv_snap_gain=0.0,
            pre_damage_skew=0.0,
        ),
        "uv_overcured": ScenarioParams(
            final_angle_ratio=0.86,
            response_alpha=0.14,
            damping=0.985,
            crack_gain=0.7,
            crack_center_ratio=0.62,
            crack_width_ratio=0.04,
            uv_delay_ratio=0.32,
            uv_snap_gain=0.22,
            pre_damage_skew=0.03,
        ),
    }
    return table[name]


def _smoothstep(u: float) -> float:
    x = min(1.0, max(0.0, u))
    return x * x * (3.0 - 2.0 * x)


def _drive_signal(u: float, params: ScenarioParams) -> float:
    if params.uv_delay_ratio > 0.0:
        shifted = (u - params.uv_delay_ratio) / max(1e-9, 1.0 - params.uv_delay_ratio)
        delayed = _smoothstep(shifted)
        snap = float(params.uv_snap_gain * np.exp(-(((u - 0.72) / 0.09) ** 2)))
        return min(1.12, max(0.0, delayed + snap))
    return _smoothstep(u)


def _curvature_weight(points: int, params: ScenarioParams) -> np.ndarray:
    s = np.linspace(0.0, 1.0, points)
    center = params.crack_center_ratio
    width = max(1e-6, params.crack_width_ratio)
    crack_peak = np.exp(-0.5 * ((s - center) / width) ** 2)
    skew = 1.0 + params.pre_damage_skew * (2.0 * s - 1.0)
    weights = np.maximum(0.2, skew + params.crack_gain * crack_peak)
    normalized = weights / np.mean(weights)
    return np.asarray(normalized, dtype=float)


def _build_shape(
    points: int,
    length_px: float,
    theta_total: float,
    weights: np.ndarray,
) -> np.ndarray:
    ds = length_px / max(1, points - 1)
    kappa0 = theta_total / max(length_px, 1e-6)
    kappa = kappa0 * weights
    theta = np.cumsum(kappa) * ds
    x = np.cumsum(np.cos(theta)) * ds
    y = np.cumsum(np.sin(theta)) * ds
    pts = np.stack((x, y), axis=1)
    pts -= pts[0]
    return pts


def _frame_metrics(points: np.ndarray) -> tuple[float, float]:
    tangents = np.diff(points, axis=0)
    start_angle = float(np.arctan2(tangents[0, 1], tangents[0, 0]))
    end_angle = float(np.arctan2(tangents[-1, 1], tangents[-1, 0]))
    bend_angle_deg = float(np.degrees(np.abs(end_angle - start_angle)))
    bend_angle_deg = min(bend_angle_deg, 360.0 - bend_angle_deg)

    second = np.diff(points, n=2, axis=0)
    curvature_mag = np.linalg.norm(second, axis=1)
    concentration = float(np.max(curvature_mag) / (np.mean(curvature_mag) + 1e-6))
    return bend_angle_deg, concentration


def _panel_thickness_px(config: SyntheticConfig) -> float:
    thickness_mm = config.panel_thickness_um / 1000.0
    return thickness_mm * config.pixels_per_mm


def generate_synthetic_bundle(output_dir: Path, config: SyntheticConfig) -> Path:
    """
    Generate side-view FPCB bending sequence from straight line to U-like profile.

    Physical assumptions:
    - centerline strain approximation: epsilon ~= t / (2R)
    - total bending angle increases by process drive + damping response
    - local stiffness damage redistributes curvature (crack/over-cure effects)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    params = _scenario_params(config.scenario)

    origin = np.array(
        [config.width * 0.42, config.height * 0.52],
        dtype=float,
    )
    weights = _curvature_weight(config.points_per_frame, params)
    final_angle = np.pi * params.final_angle_ratio
    theta = 0.0

    metrics_path = output_dir / "frame_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["frame", "time_s", "bend_angle_deg", "curvature_concentration", "est_max_strain"]
        )

        for frame_idx in range(config.frames):
            u = frame_idx / max(1, config.frames - 1)
            drive = _drive_signal(u, params)
            target_theta = final_angle * drive

            theta += params.response_alpha * (target_theta - theta)
            theta = max(0.0, min(final_angle * 1.05, theta))

            centerline = _build_shape(
                points=config.points_per_frame,
                length_px=config.panel_length_px,
                theta_total=theta,
                weights=weights,
            )

            translation = np.array(
                [
                    origin[0] + 2.0 * np.sin(2.0 * np.pi * (frame_idx / config.fps) * 0.2),
                    origin[1],
                ],
                dtype=float,
            )
            pts = centerline + translation
            pts += rng.normal(loc=0.0, scale=config.noise_std, size=pts.shape)

            pts[:, 0] = np.clip(pts[:, 0], 0, config.width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, config.height - 1)

            frame_path = output_dir / f"frame_{frame_idx:05d}.txt"
            lines = ["# x,y,index"]
            for point_idx, (x, y) in enumerate(pts, start=1):
                lines.append(f"{int(round(x))},{int(round(y))},{point_idx}")
            frame_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            bend_angle_deg = float(np.degrees(theta))
            concentration = float(np.max(weights))
            local_kappa_peak = (theta / max(config.panel_length_px, 1e-6)) * float(np.max(weights))
            est_max_strain = 0.5 * _panel_thickness_px(config) * local_kappa_peak
            writer.writerow(
                [
                    frame_idx,
                    round(frame_idx / config.fps, 5),
                    round(bend_angle_deg, 4),
                    round(concentration, 4),
                    round(float(est_max_strain), 6),
                ]
            )

    fps_path = output_dir / "fps.txt"
    fps_path.write_text(f"{config.fps}\n", encoding="utf-8")

    metadata = {
        "config": asdict(config),
        "assumptions": [
            "epsilon ~= t/(2R) strain approximation for bending.",
            "Crack/pre-damage modeled as local stiffness loss causing curvature concentration.",
            "Thick panel modeled as increased effective stiffness and lower final bend angle.",
            "UV over-cure modeled as delayed response and late-stage snap tendency.",
        ],
        "scenario_params": asdict(params),
        "outputs": {
            "frames_glob": "frame_*.txt",
            "fps_file": "fps.txt",
            "frame_metrics_file": "frame_metrics.csv",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return output_dir


def validate_synthetic_bundle(output_dir: Path, scenario: ScenarioName) -> tuple[bool, list[str]]:
    metrics_file = output_dir / "frame_metrics.csv"
    if not metrics_file.exists():
        return False, ["frame_metrics.csv not found"]

    rows: list[dict[str, float]] = []
    with metrics_file.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            rows.append(
                {
                    "bend_angle_deg": float(row["bend_angle_deg"]),
                    "curvature_concentration": float(row["curvature_concentration"]),
                    "est_max_strain": float(row["est_max_strain"]),
                }
            )

    if len(rows) < 5:
        return False, ["insufficient frames for validation"]

    bend = np.array([r["bend_angle_deg"] for r in rows], dtype=float)
    conc = np.array([r["curvature_concentration"] for r in rows], dtype=float)
    strain = np.array([r["est_max_strain"] for r in rows], dtype=float)
    d2_bend = np.diff(bend, n=2)
    monotonic_ratio = float(np.mean(np.diff(bend) >= -0.3))

    checks: list[tuple[bool, str]] = [
        (bend[-1] >= 120.0, f"final bend angle too low: {bend[-1]:.1f} deg"),
        (
            monotonic_ratio >= 0.9,
            f"bend progression unstable: monotonic_ratio={monotonic_ratio:.2f}",
        ),
    ]

    if scenario == "normal":
        checks.extend(
            [
                (bend[-1] >= 165.0, f"normal scenario final bend too small: {bend[-1]:.1f} deg"),
                (
                    conc[-1] < 4.0,
                    f"normal scenario curvature concentration too high: {conc[-1]:.2f}",
                ),
            ]
        )
    elif scenario == "crack":
        checks.extend(
            [
                (conc[-1] > 6.0, f"crack signature weak: concentration={conc[-1]:.2f}"),
            ]
        )
    elif scenario == "pre_damage":
        checks.extend(
            [
                (2.0 <= conc[-1] <= 5.0, f"pre_damage concentration out of range: {conc[-1]:.2f}"),
            ]
        )
    elif scenario == "thick_panel":
        checks.extend(
            [
                (bend[-1] < 160.0, f"thick_panel bent too much: {bend[-1]:.1f} deg"),
                (
                    np.max(strain) < 0.03,
                    f"thick_panel strain unexpectedly high: {np.max(strain):.4f}",
                ),
            ]
        )
    elif scenario == "uv_overcured":
        checks.extend(
            [
                (np.max(np.abs(d2_bend)) > 0.35, "uv_overcured snap signature not observed"),
            ]
        )

    errors = [message for ok, message in checks if not ok]
    return len(errors) == 0, errors
