from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

ScenarioName = Literal[
    "normal", "crack", "pre_damage", "thick_panel", "uv_overcured",
    "light_distortion",  # Normal + illumination-induced edge distortion
    "micro_crack",      # Subtle crack, harder to detect
]


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
    """Pixels per millimeter; 1 px = 1/pixels_per_mm mm. Used with meters_per_pixel for SI."""
    meters_per_pixel: float | None = None
    """Length scale in m/px for SI units. If None, derived as (1e-3 / pixels_per_mm)."""
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
    # Enhanced physics parameters for crack scenario
    shockwave_amplitude: float = 0.0
    """Shockwave amplitude when crack occurs (acceleration spike multiplier)."""
    shockwave_decay_rate: float = 0.0
    """Shockwave decay rate (frames^-1)."""
    vibration_frequency_hz: float = 0.0
    """Micro-vibration frequency after crack (Hz)."""
    vibration_damping: float = 0.0
    """Vibration damping coefficient."""
    vibration_duration_frames: int = 0
    """Vibration duration after crack (frames)."""


def _scenario_params(name: ScenarioName) -> ScenarioParams:
    table: dict[ScenarioName, ScenarioParams] = {
        "normal": ScenarioParams(
            final_angle_ratio=1.0,
            response_alpha=0.12,  # gentler follow → bounded d²θ/dt², realistic acceleration
            damping=0.96,
            crack_gain=0.0,
            crack_center_ratio=0.65,
            crack_width_ratio=0.05,
            uv_delay_ratio=0.0,
            uv_snap_gain=0.0,
            pre_damage_skew=0.0,
            # No shockwave/vibration for normal scenario
            shockwave_amplitude=0.0,
            shockwave_decay_rate=0.0,
            vibration_frequency_hz=0.0,
            vibration_damping=0.0,
            vibration_duration_frames=0,
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
            # Enhanced physics: shockwave and vibration when crack occurs
            shockwave_amplitude=3.5,  # Acceleration spike multiplier
            shockwave_decay_rate=0.15,  # Decay per frame
            vibration_frequency_hz=25.0,  # High-frequency vibration (~25 Hz)
            vibration_damping=0.92,  # Damping per cycle
            vibration_duration_frames=15,  # ~0.5 s at 30 fps
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
            # Mild shockwave/vibration for pre-damage
            shockwave_amplitude=1.2,
            shockwave_decay_rate=0.2,
            vibration_frequency_hz=15.0,
            vibration_damping=0.94,
            vibration_duration_frames=10,
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
            # No crack, so no shockwave/vibration
            shockwave_amplitude=0.0,
            shockwave_decay_rate=0.0,
            vibration_frequency_hz=0.0,
            vibration_damping=0.0,
            vibration_duration_frames=0,
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
            # Snap-like behavior may cause mild shockwave
            shockwave_amplitude=1.5,
            shockwave_decay_rate=0.18,
            vibration_frequency_hz=20.0,
            vibration_damping=0.93,
            vibration_duration_frames=12,
        ),
        # Light distortion: same physics as normal, extra noise applied in bundle
        "light_distortion": ScenarioParams(
            final_angle_ratio=1.0,
            response_alpha=0.12,
            damping=0.96,
            crack_gain=0.0,
            crack_center_ratio=0.65,
            crack_width_ratio=0.05,
            uv_delay_ratio=0.0,
            uv_snap_gain=0.0,
            pre_damage_skew=0.0,
            shockwave_amplitude=0.0,
            shockwave_decay_rate=0.0,
            vibration_frequency_hz=0.0,
            vibration_damping=0.0,
            vibration_duration_frames=0,
        ),
        # Micro crack: subtle, hard-to-detect crack
        "micro_crack": ScenarioParams(
            final_angle_ratio=0.97,
            response_alpha=0.18,
            damping=0.96,
            crack_gain=5.0,  # Lower than crack (16)
            crack_center_ratio=0.70,
            crack_width_ratio=0.015,  # Narrower
            uv_delay_ratio=0.0,
            uv_snap_gain=0.0,
            pre_damage_skew=0.02,
            shockwave_amplitude=1.2,  # Lower than crack (3.5)
            shockwave_decay_rate=0.25,
            vibration_frequency_hz=12.0,  # Lower
            vibration_damping=0.95,
            vibration_duration_frames=8,
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


def _smooth_drive_temporal(drive: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Smooth drive in time to bound d²(drive)/dt² and avoid unrealistic accelerations.
    Uses a causal-friendly moving average (symmetric kernel for interior).
    """
    n = len(drive)
    if n <= kernel_size or kernel_size < 2:
        return drive
    k = kernel_size
    half = k // 2
    out = np.empty_like(drive)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = float(np.mean(drive[lo:hi]))
    return out


def _curvature_weight(points: int, params: ScenarioParams) -> np.ndarray:
    s = np.linspace(0.0, 1.0, points)
    center = params.crack_center_ratio
    width = max(1e-6, params.crack_width_ratio)
    crack_peak = np.exp(-0.5 * ((s - center) / width) ** 2)
    skew = 1.0 + params.pre_damage_skew * (2.0 * s - 1.0)
    weights = np.maximum(0.2, skew + params.crack_gain * crack_peak)
    normalized = weights / np.mean(weights)
    return np.asarray(normalized, dtype=float)


def _compute_shockwave_effect(
    frame_idx: int,
    crack_frame: int,
    amplitude: float,
    decay_rate: float,
) -> float:
    """
    Compute shockwave effect at frame_idx when crack occurs at crack_frame.
    
    Models energy release as exponential decay from crack frame.
    
    Args:
        frame_idx: Current frame index
        crack_frame: Frame where crack occurs
        amplitude: Shockwave amplitude (acceleration multiplier)
        decay_rate: Decay rate per frame (0-1)
    
    Returns:
        Shockwave multiplier (1.0 = no effect, >1.0 = acceleration spike)
    """
    if frame_idx < crack_frame or amplitude <= 0.0:
        return 1.0
    frames_since_crack = frame_idx - crack_frame
    return 1.0 + amplitude * np.exp(-decay_rate * frames_since_crack)


def _compute_vibration_effect(
    frame_idx: int,
    crack_frame: int,
    frequency_hz: float,
    damping: float,
    duration_frames: int,
    fps: float,
) -> float:
    """
    Compute micro-vibration effect after crack occurrence.
    
    Models structural instability as damped oscillation.
    
    Args:
        frame_idx: Current frame index
        crack_frame: Frame where crack occurs
        frequency_hz: Vibration frequency (Hz)
        damping: Damping coefficient per cycle (0-1)
        duration_frames: Vibration duration (frames)
        fps: Frames per second
    
    Returns:
        Vibration displacement multiplier
    """
    if frame_idx < crack_frame or frequency_hz <= 0.0:
        return 0.0
    
    frames_since_crack = frame_idx - crack_frame
    if frames_since_crack > duration_frames:
        return 0.0
    
    # Time in seconds
    t = frames_since_crack / fps
    # Angular frequency
    omega = 2.0 * np.pi * frequency_hz
    # Damped oscillation: amplitude * exp(-damping*t) * sin(omega*t)
    # Damping per cycle: exp(-damping) per period
    cycles = t * frequency_hz
    amplitude = np.exp(-damping * cycles)
    vibration = amplitude * np.sin(omega * t)
    
    # Scale to reasonable displacement (pixels)
    return vibration * 0.5  # Small displacement (~0.5 px)


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


# Real FPCB bending process: start-to-finish typically ~2 s (side-view).
BEND_DURATION_S = 2.0


def high_fidelity_fpcb_config(
    frames: int | None = None,
    points_per_frame: int = 280,
    fps: float = 30.0,
    seed: int = 42,
    bend_duration_s: float = BEND_DURATION_S,
) -> SyntheticConfig:
    """
    Physics-based high-fidelity FPCB bending config aligned with real process.

    - Bend duration ~2 s (real process); frames = round(bend_duration_s * fps) when frames is None.
    - Smooth side-view progression: straight -> arc -> U-like; low noise.
    - SI scale (meters_per_pixel) for m/s, m/s².
    """
    if frames is None:
        frames = max(2, int(round(bend_duration_s * fps)))
    return SyntheticConfig(
        frames=frames,
        points_per_frame=points_per_frame,
        fps=fps,
        width=1920,
        height=1080,
        panel_length_px=280.0,
        panel_thickness_um=90.0,
        pixels_per_mm=10.0,
        meters_per_pixel=1e-4,
        noise_std=0.2,  # lower noise → less acceleration spike from i.i.d. per-frame noise
        seed=seed,
        scenario="normal",
    )


def _meters_per_pixel(config: SyntheticConfig) -> float:
    """SI length scale: meters per pixel (1 px = this many m)."""
    if config.meters_per_pixel is not None and config.meters_per_pixel > 0:
        return config.meters_per_pixel
    # Derive from pixels_per_mm: 1 mm = 1e-3 m, 1 px = 1/pixels_per_mm mm
    return 1e-3 / max(config.pixels_per_mm, 1e-9)


def generate_synthetic_bundle(
    output_dir: Path,
    config: SyntheticConfig,
    *,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    """
    Generate side-view FPCB bending sequence from straight line to U-like profile.

    extra_metadata: Optional dict merged into metadata.json (e.g. goal, label, tags for ML).

    Physical assumptions:
    - centerline strain approximation: epsilon ~= t / (2R)
    - total bending angle increases by process drive + damping response
    - local stiffness damage redistributes curvature (crack/over-cure effects)
    - SI scale: meters_per_pixel (m/px) stored in metadata for velocity m/s and acceleration m/s².
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    params = _scenario_params(config.scenario)
    m_per_px = _meters_per_pixel(config)

    origin = np.array(
        [config.width * 0.42, config.height * 0.52],
        dtype=float,
    )
    weights = _curvature_weight(config.points_per_frame, params)
    final_angle = np.pi * params.final_angle_ratio
    theta = 0.0

    # Precompute and smooth drive so d²(theta)/dt² is bounded → realistic accelerations
    raw_drive = np.array(
        [_drive_signal(frame_idx / max(1, config.frames - 1), params) for frame_idx in range(config.frames)]
    )
    drive_smoothed = _smooth_drive_temporal(raw_drive, kernel_size=5)

    # Determine crack frame (when crack_center_ratio is reached in normalized time)
    crack_frame = int(params.crack_center_ratio * (config.frames - 1)) if params.crack_gain > 0 else -1

    metrics_path = output_dir / "frame_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["frame", "time_s", "bend_angle_deg", "curvature_concentration", "est_max_strain"]
        )

        for frame_idx in range(config.frames):
            drive = drive_smoothed[frame_idx]
            target_theta = final_angle * drive

            # Apply shockwave effect to response_alpha (acceleration spike when crack occurs)
            shockwave_mult = _compute_shockwave_effect(
                frame_idx,
                crack_frame,
                params.shockwave_amplitude,
                params.shockwave_decay_rate,
            )
            effective_response_alpha = params.response_alpha * shockwave_mult

            theta += effective_response_alpha * (target_theta - theta)
            theta = max(0.0, min(final_angle * 1.05, theta))

            centerline = _build_shape(
                points=config.points_per_frame,
                length_px=config.panel_length_px,
                theta_total=theta,
                weights=weights,
            )

            # Base translation
            translation = np.array(
                [
                    origin[0] + 2.0 * np.sin(2.0 * np.pi * (frame_idx / config.fps) * 0.2),
                    origin[1],
                ],
                dtype=float,
            )

            # Add micro-vibration after crack
            vibration_y = _compute_vibration_effect(
                frame_idx,
                crack_frame,
                params.vibration_frequency_hz,
                params.vibration_damping,
                params.vibration_duration_frames,
                config.fps,
            )
            translation[1] += vibration_y

            pts = centerline + translation
            pts += rng.normal(loc=0.0, scale=config.noise_std, size=pts.shape)

            # Light distortion: illumination-induced edge detection drift (Phase 1.2 다양화)
            if config.scenario == "light_distortion":
                # Per-frame random offset: ±1~5 px (varied by seed for diversity)
                offset_scale = 1.0 + (rng.integers(0, 5) / 2.0)  # 1.0, 1.5, 2.0, 2.5, 3.0
                frame_offset = rng.uniform(-offset_scale * 2.0, offset_scale * 2.0, size=2)
                pts += frame_offset
                # Extra point-wise jitter: scale 0.8~2.0 (diverse glare)
                jitter_scale = 0.8 + rng.uniform(0, 1.2)
                pts += rng.normal(loc=0.0, scale=jitter_scale, size=pts.shape)
                # Random spikes: 1~5% of points (ghost edges from reflection)
                spike_ratio = 0.01 + rng.uniform(0, 0.04)
                n_spike = max(1, int(spike_ratio * len(pts)))
                spike_idx = rng.choice(len(pts), size=min(n_spike, len(pts)), replace=False)
                pts[spike_idx] += rng.uniform(-2.5, 2.5, size=(len(spike_idx), 2))

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

    config_dict = asdict(config)
    if config_dict.get("meters_per_pixel") is None:
        config_dict["meters_per_pixel"] = m_per_px
    total_duration_s = (config.frames - 1) / config.fps if config.frames > 1 else 0.0
    metadata = {
        "config": config_dict,
        "meters_per_pixel": m_per_px,
        "bend_duration_s": total_duration_s,
        "total_frames": config.frames,
        "fps": config.fps,
        "units": "SI (position m, velocity m/s, acceleration m/s² when scale used)",
        "process_note": "Bending timeline aligned with real FPCB side-view process (~2 s typical).",
        "assumptions": [
            "epsilon ~= t/(2R) strain approximation for bending.",
            "Crack/pre-damage modeled as local stiffness loss causing curvature concentration.",
            "Thick panel modeled as increased effective stiffness and lower final bend angle.",
            "UV over-cure modeled as delayed response and late-stage snap tendency.",
            "Drive signal smoothed in time to bound d²θ/dt²; lower response_alpha and noise for realistic acceleration (m/s²).",
        ],
        "scenario_params": asdict(params),
        "outputs": {
            "frames_glob": "frame_*.txt",
            "fps_file": "fps.txt",
            "frame_metrics_file": "frame_metrics.csv",
        },
    }
    if extra_metadata:
        metadata.update(extra_metadata)
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
    elif scenario == "light_distortion":
        # Same physics as normal; allow slightly looser curvature due to noise
        checks.extend(
            [
                (bend[-1] >= 155.0, f"light_distortion final bend too small: {bend[-1]:.1f} deg"),
                (conc[-1] < 5.0, f"light_distortion curvature concentration too high: {conc[-1]:.2f}"),
            ]
        )
    elif scenario == "micro_crack":
        # Weaker crack: elevated concentration but milder than full crack
        checks.extend(
            [
                (4.0 < conc[-1] < 8.0, f"micro_crack concentration out of range: {conc[-1]:.2f}"),
            ]
        )

    errors = [message for ok, message in checks if not ok]
    return len(errors) == 0, errors
