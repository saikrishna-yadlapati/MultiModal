"""Visualizations and graphic generators for the Speech Flaw Grounding Dashboard.

Provides rich visual representations of multimodal speech analysis:
- High-resolution, calibrated spectrogram heatmap with flaw interval bounding overlays.
- Interactive multi-layer Altair charts with synced tooltips, flaw regions, and baseline overlays.
- 4-axis contrastive acoustic radar profile (SVG) comparing Candidate vs Speaker Baseline.
- Acoustic delta gauges comparing Candidate measurements against Speaker baseline normals.
- Word-level transcript alignment ribbon with flaw-overlap indicators.
"""
from __future__ import annotations

import io
import math
from typing import Any, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import scipy.signal as signal

# Standard semantic flaw color tokens
FLAW_COLORS = {
    "long_pause": "#e4572e",       # Crimson/Coral
    "rate_deviation": "#f3a712",   # Amber/Gold
    "monotone": "#4c6ef5",         # Indigo/Blue
    "filler_word": "#20c997",      # Emerald/Teal
}

FLAW_RGB = {
    "long_pause": (228, 87, 46),
    "rate_deviation": (243, 167, 18),
    "monotone": (76, 110, 245),
    "filler_word": (32, 201, 151),
}


def generate_spectrogram_image(
    y: np.ndarray,
    sr: int,
    duration_s: float,
    evidence_df: Optional[pd.DataFrame] = None,
    max_freq: int = 4000,
    width: int = 950,
    height: int = 240,
) -> bytes:
    """Generate a high-contrast publication-quality spectrogram image with flaw interval overlays.

    Uses scipy.signal.spectrogram + 5-step perceptually graded twilight-to-gold palette,
    overlaid with detected flaw boundary bands.
    """
    if len(y) == 0:
        y = np.zeros(int(sr * 0.5), dtype=np.float64)

    nperseg = min(512, len(y))
    noverlap = min(384, nperseg // 2)
    f, t, sxx = signal.spectrogram(y, fs=sr, nperseg=nperseg, noverlap=noverlap)

    # Filter to speech frequency range (0 to max_freq Hz)
    m = f <= max_freq
    f = f[m]
    sxx = sxx[m, :]

    # Logarithmic scaling (dB)
    sxx_db = 10.0 * np.log10(np.maximum(sxx, 1e-10))
    vmin, vmax = float(sxx_db.min()), float(sxx_db.max())
    norm = (sxx_db - vmin) / (vmax - vmin + 1e-8)

    # 5-step twilight-to-gold colormap
    cmap = np.zeros((256, 3), dtype=np.uint8)
    pts = [
        (0.00, (15, 12, 41)),     # Deep dark twilight
        (0.25, (48, 43, 99)),     # Rich indigo
        (0.50, (140, 50, 110)),   # Magenta
        (0.75, (235, 100, 60)),   # Coral-orange
        (1.00, (255, 230, 100)),  # Warm yellow
    ]
    for i in range(len(pts) - 1):
        x0, c0 = pts[i]
        x1, c1 = pts[i + 1]
        idx0, idx1 = int(x0 * 255), int(x1 * 255)
        for ch in range(3):
            cmap[idx0 : idx1 + 1, ch] = np.linspace(c0[ch], c1[ch], idx1 - idx0 + 1)

    img_arr = cmap[(norm * 255).astype(np.uint8)]
    # Invert vertical axis so 0 Hz is at bottom
    img_arr = np.flipud(img_arr)
    base_img = Image.fromarray(img_arr).resize((width, height), Image.Resampling.BILINEAR).convert("RGBA")

    # Overlay flaw interval highlights
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if evidence_df is not None and not evidence_df.empty and duration_s > 0:
        for _, row in evidence_df.iterrows():
            flaw_type = str(row.get("flaw_type", "long_pause"))
            c = FLAW_RGB.get(flaw_type, (200, 200, 200))
            s_s = float(row.get("start_s", 0.0))
            e_s = float(row.get("end_s", 0.0))

            x0 = max(0, min(width - 1, int((s_s / duration_s) * width)))
            x1 = max(0, min(width, int((e_s / duration_s) * width)))
            if x1 <= x0:
                x1 = min(width, x0 + 6)

            # Shaded vertical band
            draw.rectangle([x0, 0, x1, height], fill=(c[0], c[1], c[2], 75))
            # Border lines
            draw.line([(x0, 0), (x0, height)], fill=(c[0], c[1], c[2], 240), width=2)
            draw.line([(x1, 0), (x1, height)], fill=(c[0], c[1], c[2], 240), width=2)

            # Text label
            label = flaw_type.replace("_", " ").upper()
            draw.rectangle([x0 + 2, 4, min(width, x0 + 6 + len(label) * 7), 18], fill=(20, 20, 25, 200))
            draw.text((x0 + 4, 5), label, fill=(255, 255, 255, 255))

    # Add frequency scale ticks along the left border
    freq_ticks = [1000, 2000, 3000]
    for ft in freq_ticks:
        y_pos = int(height - (ft / max_freq) * height)
        draw.line([(0, y_pos), (12, y_pos)], fill=(255, 255, 255, 180), width=1)
        draw.text((14, y_pos - 6), f"{ft}Hz", fill=(255, 255, 255, 180))

    combined = Image.alpha_composite(base_img, overlay).convert("RGB")
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return buf.getvalue()


def generate_radar_svg(scores: dict[str, float], width: int = 340, height: int = 280) -> str:
    """Generate a clean, responsive SVG radar / spider chart comparing Candidate vs Speaker Baseline.

    scores: dict with keys e.g.
      {'Pause Naturalness': 0.85, 'Pitch Dynamism': 0.40, 'Pacing Stability': 0.90, 'Fluency & Flow': 1.0}
    Values are 0.0 to 1.0. Baseline is 1.0 on all axes (dashed polygon).
    """
    cx, cy = width / 2, height / 2 + 10
    radius = min(width, height) / 2 - 45
    labels = list(scores.keys())
    n = len(labels)
    if n < 3:
        return ""

    def get_pt(r: float, idx: int) -> Tuple[float, float]:
        angle = -math.pi / 2 + (2 * math.pi * idx / n)
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    svg = [
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="max-width: {width}px; margin: auto; display: block;">'
    ]

    # Concentric reference grid
    for level in [0.25, 0.50, 0.75, 1.00]:
        r = radius * level
        pts = [get_pt(r, i) for i in range(n)]
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dash = 'stroke-dasharray="3,3"' if level < 1.0 else ""
        stroke = "#dee2e6" if level < 1.0 else "#ced4da"
        svg.append(f'<polygon points="{pts_str}" fill="none" stroke="{stroke}" stroke-width="1" {dash}/>')

    # Baseline polygon (100% ideal reference)
    base_pts = [get_pt(radius, i) for i in range(n)]
    base_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in base_pts)
    svg.append(
        f'<polygon points="{base_str}" fill="rgba(108, 117, 125, 0.05)" '
        f'stroke="#6c757d" stroke-width="1.5" stroke-dasharray="4,4"/>'
    )

    # Candidate polygon
    cand_pts = [get_pt(radius * max(0.12, min(1.0, scores[labels[i]])), i) for i in range(n)]
    cand_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in cand_pts)
    svg.append(
        f'<polygon points="{cand_str}" fill="rgba(76, 110, 245, 0.22)" '
        f'stroke="#4c6ef5" stroke-width="2.5"/>'
    )

    # Vertices, dots & text labels
    for i, (x, y) in enumerate(cand_pts):
        val = int(max(0.0, min(1.0, scores[labels[i]])) * 100)
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#4c6ef5" stroke="#ffffff" stroke-width="1.5"/>')

        lx, ly = get_pt(radius + 24, i)
        anchor = "middle" if abs(lx - cx) < 20 else ("start" if lx > cx else "end")
        val_color = "#2b8a3e" if val >= 80 else ("#e67700" if val >= 55 else "#c92a2a")
        svg.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'font-size="11" font-family="system-ui, -apple-system, sans-serif" font-weight="600" fill="#343a40">'
            f'{labels[i]} <tspan fill="{val_color}" font-weight="700">{val}%</tspan></text>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def compute_radar_metrics(evidence: list[Any], res: dict[str, Any]) -> dict[str, float]:
    """Calculate 4 contrastive delivery dimensions from actual measurements.

    Grounded in acoustic data:
    1. Pause Naturalness: Drops on anomalous unprompted pauses.
    2. Pitch Dynamism: Drops if monotone or pitch variation is suppressed.
    3. Pacing Stability: Drops if rate deviations / rushing occur.
    4. Fluency & Flow: Drops if filler words / hesitation gaps occur.
    """
    n_pause = sum(1 for e in evidence if getattr(e, "flaw_type", "") == "long_pause")
    n_mono = sum(1 for e in evidence if getattr(e, "flaw_type", "") == "monotone")
    n_rate = sum(1 for e in evidence if getattr(e, "flaw_type", "") == "rate_deviation")
    n_filler = sum(1 for e in evidence if getattr(e, "flaw_type", "") == "filler_word")

    pause_score = max(0.20, 1.0 - n_pause * 0.35)

    # For pitch dynamism, compare candidate f0 std vs baseline reference if available
    pitch_score = 1.0
    if n_mono > 0:
        cand_f0 = res.get("cand_f0")
        base_f0_std = res.get("reference", {}).get("f0_std_hz", 0.0)
        if cand_f0 is not None and not cand_f0.empty and base_f0_std > 0:
            c_std = float(cand_f0["f0_hz"].std())
            pitch_score = max(0.20, min(1.0, c_std / base_f0_std))
        else:
            pitch_score = max(0.25, 1.0 - n_mono * 0.40)

    rate_score = max(0.20, 1.0 - n_rate * 0.35)
    fluency_score = max(0.20, 1.0 - n_filler * 0.35)

    return {
        "Pause Naturalness": pause_score,
        "Pitch Dynamism": pitch_score,
        "Pacing Stability": rate_score,
        "Fluency & Flow": fluency_score,
    }


def enhanced_timeline_chart(
    rms_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    duration_s: float,
    gt: Optional[Tuple[float, float]] = None,
    base_rms_df: Optional[pd.DataFrame] = None,
) -> alt.Chart:
    """Altair timeline with rich tooltips, zoom/pan interaction, and ground truth."""
    x = alt.X(
        "time_s:Q",
        title="Time (seconds)",
        scale=alt.Scale(domain=[0, duration_s], nice=False),
        axis=alt.Axis(grid=True, gridColor="#f1f3f5"),
    )
    layers = []

    if base_rms_df is not None and not base_rms_df.empty:
        base_line = (
            alt.Chart(base_rms_df)
            .mark_line(color="#adb5bd", strokeDash=[4, 4], strokeWidth=1.5)
            .encode(
                x=x,
                y=alt.Y("rms_norm:Q", title="Loudness (0–1)", scale=alt.Scale(domain=[0, 1.05])),
                tooltip=[
                    alt.Tooltip("time_s:Q", title="Time (s)", format=".2f"),
                    alt.Tooltip("rms_norm:Q", title="Baseline Loudness", format=".2f"),
                ],
            )
        )
        layers.append(base_line)

    cand_line = (
        alt.Chart(rms_df)
        .mark_line(color="#212529", strokeWidth=2.2)
        .encode(
            x=x,
            y=alt.Y("rms_norm:Q", title="Loudness (0–1)", scale=alt.Scale(domain=[0, 1.05])),
            tooltip=[
                alt.Tooltip("time_s:Q", title="Time (s)", format=".2f"),
                alt.Tooltip("rms_norm:Q", title="Candidate Loudness", format=".2f"),
            ],
        )
    )
    layers.append(cand_line)

    if not evidence_df.empty:
        tooltips = [
            alt.Tooltip("flaw_type:N", title="Detected Flaw"),
            alt.Tooltip("start_s:Q", title="Start Time (s)", format=".2f"),
            alt.Tooltip("end_s:Q", title="End Time (s)", format=".2f"),
            alt.Tooltip("feature:N", title="Acoustic Feature"),
        ]
        if "confidence" in evidence_df.columns:
            tooltips.append(alt.Tooltip("confidence:Q", title="Confidence", format=".0%"))

        bands = (
            alt.Chart(evidence_df)
            .mark_rect(opacity=0.32)
            .encode(
                x="start_s:Q",
                x2="end_s:Q",
                color=alt.Color(
                    "flaw_type:N",
                    title="Detected Flaw",
                    scale=alt.Scale(domain=list(FLAW_COLORS), range=list(FLAW_COLORS.values())),
                ),
                tooltip=tooltips,
            )
        )
        layers.insert(0, bands)

    if gt is not None:
        gt_df = (
            alt.Chart(pd.DataFrame({"s": [gt[0]], "e": [gt[1]]}))
            .mark_rect(opacity=0.0, stroke="#000000", strokeWidth=2.5, strokeDash=[6, 3])
            .encode(x="s:Q", x2="e:Q", tooltip=[alt.Tooltip("s:Q", title="GT Start"), alt.Tooltip("e:Q", title="GT End")])
        )
        layers.append(gt_df)

    return (
        alt.layer(*layers)
        .properties(height=240)
        .interactive(bind_y=False)
    )


def enhanced_pitch_chart(
    cand_f0: pd.DataFrame,
    duration_s: float,
    base_f0: Optional[pd.DataFrame] = None,
    evidence_df: Optional[pd.DataFrame] = None,
) -> Optional[alt.Chart]:
    """A/B pitch contour chart with tooltips, mean line, baseline reference corridor, and monotone shading."""
    x = alt.X(
        "time_s:Q",
        title="Time (seconds)",
        scale=alt.Scale(domain=[0, duration_s], nice=False),
        axis=alt.Axis(grid=True, gridColor="#f1f3f5"),
    )
    layers = []

    # Monotone flaw bands highlight
    if evidence_df is not None and not evidence_df.empty:
        mono_df = evidence_df[evidence_df["flaw_type"] == "monotone"]
        if not mono_df.empty:
            mono_bands = (
                alt.Chart(mono_df)
                .mark_rect(opacity=0.25, color="#4c6ef5")
                .encode(
                    x="start_s:Q",
                    x2="end_s:Q",
                    tooltip=[
                        alt.Tooltip("flaw_type:N", title="Flaw"),
                        alt.Tooltip("start_s:Q", title="Start (s)", format=".2f"),
                        alt.Tooltip("end_s:Q", title="End (s)", format=".2f"),
                        alt.Tooltip("candidate_value:Q", title="Candidate Std (Hz)", format=".1f"),
                    ],
                )
            )
            layers.append(mono_bands)

    if base_f0 is not None and not base_f0.empty:
        base_line = (
            alt.Chart(base_f0)
            .mark_line(color="#adb5bd", strokeDash=[4, 4], strokeWidth=1.5)
            .encode(
                x=x,
                y=alt.Y("f0_hz:Q", title="Pitch F0 (Hz)"),
                tooltip=[
                    alt.Tooltip("time_s:Q", title="Time (s)", format=".2f"),
                    alt.Tooltip("f0_hz:Q", title="Baseline Pitch (Hz)", format=".1f"),
                ],
            )
        )
        layers.append(base_line)

    if cand_f0 is not None and not cand_f0.empty:
        cand_line = (
            alt.Chart(cand_f0)
            .mark_line(color="#4c6ef5", strokeWidth=2.2)
            .encode(
                x=x,
                y=alt.Y("f0_hz:Q", title="Pitch F0 (Hz)"),
                tooltip=[
                    alt.Tooltip("time_s:Q", title="Time (s)", format=".2f"),
                    alt.Tooltip("f0_hz:Q", title="Candidate Pitch (Hz)", format=".1f"),
                ],
            )
        )
        layers.append(cand_line)

        # Average pitch horizontal guide rule
        avg_pitch = float(cand_f0["f0_hz"].mean())
        mean_rule = (
            alt.Chart(pd.DataFrame({"y": [avg_pitch]}))
            .mark_rule(color="#748ffc", strokeDash=[2, 2], strokeWidth=1.2)
            .encode(y="y:Q", tooltip=[alt.Tooltip("y:Q", title="Mean Pitch (Hz)", format=".1f")])
        )
        layers.append(mean_rule)

    if not layers:
        return None

    return (
        alt.layer(*layers)
        .properties(height=210)
        .interactive(bind_y=False)
    )


def render_acoustic_delta_bar(
    feature: str,
    cand_val: float,
    base_val: float,
    flaw_type: str,
) -> str:
    """Generate an HTML/CSS visual comparison delta bar comparing Candidate vs Baseline."""
    diff = cand_val - base_val
    ratio = (cand_val / base_val) if base_val != 0 else 1.0

    color = FLAW_COLORS.get(flaw_type, "#4c6ef5")

    # Format human-readable delta
    if flaw_type == "long_pause":
        unit = "s"
        label = f"+{diff:.2f}s longer silence ({ratio:.1f}x baseline)"
    elif flaw_type == "rate_deviation":
        unit = "syl/s"
        pct = (ratio - 1.0) * 100
        sign = "+" if pct >= 0 else ""
        label = f"{sign}{pct:.0f}% pacing shift ({cand_val:.2f} vs {base_val:.2f} {unit})"
    elif flaw_type == "monotone":
        unit = "Hz std"
        pct_drop = max(0.0, (1.0 - ratio) * 100)
        label = f"-{pct_drop:.0f}% pitch variance loss (flatness)"
    else:
        unit = ""
        label = f"Measured: {cand_val:.2f} vs Reference: {base_val:.2f}"

    # Relative bar widths
    max_val = max(cand_val, base_val, 1e-5) * 1.2
    c_pct = min(100.0, max(5.0, (cand_val / max_val) * 100))
    b_pct = min(100.0, max(5.0, (base_val / max_val) * 100))

    html = f"""
    <div style="background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 12px; margin: 8px 0;">
      <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 6px;">
        <span style="color: #495057;">🔬 Acoustic Signal Deviation</span>
        <span style="color: {color}; font-weight: 700;">{label}</span>
      </div>
      <div style="margin-bottom: 6px;">
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #6c757d; margin-bottom: 2px;">
          <span>Candidate Measured: <strong>{cand_val:.2f} {unit}</strong></span>
        </div>
        <div style="background: #e9ecef; border-radius: 4px; height: 10px; width: 100%; overflow: hidden;">
          <div style="background: {color}; height: 100%; width: {c_pct:.1f}%; border-radius: 4px;"></div>
        </div>
      </div>
      <div>
        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #6c757d; margin-bottom: 2px;">
          <span>Speaker Baseline Normal: <strong>{base_val:.2f} {unit}</strong></span>
        </div>
        <div style="background: #e9ecef; border-radius: 4px; height: 8px; width: 100%; overflow: hidden;">
          <div style="background: #adb5bd; height: 100%; width: {b_pct:.1f}%; border-radius: 4px;"></div>
        </div>
      </div>
    </div>
    """
    return html


def render_word_ribbon(spans: list[Any], evidence_df: pd.DataFrame) -> str:
    """Render an interactive visual word-level alignment ribbon.

    Presents spoken words sequentially with color highlights on words that intersect detected flaw intervals.
    """
    if not spans:
        return ""

    flaw_intervals = []
    if evidence_df is not None and not evidence_df.empty:
        for _, r in evidence_df.iterrows():
            flaw_intervals.append((float(r["start_s"]), float(r["end_s"]), str(r["flaw_type"])))

    pills_html = []
    for s in spans:
        w = getattr(s, "word", "")
        w_start = getattr(s, "start_s", 0.0)
        w_end = getattr(s, "end_s", 0.0)

        # Check overlap
        matched_flaw = None
        for f_start, f_end, f_type in flaw_intervals:
            if max(w_start, f_start) < min(w_end, f_end):
                matched_flaw = f_type
                break

        if matched_flaw:
            color = FLAW_COLORS.get(matched_flaw, "#e4572e")
            pills_html.append(
                f'<span title="{w}: [{w_start:.2f}s - {w_end:.2f}s] Flaw: {matched_flaw}" '
                f'style="display: inline-block; background: {color}20; color: {color}; '
                f'border: 1.5px solid {color}; border-radius: 6px; padding: 3px 8px; '
                f'margin: 3px 4px; font-weight: 700; font-size: 13px; font-family: monospace;">'
                f'⚠️ {w}</span>'
            )
        else:
            pills_html.append(
                f'<span title="{w}: [{w_start:.2f}s - {w_end:.2f}s]" '
                f'style="display: inline-block; background: #ffffff; color: #495057; '
                f'border: 1px solid #dee2e6; border-radius: 6px; padding: 3px 8px; '
                f'margin: 3px 4px; font-size: 13px; font-family: monospace;">'
                f'{w}</span>'
            )

    container = f"""
    <div style="background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 12px; margin: 10px 0;">
      <div style="font-size: 12px; font-weight: 700; color: #495057; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px;">
        🗣️ Word-Level Spoken Transcript Alignment (Hover for Timestamps)
      </div>
      <div style="line-height: 2.2;">
        {' '.join(pills_html)}
      </div>
    </div>
    """
    return container
