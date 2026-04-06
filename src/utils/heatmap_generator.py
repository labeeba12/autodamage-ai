import cv2
import numpy as np
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

# CAR PART ZONES
# Maps each part name to a region on the top-down car diagram (x, y, w, h) as fractions of diagram size (0.0 - 1.0)
PART_ZONES = {
    # Front sections
    "hood":            (0.30, 0.05, 0.40, 0.20),
    "windshield":      (0.25, 0.22, 0.50, 0.12),
    "grille":          (0.35, 0.01, 0.30, 0.06),
    "front-bumper":    (0.25, 0.00, 0.50, 0.05),
    "headlight":       (0.15, 0.02, 0.18, 0.08),

    # Middle sections
    "roof":            (0.25, 0.34, 0.50, 0.22),
    "front-door":      (0.05, 0.30, 0.20, 0.22),
    "back-door":       (0.05, 0.52, 0.20, 0.20),
    "front-window":    (0.08, 0.28, 0.16, 0.12),
    "back-window":     (0.08, 0.50, 0.16, 0.12),
    "mirror":          (0.01, 0.30, 0.06, 0.08),

    # Rear sections
    "trunk":           (0.30, 0.74, 0.40, 0.18),
    "back-windshield": (0.25, 0.66, 0.50, 0.10),
    "back-bumper":     (0.25, 0.93, 0.50, 0.05),
    "tail-light":      (0.15, 0.88, 0.18, 0.08),

    # Side sections
    "fender":          (0.05, 0.10, 0.20, 0.18),
    "quarter-panel":   (0.05, 0.68, 0.20, 0.18),
    "rocker-panel":    (0.05, 0.40, 0.18, 0.18),

    # Wheels
    "front-wheel":     (0.02, 0.14, 0.14, 0.14),
    "back-wheel":      (0.02, 0.70, 0.14, 0.14),

    # Other
    "license-plate":   (0.38, 0.94, 0.24, 0.04),
}

# Severity colors for heatmap
SEVERITY_COLORS = {
    "minor":    (0,   200, 100),   # Green
    "moderate": (0,   140, 255),   # Orange
    "severe":   (0,   50,  255),   # Red
    "unknown":  (128, 128, 128),   # Gray
}

DIAGRAM_W = 300
DIAGRAM_H = 600


def _draw_car_outline(canvas: np.ndarray) -> np.ndarray:
    """Draws a simple top-down car outline on the canvas."""
    h, w = canvas.shape[:2]

    # Car body outline
    body_pts = np.array([
        [int(w*0.25), int(h*0.02)],
        [int(w*0.75), int(h*0.02)],
        [int(w*0.85), int(h*0.10)],
        [int(w*0.90), int(h*0.30)],
        [int(w*0.90), int(h*0.70)],
        [int(w*0.85), int(h*0.90)],
        [int(w*0.75), int(h*0.98)],
        [int(w*0.25), int(h*0.98)],
        [int(w*0.15), int(h*0.90)],
        [int(w*0.10), int(h*0.70)],
        [int(w*0.10), int(h*0.30)],
        [int(w*0.15), int(h*0.10)],
    ], dtype=np.int32)

    cv2.fillPoly(canvas, [body_pts], (40, 40, 40))
    cv2.polylines(canvas, [body_pts], True, (80, 80, 80), 2)

    # Windshield
    ws_pts = np.array([
        [int(w*0.28), int(h*0.22)],
        [int(w*0.72), int(h*0.22)],
        [int(w*0.68), int(h*0.34)],
        [int(w*0.32), int(h*0.34)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [ws_pts], (50, 70, 80))
    cv2.polylines(canvas, [ws_pts], True, (80, 100, 110), 1)

    # Rear windshield
    rws_pts = np.array([
        [int(w*0.30), int(h*0.66)],
        [int(w*0.70), int(h*0.66)],
        [int(w*0.68), int(h*0.76)],
        [int(w*0.32), int(h*0.76)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [rws_pts], (50, 70, 80))
    cv2.polylines(canvas, [rws_pts], True, (80, 100, 110), 1)

    # Wheels (4 circles)
    wheel_positions = [
        (int(w*0.18), int(h*0.18)),   # front-left
        (int(w*0.82), int(h*0.18)),   # front-right
        (int(w*0.18), int(h*0.82)),   # rear-left
        (int(w*0.82), int(h*0.82)),   # rear-right
    ]
    for wx, wy in wheel_positions:
        cv2.circle(canvas, (wx, wy), int(w*0.09), (30, 30, 30), -1)
        cv2.circle(canvas, (wx, wy), int(w*0.09), (60, 60, 60),  2)
        cv2.circle(canvas, (wx, wy), int(w*0.04), (50, 50, 50), -1)

    return canvas


def generate_heatmap(
    matched_results: list[dict],
    output_path:     str
) -> str:
    """
    Generates a top-down car heatmap showing damaged areas.

    Each damaged part is highlighted with severity color:
        Green  = minor
        Orange = moderate
        Red    = severe

    Returns: output_path of saved heatmap image
    """

    # Create canvas
    canvas = np.zeros((DIAGRAM_H, DIAGRAM_W, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)  # dark background

    # Draw car outline
    canvas = _draw_car_outline(canvas)

    # Highlight damaged parts
    damaged_parts = {}
    for item in matched_results:
        part     = item["part_name"].lower()
        severity = item.get("severity_label", "unknown")

        # Keep worst severity per part
        if part not in damaged_parts:
            damaged_parts[part] = severity
        else:
            severity_rank = {"minor": 0, "moderate": 1, "severe": 2}
            if severity_rank.get(severity, 0) > severity_rank.get(
                damaged_parts[part], 0
            ):
                damaged_parts[part] = severity

    for part_name, severity in damaged_parts.items():
        if part_name not in PART_ZONES:
            continue

        x_frac, y_frac, w_frac, h_frac = PART_ZONES[part_name]
        x1 = int(x_frac * DIAGRAM_W)
        y1 = int(y_frac * DIAGRAM_H)
        x2 = int((x_frac + w_frac) * DIAGRAM_W)
        y2 = int((y_frac + h_frac) * DIAGRAM_H)

        color   = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])
        overlay = canvas.copy()

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    # Add title and legend
    title_bar    = np.zeros((50, DIAGRAM_W, 3), dtype=np.uint8)
    title_bar[:] = (15, 15, 15)
    cv2.putText(
        title_bar, "DAMAGE HEATMAP",
        (10, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (200, 200, 200), 1
    )

    legend_bar    = np.zeros((60, DIAGRAM_W, 3), dtype=np.uint8)
    legend_bar[:] = (15, 15, 15)

    legend_items = [
        ("MINOR",    (0, 200, 100), 10),
        ("MODERATE", (0, 140, 255), 90),
        ("SEVERE",   (0, 50,  255), 190),
    ]

    for label, color, x_offset in legend_items:
        cv2.circle(legend_bar, (x_offset + 8, 30), 8, color, -1)
        cv2.putText(
            legend_bar, label,
            (x_offset + 22, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, (180, 180, 180), 1
        )

    # Stack title + diagram + legend
    final = np.vstack([title_bar, canvas, legend_bar])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final)
    logger.info(f"Heatmap saved: {output_path}")

    return output_path