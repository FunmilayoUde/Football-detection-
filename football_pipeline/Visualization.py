import cv2
import numpy as np
import supervision as sv
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from typing import Optional
from sports.configs.soccer import SoccerPitchConfiguration


def draw_pitch_voronoi_diagram_2(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """Smooth Voronoi pitch visualization with color blending."""
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coords, x_coords = np.indices((scaled_width + 2 * padding, scaled_length + 2 * padding))
    y_coords -= padding
    x_coords -= padding

    def distance_matrix(xy, x_grid, y_grid):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_grid) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_grid) ** 2)

    distances_t1 = distance_matrix(team_1_xy, x_coords, y_coords)
    distances_t2 = distance_matrix(team_2_xy, x_coords, y_coords)

    d1 = np.min(distances_t1, axis=0)
    d2 = np.min(distances_t2, axis=0)

    steepness = 15
    ratio = d2 / np.clip(d1 + d2, a_min=1e-5, a_max=None)
    blend = np.tanh((ratio - 0.5) * steepness) * 0.5 + 0.5

    for c in range(3):
        voronoi[:, :, c] = (blend * team_1_color_bgr[c] + (1 - blend) * team_2_color_bgr[c]).astype(np.uint8)

    return cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)


def render_pitch_views(
    config: SoccerPitchConfiguration,
    pitch_players_xy: np.ndarray,
    pitch_ball_xy: np.ndarray,
    players_class_id: np.ndarray,
    team_1_color: str = 'FFC000',
    team_2_color: str = '0000FF'
):
    """Create the 3 pitch visualizations (Radar + Voronoi + Blended Voronoi)."""
    team1_mask = players_class_id == 0
    team2_mask = players_class_id == 1

    annotated_pitch = draw_pitch(config)
    annotated_pitch = draw_points_on_pitch(
        config, xy=pitch_ball_xy, face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK, radius=10, pitch=annotated_pitch
    )
    annotated_pitch = draw_points_on_pitch(
        config, xy=pitch_players_xy[team1_mask],
        face_color=sv.Color.from_hex(team_1_color), edge_color=sv.Color.BLACK, radius=16, pitch=annotated_pitch
    )
    annotated_pitch = draw_points_on_pitch(
        config, xy=pitch_players_xy[team2_mask],
        face_color=sv.Color.from_hex(team_2_color), edge_color=sv.Color.BLACK, radius=16, pitch=annotated_pitch
    )

    annotated_voronoi = draw_pitch_voronoi_diagram_2(
        config, pitch_players_xy[team1_mask], pitch_players_xy[team2_mask],
        sv.Color.from_hex(team_1_color), sv.Color.from_hex(team_2_color)
    )

    annotated_blended = draw_pitch_voronoi_diagram_2(
        config, pitch_players_xy[team1_mask], pitch_players_xy[team2_mask],
        sv.Color.from_hex(team_1_color), sv.Color.from_hex(team_2_color)
    )

    annotated_blended = draw_points_on_pitch(
        config, xy=pitch_ball_xy, face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK, radius=10, pitch=annotated_blended
    )
    annotated_blended = draw_points_on_pitch(
        config, xy=pitch_players_xy[team1_mask],
        face_color=sv.Color.from_hex(team_1_color), edge_color=sv.Color.BLACK, radius=16, pitch=annotated_blended
    )
    annotated_blended = draw_points_on_pitch(
        config, xy=pitch_players_xy[team2_mask],
        face_color=sv.Color.from_hex(team_2_color), edge_color=sv.Color.BLACK, radius=16, pitch=annotated_blended
    )

    return annotated_pitch, annotated_voronoi, annotated_blended


def overlay_inset_on_frame(frame: np.ndarray, insets: list[np.ndarray], opacity: float = 0.5) -> np.ndarray:
    """Overlay multiple pitch visualizations (side-by-side) onto main broadcast frame."""
    inset_h, inset_w = 350, 600
    resized_insets = [cv2.resize(img, (inset_w, inset_h)) for img in insets]
    collage = np.hstack(resized_insets)

    pitch_rgba = cv2.cvtColor(collage, cv2.COLOR_BGR2BGRA)
    pitch_rgba[:, :, 3] = int(255 * opacity)
    real_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    y_offset = real_rgba.shape[0] - pitch_rgba.shape[0] - 20
    x_offset = (real_rgba.shape[1] - pitch_rgba.shape[1]) // 2
    overlay = real_rgba.copy()
    roi = overlay[y_offset:y_offset+pitch_rgba.shape[0], x_offset:x_offset+pitch_rgba.shape[1]]

    alpha_mask = pitch_rgba[:, :, 3:] / 255.0
    alpha_inv = 1.0 - alpha_mask

    for c in range(3):
        roi[:, :, c] = (alpha_mask[:, :, 0] * pitch_rgba[:, :, c] + alpha_inv[:, :, 0] * roi[:, :, c])

    overlay[y_offset:y_offset+pitch_rgba.shape[0], x_offset:x_offset+pitch_rgba.shape[1]] = roi
    return cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)


