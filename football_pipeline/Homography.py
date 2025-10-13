import numpy as np
import cv2
import supervision as sv
from sports.annotators.soccer import draw_pitch
from sports.configs.soccer import SoccerPitchConfiguration


class ViewTransformer:
    """Handles 2D-2D point transformations between frame and pitch coordinates."""

    def __init__(self, source, target):
        self.homography, _ = cv2.findHomography(source, target)

    def transform_points(self, points):
        """Apply the homography to a set of 2D points."""
        if points is None or len(points) == 0:
            return np.empty((0, 2))
        points_homogeneous = np.hstack([points, np.ones((len(points), 1))])
        transformed = (self.homography @ points_homogeneous.T).T
        transformed /= transformed[:, 2:3]
        return transformed[:, :2]


def estimate_homography(frame, field_model, config: SoccerPitchConfiguration, conf_threshold=0.3):
    """
    Estimate the homography matrix between the video frame and the pitch coordinates.
    Returns (pitch_reference_points, transformer)
    """

    # Run field keypoints detection
    result = field_model.predict(frame, conf=conf_threshold)[0]
    key_points = sv.KeyPoints.from_ultralytics(result)

    # Filter keypoints by confidence
    mask = key_points.confidence[0] > 0.5
    frame_reference_points = key_points.xy[0][mask]
    pitch_reference_points = np.array(config.vertices)[mask]

    if len(frame_reference_points) < 4:
        # Not enough points to compute homography
        return None, None

    transformer = ViewTransformer(source=frame_reference_points, target=pitch_reference_points)
    return pitch_reference_points, transformer


def draw_field_keypoints(frame, key_points):
    """Draw detected field keypoints for visualization."""
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex("#FF1493"), radius=8)
    annotated = frame.copy()
    annotated = vertex_annotator.annotate(scene=annotated, key_points=key_points)
    return annotated