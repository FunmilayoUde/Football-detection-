import cv2
import numpy as np
import supervision as sv
from types import SimpleNamespace
from ultralytics.trackers.bot_sort import BOTSORT


def get_video_fps(path: str, default: float = 30.0, stride: int = 1) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or default
    cap.release()
    return float(fps) / max(1, stride)


def _make_botsort_args(src_fps: float, profile: str = "human") -> SimpleNamespace:
    """
    profile="human": tuned for players/refs/goalkeepers
    profile="ball":  more permissive thresholds for small/fast object
    """
    if profile == "ball":
        
        return SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=0.20,
            track_low_thresh=0.05,
            new_track_thresh=0.20,
            track_buffer=int(src_fps * 2.0),
            match_thresh=0.70,
            proximity_thresh=0.70,
            appearance_thresh=0.25,
            gmc_method="orb",
            with_reid=False,
            fast_reid_config="",
            fast_reid_weights="",
            model="",
            fuse_score=False,
            mot20=False,
            conf_thres=0.001,
            device="cpu",
        )

    
    return SimpleNamespace(
        tracker_type="botsort",
        track_high_thresh=0.40,
        track_low_thresh=0.10,
        new_track_thresh=0.75,
        track_buffer=int(src_fps * 8.0),
        match_thresh=0.90,
        proximity_thresh=0.50,
        appearance_thresh=0.25,
        gmc_method="orb",
        with_reid=False,
        fast_reid_config="",
        fast_reid_weights="",
        model="",
        fuse_score=False,
        mot20=False,
        conf_thres=0.001,
        device="cpu",
    )


def initialize_tracker(src_fps: float = 30.0, profile: str = "human"):
    args = _make_botsort_args(src_fps, profile=profile)
    return BOTSORT(args=args, frame_rate=int(round(src_fps)))


class _SvDetectionsAdapter:
    """
    Adapter to make `sv.Detections` look like Ultralytics Results for tracker.update().
    Provides: xyxy, xywh, conf, cls, __len__, __getitem__
    """
    def __init__(self, detections: sv.Detections):
        self._xyxy = detections.xyxy.astype(float)
        self.conf = detections.confidence.astype(float) if detections.confidence is not None else np.zeros((len(detections),), dtype=float)

        if detections.class_id is None:
            self.cls = np.zeros((len(detections),), dtype=float)
        else:
            self.cls = detections.class_id.astype(float)

    @property
    def xyxy(self) -> np.ndarray:
        return self._xyxy

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self._xyxy.T
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        return np.stack([cx, cy, w, h], axis=-1)

    def __len__(self) -> int:
        return len(self._xyxy)

    def __getitem__(self, idx):
        new = object.__new__(_SvDetectionsAdapter)
        new._xyxy = self._xyxy[idx]
        new.conf = self.conf[idx]
        new.cls = self.cls[idx]
        return new


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (N,4) xyxy, b: (M,4) xyxy -> iou (N,M)
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)

    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, (ax2 - ax1)) * np.maximum(0.0, (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))

    union = area_a + area_b - inter + 1e-9
    return (inter / union).astype(float)


def update_tracks(tracker, detections: sv.Detections, frame) -> sv.Detections:
    """
    Track the given detections and return subset of detections with `tracker_id`.
    We pass `img=frame` to enable GMC (camera motion compensation).
    """
    if detections is None or len(detections) == 0:
        empty = detections[:0] if detections is not None else sv.Detections.empty()
        empty.tracker_id = np.empty((0,), dtype=int)
        return empty

    results_adapter = _SvDetectionsAdapter(detections)
    tracks_np = tracker.update(results_adapter, img=frame)

    if tracks_np is None or len(tracks_np) == 0:
        empty = detections[:0]
        empty.tracker_id = np.empty((0,), dtype=int)
        return empty

    tracks_np = np.asarray(tracks_np, dtype=np.float32)

    if tracks_np.shape[1] >= 8:
        track_ids = tracks_np[:, 4].astype(int)
        det_indices = tracks_np[:, 7].astype(int)
        det_indices = np.clip(det_indices, 0, len(detections) - 1)

        tracked = detections[det_indices]
        tracked.tracker_id = track_ids
        return tracked

    track_boxes = tracks_np[:, 0:4]
    iou = _iou_xyxy(track_boxes, detections.xyxy)
    det_indices = np.argmax(iou, axis=1)
    track_ids = tracks_np[:, 4].astype(int) if tracks_np.shape[1] >= 5 else np.arange(len(det_indices))

    tracked = detections[det_indices]
    tracked.tracker_id = track_ids
    return tracked
