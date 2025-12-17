# tools/baseline_logger.py
import json
import os
import platform
import time
from pathlib import Path

def make_run_dirs(root: str = "runs/baseline") -> Path:
    """Create baseline run directories and return the root Path."""
    root_path = Path(root)
    (root_path / "mot").mkdir(parents=True, exist_ok=True)
    (root_path / "logs").mkdir(parents=True, exist_ok=True)
    return root_path

def seq_name_from_path(p: str) -> str:
    """Derive a sequence name from a video/file path (filename without extension)."""
    return Path(p).stem

class MotWriter:
    """
    Minimal MOTChallenge writer.
    Writes lines: frame,id,x,y,w,h,conf,class,visibility
    - frame: 1-based frame index
    - id: tracker id (-1 if missing)
    - (x,y,w,h): top-left + size in pixels
    - conf: detection score (0..1)
    - class: your class id (e.g., player/referee/…)
    - visibility: 0..1 (use 1 if unknown)
    """
    def __init__(self, out_txt_path: Path | str):
        out_txt_path = Path(out_txt_path)
        out_txt_path.parent.mkdir(parents=True, exist_ok=True)
        # line-buffered for real-time writes
        self.f = open(out_txt_path, "w", buffering=1)

    def write_detections(self, frame_idx: int, detections) -> None:
        """
        Expects a supervision.Detections object with:
          - xyxy (N,4)
          - tracker_id (N,) or None
          - confidence (N,) or None
          - class_id (N,) or None
        """
        if detections is None or len(detections) == 0:
            return
        xyxy = detections.xyxy
        tids = detections.tracker_id if getattr(detections, "tracker_id", None) is not None else [-1] * len(detections)
        conf = detections.confidence if getattr(detections, "confidence", None) is not None else [1.0] * len(detections)
        cls  = detections.class_id if getattr(detections, "class_id", None) is not None else [-1] * len(detections)

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(float, xyxy[i])
            x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)
            track_id = int(tids[i]) if tids[i] is not None else -1
            score = float(conf[i]) if conf[i] is not None else 1.0
            cid = int(cls[i]) if cls[i] is not None else -1
            # visibility set to 1 (unknown)
            self.f.write(f"{frame_idx},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.3f},{cid},1\n")

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

def save_fps_csv(path: Path | str, sequence: str, frames: int, elapsed_s: float, fps: float) -> None:
    """Append one row of FPS timing to a CSV (creates header if new)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with open(path, "a") as f:
        if header_needed:
            f.write("sequence,frames,elapsed_s,fps\n")
        f.write(f"{sequence},{frames},{elapsed_s:.3f},{fps:.2f}\n")

def save_config_json(path: Path | str, cfg_dict: dict) -> None:
    """Write a frozen config snapshot as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg_dict, f, indent=2)

def snapshot_hardware(path: Path | str) -> None:
    """Capture minimal hardware/runtime info (CPU/GPU/CUDA/PyTorch)."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
    }
    try:
        import torch  # optional
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = getattr(torch.version, "cuda", None)
    except Exception:
        info["torch_probe_error"] = True

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)

class Timer:
    """Context manager for wall-clock timing."""
    def __enter__(self):
        self._start = time.time()
        self.elapsed = 0.0
        return self
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.time() - self._start
