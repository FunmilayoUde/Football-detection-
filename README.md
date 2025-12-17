# Football Detection & Tracking Pipeline

A computer-vision pipeline for football (soccer) broadcast video that:
- Detects **players, goalkeepers, referees, and ball** (YOLO)
- Tracks objects across frames (**BoT-SORT** via Ultralytics tracker)
- Classifies player teams (SigLIP embeddings → UMAP → KMeans + temporal smoothing)
- Estimates pitch homography and projects player/ball positions to pitch coordinates
- Exports:
  - Annotated output video (`output.mp4`)
  - Per-frame player coordinates (`output.csv`)

---

## Project Structure

`football_pipeline/`
- `main.py` — main pipeline runner
- `Detection.py` — detection wrapper (returns `ball_detections`, `all_detections`)
- `Tracker.py` — BoT-SORT tracking integration + detections adapter
- `Team_classification.py` — team clustering + goalkeeper assignment helper
- `Homography.py` — homography estimation + point projection
- `Visualization.py` — pitch rendering + inset overlays
- `stability.py` — team ID smoothing across frames
- `scene.py` — optional cut detector (scene cut reset)

---

## Requirements

- Python 3.10+ (tested on Windows)
- `ultralytics`, `supervision`, `opencv-python`, `numpy`, `pandas`

Install dependencies:

```bash
python -m venv venv
# Windows



venv\Scripts\activate
pip install -r football_pipeline/requirements.txt

Model Weights

You need:

Player/Object detector weights (YOLO)

Field/pitch detector weights (YOLO pose / keypoints)

Update these paths in football_pipeline/main.py, the weights are the runs folder, best.pt in each of them:
PLAYER_DETECTION_MODEL = YOLO(r"<path-to-player-detector-weights>.pt")
FIELD_DETECTION_MODEL  = YOLO(r"<path-to-field-detector-weights>.pt")

Run

From repo root:

python football_pipeline/main.py

Outputs:

output.mp4 (annotated video)

output.csv (player positions per frame)

Notes

Tracking uses BoT-SORT with camera motion compensation enabled by passing frames into the tracker update.

Team IDs shown in logs are team classification outputs (typically 0/1) and are separate from the detector class IDs.

The ball can be detected and drawn; ball tracking stability depends on consistent ball detection across frames.
