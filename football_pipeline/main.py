import pandas as pd
import cv2
import numpy as np
import supervision as sv

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from layers.detection import detect_objects
from layers.tracking import initialize_tracker, update_tracks
from layers.team_classification import TeamClassifier
from layers.homography import estimate_homography, ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from layers.visualization import render_pitch_views, overlay_inset_on_frame

# === CONFIGURATION ===
CONFIG = SoccerPitchConfiguration()
SOURCE_VIDEO_PATH = "/content/HILAL-HAZM_match_B_up10.mp4"
OUTPUT_VIDEO_PATH = "/content/MARKETING1_B-up10_csv_output.mp4"

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# === INITIALIZE MODELS ===
from ultralytics import YOLO
PLAYER_DETECTION_MODEL = YOLO("MQ_players_detection.pt") 
FIELD_DETECTION_MODEL = YOLO("MQ_field_detection.pt")  

# === INITIALIZE MODULES ===
tracker = initialize_tracker()
team_classifier = TeamClassifier(device="cuda")

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(['#FFC000', '#0000FF', '#FFD700']),
    thickness=2
)
label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(['#FFC000', '#0000FF', '#FFD700']),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#00FF00'),
    base=20, height=17
)

print("✅ Team classifier initialized (SigLIP + UMAP + KMeans)")

# === INITIAL TEAM CALIBRATION ===
calibration_crops = []
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=30)
print("🟡 Collecting initial samples for team calibration...")

for i, frame in enumerate(frame_generator):
    result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.class_id == PLAYER_ID]
    crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
    calibration_crops += crops
    if len(calibration_crops) > 100:
        break

team_classifier.fit(calibration_crops)

# === VIDEO WRITER ===
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
first_frame = next(frame_generator)
height, width = first_frame.shape[:2]
video_writer = cv2.VideoWriter(
    OUTPUT_VIDEO_PATH,
    fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
    fps=25,
    frameSize=(width, height)
)

player_data = []
frame_index = 0

# === MAIN LOOP ===
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
for frame in frame_generator:
    frame_index += 1

    # === DETECTION ===
    ball_detections, all_detections = detect_objects(
        frame=frame,
        model=PLAYER_DETECTION_MODEL,
        ball_id=BALL_ID
    )

    # === TRACKING ===
    all_detections = update_tracks(tracker, all_detections)

    # === TEAM CLASSIFICATION ===
    player_detections = all_detections[all_detections.class_id == PLAYER_ID]
    player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]

    if len(player_crops) > 0:
        team_ids = team_classifier.predict(player_crops)
        player_detections.class_id = team_ids
        print(f"Frame {frame_index}: Team IDs ->", team_ids)
    else:
        print(f"Frame {frame_index}: No players detected")

    # === GOALKEEPER CLASSIFICATION ===
    goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
    goalkeepers_team_id = team_classifier.resolve_goalkeepers_team_id(
        player_detections, goalkeepers_detections
    )
    goalkeepers_detections.class_id = goalkeepers_team_id

    # === TEAM COLORS ===
    TEAM_COLORS = {
        0: sv.Color.from_hex('#FFC000'),  # Team 1 (yellow)
        1: sv.Color.from_hex('#0000FF'),  # Team 2 (blue)
    }

    for i, det in enumerate(player_detections):
        class_id = int(det.class_id)
        ellipse_annotator.color = TEAM_COLORS.get(class_id, sv.Color.WHITE)

    # === HOMOGRAPHY ===
    pitch_reference_points, transformer = estimate_homography(
        frame=frame,
        field_model=FIELD_DETECTION_MODEL,
        config=CONFIG
    )

    if transformer is None:
        video_writer.write(frame)
        continue

    # Project coordinates
    frame_players_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(frame_players_xy)

    if len(ball_detections) > 0:
        pitch_ball_xy = transformer.transform_points(
            ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        )[0]
    else:
        pitch_ball_xy = None

        # === VISUALIZATION ===
    annotated_pitch, annotated_voronoi, annotated_blended = render_pitch_views(
        CONFIG,
        pitch_players_xy=pitch_players_xy,
        pitch_ball_xy=pitch_ball_xy,
        players_class_id=player_detections.class_id
    )

    final_frame = overlay_inset_on_frame(
        frame=frame,
        insets=[annotated_pitch, annotated_voronoi, annotated_blended],
        opacity=0.5
    )

    # === SAVE PLAYER DATA ===
    if len(player_detections) > 0:
        for idx, det in enumerate(player_detections):
            track_id = int(det.tracker_id) if det.tracker_id is not None else -1
            team_id = int(det.class_id)
            x_m, y_m = pitch_players_xy[idx]
            player_data.append({
                "frame": frame_index,
                "track_id": track_id,
                "team_id": team_id,
                "x_m": x_m,
                "y_m": y_m
            })

    video_writer.write(final_frame)

# === END OF VIDEO ===
video_writer.release()

# === SAVE PLAYER DATA TO CSV ===
df = pd.DataFrame(player_data)
csv_output_path = OUTPUT_VIDEO_PATH.replace(".mp4", ".csv")
df.to_csv(csv_output_path, index=False)
print("✅ Video saved to:", OUTPUT_VIDEO_PATH)
print("✅ Player data saved to:", csv_output_path)
