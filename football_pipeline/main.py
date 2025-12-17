# import pandas as pd
# import cv2
# import numpy as np
# import supervision as sv
# from scene import CutDetector

# from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
# from Detection import detect_objects
# from Tracker import initialize_tracker, update_tracks, get_video_fps
# from Team_classification import TeamClassifier
# from Homography import estimate_homography, ViewTransformer
# from sports.configs.soccer import SoccerPitchConfiguration
# from Visualization import render_pitch_views, overlay_inset_on_frame
# from stability import TeamStability



# from tools.baseline_logger import (
#     make_run_dirs, MotWriter, seq_name_from_path, Timer,
#     save_fps_csv, save_config_json, snapshot_hardware
# )

# cut_det = CutDetector(thresh=0.40)
# team_stab = TeamStability(window=7, min_consensus= 4)

# # === CONFIGURATION ===
# CONFIG = SoccerPitchConfiguration()
# SOURCE_VIDEO_PATH = r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\Video_sample\HIL-HAZ.mp4"
# OUTPUT_VIDEO_PATH = "output.mp4"

# BALL_ID = 0
# GOALKEEPER_ID = 1
# PLAYER_ID = 2
# REFEREE_ID = 3

# RUN_ROOT = make_run_dirs("runs/baseline")
# SEQUENCE = seq_name_from_path(SOURCE_VIDEO_PATH)
# MOT_PATH = RUN_ROOT / "mot" / f"{SEQUENCE}.txt"
# FPS_CSV = RUN_ROOT / "logs" / "per_video_fps.csv"
# CFG_JSON = RUN_ROOT / "logs" / "config.json"
# HW_JSON  = RUN_ROOT / "logs" / "hardware.json"


# # === INITIALIZE MODELS ===
# from ultralytics import YOLO
# PLAYER_DETECTION_MODEL = YOLO(r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\runs_main\y8s_e100\weights\best.pt") 
# FIELD_DETECTION_MODEL = YOLO(r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\runs_field\y8s_pose_e802\weights\best.pt")

# # === INITIALIZE MODULES ===
# src_fps = get_video_fps(r"C:\Users\Chimdi\Downloads\Football detection\Football_test\Video_sample\HIL-HAZ.mp4")
# tracker = initialize_tracker(src_fps= src_fps)
# team_classifier = TeamClassifier(device="cpu")

# ellipse_annotator = sv.EllipseAnnotator(
#     color=sv.ColorPalette.from_hex(['#FFC000', '#0000FF', '#FFD700']),
#     thickness=2
# )
# label_annotator = sv.LabelAnnotator(
#     color=sv.ColorPalette.from_hex(['#FFC000', '#0000FF', '#FFD700']),
#     text_color=sv.Color.from_hex('#000000'),
#     text_position=sv.Position.BOTTOM_CENTER
# )
# triangle_annotator = sv.TriangleAnnotator(
#     color=sv.Color.from_hex('#00FF00'),
#     base=20, height=17
# )

# print("✅ Team classifier initialized (SigLIP + UMAP + KMeans)")

# # === INITIAL TEAM CALIBRATION ===
# calibration_crops = []
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=30)
# print("🟡 Collecting initial samples for team calibration...")

# for i, frame in enumerate(frame_generator):
#     result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
#     detections = sv.Detections.from_ultralytics(result)
#     detections = detections[detections.class_id == PLAYER_ID]
#     crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
#     calibration_crops += crops
#     if len(calibration_crops) > 100:
#         break

# team_classifier.fit(calibration_crops)

# _cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
# src_fps = _cap.get(cv2.CAP_PROP_FPS) or 25.0
# _cap.release()
# print(f"[INFO] Source FPS: {src_fps:.3f}")

# # === CLIP SETTINGS ===
# CLIP_START_SEC = 0        # start at 0s (beginning of video)
# CLIP_DURATION_SEC = 30    # only process 30 seconds
# CLIP_START_FRAME = int(CLIP_START_SEC * src_fps)
# CLIP_END_FRAME = int((CLIP_START_SEC + CLIP_DURATION_SEC) * src_fps)

# print(f"[INFO] Will process frames {CLIP_START_FRAME} to {CLIP_END_FRAME} (~{CLIP_DURATION_SEC} seconds)")


# # baseline_cfg = {
# #   "video": str(SOURCE_VIDEO_PATH),
# #   "detector": {"weights":"MQ_players_detection.pt","conf":0.3,"nms_iou":0.5},
# #   "field_detector": {"weights":"MQ_field_detection.pt","conf":0.3},
# #   "classes": {"BALL_ID": BALL_ID, "GOALKEEPER_ID": GOALKEEPER_ID, "PLAYER_ID": PLAYER_ID, "REFEREE_ID": REFEREE_ID},
# #   "tracker": {"algo":"ByteTrack","defaults":True},
# #   "team_classifier": {"model":"google/siglip-base-patch16-224","umap_components":3,"kmeans_k":2},
# #   "src_fps": float(src_fps)
# # }

# baseline_cfg = {
#   "video": str(SOURCE_VIDEO_PATH),
#   "detector": {"weights": r"runs_main\y8s_e100\weights\best.pt", "conf": 0.25, "nms_iou": 0.55, "imgsz": 1280},
#   "field_detector": {"weights": None, "conf": 0.3},
#   "classes": {"BALL_ID": BALL_ID, "GOALKEEPER_ID": GOALKEEPER_ID, "PLAYER_ID": PLAYER_ID, "REFEREE_ID": REFEREE_ID},
#   "tracker": {"algo": "ByteTrack", "defaults": True},  # we'll fill actual tuned params in the next step
#   "team_classifier": {"model": "google/siglip-base-patch16-224", "umap_components": 3, "kmeans_k": 2},
#   "src_fps": float(src_fps)
# }
# save_config_json(CFG_JSON, baseline_cfg)
# snapshot_hardware(HW_JSON)


# # === VIDEO WRITER ===
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# first_frame = next(frame_generator)
# height, width = first_frame.shape[:2]
# video_writer = cv2.VideoWriter(
#     OUTPUT_VIDEO_PATH,
#     fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
#     fps=src_fps,
#     frameSize=(width, height)
# )


# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# frame_index = 0
# player_data = []


# # === MAIN LOOP ===
# for frame in frame_generator:
#     frame_index += 1

#     # 🔹 Skip frames before the clip start
#     if frame_index < CLIP_START_FRAME:
#         continue

#     # 🔹 Stop completely after the clip end
#     if frame_index > CLIP_END_FRAME:
#         print(f"[INFO] Reached end of clip at frame {frame_index}, stopping.")
#         break

#     if cut_det.is_cut(frame):
#         tracker.reset()

# # === MAIN LOOP ===
# # frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# # for frame in frame_generator:
# #     frame_index += 1

#     # if cut_det.is_cut(frame):
#     #     tracker.reset()
#     # === DETECTION ===
#     ball_detections, all_detections = detect_objects(
#         frame=frame,
#         model=PLAYER_DETECTION_MODEL,
#         ball_id=BALL_ID
#     )

#     # --- TINY-BOX FILTER (add here, before tracking) ---
#     if len(all_detections):
#         xyxy = all_detections.xyxy
#         w = xyxy[:, 2] - xyxy[:, 0]
#         h = xyxy[:, 3] - xyxy[:, 1]

#         PLAY_MIN, BALL_MIN = 10, 4   # pixels; tune per resolution
#         is_ball = (all_detections.class_id == BALL_ID)
#         keep = ((w >= PLAY_MIN) & (h >= PLAY_MIN)) | (is_ball & (w >= BALL_MIN) & (h >= BALL_MIN))
#         all_detections = all_detections[keep]

#     # === TRACKING ===
#     all_detections = update_tracks(tracker, all_detections)

    

#     # === TEAM CLASSIFICATION ===
#     player_detections = all_detections[all_detections.class_id == PLAYER_ID]
#     player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]

#     if len(player_crops) > 0:
#         team_ids = team_classifier.predict(player_crops)

#         track_ids = player_detections.tracker_id
#         if track_ids is not None and len(track_ids) == len(team_ids):
#             team_ids = team_stab.update(track_ids,team_ids)
#         player_detections.class_id = team_ids
#         print(f"Frame {frame_index}: Team IDs ->", team_ids)
#     else:
#         print(f"Frame {frame_index}: No players detected")

#     # === GOALKEEPER CLASSIFICATION ===
#     goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
#     goalkeepers_team_id = team_classifier.resolve_goalkeepers_team_id(
#         player_detections, goalkeepers_detections
#     )
#     goalkeepers_detections.class_id = goalkeepers_team_id

#     # === TEAM COLORS ===
#     TEAM_COLORS = {
#         0: sv.Color.from_hex('#FFC000'),  # Team 1 (yellow)
#         1: sv.Color.from_hex('#0000FF'),  # Team 2 (blue)
#     }

#     # for i, det in enumerate(player_detections):
#     #     print(det)
#     #     class_id = int(det.class_id)
#     #     ellipse_annotator.color = TEAM_COLORS.get(class_id, sv.Color.WHITE)
#     annotated_frame = ellipse_annotator.annotate(
#     scene=frame.copy(),
#     detections=player_detections
#         )
#     # === HOMOGRAPHY ===
#     pitch_reference_points, transformer = estimate_homography(
#         frame=frame,
#         field_model=FIELD_DETECTION_MODEL,
#         config=CONFIG
#     )

#     if transformer is None:
#         video_writer.write(frame)
#         continue

#     # Project coordinates
#     frame_players_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#     pitch_players_xy = transformer.transform_points(frame_players_xy)

#     if len(ball_detections) > 0:
#         pitch_ball_xy = transformer.transform_points(
#             ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         )[0]
#     else:
#         pitch_ball_xy = None

        
#     annotated_pitch, annotated_voronoi, annotated_blended = render_pitch_views(
#         CONFIG,
#         pitch_players_xy=pitch_players_xy,
#         pitch_ball_xy=pitch_ball_xy,
#         players_class_id=player_detections.class_id
#     )

#     final_frame = overlay_inset_on_frame(
#         frame=frame,
#         insets=[annotated_pitch, annotated_voronoi, annotated_blended],
#         opacity=0.5
#     )

#     # === SAVE PLAYER DATA ===
#     if len(player_detections) > 0:
#         for idx, det in enumerate(player_detections):
#             track_id = int(det.tracker_id) if det.tracker_id is not None else -1
#             team_id = int(det.class_id)
#             x_m, y_m = pitch_players_xy[idx]
#             player_data.append({
#                 "frame": frame_index,
#                 "track_id": track_id,
#                 "team_id": team_id,
#                 "x_m": x_m,
#                 "y_m": y_m
#             })

#     video_writer.write(final_frame)

# # === END OF VIDEO ===
# video_writer.release()

# # === SAVE PLAYER DATA TO CSV ===
# df = pd.DataFrame(player_data)
# csv_output_path = OUTPUT_VIDEO_PATH.replace(".mp4", ".csv")
# df.to_csv(csv_output_path, index=False)
# print("✅ Video saved to:", OUTPUT_VIDEO_PATH)
# print("✅ Player data saved to:", csv_output_path)

# import pandas as pd
# import cv2
# import numpy as np
# import supervision as sv
# from scene import CutDetector

# from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
# from Detection import detect_objects
# from Tracker import initialize_tracker, update_tracks, get_video_fps
# from Team_classification import TeamClassifier
# from Homography import estimate_homography, ViewTransformer
# from sports.configs.soccer import SoccerPitchConfiguration
# from Visualization import render_pitch_views, overlay_inset_on_frame
# from stability import TeamStability

# from tools.baseline_logger import (
#     make_run_dirs, MotWriter, seq_name_from_path, Timer,
#     save_fps_csv, save_config_json, snapshot_hardware
# )

# # === HELPERS / MODULES ===
# cut_det = CutDetector(thresh=0.40)
# team_stab = TeamStability(window=7, min_consensus=4)

# # === CONFIGURATION ===
# CONFIG = SoccerPitchConfiguration()
# SOURCE_VIDEO_PATH = r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\Video_sample\HIL-HAZ.mp4"
# OUTPUT_VIDEO_PATH = "output.mp4"

# BALL_ID = 0
# GOALKEEPER_ID = 1
# PLAYER_ID = 2
# REFEREE_ID = 3

# RUN_ROOT = make_run_dirs("runs/baseline")
# SEQUENCE = seq_name_from_path(SOURCE_VIDEO_PATH)
# MOT_PATH = RUN_ROOT / "mot" / f"{SEQUENCE}.txt"
# FPS_CSV = RUN_ROOT / "logs" / "per_video_fps.csv"
# CFG_JSON = RUN_ROOT / "logs" / "config.json"
# HW_JSON  = RUN_ROOT / "logs" / "hardware.json"

# # === INITIALIZE MODELS ===
# from ultralytics import YOLO
# PLAYER_DETECTION_MODEL = YOLO(
#     r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\runs_main\y8s_e100\weights\best.pt"
# )
# FIELD_DETECTION_MODEL = YOLO(
#     r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\runs_field\y8s_pose_e802\weights\best.pt"
# )

# # === INITIALIZE MODULES ===
# src_fps = get_video_fps(SOURCE_VIDEO_PATH)
# tracker = initialize_tracker(src_fps=src_fps)
# team_classifier = TeamClassifier(device="cpu")

# ellipse_annotator = sv.EllipseAnnotator(
#     color=sv.ColorPalette.from_hex(['#FFC000', '#0000FF', '#FFD700']),
#     thickness=2
# )
# label_annotator = sv.LabelAnnotator(
#     color=sv.ColorPalette.from_hex(['#FFC000', '#0000FF', '#FFD700']),
#     text_color=sv.Color.from_hex('#000000'),
#     text_position=sv.Position.BOTTOM_CENTER
# )
# triangle_annotator = sv.TriangleAnnotator(
#     color=sv.Color.from_hex('#00FF00'),
#     base=20,
#     height=17
# )

# print("✅ Team classifier initialized (SigLIP + UMAP + KMeans)")

# # === RE-CHECK FPS (optional, using OpenCV directly) ===
# _cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
# src_fps_cap = _cap.get(cv2.CAP_PROP_FPS) or src_fps or 25.0
# _cap.release()
# src_fps = src_fps_cap
# print(f"[INFO] Source FPS: {src_fps:.3f}")

# # === CLIP SETTINGS ===
# CLIP_START_SEC = 0        # start at 0s (beginning of video)
# CLIP_DURATION_SEC = 30    # only process 30 seconds
# CLIP_START_FRAME = int(CLIP_START_SEC * src_fps)
# CLIP_END_FRAME = int((CLIP_START_SEC + CLIP_DURATION_SEC) * src_fps)
# print(f"[INFO] Will process frames {CLIP_START_FRAME} to {CLIP_END_FRAME} (~{CLIP_DURATION_SEC} seconds)")

# # === BASELINE CONFIG LOGGING ===
# baseline_cfg = {
#     "video": str(SOURCE_VIDEO_PATH),
#     "detector": {
#         "weights": r"runs_main\y8s_e100\weights\best.pt",
#         "conf": 0.25,
#         "nms_iou": 0.55,
#         "imgsz": 1280
#     },
#     "field_detector": {"weights": None, "conf": 0.3},
#     "classes": {
#         "BALL_ID": BALL_ID,
#         "GOALKEEPER_ID": GOALKEEPER_ID,
#         "PLAYER_ID": PLAYER_ID,
#         "REFEREE_ID": REFEREE_ID
#     },
#     "tracker": {"algo": "ByteTrack", "defaults": True},
#     "team_classifier": {
#         "model": "google/siglip-base-patch16-224",
#         "umap_components": 3,
#         "kmeans_k": 2
#     },
#     "src_fps": float(src_fps)
# }
# save_config_json(CFG_JSON, baseline_cfg)
# snapshot_hardware(HW_JSON)

# # === INITIAL TEAM CALIBRATION ===
# calibration_crops = []
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=30)
# print("🟡 Collecting initial samples for team calibration...")

# for i, frame in enumerate(frame_generator):
#     result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
#     detections = sv.Detections.from_ultralytics(result)
#     detections = detections[detections.class_id == PLAYER_ID]
#     crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
#     calibration_crops += crops
#     if len(calibration_crops) > 100:
#         break

# team_classifier.fit(calibration_crops)

# # === VIDEO WRITER ===
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# first_frame = next(frame_generator)
# height, width = first_frame.shape[:2]
# video_writer = cv2.VideoWriter(
#     OUTPUT_VIDEO_PATH,
#     fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
#     fps=src_fps,
#     frameSize=(width, height)
# )

# # === MAIN LOOP SETUP ===
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# frame_index = 0
# player_data = []

# # === MAIN LOOP ===
# for frame in frame_generator:
#     frame_index += 1

#     # Skip frames before the clip start
#     if frame_index < CLIP_START_FRAME:
#         continue

#     # Stop after the clip end
#     if frame_index > CLIP_END_FRAME:
#         print(f"[INFO] Reached end of clip at frame {frame_index}, stopping.")
#         break

#     if cut_det.is_cut(frame):
#         tracker.reset()

#     # === DETECTION ===
#     ball_detections, all_detections = detect_objects(
#         frame=frame,
#         model=PLAYER_DETECTION_MODEL,
#         ball_id=BALL_ID
#     )

#     # --- TINY-BOX FILTER (add here, before tracking) ---
#     if len(all_detections):
#         xyxy = all_detections.xyxy
#         w = xyxy[:, 2] - xyxy[:, 0]
#         h = xyxy[:, 3] - xyxy[:, 1]

#         PLAY_MIN, BALL_MIN = 10, 4   # pixels; tune per resolution
#         is_ball = (all_detections.class_id == BALL_ID)
#         keep = ((w >= PLAY_MIN) & (h >= PLAY_MIN)) | (is_ball & (w >= BALL_MIN) & (h >= BALL_MIN))
#         all_detections = all_detections[keep]

#     # === TRACKING ===
#     all_detections = update_tracks(tracker, all_detections)

#     # === TEAM CLASSIFICATION ===
#     player_detections = all_detections[all_detections.class_id == PLAYER_ID]
#     player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]

#     if len(player_crops) > 0:
#         team_ids = team_classifier.predict(player_crops)

#         track_ids = player_detections.tracker_id
#         if track_ids is not None and len(track_ids) == len(team_ids):
#             team_ids = team_stab.update(track_ids, team_ids)
#         player_detections.class_id = team_ids
#         print(f"Frame {frame_index}: Team IDs ->", team_ids)
#     else:
#         print(f"Frame {frame_index}: No players detected")

#     # === GOALKEEPER CLASSIFICATION ===
#     goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
#     goalkeepers_team_id = team_classifier.resolve_goalkeepers_team_id(
#         player_detections, goalkeepers_detections
#     )
#     goalkeepers_detections.class_id = goalkeepers_team_id

#     # === TEAM COLORS (optional mapping, if you later want per-team colors) ===
#     TEAM_COLORS = {
#         0: sv.Color.from_hex('#FFC000'),  # Team 1 (yellow)
#         1: sv.Color.from_hex('#0000FF'),  # Team 2 (blue)
#     }

#     # === ELLIPSE ANNOTATION ON FRAME ===
#     annotated_frame = ellipse_annotator.annotate(
#         scene=frame.copy(),
#         detections=player_detections
#     )

#     # === HOMOGRAPHY ===
#     pitch_reference_points, transformer = estimate_homography(
#         frame=frame,
#         field_model=FIELD_DETECTION_MODEL,
#         config=CONFIG
#     )

#     if transformer is None:
#         # If we couldn't estimate homography, still write annotated players
#         video_writer.write(annotated_frame)
#         continue

#     # Project coordinates
#     frame_players_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#     pitch_players_xy = transformer.transform_points(frame_players_xy)

#     if len(ball_detections) > 0:
#         pitch_ball_xy = transformer.transform_points(
#             ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         )[0]
#     else:
#         pitch_ball_xy = None

#     annotated_pitch, annotated_voronoi, annotated_blended = render_pitch_views(
#         CONFIG,
#         pitch_players_xy=pitch_players_xy,
#         pitch_ball_xy=pitch_ball_xy,
#         players_class_id=player_detections.class_id
#     )

#     final_frame = overlay_inset_on_frame(
#         frame=annotated_frame,  # base = frame with ellipses
#         insets=[annotated_pitch, annotated_voronoi, annotated_blended],
#         opacity=0.5
#     )

#     # === SAVE PLAYER DATA ===
#     if len(player_detections) > 0:
#         for idx, det in enumerate(player_detections):
#             track_id = int(det.tracker_id) if det.tracker_id is not None else -1
#             team_id = int(det.class_id)
#             x_m, y_m = pitch_players_xy[idx]
#             player_data.append({
#                 "frame": frame_index,
#                 "track_id": track_id,
#                 "team_id": team_id,
#                 "x_m": x_m,
#                 "y_m": y_m
#             })

#     video_writer.write(final_frame)

# # === END OF VIDEO ===
# video_writer.release()

# # === SAVE PLAYER DATA TO CSV ===
# df = pd.DataFrame(player_data)
# csv_output_path = OUTPUT_VIDEO_PATH.replace(".mp4", ".csv")
# df.to_csv(csv_output_path, index=False)
# print("✅ Video saved to:", OUTPUT_VIDEO_PATH)
# print("✅ Player data saved to:", csv_output_path)


# import pandas as pd
# import cv2
# import numpy as np
# import supervision as sv
# from scene import CutDetector

# from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
# from Detection import detect_objects
# from Tracker import initialize_tracker, update_tracks, get_video_fps
# from Team_classification import TeamClassifier
# from Homography import estimate_homography, ViewTransformer
# from sports.configs.soccer import SoccerPitchConfiguration
# from Visualization import render_pitch_views, overlay_inset_on_frame
# from stability import TeamStability

# from tools.baseline_logger import (
#     make_run_dirs, MotWriter, seq_name_from_path, Timer,
#     save_fps_csv, save_config_json, snapshot_hardware
# )

# # === HELPERS / MODULES ===
# cut_det = CutDetector(thresh=0.40)
# team_stab = TeamStability(window=7, min_consensus=4)

# # === CONFIGURATION ===
# CONFIG = SoccerPitchConfiguration()
# SOURCE_VIDEO_PATH = r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\Video_sample\HIL-HAZ.mp4"
# OUTPUT_VIDEO_PATH = "output.mp4"

# # Class IDs (we'll debug/adjust based on model.names below)
# BALL_ID = 0
# GOALKEEPER_ID = 1
# PLAYER_ID = 2
# REFEREE_ID = 3

# RUN_ROOT = make_run_dirs("runs/baseline")
# SEQUENCE = seq_name_from_path(SOURCE_VIDEO_PATH)
# MOT_PATH = RUN_ROOT / "mot" / f"{SEQUENCE}.txt"
# FPS_CSV = RUN_ROOT / "logs" / "per_video_fps.csv"
# CFG_JSON = RUN_ROOT / "logs" / "config.json"
# HW_JSON  = RUN_ROOT / "logs" / "hardware.json"

# # === INITIALIZE MODELS ===
# from ultralytics import YOLO

# PLAYER_DETECTION_MODEL = YOLO(
#     r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\runs_main\y8s_e100\weights\best.pt"
# )
# FIELD_DETECTION_MODEL = YOLO(
#     r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\runs_field\y8s_pose_e802\weights\best.pt"
# )

# # Debug: print model class mapping so you can verify BALL_ID / PLAYER_ID / etc
# try:
#     model_names = PLAYER_DETECTION_MODEL.model.names
# except AttributeError:
#     model_names = getattr(PLAYER_DETECTION_MODEL, "names", {})
# print("🔥 Detection model class mapping:", model_names)

# # === INITIALIZE MODULES ===
# src_fps = get_video_fps(SOURCE_VIDEO_PATH)
# tracker = initialize_tracker(src_fps=src_fps)
# team_classifier = TeamClassifier(device="cpu")

# ellipse_annotator = sv.EllipseAnnotator(
#     color=sv.ColorPalette.from_hex(['#FFC000', '#0000FF', '#FFD700']),
#     thickness=2
# )
# label_annotator = sv.LabelAnnotator(
#     color=sv.ColorPalette.from_hex(['#FFC000', '#0000FF', '#FFD700']),
#     text_color=sv.Color.from_hex('#000000'),
#     text_position=sv.Position.BOTTOM_CENTER
# )
# triangle_annotator = sv.TriangleAnnotator(
#     color=sv.Color.from_hex('#00FF00'),
#     base=20,
#     height=17
# )

# print("✅ Team classifier initialized (SigLIP + UMAP + KMeans)")

# # === RE-CHECK FPS (OpenCV) ===
# _cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
# src_fps_cap = _cap.get(cv2.CAP_PROP_FPS) or src_fps or 25.0
# _cap.release()
# src_fps = src_fps_cap
# print(f"[INFO] Source FPS: {src_fps:.3f}")

# # === CLIP SETTINGS (30 seconds) ===
# CLIP_START_SEC = 0        # start at 0s (beginning of video)
# CLIP_DURATION_SEC = 30   # only process 30 seconds
# CLIP_START_FRAME = int(CLIP_START_SEC * src_fps)
# CLIP_END_FRAME = int((CLIP_START_SEC + CLIP_DURATION_SEC) * src_fps)
# print(f"[INFO] Will process frames {CLIP_START_FRAME} to {CLIP_END_FRAME} (~{CLIP_DURATION_SEC} seconds)")

# # === BASELINE CONFIG LOGGING ===
# baseline_cfg = {
#     "video": str(SOURCE_VIDEO_PATH),
#     "detector": {
#         "weights": r"runs_main\y8s_e100\weights\best.pt",
#         "conf": 0.25,
#         "nms_iou": 0.55,
#         "imgsz": 1280
#     },
#     "field_detector": {"weights": None, "conf": 0.3},
#     "classes": {
#         "BALL_ID": BALL_ID,
#         "GOALKEEPER_ID": GOALKEEPER_ID,
#         "PLAYER_ID": PLAYER_ID,
#         "REFEREE_ID": REFEREE_ID
#     },
#     "tracker": {"algo": "ByteTrack", "defaults": True},
#     "team_classifier": {
#         "model": "google/siglip-base-patch16-224",
#         "umap_components": 3,
#         "kmeans_k": 2
#     },
#     "src_fps": float(src_fps)
# }
# save_config_json(CFG_JSON, baseline_cfg)
# snapshot_hardware(HW_JSON)

# # === INITIAL TEAM CALIBRATION ===
# calibration_crops = []
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH, stride=30)
# print("🟡 Collecting initial samples for team calibration...")

# for i, frame in enumerate(frame_generator):
#     result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3)[0]
#     detections = sv.Detections.from_ultralytics(result)
#     detections = detections[detections.class_id == PLAYER_ID]
#     crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
#     calibration_crops += crops
#     if len(calibration_crops) > 100:
#         break

# team_classifier.fit(calibration_crops)

# # === VIDEO WRITER ===
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# first_frame = next(frame_generator)
# height, width = first_frame.shape[:2]
# video_writer = cv2.VideoWriter(
#     OUTPUT_VIDEO_PATH,
#     fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
#     fps=src_fps,
#     frameSize=(width, height)
# )

# # === MAIN LOOP SETUP ===
# frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# frame_index = 0
# player_data = []

# # === MAIN LOOP ===
# for frame in frame_generator:
#     frame_index += 1

#     # Clip range control
#     if frame_index < CLIP_START_FRAME:
#         continue

#     if frame_index > CLIP_END_FRAME:
#         print(f"[INFO] Reached end of clip at frame {frame_index}, stopping.")
#         break

#     # if cut_det.is_cut(frame):
#     #     tracker.reset()

#     # === DETECTION ===
#     ball_detections, all_detections = detect_objects(
#         frame=frame,
#         model=PLAYER_DETECTION_MODEL,
#         ball_id=BALL_ID
#     )

#     # Debug: what classes are present from detector?
#     if len(all_detections):
#         unique_classes = np.unique(all_detections.class_id)
#         print(f"Frame {frame_index}: class_ids={unique_classes}, num_det={len(all_detections)}")
#     else:
#         print(f"Frame {frame_index}: no detections")

#     # --- TINY-BOX FILTER (DISABLED FOR NOW WHILE DEBUGGING) ---
#     # if len(all_detections):
#     #     xyxy = all_detections.xyxy
#     #     w = xyxy[:, 2] - xyxy[:, 0]
#     #     h = xyxy[:, 3] - xyxy[:, 1]
#     #
#     #     PLAY_MIN, BALL_MIN = 10, 4   # pixels; tune per resolution
#     #     is_ball = (all_detections.class_id == BALL_ID)
#     #     keep = ((w >= PLAY_MIN) & (h >= PLAY_MIN)) | (is_ball & (w >= BALL_MIN) & (h >= BALL_MIN))
#     #     all_detections = all_detections[keep]

#     # === TRACKING ===
#     all_detections = update_tracks(tracker, all_detections, frame)

#     if len(all_detections):
#         print(f"Frame {frame_index}: tracker_ids={all_detections.tracker_id}")
#     else:
#         print(f"Frame {frame_index}: no detections after tracking")

#     # === TEAM CLASSIFICATION ===
#     player_detections = all_detections[all_detections.class_id == PLAYER_ID]
#     player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]

#     if len(player_crops) > 0:
#         team_ids = team_classifier.predict(player_crops)

#         track_ids = player_detections.tracker_id
#         if track_ids is not None and len(track_ids) == len(team_ids):
#             team_ids = team_stab.update(track_ids, team_ids)
#         player_detections.class_id = team_ids
#         print(f"Frame {frame_index}: Team IDs ->", team_ids)
#     else:
#         print(f"Frame {frame_index}: No players detected")

#     # === GOALKEEPER CLASSIFICATION ===
#     goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
#     goalkeepers_team_id = team_classifier.resolve_goalkeepers_team_id(
#         player_detections, goalkeepers_detections
#     )
#     goalkeepers_detections.class_id = goalkeepers_team_id

#     # === LABELS FOR VISUAL TRACKING ===
#     labels = []
#     for cls_id, trk_id in zip(player_detections.class_id, player_detections.tracker_id):
#         tid = int(trk_id) if trk_id is not None else -1
#         labels.append(f"id:{tid} team:{int(cls_id)}")

#     # === ANNOTATE FRAME WITH PLAYERS + LABELS ===
#     annotated_frame = ellipse_annotator.annotate(
#         scene=frame.copy(),
#         detections=player_detections
#     )
#     annotated_frame = label_annotator.annotate(
#         scene=annotated_frame,
#         detections=player_detections,
#         labels=labels
#     )

#     # === HOMOGRAPHY ===
#     pitch_reference_points, transformer = estimate_homography(
#         frame=frame,
#         field_model=FIELD_DETECTION_MODEL,
#         config=CONFIG
#     )

#     if transformer is None:
#         # No homography → still show players + IDs
#         video_writer.write(annotated_frame)
#         continue

#     # Project coordinates
#     frame_players_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#     pitch_players_xy = transformer.transform_points(frame_players_xy)

#     if len(ball_detections) > 0:
#         pitch_ball_xy = transformer.transform_points(
#             ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
#         )[0]
#     else:
#         pitch_ball_xy = None

#     annotated_pitch, annotated_voronoi, annotated_blended = render_pitch_views(
#         CONFIG,
#         pitch_players_xy=pitch_players_xy,
#         pitch_ball_xy=pitch_ball_xy,
#         players_class_id=player_detections.class_id
#     )

#     final_frame = overlay_inset_on_frame(
#         frame=annotated_frame,  # annotated base frame
#         insets=[annotated_pitch, annotated_voronoi, annotated_blended],
#         opacity=0.5
#     )

#     # === SAVE PLAYER DATA ===
#     if len(player_detections) > 0:
#         for idx, det in enumerate(player_detections):
#             track_id = int(det.tracker_id) if det.tracker_id is not None else -1
#             team_id = int(det.class_id)
#             x_m, y_m = pitch_players_xy[idx]
#             player_data.append({
#                 "frame": frame_index,
#                 "track_id": track_id,
#                 "team_id": team_id,
#                 "x_m": x_m,
#                 "y_m": y_m
#             })

#     video_writer.write(final_frame)

# # === END OF VIDEO ===
# video_writer.release()

# # === SAVE PLAYER DATA TO CSV ===
# df = pd.DataFrame(player_data)
# csv_output_path = OUTPUT_VIDEO_PATH.replace(".mp4", ".csv")
# df.to_csv(csv_output_path, index=False)
# print("✅ Video saved to:", OUTPUT_VIDEO_PATH)

# print("✅ Player data saved to:", csv_output_path)


import pandas as pd
import cv2
import numpy as np
import supervision as sv
from scene import CutDetector

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from Detection import detect_objects
from Tracker import initialize_tracker, update_tracks, get_video_fps
from Team_classification import TeamClassifier
from Homography import estimate_homography, ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from Visualization import render_pitch_views, overlay_inset_on_frame
from stability import TeamStability

from tools.baseline_logger import (
    make_run_dirs, MotWriter, seq_name_from_path, Timer,
    save_fps_csv, save_config_json, snapshot_hardware
)

# === HELPERS / MODULES ===
cut_det = CutDetector(thresh=0.40)
team_stab = TeamStability(window=7, min_consensus=4)

# === CONFIGURATION ===
CONFIG = SoccerPitchConfiguration()
SOURCE_VIDEO_PATH = r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\Video_sample\HIL-HAZ.mp4"
OUTPUT_VIDEO_PATH = "output.mp4"

# Class IDs
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

RUN_ROOT = make_run_dirs("runs/baseline")
SEQUENCE = seq_name_from_path(SOURCE_VIDEO_PATH)
MOT_PATH = RUN_ROOT / "mot" / f"{SEQUENCE}.txt"
FPS_CSV = RUN_ROOT / "logs" / "per_video_fps.csv"
CFG_JSON = RUN_ROOT / "logs" / "config.json"
HW_JSON  = RUN_ROOT / "logs" / "hardware.json"

# === INITIALIZE MODELS ===
from ultralytics import YOLO

PLAYER_DETECTION_MODEL = YOLO(
    r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\runs_main\y8s_e100\weights\best.pt"
)
FIELD_DETECTION_MODEL = YOLO(
    r"C:\Users\User\Downloads\Chimdi\Football detection\Football_test\runs_field\y8s_pose_e802\weights\best.pt"
)

# Debug: print model class mapping
try:
    model_names = PLAYER_DETECTION_MODEL.model.names
except AttributeError:
    model_names = getattr(PLAYER_DETECTION_MODEL, "names", {})
print("🔥 Detection model class mapping:", model_names)

# === INITIALIZE MODULES ===
src_fps = get_video_fps(SOURCE_VIDEO_PATH)

# ✅ Two trackers: humans + ball
human_tracker = initialize_tracker(src_fps=src_fps, profile="human")
ball_tracker  = initialize_tracker(src_fps=src_fps, profile="ball")

team_classifier = TeamClassifier(device="cpu")

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
    base=20,
    height=17
)

print("✅ Team classifier initialized (SigLIP + UMAP + KMeans)")

# === RE-CHECK FPS (OpenCV) ===
_cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
src_fps_cap = _cap.get(cv2.CAP_PROP_FPS) or src_fps or 25.0
_cap.release()
src_fps = src_fps_cap
print(f"[INFO] Source FPS: {src_fps:.3f}")

# === CLIP SETTINGS ===
CLIP_START_SEC = 0
CLIP_DURATION_SEC = 5
CLIP_START_FRAME = int(CLIP_START_SEC * src_fps)
CLIP_END_FRAME = int((CLIP_START_SEC + CLIP_DURATION_SEC) * src_fps)
print(f"[INFO] Will process frames {CLIP_START_FRAME} to {CLIP_END_FRAME} (~{CLIP_DURATION_SEC} seconds)")

# === BASELINE CONFIG LOGGING ===
baseline_cfg = {
    "video": str(SOURCE_VIDEO_PATH),
    "detector": {
        "weights": r"runs_main\y8s_e100\weights\best.pt",
        "conf": 0.25,
        "nms_iou": 0.55,
        "imgsz": 1280
    },
    "field_detector": {"weights": None, "conf": 0.3},
    "classes": {
        "BALL_ID": BALL_ID,
        "GOALKEEPER_ID": GOALKEEPER_ID,
        "PLAYER_ID": PLAYER_ID,
        "REFEREE_ID": REFEREE_ID
    },
    # ✅ Update tracker info
    "tracker": {"algo": "BoT-SORT", "gmc": True, "ball_tracker": True, "reid": False},
    "team_classifier": {
        "model": "google/siglip-base-patch16-224",
        "umap_components": 3,
        "kmeans_k": 2
    },
    "src_fps": float(src_fps)
}
save_config_json(CFG_JSON, baseline_cfg)
snapshot_hardware(HW_JSON)

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
    fps=src_fps,
    frameSize=(width, height)
)

# === MAIN LOOP SETUP ===
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
frame_index = 0
player_data = []

# === MAIN LOOP ===
for frame in frame_generator:
    frame_index += 1

    if frame_index < CLIP_START_FRAME:
        continue

    if frame_index > CLIP_END_FRAME:
        print(f"[INFO] Reached end of clip at frame {frame_index}, stopping.")
        break

    # === DETECTION ===
    ball_detections, all_detections = detect_objects(
        frame=frame,
        model=PLAYER_DETECTION_MODEL,
        ball_id=BALL_ID
    )

    # Debug: what classes are present from detector?
    if len(all_detections):
        unique_classes = np.unique(all_detections.class_id)
        print(f"Frame {frame_index}: class_ids={unique_classes}, num_det={len(all_detections)}")
    else:
        print(f"Frame {frame_index}: no detections")

    # === TRACKING ===
    # Track humans
    all_detections = update_tracks(human_tracker, all_detections, frame)
    # Track ball separately
    ball_detections = update_tracks(ball_tracker, ball_detections, frame)

    if len(all_detections):
        print(f"Frame {frame_index}: human_tracker_ids={all_detections.tracker_id}")
    else:
        print(f"Frame {frame_index}: no human detections after tracking")

    # === SPLIT DETECTIONS ===
    player_detections       = all_detections[all_detections.class_id == PLAYER_ID]
    referee_detections      = all_detections[all_detections.class_id == REFEREE_ID]
    goalkeepers_detections  = all_detections[all_detections.class_id == GOALKEEPER_ID]

    # === TEAM CLASSIFICATION (PLAYERS) ===
    player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]

    if len(player_crops) > 0:
        team_ids = team_classifier.predict(player_crops)

        track_ids = player_detections.tracker_id
        if track_ids is not None and len(track_ids) == len(team_ids):
            team_ids = team_stab.update(track_ids, team_ids)

        # overwrite class_id for players with team_id (0/1)
        player_detections.class_id = team_ids
        print(f"Frame {frame_index}: Team IDs ->", team_ids)
    else:
        print(f"Frame {frame_index}: No players detected")

    # === GOALKEEPER TEAM RESOLUTION ===
    if len(goalkeepers_detections) > 0 and len(player_detections) > 0:
        goalkeepers_team_id = team_classifier.resolve_goalkeepers_team_id(
            player_detections, goalkeepers_detections
        )
        goalkeepers_detections.class_id = goalkeepers_team_id

    # === LABELS ===
    labels = []
    for cls_id, trk_id in zip(player_detections.class_id, player_detections.tracker_id):
        tid = int(trk_id) if trk_id is not None else -1
        labels.append(f"id:{tid} team:{int(cls_id)}")

    # === ANNOTATE PLAYERS ===
    annotated_frame = ellipse_annotator.annotate(
        scene=frame.copy(),
        detections=player_detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=player_detections,
        labels=labels
    )

    # === ANNOTATE GOALKEEPERS ===
    if len(goalkeepers_detections) > 0:
        gk_labels = []
        for cls_id, trk_id in zip(goalkeepers_detections.class_id, goalkeepers_detections.tracker_id):
            tid = int(trk_id) if trk_id is not None else -1
            gk_labels.append(f"gk:{tid} team:{int(cls_id)}")

        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=goalkeepers_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=goalkeepers_detections,
            labels=gk_labels
        )

    # === ANNOTATE REFEREES ===
    if len(referee_detections) > 0:
        ref_labels = []
        for trk_id in referee_detections.tracker_id:
            tid = int(trk_id) if trk_id is not None else -1
            ref_labels.append(f"ref:{tid}")

        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=referee_detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=referee_detections,
            labels=ref_labels
        )

    # === ANNOTATE BALL (no ID text) ===
    if len(ball_detections) > 0:
        annotated_frame = triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections
        )

    # === HOMOGRAPHY ===
    pitch_reference_points, transformer = estimate_homography(
        frame=frame,
        field_model=FIELD_DETECTION_MODEL,
        config=CONFIG
    )

    if transformer is None:
        video_writer.write(annotated_frame)
        continue

    # Project coordinates (players only for pitch plot)
    frame_players_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_players_xy = transformer.transform_points(frame_players_xy)

    # Use tracked ball for pitch projection
    if len(ball_detections) > 0:
        pitch_ball_xy = transformer.transform_points(
            ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        )[0]
    else:
        pitch_ball_xy = None

    annotated_pitch, annotated_voronoi, annotated_blended = render_pitch_views(
        CONFIG,
        pitch_players_xy=pitch_players_xy,
        pitch_ball_xy=pitch_ball_xy,
        players_class_id=player_detections.class_id
    )

    final_frame = overlay_inset_on_frame(
        frame=annotated_frame,
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
