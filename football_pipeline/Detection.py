import supervision as sv
def detect_objects(frame, model, ball_id=0, conf=0.3):
    """
    Run object detection on a frame and separate detections into
    ball and all other detections (players, referees, goalkeepers).
    """
    # Run the model
    result = model.predict(frame, conf=conf)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Ball detections
    ball_detections = detections[detections.class_id == ball_id]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    # All other detections (players, referees, goalkeepers)
    all_detections = detections[detections.class_id != ball_id]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)

    return ball_detections, all_detections