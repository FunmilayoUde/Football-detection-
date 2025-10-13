import supervision as sv

def initialize_tracker():
    """
    Initialize and return a new ByteTrack tracker.
    """
    tracker = sv.ByteTrack()
    tracker.reset()
    return tracker


def update_tracks(tracker, detections):
    """
    Update tracker with the current frame's detections.
    Returns updated detections with tracker IDs.
    """
    tracked_detections = tracker.update_with_detections(detections=detections)
    return tracked_detections