import cv2
path = r"C:\Users\Chimdi\Downloads\Football detection\Football_test\Video_sample\HIL-HAZ.mp4"
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    raise RuntimeError(f"Could not open {path}")

fps = cap.get(cv2.CAP_PROP_FPS)
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"FPS: {fps}")
print(f"Frames: {frames}")
print(f"Resolution: {w}x{h}")
if fps > 0:
    print(f"Duration (s): {frames / fps:.2f}")
