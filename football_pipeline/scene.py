import cv2, numpy as np

class CutDetector:
    def __init__(self, thresh: float = 0.40):
        self.prev = None
        self.thresh = thresh

    def is_cut(self, frame) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 36))
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / (hist.sum() + 1e-6)
        if self.prev is None:
            self.prev = hist
            return False
        diff = np.abs(hist - self.prev).sum()
        self.prev = hist
        return diff > self.thresh
