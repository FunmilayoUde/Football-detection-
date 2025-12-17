from collections import deque, defaultdict
import numpy as np

class TeamStability:
    def __init__(self, window:int=7, min_consensus:int=4):
        """
        window: how many recent labels to keep per track
        min_consensus: require at least this many frames before switching to majority
        """
        self.buffers = defaultdict(lambda: deque(maxlen=window))
        self.min_consensus = min_consensus

    def update(self, track_ids, team_ids):
        """
        track_ids: array-like of tracker ids (ints or None)
        team_ids:  array-like of raw predicted team labels (0/1)
        returns: np.array of stabilized labels
        """
        stable = []
        for tid, tlabel in zip(track_ids, team_ids):
            if tid is None or tid < 0:
                stable.append(int(tlabel))
                continue
            buf = self.buffers[int(tid)]
            buf.append(int(tlabel))
            vals, counts = np.unique(list(buf), return_counts=True)
            maj = int(vals[counts.argmax()])
            if len(buf) >= self.min_consensus:
                stable.append(maj)
            else:
                stable.append(int(tlabel))
        return np.array(stable, dtype=int)