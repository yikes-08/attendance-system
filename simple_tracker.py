# simple_tracker.py
import time
import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    union = boxAArea + boxBArea - interArea + 1e-9
    return interArea / union

class Track:
    def __init__(self, bbox, track_id):
        self.bbox = bbox  # x,y,w,h
        self.id = track_id
        self.last_seen = time.time()
        self.age = 0
        self.recognized = False   # whether final attendance decision already made
        self.name = None
        self.sim = 0.0
        self.votes = []           # list of similarity scores over recent recognitions

class SimpleTracker:
    """
    Minimal short-term tracker:
    - Matches detections to tracks using IOU
    - Creates new tracks for unmatched detections
    - Removes stale tracks after max_idle seconds
    """
    def __init__(self, iou_thresh=0.3, max_idle=2.0):
        self.tracks = {}      # id -> Track
        self.next_id = 0
        self.iou_thresh = iou_thresh
        self.max_idle = max_idle

    def update(self, detections):
        """
        detections: list of dicts with 'bbox' key (x,y,w,h) and optional 'confidence'
        Returns: list of (track, detection) pairs after matching/creation.
        """
        matches = []
        assigned = set()

        # simple greedy matching by best IOU
        for det in detections:
            best_t = None
            best_iou = 0.0
            for t_id, track in self.tracks.items():
                if t_id in assigned:
                    continue
                score = iou(det['bbox'], track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_t = track
            if best_t is not None and best_iou >= self.iou_thresh:
                # update track
                best_t.bbox = det['bbox']
                best_t.last_seen = time.time()
                best_t.age += 1
                matches.append((best_t, det))
                assigned.add(best_t.id)
            else:
                # create new track
                t = Track(det['bbox'], self.next_id)
                self.tracks[self.next_id] = t
                self.next_id += 1
                matches.append((t, det))
                assigned.add(t.id)

        # cleanup stale tracks
        now = time.time()
        stale = [tid for tid, tr in list(self.tracks.items()) if now - tr.last_seen > self.max_idle]
        for tid in stale:
            del self.tracks[tid]

        return matches
