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
    def __init__(self, bbox, track_id, frame_shape):
        self.id = track_id
        self.last_seen = time.time()
        self.age = 0
        self.recognized = False
        self.name = None
        self.sim = 0.0
        self.votes = []
        # --- ROI --- Store frame dimensions to clamp ROI coordinates
        self.frame_h, self.frame_w = frame_shape[:2]
        self.update_bbox(bbox) # Use a method to set both bbox and ROI

    def update_bbox(self, bbox):
        """Updates the bounding box and recalculates the ROI."""
        self.bbox = bbox
        self.roi = self._calculate_roi(bbox)

    def _calculate_roi(self, bbox, padding_factor=0.2):
        """Calculates an expanded Region of Interest around the bbox."""
        x, y, w, h = bbox
        pad_w = int(w * padding_factor)
        pad_h = int(h * padding_factor)
        
        # Calculate expanded coordinates and clamp to frame boundaries
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(self.frame_w, x + w + pad_w)
        y2 = min(self.frame_h, y + h + pad_h)
        
        return (x1, y1, x2, y2)

class SimpleTracker:
    def __init__(self, iou_thresh=0.3, max_idle=2.0):
        self.tracks = {}
        self.next_id = 0
        self.iou_thresh = iou_thresh
        self.max_idle = max_idle

    def update(self, detections, frame_shape):
        """
        detections: list of dicts with 'bbox' key
        frame_shape: (height, width) of the frame, needed for new tracks
        Returns: list of (track, detection) pairs
        """
        matches = []
        assigned = set()

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
                best_t.update_bbox(det['bbox']) # Update both bbox and ROI
                best_t.last_seen = time.time()
                best_t.age += 1
                matches.append((best_t, det))
                assigned.add(best_t.id)
            else:
                # --- ROI --- Pass frame_shape when creating a new track
                t = Track(det['bbox'], self.next_id, frame_shape)
                self.tracks[self.next_id] = t
                self.next_id += 1
                matches.append((t, det))
                assigned.add(t.id)

        now = time.time()
        stale = [tid for tid, tr in list(self.tracks.items()) if now - tr.last_seen > self.max_idle]
        for tid in stale:
            del self.tracks[tid]

        return matches

    def get_combined_roi(self):
        """
        Combines ROIs of all active tracks into one large ROI.
        Returns (x1, y1, x2, y2) or None if no tracks exist.
        """
        if not self.tracks:
            return None

        min_x1 = float('inf')
        min_y1 = float('inf')
        max_x2 = 0
        max_y2 = 0

        for track in self.tracks.values():
            x1, y1, x2, y2 = track.roi
            min_x1 = min(min_x1, x1)
            min_y1 = min(min_y1, y1)
            max_x2 = max(max_x2, x2)
            max_y2 = max(max_y2, y2)
        
        return (int(min_x1), int(min_y1), int(max_x2), int(max_y2))