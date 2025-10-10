import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    """
    RetinaFace-based detector + aligner using InsightFace.
    Detects faces and returns cropped/aligned face images plus bounding boxes.
    """

    def __init__(self, det_size=(640, 640), ctx_id=0):
        """
        ctx_id = 0 → GPU,  -1 → CPU
        det_size sets RetinaFace input resolution.
        """
        self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect_faces(self, image):
        """
        Detect faces → return list of dicts:
        [{'bbox': (x, y, w, h), 'confidence': float, 'face_obj': face}]
        """
        faces = self.app.get(image)
        detections = []
        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            w, h = x2 - x1, y2 - y1
            detections.append({
                "bbox": (x1, y1, w, h),
                "confidence": float(f.det_score),
                "face_obj": f
            })
        return detections
