import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceDetector:
    """
    RetinaFace-based detector + aligner using InsightFace.
    Detects and returns cropped/aligned face info objects with bounding boxes.
    """

    def __init__(self, det_size=(640, 640), use_gpu=True):
        """
        det_size: (width, height) for RetinaFace model
        use_gpu: True → CUDAExecutionProvider, else CPU
        """
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)

    def detect_faces(self, image):
        """
        Detect faces and return list of dicts:
        [
            {'bbox': (x, y, w, h),
             'confidence': float,
             'face_obj': Face object from InsightFace}
        ]
        """
        if image is None or image.size == 0:
            print("⚠️ Warning: Empty image passed to FaceDetector.")
            return []

        faces = self.app.get(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # convert to RGB for InsightFace
        detections = []

        for f in faces:
            x1, y1, x2, y2 = map(int, f.bbox)
            w, h = x2 - x1, y2 - y1
            detections.append({
                "bbox": (x1, y1, w, h),
                "confidence": float(f.det_score),
                "face_obj": f
            })

        # Optional debug log
        print(f"Detected {len(detections)} faces")
        return detections
