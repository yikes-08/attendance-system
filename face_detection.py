import cv2
import numpy as np
from config import YUNET_MODEL_PATH, FACE_DETECTION_CONFIDENCE

class FaceDetector:
    def __init__(self):
        self.detector = None
        self.load_model()
    
    def load_model(self):
        """Load YuNet face detection model"""
        try:
            # Download YuNet model if not exists
            import urllib.request
            import os
            
            if not os.path.exists(YUNET_MODEL_PATH):
                os.makedirs(os.path.dirname(YUNET_MODEL_PATH), exist_ok=True)
                print("Downloading YuNet model...")
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                    YUNET_MODEL_PATH
                )
            
            # Initialize YuNet detector
            self.detector = cv2.FaceDetectorYN.create(
                YUNET_MODEL_PATH,
                "",
                (320, 320),
                FACE_DETECTION_CONFIDENCE,
                0.3,
                5000
            )
            print("YuNet model loaded successfully")
        except Exception as e:
            print(f"Error loading YuNet model: {e}")
            # Fallback to Haar Cascade
            self.detector = None
            self.haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_faces(self, frame):
        """Detect faces in the given frame"""
        if self.detector is not None:
            # Use YuNet detector
            height, width = frame.shape[:2]
            self.detector.setInputSize((width, height))
            
            _, faces = self.detector.detect(frame)
            
            if faces is not None:
                face_boxes = []
                for face in faces:
                    # YuNet returns [x, y, w, h, confidence, landmarks...]
                    x, y, w, h, confidence = face[:5]
                    if confidence > FACE_DETECTION_CONFIDENCE:
                        face_boxes.append({
                            'bbox': (int(x), int(y), int(w), int(h)),
                            'confidence': confidence,
                            'landmarks': face[5:15].reshape(5, 2) if len(face) > 15 else None
                        })
                return face_boxes
        else:
            # Fallback to Haar Cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.haar_cascade.detectMultiScale(gray, 1.1, 4)
            
            face_boxes = []
            for (x, y, w, h) in faces:
                face_boxes.append({
                    'bbox': (x, y, w, h),
                    'confidence': 1.0,
                    'landmarks': None
                })
            return face_boxes
        
        return []
    
    def extract_face_region(self, frame, face_box):
        """Extract face region from frame"""
        x, y, w, h = face_box['bbox']
        # Add some padding around the face
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_region = frame[y1:y2, x1:x2]
        return face_region
