import cv2
from face_detection import FaceDetector
from config import INSIGHTFACE_PROVIDER

det = FaceDetector()
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    print("No camera frame")
    exit(1)

faces = det.detect_faces(frame)
print(f"Detected {len(faces)} faces")
for f in faces:
    print("bbox:", f['bbox'], "conf:", f['confidence'])
    # print whether face_obj has embedding
    face_obj = f.get('face_obj')
    print("embedding present:", hasattr(face_obj, 'embedding') and face_obj.embedding is not None)

cap.release()
