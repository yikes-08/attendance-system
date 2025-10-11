import os
import onnxruntime

# Database configuration
DATABASE_PATH = "attendance.db"

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "unown20th@gmail.com"  # Change this to your email
EMAIL_PASSWORD = "dogdooeklekeunown"    # Change this to your app password

# Face detection and recognition settings
FACE_DETECTION_CONFIDENCE = 0.45        # detection confidence threshold
FACE_RECOGNITION_THRESHOLD = 0.45      # cosine similarity threshold (ArcFace embeddings)
ATTENDANCE_COOLDOWN = 30               # seconds between attendance marks for same person

# InsightFace settings
if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
    INSIGHTFACE_PROVIDER = ['CUDAExecutionProvider']
    print("✅ [INFO] GPU is available. Using CUDAExecutionProvider for InsightFace.")
else:
    INSIGHTFACE_PROVIDER = ['CPUExecutionProvider']
    print("⚠️ [INFO] GPU not found. Using CPUExecutionProvider for InsightFace.")
INSIGHTFACE_DET_SIZE = (640, 640)
INSIGHTFACE_MODEL_NAME = 'buffalo_l'   # high-quality, reasonable speed; InsightFace will auto-download

# Database / paths
DATABASE_PATH = "attendance.db"
CSV_FILENAME = os.path.join("attendance_reports", "attendance_records.csv")
os.makedirs("attendance_reports", exist_ok=True)

# Camera settings
CAMERA_INDEX = 0  # Default camera index
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Attendance settings
ATTENDANCE_EMAIL_INTERVAL = 60  # Send email every 60 minutes