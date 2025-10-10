import os

# Database configuration
DATABASE_PATH = "attendance.db"

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "sahuyash160@gmail.com"  # Change this to your email
EMAIL_PASSWORD = "mrvg yqvp gxvp sxyb"    # Change this to your app password

# Face detection and recognition settings
FACE_DETECTION_CONFIDENCE = 0.5
FACE_RECOGNITION_THRESHOLD = 0.6
ATTENDANCE_COOLDOWN = 30  # seconds between attendance marks for same person

# Camera settings
CAMERA_INDEX = 0  # Default camera index
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Model paths
YUNET_MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
FACENET_MODEL_PATH = "models/facenet_keras.h5"

# Attendance settings
ATTENDANCE_EMAIL_INTERVAL = 60  # Send email every 60 minutes
CSV_FILENAME = "attendance_records.csv"
