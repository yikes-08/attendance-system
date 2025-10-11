import cv2
import os
import sys
import sqlite3
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from config import DATABASE_PATH, ATTENDANCE_COOLDOWN

class VideoTester:
    def __init__(self):
        print("[INFO] Initializing video tester...")
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.known_faces = self.load_known_faces()
        self.recognizer.load_known_faces_from_database(self.known_faces)
        self.attendance_records = []
        self.last_seen = {}

        if not self.known_faces:
            print("⚠️ [WARNING] No enrolled faces found in the database. Please run `python enroll_dataset.py` first.")
        else:
            print(f"✅ Loaded {len(self.known_faces)} known faces from database.")

    def load_known_faces(self):
        conn = sqlite3.connect(DATABASE_PATH)
        cur = conn.cursor()
        cur.execute("SELECT id, name, encoding FROM registered_faces")
        rows = cur.fetchall()
        conn.close()

        known_faces = {}
        for pid, name, enc_bytes in rows:
            encoding = np.frombuffer(enc_bytes, dtype=np.float32)
            known_faces[pid] = {"name": name, "encoding": encoding}
        return known_faces

    def process_video(self, video_path, output_dir="attendance_reports"):
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("❌ Cannot open video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="🎬 Processing Video", unit="frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                self.process_frame(frame)
                pbar.update(1)

        cap.release()
        self.save_report(output_dir)
        print("\n✅ Video processing complete!")

    def process_frame(self, frame):
        faces = self.detector.detect_faces(frame)
        for face in faces:
            result, conf = self.recognizer.recognize_face(face)
            if result:
                pid = result["person_id"]
                name = result["person_name"]
                self.mark_attendance(pid, name, conf)

    def mark_attendance(self, pid, name, conf):
        now = datetime.now()
        if pid in self.last_seen and (now - self.last_seen[pid]).total_seconds() < ATTENDANCE_COOLDOWN:
            return
        self.last_seen[pid] = now
        self.attendance_records.append({
            "PersonID": pid,
            "Name": name,
            "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "Confidence": round(conf, 3)
        })

    def save_report(self, output_dir):
        if not self.attendance_records:
            print("🤷 No attendance records detected.")
            return

        df = pd.DataFrame(self.attendance_records)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = os.path.join(output_dir, f"Video_Attendance_Report_{timestamp}.csv")
        df.to_csv(file_path, index=False)

        print(f"📊 Saved attendance report: {file_path}")
        print(f"  → Total Records: {len(df)} | Unique People: {df['Name'].nunique()}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]
    tester = VideoTester()
    tester.process_video(video_path)
