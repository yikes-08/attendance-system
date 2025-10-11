# attendance_system.py
import cv2
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import onnxruntime as ort
import time
import os

from config import (
    DATABASE_PATH,
    CSV_FILENAME,
    FACE_RECOGNITION_THRESHOLD,
    ATTENDANCE_COOLDOWN
)
from face_detection import FaceDetector
from face_recognition import FaceRecognizer


class AttendanceSystem:
    def __init__(self):
        # Auto-detect GPU
        self.use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
        print(f"⚙️ Initializing in {'GPU' if self.use_gpu else 'CPU'} mode")

        self.detector = FaceDetector(use_gpu=self.use_gpu)
        self.recognizer = FaceRecognizer()

        self.known_faces = self.load_known_faces()
        self.recognizer.load_known_faces_from_database(self.known_faces)

        self.attendance_log = {}
        print(f"[INFO] Loaded {len(self.known_faces)} enrolled faces ✅")

    def load_known_faces(self):
        """Load stored embeddings from SQLite"""
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

    def mark_attendance(self, person_id, person_name, confidence):
        """Mark attendance in DB and CSV, with cooldown control"""
        now = datetime.now()
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%Y-%m-%d")

        # Prevent duplicates within cooldown period
        last_entry = self.attendance_log.get(person_id)
        if last_entry and (now - last_entry).total_seconds() < ATTENDANCE_COOLDOWN:
            return

        self.attendance_log[person_id] = now
        print(f"[MARKED] {person_name} @ {time_str} ({confidence:.2f})")

        conn = sqlite3.connect(DATABASE_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                name TEXT,
                date TEXT,
                time TEXT,
                confidence REAL
            )
        """)
        cur.execute("INSERT INTO attendance (person_id, name, date, time, confidence) VALUES (?, ?, ?, ?, ?)",
                    (person_id, person_name, date_str, time_str, confidence))
        conn.commit()
        conn.close()

        # Append to CSV safely
        row = pd.DataFrame([[person_id, person_name, date_str, time_str, confidence]],
                           columns=["PersonID", "Name", "Date", "Time", "Confidence"])
        file_exists = os.path.exists(CSV_FILENAME)
        row.to_csv(CSV_FILENAME, mode='a', header=not file_exists, index=False)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return

        print("🎥 Starting Real-Time Attendance System")
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            faces = self.detector.detect_faces(frame)

            # Process every frame (can adjust to skip for FPS boost)
            for face in faces:
                result, conf = self.recognizer.recognize_face(face)
                x, y, w, h = face['bbox']

                if result:
                    name = result["person_name"]
                    pid = result["person_id"]
                    self.mark_attendance(pid, name, conf)
                    color = (0, 255, 0)
                    text = f"{name} ({conf:.2f})"
                else:
                    color = (0, 0, 255)
                    text = f"Unknown ({conf:.2f})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # FPS counter
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Face Attendance", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("🟢 Attendance session ended")


if __name__ == "__main__":
    system = AttendanceSystem()
    system.run()
