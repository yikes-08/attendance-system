# test_video.py
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
from simple_tracker import SimpleTracker
from db_writer import DBWriter
from config import DATABASE_PATH, ATTENDANCE_COOLDOWN, CSV_FILENAME, FACE_RECOGNITION_THRESHOLD

class VideoTester:
    def __init__(self, use_faiss=False):
        print("[INFO] Initializing video tester (optimized)...")
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer(use_gpu=('CUDAExecutionProvider' in getattr(__import__('onnxruntime'), 'get_available_providers', lambda: [])()), use_faiss=use_faiss)
        self.tracker = SimpleTracker(iou_thresh=0.3, max_idle=2.0)
        self.db_writer = DBWriter()
        self.attendance_records = []
        self.last_seen = {}

        # load known faces from DB (robust)
        self.known_faces = self.load_known_faces()
        self.recognizer.load_known_faces_from_database(self.known_faces)

        if not self.known_faces:
            print("⚠️ [WARNING] No enrolled faces found in the database. Please run `python enroll_dataset.py` first.")
        else:
            print(f"✅ Loaded {len(self.known_faces)} known faces from database.")

        # recognition tuning (same as attendance_system)
        self.RECOG_PERIOD = 3
        self.VOTE_WINDOW = 3
        self.REQUIRED_VOTES = 2

    def load_known_faces(self):
        conn = sqlite3.connect(DATABASE_PATH)
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, name, encoding FROM registered_faces")
            rows = cur.fetchall()
        except Exception:
            rows = []
        conn.close()

        known_faces = {}
        for pid, name, enc_bytes in rows:
            known_faces[pid] = {"name": name, "encoding": enc_bytes}
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

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        with tqdm(total=total_frames, desc="🎬 Processing Video", unit="frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.detect_faces(frame)
                matches = self.tracker.update(detections)

                for track, det in matches:
                    do_recog = (not track.recognized) or (track.age % self.RECOG_PERIOD == 0)
                    if do_recog:
                        result, sim = self.recognizer.recognize_face(det)
                        track.votes.append(sim)
                        if len(track.votes) > self.VOTE_WINDOW:
                            track.votes = track.votes[-self.VOTE_WINDOW:]

                        if result:
                            track.name = result["person_name"]
                            track.sim = sim
                        else:
                            track.sim = sim

                        positive_votes = sum(1 for v in track.votes if v >= FACE_RECOGNITION_THRESHOLD)
                        if positive_votes >= self.REQUIRED_VOTES and track.name is not None:
                            pid = result["person_id"] if result else None
                            if pid is not None:
                                # maintain cooldown in-memory per-video
                                now = datetime.now()
                                last = self.last_seen.get(pid)
                                if not last or (now - last).total_seconds() >= ATTENDANCE_COOLDOWN:
                                    self.last_seen[pid] = now
                                    # append to local records (for report)
                                    rec = {
                                        "PersonID": pid,
                                        "Name": track.name,
                                        "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                                        "Confidence": round(float(track.sim), 3)
                                    }
                                    self.attendance_records.append(rec)
                                    # also enqueue to persistent DB/CSV
                                    self.db_writer.enqueue(pid, track.name, rec["Timestamp"], rec["Confidence"])
                                    track.recognized = True

                pbar.update(1)

        cap.release()
        self.db_writer.stop()
        self.save_report(output_dir)
        print("\n✅ Video processing complete!")

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
