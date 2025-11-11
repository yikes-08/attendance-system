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
from config import DATABASE_PATH, ATTENDANCE_COOLDOWN, FACE_RECOGNITION_THRESHOLD
from db_init import ensure_registered_faces_table

class VideoTester:
    def __init__(self, use_faiss=False):
        print("[INFO] Initializing video tester...")
        try:
            import onnxruntime as ort
            has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
        except Exception:
            has_cuda = False

        # Ensure database table exists before proceeding (only for registered faces)
        ensure_registered_faces_table()

        self.detector = FaceDetector(use_gpu=has_cuda)
        self.recognizer = FaceRecognizer(use_gpu=has_cuda, use_faiss=use_faiss)
        self.tracker = SimpleTracker(iou_thresh=0.3, max_idle=2.0)
        
        # Generate a unique CSV for the background writer
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        db_writer_csv_path = os.path.join("attendance_reports", f"Video_Processing_Log_{timestamp}.csv")
        
        # Pass the unique path to the DBWriter
        self.db_writer = DBWriter(csv_path=db_writer_csv_path)
        self.attendance_records = []
        self.last_seen = {}

        # Load known faces
        self.known_faces = self.load_known_faces()
        self.recognizer.load_known_faces_from_database(self.known_faces)

        if not self.known_faces:
            print("⚠️ [WARNING] No enrolled faces found. Please run enrollment first.")
        else:
            print(f"✅ Loaded {len(self.known_faces)} known faces from database.")

        # Recognition tuning
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
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup for saving annotated video output
        output_video_path = os.path.join(output_dir, f"annotated_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        with tqdm(total=total_frames, desc="🎬 Processing Video", unit="frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.detect_faces(frame)
                
                # Pass frame.shape to the tracker update method
                matches = self.tracker.update(detections, frame.shape)

                for track, det in matches:
                    x, y, w, h = det['bbox']
                    color = (0, 255, 0) if getattr(track, "recognized", False) else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    label = f"ID {track.id}: " + (track.name if getattr(track, "name", None) else "Unknown")
                    conf = getattr(track, "sim", 0.0)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x, max(15, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Recognition logic
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
                                now = datetime.now()
                                last = self.last_seen.get(pid)
                                if not last or (now - last).total_seconds() >= ATTENDANCE_COOLDOWN:
                                    self.last_seen[pid] = now
                                    rec = {
                                        "PersonID": pid,
                                        "Name": track.name,
                                        "Timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                                        "Confidence": round(float(track.sim), 3)
                                    }
                                    self.attendance_records.append(rec)
                                    self.db_writer.enqueue(pid, track.name, rec["Timestamp"], rec["Confidence"])
                                    track.recognized = True

                out_writer.write(frame)
                cv2.imshow("Test Video - Face Recognition", frame) # Optional: comment out for faster processing
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n🛑 Video processing stopped manually.")
                    break

                pbar.update(1)

        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()
        self.db_writer.stop()
        self.save_report(output_dir)
        print(f"\n✅ Video processing complete!")
        print(f"📹 Annotated output saved to: {output_video_path}")

    def save_report(self, output_dir):
        if not self.attendance_records:
            print("🤷 No attendance records were detected.")
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