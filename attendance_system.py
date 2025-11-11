# attendance_system.py
import cv2
import sqlite3
import numpy as np
from datetime import datetime
import onnxruntime as ort
import time
import os
import pandas as pd
import glob

from config import (
    DATABASE_PATH,
    FACE_RECOGNITION_THRESHOLD,
    ATTENDANCE_COOLDOWN
)
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from simple_tracker import SimpleTracker
from db_writer import DBWriter
from realsense_camera import RealSenseCamera  # Import the new camera class
from db_init import ensure_registered_faces_table

class DynamicFrameSkipper:
    def __init__(self, fast_rate=2, slow_rate=5):
        self.fast_rate = fast_rate
        self.slow_rate = slow_rate
        self.tracking_active = False
        self.frame_counter = 0

    def update_status(self, tracking_active):
        self.tracking_active = tracking_active

    def should_process(self):
        self.frame_counter += 1
        rate = self.slow_rate if self.tracking_active else self.fast_rate
        return self.frame_counter % rate == 0

class AttendanceSystem:
    def __init__(self, use_faiss=False):
        self.use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
        print(f"⚙️ Initializing in {'GPU' if self.use_gpu else 'CPU'} mode")

        # Ensure database table exists before proceeding (only for registered faces)
        ensure_registered_faces_table()

        self.detector = FaceDetector(use_gpu=self.use_gpu)
        self.recognizer = FaceRecognizer(use_gpu=self.use_gpu, use_faiss=use_faiss)
        self.tracker = SimpleTracker(iou_thresh=0.3, max_idle=2.0)
        self.frame_skipper = DynamicFrameSkipper(fast_rate=2, slow_rate=5)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        session_csv_path = os.path.join("attendance_reports", f"Attendance_Log_{timestamp}.csv")
        print(f"📝 This session's attendance log will be saved to: {session_csv_path}")
        self.db_writer = DBWriter(csv_path=session_csv_path)

        self.known_faces = self.load_known_faces()
        self.recognizer.load_known_faces_from_database(self.known_faces)

        self.marked_this_session = set()
        print(f"[INFO] Loaded {len(self.known_faces)} enrolled faces ✅")

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

    def _should_mark(self, person_id):
        if person_id in self.marked_this_session:
            return False
        try:
            # Check CSV files for last attendance record (attendance stored only in CSV)
            attendance_dir = "attendance_reports"
            if not os.path.exists(attendance_dir):
                return True
            
            # Get all CSV files in attendance_reports directory
            csv_files = glob.glob(os.path.join(attendance_dir, "*.csv"))
            if not csv_files:
                return True
            
            # Read all CSV files and find the most recent attendance for this person
            last_timestamp = None
            for csv_file in csv_files:
                try:
                    df = pd.read_csv(csv_file)
                    if 'PersonID' in df.columns and 'Date' in df.columns and 'Time' in df.columns:
                        person_records = df[df['PersonID'] == person_id]
                        if not person_records.empty:
                            # Get the most recent record from this file
                            for _, row in person_records.iterrows():
                                try:
                                    ts_str = f"{row['Date']} {row['Time']}"
                                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
                                    if last_timestamp is None or ts > last_timestamp:
                                        last_timestamp = ts
                                except Exception:
                                    continue
                except Exception:
                    continue
            
            if last_timestamp:
                time_diff = (datetime.now() - last_timestamp).total_seconds()
                if time_diff < ATTENDANCE_COOLDOWN:
                    return False
        except Exception as e:
            print(f"⚠️ CSV check error in _should_mark: {e}")
        return True

    def _enqueue_mark(self, person_id, person_name, confidence):
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        self.marked_this_session.add(person_id)
        self.db_writer.enqueue(person_id, person_name, ts, confidence)
        print(f"[MARKED] {person_name} @ {ts} ({confidence:.2f})")

    def run(self, camera_index=0, use_realsense=False):
        if use_realsense:
            cap = RealSenseCamera(width=1280, height=720, fps=30)
            cap.start()
        else:
            print("📷 Initializing Standard Webcam...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                print("✅ Standard Webcam started.")

        if not cap.isOpened():
            print("❌ Cannot access camera")
            return

        print("🎥 Starting Real-Time Attendance System (Fully Optimized)")
        frame_count = 0
        start_time = time.time()
        last_known_boxes = {}

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count +=1

                if self.frame_skipper.should_process():
                    detections = []
                    combined_roi = self.tracker.get_combined_roi()
                    if combined_roi:
                        x1, y1, x2, y2 = combined_roi
                        if x1 < x2 and y1 < y2:
                            roi_frame = frame[y1:y2, x1:x2]
                            detections_raw = self.detector.detect_faces(roi_frame)
                            for det in detections_raw:
                                x, y, w, h = det['bbox']
                                det['bbox'] = (x + x1, y + y1, w, h)
                                detections.append(det)
                    else:
                        detections = self.detector.detect_faces(frame)

                    matches = self.tracker.update(detections, frame.shape)
                    self.frame_skipper.update_status(bool(self.tracker.tracks))

                    current_boxes = {}
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
                                if pid is not None and self._should_mark(pid):
                                    self._enqueue_mark(pid, track.name, float(track.sim))
                                    track.recognized = True
                        
                        current_boxes[track.id] = (det['bbox'], track.name, track.sim)
                    last_known_boxes = current_boxes

                for track_id, (bbox, name, sim) in last_known_boxes.items():
                    x, y, w, h = bbox
                    color = (0, 255, 0) if name else (0, 0, 255)
                    text = f"{name} ({sim:.2f})" if name else f"Unknown ({sim:.2f})"
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow("Face Attendance", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            try:
                self.db_writer.stop()
            except Exception:
                pass
            print("🟢 Attendance session ended")