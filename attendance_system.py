# attendance_system.py
import cv2
import sqlite3
import numpy as np
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
from simple_tracker import SimpleTracker
from db_writer import DBWriter

class DynamicFrameSkipper:
    """
    Intelligently decides whether to process a frame to save resources.
    - Processes frequently when no faces are tracked (to find new ones).
    - Processes less frequently when faces are already being tracked.
    """
    def __init__(self, fast_rate=2, slow_rate=5):
        self.fast_rate = fast_rate  # Process every 2nd frame when searching
        self.slow_rate = slow_rate  # Process every 5th frame when tracking
        self.tracking_active = False
        self.frame_counter = 0

    def update_status(self, tracking_active):
        """Update the tracker's status from the main loop."""
        self.tracking_active = tracking_active

    def should_process(self):
        """Returns True if the current frame should be processed."""
        self.frame_counter += 1
        rate = self.slow_rate if self.tracking_active else self.fast_rate
        if self.frame_counter % rate == 0:
            return True
        return False

class AttendanceSystem:
    # __init__, load_known_faces, _should_mark, _enqueue_mark methods remain unchanged.
    def __init__(self, use_faiss=False):
        # Auto-detect GPU
        self.use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
        print(f"⚙️ Initializing in {'GPU' if self.use_gpu else 'CPU'} mode")

        self.detector = FaceDetector(use_gpu=self.use_gpu)
        self.recognizer = FaceRecognizer(use_gpu=self.use_gpu, use_faiss=use_faiss)

        # tracker & writer
        self.tracker = SimpleTracker(iou_thresh=0.3, max_idle=2.0)
        self.db_writer = DBWriter()
        
        # --- 🧠 Dynamic Frame Skipper ---
        self.frame_skipper = DynamicFrameSkipper(fast_rate=2, slow_rate=5)

        # load known faces from database
        self.known_faces = self.load_known_faces()
        self.recognizer.load_known_faces_from_database(self.known_faces)

        # short-term attendance control
        self.attendance_log = {}
        print(f"[INFO] Loaded {len(self.known_faces)} enrolled faces ✅")

        # recognition tuning
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
        now = datetime.now()
        last = self.attendance_log.get(person_id)
        if last and (now - last).total_seconds() < ATTENDANCE_COOLDOWN:
            return False
        return True

    def _enqueue_mark(self, person_id, person_name, confidence):
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S")
        self.attendance_log[person_id] = now
        self.db_writer.enqueue(person_id, person_name, ts, confidence)
        print(f"[MARKED] {person_name} @ {ts} ({confidence:.2f})")

    def run(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return

        print("🎥 Starting Real-Time Attendance System (Optimized with ROI + Dynamic Skip)")
        frame_count = 0
        start_time = time.time()
        
        # Store last known bounding boxes to draw on skipped frames
        last_known_boxes = {}

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count +=1

                # --- 🧠 Dynamic Frame Skipping Logic ---
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
                    
                    # Update skipper with current tracking status
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

                # --- Draw on EVERY frame using last known data for a smooth display ---
                for track_id, (bbox, name, sim) in last_known_boxes.items():
                    x, y, w, h = bbox
                    if name:
                        color = (0, 255, 0)
                        text = f"{name} ({sim:.2f})"
                    else:
                        color = (0, 0, 255)
                        text = f"Unknown ({sim:.2f})"
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