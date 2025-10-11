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

class AttendanceSystem:
    def __init__(self, use_faiss=False):
        # Auto-detect GPU
        self.use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
        print(f"⚙️ Initializing in {'GPU' if self.use_gpu else 'CPU'} mode")

        self.detector = FaceDetector(use_gpu=self.use_gpu)
        self.recognizer = FaceRecognizer(use_gpu=self.use_gpu, use_faiss=use_faiss)

        # tracker & writer
        self.tracker = SimpleTracker(iou_thresh=0.3, max_idle=2.0)
        self.db_writer = DBWriter()

        # load known faces from database (robust to pickled lists or raw bytes)
        self.known_faces = self.load_known_faces()
        self.recognizer.load_known_faces_from_database(self.known_faces)

        # short-term attendance control
        self.attendance_log = {}  # person_id -> last marked datetime
        print(f"[INFO] Loaded {len(self.known_faces)} enrolled faces ✅")

        # recognition tuning
        self.RECOG_PERIOD = 3      # do embedded recognition on new track or every RECOG_PERIOD track.age
        self.VOTE_WINDOW = 3       # consider last VOTE_WINDOW recognitions per track
        self.REQUIRED_VOTES = 2    # number of recognitions >= threshold to mark attendance

    def load_known_faces(self):
        """Load stored embeddings from SQLite. Handles both pickled-lists and legacy raw bytes."""
        conn = sqlite3.connect(DATABASE_PATH)
        cur = conn.cursor()
        try:
            cur.execute("SELECT id, name, encoding FROM registered_faces")
            rows = cur.fetchall()
        except Exception:
            # fallback to alternate table name or schema; return empty
            rows = []
        conn.close()

        known_faces = {}
        for pid, name, enc_bytes in rows:
            # store as-is; FaceRecognizer will unpickle or fallback
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
        # update in-memory cooldown
        self.attendance_log[person_id] = now
        # enqueue asynchronous DB write
        self.db_writer.enqueue(person_id, person_name, ts, confidence)
        print(f"[MARKED] {person_name} @ {ts} ({confidence:.2f})")

    def run(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return

        print("🎥 Starting Real-Time Attendance System (optimized)")
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Detect faces on every frame (you can change to process_every_n_frames)
                detections = self.detector.detect_faces(frame)  # returns list of {bbox, confidence, face_obj}
                matches = self.tracker.update(detections)  # list of (Track, detection)

                for track, det in matches:
                    # only attempt recognition on new track or periodically
                    do_recog = (not track.recognized) or (track.age % self.RECOG_PERIOD == 0)
                    if do_recog:
                        result, sim = self.recognizer.recognize_face(det)
                        # store vote
                        track.votes.append(sim)
                        # trim votes
                        if len(track.votes) > self.VOTE_WINDOW:
                            track.votes = track.votes[-self.VOTE_WINDOW:]

                        if result:
                            # update track info
                            track.name = result["person_name"]
                            track.sim = sim
                        else:
                            # unknown; optionally keep sim for diagnostics
                            track.sim = sim

                        # check voting: count how many votes >= threshold
                        positive_votes = sum(1 for v in track.votes if v >= FACE_RECOGNITION_THRESHOLD)
                        if positive_votes >= self.REQUIRED_VOTES and track.name is not None:
                            pid = result["person_id"] if result else None
                            if pid is not None and self._should_mark(pid):
                                # mark attendance asynchronously
                                self._enqueue_mark(pid, track.name, float(track.sim))
                                track.recognized = True  # prevent re-marking on same track

                    # draw visual overlay
                    x, y, w, h = det['bbox']
                    if track.name:
                        color = (0, 255, 0)
                        text = f"{track.name} ({track.sim:.2f})"
                    else:
                        color = (0, 0, 255)
                        text = f"Unknown ({track.sim:.2f})"
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
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            # ensure writer stopped gracefully
            try:
                self.db_writer.stop()
            except Exception:
                pass
            print("🟢 Attendance session ended")
