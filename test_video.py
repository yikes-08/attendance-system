# test_video.py (fixed bbox handling)
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
        # try to detect GPU via onnxruntime available providers
        try:
            import onnxruntime as ort
            has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
        except Exception:
            has_cuda = False

        self.detector = FaceDetector(use_gpu=has_cuda)
        self.recognizer = FaceRecognizer(use_gpu=has_cuda, use_faiss=use_faiss)

        self.tracker = SimpleTracker(iou_thresh=0.3, max_idle=2.0)
        self.db_writer = DBWriter()
        self.attendance_records = []
        self.last_seen = {}

        # Load known faces
        self.known_faces = self.load_known_faces()
        self.recognizer.load_known_faces_from_database(self.known_faces)

        if not self.known_faces:
            print("⚠️ [WARNING] No enrolled faces found in the database. Please run `python enroll_dataset.py` first.")
        else:
            print(f"✅ Loaded {len(self.known_faces)} known faces from database.")

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

    def _extract_bbox(self, det, frame_shape):
        """
        Return (x, y, w, h) as ints.
        Handles:
          - detector dicts with 'bbox' already as (x,y,w,h)
          - legacy/other formats (x1,y1,x2,y2) → converted to w,h
          - list/tuple/ndarray formats with heuristic fallback
        """
        h_frame, w_frame = frame_shape[:2]

        # dict with 'bbox'
        if isinstance(det, dict) and "bbox" in det:
            bx = det["bbox"]
            if len(bx) >= 4:
                x, y, w, h = map(int, bx[:4])
                # If bbox looks like x1,y1,x2,y2 (x2 > x1 and x2 within frame), convert
                if w > w_frame or h > h_frame or (w > 0 and h > 0 and (x + w) > w_frame or (y + h) > h_frame):
                    # maybe stored as x1,y1,x2,y2
                    x1, y1, x2, y2 = x, y, w, h
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    x, y = x1, y1
                return max(0, x), max(0, y), max(0, w), max(0, h)

        # list/tuple/ndarray fallback
        if isinstance(det, (list, tuple, np.ndarray)):
            if len(det) >= 4:
                x1, y1, a3, a4 = map(int, det[:4])
                # heuristics: if a3,a4 are coordinates within frame and greater than x1,y1 -> treat as x2,y2
                if (0 <= a3 <= w_frame and 0 <= a4 <= h_frame and a3 > x1 and a4 > y1):
                    x, y = x1, y1
                    w = max(0, a3 - x1)
                    h = max(0, a4 - y1)
                    return x, y, w, h
                else:
                    # treat a3,a4 as w,h
                    return max(0, x1), max(0, y1), max(0, a3), max(0, a4)

        # object with .bbox (e.g., insightface face obj)
        if hasattr(det, "bbox"):
            try:
                bx = list(map(int, det.bbox))
                if len(bx) == 4:
                    x1, y1, x2, y2 = bx
                    w = max(0, x2 - x1)
                    h = max(0, y2 - y1)
                    return max(0, x1), max(0, y1), w, h
            except Exception:
                pass

        # give a safe empty bbox if nothing matches
        return 0, 0, 0, 0

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
        fps = cap.get(cv2.CAP_PROP_FPS) or 25  # fallback to 25 if fps is 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Optional: save annotated video
        output_video_path = os.path.join(output_dir, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        with tqdm(total=total_frames, desc="🎬 Processing Video", unit="frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.detect_faces(frame)
                
                # --- ✅ FIX: Pass frame.shape to the tracker update method ---
                matches = self.tracker.update(detections, frame.shape)

                for track, det in matches:
                    # Extract bbox as (x, y, w, h)
                    x, y, w, h = self._extract_bbox(det, frame.shape)

                    # draw rectangle using x,y,w,h
                    color = (0, 255, 0) if getattr(track, "recognized", False) else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Label text: include tracker ID
                    label = f"ID {track.id}: " + (track.name if getattr(track, "name", None) else "Unknown")
                    conf = getattr(track, "sim", 0.0)
                    cv2.putText(frame, f"{label} ({conf:.2f})", (x, max(15, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Recognition
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

                # --- Display live frame and save video ---
                cv2.imshow("Test Video - Face Recognition", frame)
                out_writer.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n🛑 Video manually stopped.")
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