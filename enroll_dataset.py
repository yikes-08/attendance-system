import os
import cv2
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=""):
        print(f"{desc}...")
        for item in iterable:
            yield item

from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from database import AttendanceDatabase

DATA_DIR = "data/train"  # root of your dataset

def get_face_crops(detector, img_bgr):
    faces = detector.detect_faces(img_bgr)
    crops = []
    for f in faces:
        crop = detector.extract_face_region(img_bgr, f)
        if crop is not None and crop.size > 0:
            crops.append(crop)
    return crops

def main():
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    db = AttendanceDatabase()

    # For each person, compute centroid embedding from all valid face crops
    for person_name in sorted(os.listdir(DATA_DIR)):
        person_dir = os.path.join(DATA_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        img_paths = []
        for fn in os.listdir(person_dir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                img_paths.append(os.path.join(person_dir, fn))

        if not img_paths:
            print(f"Skipping {person_name}: no images")
            continue

        for p in tqdm(img_paths, desc=f"Processing {person_name}"):
            img = cv2.imread(p)
            if img is None:
                continue
            crops = get_face_crops(detector, img)
            # Use the most confident/first crop found
            if not crops:
                continue
            emb = recognizer.get_face_encoding(crops[0])
            if emb is not None and np.all(np.isfinite(emb)):
                embeddings.append(emb)

        if not embeddings:
            print(f"Skipping {person_name}: no valid embeddings")
            continue

        # Compute centroid embedding
        centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

        # Use a stable person_id (e.g., same as name or a slug)
        person_id = person_name

        # Persist centroid in DB
        db.add_known_face(person_id, person_name, centroid)
        print(f"Enrolled {person_name}: {len(embeddings)} images -> centroid saved")

    print("Enrollment complete.")

if __name__ == "__main__":
    main()