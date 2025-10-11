# enroll_dataset.py
import os
import cv2
import numpy as np
import sqlite3
import pickle
from tqdm import tqdm
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from config import DATABASE_PATH
import onnxruntime as ort

def create_database():
    """Create SQLite database and table if not exists"""
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS registered_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def store_embeddings_bulk(data):
    """
    Store multiple embeddings (name, pickled_embeddings_bytes) in one transaction.
    data: [(name, pickled_embeddings_bytes), ...]
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.executemany(
        "INSERT INTO registered_faces (name, encoding) VALUES (?, ?)",
        data
    )
    conn.commit()
    conn.close()

def load_images_from_folder(folder_path):
    """Load all image paths in folder"""
    exts = ('.jpg', '.jpeg', '.png')
    return [os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(exts)]

def enroll_from_folder(dataset_root):
    """
    dataset_root/
      person1/
        img1.jpg ...
      person2/
        img1.jpg ...
    This stores a PICKLED LIST of normalized embeddings for each person.
    """
    # --- Setup environment ---
    create_database()
    use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
    print(f"⚙️ Using {'GPU' if use_gpu else 'CPU'} mode for enrollment")

    detector = FaceDetector(use_gpu=use_gpu)
    recognizer = FaceRecognizer(use_gpu=use_gpu)

    persons = [p for p in os.listdir(dataset_root)
               if os.path.isdir(os.path.join(dataset_root, p))]

    all_to_store = []

    for person_name in persons:
        folder = os.path.join(dataset_root, person_name)
        image_files = load_images_from_folder(folder)
        if not image_files:
            print(f"⚠️ No images found for {person_name}")
            continue

        embeddings = []
        for img_path in tqdm(image_files, desc=f"Processing {person_name}", unit="img"):
            img = cv2.imread(img_path)
            if img is None:
                continue

            faces = detector.detect_faces(img)
            if not faces:
                continue

            # pick highest-confidence detection
            faces.sort(key=lambda f: f['confidence'], reverse=True)
            face_obj = faces[0]['face_obj']
            emb = recognizer.get_face_embedding(face_obj)
            if emb is not None:
                emb = emb / (np.linalg.norm(emb) + 1e-9)  # normalize
                embeddings.append(emb.astype(np.float32))

        if embeddings:
            # Optionally: augment or deduplicate embeddings here
            serialized = pickle.dumps(embeddings)
            all_to_store.append((person_name, serialized))
            print(f"[+] Stored {person_name} with {len(embeddings)} embeddings")
        else:
            print(f"[!] No valid faces found for {person_name}")

    # --- Commit all at once ---
    if all_to_store:
        store_embeddings_bulk(all_to_store)
        print(f"✅ Enrollment complete — {len(all_to_store)} persons registered")
    else:
        print("⚠️ No embeddings to store!")

if __name__ == "__main__":
    dataset_root = input("Enter dataset folder path (e.g. ./datasets): ").strip()
    enroll_from_folder(dataset_root)
