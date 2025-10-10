import os
import cv2
import numpy as np
import sqlite3
from tqdm import tqdm
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from config import DATABASE_PATH

def create_database():
    """Create SQLite database and table if not exists"""
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS registered_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def store_embedding_in_db(name, embedding):
    """Store person’s name and mean embedding vector"""
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO registered_faces (name, encoding) VALUES (?, ?)",
                (name, embedding.tobytes()))
    conn.commit()
    conn.close()

def load_images_from_folder(folder_path):
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
    """
    create_database()
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    persons = [p for p in os.listdir(dataset_root)
               if os.path.isdir(os.path.join(dataset_root, p))]

    for person_name in persons:
        folder = os.path.join(dataset_root, person_name)
        image_files = load_images_from_folder(folder)
        if not image_files:
            continue

        embeddings = []
        for img_path in tqdm(image_files, desc=f"Processing {person_name}"):
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = detector.detect_faces(img)
            if not faces:
                continue
            # take the most confident face
            faces.sort(key=lambda f: f['confidence'], reverse=True)
            emb = recognizer.get_face_encoding(faces[0]['face_obj'])
            if emb is not None:
                embeddings.append(emb)

        if embeddings:
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-9)
            store_embedding_in_db(person_name, mean_emb)
            print(f"[+] Stored {person_name} with {len(embeddings)} images")
        else:
            print(f"[!] No valid faces for {person_name}")

    print("Enrollment complete ✅")

if __name__ == "__main__":
    dataset_root = input("Enter dataset folder path (e.g. ./dataset): ").strip()
    enroll_from_folder(dataset_root)
