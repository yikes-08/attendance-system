import cv2
import os
import sys
import sqlite3
import pickle
import numpy as np
from tqdm import tqdm

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATABASE_PATH, FACE_DETECTION_CONFIDENCE
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from db_init import ensure_registered_faces_table

class DatasetEnroller:
    def __init__(self):
        try:
            import onnxruntime as ort
            has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
        except Exception:
            has_cuda = False
            
        self.detector = FaceDetector(use_gpu=has_cuda)
        self.recognizer = FaceRecognizer(use_gpu=has_cuda)
        # Ensure the table exists before connecting
        ensure_registered_faces_table()
        self.conn = sqlite3.connect(DATABASE_PATH)
        self.cur = self.conn.cursor()

    def setup_database(self):
        """Legacy method - kept for compatibility, but table is now ensured in __init__"""
        ensure_registered_faces_table()

    def enroll_from_directory(self, dataset_path):
        if not os.path.isdir(dataset_path):
            print(f"❌ Error: Directory not found at {dataset_path}")
            return

        person_folders = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        print(f"Found {len(person_folders)} people to enroll...")

        for person_name in tqdm(person_folders, desc="Enrolling People"):
            person_path = os.path.join(dataset_path, person_name)
            image_files = [f for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not image_files:
                print(f"⚠️ No images found for {person_name}, skipping.")
                continue

            person_embeddings = []
            for image_file in image_files:
                image_path = os.path.join(person_path, image_file)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"⚠️ Failed to read {image_path}, skipping.")
                    continue
                
                detections = self.detector.detect_faces(image)
                if detections:
                    # Use the first and most confident detection
                    best_detection = detections[0]
                    if best_detection['confidence'] > FACE_DETECTION_CONFIDENCE:
                        embedding = self.recognizer.get_face_embedding(best_detection['face_obj'])
                        if embedding is not None:
                            person_embeddings.append(embedding)

            if person_embeddings:
                # Serialize the list of embeddings using pickle
                embeddings_blob = pickle.dumps(person_embeddings)
                
                try:
                    self.cur.execute(
                        "INSERT OR REPLACE INTO registered_faces (name, encoding) VALUES (?, ?)",
                        (person_name, embeddings_blob)
                    )
                    self.conn.commit()
                except sqlite3.Error as e:
                    print(f"❌ Database error for {person_name}: {e}")
            else:
                print(f"⚠️ Could not generate any valid embeddings for {person_name}.")
        
        print("\n✅ Enrollment process completed.")
        self.conn.close()

# ✅ NEW: Create the function that main.py will import
def enroll_new_user(dataset_path):
    """
    Main function to handle the enrollment process.
    """
    enroller = DatasetEnroller()
    enroller.enroll_from_directory(dataset_path)

if __name__ == '__main__':
    # This block allows the script to still be run directly
    if len(sys.argv) > 1:
        enroll_new_user(sys.argv[1])
    else:
        print("Usage: python enroll_dataset.py <path_to_dataset_directory>")