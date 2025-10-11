# face_recognition.py
import numpy as np
from insightface.app import FaceAnalysis
from config import FACE_RECOGNITION_THRESHOLD

class FaceRecognizer:
    """
    InsightFace FaceAnalysis wrapper (RetinaFace + ArcFace embedding)
    """

    def __init__(self, use_gpu=True):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

        # Initialize the combined model (detection + embedding)
        self.model = FaceAnalysis(name='buffalo_l', providers=providers)
        self.model.prepare(ctx_id=0, det_size=(640, 640))
        self.known_faces = []

    def get_face_embedding(self, face_obj):
        """
        Extract 512-D embedding vector from a detected face object.
        """
        if not hasattr(face_obj, 'normed_embedding'):
            return None
        return face_obj.normed_embedding.astype(np.float32)

    def load_known_faces_from_database(self, known_faces):
        """
        Accept dict of {id: {'name', 'encoding'}} from SQLite
        """
        self.known_faces = [
            (pid, data["name"], np.array(data["encoding"], dtype=np.float32))
            for pid, data in known_faces.items()
        ]

    def recognize_face(self, face):
        """
        Compare embedding with known embeddings → return match + confidence
        """
        if not self.known_faces:
            return None, 0.0

        emb = self.get_face_embedding(face['face_obj'])
        if emb is None:
            return None, 0.0

        similarities = []
        for pid, name, known_emb in self.known_faces:
            sim = np.dot(emb, known_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(known_emb) + 1e-9
            )
            similarities.append((pid, name, sim))

        pid, name, best_sim = max(similarities, key=lambda x: x[2])
        if best_sim >= FACE_RECOGNITION_THRESHOLD:
            return {"person_id": pid, "person_name": name}, best_sim
        else:
            return None, best_sim
