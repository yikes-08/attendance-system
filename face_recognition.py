import numpy as np
from insightface.model_zoo import get_model
from config import FACE_RECOGNITION_THRESHOLD

class FaceRecognizer:
    """
    ArcFace-based embedding + similarity comparison
    """

    def __init__(self):
        # Load ArcFace model from buffalo_l pack (InsightFace)
        self.model = get_model('arcface_r100_v1', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0)
        self.known_faces = []

    def get_face_encoding(self, face_obj):
        """
        Extract 512-D embedding vector from a detected/Aligned face object
        """
        if not hasattr(face_obj, 'normed_embedding'):
            return None
        emb = face_obj.normed_embedding.astype(np.float32)
        return emb

    def load_known_faces_from_database(self, known_faces):
        """
        Accept dict of {id: {'name', 'encoding'}} from SQLite
        """
        self.known_faces = [
            (pid, data["name"], data["encoding"]) for pid, data in known_faces.items()
        ]

    def recognize_face(self, face):
        """
        Compare embedding with known embeddings → return match + confidence
        """
        if not self.known_faces:
            return None, 0.0

        emb = self.get_face_encoding(face['face_obj'])
        if emb is None:
            return None, 0.0

        similarities = []
        for pid, name, known_emb in self.known_faces:
            sim = np.dot(emb, known_emb) / (np.linalg.norm(emb) * np.linalg.norm(known_emb) + 1e-9)
            similarities.append((pid, name, sim))

        pid, name, best_sim = max(similarities, key=lambda x: x[2])
        if best_sim >= FACE_RECOGNITION_THRESHOLD:
            return {"person_id": pid, "person_name": name}, best_sim
        else:
            return None, best_sim
