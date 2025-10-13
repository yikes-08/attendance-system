import numpy as np
import pickle
from config import FACE_RECOGNITION_THRESHOLD
from insightface.app import FaceAnalysis

class FaceRecognizer:
    """
    InsightFace FaceAnalysis wrapper (RetinaFace + ArcFace embedding)
    Supports multiple stored embeddings per person (pickled list).
    Optional FAISS index for faster lookups (disabled by default).
    """

    def __init__(self, use_gpu=True, use_faiss=False):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        # Initialize model (detection + embedding)
        self.model = FaceAnalysis(name='buffalo_l', providers=providers)
        try:
            self.model.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
        except Exception as e:
            # fallback prepare without ctx_id in rare cases
            self.model.prepare(det_size=(640, 640))

        self.use_faiss = use_faiss
        self.known_faces = []  # list of (id, name, [embs])
        self.faiss_index = None
        self.id_map = []  # index -> (pid, name)

    def get_face_embedding(self, face_obj):
        """
        Extract normalized embedding vector (float32) from InsightFace face object.
        """
        # InsightFace uses attributes like .normed_embedding or .embedding depending on version
        emb = None
        if hasattr(face_obj, 'normed_embedding') and face_obj.normed_embedding is not None:
            emb = face_obj.normed_embedding
        elif hasattr(face_obj, 'embedding') and face_obj.embedding is not None:
            emb = face_obj.embedding
        if emb is None:
            return None
        emb = np.array(emb, dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        return emb

    def load_known_faces_from_database(self, known_faces_db, use_faiss=None):
        """
        known_faces_db: dict {id: {'name': name, 'encoding': bytes_or_array}}
        The encoding stored may be:
          - pickled list of embeddings (preferred for new enrollments),
          - raw float32 bytes (legacy): we'll fallback to np.frombuffer.
        """
        self.known_faces = []
        vectors = []
        self.id_map = []

        for pid, data in known_faces_db.items():
            name = data.get('name') or data.get('person_name') or str(pid)
            enc = data.get('encoding') or data.get('encoding_bytes') or data.get('enc_bytes')
            embs_list = []

            if enc is None:
                # nothing to load
                continue

            # try unpickle (preferred)
            try:
                loaded = pickle.loads(enc)
                # if single array was pickled, make it a list
                if isinstance(loaded, (list, tuple)):
                    embs_list = [np.array(e, dtype=np.float32) for e in loaded]
                else:
                    embs_list = [np.array(loaded, dtype=np.float32)]
            except Exception:
                # fallback: maybe it's raw float32 bytes
                try:
                    arr = np.frombuffer(enc, dtype=np.float32)
                    if arr.size > 0:
                        # if arr is a 1D embedding, keep as one sample
                        embs_list = [arr.astype(np.float32)]
                except Exception:
                    # can't decode; skip
                    continue

            # normalize stored embeddings
            embs_list = [e / (np.linalg.norm(e) + 1e-9) for e in embs_list]
            self.known_faces.append((pid, name, embs_list))

            # build flat vector list for FAISS mapping (if enabled)
            if (use_faiss if use_faiss is not None else self.use_faiss) and embs_list:
                for e in embs_list:
                    vectors.append(e)
                    self.id_map.append((pid, name))

        # optionally build FAISS index
        if (use_faiss if use_faiss is not None else self.use_faiss) and vectors:
            try:
                import faiss
                d = vectors[0].shape[0]
                index = faiss.IndexFlatIP(d)  # inner product works for normalized vectors
                vecs = np.vstack(vectors).astype('float32')
                faiss.normalize_L2(vecs)
                index.add(vecs)
                self.faiss_index = index
                self.use_faiss = True
            except Exception as e:
                print(f"⚠️ FAISS not available or failed to init: {e}")
                self.faiss_index = None
                self.use_faiss = False

    def recognize_face(self, face):
        """
        Compare embedding with known embeddings → return match + confidence
        Returns: ({"person_id": pid, "person_name": name}, similarity) or (None, best_sim)
        """
        if not self.known_faces and not self.faiss_index:
            return None, 0.0

        emb = self.get_face_embedding(face['face_obj'])
        if emb is None:
            return None, 0.0

        # Option A: FAISS fast lookup
        if self.use_faiss and self.faiss_index is not None:
            try:
                import faiss
                q = emb.reshape(1, -1).astype('float32')
                faiss.normalize_L2(q)
                D, I = self.faiss_index.search(q, k=1)  # top-1
                best_sim = float(D[0][0])
                best_idx = int(I[0][0])
                if best_idx < 0 or best_idx >= len(self.id_map):
                    return None, best_sim
                pid, name = self.id_map[best_idx]
                if best_sim >= FACE_RECOGNITION_THRESHOLD:
                    return {"person_id": pid, "person_name": name}, best_sim
                else:
                    return None, best_sim
            except Exception as e:
                # on any FAISS error, fallback to brute-force below
                print(f"⚠️ FAISS search error, falling back: {e}")

        # Option B: brute-force per-person (max of per-person samples)
        best_sim = -1.0
        best_pid = None
        best_name = None
        for pid, name, embs in self.known_faces:
            if not embs:
                continue
            # compute dot product (vectors are normalized => dot == cosine)
            sims = [float(np.dot(emb, e) / (np.linalg.norm(e) + 1e-9)) for e in embs]
            local_max = max(sims) if sims else -1.0
            if local_max > best_sim:
                best_sim = local_max
                best_pid = pid
                best_name = name

        if best_sim >= FACE_RECOGNITION_THRESHOLD:
            return {"person_id": best_pid, "person_name": best_name}, best_sim
        else:
            return None, best_sim
