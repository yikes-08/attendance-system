import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from config import FACENET_MODEL_PATH, FACE_RECOGNITION_THRESHOLD
try:
    from keras_facenet import FaceNet
    _FACENET_AVAILABLE = True
except Exception:
    _FACENET_AVAILABLE = False

class FaceRecognizer:
    def __init__(self):
        self.model = None
        self.known_faces = {}
        self.load_model()
    
    def load_model(self):
        """Load FaceNet model for face recognition"""
        try:
            if _FACENET_AVAILABLE:
                # Use keras-facenet which bundles FaceNet and preprocessing
                self.model = FaceNet()
                print("FaceNet (keras-facenet) loaded successfully")
            else:
                raise RuntimeError("keras-facenet not available")
        except Exception as e:
            print(f"Error loading FaceNet model: {e}")
            print("Using a simplified face recognition approach...")
            self.model = None
    
    def preprocess_face(self, face_image):
        """Preprocess face image for recognition"""
        if face_image is None or face_image.size == 0:
            return None
        
        # Resize to 160x160 (FaceNet input size)
        face_resized = cv2.resize(face_image, (160, 160))
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        face_normalized = face_rgb.astype('float32') / 255.0
        
        # Add batch dimension
        face_batch = np.expand_dims(face_normalized, axis=0)
        
        return face_batch
    
    def get_face_encoding(self, face_image):
        """Get face encoding using FaceNet model"""
        if self.model is None:
            return self.get_simple_face_features(face_image)

        try:
            # keras-facenet expects RGB 160x160 but handles preprocessing internally
            face_rgb = cv2.cvtColor(cv2.resize(face_image, (160, 160)), cv2.COLOR_BGR2RGB)
            embedding = self.model.embeddings([face_rgb])[0]
            return embedding
        except Exception as e:
            print(f"Error getting face encoding: {e}")
            return self.get_simple_face_features(face_image)
    
    def get_simple_face_features(self, face_image):
        """Fallback method using simple image features"""
        if face_image is None or face_image.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray_resized = cv2.resize(gray, (64, 64))
        
        # Use histogram as features
        hist = cv2.calcHist([gray_resized], [0], None, [256], [0, 256])
        
        # Normalize histogram
        hist_normalized = hist.flatten() / np.sum(hist)
        
        return hist_normalized
    
    def add_known_face(self, person_id, person_name, face_image):
        """Add a new known face to the system"""
        encoding = self.get_face_encoding(face_image)
        if encoding is not None:
            self.known_faces[person_id] = {
                'name': person_name,
                'encoding': encoding
            }
            return True
        return False
    
    def recognize_face(self, face_image):
        """Recognize a face from the given image"""
        if not self.known_faces:
            return None, 0.0
        
        face_encoding = self.get_face_encoding(face_image)
        if face_encoding is None:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for person_id, face_data in self.known_faces.items():
            known_encoding = face_data['encoding']
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                face_encoding.reshape(1, -1),
                known_encoding.reshape(1, -1)
            )[0][0]
            
            if similarity > best_similarity and similarity > FACE_RECOGNITION_THRESHOLD:
                best_similarity = similarity
                best_match = {
                    'person_id': person_id,
                    'person_name': face_data['name'],
                    'confidence': similarity
                }
        
        return best_match, best_similarity
    
    def load_known_faces_from_database(self, known_faces_dict):
        """Load known faces from database"""
        self.known_faces = known_faces_dict
    
    def get_face_landmarks(self, face_image):
        """Extract face landmarks (simplified version)"""
        # This is a simplified version - in production, you might want to use
        # a more sophisticated landmark detection model
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Simple feature points based on face geometry
        h, w = gray.shape
        landmarks = np.array([
            [w//4, h//3],      # Left eye
            [3*w//4, h//3],    # Right eye
            [w//2, h//2],      # Nose
            [w//4, 2*h//3],    # Left mouth corner
            [3*w//4, 2*h//3]   # Right mouth corner
        ])
        
        return landmarks
