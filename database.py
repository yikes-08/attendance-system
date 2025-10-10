import sqlite3
import datetime
from config import DATABASE_PATH

class AttendanceDatabase:
    def __init__(self):
        self.db_path = DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL,
                person_name TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                confidence REAL NOT NULL
            )
        ''')
        
        # Create known_faces table for storing face encodings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT UNIQUE NOT NULL,
                person_name TEXT NOT NULL,
                face_encoding BLOB NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_attendance_record(self, person_id, person_name, confidence):
        """Add a new attendance record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO attendance (person_id, person_name, timestamp, confidence)
            VALUES (?, ?, ?, ?)
        ''', (person_id, person_name, datetime.datetime.now(), confidence))
        
        conn.commit()
        conn.close()
    
    def add_known_face(self, person_id, person_name, face_encoding):
        """Add a new known face to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Convert numpy array to bytes for storage
        import pickle
        encoding_bytes = pickle.dumps(face_encoding)
        
        cursor.execute('''
            INSERT OR REPLACE INTO known_faces (person_id, person_name, face_encoding)
            VALUES (?, ?, ?)
        ''', (person_id, person_name, encoding_bytes))
        
        conn.commit()
        conn.close()
    
    def get_known_faces(self):
        """Retrieve all known faces from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT person_id, person_name, face_encoding FROM known_faces')
        results = cursor.fetchall()
        
        known_faces = {}
        for person_id, person_name, face_encoding_bytes in results:
            import pickle
            face_encoding = pickle.loads(face_encoding_bytes)
            known_faces[person_id] = {
                'name': person_name,
                'encoding': face_encoding
            }
        
        conn.close()
        return known_faces
    
    def get_attendance_records(self, start_date=None, end_date=None):
        """Get attendance records within a date range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if start_date and end_date:
            cursor.execute('''
                SELECT person_id, person_name, timestamp, confidence
                FROM attendance
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp DESC
            ''', (start_date, end_date))
        else:
            cursor.execute('''
                SELECT person_id, person_name, timestamp, confidence
                FROM attendance
                ORDER BY timestamp DESC
            ''')
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def get_recent_attendance(self, person_id, minutes=5):
        """Check if person has been marked present recently"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
        cursor.execute('''
            SELECT COUNT(*) FROM attendance
            WHERE person_id = ? AND timestamp > ?
        ''', (person_id, cutoff_time))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
