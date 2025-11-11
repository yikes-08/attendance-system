# db_init.py
"""
Shared database initialization module to ensure consistent table creation
across all components of the attendance system.
"""
import sqlite3
from config import DATABASE_PATH


def initialize_database():
    """
    Initialize the database with all required tables.
    This ensures both registered_faces and attendance tables exist.
    Should be called by all modules that interact with the database.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    
    try:
        # Create registered_faces table for storing enrolled face encodings
        cur.execute("""
            CREATE TABLE IF NOT EXISTS registered_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                UNIQUE(name)
            )
        """)
        
        # Create attendance table for storing attendance records
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                name TEXT,
                date TEXT,
                time TEXT,
                confidence REAL
            )
        """)
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"❌ Database initialization error: {e}")
        conn.rollback()
    finally:
        conn.close()


def ensure_registered_faces_table():
    """Ensure the registered_faces table exists."""
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS registered_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                encoding BLOB NOT NULL,
                UNIQUE(name)
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"❌ Error ensuring registered_faces table: {e}")
        conn.rollback()
    finally:
        conn.close()


def ensure_attendance_table():
    """Ensure the attendance table exists."""
    conn = sqlite3.connect(DATABASE_PATH)
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER,
                name TEXT,
                date TEXT,
                time TEXT,
                confidence REAL
            )
        """)
        conn.commit()
    except sqlite3.Error as e:
        print(f"❌ Error ensuring attendance table: {e}")
        conn.rollback()
    finally:
        conn.close()

