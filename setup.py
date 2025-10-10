#!/usr/bin/env python3
"""
Setup script for the Face Detection Biometric Attendance System
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        project_root = os.path.dirname(os.path.abspath(__file__))
        req_path = os.path.join(project_root, "requirements.txt")
        if not os.path.isfile(req_path):
            raise FileNotFoundError(f"requirements.txt not found at {req_path}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "models",
        "reports",
        "data"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_models():
    """Download required models"""
    print("Downloading models...")
    
    # YuNet model will be downloaded automatically when first used
    # FaceNet model will be downloaded automatically when first used
    
    print("Models will be downloaded automatically on first run.")

def setup_database():
    """Initialize database"""
    print("Setting up database...")
    try:
        from database import AttendanceDatabase
        db = AttendanceDatabase()
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False
    return True

def main():
    print("Setting up Face Detection Biometric Attendance System...")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please check your Python environment.")
        return
    
    # Create directories
    create_directories()
    
    # Download models
    download_models()
    
    # Setup database
    if not setup_database():
        print("Failed to setup database.")
        return
    
    print("=" * 50)
    print("Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update config.py with your email settings")
    print("2. Run: python main.py --test-email")
    print("3. Run: python main.py")
    print("\nControls:")
    print("- Press 'q' to quit")
    print("- Press 'a' to add new person")
    print("- Press 's' to send manual report")

if __name__ == "__main__":
    main()
