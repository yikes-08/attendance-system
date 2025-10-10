import cv2
import numpy as np
import time
import threading
from datetime import datetime, timedelta
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from database import AttendanceDatabase
from email_notification import EmailNotifier
from config import *

class AttendanceSystem:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.database = AttendanceDatabase()
        self.email_notifier = EmailNotifier()
        
        # Load known faces from database
        known_faces = self.database.get_known_faces()
        self.face_recognizer.load_known_faces_from_database(known_faces)
        
        # Attendance tracking
        self.last_attendance = {}  # Track last attendance time for each person
        self.attendance_cooldown = ATTENDANCE_COOLDOWN
        
        # Email scheduling
        self.last_email_sent = datetime.now()
        self.email_interval = ATTENDANCE_EMAIL_INTERVAL
        
        # Camera
        self.camera = None
        self.is_running = False
        
        print("Attendance System initialized successfully")
    
    def initialize_camera(self):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(CAMERA_INDEX)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            
            if not self.camera.isOpened():
                raise Exception("Could not open camera")
            
            print(f"Camera initialized successfully (Index: {CAMERA_INDEX})")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def add_person(self, person_id, person_name, face_image):
        """Add a new person to the system"""
        try:
            # Add face encoding to recognizer
            success = self.face_recognizer.add_known_face(person_id, person_name, face_image)
            
            if success:
                # Save to database
                encoding = self.face_recognizer.known_faces[person_id]['encoding']
                self.database.add_known_face(person_id, person_name, encoding)
                print(f"Person {person_name} added successfully")
                return True
            else:
                print(f"Failed to add person {person_name}")
                return False
        except Exception as e:
            print(f"Error adding person: {e}")
            return False
    
    def mark_attendance(self, person_id, person_name, confidence):
        """Mark attendance for a person"""
        current_time = datetime.now()
        
        # Check cooldown period
        if person_id in self.last_attendance:
            time_diff = (current_time - self.last_attendance[person_id]).total_seconds()
            if time_diff < self.attendance_cooldown:
                return False
        
        # Mark attendance
        self.database.add_attendance_record(person_id, person_name, confidence)
        self.last_attendance[person_id] = current_time
        
        print(f"Attendance marked for {person_name} at {current_time.strftime('%H:%M:%S')}")
        
        # Send immediate notification
        self.email_notifier.send_immediate_notification(
            person_name, 
            current_time.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        return True
    
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        for face in faces:
            # Extract face region
            face_region = self.face_detector.extract_face_region(frame, face)
            
            if face_region is not None and face_region.size > 0:
                # Recognize face
                recognition_result, confidence = self.face_recognizer.recognize_face(face_region)
                
                if recognition_result:
                    # Mark attendance
                    self.mark_attendance(
                        recognition_result['person_id'],
                        recognition_result['person_name'],
                        confidence
                    )
                    
                    # Draw bounding box and label
                    x, y, w, h = face['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{recognition_result['person_name']} ({confidence:.2f})", 
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Unknown face
                    x, y, w, h = face['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def check_email_schedule(self):
        """Check if it's time to send scheduled email"""
        current_time = datetime.now()
        if (current_time - self.last_email_sent).total_seconds() >= self.email_interval * 60:
            # Send daily report
            attendance_records = self.database.get_attendance_records()
            if attendance_records:
                self.email_notifier.send_attendance_report(attendance_records)
                self.last_email_sent = current_time
    
    def run(self):
        """Main loop for the attendance system"""
        if not self.initialize_camera():
            return
        
        self.is_running = True
        print("Starting attendance system...")
        print("Press 'q' to quit, 'a' to add new person, 's' to send report")
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Check email schedule
            self.check_email_schedule()
            
            # Display frame
            cv2.imshow('Attendance System', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.add_person_interactive(processed_frame)
            elif key == ord('s'):
                self.send_manual_report()
        
        self.cleanup()
    
    def add_person_interactive(self, frame):
        """Interactive method to add a new person"""
        print("Adding new person...")
        person_id = input("Enter person ID: ")
        person_name = input("Enter person name: ")
        
        # Detect faces in current frame
        faces = self.face_detector.detect_faces(frame)
        
        if faces:
            # Use the first detected face
            face_region = self.face_detector.extract_face_region(frame, faces[0])
            if face_region is not None:
                success = self.add_person(person_id, person_name, face_region)
                if success:
                    print(f"Person {person_name} added successfully!")
                else:
                    print("Failed to add person. Please try again.")
            else:
                print("Could not extract face region. Please try again.")
        else:
            print("No face detected. Please position face in camera view and try again.")
    
    def send_manual_report(self):
        """Send manual attendance report"""
        print("Sending manual attendance report...")
        attendance_records = self.database.get_attendance_records()
        if attendance_records:
            success = self.email_notifier.send_attendance_report(attendance_records)
            if success:
                print("Report sent successfully!")
            else:
                print("Failed to send report.")
        else:
            print("No attendance records found.")
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("Attendance system stopped.")
    
    def get_statistics(self):
        """Get attendance statistics"""
        records = self.database.get_attendance_records()
        if not records:
            return "No attendance records found."
        
        df = pd.DataFrame(records, columns=['Person ID', 'Person Name', 'Timestamp', 'Confidence'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        stats = {
            'Total Records': len(df),
            'Unique People': df['Person Name'].nunique(),
            'Today Records': len(df[df['Timestamp'].dt.date == datetime.now().date()]),
            'Last 7 Days': len(df[df['Timestamp'] >= datetime.now() - timedelta(days=7)])
        }
        
        return stats
