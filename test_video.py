#!/usr/bin/env python3
"""
Video Testing Script for Face Detection Attendance System
Processes recorded video footage and generates local attendance reports
"""

import cv2
import os
import pandas as pd
from datetime import datetime, timedelta
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from database import AttendanceDatabase
from config import ATTENDANCE_COOLDOWN

class VideoTester:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.database = AttendanceDatabase()
        
        # Load known faces from database
        known_faces = self.database.get_known_faces()
        self.face_recognizer.load_known_faces_from_database(known_faces)
        
        # Attendance tracking
        self.attendance_records = []
        self.last_attendance = {}
        self.attendance_cooldown = ATTENDANCE_COOLDOWN
        
        print(f"Video Tester initialized with {len(known_faces)} known faces")
        for person_id, face_data in known_faces.items():
            print(f"  - {face_data['name']} (ID: {person_id})")
    
    def process_video(self, video_path, output_dir="attendance_reports"):
        """Process video and generate attendance report"""
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            return False
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Processing video: {os.path.basename(video_path)}")
        print(f"Duration: {duration:.2f} seconds, FPS: {fps}, Total frames: {total_frames}")
        
        frame_count = 0
        processed_frames = 0
        detection_interval = max(1, fps // 2)  # Process every half second
        
        print("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every nth frame (to avoid duplicate detections)
            if frame_count % detection_interval == 0:
                self.process_frame(frame, frame_count / fps)
                processed_frames += 1
                
                # Progress indicator
                if processed_frames % 10 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% - Processed {processed_frames} frames")
            
            # Show frame with detections (optional)
            self.draw_detections(frame)
            cv2.imshow('Video Processing', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nVideo processing complete!")
        print(f"Total frames processed: {processed_frames}")
        print(f"Attendance records: {len(self.attendance_records)}")
        
        # Generate attendance report
        self.generate_attendance_report(output_dir)
        return True
    
    def process_frame(self, frame, timestamp):
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
                    # Mark attendance (with cooldown check)
                    self.mark_attendance(
                        recognition_result['person_id'],
                        recognition_result['person_name'],
                        confidence,
                        timestamp
                    )
    
    def mark_attendance(self, person_id, person_name, confidence, timestamp):
        """Mark attendance for a person (with cooldown)"""
        current_time = datetime.now()  # Use system time instead of video time
        
        # Check cooldown period
        if person_id in self.last_attendance:
            time_diff = abs((current_time - self.last_attendance[person_id]).total_seconds())
            if time_diff < self.attendance_cooldown:
                return False
        
        # Mark attendance
        attendance_record = {
            'Name': person_name,
            'Attendance': 'Present',
            'Timestamp': current_time,
            'Confidence': confidence,
            'Video_Time': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}"
        }
        
        self.attendance_records.append(attendance_record)
        self.last_attendance[person_id] = current_time
        
        print(f"✓ Attendance marked: {person_name} at {current_time.strftime('%H:%M:%S')} (confidence: {confidence:.3f})")
        return True
    
    def draw_detections(self, frame):
        """Draw face detection boxes and labels on frame"""
        faces = self.face_detector.detect_faces(frame)
        
        for face in faces:
            x, y, w, h = face['bbox']
            
            # Extract face region and recognize
            face_region = self.face_detector.extract_face_region(frame, face)
            if face_region is not None:
                recognition_result, confidence = self.face_recognizer.recognize_face(face_region)
                
                if recognition_result:
                    # Known face - green box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{recognition_result['person_name']} ({confidence:.2f})", 
                              (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # Unknown face - red box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def generate_attendance_report(self, output_dir):
        """Generate attendance report and save to file"""
        if not self.attendance_records:
            print("No attendance records found to generate report")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.attendance_records)
        
        # Sort by timestamp
        df = df.sort_values('Timestamp')
        
        # Create summary
        today = datetime.now().strftime('%Y-%m-%d')
        filename = f"Today's_Attendance_{today}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Format timestamp for CSV
        df_csv = df.copy()
        df_csv['Timestamp'] = df_csv['Timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save detailed report
        df_csv.to_csv(filepath, index=False)
        
        # Create summary report
        summary_filename = f"Attendance_Summary_{today}.txt"
        summary_filepath = os.path.join(output_dir, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            f.write(f"ATTENDANCE REPORT - {today}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Records: {len(df)}\n")
            f.write(f"Unique People: {df['Name'].nunique()}\n")
            f.write(f"Video Processing Time: {datetime.now().strftime('%H:%M:%S')}\n\n")
            
            f.write("PEOPLE DETECTED:\n")
            f.write("-" * 30 + "\n")
            for person in df['Name'].unique():
                person_records = df[df['Name'] == person]
                first_seen = person_records['Timestamp'].min().strftime('%H:%M:%S')
                last_seen = person_records['Timestamp'].max().strftime('%H:%M:%S')
                avg_confidence = person_records['Confidence'].mean()
                f.write(f"{person}: First seen {first_seen}, Last seen {last_seen}, Avg confidence: {avg_confidence:.3f}\n")
            
            f.write("\nDETAILED RECORDS:\n")
            f.write("-" * 50 + "\n")
            for _, record in df.iterrows():
                f.write(f"{record['Name']} - {record['Timestamp'].strftime('%H:%M:%S')} - {record['Attendance']} - Confidence: {record['Confidence']:.3f}\n")
        
        print(f"\n📊 ATTENDANCE REPORTS GENERATED:")
        print(f"📁 Detailed CSV: {filepath}")
        print(f"📄 Summary: {summary_filepath}")
        
        # Print summary to console
        print(f"\n📈 SUMMARY:")
        print(f"Total attendance records: {len(df)}")
        print(f"Unique people detected: {df['Name'].nunique()}")
        print(f"People detected:")
        for person in df['Name'].unique():
            count = len(df[df['Name'] == person])
            avg_conf = df[df['Name'] == person]['Confidence'].mean()
            print(f"  - {person}: {count} detections (avg confidence: {avg_conf:.3f})")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test attendance system with video footage')
    parser.add_argument('video_path', help='Path to the video file to process')
    parser.add_argument('--output', default='attendance_reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Create video tester
    tester = VideoTester()
    
    # Process video
    success = tester.process_video(args.video_path, args.output)
    
    if success:
        print("\n✅ Video processing completed successfully!")
    else:
        print("\n❌ Video processing failed!")

if __name__ == "__main__":
    main()
