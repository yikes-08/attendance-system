#!/usr/bin/env python3
"""
Face Detection Biometric Attendance System
Using YuNet for face detection and FaceNet for face recognition
"""

import sys
import os
import argparse
from attendance_system import AttendanceSystem

def main():
    parser = argparse.ArgumentParser(description='Face Detection Biometric Attendance System')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--test-email', action='store_true', help='Test email configuration')
    parser.add_argument('--add-person', action='store_true', help='Add a new person to the system')
    parser.add_argument('--send-report', action='store_true', help='Send attendance report manually')
    parser.add_argument('--stats', action='store_true', help='Show attendance statistics')
    
    args = parser.parse_args()
    
    # Create attendance system
    attendance_system = AttendanceSystem()
    
    if args.test_email:
        print("Testing email configuration...")
        if attendance_system.email_notifier.test_email_connection():
            print("Email configuration is working correctly!")
        else:
            print("Email configuration failed. Please check your settings in config.py")
        return
    
    if args.add_person:
        print("Adding new person to the system...")
        # This would require camera access and interactive input
        print("Please run the main system and press 'a' to add a new person interactively.")
        return
    
    if args.send_report:
        print("Sending manual attendance report...")
        attendance_system.send_manual_report()
        return
    
    if args.stats:
        print("Attendance Statistics:")
        stats = attendance_system.get_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        return
    
    # Run the main attendance system
    try:
        attendance_system.run()
    except KeyboardInterrupt:
        print("\nSystem interrupted by user")
        attendance_system.cleanup()
    except Exception as e:
        print(f"Error running attendance system: {e}")
        attendance_system.cleanup()

if __name__ == "__main__":
    main()
