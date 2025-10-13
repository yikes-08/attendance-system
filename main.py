import argparse
import os
from attendance_system import AttendanceSystem
from enroll_dataset import enroll_new_user  # This import will now succeed
from test_video import VideoTester

def main():
    parser = argparse.ArgumentParser(description="Face Recognition Attendance System")
    parser.add_argument("--start", action="store_true", help="Start the real-time attendance system.")
    parser.add_argument("--enroll", type=str, help="Path to the dataset directory for enrolling new users.")
    parser.add_argument("--test-video", type=str, help="Path to a video file to test the system.")
    
    parser.add_argument(
        "--camera", 
        type=str, 
        default="webcam0", 
        choices=["realsense", "webcam0", "webcam1"],
        help="Specify the camera source: 'realsense', 'webcam0' (built-in), or 'webcam1' (external)."
    )
    
    args = parser.parse_args()
    
    if args.start:
        system = AttendanceSystem()
        
        if args.camera == "realsense":
            print("Selected camera: Intel RealSense")
            system.run(use_realsense=True)
        elif args.camera == "webcam0":
            print("Selected camera: Built-in Webcam (Index 0)")
            system.run(use_realsense=False, camera_index=0)
        elif args.camera == "webcam1":
            print("Selected camera: External Webcam (Index 1)")
            system.run(use_realsense=False, camera_index=1)
            
    elif args.enroll:
        if os.path.isdir(args.enroll):
            enroll_new_user(args.enroll) # This call is now valid
        else:
            print(f"Error: The provided path '{args.enroll}' is not a valid directory.")
            
    elif args.test_video:
        tester = VideoTester()
        tester.process_video(args.test_video)
        
    else:
        print("No action specified. Use --start, --enroll, or --test-video.")
        parser.print_help()

if __name__ == "__main__":
    main()