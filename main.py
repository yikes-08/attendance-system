import argparse
import sys
import os

# Add the project's root directory to the Python path to ensure all modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main entry point for the Face Recognition Attendance System.
    Provides command-line arguments to run different components of the system.
    """
    parser = argparse.ArgumentParser(
        description="Face Recognition Attendance System",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  - To run the real-time attendance camera:
    python main.py

  - To enroll new faces from a dataset folder:
    python main.py --enroll ./path/to/dataset

  - To process a pre-recorded video and generate a report:
    python main.py --test-video ./path/to/video.mp4

  - To verify your email configuration in config.py:
    python main.py --test-email
"""
    )

    parser.add_argument(
        '--enroll',
        type=str,
        metavar="PATH",
        help="Path to the dataset folder to enroll new faces."
    )
    parser.add_argument(
        '--test-video',
        type=str,
        metavar="PATH",
        help="Path to a video file to process for attendance."
    )
    parser.add_argument(
        '--test-email',
        action='store_true',
        help="Sends a test email to verify SMTP configuration."
    )

    args = parser.parse_args()

    # --- Logic to run different parts of the application ---

    if args.enroll:
        from enroll_dataset import enroll_from_folder
        if not os.path.isdir(args.enroll):
            print(f"❌ Error: Dataset directory not found at '{args.enroll}'")
            return
        print(f"🚀 Starting enrollment process for folder: {args.enroll}")
        enroll_from_folder(args.enroll)
        print("✅ Enrollment complete.")
        return

    if args.test_video:
        from test_video import VideoTester
        if not os.path.isfile(args.test_video):
            print(f"❌ Error: Video file not found at '{args.test_video}'")
            return
        print(f"🎥 Starting video testing for file: {args.test_video}")
        tester = VideoTester()
        tester.process_video(args.test_video)
        return

    if args.test_email:
        from email_notification import EmailNotifier
        print("📧 Testing email configuration...")
        notifier = EmailNotifier()
        if notifier.test_email_connection():
            print("✅ Email configuration is working correctly!")
        else:
            print("❌ Email configuration failed. Please check your settings in config.py")
        return

    # --- If no arguments are provided, run the main real-time system ---
    try:
        from attendance_system import AttendanceSystem
        print("✨ Starting the real-time attendance system...")
        print("Press 'q' in the camera window to exit.")
        system = AttendanceSystem()
        system.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Shutting down.")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
    finally:
        print("System closed.")


if __name__ == "__main__":
    main()