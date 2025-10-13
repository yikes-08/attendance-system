# test_performance.py
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Ensure local modules can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from face_detection import FaceDetector
from face_recognition import FaceRecognizer
from simple_tracker import SimpleTracker

def run_performance_test(video_path, config, duration_seconds=10):
    """
    Runs face detection and tracking on a video for a fixed duration
    and returns the average FPS.
    """
    print(f"--- Testing Config: {config['name']} ---")
    
    # Unpack config
    frame_width, frame_height, det_size = config['params']
    
    # Initialize components
    try:
        import onnxruntime as ort
        has_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
    except Exception:
        has_cuda = False
    
    detector = FaceDetector(det_size=det_size, use_gpu=has_cuda)
    tracker = SimpleTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return 0

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    total_frames_to_process = int(fps_video * duration_seconds)
    
    start_time = time.time()
    
    while frame_count < total_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame according to config
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        detections = detector.detect_faces(frame)
        tracker.update(detections, frame.shape)
        
        frame_count += 1

    end_time = time.time()
    cap.release()
    
    elapsed_time = end_time - start_time
    avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
    
    print(f"Processed {frame_count} frames in {elapsed_time:.2f}s. Average FPS: {avg_fps:.2f}\n")
    return avg_fps

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_performance.py <path_to_video_file>")
        sys.exit(1)
        
    video_file = sys.argv[1]
    
    # Define the configurations to test
    configs_to_test = [
        {
            "name": "Low Res (480x360)",
            # ✅ FIX: Changed detection size from (320, 240) to (320, 224)
            "params": (480, 360, (320, 224)) 
        },
        {
            "name": "Standard (640x480)",
            "params": (640, 480, (640, 480))
        },
        {
            "name": "High Res (1280x720)",
            "params": (1280, 720, (640, 640)) 
        },
    ]

    results = {}
    for config in configs_to_test:
        fps = run_performance_test(video_file, config)
        results[config['name']] = fps

    # Plotting the results
    names = list(results.keys())
    values = list(results.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, values, color=['#ff9999','#66b3ff','#99ff99'])
    plt.ylabel('Average Frames Per Second (FPS)')
    plt.xlabel('Resolution Configuration')
    plt.title('System Performance vs. Input Resolution')
    plt.ylim(0, max(values) * 1.2 if values else 10) 

    # Add FPS value on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f} FPS', va='bottom', ha='center')

    # Save the plot
    plt.savefig("performance_graph.png")
    print("✅ Performance graph saved to performance_graph.png")
    plt.show()