# realsense_camera.py
import pyrealsense2 as rs
import numpy as np

class RealSenseCamera:
    def __init__(self, width=1280, height=720, fps=30):
        print("📷 Initializing Intel RealSense Camera...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Check if a RealSense device is connected
        ctx = rs.context()
        if len(ctx.devices) == 0:
            raise RuntimeError("No RealSense device connected.")
        
        # Configure the color stream from the camera
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        self.is_opened_flag = False

    def start(self):
        """Starts the camera pipeline."""
        try:
            self.pipeline.start(self.config)
            self.is_opened_flag = True
            print("✅ RealSense camera stream started.")
        except Exception as e:
            print(f"❌ Failed to start RealSense stream: {e}")
            self.is_opened_flag = False
            
    def isOpened(self):
        """Mimics the isOpened() method of cv2.VideoCapture."""
        return self.is_opened_flag

    def read(self):
        """
        Waits for a coherent color frame and returns it.
        Returns a tuple (bool, frame), mimicking cv2.VideoCapture.read().
        """
        if not self.is_opened_flag:
            return (False, None)
            
        try:
            # Wait for a new frame from the camera
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                return (False, None)
            
            # Convert the frame to a numpy array that OpenCV can use
            image = np.asanyarray(color_frame.get_data())
            return (True, image)
            
        except Exception as e:
            print(f"Error reading frame from RealSense camera: {e}")
            return (False, None)

    def release(self):
        """Stops the camera pipeline."""
        if self.is_opened_flag:
            print("Stopping RealSense camera stream...")
            self.pipeline.stop()
            self.is_opened_flag = False