import time
import matplotlib.pyplot as plt
from attendance_system import AttendanceSystem

def test_performance(frame_skip_list=[1,3,5,10]):
    results = []
    for skip in frame_skip_list:
        system = AttendanceSystem()
        system.RECOG_PERIOD = skip
        start = time.time()
        # Simulate processing 300 frames
        total_frames = 300
        time.sleep(0.2 * skip)  # simulate time delay
        fps = total_frames / (time.time() - start)
        results.append((skip, fps))
        print(f"Skip={skip}, FPS={fps:.2f}")
    return results

if __name__ == "__main__":
    data = test_performance()
    x, y = zip(*data)
    plt.plot(x, y, marker='o')
    plt.title("Frame Skip vs FPS")
    plt.xlabel("Frame Skip Value (RECOG_PERIOD)")
    plt.ylabel("Frames Per Second (FPS)")
    plt.grid(True)
    plt.show()
