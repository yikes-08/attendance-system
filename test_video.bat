@echo off
echo Face Detection Attendance System - Video Testing
echo ================================================
echo.

set /p video_path="Enter the path to your video file: "

if "%video_path%"=="" (
    echo Error: No video path provided
    pause
    exit /b 1
)

if not exist "%video_path%" (
    echo Error: Video file not found at: %video_path%
    pause
    exit /b 1
)

echo.
echo Processing video: %video_path%
echo Press 'q' in the video window to stop processing early
echo.

python test_video.py "%video_path%"

echo.
echo Video processing complete!
echo Check the 'attendance_reports' folder for your results.
echo.
pause
