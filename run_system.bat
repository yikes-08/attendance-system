@echo off
REM This batch file provides a menu to run the attendance system.

:menu
cls
echo ===================================================
echo      Face Recognition Attendance System Menu
echo ===================================================
echo.
echo   1. Start Real-Time Attendance System
echo   2. Enroll New Faces from a Dataset
echo   3. Test a Pre-recorded Video File
echo   4. Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" goto start_menu
if "%choice%"=="2" goto enroll
if "%choice%"=="3" goto test_video
if "%choice%"=="4" goto exit
goto menu

:start_menu
cls
echo ===================================================
echo          Select Camera Source
echo ===================================================
echo.
echo   1. Intel RealSense Camera
echo   2. Laptop Webcam (Default)
echo   3. External USB Webcam
echo   4. Back to Main Menu
echo.
set /p camera_choice="Enter your choice (1-4): "

if "%camera_choice%"=="1" (
    python main.py --start --camera realsense
    pause
    goto menu
)
if "%camera_choice%"=="2" (
    python main.py --start --camera webcam0
    pause
    goto menu
)
if "%camera_choice%"=="3" (
    python main.py --start --camera webcam1
    pause
    goto menu
)
if "%camera_choice%"=="4" goto menu
goto start_menu

:enroll
cls
echo ===================================================
echo           Enroll New Faces
echo ===================================================
echo.
set /p dataset_path="Enter the full path to the dataset folder: "
python main.py --enroll "%dataset_path%"
echo.
echo Enrollment process finished. Press any key to return to the menu.
pause
goto menu

:test_video
cls
echo ===================================================
echo           Test a Video File
echo ===================================================
echo.
set /p video_path="Enter the full path to the video file: "
python main.py --test-video "%video_path%"
echo.
echo Video processing finished. Press any key to return to the menu.
pause
goto menu

:exit
exit /b