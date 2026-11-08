@echo off
title AI OpenCV Shape Detector
echo ======================================================
echo    AI Real-Time Shape Detection & Logger App
echo ======================================================
echo.
set /p TARGET="Enter Target Shape (e.g., red-circle, blue-square, yellow-triangle) [default: red-circle]: "
if "%TARGET%"=="" set TARGET=red-circle

echo.
echo Starting OpenCV Camera with Target: %TARGET%
echo Press 'q' on the camera window to exit.
echo.

python opencv_detection_app.py --target %TARGET% --camera 0

pause
