========================================================================
             AI Shape Detection & Target Logger System
========================================================================

Project Files:
------------------------
1. opencv_detection_app.py:
   Interactive camera application using OpenCV. Opens webcam feed, draws bounding boxes,
   logs millisecond timestamps to detection_log.txt, and triggers buzzer alert when target shape is detected.

2. detection_logger.py:
   CLI-based detection and logging script without an OpenCV GUI window.

3. best.pt:
   YOLO11m model weights trained on 15 geometric color and shape classes.

4. run_camera.bat:
   Quick launcher script. Double-click to start the application directly and specify the target shape.

5. requirements.txt:
   Required Python packages (ultralytics, opencv-python, numpy).

6. inference.py:
   Standalone script to run inference on static images using best.pt.

7. model_architecture.py:
   PyTorch architecture definition of the YOLO11m model.

------------------------------------------------------------------------
How to Run:
------------------------
- Run Live Camera Detection (OpenCV):
  python opencv_detection_app.py --target red-circle --camera 0

- Run via Batch Launcher:
  Double-click run_camera.bat

- Exit Application:
  Press 'q' on the camera window to safely exit.
