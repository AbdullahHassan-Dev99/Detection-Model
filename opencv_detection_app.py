# ==============================================================================
#  AI OpenCV Live Camera Target Detector & Logger with Buzzer Alert
#  Model: best.pt (15 Geometric Color Classes)
# ==============================================================================

import os
import sys
import time
import logging
from datetime import datetime
import argparse
import cv2

# Buzzer support for Windows
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Ultralytics YOLO import
try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics package is not installed. Please install via: pip install ultralytics")
    sys.exit(1)

# 15 Geometric & Color Classes friendly display names
CLASS_DISPLAY_NAMES = {
    "blue-circle": "Blue Circle",
    "blue-square": "Blue Square",
    "blue-triangle": "Blue Triangle",
    "green-circle": "Green Circle",
    "green-square": "Green Square",
    "green-triangle": "Green Triangle",
    "orange-circle": "Orange Circle",
    "orange-square": "Orange Square",
    "orange-triangle": "Orange Triangle",
    "red-circle": "Red Circle",
    "red-square": "Red Square",
    "red-triangle": "Red Triangle",
    "yellow-circle": "Yellow Circle",
    "yellow-square": "Yellow Square",
    "yellow-triangle": "Yellow Triangle",
}

# Bounding box color palette (BGR format for OpenCV)
CLASS_COLORS = {
    "blue-circle": (255, 100, 0),
    "blue-square": (255, 100, 0),
    "blue-triangle": (255, 100, 0),
    "green-circle": (0, 255, 0),
    "green-square": (0, 255, 0),
    "green-triangle": (0, 255, 0),
    "orange-circle": (0, 140, 255),
    "orange-square": (0, 140, 255),
    "orange-triangle": (0, 140, 255),
    "red-circle": (0, 0, 255),
    "red-square": (0, 0, 255),
    "red-triangle": (0, 0, 255),
    "yellow-circle": (0, 255, 255),
    "yellow-square": (0, 255, 255),
    "yellow-triangle": (0, 255, 255),
}

def setup_logger(log_file="detection_log.txt"):
    """Setup logger to write formatted millisecond timestamps to console and log file."""
    logger = logging.getLogger("OpenCVDetector")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File Handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def trigger_buzzer():
    """Trigger Windows Buzzer beep sound."""
    if HAS_WINSOUND:
        winsound.Beep(2200, 500)
    else:
        sys.stdout.write('\a')
        sys.stdout.flush()

def start_opencv_detection(target_shape, camera_id=0, model_path="best.pt", conf_thresh=0.25, log_file="detection_log.txt"):
    """Open CV2 camera stream, detect objects using YOLO, draw boxes, log events, and beep on target."""
    logger = setup_logger(log_file)
    display_target = CLASS_DISPLAY_NAMES.get(target_shape.lower(), target_shape)
    
    logger.info("==================================================")
    logger.info("Starting AI OpenCV Real-time Detection Application")
    logger.info(f"Target Shape Specified: '{target_shape}' ({display_target})")
    logger.info(f"Log Output File: {os.path.abspath(log_file)}")
    logger.info(f"Opening Camera Stream (ID: {camera_id})...")
    logger.info("==================================================")

    if not os.path.exists(model_path):
        logger.error(f"Model file '{model_path}' not found! Please make sure best.pt exists.")
        return

    logger.info(f"Loading YOLO Model from {model_path}...")
    model = YOLO(model_path)
    logger.info("YOLO Model loaded successfully!")

    # Initialize OpenCV Camera VideoCapture
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"Cannot open camera with ID {camera_id}. Check webcam connection.")
        return

    # Set camera resolution (640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    logger.info("Webcam feed active. Press 'q' on the OpenCV window to exit.")
    
    frame_count = 0
    last_beep_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from webcam feed.")
                break

            frame_count += 1
            
            # Run YOLO prediction on current frame
            results = model.predict(source=frame, conf=conf_thresh, verbose=False)
            
            detected_objects = []
            target_detected = False
            target_info = None

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    confidence = float(box.conf[0].item()) * 100
                    display_name = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)

                    # Get coordinates
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    bbox = [x1, y1, x2, y2]

                    detected_objects.append(f"{cls_name} ({confidence:.1f}%)")

                    # Default box color
                    box_color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                    thickness = 2

                    # Check if target shape matched
                    if cls_name.lower() == target_shape.lower():
                        target_detected = True
                        target_info = {
                            "name": cls_name,
                            "display": display_name,
                            "conf": confidence,
                            "bbox": bbox
                        }
                        # Highlight target box with thick border
                        box_color = (0, 0, 255) if cls_name.startswith("red") else (0, 255, 0)
                        thickness = 4

                    # Draw Bounding Box on OpenCV Frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

                    # Draw Label Tag Background & Text
                    label_text = f"{cls_name} {confidence:.1f}%"
                    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), box_color, -1)
                    cv2.putText(frame, label_text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Log frame detection status with Timestamp
            if detected_objects:
                logger.info(f"[Frame #{frame_count}] Detected: {', '.join(detected_objects)}")
            else:
                logger.info(f"[Frame #{frame_count}] Scanning... Target '{target_shape}' not in frame.")

            # Top Status Bar Overlay on OpenCV Window
            status_text = f"Target: {target_shape} | Status: {'SEARCHING...' if not target_detected else 'TARGET DETECTED!'}"
            status_color = (0, 0, 255) if target_detected else (255, 255, 255)
            cv2.rectangle(frame, (0, 0), (640, 40), (40, 40, 40), -1)
            cv2.putText(frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            # Trigger Target Detection Alert & Buzzer
            if target_detected:
                # Big Alert Overlay on Screen
                cv2.rectangle(frame, (100, 200), (540, 280), (0, 0, 255), -1)
                cv2.putText(frame, "TARGET DETECTED!", (120, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

                logger.info("--------------------------------------------------")
                logger.info(f"🚨 TARGET DETECTED: {target_info['name']} ({target_info['display']})")
                logger.info(f"📊 Confidence: {target_info['conf']:.2f}% | BBox: {target_info['bbox']}")
                logger.info("🔔 BUZZER BEEP ALERT!")
                logger.info("--------------------------------------------------")

                # Sound Buzzer (limit sound rate to once every 1 second)
                current_time = time.time()
                if current_time - last_beep_time > 1.0:
                    trigger_buzzer()
                    last_beep_time = current_time

            # Show OpenCV Window
            cv2.imshow("AI Shape Detection System - OpenCV", frame)

            # Press 'q' key to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Exit key 'q' pressed by user.")
                break

    except Exception as e:
        logger.error(f"Error in OpenCV video stream loop: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam released and OpenCV windows closed clean.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI OpenCV Live Camera Target Detector")
    parser.add_argument("--target", type=str, default="red-circle", help="Target shape name (e.g. red-circle, blue-square)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device ID (default 0)")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to best.pt model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25)")
    parser.add_argument("--logfile", type=str, default="detection_log.txt", help="Log output filename")

    args = parser.parse_args()

    start_opencv_detection(
        target_shape=args.target,
        camera_id=args.camera,
        model_path=args.model,
        conf_thresh=args.conf,
        log_file=args.logfile
    )
