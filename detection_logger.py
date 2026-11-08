# ==============================================================================
#  AI Object Detection & Target Logger with Buzzer Alert & Timestamps
#  Model: best.pt (Trained on 15 Geometric Color Classes)
# ==============================================================================

import os
import sys
import time
import logging
from datetime import datetime
import argparse

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

# Class dictionary (15 Geometric & Color Classes friendly display names)
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

def setup_logger(log_file="detection_log.txt"):
    """Setup logging to file and console with millisecond timestamps."""
    logger = logging.getLogger("ShapeDetector")
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
    """Trigger Windows Buzzer beep (2000Hz for 800ms) or fallback bell."""
    if HAS_WINSOUND:
        winsound.Beep(2000, 800)
    else:
        sys.stdout.write('\a')
        sys.stdout.flush()

def run_detection(target_shape, source="0", model_path="best.pt", conf_thresh=0.25, log_file="detection_log.txt"):
    """Run object detection loop, logging shapes and alerting on target detection."""
    logger = setup_logger(log_file)
    display_target = CLASS_DISPLAY_NAMES.get(target_shape.lower(), target_shape)
    
    logger.info("==================================================")
    logger.info("Starting AI Object Detection & Target Logger System")
    logger.info(f"Log File Path: {os.path.abspath(log_file)}")
    logger.info(f"Target Shape Specified: {target_shape} ({display_target})")
    logger.info(f"Input Source: {source}")
    logger.info("==================================================")

    if not os.path.exists(model_path):
        logger.error(f"Model weights file not found: {model_path}")
        return

    logger.info(f"Loading YOLO model from {model_path}...")
    model = YOLO(model_path)
    logger.info("YOLO Model loaded successfully!")

    if str(source).isdigit():
        source = int(source)

    frame_count = 0

    try:
        for result in model.predict(source=source, stream=True, conf=conf_thresh, verbose=False):
            frame_count += 1
            detected_objects = []
            target_detected_in_frame = False
            target_info = None

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    cls_name = model.names.get(cls_id, f"class_{cls_id}")
                    confidence = float(box.conf[0].item()) * 100
                    display_name = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)
                    bbox = [round(coord, 1) for coord in box.xyxy[0].tolist()]

                    detected_objects.append(f"{cls_name} ({display_name}) [{confidence:.1f}%]")

                    if cls_name.lower() == target_shape.lower():
                        target_detected_in_frame = True
                        target_info = {
                            "name": cls_name,
                            "display": display_name,
                            "conf": confidence,
                            "bbox": bbox
                        }

            # Log status of current frame
            if detected_objects:
                objs_str = ", ".join(detected_objects)
                logger.info(f"[Frame #{frame_count}] Scanning... Detected: {objs_str}")
            else:
                logger.info(f"[Frame #{frame_count}] Scanning... Searching for '{target_shape}' (No shapes detected).")

            # Target detection alert
            if target_detected_in_frame:
                logger.info("--------------------------------------------------")
                logger.info(f"🚨 TARGET DETECTED! Found target shape '{target_info['name']}' ({target_info['display']})!")
                logger.info(f"📊 Confidence: {target_info['conf']:.2f}%")
                logger.info(f"📍 Bounding Box: {target_info['bbox']}")
                logger.info("🔔 Triggering Buzzer Alarm (winsound.Beep)...")
                logger.info("--------------------------------------------------")
                
                trigger_buzzer()

            time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("Detection process stopped by user.")
    except Exception as e:
        logger.error(f"Error during detection execution: {e}")

    logger.info("Detection process ended.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Target Detection Logger & Buzzer Alert")
    parser.add_argument("--target", type=str, default="red-circle", help="Target shape (e.g., red-circle, blue-square)")
    parser.add_argument("--source", type=str, default="0", help="Input source (0 for webcam or path to image/video)")
    parser.add_argument("--model", type=str, default="best.pt", help="Path to YOLO best.pt model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--logfile", type=str, default="detection_log.txt", help="Log filename")

    args = parser.parse_args()
    
    run_detection(
        target_shape=args.target,
        source=args.source,
        model_path=args.model,
        conf_thresh=args.conf,
        log_file=args.logfile
    )
