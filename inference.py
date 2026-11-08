# =========================================================
#  YOLO11m Object Detection - Inference Script
#  Model: best.pt (Trained on 15 Geometric Color Classes)
# =========================================================

"""
Inference Script:
Loads the trained model weights (best.pt) and runs inference
on input images to detect objects across the 15 geometric color classes and draw bounding boxes.
"""

import sys  # System module for reading command line arguments
from ultralytics import YOLO  # Ultralytics YOLO module for model loading and prediction

def main():  # Main execution function
    # 1. Define and load model weights
    model_path = "best.pt"  # Path to trained model weights
    print(f"Loading model from file: {model_path}...")
    model = YOLO(model_path)  # Load model and initialize YOLO11 network in memory

    # 2. Display model information and list of 15 classes
    print("\n--- Model Information & Class List ---")
    print(f"Total number of classes (nc): {model.model.nc}")
    print("Class names:")
    for idx, name in model.names.items():
        print(f"  [{idx}]: {name}")

    # 3. Perform detection and prediction if an image path is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nRunning detection on image: {image_path}")
        results = model.predict(source=image_path, save=True, conf=0.25)
        print(f"Operation completed successfully and image saved. Detected objects: {len(results[0].boxes)}")
    else:
        print("\nTo run inference on an image, use the following command:")
        print("python inference.py <path_to_image.jpg>")

if __name__ == "__main__":
    main()
