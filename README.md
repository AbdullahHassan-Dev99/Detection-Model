# AI Real-Time Shape Detection & Target Logger

A real-time geometric shape and color detection system powered by **YOLO11m** and **OpenCV**, featuring automatic timestamp logging, visual bounding box overlays, and buzzer alert integration.

---

## 🚀 Overview

This repository provides an end-to-end computer vision solution trained to detect **15 geometric color classes** (circles, squares, and triangles in blue, green, orange, red, and yellow). When a target shape is detected in the video stream, the system logs the detection with millisecond precision and triggers an audio alert (buzzer).

---

## 📁 Repository Structure

| File | Description |
| :--- | :--- |
| `opencv_detection_app.py` | Interactive OpenCV live camera feed with real-time detection, HUD status overlay, and buzzer alert. |
| `detection_logger.py` | CLI-based detection script for headless logging to console and text files. |
| `inference.py` | Standalone script to run predictions on static images. |
| `model_architecture.py` | PyTorch implementation of the YOLO11m architecture (Backbone, SPPF, C2PSA, and Detect Head). |
| `best.pt` | Pretrained YOLO11m weights for the 15 geometric color classes. |
| `run_camera.bat` | One-click Windows batch launcher to select a target shape and start the camera. |
| `requirements.txt` | Python dependencies. |

---

## 🎯 Supported Classes (15 Geometric & Color Combinations)

- **Circles:** `blue-circle`, `green-circle`, `orange-circle`, `red-circle`, `yellow-circle`
- **Squares:** `blue-square`, `green-square`, `orange-square`, `red-square`, `yellow-square`
- **Triangles:** `blue-triangle`, `green-triangle`, `orange-triangle`, `red-triangle`, `yellow-triangle`

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd "detection model"
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Live OpenCV Camera Feed (Recommended)
Run real-time detection specifying your target shape:
```bash
python opencv_detection_app.py --target red-circle --camera 0
```
- Press **`q`** on the camera window to exit safely.

### 2. Windows Batch Launcher
Double-click `run_camera.bat` and enter your desired target shape when prompted.

### 3. CLI Logger Mode
```bash
python detection_logger.py --target blue-square --source 0
```

### 4. Single Image Inference
```bash
python inference.py path/to/sample_image.jpg
```
