"""
Centralized configuration for the person detection pipeline.
Supports two inference modes: cpu (ONNXRuntime) and hailo (HailoRT).
"""

import os

# --- Mode ---
# Set via environment variable: MODE=cpu or MODE=hailo
MODE = os.environ.get("MODE", "cpu").lower()

# --- RTSP / Camera ---
# Set via environment variable or .env file. Do NOT hardcode credentials here.
RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://user:password@192.168.1.100:554/stream1"  # placeholder
)

# --- Capture ---
CAPTURE_QUEUE_SIZE = 2
CAPTURE_RECONNECT_DELAY = 5.0

# --- Model (shared) ---
MODEL_IMGSZ = 640               # Native YOLO input. Both modes must match.
MODEL_CONF = 0.40
MODEL_IOU = 0.45
PERSON_CLASS_ID = 0              # COCO class 0 = person

# --- CPU mode (ONNXRuntime) ---
ONNX_MODEL_PATH = "yolov8n.onnx"
ONNX_NUM_THREADS = 4             # ORT intra-op threads (match core count)

# --- Hailo mode ---
HEF_MODEL_PATH = "yolov8n.hef"  # Pre-compiled Hailo Executable Format
HAILO_BATCH_SIZE = 1

# --- Display ---
SHOW_VIDEO = False
CONSOLE_UPDATE_INTERVAL = 1.0

# --- Monitor ---
MONITOR_SAMPLE_INTERVAL = 1.0

# --- Logging ---
LOG_DIR = "logs"                 # CSV logs written here
LOG_INTERVAL = 1.0               # One CSV row per second

# --- Performance ---
SKIP_FRAMES = 0
