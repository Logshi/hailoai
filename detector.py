"""
Person detector using Ultralytics YOLOv8-nano.

Wraps model loading, inference, and result parsing.
Only returns detections for class 0 (person).
"""

import time
import numpy as np
from ultralytics import YOLO
import config


class Detection:
    """Single detected person."""
    __slots__ = ("x1", "y1", "x2", "y2", "confidence")

    def __init__(self, x1, y1, x2, y2, confidence):
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.confidence = float(confidence)


class PersonDetector:
    def __init__(self, model_path=None):
        path = model_path or config.MODEL_PATH
        print(f"[detector] Loading model: {path}  (imgsz={config.MODEL_IMGSZ})")
        self.model = YOLO(path)
        # Warm-up: run one dummy inference so first real frame isn't slow.
        dummy = np.zeros((config.MODEL_IMGSZ, config.MODEL_IMGSZ, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=config.MODEL_IMGSZ, verbose=False)
        print("[detector] Model loaded and warmed up")

        self.last_inference_ms = 0.0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a BGR frame. Returns list of Detection objects."""
        t0 = time.perf_counter()

        results = self.model.predict(
            frame,
            imgsz=config.MODEL_IMGSZ,
            conf=config.MODEL_CONF,
            iou=config.MODEL_IOU,
            classes=[config.PERSON_CLASS_ID],
            verbose=False,
        )

        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                detections.append(Detection(xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf))

        return detections
