"""
CPU-based person detector using ONNXRuntime.

Performs the same preprocessing that Ultralytics YOLOv8 uses:
  1. Letterbox resize to MODEL_IMGSZ x MODEL_IMGSZ
  2. BGR -> RGB
  3. HWC -> CHW, float32 [0, 1]
  4. Add batch dimension

Postprocessing:
  YOLOv8 ONNX output shape is [1, 84, 8400] for imgsz=640.
  - 84 = 4 (cx, cy, w, h) + 80 (class scores)
  - 8400 = number of candidate boxes
  We transpose to [8400, 84], filter person class, apply confidence
  threshold, then NMS.
"""

import time
import numpy as np
import cv2
import config
from detection import Detection

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def _letterbox(img: np.ndarray, new_shape: int):
    """Resize with padding to maintain aspect ratio. Returns (padded_img, ratio, (dw, dh))."""
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_unpad_w = int(round(w * r))
    new_unpad_h = int(round(h * r))
    dw = (new_shape - new_unpad_w) / 2
    dh = (new_shape - new_unpad_h) / 2

    if (w, h) != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Standard greedy NMS. boxes: [N, 4] as x1y1x2y2, scores: [N]."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]
    return keep


class CPUDetector:
    """ONNXRuntime-based YOLOv8 person detector."""

    def __init__(self, model_path: str | None = None):
        if ort is None:
            raise ImportError("onnxruntime is not installed. "
                              "Install with: pip install onnxruntime")

        path = model_path or config.ONNX_MODEL_PATH
        print(f"[detector_cpu] Loading ONNX model: {path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = config.ONNX_NUM_THREADS
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(path, sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.imgsz = config.MODEL_IMGSZ

        # Warm-up
        dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})
        print("[detector_cpu] Model loaded and warmed up")

        self.last_inference_ms = 0.0
        self.last_e2e_ms = 0.0
        self.model_name = path

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run full pipeline: preprocess -> infer -> postprocess."""
        t_start = time.perf_counter()

        # --- Preprocess ---
        img_lb, ratio, (dw, dh) = _letterbox(frame, self.imgsz)
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        blob = img_rgb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, H, W]

        # --- Inference ---
        t_infer = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: blob})
        self.last_inference_ms = (time.perf_counter() - t_infer) * 1000.0

        # --- Postprocess ---
        # YOLOv8 output: [1, 84, N] -> transpose to [N, 84]
        preds = outputs[0]                          # [1, 84, 8400]
        preds = np.squeeze(preds, axis=0).T         # [8400, 84]

        # Columns: cx, cy, w, h, class_scores[0..79]
        cx = preds[:, 0]
        cy = preds[:, 1]
        w  = preds[:, 2]
        h  = preds[:, 3]
        class_scores = preds[:, 4:]                 # [8400, 80]

        # Person class only
        person_scores = class_scores[:, config.PERSON_CLASS_ID]
        mask = person_scores >= config.MODEL_CONF
        if not np.any(mask):
            self.last_e2e_ms = (time.perf_counter() - t_start) * 1000.0
            return []

        cx = cx[mask]
        cy = cy[mask]
        w  = w[mask]
        h  = h[mask]
        scores = person_scores[mask]

        # Convert cxcywh -> x1y1x2y2 in letterboxed coords
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # NMS
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(boxes, scores, config.MODEL_IOU)
        boxes = boxes[keep]
        scores = scores[keep]

        # Map back to original frame coordinates
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio

        # Clip to frame
        fh, fw = frame.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, fw)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, fh)

        self.last_e2e_ms = (time.perf_counter() - t_start) * 1000.0

        return [Detection(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(s))
                for b, s in zip(boxes, scores)]
