"""
Hailo-8 accelerated person detector using HailoRT Python API.

Supports two HEF output formats:
  A) NMS-postprocessed: shape (num_classes, 5, max_detections)
     - 5 values = y1, x1, y2, x2, confidence (normalized 0-1)
     - Already NMS'd by the Hailo chip, no CPU-side NMS needed
  B) Raw YOLOv8: shape (1, 84, N) or multiple head tensors
     - Requires CPU-side NMS (same as ONNX path)

Prerequisites:
  - Hailo-8 module connected (M.2 / RPi AI Kit)
  - HailoRT >= 4.17 installed
  - A compiled .hef file

Environment setup on Raspberry Pi 5:
  source ~/hailo-apps/setup_env.sh
"""

import time
import numpy as np
import cv2
import config
from detection import Detection

try:
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        ConfigureParams,
        InferVStreams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False


def _letterbox(img: np.ndarray, new_shape: int):
    """Resize with padding, identical to CPU mode for fair comparison."""
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
    """Standard greedy NMS (only used for raw-output HEFs)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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


class HailoDetector:
    """Hailo-8 accelerated YOLOv8 person detector."""

    def __init__(self, hef_path: str | None = None):
        if not HAILO_AVAILABLE:
            raise ImportError(
                "HailoRT Python bindings not found.\n"
                "Install: pip install hailort\n"
                "Or: source ~/hailo-apps/setup_env.sh"
            )

        path = hef_path or config.HEF_MODEL_PATH
        print(f"[detector_hailo] Loading HEF: {path}")

        # --- Open device ---
        self._vdevice = VDevice()

        # --- Load HEF ---
        self._hef = HEF(path)

        # --- Configure network group ---
        configure_params = ConfigureParams.create_from_hef(
            hef=self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._vdevice.configure(self._hef, configure_params)[0]

        # --- Discover input/output metadata ---
        self._input_vstream_info = self._hef.get_input_vstream_infos()
        self._output_vstream_info = self._hef.get_output_vstream_infos()

        input_info = self._input_vstream_info[0]
        self._input_name = input_info.name
        self._input_shape = input_info.shape  # e.g. (640, 640, 3)
        self.imgsz = self._input_shape[0]

        print(f"[detector_hailo] Input: {self._input_name} shape={self._input_shape}")

        # Detect output format
        self._nms_output = False
        self._output_shapes = {}
        for oi in self._output_vstream_info:
            print(f"[detector_hailo] Output: {oi.name} shape={oi.shape}")
            self._output_shapes[oi.name] = oi.shape

        # Check if this is an NMS-postprocessed HEF:
        # Single output with shape (num_classes, 5, max_det) e.g. (80, 5, 100)
        if len(self._output_vstream_info) == 1:
            oi = self._output_vstream_info[0]
            if len(oi.shape) == 3 and oi.shape[1] == 5:
                self._nms_output = True
                self._num_classes = oi.shape[0]
                self._max_det = oi.shape[2]
                print(f"[detector_hailo] NMS-postprocessed output detected: "
                      f"{self._num_classes} classes, max {self._max_det} detections/class")

        # --- Create vstream params ---
        self._input_params = InputVStreamParams.make(
            self._network_group,
            quantized=False,
            format_type=FormatType.UINT8,
        )
        self._output_params = OutputVStreamParams.make(
            self._network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        # --- Activate network group (required before inference) ---
        self._ng_context = self._network_group.activate()
        self._ng_context.__enter__()

        # --- Warm up ---
        dummy = np.zeros((1, *self._input_shape), dtype=np.uint8)
        self._run_inference(dummy)
        print("[detector_hailo] Model loaded and warmed up")

        self.last_inference_ms = 0.0
        self.last_e2e_ms = 0.0
        self.model_name = path

    def _run_inference(self, input_data: np.ndarray) -> dict:
        """Send a batch through Hailo and return output dict {name: ndarray}."""
        input_dict = {self._input_name: input_data}
        with InferVStreams(self._network_group,
                          self._input_params,
                          self._output_params) as pipeline:
            results = pipeline.infer(input_dict)
        return results

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Full pipeline: preprocess -> Hailo infer -> postprocess."""
        t_start = time.perf_counter()

        # --- Preprocess ---
        img_lb, ratio, (dw, dh) = _letterbox(frame, self.imgsz)
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        input_batch = img_rgb[np.newaxis, ...].astype(np.uint8)

        # --- Inference on Hailo ---
        t_infer = time.perf_counter()
        raw_outputs = self._run_inference(input_batch)
        self.last_inference_ms = (time.perf_counter() - t_infer) * 1000.0

        # --- Postprocess ---
        fh, fw = frame.shape[:2]

        if self._nms_output:
            detections = self._decode_nms_output(raw_outputs, ratio, dw, dh, fw, fh)
        else:
            detections = self._decode_raw_output(raw_outputs, ratio, dw, dh, fw, fh)

        self.last_e2e_ms = (time.perf_counter() - t_start) * 1000.0
        return detections

    def _decode_nms_output(self, raw_outputs: dict,
                           ratio: float, dw: float, dh: float,
                           fw: int, fh: int) -> list[Detection]:
        """
        Decode Hailo YOLOv8 NMS-postprocessed output.

        HailoRT returns a list of length num_classes (80 for COCO).
        Each element is an ndarray of shape (N_detections, 5) where
        columns are [y1, x1, y2, x2, score] in pixel coordinates
        relative to the model input size (e.g. 640x640).
        If a class has no detections, the array may be empty or None.

        We only extract person detections (class 0).
        """
        output_name = list(raw_outputs.keys())[0]
        preds = raw_outputs[output_name]

        # Debug: print structure once to understand the format
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print(f"[debug] output type: {type(preds)}")
            if isinstance(preds, (list, tuple)):
                print(f"[debug] len: {len(preds)}")
                first = preds[0]
                print(f"[debug] preds[0] type: {type(first)}")
                if isinstance(first, (list, tuple)):
                    print(f"[debug] preds[0] len: {len(first)}")
                    inner = first[0] if len(first) > 0 else None
                    print(f"[debug] preds[0][0] type: {type(inner)}")
                    if hasattr(inner, 'shape'):
                        print(f"[debug] preds[0][0] shape: {inner.shape}")
                elif hasattr(first, 'shape'):
                    print(f"[debug] preds[0] shape: {first.shape}")
            elif hasattr(preds, 'shape'):
                print(f"[debug] preds shape: {preds.shape}")

        # Unwrap nested structure: dict -> list(batch) -> list(classes) -> ndarray
        # Peel off batch dimension if present
        if isinstance(preds, (list, tuple)) and len(preds) > 0:
            first = preds[0]
            if isinstance(first, (list, tuple)) and len(first) > 0:
                # Nested: preds[batch][class] -> ndarray
                class_list = first
            elif isinstance(first, np.ndarray):
                # Flat: preds[class] -> ndarray
                class_list = preds
            else:
                return []
        else:
            return []

        # Extract person class
        if config.PERSON_CLASS_ID >= len(class_list):
            return []

        person_preds = class_list[config.PERSON_CLASS_ID]
        if person_preds is None:
            return []

        person_preds = np.asarray(person_preds, dtype=np.float32)
        if person_preds.ndim != 2 or person_preds.shape[0] == 0:
            return []

        # Each row: [y1, x1, y2, x2, score] in model-input pixel coords
        # OR:       [x1, y1, x2, y2, score] — we detect which format below
        scores = person_preds[:, 4]
        mask = scores >= config.MODEL_CONF
        if not np.any(mask):
            return []

        person_preds = person_preds[mask]
        scores = person_preds[:, 4]

        # Hailo NMS typically outputs [y1, x1, y2, x2, score]
        y1_px = person_preds[:, 0]
        x1_px = person_preds[:, 1]
        y2_px = person_preds[:, 2]
        x2_px = person_preds[:, 3]

        # Map from letterboxed model-input coords to original frame coords
        x1_orig = (x1_px - dw) / ratio
        y1_orig = (y1_px - dh) / ratio
        x2_orig = (x2_px - dw) / ratio
        y2_orig = (y2_px - dh) / ratio

        # Clip to frame
        x1_orig = np.clip(x1_orig, 0, fw)
        y1_orig = np.clip(y1_orig, 0, fh)
        x2_orig = np.clip(x2_orig, 0, fw)
        y2_orig = np.clip(y2_orig, 0, fh)

        return [Detection(int(x1), int(y1), int(x2), int(y2), float(s))
                for x1, y1, x2, y2, s
                in zip(x1_orig, y1_orig, x2_orig, y2_orig, scores)]

    def _decode_raw_output(self, raw_outputs: dict,
                           ratio: float, dw: float, dh: float,
                           fw: int, fh: int) -> list[Detection]:
        """
        Decode raw (non-NMS) YOLOv8 output tensors.
        Handles single [1, 84, N] or multiple head tensors.
        """
        output_names = list(raw_outputs.keys())

        # Single tensor [1, 84, N]
        if len(output_names) == 1:
            preds = raw_outputs[output_names[0]]
            if preds.ndim == 3 and preds.shape[1] == 84:
                return self._decode_yolov8_raw(
                    np.squeeze(preds, 0).T, ratio, dw, dh, fw, fh)

        # Multiple heads: flatten and concatenate
        all_preds = []
        for name in sorted(output_names):
            tensor = raw_outputs[name]
            batch = np.squeeze(tensor, axis=0)
            all_preds.append(batch.reshape(-1))

        concat = np.concatenate(all_preds)
        n_attrs = 4 + 80
        if concat.size % n_attrs == 0:
            n_candidates = concat.size // n_attrs
            preds = concat.reshape(n_candidates, n_attrs)
            return self._decode_yolov8_raw(preds, ratio, dw, dh, fw, fh)

        print(f"[detector_hailo] WARNING: Unrecognized output shapes: "
              f"{[(n, raw_outputs[n].shape) for n in output_names]}")
        return []

    def _decode_yolov8_raw(self, preds: np.ndarray, ratio: float,
                           dw: float, dh: float,
                           fw: int, fh: int) -> list[Detection]:
        """Decode [N, 84] raw YOLOv8 predictions with NMS."""
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        person_scores = preds[:, 4 + config.PERSON_CLASS_ID]

        mask = person_scores >= config.MODEL_CONF
        if not np.any(mask):
            return []

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        scores = person_scores[mask]

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        keep = _nms(boxes, scores, config.MODEL_IOU)
        boxes = boxes[keep]
        scores = scores[keep]

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, fw)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, fh)

        return [Detection(int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(s))
                for b, s in zip(boxes, scores)]

    def shutdown(self):
        """Clean up Hailo resources."""
        try:
            if hasattr(self, '_ng_context') and self._ng_context:
                self._ng_context.__exit__(None, None, None)
        except Exception:
            pass
        try:
            if hasattr(self, '_vdevice') and self._vdevice:
                self._vdevice.release()
        except Exception:
            pass

    def __del__(self):
        self.shutdown()
