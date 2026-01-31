#!/usr/bin/env python3
"""
Real-time person detection pipeline for Raspberry Pi 5.
Supports two inference backends: CPU (ONNXRuntime) and Hailo-8 (HailoRT).

Usage:
    MODE=cpu   python main.py                              # CPU / ONNX
    MODE=hailo python main.py --hef yolov8n.hef            # Hailo-8
    MODE=cpu   python main.py --source 0 --show            # USB cam + display
    MODE=cpu   python main.py --source rtsp://... --imgsz 320
"""

import argparse
import os
import signal
import sys
import time

import cv2

# Load .env file if present (no extra dependency needed)
def _load_dotenv(path=".env"):
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

import config
from capture import FrameCapture
from display import Display
from logger import MetricsLogger
from monitor import SystemMonitor


def parse_args():
    p = argparse.ArgumentParser(description="Person detection benchmark - RPi 5")
    p.add_argument("--mode", default=None, choices=["cpu", "hailo"],
                   help="Inference mode (default: MODE env var or 'cpu')")
    p.add_argument("--source", default=None,
                   help="RTSP URL or camera index (default: config.RTSP_URL)")
    p.add_argument("--show", action="store_true", default=config.SHOW_VIDEO,
                   help="Show video window with overlay")
    p.add_argument("--imgsz", type=int, default=config.MODEL_IMGSZ,
                   help="Model input size (default: %(default)s)")
    p.add_argument("--conf", type=float, default=config.MODEL_CONF,
                   help="Confidence threshold (default: %(default)s)")
    p.add_argument("--skip", type=int, default=config.SKIP_FRAMES,
                   help="Process every Nth frame, 0=all (default: %(default)s)")
    # Hailo-specific
    p.add_argument("--hef", default=None,
                   help="Path to .hef model file (Hailo mode)")
    # CPU-specific
    p.add_argument("--onnx", default=None,
                   help="Path to .onnx model file (CPU mode)")
    return p.parse_args()


def build_detector(mode: str, args):
    """Factory: create the right detector based on mode."""
    if mode == "hailo":
        from detector_hailo import HailoDetector
        return HailoDetector(hef_path=args.hef)
    else:
        from detector_cpu import CPUDetector
        return CPUDetector(model_path=args.onnx)


def main():
    args = parse_args()

    # Resolve mode: CLI flag > env var > default
    mode = args.mode or config.MODE
    config.MODE = mode
    config.MODEL_IMGSZ = args.imgsz
    config.MODEL_CONF = args.conf

    # Resolve source
    source = args.source if args.source is not None else config.RTSP_URL
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    # --- Initialize components ---
    print(f"[main] Mode: {mode.upper()}")
    print(f"[main] Source: {source}")
    print(f"[main] Input size: {config.MODEL_IMGSZ}")
    print()

    detector = build_detector(mode, args)
    capture = FrameCapture(source)
    monitor = SystemMonitor()
    display = Display(mode=mode, model_name=detector.model_name)
    logger = MetricsLogger(mode=mode, model_name=detector.model_name)

    # Graceful shutdown
    def _shutdown(sig, frame):
        print("\n[main] Shutting down...")
        capture.stop()
        monitor.stop()
        logger.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown)

    # --- Start ---
    capture.start()
    monitor.start()
    display.print_header()

    frame_idx = 0
    skip = max(args.skip, 0)

    # --- Main loop ---
    while True:
        frame = capture.read(timeout=3.0)
        if frame is None:
            print("[main] No frame received (stream down?). Waiting...")
            time.sleep(1)
            continue

        frame_idx += 1
        if skip > 0 and (frame_idx % (skip + 1)) != 0:
            continue

        # --- Detect ---
        detections = detector.detect(frame)
        inference_ms = detector.last_inference_ms
        e2e_ms = detector.last_e2e_ms

        # --- Metrics ---
        display.tick()
        snap = monitor.snapshot()

        # Console
        display.print_metrics(snap, inference_ms, e2e_ms, len(detections))

        # CSV log
        logger.log(
            num_persons=len(detections),
            inference_ms=inference_ms,
            e2e_ms=e2e_ms,
            pipeline_fps=display.fps,
            mon_snap=snap,
        )

        # Video window
        if args.show:
            display.draw_overlay(frame, detections, snap, inference_ms, e2e_ms)
            cv2.imshow("Person Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    _shutdown(None, None)


if __name__ == "__main__":
    main()
