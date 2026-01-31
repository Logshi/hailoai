"""
Output module: console ASCII dashboard + optional OpenCV overlay.
Updated for dual-mode (CPU / Hailo) benchmarking display.
"""

import time
import cv2
import config
from detection import Detection


class Display:
    def __init__(self, mode: str, model_name: str):
        self.mode = mode.upper()
        self.model_name = model_name
        self._last_console_time = 0.0
        self._frame_count = 0
        self._fps_start = time.perf_counter()
        self.fps = 0.0

    def tick(self):
        """Call once per processed frame to update FPS counter."""
        self._frame_count += 1
        elapsed = time.perf_counter() - self._fps_start
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start = time.perf_counter()

    # -- console output --

    def print_metrics(self, mon_snap: dict, inference_ms: float,
                      e2e_ms: float, num_persons: int):
        """Print a compact metrics block to the console (rate-limited)."""
        now = time.time()
        if now - self._last_console_time < config.CONSOLE_UPDATE_INTERVAL:
            return
        self._last_console_time = now

        cores = mon_snap.get("per_core", [])
        core_str = " | ".join(f"{c:5.1f}%" for c in cores) if cores else "N/A"

        # Truncate long model names (e.g. full HEF paths)
        model_display = self.model_name
        if len(model_display) > 35:
            model_display = "..." + model_display[-32:]

        W = 62
        sep = "+" + "-" * W + "+"
        dsep = "+" + "=" * W + "+"

        lines = [
            dsep,
            f"|{'PERSON DETECTION BENCHMARK':^{W}}|",
            dsep,
            f"| {'Mode':<20}: {self.mode:<{W-23}}|",
            f"| {'Model':<20}: {model_display:<{W-23}}|",
            f"| {'Input size':<20}: {config.MODEL_IMGSZ:<{W-23}}|",
            sep,
            f"| {'Persons detected':<20}: {num_persons:<{W-23}}|",
            f"| {'Inference (model)':<20}: {inference_ms:>8.1f} ms{'':<{W-34}}|",
            f"| {'End-to-end (frame)':<20}: {e2e_ms:>8.1f} ms{'':<{W-34}}|",
            f"| {'Pipeline FPS':<20}: {self.fps:>8.2f}{'':<{W-31}}|",
            sep,
            f"| {'CPU overall':<20}: {mon_snap['cpu_percent']:>5.1f} %{'':<{W-30}}|",
            f"| {'Per-core':<20}: {core_str} |",
            f"| {'RAM used / total':<20}: {mon_snap['ram_used_mb']:>7.0f} / {mon_snap['ram_total_mb']:>7.0f} MB{'':<{W-42}}|",
            f"| {'RAM available':<20}: {mon_snap['ram_available_mb']:>7.0f} MB{'':<{W-33}}|",
            f"| {'Process RSS':<20}: {mon_snap['process_rss_mb']:>7.1f} MB{'':<{W-33}}|",
            f"| {'CPU temperature':<20}: {mon_snap['cpu_temp_c']:>5.1f} C{'':<{W-30}}|",
            f"| {'Throttle status':<20}: {str(mon_snap['throttle']):<{W-23}}|",
            dsep,
        ]
        # ANSI: move cursor up to overwrite previous block
        print(f"\033[{len(lines)}A" + "\n".join(lines), flush=True)

    def print_header(self):
        """Print blank lines so the first ANSI overwrite works cleanly."""
        print("\n" * 22, flush=True)

    # -- video overlay --

    def draw_overlay(self, frame, detections: list[Detection],
                     mon_snap: dict, inference_ms: float, e2e_ms: float):
        """Draw bounding boxes and HUD on the frame (in-place)."""
        for d in detections:
            cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (0, 255, 0), 2)
            label = f"person {d.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (d.x1, d.y1 - th - 6),
                          (d.x1 + tw, d.y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (d.x1, d.y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        hud = [
            f"Mode: {self.mode}",
            f"FPS: {self.fps:.1f}",
            f"Infer: {inference_ms:.0f}ms  E2E: {e2e_ms:.0f}ms",
            f"CPU: {mon_snap['cpu_percent']:.0f}%  Temp: {mon_snap['cpu_temp_c']:.1f}C",
            f"Persons: {len(detections)}",
        ]
        y = 24
        for line in hud:
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            y += 26

        return frame
