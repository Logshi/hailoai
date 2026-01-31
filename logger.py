"""
CSV logger for benchmark metrics.

Writes one row per second with all pipeline and system metrics.
File is created in LOG_DIR with a timestamped name.
"""

import csv
import os
import time
from datetime import datetime
import config

CSV_FIELDS = [
    "timestamp",
    "mode",
    "model_name",
    "input_size",
    "num_persons",
    "inference_ms",
    "e2e_ms",
    "pipeline_fps",
    "cpu_percent",
    "core0", "core1", "core2", "core3",
    "ram_used_mb",
    "ram_total_mb",
    "ram_available_mb",
    "process_rss_mb",
    "cpu_temp_c",
    "throttle_status",
]


class MetricsLogger:
    def __init__(self, mode: str, model_name: str):
        self.mode = mode
        self.model_name = model_name
        self._last_write = 0.0

        os.makedirs(config.LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{mode}_{ts}.csv"
        self._path = os.path.join(config.LOG_DIR, filename)
        self._file = open(self._path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._file.flush()
        print(f"[logger] Logging to {self._path}")

    def log(self, num_persons: int, inference_ms: float, e2e_ms: float,
            pipeline_fps: float, mon_snap: dict):
        """Write a row if enough time has passed since last write."""
        now = time.time()
        if now - self._last_write < config.LOG_INTERVAL:
            return
        self._last_write = now

        per_core = mon_snap.get("per_core", [0, 0, 0, 0])
        # Pad to 4 cores if fewer reported
        while len(per_core) < 4:
            per_core.append(0.0)

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode,
            "model_name": self.model_name,
            "input_size": config.MODEL_IMGSZ,
            "num_persons": num_persons,
            "inference_ms": round(inference_ms, 2),
            "e2e_ms": round(e2e_ms, 2),
            "pipeline_fps": round(pipeline_fps, 2),
            "cpu_percent": round(mon_snap["cpu_percent"], 1),
            "core0": round(per_core[0], 1),
            "core1": round(per_core[1], 1),
            "core2": round(per_core[2], 1),
            "core3": round(per_core[3], 1),
            "ram_used_mb": round(mon_snap["ram_used_mb"], 0),
            "ram_total_mb": round(mon_snap["ram_total_mb"], 0),
            "ram_available_mb": round(mon_snap["ram_available_mb"], 0),
            "process_rss_mb": round(mon_snap["process_rss_mb"], 1),
            "cpu_temp_c": round(mon_snap["cpu_temp_c"], 1),
            "throttle_status": mon_snap["throttle"],
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        if self._file and not self._file.closed:
            self._file.close()
            print(f"[logger] Log saved: {self._path}")
