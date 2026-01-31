"""
System resource monitor.

Collects CPU, RAM, temperature, and throttle status in a background thread.
All values are exposed as plain attributes — no locks needed for atomic reads
of Python floats/ints on CPython.
"""

import os
import threading
import time
import psutil
import config


class SystemMonitor:
    def __init__(self, pid=None):
        self.pid = pid or os.getpid()
        self._process = psutil.Process(self.pid)
        self._stop = threading.Event()
        self._thread = None

        # Latest readings (written by background thread, read by main thread).
        self.cpu_percent = 0.0           # Overall CPU %
        self.per_core_percent = []       # Per-core list
        self.ram_used_mb = 0.0
        self.ram_available_mb = 0.0
        self.ram_total_mb = 0.0
        self.process_rss_mb = 0.0        # Resident set size of this process
        self.cpu_temp_c = 0.0
        self.throttle_status = "N/A"     # Raspberry Pi specific

    def start(self):
        # Prime psutil's cpu_percent (first call always returns 0).
        psutil.cpu_percent(percpu=True)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="monitor")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    # -- snapshot for display --

    def snapshot(self) -> dict:
        """Return a copy of the latest metrics as a dict."""
        return {
            "cpu_percent": self.cpu_percent,
            "per_core": list(self.per_core_percent),
            "ram_used_mb": self.ram_used_mb,
            "ram_available_mb": self.ram_available_mb,
            "ram_total_mb": self.ram_total_mb,
            "process_rss_mb": self.process_rss_mb,
            "cpu_temp_c": self.cpu_temp_c,
            "throttle": self.throttle_status,
        }

    # -- internals --

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(config.MONITOR_SAMPLE_INTERVAL)

    def _sample(self):
        # CPU
        self.per_core_percent = psutil.cpu_percent(interval=0, percpu=True)
        self.cpu_percent = sum(self.per_core_percent) / max(len(self.per_core_percent), 1)

        # RAM
        vm = psutil.virtual_memory()
        self.ram_total_mb = vm.total / (1024 * 1024)
        self.ram_used_mb = vm.used / (1024 * 1024)
        self.ram_available_mb = vm.available / (1024 * 1024)

        # Process memory
        try:
            mem = self._process.memory_info()
            self.process_rss_mb = mem.rss / (1024 * 1024)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        # Temperature — Raspberry Pi exposes thermal_zone0
        self.cpu_temp_c = self._read_temp()

        # Throttle — Raspberry Pi vcgencmd
        self.throttle_status = self._read_throttle()

    @staticmethod
    def _read_temp() -> float:
        # Method 1: psutil (works on most Linux)
        temps = psutil.sensors_temperatures()
        if temps:
            for name in ("cpu_thermal", "cpu-thermal", "coretemp", "soc_thermal"):
                if name in temps and temps[name]:
                    return temps[name][0].current
            # Fallback: first available sensor
            first = next(iter(temps.values()))
            if first:
                return first[0].current

        # Method 2: sysfs (Raspberry Pi)
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except (FileNotFoundError, ValueError, PermissionError):
            pass

        return 0.0

    @staticmethod
    def _read_throttle() -> str:
        """Read Raspberry Pi throttle register via vcgencmd."""
        try:
            import subprocess
            result = subprocess.run(
                ["vcgencmd", "get_throttled"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                # Output: throttled=0x0
                val = result.stdout.strip().split("=")[-1]
                code = int(val, 16)
                if code == 0:
                    return "OK"
                flags = []
                if code & 0x1:
                    flags.append("UNDERVOLTAGE")
                if code & 0x2:
                    flags.append("FREQ_CAPPED")
                if code & 0x4:
                    flags.append("THROTTLED")
                if code & 0x8:
                    flags.append("SOFT_TEMP_LIMIT")
                return "|".join(flags) if flags else f"0x{code:x}"
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        return "N/A"
