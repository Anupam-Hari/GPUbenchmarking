from __future__ import annotations

import threading
import time

import pynvml


class GPUMonitor:
    def __init__(self, device_index: int = 0, interval: float = 0.1):
        pynvml.nvmlInit()

        self.interval = interval
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

        self._running = False
        self._thread = None

        self.util_samples = []
        self.mem_samples = []

    def _run(self):
        while self._running:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            self.util_samples.append(util.gpu)
            self.mem_samples.append(mem.used / 1024**2)   # MB

            time.sleep(self.interval)

    def start(self):
        self.util_samples.clear()
        self.mem_samples.clear()

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()

    @property
    def average_gpu_util(self):
        return sum(self.util_samples) / len(self.util_samples) if self.util_samples else 0

    @property
    def peak_gpu_util(self):
        return max(self.util_samples) if self.util_samples else 0

    @property
    def average_memory(self):
        return sum(self.mem_samples) / len(self.mem_samples) if self.mem_samples else 0

    @property
    def peak_memory(self):
        return max(self.mem_samples) if self.mem_samples else 0