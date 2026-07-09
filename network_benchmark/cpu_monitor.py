from __future__ import annotations

import threading

import psutil


class ProcessCPUMonitor:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.process = psutil.Process()

        self._running = False
        self._thread = None
        self.samples = []

    def _run(self):
        self.process.cpu_percent(None)  # initialize

        while self._running:
            usage = self.process.cpu_percent(interval=self.interval)
            self.samples.append(usage)

    def start(self):
        self.samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join()

    @property
    def average(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def peak(self):
        return max(self.samples) if self.samples else 0.0