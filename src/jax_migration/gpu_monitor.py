"""
GPU Monitoring Utility for JAX Training

Monitors GPU utilization, memory usage, and temperature in real-time
using NVIDIA Management Library (NVML) via pynvml.
"""

import threading
import time
from typing import Dict, Optional
from dataclasses import dataclass, field


@dataclass
class GPUStats:
    """GPU statistics snapshot."""
    utilization_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    temperature_celsius: float = 0.0
    timestamp: float = field(default_factory=time.time)


class GPUMonitor:
    """
    Real-time GPU monitoring with background thread.

    Usage:
        monitor = GPUMonitor(device_id=0, interval=1.0)
        monitor.start_monitoring()
        # ... training code ...
        stats = monitor.get_stats()
        monitor.stop_monitoring()

    Attributes:
        device_id: GPU device index (default 0)
        interval: Polling interval in seconds (default 1.0)
    """

    def __init__(self, device_id: int = 0, interval: float = 1.0):
        """
        Initialize GPU monitor.

        Args:
            device_id: GPU device index to monitor
            interval: Polling interval in seconds
        """
        self.device_id = device_id
        self.interval = interval

        # Monitoring state
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Statistics
        self._samples: list[GPUStats] = []
        self._is_initialized = False
        self._nvml_available = False

        # Try to initialize NVML
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self._nvml_available = True
            self._is_initialized = True

            # Get device name
            device_name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode('utf-8')
            print(f"[GPU Monitor] Initialized for device {device_id}: {device_name}")

        except ImportError:
            print("[GPU Monitor] pynvml not installed. GPU monitoring disabled.")
            print("  Install with: pip install pynvml>=11.5.0")
        except Exception as e:
            print(f"[GPU Monitor] Failed to initialize NVML: {e}")
            print("  GPU monitoring will be disabled.")

    def _get_current_stats(self) -> GPUStats:
        """Query current GPU statistics."""
        if not self._nvml_available:
            return GPUStats()

        try:
            # GPU utilization
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            utilization = float(util.gpu)

            # Memory info
            mem_info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
            memory_used_gb = mem_info.used / (1024 ** 3)
            memory_total_gb = mem_info.total / (1024 ** 3)

            # Temperature
            try:
                temperature = float(self._nvml.nvmlDeviceGetTemperature(
                    self._handle,
                    self._nvml.NVML_TEMPERATURE_GPU
                ))
            except:
                temperature = 0.0

            return GPUStats(
                utilization_percent=utilization,
                memory_used_gb=memory_used_gb,
                memory_total_gb=memory_total_gb,
                temperature_celsius=temperature,
                timestamp=time.time()
            )

        except Exception as e:
            print(f"[GPU Monitor] Error querying GPU: {e}")
            return GPUStats()

    def _monitoring_loop(self):
        """Background thread monitoring loop."""
        while not self._stop_event.is_set():
            stats = self._get_current_stats()

            with self._lock:
                self._samples.append(stats)

            time.sleep(self.interval)

    def start_monitoring(self):
        """Start background monitoring thread."""
        if not self._is_initialized:
            print("[GPU Monitor] Cannot start - NVML not initialized")
            return

        if self._thread is not None and self._thread.is_alive():
            print("[GPU Monitor] Already monitoring")
            return

        self._stop_event.clear()
        self._samples.clear()

        self._thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="GPUMonitorThread"
        )
        self._thread.start()
        print(f"[GPU Monitor] Started monitoring (interval={self.interval}s)")

    def stop_monitoring(self):
        """Stop background monitoring thread."""
        if self._thread is None or not self._thread.is_alive():
            return

        self._stop_event.set()
        self._thread.join(timeout=5.0)
        print("[GPU Monitor] Stopped monitoring")

    def get_stats(self) -> Dict[str, float]:
        """
        Get summary statistics from monitoring session.

        Returns:
            Dictionary with keys:
                - avg_utilization: Average GPU utilization (%)
                - peak_utilization: Peak GPU utilization (%)
                - avg_memory_gb: Average GPU memory used (GB)
                - peak_memory_gb: Peak GPU memory used (GB)
                - total_memory_gb: Total GPU memory available (GB)
                - avg_temperature: Average GPU temperature (°C)
                - peak_temperature: Peak GPU temperature (°C)
                - num_samples: Number of samples collected
                - duration_seconds: Monitoring duration
        """
        with self._lock:
            samples = list(self._samples)

        if not samples:
            return {
                'avg_utilization': 0.0,
                'peak_utilization': 0.0,
                'avg_memory_gb': 0.0,
                'peak_memory_gb': 0.0,
                'total_memory_gb': 0.0,
                'avg_temperature': 0.0,
                'peak_temperature': 0.0,
                'num_samples': 0,
                'duration_seconds': 0.0
            }

        # Calculate statistics
        utilizations = [s.utilization_percent for s in samples]
        memory_used = [s.memory_used_gb for s in samples]
        temperatures = [s.temperature_celsius for s in samples]

        duration = samples[-1].timestamp - samples[0].timestamp if len(samples) > 1 else 0.0

        return {
            'avg_utilization': sum(utilizations) / len(utilizations),
            'peak_utilization': max(utilizations),
            'avg_memory_gb': sum(memory_used) / len(memory_used),
            'peak_memory_gb': max(memory_used),
            'total_memory_gb': samples[0].memory_total_gb,
            'avg_temperature': sum(temperatures) / len(temperatures) if temperatures else 0.0,
            'peak_temperature': max(temperatures) if temperatures else 0.0,
            'num_samples': len(samples),
            'duration_seconds': duration
        }

    def get_current(self) -> GPUStats:
        """Get current GPU statistics (immediate query, not from monitoring thread)."""
        return self._get_current_stats()

    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()

    def __del__(self):
        """Cleanup on deletion."""
        self.stop_monitoring()
        if self._nvml_available:
            try:
                self._nvml.nvmlShutdown()
            except:
                pass


if __name__ == "__main__":
    # Test GPU monitoring
    import sys

    print("Testing GPU Monitor...")
    monitor = GPUMonitor(device_id=0, interval=0.5)

    if not monitor._is_initialized:
        print("Failed to initialize GPU monitor")
        sys.exit(1)

    # Test immediate query
    current = monitor.get_current()
    print(f"\nCurrent GPU stats:")
    print(f"  Utilization: {current.utilization_percent:.1f}%")
    print(f"  Memory: {current.memory_used_gb:.2f} / {current.memory_total_gb:.2f} GB")
    print(f"  Temperature: {current.temperature_celsius:.1f}°C")

    # Test monitoring
    print("\nStarting 5-second monitoring test...")
    monitor.start_monitoring()
    time.sleep(5.0)
    monitor.stop_monitoring()

    stats = monitor.get_stats()
    print(f"\nMonitoring summary:")
    print(f"  Average utilization: {stats['avg_utilization']:.1f}%")
    print(f"  Peak utilization: {stats['peak_utilization']:.1f}%")
    print(f"  Average memory: {stats['avg_memory_gb']:.2f} GB")
    print(f"  Peak memory: {stats['peak_memory_gb']:.2f} GB")
    print(f"  Average temperature: {stats['avg_temperature']:.1f}°C")
    print(f"  Samples collected: {stats['num_samples']}")
    print(f"  Duration: {stats['duration_seconds']:.1f}s")

    print("\n✓ GPU monitor test complete")
