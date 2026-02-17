"""
GPU Session Manager Module

Handles GPU detection and NVENC session tracking for parallel video processing.

Components:
- get_gpu_info(): Detect GPU type, count, and NVENC session limits
- NVENCSessionTracker: Thread-safe tracker for active NVENC sessions per GPU
"""

import subprocess
import threading
from typing import Dict, Any

from .constants import NVENC_SESSION_LIMITS, DATACENTER_GPU_KEYWORDS


def get_gpu_info() -> Dict[str, Any]:
    """
    Detect GPU type, count GPUs, and determine NVENC session limit.
    For multi-GPU workers, scales sessions by number of GPUs.

    Returns dict with gpu_name, gpu_type, gpu_count, and nvenc_sessions.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            gpu_count = len(gpu_lines)
            gpu_name = gpu_lines[0] if gpu_lines else 'Unknown'

            # Determine GPU type
            is_datacenter = any(kw in gpu_name for kw in DATACENTER_GPU_KEYWORDS)

            gpu_type = 'datacenter' if is_datacenter else 'consumer'
            base_sessions = NVENC_SESSION_LIMITS[gpu_type]

            # Scale sessions by number of GPUs (multi-GPU workers)
            # Each GPU can handle its own NVENC sessions independently
            nvenc_sessions = base_sessions * gpu_count

            print(f"[GPU Detection] Found {gpu_count} GPU(s): {gpu_name}")
            print(f"[GPU Detection] Type: {gpu_type}, Sessions per GPU: {base_sessions}, Total: {nvenc_sessions}")

            return {
                'gpu_name': gpu_name,
                'gpu_type': gpu_type,
                'gpu_count': gpu_count,
                'nvenc_sessions': nvenc_sessions
            }
    except Exception as e:
        print(f"[GPU Detection] Error: {e}")

    return {
        'gpu_name': 'Unknown',
        'gpu_type': 'default',
        'gpu_count': 1,
        'nvenc_sessions': NVENC_SESSION_LIMITS['default']
    }


class NVENCSessionTracker:
    """
    Thread-safe tracker for active NVENC sessions per GPU.
    Uses semaphores to enforce REAL limits per GPU - critical for 9+ GPU scaling.

    Key improvement: Semaphores block when GPU is at capacity, preventing
    over-subscription and memory issues with many GPUs.
    """

    def __init__(self, gpu_count: int, sessions_per_gpu: int):
        self.gpu_count = gpu_count
        self.sessions_per_gpu = sessions_per_gpu
        self.active_sessions = {i: 0 for i in range(gpu_count)}
        self._lock = threading.Lock()

        # CRITICAL: Semaphores enforce REAL limits per GPU
        # This prevents over-subscription when scaling to 9+ GPUs
        self._gpu_semaphores = {i: threading.Semaphore(sessions_per_gpu) for i in range(gpu_count)}

        # Track total sessions for load balancing decisions
        self._total_assigned = {i: 0 for i in range(gpu_count)}

    def acquire_gpu(self, blocking: bool = True, timeout: float = None) -> int:
        """
        Get GPU with capacity available. Uses semaphores for REAL enforcement.

        Args:
            blocking: If True, wait for GPU to become available
            timeout: Max seconds to wait (None = infinite)

        Returns GPU ID (0-indexed), or -1 if non-blocking and none available.
        """
        # First, find the GPU with least total assignments (for fairness)
        with self._lock:
            # Sort GPUs by total assigned (prefer less used)
            gpu_order = sorted(range(self.gpu_count), key=lambda i: self._total_assigned[i])

        # Try to acquire from least-used GPU first
        for gpu_id in gpu_order:
            acquired = self._gpu_semaphores[gpu_id].acquire(blocking=False)
            if acquired:
                with self._lock:
                    self.active_sessions[gpu_id] += 1
                    self._total_assigned[gpu_id] += 1
                return gpu_id

        # If non-blocking and none available, return -1
        if not blocking:
            return -1

        # Blocking mode: wait for ANY GPU to become available
        # Use round-robin starting from least-used
        start_gpu = gpu_order[0]
        for i in range(self.gpu_count):
            gpu_id = (start_gpu + i) % self.gpu_count
            acquired = self._gpu_semaphores[gpu_id].acquire(blocking=True, timeout=timeout)
            if acquired:
                with self._lock:
                    self.active_sessions[gpu_id] += 1
                    self._total_assigned[gpu_id] += 1
                return gpu_id

        # Timeout expired on all GPUs
        return -1

    def release_gpu(self, gpu_id: int):
        """Release a session from the specified GPU."""
        if gpu_id < 0 or gpu_id >= self.gpu_count:
            return

        with self._lock:
            if self.active_sessions[gpu_id] > 0:
                self.active_sessions[gpu_id] -= 1

        # Release semaphore - allows waiting thread to proceed
        self._gpu_semaphores[gpu_id].release()

    def get_stats(self) -> Dict[str, Any]:
        """Get current session counts per GPU."""
        with self._lock:
            return {
                'active': dict(self.active_sessions),
                'total_assigned': dict(self._total_assigned),
                'capacity_per_gpu': self.sessions_per_gpu,
                'total_capacity': self.sessions_per_gpu * self.gpu_count
            }
