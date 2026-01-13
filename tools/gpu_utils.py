"""
GPU Utilities Module - Unified Multi-GPU Infrastructure

This module provides a centralized GPU management system for all processors.
It handles GPU detection, NVENC session management, round-robin GPU assignment,
and FFmpeg encoding with automatic CPU fallback.

Usage:
    from tools.gpu_utils import (
        get_gpu_count,
        get_gpu_info,
        get_nvenc_sessions_per_gpu,
        assign_gpu,
        build_ffmpeg_nvenc_args,
        build_ffmpeg_cpu_fallback_args,
        run_ffmpeg_with_fallback,
        GPUManager
    )

    # Get GPU information
    gpu_count = get_gpu_count()
    info = get_gpu_info()

    # Assign GPU for a task (round-robin)
    gpu_id = assign_gpu(task_index=0)

    # Build FFmpeg encoding arguments
    nvenc_args = build_ffmpeg_nvenc_args(gpu_id=0, bitrate=5000)
    cpu_args = build_ffmpeg_cpu_fallback_args(bitrate=5000)

    # Run FFmpeg with automatic fallback
    success, result = run_ffmpeg_with_fallback(
        cmd_nvenc=['ffmpeg', ...],
        cmd_cpu=['ffmpeg', ...],
        progress_cb=my_callback,
        duration=60.0
    )
"""

import os
import subprocess
import threading
import re
from typing import Callable, Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing


# =============================================================================
# CONSTANTS
# =============================================================================

# NVENC session limits by GPU type (per GPU)
# Datacenter GPUs: Unlimited NVENC sessions (we use 12 as practical limit)
# Consumer GPUs: Limited to 3-5 sessions by NVIDIA driver
NVENC_SESSION_LIMITS = {
    'datacenter': 12,   # A100, A6000, A5000, H100, L40, L40S, RTX 6000 Ada, etc.
    'consumer': 3,      # RTX 4090, 3090, 4080, 3080, etc.
    'default': 2,       # Fallback for unknown GPUs
}

# GPU type detection keywords
DATACENTER_GPU_KEYWORDS = [
    # NVIDIA A-series (Ampere datacenter)
    'A100', 'A6000', 'A5000', 'A4500', 'A4000', 'A40', 'A30', 'A10', 'A16', 'A2',
    # NVIDIA H-series (Hopper)
    'H100', 'H200',
    # NVIDIA L-series (Ada Lovelace datacenter)
    'L40', 'L40S', 'L4',
    # NVIDIA RTX Ada (professional/datacenter)
    'RTX 6000 Ada', 'RTX 5000 Ada', 'RTX 4500 Ada', 'RTX 4000 Ada',
    'RTX A6000', 'RTX A5000', 'RTX A4500', 'RTX A4000',
    # Legacy datacenter
    'V100', 'T4', 'P100', 'P40', 'P4',
    # Quadro/Tesla (professional)
    'Quadro', 'Tesla',
]

# Consumer GPUs that look datacenter-ish but are limited
CONSUMER_GPU_KEYWORDS = [
    'RTX 4090', 'RTX 4080', 'RTX 4070', 'RTX 4060',
    'RTX 3090', 'RTX 3080', 'RTX 3070', 'RTX 3060',
    'RTX 2080', 'RTX 2070', 'RTX 2060',
    'GTX 16', 'GTX 10',
]


# =============================================================================
# GPU MANAGER SINGLETON
# =============================================================================

class GPUManager:
    """
    Singleton class for GPU management.

    Provides:
    - GPU detection and enumeration
    - NVENC session tracking per GPU
    - Round-robin GPU assignment
    - Thread-safe operations

    Example:
        manager = GPUManager.get_instance()
        gpu_id = manager.assign_gpu(task_index=0)
        sessions = manager.get_available_sessions(gpu_id)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'GPUManager':
        """Get the singleton instance."""
        return cls()

    def __init__(self):
        if self._initialized:
            return

        self._gpu_count = 0
        self._gpu_names: List[str] = []
        self._gpu_types: List[str] = []
        self._gpu_memory: List[int] = []  # Memory in MB
        self._nvenc_sessions_per_gpu: List[int] = []
        self._active_sessions: Dict[int, int] = {}  # gpu_id -> active_sessions
        self._session_lock = threading.Lock()

        self._detect_gpus()
        self._initialized = True

    def _detect_gpus(self):
        """Detect all available GPUs using nvidia-smi."""
        try:
            # Query GPU names
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                self._gpu_count = len(lines)

                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    gpu_name = parts[0] if parts else 'Unknown'
                    memory_mb = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

                    self._gpu_names.append(gpu_name)
                    self._gpu_memory.append(memory_mb)

                    # Determine GPU type
                    gpu_type = self._detect_gpu_type(gpu_name)
                    self._gpu_types.append(gpu_type)

                    # Set NVENC sessions based on type
                    sessions = NVENC_SESSION_LIMITS.get(gpu_type, NVENC_SESSION_LIMITS['default'])
                    self._nvenc_sessions_per_gpu.append(sessions)

                # Initialize active session counters
                for i in range(self._gpu_count):
                    self._active_sessions[i] = 0

                print(f"[GPUManager] Detected {self._gpu_count} GPU(s)")
                for i, name in enumerate(self._gpu_names):
                    print(f"[GPUManager]   GPU {i}: {name} ({self._gpu_types[i]}, "
                          f"{self._gpu_memory[i]}MB, {self._nvenc_sessions_per_gpu[i]} NVENC sessions)")
            else:
                print(f"[GPUManager] nvidia-smi failed or no output: {result.stderr}")
                self._set_defaults()

        except FileNotFoundError:
            print("[GPUManager] nvidia-smi not found - no NVIDIA GPUs available")
            self._set_defaults()
        except subprocess.TimeoutExpired:
            print("[GPUManager] nvidia-smi timed out")
            self._set_defaults()
        except Exception as e:
            print(f"[GPUManager] Error detecting GPUs: {e}")
            self._set_defaults()

    def _detect_gpu_type(self, gpu_name: str) -> str:
        """Determine GPU type from name."""
        gpu_name_upper = gpu_name.upper()

        # Check consumer GPUs first (more specific matches)
        for keyword in CONSUMER_GPU_KEYWORDS:
            if keyword.upper() in gpu_name_upper:
                return 'consumer'

        # Check datacenter GPUs
        for keyword in DATACENTER_GPU_KEYWORDS:
            if keyword.upper() in gpu_name_upper:
                return 'datacenter'

        # Default to consumer for safety
        return 'consumer'

    def _set_defaults(self):
        """Set default values when GPU detection fails."""
        self._gpu_count = 0
        self._gpu_names = []
        self._gpu_types = []
        self._gpu_memory = []
        self._nvenc_sessions_per_gpu = []

    @property
    def gpu_count(self) -> int:
        """Return number of detected GPUs."""
        return self._gpu_count

    def get_gpu_info(self, gpu_id: int = 0) -> Dict[str, Any]:
        """
        Get detailed information about a specific GPU.

        Args:
            gpu_id: GPU index (default 0)

        Returns:
            Dict with gpu_name, gpu_type, memory_mb, nvenc_sessions
        """
        if gpu_id < 0 or gpu_id >= self._gpu_count:
            return {
                'gpu_name': 'Unknown',
                'gpu_type': 'default',
                'memory_mb': 0,
                'nvenc_sessions': NVENC_SESSION_LIMITS['default']
            }

        return {
            'gpu_name': self._gpu_names[gpu_id],
            'gpu_type': self._gpu_types[gpu_id],
            'memory_mb': self._gpu_memory[gpu_id],
            'nvenc_sessions': self._nvenc_sessions_per_gpu[gpu_id]
        }

    def get_all_gpus_info(self) -> List[Dict[str, Any]]:
        """Get information about all GPUs."""
        return [self.get_gpu_info(i) for i in range(self._gpu_count)]

    def get_nvenc_sessions(self, gpu_id: int = 0) -> int:
        """Get max NVENC sessions for a GPU."""
        if 0 <= gpu_id < len(self._nvenc_sessions_per_gpu):
            return self._nvenc_sessions_per_gpu[gpu_id]
        return NVENC_SESSION_LIMITS['default']

    def get_total_nvenc_sessions(self) -> int:
        """Get total NVENC sessions across all GPUs."""
        return sum(self._nvenc_sessions_per_gpu) if self._nvenc_sessions_per_gpu else NVENC_SESSION_LIMITS['default']

    def assign_gpu(self, task_index: int) -> int:
        """
        Assign a GPU to a task using round-robin distribution.

        Args:
            task_index: Index of the task (0, 1, 2, ...)

        Returns:
            GPU ID to use for this task
        """
        if self._gpu_count == 0:
            return 0
        return task_index % self._gpu_count

    def acquire_session(self, gpu_id: int) -> bool:
        """
        Try to acquire an NVENC session on a GPU.
        Thread-safe.

        Returns:
            True if session acquired, False if GPU at capacity
        """
        with self._session_lock:
            if gpu_id not in self._active_sessions:
                return False

            max_sessions = self.get_nvenc_sessions(gpu_id)
            if self._active_sessions[gpu_id] < max_sessions:
                self._active_sessions[gpu_id] += 1
                return True
            return False

    def release_session(self, gpu_id: int):
        """Release an NVENC session on a GPU. Thread-safe."""
        with self._session_lock:
            if gpu_id in self._active_sessions:
                self._active_sessions[gpu_id] = max(0, self._active_sessions[gpu_id] - 1)

    def get_active_sessions(self, gpu_id: int) -> int:
        """Get current active sessions on a GPU."""
        with self._session_lock:
            return self._active_sessions.get(gpu_id, 0)

    def get_available_sessions(self, gpu_id: int) -> int:
        """Get remaining available sessions on a GPU."""
        with self._session_lock:
            max_sessions = self.get_nvenc_sessions(gpu_id)
            active = self._active_sessions.get(gpu_id, 0)
            return max(0, max_sessions - active)

    def reset_sessions(self):
        """Reset all session counters. Useful for testing."""
        with self._session_lock:
            for gpu_id in self._active_sessions:
                self._active_sessions[gpu_id] = 0


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

def get_gpu_count() -> int:
    """
    Get the number of available GPUs.

    Returns:
        Number of NVIDIA GPUs detected (0 if none)
    """
    return GPUManager.get_instance().gpu_count


def get_gpu_info(gpu_id: int = None) -> Dict[str, Any]:
    """
    Get detailed GPU information.

    Args:
        gpu_id: Specific GPU index, or None for summary of all GPUs

    Returns:
        If gpu_id specified: Dict with gpu_name, gpu_type, memory_mb, nvenc_sessions
        If gpu_id is None: Dict with gpu_count, gpu_names, total_nvenc_sessions, gpus list
    """
    manager = GPUManager.get_instance()

    if gpu_id is not None:
        return manager.get_gpu_info(gpu_id)

    # Return summary of all GPUs
    gpus = manager.get_all_gpus_info()
    return {
        'gpu_count': manager.gpu_count,
        'gpu_names': [g['gpu_name'] for g in gpus],
        'total_nvenc_sessions': manager.get_total_nvenc_sessions(),
        'gpus': gpus
    }


def get_nvenc_sessions_per_gpu(gpu_id: int = 0) -> int:
    """
    Get max NVENC sessions for a specific GPU.

    Args:
        gpu_id: GPU index (default 0)

    Returns:
        Maximum NVENC sessions for this GPU based on type:
        - Datacenter GPUs (A100, A6000, H100, L40, etc): 12 sessions
        - Consumer GPUs (RTX 4090, 3090, etc): 3 sessions
        - Default/unknown: 2 sessions
    """
    return GPUManager.get_instance().get_nvenc_sessions(gpu_id)


def assign_gpu(task_index: int) -> int:
    """
    Assign a GPU to a task using round-robin distribution.

    Args:
        task_index: Index of the task (0, 1, 2, ...)

    Returns:
        GPU ID to use for this task (distributes evenly across GPUs)

    Example:
        # With 2 GPUs:
        assign_gpu(0)  # Returns 0
        assign_gpu(1)  # Returns 1
        assign_gpu(2)  # Returns 0
        assign_gpu(3)  # Returns 1
    """
    return GPUManager.get_instance().assign_gpu(task_index)


# =============================================================================
# FFMPEG ARGUMENT BUILDERS
# =============================================================================

def build_ffmpeg_nvenc_args(
    gpu_id: int = 0,
    bitrate: int = 5000,
    preset: str = 'p4',
    cq: int = 23,
    lookahead: int = 8
) -> List[str]:
    """
    Build common NVENC encoding arguments for FFmpeg.

    Args:
        gpu_id: GPU to use for encoding (for multi-GPU systems)
        bitrate: Target bitrate in kbps (default 5000)
        preset: NVENC preset (p1=fastest to p7=best quality, default p4)
        cq: Constant quality value (0-51, lower=better, default 23)
        lookahead: RC lookahead frames (default 8)

    Returns:
        List of FFmpeg arguments for NVENC encoding

    Example:
        args = build_ffmpeg_nvenc_args(gpu_id=0, bitrate=5000)
        cmd = ['ffmpeg', '-i', 'input.mp4', *args, 'output.mp4']
    """
    return [
        '-c:v', 'h264_nvenc',
        '-gpu', str(gpu_id),
        '-preset', preset,
        '-rc', 'vbr',
        '-cq', str(cq),
        '-b:v', f'{bitrate}k',
        '-maxrate', f'{int(bitrate * 1.5)}k',
        '-bufsize', f'{bitrate * 2}k',
        '-rc-lookahead', str(lookahead),
    ]


def build_ffmpeg_cpu_fallback_args(
    bitrate: int = 5000,
    preset: str = 'fast',
    crf: int = 23
) -> List[str]:
    """
    Build CPU fallback (libx264) encoding arguments for FFmpeg.

    Args:
        bitrate: Target bitrate in kbps (default 5000)
        preset: x264 preset (ultrafast to veryslow, default fast)
        crf: Constant rate factor (0-51, lower=better, default 23)

    Returns:
        List of FFmpeg arguments for CPU encoding

    Example:
        args = build_ffmpeg_cpu_fallback_args(bitrate=5000)
        cmd = ['ffmpeg', '-i', 'input.mp4', *args, 'output.mp4']
    """
    return [
        '-c:v', 'libx264',
        '-preset', preset,
        '-crf', str(crf),
        '-b:v', f'{bitrate}k',
        '-maxrate', f'{int(bitrate * 1.5)}k',
        '-bufsize', f'{bitrate * 2}k',
        '-pix_fmt', 'yuv420p',
    ]


def build_ffmpeg_audio_args(
    codec: str = 'aac',
    bitrate: int = 128,
    keep_audio: bool = True
) -> List[str]:
    """
    Build audio encoding arguments for FFmpeg.

    Args:
        codec: Audio codec (default 'aac')
        bitrate: Audio bitrate in kbps (default 128)
        keep_audio: Whether to include audio (default True)

    Returns:
        List of FFmpeg arguments for audio
    """
    if not keep_audio:
        return ['-an']
    return ['-c:a', codec, '-b:a', f'{bitrate}k']


def build_complete_ffmpeg_cmd(
    input_path: str,
    output_path: str,
    video_filter: Optional[str] = None,
    gpu_id: int = 0,
    bitrate: int = 5000,
    keep_audio: bool = True,
    use_nvenc: bool = True,
    extra_input_args: Optional[List[str]] = None,
    extra_output_args: Optional[List[str]] = None
) -> List[str]:
    """
    Build a complete FFmpeg command for video processing.

    Args:
        input_path: Input video path
        output_path: Output video path
        video_filter: Video filter string (e.g., "scale=1920:1080")
        gpu_id: GPU to use for NVENC
        bitrate: Target video bitrate in kbps
        keep_audio: Whether to include audio
        use_nvenc: Whether to use NVENC (True) or CPU (False)
        extra_input_args: Additional arguments before input
        extra_output_args: Additional arguments before output

    Returns:
        Complete FFmpeg command as list
    """
    cmd = ['ffmpeg', '-y']

    # Extra input args (e.g., -hwaccel cuda)
    if extra_input_args:
        cmd.extend(extra_input_args)

    cmd.extend(['-i', input_path])

    # Video filter
    if video_filter:
        cmd.extend(['-vf', video_filter])

    # Video encoding
    if use_nvenc:
        cmd.extend(build_ffmpeg_nvenc_args(gpu_id, bitrate))
    else:
        cmd.extend(build_ffmpeg_cpu_fallback_args(bitrate))

    # Audio
    cmd.extend(build_ffmpeg_audio_args(keep_audio=keep_audio))

    # Extra output args
    if extra_output_args:
        cmd.extend(extra_output_args)

    # Output
    cmd.extend(['-movflags', '+faststart', output_path])

    return cmd


# =============================================================================
# FFMPEG EXECUTION WITH FALLBACK
# =============================================================================

def run_ffmpeg_with_fallback(
    cmd_nvenc: List[str],
    cmd_cpu: List[str],
    progress_cb: Optional[Callable[[float, str], None]] = None,
    duration: float = 0,
    timeout: int = 900,
    gpu_id: int = 0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run FFmpeg with NVENC, automatically falling back to CPU if NVENC fails.

    Args:
        cmd_nvenc: FFmpeg command with NVENC encoding
        cmd_cpu: FFmpeg command with CPU encoding (fallback)
        progress_cb: Optional callback(progress: 0-1, message: str)
        duration: Video duration in seconds (for progress calculation)
        timeout: Max execution time in seconds (default 900 = 15 minutes)
        gpu_id: GPU ID being used (for session tracking)

    Returns:
        Tuple of (success: bool, result: dict)
        result contains: encoder_used, stderr (on failure), duration

    Example:
        nvenc_cmd = ['ffmpeg', '-y', '-i', 'in.mp4', ...nvenc_args..., 'out.mp4']
        cpu_cmd = ['ffmpeg', '-y', '-i', 'in.mp4', ...cpu_args..., 'out.mp4']

        success, result = run_ffmpeg_with_fallback(nvenc_cmd, cpu_cmd, my_progress, 60.0)
        if success:
            print(f"Encoded with {result['encoder_used']}")
    """
    manager = GPUManager.get_instance()

    def report_progress(progress: float, message: str):
        if progress_cb:
            progress_cb(progress, message)

    def run_ffmpeg(cmd: List[str], mode_name: str) -> Tuple[bool, str]:
        """Run FFmpeg and return (success, stderr)."""
        stderr_lines = []

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )

            while True:
                line = process.stderr.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue

                stderr_lines.append(line)
                # Keep last 50 lines
                if len(stderr_lines) > 50:
                    stderr_lines.pop(0)

                # Parse progress from FFmpeg output
                if 'time=' in line and duration > 0:
                    try:
                        time_match = re.search(r'time=(\d+):(\d+):(\d+\.?\d*)', line)
                        if time_match:
                            h, m, s = time_match.groups()
                            current_time = int(h) * 3600 + int(m) * 60 + float(s)
                            progress = min(0.1 + (current_time / duration) * 0.85, 0.95)
                            report_progress(progress, f"Encoding ({mode_name})... {int(current_time)}s / {int(duration)}s")
                    except Exception:
                        pass

            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                return False, "FFmpeg process timed out"

            stderr_output = ''.join(stderr_lines)
            return process.returncode == 0, stderr_output

        except Exception as e:
            return False, str(e)

    # Try NVENC first
    report_progress(0.05, "Starting NVENC encoding...")

    # Acquire NVENC session
    session_acquired = manager.acquire_session(gpu_id)

    try:
        success, stderr = run_ffmpeg(cmd_nvenc, "NVENC")
    finally:
        if session_acquired:
            manager.release_session(gpu_id)

    if success:
        report_progress(1.0, "Complete (NVENC)")
        return True, {'encoder_used': 'nvenc', 'gpu_id': gpu_id}

    # NVENC failed, try CPU fallback
    print(f"[gpu_utils] NVENC failed on GPU {gpu_id}, trying CPU fallback...")
    print(f"[gpu_utils] NVENC error: {stderr[-500:] if len(stderr) > 500 else stderr}")

    report_progress(0.1, "NVENC failed, using CPU fallback...")

    success, stderr = run_ffmpeg(cmd_cpu, "CPU")

    if success:
        report_progress(1.0, "Complete (CPU)")
        return True, {'encoder_used': 'cpu'}

    # Both failed
    report_progress(0.0, "Encoding failed")
    return False, {'encoder_used': None, 'stderr': stderr}


def run_ffmpeg_simple(
    input_path: str,
    output_path: str,
    video_filter: Optional[str] = None,
    gpu_id: int = 0,
    bitrate: int = 5000,
    keep_audio: bool = True,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    duration: float = 0
) -> Tuple[bool, Dict[str, Any]]:
    """
    Simple wrapper to run FFmpeg with automatic NVENC/CPU fallback.

    Args:
        input_path: Input video path
        output_path: Output video path
        video_filter: Optional video filter string
        gpu_id: GPU to use for NVENC
        bitrate: Target bitrate in kbps
        keep_audio: Whether to keep audio
        progress_cb: Optional progress callback
        duration: Video duration for progress

    Returns:
        Tuple of (success, result_dict)
    """
    cmd_nvenc = build_complete_ffmpeg_cmd(
        input_path, output_path, video_filter,
        gpu_id=gpu_id, bitrate=bitrate, keep_audio=keep_audio, use_nvenc=True
    )

    cmd_cpu = build_complete_ffmpeg_cmd(
        input_path, output_path, video_filter,
        gpu_id=gpu_id, bitrate=bitrate, keep_audio=keep_audio, use_nvenc=False
    )

    return run_ffmpeg_with_fallback(cmd_nvenc, cmd_cpu, progress_cb, duration, gpu_id=gpu_id)


# =============================================================================
# PARALLEL PROCESSING UTILITIES
# =============================================================================

def create_thread_pool(max_workers: int = None) -> ThreadPoolExecutor:
    """
    Create a ThreadPoolExecutor sized for video processing.

    Args:
        max_workers: Max workers, or None to auto-detect based on NVENC sessions

    Returns:
        ThreadPoolExecutor configured for the system
    """
    if max_workers is None:
        manager = GPUManager.get_instance()
        max_workers = manager.get_total_nvenc_sessions()
        max_workers = max(1, max_workers)

    return ThreadPoolExecutor(max_workers=max_workers)


def create_process_pool(max_workers: int = None) -> ProcessPoolExecutor:
    """
    Create a ProcessPoolExecutor for image processing.
    Uses 'spawn' context to avoid CUDA issues with fork.

    Args:
        max_workers: Max workers, or None to use CPU count

    Returns:
        ProcessPoolExecutor with spawn context
    """
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context('spawn')
    )


def create_video_process_pool(max_workers: int = None) -> ProcessPoolExecutor:
    """
    Create a ProcessPoolExecutor sized for parallel video encoding.

    Args:
        max_workers: Max workers, or None to use total NVENC sessions

    Returns:
        ProcessPoolExecutor configured for parallel video encoding
    """
    if max_workers is None:
        manager = GPUManager.get_instance()
        max_workers = manager.get_total_nvenc_sessions()
        max_workers = max(1, max_workers)

    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context('spawn')
    )


# =============================================================================
# VIDEO INFO UTILITIES
# =============================================================================

def get_video_info(path: str) -> Dict[str, Any]:
    """
    Get video metadata using ffprobe.

    Args:
        path: Path to video file

    Returns:
        Dict with width, height, fps, duration, has_audio
    """
    try:
        import json

        cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams', '-show_format',
            path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return _default_video_info()

        info = json.loads(result.stdout)

        video_stream = next(
            (s for s in info.get('streams', []) if s.get('codec_type') == 'video'),
            {}
        )

        audio_stream = next(
            (s for s in info.get('streams', []) if s.get('codec_type') == 'audio'),
            None
        )

        # Parse FPS
        fps_str = video_stream.get('r_frame_rate', '30/1')
        try:
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = int(num) / int(den) if int(den) != 0 else 30
            else:
                fps = float(fps_str)
        except Exception:
            fps = 30

        return {
            'width': int(video_stream.get('width', 1920)),
            'height': int(video_stream.get('height', 1080)),
            'fps': fps,
            'duration': float(info.get('format', {}).get('duration', 0)),
            'has_audio': audio_stream is not None
        }

    except Exception as e:
        print(f"[gpu_utils] Error getting video info: {e}")
        return _default_video_info()


def _default_video_info() -> Dict[str, Any]:
    """Return default video info."""
    return {
        'width': 1920,
        'height': 1080,
        'fps': 30,
        'duration': 0,
        'has_audio': True
    }


# =============================================================================
# DIAGNOSTIC FUNCTIONS
# =============================================================================

def print_gpu_diagnostics():
    """Print detailed GPU diagnostics to stdout."""
    manager = GPUManager.get_instance()

    print("\n" + "=" * 60)
    print("GPU DIAGNOSTICS")
    print("=" * 60)

    print(f"\nTotal GPUs detected: {manager.gpu_count}")

    if manager.gpu_count == 0:
        print("No NVIDIA GPUs found!")
        print("Make sure nvidia-smi is available and NVIDIA drivers are installed.")
    else:
        print(f"Total NVENC sessions available: {manager.get_total_nvenc_sessions()}")
        print("\nPer-GPU details:")

        for i, info in enumerate(manager.get_all_gpus_info()):
            print(f"\n  GPU {i}: {info['gpu_name']}")
            print(f"    Type: {info['gpu_type']}")
            print(f"    Memory: {info['memory_mb']} MB")
            print(f"    NVENC sessions: {info['nvenc_sessions']}")
            print(f"    Active sessions: {manager.get_active_sessions(i)}")
            print(f"    Available sessions: {manager.get_available_sessions(i)}")

    print("\n" + "=" * 60 + "\n")


def test_nvenc_available() -> bool:
    """
    Test if NVENC encoding is available by running a quick FFmpeg test.

    Returns:
        True if NVENC works, False otherwise
    """
    try:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'testsrc=duration=0.1:size=64x64',
            '-c:v', 'h264_nvenc',
            '-f', 'null', '-'
        ]

        result = subprocess.run(cmd, capture_output=True, timeout=10)
        return result.returncode == 0

    except Exception:
        return False


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

# Initialize the singleton on module import
_manager = GPUManager.get_instance()


if __name__ == "__main__":
    # Run diagnostics when executed directly
    print_gpu_diagnostics()

    print("Testing NVENC availability...")
    if test_nvenc_available():
        print("NVENC is available and working!")
    else:
        print("NVENC is NOT available - FFmpeg will use CPU encoding")

    print("\nExample usage:")
    print(f"  GPU count: {get_gpu_count()}")
    print(f"  Total NVENC sessions: {get_gpu_info()['total_nvenc_sessions']}")

    for i in range(4):
        print(f"  Task {i} -> GPU {assign_gpu(i)}")
