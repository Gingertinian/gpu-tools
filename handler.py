"""
RunPod Serverless Handler for Farmium GPU Tools

This handler processes GPU jobs for:
- Vignettes: Video overlay effects
- Spoofer: Image/video transformations for duplicate detection evasion
- Captioner: Add text captions to images/videos
- Resize: Resize/crop images/videos to target resolution
- Pic to Video: Convert images to static videos
- BG Remove: Remove background from images
- Upscale: AI upscaling with Real-ESRGAN
- Face Swap: Swap faces using InsightFace
- Video Gen: AI video generation from images

Usage:
    Deploy to RunPod Serverless with network volume containing tools/ directory
"""

import runpod
import os
import sys
import requests
import tempfile
import shutil
import zipfile
import subprocess
import asyncio
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from dataclasses import dataclass
from typing import Optional, List, Callable, Any, Tuple

# ==================== Constants ====================

# Already-compressed file formats (ZIP_STORED = no compression, saves CPU)
COMPRESSED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.gif',  # Images (already compressed)
    '.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v',  # Videos (already compressed)
    '.mp3', '.aac', '.ogg', '.flac',  # Audio (already compressed)
    '.zip', '.gz', '.bz2', '.xz', '.7z',  # Archives
}

# Larger chunk size for faster downloads (1MB instead of 8KB)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB

# =============================================================================
# MEGA-GPU I/O CONFIGURATION
# =============================================================================
# With 20+ GPUs, the bottleneck shifts from GPU to I/O. We need enough
# download/upload workers to keep all GPUs saturated.
#
# Formula: I/O workers = max(BASE_WORKERS, GPU_COUNT × WORKERS_PER_GPU)
#
# Example scaling:
#   1 GPU:  max(50, 1×10) = 50 workers
#   4 GPUs: max(50, 4×10) = 50 workers
#   9 GPUs: max(50, 9×10) = 90 workers
#   20 GPUs: max(50, 20×10) = 200 workers

BASE_DOWNLOAD_WORKERS = 50    # Minimum download connections
BASE_UPLOAD_WORKERS = 50      # Minimum upload connections
IO_WORKERS_PER_GPU = 10       # Additional I/O workers per GPU

# Processing configuration
PROCESSING_OVERPROVISION = 1.2  # Reduced from 1.5, semaphores handle blocking


def get_optimal_io_workers(gpu_count: int) -> tuple:
    """
    Calculate optimal download/upload workers based on GPU count.
    For mega-GPU workers (9-20+ GPUs), we need more I/O parallelism.
    """
    download_workers = max(BASE_DOWNLOAD_WORKERS, gpu_count * IO_WORKERS_PER_GPU)
    upload_workers = max(BASE_UPLOAD_WORKERS, gpu_count * IO_WORKERS_PER_GPU)

    # Cap at reasonable maximum to avoid file descriptor issues
    MAX_IO_WORKERS = 500
    download_workers = min(download_workers, MAX_IO_WORKERS)
    upload_workers = min(upload_workers, MAX_IO_WORKERS)

    return download_workers, upload_workers

# Add tools directory to path
WORKSPACE = os.environ.get('WORKSPACE', '/workspace')
sys.path.insert(0, os.path.join(WORKSPACE, 'tools'))

# ==================== Import Tool Processors ====================

# Vignettes
try:
    from vignettes.processor import process_vignettes
except ImportError:
    process_vignettes = None

# Spoofer
try:
    from spoofer.processor import process_spoofer
except ImportError:
    process_spoofer = None

try:
    from spoofer.processor_fast import process_spoofer_fast
except ImportError:
    process_spoofer_fast = None

# Captioner
try:
    from captioner.processor import process_captioner
except ImportError:
    process_captioner = None

# Resize
try:
    from resize.processor import process_resize
except ImportError:
    process_resize = None

# Pic to Video
try:
    from pic_to_video.processor import process_pic_to_video
except ImportError:
    process_pic_to_video = None

# BG Remove
try:
    from bg_remove.processor import process_bg_remove
except ImportError:
    process_bg_remove = None

# Upscale
try:
    from upscale.processor import process_upscale, process_upscale_video
except ImportError:
    process_upscale = None
    process_upscale_video = None

# Face Swap
try:
    from face_swap.processor import process_face_swap, process_face_swap_video
except ImportError:
    process_face_swap = None
    process_face_swap_video = None

# Video Gen
try:
    from video_gen.processor import process_video_gen
except ImportError:
    process_video_gen = None

# Video Reframe - Use GPU-optimized processor
try:
    from video_reframe.processor_gpu import process_video_reframe
    print("[Handler] Using GPU-optimized video_reframe processor")
except ImportError:
    try:
        from video_reframe.processor import process_video_reframe
        print("[Handler] Using standard video_reframe processor")
    except ImportError:
        process_video_reframe = None


# ==================== Helper Functions ====================

def get_zip_compression(filepath: str) -> int:
    """
    Get optimal ZIP compression method for a file.
    Returns ZIP_STORED for already-compressed formats (saves CPU).
    Returns ZIP_DEFLATED for uncompressed formats.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in COMPRESSED_EXTENSIONS:
        return zipfile.ZIP_STORED  # No compression - file already compressed
    return zipfile.ZIP_DEFLATED  # Compress uncompressed files


def download_file(url: str, path: str) -> None:
    """Download file from presigned URL with optimized chunk size"""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
            f.write(chunk)


def upload_file(path: str, url: str, max_retries: int = 3) -> dict:
    """
    Upload file to presigned URL with retry logic and verification.
    Returns dict with upload status and details.
    Raises exception on failure after all retries.
    """
    ext = os.path.splitext(path)[1].lower()
    content_types = {
        '.zip': 'application/zip',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm',
        '.mkv': 'video/x-matroska',
        '.avi': 'video/x-msvideo',
        '.m4v': 'video/x-m4v',
    }
    content_type = content_types.get(ext, 'application/octet-stream')

    # Verify file exists and has content
    if not os.path.exists(path):
        print(f"[Upload] ERROR: File not found: {path}")
        raise FileNotFoundError(f"Output file not found: {path}")

    file_size = os.path.getsize(path)
    if file_size == 0:
        print(f"[Upload] ERROR: File is empty: {path}")
        raise ValueError(f"Output file is empty: {path}")

    # Extract destination key from URL for logging (hide signature)
    url_path = url.split('?')[0].split('/')[-2:] if '?' in url else ['unknown']
    dest_key = '/'.join(url_path)
    print(f"[Upload] Starting: {os.path.basename(path)} ({file_size/1024/1024:.2f}MB) -> {dest_key}")

    last_error = None
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            with open(path, 'rb') as f:
                response = requests.put(
                    url,
                    data=f,
                    headers={'Content-Type': content_type},
                    timeout=600
                )
                response.raise_for_status()

            elapsed = time.time() - start_time
            speed_mbps = (file_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
            print(f"[Upload] SUCCESS: {os.path.basename(path)} uploaded in {elapsed:.1f}s ({speed_mbps:.1f} MB/s) - Status: {response.status_code}")

            # Upload succeeded
            return {
                "uploaded": True,
                "size": file_size,
                "contentType": content_type,
                "attempts": attempt + 1
            }
        except requests.exceptions.RequestException as e:
            last_error = e
            error_detail = str(e)
            if hasattr(e, 'response') and e.response is not None:
                error_detail = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            print(f"[Upload] FAILED attempt {attempt+1}/{max_retries}: {error_detail}")
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = (2 ** attempt)
                print(f"[Upload] Retrying in {wait_time}s...")
                time.sleep(wait_time)

    # All retries failed
    print(f"[Upload] FATAL: All {max_retries} attempts failed for {path}")
    raise RuntimeError(f"Upload failed after {max_retries} attempts: {last_error}")


def create_progress_callback(job):
    """Create a progress callback that sends updates to RunPod"""
    def callback(progress: float, message: str = None):
        scaled_progress = int(10 + progress * 80)
        update = {"progress": scaled_progress}
        if message:
            update["status"] = message
        runpod.serverless.progress_update(job, update)
    return callback


def get_file_extension(url: str, config: dict) -> str:
    """Extract file extension from URL or config"""
    ext = config.get("inputExtension")
    if not ext:
        url_path = url.split('?')[0]
        ext = os.path.splitext(url_path)[1].lower() or ".jpg"
    return ext


def is_image(ext: str) -> bool:
    """Check if extension is an image format"""
    return ext.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']


def is_video(ext: str) -> bool:
    """Check if extension is a video format"""
    return ext.lower() in ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v']


# ==================== Async Pipeline Data Classes ====================

@dataclass
class WorkItem:
    """Represents a single item in the processing pipeline"""
    index: int
    input_url: str
    output_url: str
    input_path: str
    output_path: str
    tool: str
    config: dict
    gpu_id: int = 0
    # Status tracking
    download_complete: bool = False
    process_complete: bool = False
    upload_complete: bool = False
    error: Optional[str] = None
    result: Optional[dict] = None
    # Timing
    download_time: float = 0
    process_time: float = 0
    upload_time: float = 0
    input_size: int = 0
    output_size: int = 0


# ==================== Batch Mode Processor ====================

# Global cache for GPU info (GPU detection is slow)
_GPU_INFO_CACHE = None
_GPU_LOAD_TRACKER = None  # Tracks current load per GPU


def get_gpu_info(force_refresh: bool = False) -> dict:
    """
    Detect GPU type, count GPUs, memory, and determine NVENC session limit.
    Results are cached for performance (GPU detection via nvidia-smi is slow).

    Returns:
        dict with keys:
        - gpu_name: str - Name of first GPU
        - gpu_names: list - Names of all GPUs
        - gpu_type: str - 'datacenter' or 'consumer'
        - gpu_count: int - Number of GPUs
        - gpu_memory_mb: list - Memory per GPU in MB
        - gpu_memory_free_mb: list - Free memory per GPU in MB
        - nvenc_sessions: int - Total NVENC sessions available
        - nvenc_sessions_per_gpu: int - NVENC sessions per GPU
    """
    global _GPU_INFO_CACHE

    # Return cached result if available
    if _GPU_INFO_CACHE is not None and not force_refresh:
        return _GPU_INFO_CACHE

    try:
        # Query comprehensive GPU info: name, memory total, memory free
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            gpu_count = len(gpu_lines)

            gpu_names = []
            gpu_memory_mb = []
            gpu_memory_free_mb = []
            gpu_utilization = []

            for line in gpu_lines:
                parts = [p.strip() for p in line.split(',')]
                gpu_names.append(parts[0] if len(parts) > 0 else 'Unknown')
                gpu_memory_mb.append(int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0)
                gpu_memory_free_mb.append(int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0)
                gpu_utilization.append(int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0)

            gpu_name = gpu_names[0] if gpu_names else 'Unknown'

            # Datacenter GPUs have unlimited NVENC sessions
            datacenter_keywords = ['A100', 'A6000', 'A5000', 'A4000', 'A4500', 'A40', 'A30', 'A10',
                                   'V100', 'T4', 'Quadro', 'Tesla', 'H100', 'L40', 'L4', 'RTX 6000', 'RTX 4090']
            is_datacenter = any(kw in gpu_name for kw in datacenter_keywords)
            gpu_type = 'datacenter' if is_datacenter else 'consumer'

            # NVENC session limits: datacenter = unlimited (use 12), consumer = 3
            base_sessions = 12 if is_datacenter else 3
            nvenc_sessions = base_sessions * gpu_count

            print(f"[GPU Detection] Found {gpu_count} GPU(s): {gpu_name}")
            print(f"[GPU Detection] Type: {gpu_type}, Sessions/GPU: {base_sessions}, Total: {nvenc_sessions}")
            print(f"[GPU Detection] Memory: {gpu_memory_mb} MB, Free: {gpu_memory_free_mb} MB")
            print(f"[GPU Detection] Utilization: {gpu_utilization}%")

            _GPU_INFO_CACHE = {
                'gpu_name': gpu_name,
                'gpu_names': gpu_names,
                'gpu_type': gpu_type,
                'gpu_count': gpu_count,
                'gpu_memory_mb': gpu_memory_mb,
                'gpu_memory_free_mb': gpu_memory_free_mb,
                'gpu_utilization': gpu_utilization,
                'nvenc_sessions': nvenc_sessions,
                'nvenc_sessions_per_gpu': base_sessions
            }
            return _GPU_INFO_CACHE

    except Exception as e:
        print(f"[GPU Detection] Error: {e}")

    # Fallback for no GPU or error
    _GPU_INFO_CACHE = {
        'gpu_name': 'Unknown',
        'gpu_names': ['Unknown'],
        'gpu_type': 'default',
        'gpu_count': 1,
        'gpu_memory_mb': [0],
        'gpu_memory_free_mb': [0],
        'gpu_utilization': [0],
        'nvenc_sessions': 2,
        'nvenc_sessions_per_gpu': 2
    }
    return _GPU_INFO_CACHE


class GPULoadTracker:
    """
    Tracks current load (active jobs) per GPU for smart work distribution.
    Thread-safe for concurrent batch processing.

    IMPROVED: Uses semaphores to enforce REAL limits per GPU.
    Critical for scaling to 9+ GPUs without over-subscription.
    """
    def __init__(self, gpu_count: int, max_sessions_per_gpu: int):
        self.gpu_count = gpu_count
        self.max_sessions_per_gpu = max_sessions_per_gpu
        self.active_jobs = [0] * gpu_count  # Current active jobs per GPU
        self.total_jobs = [0] * gpu_count   # Total jobs assigned per GPU
        self.lock = threading.Lock()

        # CRITICAL: Semaphores enforce REAL limits per GPU
        # Prevents over-subscription when scaling to many GPUs
        self._gpu_semaphores = {i: threading.Semaphore(max_sessions_per_gpu) for i in range(gpu_count)}

    def get_best_gpu(self, blocking: bool = True, timeout: float = None) -> int:
        """
        Get GPU with capacity available. Uses semaphores for REAL enforcement.

        Args:
            blocking: If True, wait for GPU to become available
            timeout: Max seconds to wait (None = infinite)

        Returns GPU ID (0-indexed), or -1 if non-blocking and none available.
        """
        # First, find GPUs sorted by total jobs (prefer less used)
        with self.lock:
            gpu_order = sorted(range(self.gpu_count), key=lambda i: self.total_jobs[i])

        # Try to acquire from least-used GPU first (non-blocking)
        for gpu_id in gpu_order:
            acquired = self._gpu_semaphores[gpu_id].acquire(blocking=False)
            if acquired:
                with self.lock:
                    self.active_jobs[gpu_id] += 1
                    self.total_jobs[gpu_id] += 1
                return gpu_id

        # If non-blocking and none available
        if not blocking:
            return -1

        # Blocking: wait for ANY GPU
        start_gpu = gpu_order[0]
        for i in range(self.gpu_count):
            gpu_id = (start_gpu + i) % self.gpu_count
            acquired = self._gpu_semaphores[gpu_id].acquire(blocking=True, timeout=timeout)
            if acquired:
                with self.lock:
                    self.active_jobs[gpu_id] += 1
                    self.total_jobs[gpu_id] += 1
                return gpu_id

        return -1  # Timeout

    def assign_job(self, gpu_id: int) -> None:
        """Mark a job as assigned to a GPU (legacy compatibility)."""
        # Note: With semaphore-based get_best_gpu(), this is called internally
        pass  # Assignment is handled in get_best_gpu()

    def complete_job(self, gpu_id: int) -> None:
        """Mark a job as completed on a GPU."""
        if gpu_id < 0 or gpu_id >= self.gpu_count:
            return

        with self.lock:
            self.active_jobs[gpu_id] = max(0, self.active_jobs[gpu_id] - 1)

        # Release semaphore - allows waiting thread to proceed
        self._gpu_semaphores[gpu_id].release()

    def get_stats(self) -> dict:
        """Get current load statistics."""
        with self.lock:
            return {
                'active_jobs': list(self.active_jobs),
                'total_jobs': list(self.total_jobs),
                'capacity_per_gpu': self.max_sessions_per_gpu,
                'total_capacity': self.max_sessions_per_gpu * self.gpu_count
            }


# ==================== Async Pipeline Implementation ====================

class AsyncPipelineProcessor:
    """
    Async I/O pipeline processor that overlaps downloads, processing, and uploads
    to maximize GPU utilization.

    MEGA-GPU OPTIMIZED Architecture:
    - Download workers: Scales with GPU count (10 per GPU, min 50)
    - Processing workers: NVENC sessions × 1.2 (semaphores handle blocking)
    - Upload workers: Scales with GPU count (10 per GPU, min 50)

    This ensures:
    1. GPU never waits for downloads (prefetch buffer scaled to GPU count)
    2. Uploads don't block processing (async upload queue)
    3. Full overlap of all three stages
    4. SCALES to 20+ GPUs without bottleneck
    """

    def __init__(
        self,
        gpu_info: dict,
        job,
        temp_dir: str,
        download_workers: int = None,
        upload_workers: int = None
    ):
        self.gpu_info = gpu_info
        self.job = job
        self.temp_dir = temp_dir

        # MEGA-GPU: Scale I/O workers based on GPU count
        gpu_count = gpu_info.get('gpu_count', 1)
        optimal_download, optimal_upload = get_optimal_io_workers(gpu_count)

        self.download_workers = download_workers or optimal_download
        self.upload_workers = upload_workers or optimal_upload

        # Calculate processing workers: smaller buffer, semaphores handle blocking
        nvenc_sessions = gpu_info.get('nvenc_sessions', 6)
        self.processing_workers = int(nvenc_sessions * PROCESSING_OVERPROVISION)

        # GPU load tracking
        gpu_count = gpu_info.get('gpu_count', 1)
        self.gpu_tracker = GPULoadTracker(gpu_count, gpu_info.get('nvenc_sessions_per_gpu', 3))

        # Thread-safe queues for pipeline stages
        self.download_queue: Queue[WorkItem] = Queue()  # Items ready to download
        self.process_queue: Queue[WorkItem] = Queue()   # Items ready to process (downloaded)
        self.upload_queue: Queue[WorkItem] = Queue()    # Items ready to upload (processed)

        # Results tracking
        self.results: List[Optional[dict]] = []
        self.results_lock = threading.Lock()

        # Completion tracking
        self.total_items = 0
        self.completed_downloads = 0
        self.completed_processing = 0
        self.completed_uploads = 0
        self.failed_count = 0
        self.stats_lock = threading.Lock()

        # Shutdown flag
        self.shutdown = False

        # Download cache for spoofer batch mode (same input, multiple variations)
        # Key: input_path, Value: threading.Event (set when download complete)
        self._download_cache: Dict[str, threading.Event] = {}
        self._download_cache_lock = threading.Lock()
        self._download_cache_errors: Dict[str, str] = {}  # Track download errors

        # Thread pools
        self.download_executor: Optional[ThreadPoolExecutor] = None
        self.process_executor: Optional[ThreadPoolExecutor] = None
        self.upload_executor: Optional[ThreadPoolExecutor] = None

        # Timing metrics
        self.start_time = 0
        self.pipeline_metrics = {
            'download_wait_time': 0,
            'process_wait_time': 0,
            'upload_wait_time': 0,
            'gpu_idle_time': 0
        }

    def _download_worker(self, item: WorkItem) -> None:
        """Download a single file and add to process queue.

        Uses caching to avoid downloading the same input file multiple times
        (important for spoofer batch mode with variations).
        """
        start_time = time.time()
        input_path = item.input_path
        should_download = False
        download_event = None

        # Check download cache - avoid downloading same file multiple times
        with self._download_cache_lock:
            if input_path in self._download_cache:
                # Another worker is downloading or has downloaded this file
                download_event = self._download_cache[input_path]
            else:
                # First worker for this input path - we'll do the download
                download_event = threading.Event()
                self._download_cache[input_path] = download_event
                should_download = True

        try:
            if should_download:
                # We're responsible for downloading this file
                download_file(item.input_url, item.input_path)
                item.input_size = os.path.getsize(item.input_path) if os.path.exists(item.input_path) else 0
                item.download_time = time.time() - start_time

                speed_mbps = (item.input_size / 1024 / 1024) / item.download_time if item.download_time > 0 else 0
                print(f"[Download {item.index}] {item.input_size/1024/1024:.2f}MB in {item.download_time:.2f}s ({speed_mbps:.2f} MB/s)")

                # Signal that download is complete
                download_event.set()
            else:
                # Wait for another worker to finish downloading
                print(f"[Download {item.index}] Waiting for cached download of {os.path.basename(input_path)}...")
                download_event.wait(timeout=300)  # 5 min timeout

                # Check if download failed
                with self._download_cache_lock:
                    if input_path in self._download_cache_errors:
                        raise Exception(self._download_cache_errors[input_path])

                item.input_size = os.path.getsize(item.input_path) if os.path.exists(item.input_path) else 0
                item.download_time = time.time() - start_time
                print(f"[Download {item.index}] Using cached file (waited {item.download_time:.2f}s)")

            item.download_complete = True

            # Add to process queue immediately
            self.process_queue.put(item)

            with self.stats_lock:
                self.completed_downloads += 1

        except Exception as e:
            item.error = f"Download failed: {str(e)}"
            item.download_complete = False

            print(f"[Download {item.index}] FAILED: {str(e)[:100]}")

            # Record error in cache so other workers waiting on this file know it failed
            if should_download:
                with self._download_cache_lock:
                    self._download_cache_errors[input_path] = str(e)
                download_event.set()  # Signal completion (even if failed)

            # Record failure
            with self.stats_lock:
                self.failed_count += 1
                self.completed_downloads += 1

            # Store result
            with self.results_lock:
                self.results[item.index] = {
                    "index": item.index,
                    "status": "failed",
                    "error": item.error
                }

    def _process_worker(self) -> None:
        """
        Processing worker that pulls from process queue.
        Runs continuously until shutdown signal.
        """
        while not self.shutdown:
            try:
                # Wait for item with timeout to check shutdown
                wait_start = time.time()
                try:
                    item = self.process_queue.get(timeout=0.1)
                except Empty:
                    continue

                wait_time = time.time() - wait_start
                with self.stats_lock:
                    self.pipeline_metrics['process_wait_time'] += wait_time

                if item is None:  # Poison pill
                    break

                # Skip if download failed
                if not item.download_complete or item.error:
                    continue

                # Assign GPU and process
                gpu_id = self.gpu_tracker.get_best_gpu()
                item.gpu_id = gpu_id
                self.gpu_tracker.assign_job(gpu_id)

                start_time = time.time()

                try:
                    result = self._process_single_item(item)
                    item.process_time = time.time() - start_time

                    if result.get("status") == "completed":
                        item.process_complete = True
                        item.output_size = result.get("output_size", 0)
                        item.result = result

                        print(f"[Process {item.index}] GPU {gpu_id} completed in {item.process_time:.2f}s")

                        # Add to upload queue immediately
                        self.upload_queue.put(item)
                    else:
                        item.error = result.get("error", "Unknown processing error")
                        print(f"[Process {item.index}] GPU {gpu_id} FAILED: {item.error[:100]}")

                        with self.stats_lock:
                            self.failed_count += 1

                        with self.results_lock:
                            self.results[item.index] = {
                                "index": item.index,
                                "status": "failed",
                                "error": item.error,
                                "gpu_id": gpu_id
                            }

                except Exception as e:
                    item.error = str(e)
                    item.process_time = time.time() - start_time

                    print(f"[Process {item.index}] GPU {gpu_id} EXCEPTION: {str(e)[:100]}")

                    with self.stats_lock:
                        self.failed_count += 1

                    with self.results_lock:
                        self.results[item.index] = {
                            "index": item.index,
                            "status": "failed",
                            "error": item.error,
                            "gpu_id": gpu_id
                        }
                finally:
                    self.gpu_tracker.complete_job(gpu_id)

                    with self.stats_lock:
                        self.completed_processing += 1

            except Exception as e:
                print(f"[Process Worker] Unexpected error: {e}")

    def _process_single_item(self, item: WorkItem) -> dict:
        """Process a single work item with the specified tool"""
        input_ext = os.path.splitext(item.input_path)[1].lower()
        is_video_file = is_video(input_ext)

        # GPU config for processors
        gpu_config = {
            **item.config,
            "_gpu_id": item.gpu_id,
            "_cuda_device": item.gpu_id,
            "_ffmpeg_loglevel": "info"
        }

        tool = item.tool

        # Process based on tool type
        if tool == "spoofer":
            if is_image(input_ext) and process_spoofer_fast is not None:
                process_spoofer_fast(item.input_path, item.output_path, gpu_config)
            elif process_spoofer is not None:
                process_spoofer(item.input_path, item.output_path, gpu_config)
            else:
                return {"status": "failed", "error": "Spoofer not available"}

        elif tool == "captioner":
            if process_captioner is not None:
                captioner_config = {**gpu_config, "imageIndex": item.index}
                process_captioner(item.input_path, item.output_path, captioner_config)
            else:
                return {"status": "failed", "error": "Captioner not available"}

        elif tool == "vignettes":
            if process_vignettes is not None:
                process_vignettes(item.input_path, item.output_path, gpu_config)
            else:
                return {"status": "failed", "error": "Vignettes not available"}

        elif tool == "resize":
            if process_resize is not None:
                process_resize(item.input_path, item.output_path, gpu_config)
            else:
                return {"status": "failed", "error": "Resize not available"}

        elif tool == "pic_to_video":
            if process_pic_to_video is not None:
                process_pic_to_video(item.input_path, item.output_path, gpu_config)
            else:
                return {"status": "failed", "error": "Pic to Video not available"}

        elif tool == "bg_remove":
            if process_bg_remove is not None:
                process_bg_remove(item.input_path, item.output_path, gpu_config)
            else:
                return {"status": "failed", "error": "BG Remove not available"}

        elif tool == "upscale":
            if is_video_file and process_upscale_video is not None:
                process_upscale_video(item.input_path, item.output_path, gpu_config)
            elif process_upscale is not None:
                process_upscale(item.input_path, item.output_path, gpu_config)
            else:
                return {"status": "failed", "error": "Upscale not available"}

        elif tool == "face_swap":
            if is_video_file and process_face_swap_video is not None:
                process_face_swap_video(item.input_path, item.output_path, gpu_config)
            elif process_face_swap is not None:
                process_face_swap(item.input_path, item.output_path, gpu_config)
            else:
                return {"status": "failed", "error": "Face Swap not available"}

        elif tool == "video_reframe":
            if process_video_reframe is not None:
                result = process_video_reframe(item.input_path, item.output_path, gpu_config)
                # video_reframe may change output path for images
                if result and result.get('outputPath'):
                    item.output_path = result['outputPath']
            else:
                return {"status": "failed", "error": "Video Reframe not available"}

        else:
            return {"status": "failed", "error": f"Unknown tool: {tool}"}

        # Check output
        if os.path.exists(item.output_path) and os.path.getsize(item.output_path) > 0:
            output_size = os.path.getsize(item.output_path)
            return {
                "status": "completed",
                "output_path": item.output_path,
                "output_size": output_size,
                "gpu_id": item.gpu_id
            }
        else:
            return {"status": "failed", "error": "Output file not created"}

    def _upload_worker(self) -> None:
        """
        Upload worker that pulls from upload queue.
        Runs continuously until shutdown signal.
        """
        while not self.shutdown:
            try:
                # Wait for item with timeout to check shutdown
                wait_start = time.time()
                try:
                    item = self.upload_queue.get(timeout=0.1)
                except Empty:
                    continue

                wait_time = time.time() - wait_start
                with self.stats_lock:
                    self.pipeline_metrics['upload_wait_time'] += wait_time

                if item is None:  # Poison pill
                    break

                # Skip if processing failed
                if not item.process_complete or item.error:
                    continue

                start_time = time.time()

                try:
                    upload_result = upload_file(item.output_path, item.output_url)
                    item.upload_time = time.time() - start_time
                    item.upload_complete = True

                    speed_mbps = (item.output_size / 1024 / 1024) / item.upload_time if item.upload_time > 0 else 0
                    print(f"[Upload {item.index}] {item.output_size/1024/1024:.2f}MB in {item.upload_time:.2f}s ({speed_mbps:.2f} MB/s)")

                    # Record success
                    with self.results_lock:
                        self.results[item.index] = {
                            "index": item.index,
                            "status": "completed",
                            "output_path": item.output_path,
                            "gpu_id": item.gpu_id,
                            "processing_time": item.process_time,
                            "input_size": item.input_size,
                            "output_size": item.output_size,
                            "download_time": item.download_time,
                            "upload_time": item.upload_time
                        }

                except Exception as e:
                    item.error = f"Upload failed: {str(e)}"
                    item.upload_time = time.time() - start_time

                    print(f"[Upload {item.index}] FAILED: {str(e)[:100]}")

                    with self.stats_lock:
                        self.failed_count += 1

                    with self.results_lock:
                        self.results[item.index] = {
                            "index": item.index,
                            "status": "upload_failed",
                            "error": item.error,
                            "gpu_id": item.gpu_id
                        }
                finally:
                    with self.stats_lock:
                        self.completed_uploads += 1

            except Exception as e:
                print(f"[Upload Worker] Unexpected error: {e}")

    def process_batch(
        self,
        input_urls: List[str],
        output_urls: List[str],
        tool: str,
        config: dict
    ) -> dict:
        """
        Process a batch of files using the async pipeline.

        This method orchestrates:
        1. Starting all worker threads
        2. Queuing downloads (starts processing as files become ready)
        3. Waiting for all stages to complete
        4. Returning aggregated results
        """
        self.start_time = time.time()
        self.total_items = len(input_urls)
        self.results = [None] * self.total_items

        gpu_count = self.gpu_info.get('gpu_count', 1)
        nvenc_sessions = self.gpu_info.get('nvenc_sessions', 6)
        gpu_name = self.gpu_info.get('gpu_name', 'Unknown')
        sessions_per_gpu = self.gpu_info.get('nvenc_sessions_per_gpu', 16)

        # Calculate theoretical throughput (assuming ~2s per video encode)
        theoretical_throughput = nvenc_sessions / 2.0  # videos/second
        estimated_time = self.total_items / theoretical_throughput if theoretical_throughput > 0 else 0

        print(f"[AsyncPipeline] =============== MEGA-GPU BATCH START ===============")
        print(f"[AsyncPipeline] GPU Model: {gpu_name}")
        print(f"[AsyncPipeline] GPU Count: {gpu_count} | Sessions/GPU: {sessions_per_gpu}")
        print(f"[AsyncPipeline] Total NVENC Capacity: {nvenc_sessions} parallel encodes")
        print(f"[AsyncPipeline] ---")
        print(f"[AsyncPipeline] Items to Process: {self.total_items} | Tool: {tool}")
        print(f"[AsyncPipeline] Download Workers: {self.download_workers}")
        print(f"[AsyncPipeline] Process Workers: {self.processing_workers}")
        print(f"[AsyncPipeline] Upload Workers: {self.upload_workers}")
        print(f"[AsyncPipeline] ---")
        print(f"[AsyncPipeline] Theoretical Throughput: ~{theoretical_throughput:.1f} videos/sec")
        print(f"[AsyncPipeline] Estimated Time: ~{estimated_time:.1f}s ({estimated_time/60:.1f} min)")
        print(f"[AsyncPipeline] =======================================================")

        # Check for spoofer variations - need to expand work items
        # Supports both 'variations' and legacy 'copies' for backward compatibility
        variations = config.get("variations") or config.get("copies") or config.get("options", {}).get("variations") or config.get("options", {}).get("copies", 1)
        is_spoofer_batch = tool == "spoofer" and variations > 1

        if is_spoofer_batch:
            print(f"[AsyncPipeline] SPOOFER BATCH MODE: {len(input_urls)} inputs × {variations} variations = {len(input_urls) * variations} outputs")

        # Prepare work items
        work_items = []
        work_index = 0

        for i, (input_url, output_url) in enumerate(zip(input_urls, output_urls)):
            ext = get_file_extension(input_url, config)

            # Determine output extension
            if tool == "bg_remove":
                output_ext = ".png"
            elif tool in ["pic_to_video", "video_gen"]:
                output_ext = ".mp4"
            elif tool in ["spoofer", "captioner", "vignettes", "resize", "video_reframe"]:
                output_ext = ext if is_video(ext) else '.jpg'
            else:
                output_ext = ext

            if is_spoofer_batch:
                # For spoofer with variations > 1, create N work items per input
                # Each variation gets a unique seed and output path
                for var_idx in range(variations):
                    var_config = {
                        **config,
                        "variations": 1,  # Process single variation per work item
                        "copies": 1,  # Legacy support
                        "_variation_index": var_idx,
                        "_seed": int(time.time() * 1000) + work_index * 12345,  # Unique seed per variation
                    }
                    # Remove options.copies/variations if present (set to 1)
                    if "options" in var_config:
                        var_config["options"] = {**var_config["options"], "variations": 1, "copies": 1}

                    item = WorkItem(
                        index=work_index,
                        input_url=input_url,  # Same input URL for all variations
                        output_url=f"{output_url.rsplit('.', 1)[0]}_v{var_idx:04d}.{output_url.rsplit('.', 1)[1]}" if '.' in output_url else f"{output_url}_v{var_idx:04d}",
                        input_path=os.path.join(self.temp_dir, f"input_{i}{ext}"),  # Share input file
                        output_path=os.path.join(self.temp_dir, f"output_{i}_v{var_idx:04d}{output_ext}"),
                        tool=tool,
                        config=var_config
                    )
                    work_items.append(item)
                    work_index += 1
            else:
                # Standard 1:1 mapping for non-batch tools
                item = WorkItem(
                    index=work_index,
                    input_url=input_url,
                    output_url=output_url,
                    input_path=os.path.join(self.temp_dir, f"input_{i}{ext}"),
                    output_path=os.path.join(self.temp_dir, f"output_{i}{output_ext}"),
                    tool=tool,
                    config=config
                )
                work_items.append(item)
                work_index += 1

        # Update total items count for progress tracking
        self.total_items = len(work_items)
        self.results = [None] * self.total_items

        if is_spoofer_batch:
            print(f"[AsyncPipeline] Created {self.total_items} work items for spoofer batch")

        # Start thread pools
        self.download_executor = ThreadPoolExecutor(max_workers=self.download_workers)
        self.upload_executor = ThreadPoolExecutor(max_workers=self.upload_workers)

        # Start processing workers (persistent threads that pull from queue)
        process_threads = []
        for _ in range(self.processing_workers):
            t = threading.Thread(target=self._process_worker, daemon=True)
            t.start()
            process_threads.append(t)

        # Start upload workers (persistent threads that pull from queue)
        upload_threads = []
        for _ in range(self.upload_workers):
            t = threading.Thread(target=self._upload_worker, daemon=True)
            t.start()
            upload_threads.append(t)

        # Submit all downloads (they will feed the process queue as they complete)
        download_futures = []
        for item in work_items:
            future = self.download_executor.submit(self._download_worker, item)
            download_futures.append(future)

        # Progress reporting thread
        def progress_reporter():
            while not self.shutdown:
                with self.stats_lock:
                    total_done = self.completed_uploads + self.failed_count
                    # Avoid counting failed items twice
                    downloads = self.completed_downloads
                    processing = self.completed_processing
                    uploads = self.completed_uploads

                # Calculate progress (weight: download 10%, process 70%, upload 20%)
                download_progress = (downloads / self.total_items) * 10
                process_progress = (processing / self.total_items) * 70
                upload_progress = (uploads / self.total_items) * 20
                overall_progress = int(download_progress + process_progress + upload_progress)

                runpod.serverless.progress_update(self.job, {
                    "progress": min(overall_progress, 99),
                    "status": f"D:{downloads}/{self.total_items} P:{processing}/{self.total_items} U:{uploads}/{self.total_items}"
                })

                if total_done >= self.total_items:
                    break

                time.sleep(0.5)

        progress_thread = threading.Thread(target=progress_reporter, daemon=True)
        progress_thread.start()

        # Wait for all downloads to complete
        for future in download_futures:
            future.result()

        print(f"[AsyncPipeline] All downloads complete, waiting for processing...")

        # Wait for process queue to drain
        while True:
            with self.stats_lock:
                if self.completed_processing >= self.total_items - self.failed_count:
                    break
            time.sleep(0.1)

        print(f"[AsyncPipeline] All processing complete, waiting for uploads...")

        # Wait for upload queue to drain
        while True:
            with self.stats_lock:
                total_done = self.completed_uploads + self.failed_count
                if total_done >= self.total_items:
                    break
            time.sleep(0.1)

        # Signal shutdown
        self.shutdown = True

        # Send poison pills to wake up waiting workers
        for _ in range(self.processing_workers):
            self.process_queue.put(None)
        for _ in range(self.upload_workers):
            self.upload_queue.put(None)

        # Wait for worker threads to finish
        for t in process_threads:
            t.join(timeout=1.0)
        for t in upload_threads:
            t.join(timeout=1.0)

        # Shutdown executors
        self.download_executor.shutdown(wait=False)
        self.upload_executor.shutdown(wait=False)

        # Calculate final stats
        elapsed = time.time() - self.start_time

        completed_count = sum(1 for r in self.results if r and r.get("status") == "completed")
        failed_count = self.total_items - completed_count

        # GPU distribution
        gpu_distribution = [0] * gpu_count
        total_processing_time = 0
        total_input_size = 0
        total_output_size = 0

        for r in self.results:
            if r and r.get("status") == "completed":
                gpu_id = r.get("gpu_id", 0)
                if 0 <= gpu_id < gpu_count:
                    gpu_distribution[gpu_id] += 1
                total_processing_time += r.get("processing_time", 0)
                total_input_size += r.get("input_size", 0)
                total_output_size += r.get("output_size", 0)

        throughput_mbps = (total_input_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
        parallelism_efficiency = (total_processing_time / elapsed / nvenc_sessions * 100) if elapsed > 0 else 0

        print(f"[AsyncPipeline] ===== BATCH COMPLETE =====")
        print(f"[AsyncPipeline] Total time: {elapsed:.2f}s")
        print(f"[AsyncPipeline] Completed: {completed_count}/{self.total_items}, Failed: {failed_count}")
        print(f"[AsyncPipeline] Throughput: {throughput_mbps:.2f} MB/s")
        print(f"[AsyncPipeline] GPU distribution: {gpu_distribution}")
        print(f"[AsyncPipeline] Parallelism efficiency: {parallelism_efficiency:.1f}%")
        print(f"[AsyncPipeline] Pipeline metrics: {self.pipeline_metrics}")

        runpod.serverless.progress_update(self.job, {
            "progress": 100,
            "status": "Completed"
        })

        return {
            "status": "completed",
            "mode": "async_pipeline",
            "gpu": self.gpu_info['gpu_name'],
            "gpu_type": self.gpu_info['gpu_type'],
            "gpu_count": gpu_count,
            "parallel_sessions": nvenc_sessions,
            "total": self.total_items,
            "completed": completed_count,
            "failed": failed_count,
            "gpu_distribution": gpu_distribution,
            "job_time_seconds": elapsed,
            "total_processing_time_seconds": total_processing_time,
            "throughput_mbps": throughput_mbps,
            "parallelism_efficiency": parallelism_efficiency,
            "pipeline_metrics": self.pipeline_metrics,
            "results": self.results
        }


# ==================== Legacy Functions (for backwards compatibility) ====================

def download_files_parallel(urls: list, paths: list, max_workers: int = 50) -> list:
    """
    Download multiple files in parallel. Returns list of results.

    Optimized for high throughput with 50 concurrent downloads to maximize
    network utilization and minimize GPU idle time waiting for data.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(urls)
    download_start = time.time()

    def download_one(idx: int, url: str, path: str):
        file_start = time.time()
        try:
            download_file(url, path)
            file_size = os.path.getsize(path) if os.path.exists(path) else 0
            elapsed = time.time() - file_start
            speed_mbps = (file_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
            print(f"[Download {idx}] {file_size/1024/1024:.2f}MB in {elapsed:.2f}s ({speed_mbps:.2f} MB/s)")
            return {"index": idx, "success": True, "path": path, "size": file_size, "time": elapsed}
        except Exception as e:
            return {"index": idx, "success": False, "error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_one, i, u, p) for i, (u, p) in enumerate(zip(urls, paths))]
        for future in as_completed(futures):
            result = future.result()
            results[result["index"]] = result

    total_time = time.time() - download_start
    total_size = sum(r.get("size", 0) for r in results if r and r.get("success"))
    print(f"[Download Complete] {len(urls)} files, {total_size/1024/1024:.2f}MB total in {total_time:.2f}s")

    return results


def upload_files_parallel(files: list, urls: list, max_workers: int = 50) -> list:
    """
    Upload multiple files in parallel to their respective presigned URLs.
    Returns list of {"index": i, "success": bool, "size": int, "error": str?}

    Optimized for high throughput with 50 concurrent uploads to maximize
    network utilization and minimize total job time.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(files)
    upload_start = time.time()

    def upload_one(idx: int, file_path: str, url: str):
        file_start = time.time()
        try:
            result = upload_file(file_path, url)
            elapsed = time.time() - file_start
            file_size = result.get("size", 0)
            speed_mbps = (file_size / 1024 / 1024) / elapsed if elapsed > 0 else 0
            print(f"[Upload {idx}] {file_size/1024/1024:.2f}MB in {elapsed:.2f}s ({speed_mbps:.2f} MB/s)")
            return {"index": idx, "success": True, "size": file_size, "time": elapsed}
        except Exception as e:
            return {"index": idx, "success": False, "error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (f, u) in enumerate(zip(files, urls)):
            futures.append(executor.submit(upload_one, i, f, u))

        for future in as_completed(futures):
            result = future.result()
            results[result["index"]] = result

    total_time = time.time() - upload_start
    total_size = sum(r.get("size", 0) for r in results if r and r.get("success"))
    print(f"[Upload Complete] {len(files)} files, {total_size/1024/1024:.2f}MB total in {total_time:.2f}s")

    return results


def pre_distribute_work(items: list, gpu_count: int, gpu_memory_free_mb: list = None) -> list:
    """
    Pre-distribute work items across GPUs before processing starts.
    Uses memory-aware distribution if memory info is available.

    Args:
        items: List of (input_path, output_path, tool, config, file_index) tuples
        gpu_count: Number of available GPUs
        gpu_memory_free_mb: Optional list of free memory per GPU

    Returns:
        List of (input_path, output_path, tool, config, file_index, gpu_id) tuples
    """
    if gpu_count <= 1:
        # Single GPU: all items go to GPU 0
        return [(item[0], item[1], item[2], item[3], item[4], 0) for item in items]

    # Estimate file sizes for smarter distribution
    file_sizes = []
    for item in items:
        try:
            size = os.path.getsize(item[0]) if os.path.exists(item[0]) else 0
        except:
            size = 0
        file_sizes.append(size)

    # Sort items by size (descending) for better load balancing
    indexed_items = list(enumerate(items))
    indexed_items.sort(key=lambda x: file_sizes[x[0]], reverse=True)

    # Distribute using greedy algorithm (assign to GPU with least total size)
    gpu_loads = [0] * gpu_count
    assignments = [None] * len(items)

    for original_idx, item in indexed_items:
        # Find GPU with minimum load
        if gpu_memory_free_mb:
            # Weight by available memory
            scores = [gpu_loads[i] / max(gpu_memory_free_mb[i], 1) for i in range(gpu_count)]
            best_gpu = scores.index(min(scores))
        else:
            best_gpu = gpu_loads.index(min(gpu_loads))

        gpu_loads[best_gpu] += file_sizes[original_idx]
        assignments[original_idx] = (item[0], item[1], item[2], item[3], item[4], best_gpu)

    print(f"[Pre-distribute] Assigned {len(items)} items across {gpu_count} GPUs")
    print(f"[Pre-distribute] Load per GPU (bytes): {gpu_loads}")

    return assignments


def process_single_file_for_batch(args: tuple, gpu_tracker: GPULoadTracker = None) -> dict:
    """
    Worker function for batch processing.
    Processes a single file with the specified tool.
    Supports multi-GPU via gpu_id parameter.

    Args:
        args: Tuple of (input_path, output_path, tool, config, file_index, gpu_id)
              or (input_path, output_path, tool, config, file_index) for backwards compat
        gpu_tracker: Optional GPULoadTracker for dynamic load balancing

    Returns:
        dict with index, status, output_path/error, gpu_id, processing_time
    """
    # Support both old (5-tuple) and new (6-tuple) format with gpu_id
    if len(args) == 6:
        input_path, output_path, tool, config, file_index, gpu_id = args
    else:
        input_path, output_path, tool, config, file_index = args
        gpu_id = 0

    # NOTE: Do NOT set os.environ['CUDA_VISIBLE_DEVICES'] here!
    # In ThreadPoolExecutor, all threads share the same environment, so setting this
    # would affect all threads (race condition). Instead, each processor should use
    # GPU-specific flags like FFmpeg's -gpu X. For PyTorch, pass gpu_id in config.

    # Start timing for performance tracking
    start_time = time.time()

    try:
        input_ext = os.path.splitext(input_path)[1].lower()
        is_video_file = is_video(input_ext)
        input_size = os.path.getsize(input_path) if os.path.exists(input_path) else 0

        # Common config with GPU ID for all processors
        # Also add FFmpeg loglevel hint for GPU verification
        gpu_config = {
            **config,
            "_gpu_id": gpu_id,
            "_cuda_device": gpu_id,
            "_ffmpeg_loglevel": "info"  # Enable to see encoder initialization
        }

        print(f"[Batch Worker {file_index}] GPU {gpu_id} START: {os.path.basename(input_path)} ({input_size/1024/1024:.2f}MB)")

        # Process based on tool type
        if tool == "spoofer":
            if is_image(input_ext) and process_spoofer_fast is not None:
                # Use fast CPU mode for images
                result = process_spoofer_fast(input_path, output_path, gpu_config)
            elif process_spoofer is not None:
                result = process_spoofer(input_path, output_path, gpu_config)
            else:
                return {"index": file_index, "status": "failed", "error": "Spoofer not available", "gpu_id": gpu_id}

        elif tool == "captioner":
            if process_captioner is not None:
                # Pass file index for batch caption matching + GPU ID
                captioner_config = {**gpu_config, "imageIndex": file_index}
                result = process_captioner(input_path, output_path, captioner_config)
            else:
                return {"index": file_index, "status": "failed", "error": "Captioner not available", "gpu_id": gpu_id}

        elif tool == "vignettes":
            if process_vignettes is not None:
                result = process_vignettes(input_path, output_path, gpu_config)
            else:
                return {"index": file_index, "status": "failed", "error": "Vignettes not available", "gpu_id": gpu_id}

        elif tool == "resize":
            if process_resize is not None:
                result = process_resize(input_path, output_path, gpu_config)
            else:
                return {"index": file_index, "status": "failed", "error": "Resize not available", "gpu_id": gpu_id}

        elif tool == "pic_to_video":
            if process_pic_to_video is not None:
                result = process_pic_to_video(input_path, output_path, gpu_config)
            else:
                return {"index": file_index, "status": "failed", "error": "Pic to Video not available", "gpu_id": gpu_id}

        elif tool == "bg_remove":
            if process_bg_remove is not None:
                result = process_bg_remove(input_path, output_path, gpu_config)
            else:
                return {"index": file_index, "status": "failed", "error": "BG Remove not available", "gpu_id": gpu_id}

        elif tool == "upscale":
            if is_video_file and process_upscale_video is not None:
                result = process_upscale_video(input_path, output_path, gpu_config)
            elif process_upscale is not None:
                result = process_upscale(input_path, output_path, gpu_config)
            else:
                return {"index": file_index, "status": "failed", "error": "Upscale not available", "gpu_id": gpu_id}

        elif tool == "face_swap":
            if is_video_file and process_face_swap_video is not None:
                result = process_face_swap_video(input_path, output_path, gpu_config)
            elif process_face_swap is not None:
                result = process_face_swap(input_path, output_path, gpu_config)
            else:
                return {"index": file_index, "status": "failed", "error": "Face Swap not available", "gpu_id": gpu_id}

        elif tool == "video_reframe":
            if process_video_reframe is not None:
                result = process_video_reframe(input_path, output_path, gpu_config)
                # video_reframe may change output path for images
                if result and result.get('outputPath'):
                    output_path = result['outputPath']
            else:
                return {"index": file_index, "status": "failed", "error": "Video Reframe not available", "gpu_id": gpu_id}

        else:
            return {"index": file_index, "status": "failed", "error": f"Unknown tool: {tool}", "gpu_id": gpu_id}

        processing_time = time.time() - start_time

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            output_size = os.path.getsize(output_path)
            print(f"[Batch Worker {file_index}] GPU {gpu_id} DONE: {processing_time:.2f}s, output {output_size/1024/1024:.2f}MB")
            return {
                "index": file_index,
                "status": "completed",
                "output_path": output_path,
                "result": result,
                "gpu_id": gpu_id,
                "processing_time": processing_time,
                "input_size": input_size,
                "output_size": output_size
            }
        else:
            print(f"[Batch Worker {file_index}] GPU {gpu_id} FAILED: No output after {processing_time:.2f}s")
            return {"index": file_index, "status": "failed", "error": "Output file not created", "gpu_id": gpu_id, "processing_time": processing_time}

    except Exception as e:
        import traceback
        processing_time = time.time() - start_time
        print(f"[Batch Worker {file_index}] GPU {gpu_id} ERROR after {processing_time:.2f}s: {str(e)[:100]}")
        return {
            "index": file_index,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "gpu_id": gpu_id,
            "processing_time": processing_time
        }


def process_batch_videos(
    work_items: list,
    gpu_info: dict,
    job,
    results: list,
    progress_base: int = 30,
    progress_range: int = 50
) -> tuple:
    """
    Process video batch using ThreadPoolExecutor with proper GPU affinity
    and NVENC session limit handling.

    Optimized for FFmpeg-based processing where the subprocess does the heavy lifting.
    Uses dynamic GPU assignment based on current load.

    Args:
        work_items: List of (input_path, output_path, tool, config, file_index, gpu_id) tuples
        gpu_info: GPU information dict from get_gpu_info()
        job: RunPod job for progress updates
        results: Results list to update in place
        progress_base: Base progress percentage
        progress_range: Progress range for this phase

    Returns:
        (completed_count, failed_count)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    batch_start_time = time.time()

    gpu_count = gpu_info.get('gpu_count', 1)
    nvenc_sessions_per_gpu = gpu_info.get('nvenc_sessions_per_gpu', 3)
    max_parallel = gpu_info.get('nvenc_sessions', nvenc_sessions_per_gpu * gpu_count)

    # Create GPU load tracker for dynamic assignment
    gpu_tracker = GPULoadTracker(gpu_count, nvenc_sessions_per_gpu)

    completed = 0
    failed = 0
    total_items = len(work_items)

    # Calculate total input size for throughput metrics
    total_input_size = sum(os.path.getsize(item[0]) for item in work_items if os.path.exists(item[0]))

    print(f"[Batch Videos] ===== VIDEO PROCESSING START =====")
    print(f"[Batch Videos] Processing {total_items} videos ({total_input_size/1024/1024:.2f}MB total)")
    print(f"[Batch Videos] GPU count: {gpu_count}, NVENC sessions/GPU: {nvenc_sessions_per_gpu}, Max parallel: {max_parallel}")

    def process_with_tracking(item):
        """Wrapper that handles GPU tracking."""
        input_path, output_path, tool, config, file_index, assigned_gpu = item

        # Use dynamic GPU assignment if multiple GPUs
        if gpu_count > 1:
            gpu_id = gpu_tracker.get_best_gpu()
        else:
            gpu_id = assigned_gpu

        gpu_tracker.assign_job(gpu_id)
        try:
            # Update item with dynamically assigned GPU
            updated_item = (input_path, output_path, tool, config, file_index, gpu_id)
            result = process_single_file_for_batch(updated_item)
            return result
        finally:
            gpu_tracker.complete_job(gpu_id)

    # Use ThreadPoolExecutor - FFmpeg subprocess handles parallelism well
    # Limit workers to NVENC session count to avoid session exhaustion
    max_workers = min(max_parallel, total_items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for item in work_items:
            future = executor.submit(process_with_tracking, item)
            future_to_idx[future] = item[4]  # file_index

        for future in as_completed(future_to_idx):
            try:
                result = future.result(timeout=900)  # 15 min timeout for videos
            except Exception as e:
                idx = future_to_idx[future]
                result = {"index": idx, "status": "failed", "error": str(e), "gpu_id": -1}

            idx = result["index"]
            results[idx] = result

            if result["status"] == "completed":
                completed += 1
            else:
                failed += 1

            # Update progress
            progress = progress_base + int(((completed + failed) / total_items) * progress_range)
            runpod.serverless.progress_update(job, {
                "progress": progress,
                "status": f"Videos: {completed + failed}/{total_items} ({failed} failed)"
            })

    # Log final GPU distribution stats and performance metrics
    batch_elapsed = time.time() - batch_start_time
    stats = gpu_tracker.get_stats()

    # Calculate performance metrics
    total_processing_time = sum(
        results[i].get("processing_time", 0)
        for i in range(len(results))
        if results[i] and results[i].get("status") == "completed"
    )
    avg_time_per_video = total_processing_time / completed if completed > 0 else 0
    throughput_mbps = (total_input_size / 1024 / 1024) / batch_elapsed if batch_elapsed > 0 else 0
    parallelism_efficiency = (total_processing_time / batch_elapsed / max_parallel * 100) if batch_elapsed > 0 else 0

    print(f"[Batch Videos] ===== VIDEO PROCESSING COMPLETE =====")
    print(f"[Batch Videos] Total time: {batch_elapsed:.2f}s, Videos: {completed}/{total_items}")
    print(f"[Batch Videos] Throughput: {throughput_mbps:.2f} MB/s, Avg time/video: {avg_time_per_video:.2f}s")
    print(f"[Batch Videos] GPU distribution: {stats['total_jobs']}")
    print(f"[Batch Videos] Parallelism efficiency: {parallelism_efficiency:.1f}% (of {max_parallel} max sessions)")

    return completed, failed


def process_batch_images(
    work_items: list,
    gpu_info: dict,
    job,
    results: list,
    progress_base: int = 30,
    progress_range: int = 50
) -> tuple:
    """
    Process image batch using ThreadPoolExecutor.

    Note: Changed from ProcessPoolExecutor because:
    - PIL operations are not that CPU-heavy
    - Process spawning overhead outweighs parallelism benefits for images
    - ThreadPoolExecutor has lower overhead for I/O-bound image operations

    Args:
        work_items: List of (input_path, output_path, tool, config, file_index, gpu_id) tuples
        gpu_info: GPU information dict from get_gpu_info()
        job: RunPod job for progress updates
        results: Results list to update in place
        progress_base: Base progress percentage
        progress_range: Progress range for this phase

    Returns:
        (completed_count, failed_count)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    gpu_count = gpu_info.get('gpu_count', 1)
    # For images, we can have more parallelism (no NVENC limit)
    # But limit based on CPU cores
    import multiprocessing
    cpu_cores = multiprocessing.cpu_count()
    max_parallel = min(cpu_cores * 2, len(work_items), 32)  # Cap at 32

    completed = 0
    failed = 0
    total_items = len(work_items)

    print(f"[Batch Images] Processing {total_items} images with max {max_parallel} parallel workers")

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_idx = {executor.submit(process_single_file_for_batch, item): item[4] for item in work_items}

        for future in as_completed(future_to_idx):
            try:
                result = future.result(timeout=300)  # 5 min timeout for images
            except Exception as e:
                idx = future_to_idx[future]
                result = {"index": idx, "status": "failed", "error": str(e), "gpu_id": 0}

            idx = result["index"]
            results[idx] = result

            if result["status"] == "completed":
                completed += 1
            else:
                failed += 1

            # Update progress
            progress = progress_base + int(((completed + failed) / total_items) * progress_range)
            runpod.serverless.progress_update(job, {
                "progress": progress,
                "status": f"Images: {completed + failed}/{total_items} ({failed} failed)"
            })

    return completed, failed


def process_batch_mode(job, job_input: dict) -> dict:
    """
    Process multiple files in parallel using async I/O pipeline.
    Optimized for maximum GPU utilization with overlapped downloads, processing, and uploads.

    Input format:
    {
        "tool": "batch",
        "processor": "spoofer" | "captioner" | "vignettes" | "resize" | "pic_to_video" | etc,
        "inputUrls": ["url1", "url2", ...],
        "outputUrls": ["url1", "url2", ...],
        "config": { tool-specific configuration }
    }

    Returns:
    {
        "status": "completed",
        "mode": "async_pipeline",
        "total": N,
        "completed": X,
        "failed": Y,
        "results": [...],
        "gpu_distribution": [jobs_gpu0, jobs_gpu1, ...]
    }
    """
    job_start_time = time.time()

    processor = job_input.get("processor", "").lower()
    input_urls = job_input.get("inputUrls", [])
    output_urls = job_input.get("outputUrls", [])
    config = job_input.get("batchConfig", job_input.get("config", {}))
    job_id = job["id"]

    # Validate
    if not processor:
        return {"error": "Missing 'processor' parameter for batch mode"}
    if not input_urls or not output_urls:
        return {"error": "Missing 'inputUrls' or 'outputUrls' for batch mode"}
    if len(input_urls) != len(output_urls):
        return {"error": "inputUrls and outputUrls must have same length"}

    total_files = len(input_urls)

    # Detect GPU capabilities (cached)
    gpu_info = get_gpu_info()
    gpu_count = gpu_info.get('gpu_count', 1)
    max_parallel = gpu_info['nvenc_sessions']

    print(f"[Batch Mode] ===== JOB START: {job_id} =====")
    print(f"[Batch Mode] GPU: {gpu_info['gpu_name']}, Type: {gpu_info['gpu_type']}")
    print(f"[Batch Mode] GPUs: {gpu_count}, Max parallel: {max_parallel}")
    print(f"[Batch Mode] Processing {total_files} files with processor: {processor}")
    print(f"[Batch Mode] Config received: {json.dumps({k: v for k, v in config.items() if not str(k).startswith('_')}, default=str)}")

    runpod.serverless.progress_update(job, {
        "progress": 5,
        "status": f"Batch mode: {total_files} files, {gpu_count} GPU(s), {max_parallel} sessions"
    })

    temp_dir = tempfile.mkdtemp(prefix=f"farmium_batch_{job_id}_")

    # Get optimal I/O workers based on GPU count
    download_workers, upload_workers = get_optimal_io_workers(gpu_count)
    print(f"[Batch Mode] I/O workers: {download_workers} download, {upload_workers} upload")

    try:
        # Use the new async pipeline processor
        pipeline = AsyncPipelineProcessor(
            gpu_info=gpu_info,
            job=job,
            temp_dir=temp_dir,
            download_workers=download_workers,
            upload_workers=upload_workers
        )

        result = pipeline.process_batch(
            input_urls=input_urls,
            output_urls=output_urls,
            tool=processor,
            config=config
        )

        # Add job timing
        result['job_time_seconds'] = time.time() - job_start_time

        return result

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "mode": "async_pipeline"
        }

    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# ==================== Pipeline Processor ====================

def process_pipeline(job, temp_dir: str, input_path: str, output_url: str, pipeline: list, progress_callback, output_urls: list = None) -> dict:
    """
    Process a pipeline of tools sequentially without intermediate uploads.

    Each tool's output becomes the next tool's input.

    If output_urls is provided (array of presigned URLs), uploads each file directly
    to its corresponding URL - much faster than ZIP + re-upload flow.

    If output_url is provided (single URL), creates ZIP and uploads (legacy mode).

    Args:
        job: RunPod job object for progress updates
        temp_dir: Temporary directory for processing
        input_path: Path to downloaded input file
        output_url: Presigned URL for final ZIP upload (legacy mode)
        pipeline: List of {"tool": "...", "config": {...}} dicts
        progress_callback: Callback for progress updates
        output_urls: Array of presigned URLs for direct individual uploads (optional)

    Returns:
        dict with status and results
    """
    from glob import glob

    if not pipeline or len(pipeline) == 0:
        return {"error": "Pipeline is empty"}

    # Track current files being processed
    # Start with the single input file
    current_files = [input_path]
    total_steps = len(pipeline)

    for step_idx, step in enumerate(pipeline):
        tool = step.get("tool", "").lower()
        config = step.get("config", {})

        step_progress_base = int((step_idx / total_steps) * 80) + 10

        runpod.serverless.progress_update(job, {
            "progress": step_progress_base,
            "status": f"Step {step_idx + 1}/{total_steps}: {tool}"
        })

        # Create output directory for this step
        step_output_dir = os.path.join(temp_dir, f"step_{step_idx}_{tool}")
        os.makedirs(step_output_dir, exist_ok=True)

        next_files = []

        # Process each current file through this tool
        for file_idx, current_file in enumerate(current_files):
            input_ext = os.path.splitext(current_file)[1].lower()

            # Determine output extension based on tool
            if tool == "bg_remove":
                output_ext = ".png"
            elif tool in ["pic_to_video", "video_gen"]:
                output_ext = ".mp4"
            else:
                output_ext = input_ext if input_ext else ".jpg"

            # For spoofer with variations > 1, output is a directory of files
            variations = config.get("variations") or config.get("copies") or config.get("options", {}).get("variations") or config.get("options", {}).get("copies", 1)
            is_batch_spoofer = tool == "spoofer" and variations > 1

            if is_batch_spoofer:
                # Spoofer batch mode: output to a subdirectory
                spoofer_output_dir = os.path.join(step_output_dir, f"batch_{file_idx}")
                os.makedirs(spoofer_output_dir, exist_ok=True)
                output_path = spoofer_output_dir  # Pass directory, not file

                # Modify config to output individual files, not ZIP
                batch_config = {**config, "outputMode": "directory", "outputDir": spoofer_output_dir}
            else:
                output_path = os.path.join(step_output_dir, f"output_{file_idx}{output_ext}")
                batch_config = config

            # Create step-specific progress callback
            def step_callback(progress: float, message: str = None):
                file_progress = (file_idx + progress) / len(current_files)
                overall = step_progress_base + int(file_progress * (80 / total_steps))
                update = {"progress": min(overall, 90)}
                if message:
                    update["status"] = f"Step {step_idx + 1}: {tool} - {message}"
                runpod.serverless.progress_update(job, update)

            # Process based on tool type
            try:
                if tool == "spoofer":
                    input_is_image = is_image(input_ext)
                    if input_is_image and process_spoofer_fast is not None:
                        result = process_spoofer_fast(current_file, output_path, batch_config, progress_callback=step_callback)
                    elif process_spoofer is not None:
                        result = process_spoofer(current_file, output_path, batch_config, progress_callback=step_callback)
                    else:
                        return {"error": "Spoofer processor not available"}

                    # Collect output files
                    if is_batch_spoofer:
                        # Collect all files from the batch output directory (images and videos)
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.mp4', '*.mov', '*.avi', '*.webm']:
                            next_files.extend(glob(os.path.join(spoofer_output_dir, ext)))
                    else:
                        if os.path.exists(output_path):
                            next_files.append(output_path)

                elif tool == "captioner":
                    if process_captioner is None:
                        return {"error": "Captioner processor not available"}

                    # For captioner, pass the image index for batch caption matching
                    captioner_config = {**batch_config, "imageIndex": file_idx}
                    result = process_captioner(current_file, output_path, captioner_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "resize":
                    if process_resize is None:
                        return {"error": "Resize processor not available"}
                    result = process_resize(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "vignettes":
                    if process_vignettes is None:
                        return {"error": "Vignettes processor not available"}
                    result = process_vignettes(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "bg_remove":
                    if process_bg_remove is None:
                        return {"error": "BG Remove processor not available"}
                    result = process_bg_remove(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "upscale":
                    if process_upscale is None:
                        return {"error": "Upscale processor not available"}
                    if is_video(input_ext) and process_upscale_video:
                        result = process_upscale_video(current_file, output_path, batch_config, progress_callback=step_callback)
                    else:
                        result = process_upscale(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "face_swap":
                    if process_face_swap is None:
                        return {"error": "Face Swap processor not available"}
                    if is_video(input_ext) and process_face_swap_video:
                        result = process_face_swap_video(current_file, output_path, batch_config, progress_callback=step_callback)
                    else:
                        result = process_face_swap(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "pic_to_video":
                    if process_pic_to_video is None:
                        return {"error": "Pic to Video processor not available"}
                    result = process_pic_to_video(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "video_gen":
                    if process_video_gen is None:
                        return {"error": "Video Gen processor not available"}
                    result = process_video_gen(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "video_reframe":
                    if process_video_reframe is None:
                        return {"error": "Video Reframe processor not available"}
                    result = process_video_reframe(current_file, output_path, batch_config, progress_callback=step_callback)
                    # video_reframe may change extension for images (mp4 -> jpg)
                    actual_path = result.get('outputPath', output_path) if result else output_path
                    if os.path.exists(actual_path):
                        next_files.append(actual_path)

                else:
                    return {"error": f"Unknown tool in pipeline: {tool}"}

            except Exception as e:
                import traceback
                return {
                    "error": f"Pipeline step {step_idx + 1} ({tool}) failed: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "step": step_idx,
                    "tool": tool
                }

        # Update current files for next step
        if not next_files:
            return {"error": f"Pipeline step {step_idx + 1} ({tool}) produced no output files"}

        current_files = next_files

    # ==================== DIRECT UPLOAD MODE ====================
    # If output_urls array is provided, upload each file directly (FAST)
    if output_urls and len(output_urls) >= len(current_files):
        runpod.serverless.progress_update(job, {
            "progress": 92,
            "status": "Uploading files directly"
        })

        # Upload files in parallel (50 concurrent uploads for maximum throughput)
        upload_results = upload_files_parallel(
            current_files[:len(output_urls)],
            output_urls[:len(current_files)],
            max_workers=50
        )

        # Check for failures
        failed = [r for r in upload_results if not r.get("success")]
        if failed:
            return {
                "error": f"Failed to upload {len(failed)} files: {failed[0].get('error', 'unknown')}",
                "mode": "pipeline_direct",
                "steps": len(pipeline),
                "outputFiles": len(current_files),
                "uploadedFiles": len(current_files) - len(failed),
                "failedUploads": len(failed)
            }

        total_size = sum(r.get("size", 0) for r in upload_results)

        runpod.serverless.progress_update(job, {
            "progress": 100,
            "status": "Completed"
        })

        return {
            "status": "completed",
            "mode": "pipeline_direct",
            "steps": len(pipeline),
            "outputFiles": len(current_files),
            "uploadedFiles": len(current_files),
            "totalSize": total_size
        }

    # ==================== LEGACY ZIP MODE ====================
    # Fallback: Create ZIP and upload (for backwards compatibility)
    runpod.serverless.progress_update(job, {
        "progress": 92,
        "status": "Creating ZIP"
    })

    final_zip_path = os.path.join(temp_dir, "pipeline_output.zip")
    with zipfile.ZipFile(final_zip_path, 'w') as zf:
        for idx, file_path in enumerate(current_files):
            ext = os.path.splitext(file_path)[1]
            arcname = f"output_{idx + 1:04d}{ext}"
            # Use smart compression: ZIP_STORED for already-compressed files
            compression = get_zip_compression(file_path)
            zf.write(file_path, arcname, compress_type=compression)

    # Upload final ZIP with error handling
    runpod.serverless.progress_update(job, {
        "progress": 95,
        "status": "Uploading"
    })

    try:
        upload_result = upload_file(final_zip_path, output_url)
    except Exception as e:
        return {
            "error": f"Failed to upload pipeline output: {str(e)}",
            "mode": "pipeline",
            "steps": len(pipeline),
            "outputFiles": len(current_files)
        }

    return {
        "status": "completed",
        "mode": "pipeline",
        "steps": len(pipeline),
        "outputFiles": len(current_files),
        "outputSize": os.path.getsize(final_zip_path),
        "uploadAttempts": upload_result.get("attempts", 1)
    }


# ==================== Batch Pipeline Processor ====================

def process_single_pipeline_for_batch(args: tuple) -> dict:
    """
    Worker function for batch pipeline processing.
    Processes a single file through an entire pipeline on its assigned GPU.

    Args:
        args: Tuple of (input_path, output_dir, pipeline, config, file_index, gpu_id)

    Returns:
        dict with index, status, output_files/error, gpu_id
    """
    from glob import glob

    input_path, output_dir, pipeline, config, file_index, gpu_id = args

    # NOTE: Do NOT set os.environ['CUDA_VISIBLE_DEVICES'] - see comment in process_single_file_for_batch

    try:
        print(f"[Pipeline Worker {file_index}] Processing on GPU {gpu_id}: {os.path.basename(input_path)}")

        # Track current files through pipeline
        current_files = [input_path]

        for step_idx, step in enumerate(pipeline):
            tool = step.get("tool", "").lower()
            step_config = {**step.get("config", {}), "_gpu_id": gpu_id, "_cuda_device": gpu_id}

            # Create step output directory
            step_output_dir = os.path.join(output_dir, f"step_{step_idx}_{tool}")
            os.makedirs(step_output_dir, exist_ok=True)

            next_files = []

            for sub_idx, current_file in enumerate(current_files):
                input_ext = os.path.splitext(current_file)[1].lower()

                # Determine output extension
                if tool == "bg_remove":
                    output_ext = ".png"
                elif tool in ["pic_to_video", "video_gen"]:
                    output_ext = ".mp4"
                else:
                    output_ext = input_ext if input_ext else ".jpg"

                output_path = os.path.join(step_output_dir, f"output_{sub_idx}{output_ext}")

                # Process based on tool type
                if tool == "spoofer":
                    if is_image(input_ext) and process_spoofer_fast is not None:
                        process_spoofer_fast(current_file, output_path, step_config)
                    elif process_spoofer is not None:
                        process_spoofer(current_file, output_path, step_config)
                elif tool == "captioner" and process_captioner is not None:
                    process_captioner(current_file, output_path, {**step_config, "imageIndex": file_index})
                elif tool == "vignettes" and process_vignettes is not None:
                    process_vignettes(current_file, output_path, step_config)
                elif tool == "resize" and process_resize is not None:
                    process_resize(current_file, output_path, step_config)
                elif tool == "bg_remove" and process_bg_remove is not None:
                    process_bg_remove(current_file, output_path, step_config)
                elif tool == "upscale":
                    if is_video(input_ext) and process_upscale_video is not None:
                        process_upscale_video(current_file, output_path, step_config)
                    elif process_upscale is not None:
                        process_upscale(current_file, output_path, step_config)
                elif tool == "face_swap":
                    if is_video(input_ext) and process_face_swap_video is not None:
                        process_face_swap_video(current_file, output_path, step_config)
                    elif process_face_swap is not None:
                        process_face_swap(current_file, output_path, step_config)
                elif tool == "pic_to_video" and process_pic_to_video is not None:
                    process_pic_to_video(current_file, output_path, step_config)
                elif tool == "video_gen" and process_video_gen is not None:
                    process_video_gen(current_file, output_path, step_config)
                elif tool == "video_reframe" and process_video_reframe is not None:
                    reframe_result = process_video_reframe(current_file, output_path, step_config)
                    # video_reframe may change extension for images (mp4 -> jpg)
                    if reframe_result and reframe_result.get('outputPath'):
                        output_path = reframe_result['outputPath']

                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    next_files.append(output_path)

            if not next_files:
                return {
                    "index": file_index,
                    "status": "failed",
                    "error": f"Pipeline step {step_idx + 1} ({tool}) produced no output",
                    "gpu_id": gpu_id
                }

            current_files = next_files

        return {
            "index": file_index,
            "status": "completed",
            "output_files": current_files,
            "gpu_id": gpu_id
        }

    except Exception as e:
        import traceback
        return {
            "index": file_index,
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc(),
            "gpu_id": gpu_id
        }


def process_batch_pipeline(job, job_input: dict) -> dict:
    """
    Process multiple files through a pipeline in parallel.
    Each file goes through the full pipeline on its assigned GPU.

    Input format:
    {
        "tool": "batch_pipeline",
        "inputUrls": ["url1", "url2", ...],
        "outputUrls": ["url1", "url2", ...],  # One per input file
        "pipeline": [{"tool": "...", "config": {...}}, ...],
        "config": { optional global configuration }
    }

    Returns:
    {
        "status": "completed",
        "mode": "batch_pipeline",
        "total": N,
        "completed": X,
        "failed": Y,
        "gpu_distribution": [jobs_gpu0, jobs_gpu1, ...],
        "results": [...]
    }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    input_urls = job_input.get("inputUrls", [])
    output_urls = job_input.get("outputUrls", [])
    pipeline = job_input.get("pipeline", [])
    config = job_input.get("config", {})
    job_id = job["id"]

    # Validate
    if not input_urls or not output_urls:
        return {"error": "Missing 'inputUrls' or 'outputUrls' for batch pipeline"}
    if len(input_urls) != len(output_urls):
        return {"error": "inputUrls and outputUrls must have same length"}
    if not pipeline:
        return {"error": "Missing 'pipeline' parameter"}

    total_files = len(input_urls)

    # Detect GPU capabilities (cached)
    gpu_info = get_gpu_info()
    gpu_count = gpu_info.get('gpu_count', 1)
    max_parallel = gpu_info.get('nvenc_sessions', gpu_count * 3)
    gpu_memory_free = gpu_info.get('gpu_memory_free_mb', [])

    print(f"[Batch Pipeline] GPU: {gpu_info['gpu_name']}, GPUs: {gpu_count}")
    print(f"[Batch Pipeline] Processing {total_files} files through {len(pipeline)} pipeline steps")

    runpod.serverless.progress_update(job, {
        "progress": 5,
        "status": f"Batch pipeline: {total_files} files, {len(pipeline)} steps, {gpu_count} GPU(s)"
    })

    temp_dir = tempfile.mkdtemp(prefix=f"farmium_batch_pipeline_{job_id}_")

    try:
        # 1. Download all input files in parallel
        runpod.serverless.progress_update(job, {
            "progress": 10,
            "status": f"Downloading {total_files} files..."
        })

        input_paths = []
        for i, url in enumerate(input_urls):
            ext = get_file_extension(url, config)
            input_paths.append(os.path.join(temp_dir, f"input_{i}{ext}"))

        # Download with high parallelism (50 concurrent) to minimize GPU idle time
        download_results = download_files_parallel(input_urls, input_paths, max_workers=50)

        # 2. Prepare work items
        results = [None] * total_files
        work_items = []
        failed = 0

        for i in range(total_files):
            if not download_results[i].get("success"):
                results[i] = {
                    "index": i,
                    "status": "failed",
                    "error": f"Download failed: {download_results[i].get('error')}"
                }
                failed += 1
            else:
                # Each file gets its own output directory
                file_output_dir = os.path.join(temp_dir, f"file_{i}")
                os.makedirs(file_output_dir, exist_ok=True)
                work_items.append((input_paths[i], file_output_dir, pipeline, config, i))

        # 3. Pre-distribute work across GPUs
        if work_items:
            work_items = pre_distribute_work(work_items, gpu_count, gpu_memory_free)

        # 4. Process pipelines in parallel
        runpod.serverless.progress_update(job, {
            "progress": 25,
            "status": f"Processing {len(work_items)} pipelines in parallel..."
        })

        completed = 0
        # Create GPU tracker for dynamic assignment
        gpu_tracker = GPULoadTracker(gpu_count, max_parallel // gpu_count)

        def process_with_tracking(item):
            """Wrapper for GPU tracking."""
            input_path, output_dir, pl, cfg, file_index, assigned_gpu = item
            if gpu_count > 1:
                gpu_id = gpu_tracker.get_best_gpu()
            else:
                gpu_id = assigned_gpu
            gpu_tracker.assign_job(gpu_id)
            try:
                updated_item = (input_path, output_dir, pl, cfg, file_index, gpu_id)
                return process_single_pipeline_for_batch(updated_item)
            finally:
                gpu_tracker.complete_job(gpu_id)

        # Use ThreadPoolExecutor - pipelines often involve FFmpeg subprocesses
        max_workers = min(max_parallel, len(work_items))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for item in work_items:
                future = executor.submit(process_with_tracking, item)
                future_to_idx[future] = item[4]  # file_index

            for future in as_completed(future_to_idx):
                try:
                    result = future.result(timeout=1800)  # 30 min timeout per pipeline
                except Exception as e:
                    idx = future_to_idx[future]
                    result = {"index": idx, "status": "failed", "error": str(e), "gpu_id": -1}

                idx = result["index"]
                results[idx] = result

                if result["status"] == "completed":
                    completed += 1
                else:
                    failed += 1

                # Update progress
                progress = 25 + int(((completed + failed) / total_files) * 55)
                runpod.serverless.progress_update(job, {
                    "progress": progress,
                    "status": f"Pipelines: {completed + failed}/{total_files} ({failed} failed)"
                })

        # 5. Upload output files
        runpod.serverless.progress_update(job, {
            "progress": 85,
            "status": f"Uploading {completed} processed files..."
        })

        files_to_upload = []
        urls_to_upload = []
        upload_indices = []

        for i, result in enumerate(results):
            if result and result.get("status") == "completed" and result.get("output_files"):
                # Upload the first/primary output file from the pipeline
                output_files = result["output_files"]
                if output_files and os.path.exists(output_files[0]):
                    files_to_upload.append(output_files[0])
                    urls_to_upload.append(output_urls[i])
                    upload_indices.append(i)

        if files_to_upload:
            # Upload with high parallelism (50 concurrent) to minimize total job time
            upload_results = upload_files_parallel(files_to_upload, urls_to_upload, max_workers=50)
            for upload_result in upload_results:
                if not upload_result.get("success"):
                    idx = upload_indices[upload_result["index"]]
                    results[idx]["status"] = "upload_failed"
                    results[idx]["upload_error"] = upload_result.get("error")
                    failed += 1
                    completed -= 1

        # 6. Calculate GPU distribution
        gpu_distribution = [0] * gpu_count
        for result in results:
            if result and result.get("status") == "completed":
                gpu_id = result.get("gpu_id", 0)
                if 0 <= gpu_id < gpu_count:
                    gpu_distribution[gpu_id] += 1

        runpod.serverless.progress_update(job, {
            "progress": 100,
            "status": "Completed"
        })

        return {
            "status": "completed",
            "mode": "batch_pipeline",
            "gpu": gpu_info['gpu_name'],
            "gpu_count": gpu_count,
            "pipeline_steps": len(pipeline),
            "total": total_files,
            "completed": completed,
            "failed": failed,
            "gpu_distribution": gpu_distribution,
            "results": results
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "mode": "batch_pipeline"
        }

    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# ==================== Main Handler ====================

def handler(job):
    """
    Main handler for GPU processing jobs

    Input format:
    {
        "tool": "vignettes" | "spoofer" | "captioner" | "resize" | "pic_to_video" |
                "bg_remove" | "upscale" | "face_swap" | "video_gen" | "batch" | "batch_pipeline",
        "inputUrl": "presigned download URL",
        "outputUrl": "presigned upload URL",
        "config": { tool-specific configuration }
    }

    For batch mode:
    {
        "tool": "batch",
        "processor": "spoofer" | "captioner" | etc,
        "inputUrls": ["url1", "url2", ...],
        "outputUrls": ["url1", "url2", ...],
        "config": { tool-specific configuration }
    }

    For batch_pipeline mode:
    {
        "tool": "batch_pipeline",
        "inputUrls": ["url1", "url2", ...],
        "outputUrls": ["url1", "url2", ...],
        "pipeline": [{"tool": "...", "config": {...}}, ...],
        "config": { optional global configuration }
    }
    """
    job_input = job["input"]
    job_id = job["id"]

    tool = job_input.get("tool", "").lower()
    input_url = job_input.get("inputUrl")
    output_url = job_input.get("outputUrl")
    config = job_input.get("config", {})


    # Validate inputs
    if not tool:
        return {"error": "Missing 'tool' parameter"}

    # ==================== BATCH MODE ====================
    # Process multiple files in parallel using multiple NVENC sessions
    # Optimized for datacenter GPUs (A5000, A6000) with unlimited NVENC sessions
    if tool == "batch":
        return process_batch_mode(job, job_input)

    # ==================== BATCH PIPELINE MODE ====================
    # Process multiple files through a full pipeline in parallel
    # Each file is processed through all pipeline steps on its assigned GPU
    if tool == "batch_pipeline":
        return process_batch_pipeline(job, job_input)

    if not input_url:
        return {"error": "Missing 'inputUrl' parameter"}
    if not output_url:
        return {"error": "Missing 'outputUrl' parameter"}

    # ==================== PIPELINE MODE ====================
    # Process entire workflow in one job without intermediate uploads
    if tool == "pipeline":
        pipeline = job_input.get("pipeline", [])
        if not pipeline:
            return {"error": "Missing 'pipeline' parameter for pipeline mode"}

        # Direct upload mode: array of presigned URLs for individual files (FAST)
        output_urls = job_input.get("outputUrls", [])

        temp_dir = tempfile.mkdtemp(prefix=f"farmium_{job_id}_")
        try:
            # Download input
            runpod.serverless.progress_update(job, {
                "progress": 5,
                "status": "downloading"
            })
            input_ext = get_file_extension(input_url, config)
            input_path = os.path.join(temp_dir, f"input{input_ext}")
            download_file(input_url, input_path)

            # Process pipeline
            progress_callback = create_progress_callback(job)
            result = process_pipeline(
                job, temp_dir, input_path, output_url, pipeline, progress_callback,
                output_urls=output_urls if output_urls else None
            )

            return result

        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "tool": "pipeline"
            }
        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    # Create temp directory for this job
    temp_dir = tempfile.mkdtemp(prefix=f"farmium_{job_id}_")

    try:
        # Determine file extensions
        input_ext = get_file_extension(input_url, config)
        output_ext = config.get("outputExtension", input_ext)

        # Tool-specific output extension handling
        if tool == "spoofer":
            variations = config.get("variations") or config.get("copies") or config.get("options", {}).get("variations") or config.get("options", {}).get("copies", 1)
            if variations > 1:
                output_ext = ".zip"
        elif tool == "bg_remove":
            output_ext = ".png"  # Always PNG for transparency
        elif tool == "pic_to_video":
            output_ext = ".mp4"
        elif tool == "video_gen":
            output_ext = ".mp4"

        input_path = os.path.join(temp_dir, f"input{input_ext}")
        output_path = os.path.join(temp_dir, f"output{output_ext}")

        # Download input file
        runpod.serverless.progress_update(job, {
            "progress": 5,
            "status": "downloading"
        })
        download_file(input_url, input_path)

        # Process based on tool type
        runpod.serverless.progress_update(job, {
            "progress": 10,
            "status": "processing"
        })

        progress_callback = create_progress_callback(job)
        result = {}

        # ==================== VIGNETTES ====================
        if tool == "vignettes":
            if process_vignettes is None:
                return {"error": "Vignettes processor not available"}
            result = process_vignettes(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== SPOOFER ====================
        elif tool == "spoofer":
            input_is_image = is_image(input_ext)

            try:
                if input_is_image and process_spoofer_fast is not None:
                    runpod.serverless.progress_update(job, {
                        "progress": 12,
                        "status": "processing (CPU fast mode)"
                    })
                    result = process_spoofer_fast(
                        input_path, output_path, config,
                        progress_callback=progress_callback
                    )
                elif process_spoofer is not None:
                    runpod.serverless.progress_update(job, {
                        "progress": 12,
                        "status": "processing (GPU mode)"
                    })
                    result = process_spoofer(
                        input_path, output_path, config,
                        progress_callback=progress_callback
                    )
                else:
                    return {"error": "Spoofer processor not available"}
            except Exception as e:
                import traceback
                return {
                    "error": f"Spoofer processing error: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "tool": tool,
                    "mode": "cpu_fast" if input_is_image else "gpu"
                }

        # ==================== CAPTIONER ====================
        elif tool == "captioner":
            if process_captioner is None:
                return {"error": "Captioner processor not available"}
            result = process_captioner(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== RESIZE ====================
        elif tool == "resize":
            if process_resize is None:
                return {"error": "Resize processor not available"}
            result = process_resize(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== PIC TO VIDEO ====================
        elif tool == "pic_to_video":
            if process_pic_to_video is None:
                return {"error": "Pic to Video processor not available"}
            output_path = os.path.join(temp_dir, "output.mp4")
            result = process_pic_to_video(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== BG REMOVE ====================
        elif tool == "bg_remove":
            if process_bg_remove is None:
                return {"error": "BG Remove processor not available. Install: pip install rembg"}
            output_path = os.path.join(temp_dir, "output.png")
            result = process_bg_remove(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== UPSCALE ====================
        elif tool == "upscale":
            if process_upscale is None:
                return {"error": "Upscale processor not available. Install: pip install realesrgan basicsr"}

            if is_video(input_ext):
                if process_upscale_video is None:
                    return {"error": "Video upscale not available"}
                result = process_upscale_video(
                    input_path, output_path, config,
                    progress_callback=progress_callback
                )
            else:
                result = process_upscale(
                    input_path, output_path, config,
                    progress_callback=progress_callback
                )

        # ==================== FACE SWAP ====================
        elif tool == "face_swap":
            if process_face_swap is None:
                return {"error": "Face Swap processor not available. Install: pip install insightface onnxruntime-gpu"}

            if is_video(input_ext):
                if process_face_swap_video is None:
                    return {"error": "Video face swap not available"}
                result = process_face_swap_video(
                    input_path, output_path, config,
                    progress_callback=progress_callback
                )
            else:
                result = process_face_swap(
                    input_path, output_path, config,
                    progress_callback=progress_callback
                )

        # ==================== VIDEO GEN ====================
        elif tool == "video_gen":
            if process_video_gen is None:
                return {"error": "Video Gen processor not available. Install: pip install replicate diffusers"}
            output_path = os.path.join(temp_dir, "output.mp4")
            result = process_video_gen(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== VIDEO REFRAME ====================
        elif tool == "video_reframe":
            if process_video_reframe is None:
                return {"error": "Video Reframe processor not available"}
            print(f"[VIDEO_REFRAME] Single mode - Config received: {json.dumps({k: v for k, v in config.items() if not str(k).startswith('_')}, default=str)}")
            output_path = os.path.join(temp_dir, "output.mp4")
            print(f"[VIDEO_REFRAME] Initial output_path: {output_path}")
            result = process_video_reframe(
                input_path, output_path, config,
                progress_callback=progress_callback
            )
            print(f"[VIDEO_REFRAME] Processor result: {result}")
            # IMPORTANT: video_reframe may change extension for images (mp4 -> jpg)
            # Use actual output path from result if available
            if result and result.get('outputPath'):
                old_path = output_path
                output_path = result['outputPath']
                print(f"[VIDEO_REFRAME] Output path updated: {old_path} -> {output_path}")
            else:
                print(f"[VIDEO_REFRAME] WARNING: No outputPath in result, using original: {output_path}")

        # ==================== UNKNOWN TOOL ====================
        else:
            return {"error": f"Unknown tool: {tool}"}

        # Check if output was created
        if not os.path.exists(output_path):
            return {"error": "Processing failed - no output file created"}

        output_size = os.path.getsize(output_path)
        if output_size == 0:
            return {"error": "Processing failed - output file is empty"}

        # Upload result with error handling
        runpod.serverless.progress_update(job, {
            "progress": 95,
            "status": "uploading"
        })

        try:
            upload_result = upload_file(output_path, output_url)
        except Exception as e:
            return {
                "error": f"Failed to upload output: {str(e)}",
                "tool": tool,
                "outputSize": output_size
            }

        # Success
        return {
            "status": "completed",
            "tool": tool,
            "outputSize": output_size,
            "uploadAttempts": upload_result.get("attempts", 1),
            **result
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": tool
        }

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# Start the serverless handler
runpod.serverless.start({"handler": handler})

# Force rebuild 1768419629
