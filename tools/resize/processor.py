"""
Resize Processor - Resize and crop images/videos to target resolution

Converts media to target resolution (default 1080x1920) using:
- Center crop to match aspect ratio
- Scale to exact dimensions
- GPU acceleration for videos (NVENC)
- Multi-GPU support for parallel batch processing

Config structure:
{
    "width": 1080,        # Target width (default 1080)
    "height": 1920,       # Target height (default 1920)
    "skipIfCorrect": true,# Skip if already at target resolution
    "_gpu_id": 0          # Optional: specific GPU to use (0-indexed)
}
"""

import os
import logging
import subprocess
import tempfile
import zipfile
import shutil
from typing import Callable, Optional, Dict, Any, Tuple, List
from PIL import Image
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


def safe_extract(zf: zipfile.ZipFile, name: str, extract_dir: str) -> str:
    """Safely extract a zip entry, preventing path traversal attacks (ZipSlip)."""
    target_path = os.path.realpath(os.path.join(extract_dir, name))
    extract_dir_real = os.path.realpath(extract_dir)
    if not target_path.startswith(extract_dir_real + os.sep) and target_path != extract_dir_real:
        raise ValueError(f"Attempted path traversal in zip entry: {name}")
    zf.extract(name, extract_dir)
    return target_path


@dataclass
class GPUInfo:
    """Information about available GPUs."""
    count: int
    names: List[str]
    memory_mb: List[int]

    def __repr__(self):
        return f"GPUInfo(count={self.count}, names={self.names})"


def get_gpu_info() -> GPUInfo:
    """
    Detect available NVIDIA GPUs using nvidia-smi.

    Returns:
        GPUInfo with count, names, and memory for each GPU
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return GPUInfo(count=0, names=[], memory_mb=[])

        names = []
        memory_mb = []

        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    names.append(parts[1])
                    try:
                        memory_mb.append(int(float(parts[2])))
                    except ValueError:
                        memory_mb.append(0)

        return GPUInfo(count=len(names), names=names, memory_mb=memory_mb)

    except FileNotFoundError:
        # nvidia-smi not found - no NVIDIA GPU
        return GPUInfo(count=0, names=[], memory_mb=[])
    except subprocess.TimeoutExpired:
        return GPUInfo(count=0, names=[], memory_mb=[])
    except Exception as e:
        print(f"[Resize] GPU detection error: {e}")
        return GPUInfo(count=0, names=[], memory_mb=[])


def get_video_dimensions(video_path: str) -> Optional[Tuple[int, int]]:
    """Get video dimensions using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height',
        '-of', 'csv=p=0',
        video_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    except subprocess.TimeoutExpired:
        logger.warning(f"ffprobe timed out getting dimensions for {video_path}")
    except (subprocess.SubprocessError, ValueError, OSError) as e:
        logger.debug(f"Could not get video dimensions for {video_path}: {e}")
    return None


def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """Get image dimensions using PIL."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except (IOError, OSError) as e:
        logger.debug(f"Could not get image dimensions for {image_path}: {e}")
        return None


def process_resize(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Resize image or video to target resolution.

    Args:
        input_path: Path to input file
        output_path: Path to output file
        config: Resize configuration
        progress_callback: Optional callback(progress: 0-1, message: str)

    Returns:
        Dict with processing results
    """

    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    # Get config
    target_width = config.get('width', 1080)
    target_height = config.get('height', 1920)
    skip_if_correct = config.get('skipIfCorrect', True)
    gpu_id = config.get('_gpu_id', None)

    # Check for batch mode (list of files or ZIP)
    if isinstance(input_path, list):
        # Batch mode: list of video paths
        report_progress(0.01, f"Batch mode: processing {len(input_path)} files...")
        return process_videos_parallel(input_path, output_path, config, progress_callback)

    # Check if input is a ZIP file
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.zip':
        report_progress(0.01, "ZIP mode: extracting and processing...")
        return _process_zip_batch(input_path, output_path, config, progress_callback)

    # Detect file type for single file
    is_video = ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v', '.flv', '.wmv']

    report_progress(0.05, "Analyzing file...")

    # Get current dimensions
    if is_video:
        dims = get_video_dimensions(input_path)
    else:
        dims = get_image_dimensions(input_path)

    if dims is None:
        raise RuntimeError(f"Could not read dimensions from {input_path}")

    current_width, current_height = dims

    # Check if already at target resolution
    if skip_if_correct and current_width == target_width and current_height == target_height:
        report_progress(1.0, "Already at target resolution, copying...")
        # Just copy the file
        shutil.copy2(input_path, output_path)
        return {
            'skipped': True,
            'reason': 'Already at target resolution',
            'original_resolution': f"{current_width}x{current_height}",
            'target_resolution': f"{target_width}x{target_height}"
        }

    report_progress(0.1, f"Resizing from {current_width}x{current_height} to {target_width}x{target_height}...")

    if is_video:
        result = process_video_resize(input_path, output_path, current_width, current_height,
                                       target_width, target_height, report_progress, gpu_id)
    else:
        result = process_image_resize(input_path, output_path, target_width, target_height, report_progress)

    result['original_resolution'] = f"{current_width}x{current_height}"
    result['target_resolution'] = f"{target_width}x{target_height}"

    return result


def process_image_resize(
    input_path: str,
    output_path: str,
    target_width: int,
    target_height: int,
    report_progress: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Resize image using PIL with center crop."""

    report_progress(0.2, "Loading image...")

    img = Image.open(input_path)

    # Handle EXIF rotation
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img)

    # Convert to RGB if necessary
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    original_width, original_height = img.size
    target_aspect = target_width / target_height
    original_aspect = original_width / original_height

    report_progress(0.4, "Cropping to aspect ratio...")

    # Crop to match target aspect ratio
    if original_aspect > target_aspect:
        # Image is wider, crop horizontally
        new_width = int(target_aspect * original_height)
        left = (original_width - new_width) // 2
        img = img.crop((left, 0, left + new_width, original_height))
    elif original_aspect < target_aspect:
        # Image is taller, crop vertically
        new_height = int(original_width / target_aspect)
        top = (original_height - new_height) // 2
        img = img.crop((0, top, original_width, top + new_height))

    report_progress(0.6, "Scaling to target resolution...")

    # Resize to target dimensions
    img = img.resize((target_width, target_height), Image.LANCZOS)

    report_progress(0.8, "Saving output...")

    # Save
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext in ['.jpg', '.jpeg']:
        img.save(output_path, 'JPEG', quality=95)
    elif output_ext == '.png':
        img.save(output_path, 'PNG')
    else:
        img.save(output_path, 'JPEG', quality=95)

    report_progress(1.0, "Complete")

    return {
        'skipped': False,
        'type': 'image'
    }


def process_video_resize(
    input_path: str,
    output_path: str,
    current_width: int,
    current_height: int,
    target_width: int,
    target_height: int,
    report_progress: Callable[[float, str], None],
    gpu_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Resize video using FFmpeg with GPU acceleration and CPU fallback.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        current_width: Current video width
        current_height: Current video height
        target_width: Target video width
        target_height: Target video height
        report_progress: Progress callback function
        gpu_id: Optional GPU index for multi-GPU systems (0-indexed)

    Returns:
        Dict with processing results
    """

    report_progress(0.15, "Building resize filter...")

    target_aspect = target_width / target_height
    original_aspect = current_width / current_height

    # Get video duration for progress
    duration = get_video_duration(input_path)

    # Determine GPU to use
    gpu_str = str(gpu_id) if gpu_id is not None else "0"

    def build_ffmpeg_cmd(use_nvenc: bool = True) -> list:
        """Build FFmpeg command with GPU or CPU encoding."""
        # Build video filter based on encoding mode
        if use_nvenc:
            if abs(original_aspect - target_aspect) < 0.01:
                vf = f"scale_cuda={target_width}:{target_height}"
            elif original_aspect > target_aspect:
                new_width = int(target_aspect * current_height)
                x = (current_width - new_width) // 2
                vf = f"crop={new_width}:{current_height}:{x}:0,scale_cuda={target_width}:{target_height}"
            else:
                new_height = int(current_width / target_aspect)
                y = (current_height - new_height) // 2
                vf = f"crop={current_width}:{new_height}:0:{y},scale_cuda={target_width}:{target_height}"

            cmd = [
                'ffmpeg', '-y',
                '-hwaccel', 'cuda',
                '-hwaccel_device', gpu_str,
                '-hwaccel_output_format', 'cuda',
                '-i', input_path,
                '-vf', vf,
                '-c:v', 'h264_nvenc',
                '-gpu', gpu_str,
                '-preset', 'p4',
                '-b:v', '5000k',
                '-maxrate', '7500k',
                '-bufsize', '10000k',
            ]
        else:
            # CPU fallback - use standard scale filter
            if abs(original_aspect - target_aspect) < 0.01:
                vf = f"scale={target_width}:{target_height}"
            elif original_aspect > target_aspect:
                new_width = int(target_aspect * current_height)
                x = (current_width - new_width) // 2
                vf = f"crop={new_width}:{current_height}:{x}:0,scale={target_width}:{target_height}"
            else:
                new_height = int(current_width / target_aspect)
                y = (current_height - new_height) // 2
                vf = f"crop={current_width}:{new_height}:0:{y},scale={target_width}:{target_height}"

            cmd = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', vf,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
            ]

        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path
        ])
        return cmd

    def run_ffmpeg_with_progress(cmd: list, mode_name: str) -> tuple:
        """Run FFmpeg and capture stderr properly. Returns (success, stderr_output)."""
        stderr_lines = []

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Read stderr line by line for progress
        try:
            while True:
                line = process.stderr.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue

                stderr_lines.append(line)
                # Keep last 50 lines to avoid memory issues
                if len(stderr_lines) > 50:
                    stderr_lines.pop(0)

                if 'time=' in line:
                    try:
                        time_str = line.split('time=')[1].split()[0]
                        h, m, s = time_str.split(':')
                        current_time = int(h) * 3600 + int(m) * 60 + float(s)
                        if duration > 0:
                            progress = min(0.2 + (current_time / duration) * 0.75, 0.95)
                            report_progress(progress, f"Encoding ({mode_name})... {int(current_time)}s / {int(duration)}s")
                    except Exception:
                        pass

            # Wait for process to complete with timeout
            try:
                process.wait(timeout=600)  # 10 minute timeout
            except subprocess.TimeoutExpired:
                process.kill()
                return False, "FFmpeg process timed out after 600 seconds"

        except Exception as e:
            process.kill()
            return False, f"Error during encoding: {str(e)}"

        stderr_output = ''.join(stderr_lines)
        return process.returncode == 0, stderr_output

    report_progress(0.2, "Encoding with NVENC...")

    # Try NVENC first
    cmd = build_ffmpeg_cmd(use_nvenc=True)
    success, stderr = run_ffmpeg_with_progress(cmd, "NVENC")

    # If NVENC failed, try CPU fallback
    if not success:
        print(f"[Resize] NVENC failed, trying CPU fallback. Error: {stderr[-500:]}")
        report_progress(0.2, "NVENC failed, trying CPU encoding...")
        cmd = build_ffmpeg_cmd(use_nvenc=False)
        success, stderr = run_ffmpeg_with_progress(cmd, "CPU")

    if not success:
        raise RuntimeError(f"FFmpeg failed with all encoders: {stderr[-1000:]}")

    report_progress(1.0, "Complete")

    return {
        'skipped': False,
        'type': 'video',
        'duration': duration
    }


def get_video_duration(path: str) -> float:
    """Get video duration in seconds."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
        return float(info.get('format', {}).get('duration', 0))
    except subprocess.TimeoutExpired:
        logger.warning(f"ffprobe timed out getting duration for {path}")
        return 0
    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.debug(f"Could not get video duration for {path}: {e}")
        return 0


# ==============================================================================
# Multi-GPU Parallel Processing Functions
# ==============================================================================

@dataclass
class BatchProgress:
    """Thread-safe progress tracker for batch processing."""
    total_files: int
    completed: int = 0
    failed: int = 0
    _lock: Lock = None

    def __post_init__(self):
        self._lock = Lock()

    def increment_completed(self):
        with self._lock:
            self.completed += 1

    def increment_failed(self):
        with self._lock:
            self.failed += 1

    @property
    def progress(self) -> float:
        with self._lock:
            return (self.completed + self.failed) / max(1, self.total_files)


def process_single_video_resize_worker(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    gpu_id: int,
    file_index: int,
    batch_progress: BatchProgress
) -> Dict[str, Any]:
    """
    Worker function for parallel video processing on a specific GPU.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        config: Resize configuration
        gpu_id: GPU index to use for this video
        file_index: Index of this file in the batch
        batch_progress: Shared progress tracker

    Returns:
        Dict with processing result and metadata
    """
    try:
        # Create config copy with assigned GPU
        worker_config = config.copy()
        worker_config['_gpu_id'] = gpu_id

        # Silent progress callback for workers (to avoid console spam)
        def worker_progress(progress: float, message: str):
            pass  # Suppress individual file progress in batch mode

        # Get video dimensions
        dims = get_video_dimensions(input_path)
        if dims is None:
            raise RuntimeError(f"Could not read dimensions from {input_path}")

        current_width, current_height = dims
        target_width = config.get('width', 1080)
        target_height = config.get('height', 1920)

        # Check skip condition
        skip_if_correct = config.get('skipIfCorrect', True)
        if skip_if_correct and current_width == target_width and current_height == target_height:
            shutil.copy2(input_path, output_path)
            batch_progress.increment_completed()
            return {
                'success': True,
                'file_index': file_index,
                'input': input_path,
                'output': output_path,
                'gpu_id': gpu_id,
                'skipped': True,
                'reason': 'Already at target resolution'
            }

        # Process video
        result = process_video_resize(
            input_path, output_path,
            current_width, current_height,
            target_width, target_height,
            worker_progress,
            gpu_id
        )

        batch_progress.increment_completed()

        return {
            'success': True,
            'file_index': file_index,
            'input': input_path,
            'output': output_path,
            'gpu_id': gpu_id,
            'skipped': False,
            'original_resolution': f"{current_width}x{current_height}",
            'target_resolution': f"{target_width}x{target_height}",
            'duration': result.get('duration', 0)
        }

    except Exception as e:
        batch_progress.increment_failed()
        return {
            'success': False,
            'file_index': file_index,
            'input': input_path,
            'output': output_path,
            'gpu_id': gpu_id,
            'error': str(e)
        }


def process_videos_parallel(
    input_paths: List[str],
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process multiple videos in parallel across available GPUs.

    Args:
        input_paths: List of input video paths
        output_dir: Directory for output files (or output ZIP path)
        config: Resize configuration
        progress_callback: Optional progress callback

    Returns:
        Dict with batch processing results
    """
    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    report_progress(0.01, "Detecting GPUs...")

    # Detect available GPUs
    gpu_info = get_gpu_info()
    num_gpus = max(1, gpu_info.count)  # At least 1 for CPU fallback

    print(f"[Resize] Detected {gpu_info.count} GPUs: {gpu_info.names}")
    report_progress(0.02, f"Found {gpu_info.count} GPUs")

    # Prepare output paths
    output_is_zip = output_dir.lower().endswith('.zip')
    if output_is_zip:
        # Create temp directory for outputs
        temp_output_dir = tempfile.mkdtemp(prefix='resize_batch_')
    else:
        temp_output_dir = output_dir
        os.makedirs(temp_output_dir, exist_ok=True)

    try:
        # Generate output paths for each input
        output_paths = []
        for input_path in input_paths:
            basename = os.path.basename(input_path)
            name, ext = os.path.splitext(basename)
            output_name = f"{name}_resized{ext}"
            output_paths.append(os.path.join(temp_output_dir, output_name))

        # Initialize progress tracker
        batch_progress = BatchProgress(total_files=len(input_paths))

        report_progress(0.05, f"Processing {len(input_paths)} files on {num_gpus} GPU(s)...")

        # Determine max workers (one per GPU to avoid memory issues)
        max_workers = num_gpus

        results = []

        # Process in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for idx, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
                # Round-robin GPU assignment
                assigned_gpu = idx % num_gpus

                future = executor.submit(
                    process_single_video_resize_worker,
                    input_path,
                    output_path,
                    config,
                    assigned_gpu,
                    idx,
                    batch_progress
                )
                futures[future] = idx

            # Collect results and report progress
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # Update progress
                progress = 0.05 + (batch_progress.progress * 0.90)
                completed = batch_progress.completed
                failed = batch_progress.failed
                total = batch_progress.total_files

                status = f"Completed {completed}/{total}"
                if failed > 0:
                    status += f" ({failed} failed)"

                report_progress(progress, status)

        # Sort results by file index
        results.sort(key=lambda r: r.get('file_index', 0))

        # If output should be ZIP, create it
        if output_is_zip:
            report_progress(0.96, "Creating output ZIP...")
            with zipfile.ZipFile(output_dir, 'w', zipfile.ZIP_DEFLATED) as zf:
                for result in results:
                    if result.get('success', False):
                        output_path = result['output']
                        if os.path.exists(output_path):
                            arcname = os.path.basename(output_path)
                            zf.write(output_path, arcname)

        report_progress(1.0, "Batch processing complete")

        # Summary
        successful = sum(1 for r in results if r.get('success', False))
        failed = sum(1 for r in results if not r.get('success', False))
        skipped = sum(1 for r in results if r.get('skipped', False))

        return {
            'batch': True,
            'total_files': len(input_paths),
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'gpus_used': num_gpus,
            'gpu_names': gpu_info.names,
            'results': results
        }

    finally:
        # Clean up temp directory only if we created one for ZIP output
        if output_is_zip:
            shutil.rmtree(temp_output_dir, ignore_errors=True)


def _process_zip_batch(
    input_zip: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Extract ZIP, process all videos, and create output ZIP.

    Args:
        input_zip: Path to input ZIP file
        output_path: Path to output ZIP file
        config: Resize configuration
        progress_callback: Optional progress callback

    Returns:
        Dict with batch processing results
    """
    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    report_progress(0.01, "Extracting input ZIP...")

    # Create temp directory for extraction
    temp_extract_dir = tempfile.mkdtemp(prefix='resize_extract_')

    try:
        # Extract ZIP (with path traversal protection)
        with zipfile.ZipFile(input_zip, 'r') as zf:
            for name in zf.namelist():
                safe_extract(zf, name, temp_extract_dir)

        # Find video files
        video_extensions = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v', '.flv', '.wmv'}
        video_files = []

        for root, dirs, files in os.walk(temp_extract_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    video_files.append(os.path.join(root, file))

        if not video_files:
            raise RuntimeError("No video files found in ZIP")

        report_progress(0.02, f"Found {len(video_files)} videos in ZIP")

        # Process videos in parallel
        result = process_videos_parallel(
            video_files,
            output_path,
            config,
            progress_callback
        )

        return result

    finally:
        # Clean up extraction directory
        shutil.rmtree(temp_extract_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python processor.py <command> [args...]")
        print("")
        print("Commands:")
        print("  resize <input> <output> [width] [height] [gpu_id]  - Resize single file")
        print("  batch <input_dir> <output.zip> [width] [height]    - Batch process directory")
        print("  gpuinfo                                            - Show GPU information")
        print("")
        print("Examples:")
        print("  python processor.py resize video.mp4 out.mp4 1080 1920")
        print("  python processor.py resize video.mp4 out.mp4 1080 1920 1  # Use GPU 1")
        print("  python processor.py batch ./videos output.zip 1080 1920")
        print("  python processor.py gpuinfo")
        sys.exit(1)

    command = sys.argv[1].lower()

    def progress(p, msg):
        print(f"[{int(p*100):3d}%] {msg}")

    if command == 'gpuinfo':
        # Show GPU information
        info = get_gpu_info()
        print(f"\nDetected {info.count} NVIDIA GPU(s):")
        for i, (name, mem) in enumerate(zip(info.names, info.memory_mb)):
            print(f"  GPU {i}: {name} ({mem} MB)")
        if info.count == 0:
            print("  No NVIDIA GPUs detected (will use CPU fallback)")

    elif command == 'resize':
        if len(sys.argv) < 4:
            print("Usage: python processor.py resize <input> <output> [width] [height] [gpu_id]")
            sys.exit(1)

        test_config = {
            'width': int(sys.argv[4]) if len(sys.argv) > 4 else 1080,
            'height': int(sys.argv[5]) if len(sys.argv) > 5 else 1920,
            'skipIfCorrect': True
        }

        # Optional GPU ID
        if len(sys.argv) > 6:
            test_config['_gpu_id'] = int(sys.argv[6])

        result = process_resize(sys.argv[2], sys.argv[3], test_config, progress)
        print(f"\nResult: {json.dumps(result, indent=2)}")

    elif command == 'batch':
        if len(sys.argv) < 4:
            print("Usage: python processor.py batch <input_dir_or_zip> <output.zip> [width] [height]")
            sys.exit(1)

        input_path = sys.argv[2]
        output_path = sys.argv[3]

        test_config = {
            'width': int(sys.argv[4]) if len(sys.argv) > 4 else 1080,
            'height': int(sys.argv[5]) if len(sys.argv) > 5 else 1920,
            'skipIfCorrect': True
        }

        # Check if input is directory or ZIP
        if os.path.isdir(input_path):
            # Find all video files in directory
            video_extensions = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v', '.flv', '.wmv'}
            video_files = []
            for file in os.listdir(input_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    video_files.append(os.path.join(input_path, file))

            if not video_files:
                print(f"No video files found in {input_path}")
                sys.exit(1)

            print(f"Found {len(video_files)} video files")
            result = process_videos_parallel(video_files, output_path, test_config, progress)
        else:
            # Assume it's a ZIP file
            result = process_resize(input_path, output_path, test_config, progress)

        print(f"\nResult: {json.dumps(result, indent=2, default=str)}")

    else:
        # Legacy mode: treat first arg as input file
        if len(sys.argv) < 3:
            print("Usage: python processor.py input_file output_file [width] [height]")
            sys.exit(1)

        test_config = {
            'width': int(sys.argv[3]) if len(sys.argv) > 3 else 1080,
            'height': int(sys.argv[4]) if len(sys.argv) > 4 else 1920,
            'skipIfCorrect': True
        }

        result = process_resize(sys.argv[1], sys.argv[2], test_config, progress)
        print(f"\nResult: {json.dumps(result, indent=2)}")
