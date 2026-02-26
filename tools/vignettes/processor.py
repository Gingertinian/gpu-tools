"""
Vignettes Processor - Video overlay effects

Applies vignette overlays, borders, and visual effects to videos.
Uses NVENC for hardware-accelerated encoding with multi-GPU support.

Config structure:
{
    "overlayType": "vignette" | "border" | "frame" | "blur_edges",
    "intensity": 0-100,
    "color": "#hexcolor",
    "borderWidth": pixels,
    "cornerRadius": pixels,
    "blurAmount": 0-100,
    "customOverlay": "url to overlay image",
    "_gpu_id": int (optional, for multi-GPU selection)
}

Multi-GPU Features:
- Automatic GPU detection (count and type)
- Round-robin GPU assignment for batch processing
- Parallel video processing with ThreadPoolExecutor
- ZIP input support for batch mode
"""

import os
import logging
import subprocess
import tempfile
import zipfile
import shutil
import time
from typing import Callable, Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json

logger = logging.getLogger(__name__)

# Video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'}

# NVENC session limits by GPU type (per GPU)
# Consumer GPUs (GeForce): Limited to 3-5 sessions
# Datacenter GPUs (A-series, Quadro): Unlimited sessions
NVENC_SESSION_LIMITS = {
    'consumer': 3,      # RTX 3090, 4090, 4080, etc.
    'datacenter': 12,   # A5000, A6000, A100, etc.
    'default': 2,       # Fallback
}


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
            datacenter_keywords = ['A100', 'A6000', 'A5000', 'A4000', 'A4500', 'A40', 'A30', 'A10',
                                   'V100', 'T4', 'Quadro', 'Tesla', 'H100', 'L40', 'RTX 4090', 'RTX 6000']
            is_datacenter = any(kw in gpu_name for kw in datacenter_keywords)

            gpu_type = 'datacenter' if is_datacenter else 'consumer'
            base_sessions = NVENC_SESSION_LIMITS[gpu_type]

            # Scale sessions by number of GPUs (multi-GPU workers)
            nvenc_sessions = base_sessions * gpu_count

            print(f"[Vignettes GPU Detection] Found {gpu_count} GPU(s): {gpu_name}")
            print(f"[Vignettes GPU Detection] Type: {gpu_type}, Sessions per GPU: {base_sessions}, Total: {nvenc_sessions}")

            return {
                'gpu_name': gpu_name,
                'gpu_type': gpu_type,
                'gpu_count': gpu_count,
                'nvenc_sessions': nvenc_sessions
            }
    except Exception as e:
        print(f"[Vignettes GPU Detection] Error: {e}")

    return {
        'gpu_name': 'Unknown',
        'gpu_type': 'default',
        'gpu_count': 1,
        'nvenc_sessions': NVENC_SESSION_LIMITS['default']
    }


def safe_extract(zf: zipfile.ZipFile, name: str, extract_dir: str) -> str:
    """Safely extract a zip entry, preventing path traversal attacks (ZipSlip)."""
    target_path = os.path.realpath(os.path.join(extract_dir, name))
    extract_dir_real = os.path.realpath(extract_dir)
    if not target_path.startswith(extract_dir_real + os.sep) and target_path != extract_dir_real:
        raise ValueError(f"Attempted path traversal in zip entry: {name}")
    zf.extract(name, extract_dir)
    return target_path


def extract_videos_from_zip(zip_path: str, extract_dir: str) -> List[str]:
    """
    Extract video files from a ZIP archive.
    Returns list of paths to extracted video files.
    """
    video_paths = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            # Skip directories and hidden files
            if name.endswith('/') or name.startswith('__MACOSX') or name.startswith('.'):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                # Extract to temp dir
                extracted_path = safe_extract(zf, name, extract_dir)
                video_paths.append(extracted_path)
    return video_paths


def process_vignettes(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process video with vignette/overlay effects.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        config: Vignette configuration
        progress_callback: Optional callback(progress: 0-1, message: str)

    Returns:
        Dict with processing results
    """

    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    report_progress(0.05, "Analyzing video...")

    # Get video info
    video_info = get_video_info(input_path)
    width = video_info.get('width', 1920)
    height = video_info.get('height', 1080)
    duration = video_info.get('duration', 0)
    fps = video_info.get('fps', 30)

    report_progress(0.1, "Creating overlay...")

    # Extract config
    overlay_type = config.get('overlayType', 'vignette')
    intensity = config.get('intensity', 50) / 100
    color = config.get('color', '#000000')
    border_width = config.get('borderWidth', 20)
    corner_radius = config.get('cornerRadius', 0)
    blur_amount = config.get('blurAmount', 50) / 100

    effects_applied = []

    # Create temp directory for overlays
    temp_dir = tempfile.mkdtemp(prefix='vignettes_')
    overlay_path = os.path.join(temp_dir, 'overlay.png')

    try:
        # Generate overlay based on type
        if overlay_type == 'vignette':
            create_vignette_overlay(overlay_path, width, height, intensity, color)
            effects_applied.append(f"vignette:{int(intensity*100)}%")

        elif overlay_type == 'border':
            create_border_overlay(overlay_path, width, height, border_width, color, corner_radius)
            effects_applied.append(f"border:{border_width}px")

        elif overlay_type == 'frame':
            create_frame_overlay(overlay_path, width, height, border_width, color, corner_radius)
            effects_applied.append(f"frame:{border_width}px")

        elif overlay_type == 'blur_edges':
            # Blur edges requires different approach - edge blur filter
            pass

        elif overlay_type == 'custom' and config.get('customOverlay'):
            # Download and prepare custom overlay
            # For now, skip custom overlays
            pass

        report_progress(0.3, "Building filter chain...")

        # Build FFmpeg command
        filters = []

        if overlay_type == 'blur_edges':
            # Blur edges effect using FFmpeg filters
            blur_radius = int(blur_amount * 50)
            # Create edge mask and apply blur
            filters.append(
                f"split[original][blur];"
                f"[blur]boxblur={blur_radius}:{blur_radius}[blurred];"
                f"[original][blurred]blend=all_expr='if(gt(abs(X-W/2)/(W/2)+abs(Y-H/2)/(H/2),1.5-{intensity}),B,A)'"
            )
            effects_applied.append(f"blur_edges:{int(blur_amount*100)}%")
        elif os.path.exists(overlay_path):
            # Overlay the generated image
            filters.append(f"[0:v][1:v]overlay=0:0")

        report_progress(0.4, "Encoding with NVENC...")

        # Get GPU ID from config (for multi-GPU batch processing)
        gpu_id = config.get('_gpu_id', 0)
        if gpu_id > 0:
            print(f"[Vignettes] Using GPU {gpu_id} for NVENC encoding")

        # Build FFmpeg command - try NVENC first, then CPU fallback
        def build_ffmpeg_cmd(use_nvenc: bool = True, target_gpu: int = 0) -> list:
            cmd = ['ffmpeg', '-y']
            if use_nvenc:
                cmd.extend(['-hwaccel', 'cuda', '-hwaccel_device', str(target_gpu)])
            cmd.extend(['-i', input_path])

            if overlay_type != 'blur_edges' and os.path.exists(overlay_path):
                cmd.extend(['-i', overlay_path])

            filter_str = ';'.join(filters) if filters else None
            if filter_str:
                cmd.extend(['-filter_complex', filter_str])

            if use_nvenc:
                # NVENC encoding with GPU selection
                cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-gpu', str(target_gpu),  # Multi-GPU: select specific GPU
                    '-preset', 'p4',
                    '-b:v', '5000k',
                    '-maxrate', '7500k',
                    '-bufsize', '10000k',
                    '-profile:v', 'high',
                ])
            else:
                # CPU fallback with libx264
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-profile:v', 'high',
                ])

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
                                progress = min(0.4 + (current_time / duration) * 0.55, 0.95)
                                report_progress(progress, f"Encoding ({mode_name})... {int(current_time)}s / {int(duration)}s")
                        except (ValueError, IndexError):
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

        # Try NVENC first with specified GPU
        cmd = build_ffmpeg_cmd(use_nvenc=True, target_gpu=gpu_id)
        success, stderr = run_ffmpeg_with_progress(cmd, f"NVENC-GPU{gpu_id}")

        # If NVENC failed, try CPU fallback
        if not success:
            print(f"[Vignettes] NVENC on GPU {gpu_id} failed, trying CPU fallback. Error: {stderr[-500:]}")
            report_progress(0.4, "NVENC failed, trying CPU encoding...")
            cmd = build_ffmpeg_cmd(use_nvenc=False, target_gpu=0)
            success, stderr = run_ffmpeg_with_progress(cmd, "CPU")

        if not success:
            raise RuntimeError(f"FFmpeg failed with all encoders: {stderr[-1000:]}")

        report_progress(1.0, "Complete")

        return {
            'effects_applied': effects_applied,
            'overlay_type': overlay_type,
            'resolution': f"{width}x{height}",
            'duration': duration
        }

    finally:
        # Cleanup temp files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except (IOError, OSError) as e:
            logger.debug(f"Failed to cleanup vignettes temp dir: {e}")


def get_video_info(path: str) -> Dict[str, Any]:
    """Get video metadata using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-show_format', path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)

        video_stream = next(
            (s for s in info.get('streams', []) if s.get('codec_type') == 'video'),
            {}
        )

        fps_str = video_stream.get('r_frame_rate', '30/1')
        try:
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = int(num) / int(den)
            else:
                fps = float(fps_str)
        except (ValueError, ZeroDivisionError):
            fps = 30

        return {
            'width': int(video_stream.get('width', 1920)),
            'height': int(video_stream.get('height', 1080)),
            'fps': fps,
            'duration': float(info.get('format', {}).get('duration', 0))
        }
    except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, KeyError, OSError) as e:
        logger.debug(f"Could not get video info: {e}")
        return {'width': 1920, 'height': 1080, 'fps': 30, 'duration': 0}


def create_vignette_overlay(path: str, width: int, height: int, intensity: float, color: str):
    """Create a vignette overlay image with transparency."""
    # Create RGBA image
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # Parse color
    r, g, b = hex_to_rgb(color)

    # Create radial gradient
    arr = np.zeros((height, width, 4), dtype=np.uint8)

    # Calculate center
    center_x, center_y = width / 2, height / 2
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

    # Generate gradient
    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    dist = dist / max_dist

    # Vignette alpha (more transparent in center, opaque at edges)
    alpha = (dist ** 1.5) * intensity * 255
    alpha = np.clip(alpha, 0, 255).astype(np.uint8)

    arr[:, :, 0] = r
    arr[:, :, 1] = g
    arr[:, :, 2] = b
    arr[:, :, 3] = alpha

    img = Image.fromarray(arr, 'RGBA')
    img.save(path, 'PNG')


def create_border_overlay(path: str, width: int, height: int, border_width: int, color: str, corner_radius: int = 0):
    """Create a border overlay."""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    r, g, b = hex_to_rgb(color)

    # Draw border rectangles
    # Top
    draw.rectangle([0, 0, width, border_width], fill=(r, g, b, 255))
    # Bottom
    draw.rectangle([0, height - border_width, width, height], fill=(r, g, b, 255))
    # Left
    draw.rectangle([0, 0, border_width, height], fill=(r, g, b, 255))
    # Right
    draw.rectangle([width - border_width, 0, width, height], fill=(r, g, b, 255))

    # Apply corner radius if specified
    if corner_radius > 0:
        # Create mask with rounded corners
        mask = Image.new('L', (width, height), 255)
        mask_draw = ImageDraw.Draw(mask)

        # Cut out rounded rectangle from center
        inner_rect = [
            border_width, border_width,
            width - border_width, height - border_width
        ]
        mask_draw.rounded_rectangle(inner_rect, radius=corner_radius, fill=0)

        img.putalpha(mask)

    img.save(path, 'PNG')


def create_frame_overlay(path: str, width: int, height: int, frame_width: int, color: str, corner_radius: int = 0):
    """Create a decorative frame overlay."""
    img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    r, g, b = hex_to_rgb(color)

    # Outer rectangle
    outer = [0, 0, width, height]
    # Inner rectangle (cutout)
    inner = [frame_width, frame_width, width - frame_width, height - frame_width]

    # Draw frame
    if corner_radius > 0:
        draw.rounded_rectangle(outer, radius=corner_radius + frame_width, fill=(r, g, b, 255))
        draw.rounded_rectangle(inner, radius=corner_radius, fill=(0, 0, 0, 0))
    else:
        draw.rectangle(outer, fill=(r, g, b, 255))
        draw.rectangle(inner, fill=(0, 0, 0, 0))

    img.save(path, 'PNG')


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])

    try:
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, IndexError) as e:
        logger.debug(f"Invalid hex color '{hex_color}': {e}")
        return (0, 0, 0)


# ═══════════════════════════════════════════════════════════════
# PARALLEL VIDEO PROCESSING (Multi-GPU Support)
# ═══════════════════════════════════════════════════════════════

def process_single_video_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel video processing.
    Designed to be called from ThreadPoolExecutor.
    FFmpeg handles GPU work, so ThreadPoolExecutor is appropriate.

    Args tuple: (input_path, output_path, config, video_index, gpu_id)
    Returns: dict with status, output_path, and any errors
    """
    input_path, output_path, config, video_index, gpu_id = args

    try:
        # Create config copy with GPU ID
        worker_config = config.copy()
        worker_config['_gpu_id'] = gpu_id

        print(f"[Vignettes Worker {video_index}] Processing on GPU {gpu_id}: {os.path.basename(input_path)}")

        # Get video info first
        video_info = get_video_info(input_path)
        duration = video_info.get('duration', 0)

        # Process the video (reuse the main processing logic)
        result = process_single_video_internal(input_path, output_path, worker_config)

        return {
            'status': 'completed',
            'index': video_index,
            'output_path': output_path,
            'duration': duration,
            'gpu_id': gpu_id,
            'result': result
        }

    except Exception as e:
        import traceback
        print(f"[Vignettes Worker {video_index}] Error: {str(e)}")
        print(traceback.format_exc())
        return {
            'status': 'failed',
            'index': video_index,
            'error': str(e),
            'gpu_id': gpu_id
        }


def process_single_video_internal(
    input_path: str,
    output_path: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Internal video processing function used by workers.
    Processes a single video with vignette effects using the specified GPU.
    """
    # Get video info
    video_info = get_video_info(input_path)
    width = video_info.get('width', 1920)
    height = video_info.get('height', 1080)
    duration = video_info.get('duration', 0)

    # Extract config
    overlay_type = config.get('overlayType', 'vignette')
    intensity = config.get('intensity', 50) / 100
    color = config.get('color', '#000000')
    border_width = config.get('borderWidth', 20)
    corner_radius = config.get('cornerRadius', 0)
    blur_amount = config.get('blurAmount', 50) / 100
    gpu_id = config.get('_gpu_id', 0)

    effects_applied = []

    # Create temp directory for overlays
    temp_dir = tempfile.mkdtemp(prefix='vignettes_worker_')
    overlay_path = os.path.join(temp_dir, 'overlay.png')

    try:
        # Generate overlay based on type
        if overlay_type == 'vignette':
            create_vignette_overlay(overlay_path, width, height, intensity, color)
            effects_applied.append(f"vignette:{int(intensity*100)}%")
        elif overlay_type == 'border':
            create_border_overlay(overlay_path, width, height, border_width, color, corner_radius)
            effects_applied.append(f"border:{border_width}px")
        elif overlay_type == 'frame':
            create_frame_overlay(overlay_path, width, height, border_width, color, corner_radius)
            effects_applied.append(f"frame:{border_width}px")

        # Build filter chain
        filters = []
        if overlay_type == 'blur_edges':
            blur_radius = int(blur_amount * 50)
            filters.append(
                f"split[original][blur];"
                f"[blur]boxblur={blur_radius}:{blur_radius}[blurred];"
                f"[original][blurred]blend=all_expr='if(gt(abs(X-W/2)/(W/2)+abs(Y-H/2)/(H/2),1.5-{intensity}),B,A)'"
            )
            effects_applied.append(f"blur_edges:{int(blur_amount*100)}%")
        elif os.path.exists(overlay_path):
            filters.append(f"[0:v][1:v]overlay=0:0")

        # Build FFmpeg command with multi-GPU support
        def build_cmd(use_nvenc: bool, target_gpu: int) -> list:
            cmd = ['ffmpeg', '-y']
            if use_nvenc:
                cmd.extend(['-hwaccel', 'cuda', '-hwaccel_device', str(target_gpu)])
            cmd.extend(['-i', input_path])

            if overlay_type != 'blur_edges' and os.path.exists(overlay_path):
                cmd.extend(['-i', overlay_path])

            filter_str = ';'.join(filters) if filters else None
            if filter_str:
                cmd.extend(['-filter_complex', filter_str])

            if use_nvenc:
                cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-gpu', str(target_gpu),
                    '-preset', 'p4',
                    '-b:v', '5000k',
                    '-maxrate', '7500k',
                    '-bufsize', '10000k',
                    '-profile:v', 'high',
                ])
            else:
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '23',
                    '-profile:v', 'high',
                ])

            cmd.extend(['-c:a', 'aac', '-b:a', '128k', output_path])
            return cmd

        # Try NVENC first
        cmd = build_cmd(use_nvenc=True, target_gpu=gpu_id)
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=900)

        if process.returncode != 0:
            # NVENC failed, try CPU
            print(f"[Vignettes Internal] NVENC on GPU {gpu_id} failed, trying CPU")
            cmd = build_cmd(use_nvenc=False, target_gpu=0)
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {process.stderr[-500:] if process.stderr else 'Unknown'}")

        return {
            'effects_applied': effects_applied,
            'overlay_type': overlay_type,
            'resolution': f"{width}x{height}",
            'duration': duration,
            'gpu_id': gpu_id
        }

    finally:
        # Cleanup temp files
        try:
            shutil.rmtree(temp_dir)
        except (IOError, OSError) as e:
            logger.debug(f"Failed to clean up temp dir {temp_dir}: {e}")


def process_videos_parallel(
    video_paths: List[str],
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    max_parallel: int = None
) -> Dict[str, Any]:
    """
    Process multiple videos in parallel using multiple NVENC sessions.
    Uses ThreadPoolExecutor since FFmpeg handles the GPU work externally.

    Args:
        video_paths: List of input video paths
        output_dir: Directory for output videos
        config: Vignette configuration
        progress_callback: Optional progress callback
        max_parallel: Max parallel threads (auto-detected if None)

    Returns:
        Dict with results including processed count and any errors
    """
    if not video_paths:
        return {'error': 'No videos to process'}

    # Auto-detect parallel limit based on GPU
    gpu_info = get_gpu_info()
    gpu_count = gpu_info['gpu_count']

    if max_parallel is None:
        max_parallel = gpu_info['nvenc_sessions']

    print(f"[Vignettes Parallel] GPU: {gpu_info['gpu_name']}, Type: {gpu_info['gpu_type']}, "
          f"GPUs: {gpu_count}, Max parallel: {max_parallel}")

    # Prepare work items with GPU assignment (round-robin across GPUs)
    work_items = []
    for i, video_path in enumerate(video_paths):
        basename = os.path.basename(video_path)
        name, ext = os.path.splitext(basename)
        # Preserve original extension, default to .mp4
        out_ext = ext if ext.lower() in VIDEO_EXTENSIONS else '.mp4'
        output_path = os.path.join(output_dir, f"{name}_vignette{out_ext}")
        gpu_id = i % gpu_count  # Round-robin GPU assignment
        work_items.append((video_path, output_path, config, i, gpu_id))

    total = len(work_items)
    completed = 0
    failed = 0
    results = []

    def report_progress(msg=""):
        if progress_callback:
            progress = (completed + failed) / total if total > 0 else 0
            progress_callback(progress, msg)

    report_progress(f"Processing {total} videos with {max_parallel} parallel sessions across {gpu_count} GPU(s)...")

    # Process in parallel using ThreadPoolExecutor
    # ThreadPoolExecutor is appropriate here because FFmpeg does the GPU work
    # and we're just launching/managing external processes
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_item = {executor.submit(process_single_video_worker, item): item
                        for item in work_items}

        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)

            if result['status'] == 'completed':
                completed += 1
                print(f"[Vignettes Parallel] Completed video {result['index']+1}/{total} on GPU {result['gpu_id']}")
            else:
                failed += 1
                print(f"[Vignettes Parallel] Failed video {result['index']}: {result.get('error', 'unknown')}")

            report_progress(f"Completed {completed}/{total} videos ({failed} failed)")

    return {
        'status': 'completed',
        'total': total,
        'completed': completed,
        'failed': failed,
        'results': results,
        'parallel_sessions': max_parallel,
        'gpu_count': gpu_count,
        'gpu_type': gpu_info['gpu_type']
    }


def process_vignettes_batch(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process vignettes with automatic detection of ZIP input for batch mode.

    If input is a ZIP file containing videos:
    - Extracts videos
    - Processes them in parallel using multiple GPUs
    - Creates output ZIP with processed videos

    If input is a single video:
    - Processes normally with the standard process_vignettes function
    """
    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    ext = os.path.splitext(input_path)[1].lower()

    # Check if input is a ZIP
    if ext == '.zip':
        report_progress(0.05, "Detected ZIP input, extracting videos...")

        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix='vignettes_batch_')

        try:
            # Extract videos from ZIP
            video_paths = extract_videos_from_zip(input_path, temp_dir)

            if not video_paths:
                return {'error': 'No video files found in ZIP archive'}

            report_progress(0.1, f"Found {len(video_paths)} videos, starting parallel processing...")

            # Create output directory
            output_dir = os.path.join(temp_dir, 'output')
            os.makedirs(output_dir, exist_ok=True)

            # Process videos in parallel
            result = process_videos_parallel(
                video_paths,
                output_dir,
                config,
                progress_callback=lambda p, m: report_progress(0.1 + p * 0.8, m)
            )

            if result.get('error'):
                return result

            # Create output ZIP with processed videos
            report_progress(0.92, "Creating output ZIP...")

            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zf:
                for r in result.get('results', []):
                    if r.get('status') == 'completed' and r.get('output_path'):
                        if os.path.exists(r['output_path']):
                            arcname = os.path.basename(r['output_path'])
                            zf.write(r['output_path'], arcname)

            report_progress(1.0, "Complete")

            return {
                'status': 'completed',
                'mode': 'parallel_video_batch',
                'videos_processed': result.get('completed', 0),
                'videos_failed': result.get('failed', 0),
                'parallel_sessions': result.get('parallel_sessions', 1),
                'gpu_count': result.get('gpu_count', 1),
                'gpu_type': result.get('gpu_type', 'unknown'),
                'output_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
            }

        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir)
            except (IOError, OSError) as e:
                logger.debug(f"Failed to clean up temp dir {temp_dir}: {e}")

    # Single video - use standard processing
    return process_vignettes(input_path, output_path, config, progress_callback)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor.py input_file output_file [--batch]")
        print("")
        print("Options:")
        print("  input_file   - Single video file or ZIP containing multiple videos")
        print("  output_file  - Output video or ZIP file")
        print("  --batch      - Force batch processing mode (auto-detected for .zip)")
        print("")
        print("Multi-GPU Support:")
        print("  - Automatically detects number of GPUs")
        print("  - Uses round-robin GPU assignment for parallel processing")
        print("  - Supports datacenter GPUs (A5000, A6000, etc.) with unlimited NVENC sessions")
        sys.exit(1)

    # Print GPU info
    print("\n=== GPU Detection ===")
    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info['gpu_name']}")
    print(f"Type: {gpu_info['gpu_type']}")
    print(f"Count: {gpu_info['gpu_count']}")
    print(f"Max NVENC sessions: {gpu_info['nvenc_sessions']}")
    print("=====================\n")

    test_config = {
        'overlayType': 'vignette',
        'intensity': 50,
        'color': '#000000'
    }

    def progress(p, msg):
        print(f"[{int(p*100):3d}%] {msg}")

    # Check if batch mode (ZIP input or --batch flag)
    use_batch = '--batch' in sys.argv or sys.argv[1].lower().endswith('.zip')

    if use_batch:
        print("Using batch processing mode...")
        result = process_vignettes_batch(sys.argv[1], sys.argv[2], test_config, progress)
    else:
        print("Using single video processing mode...")
        result = process_vignettes(sys.argv[1], sys.argv[2], test_config, progress)

    print(f"\nResult: {result}")
