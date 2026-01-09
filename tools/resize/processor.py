"""
Resize Processor - Resize and crop images/videos to target resolution

Converts media to target resolution (default 1080x1920) using:
- Center crop to match aspect ratio
- Scale to exact dimensions
- GPU acceleration for videos (NVENC)

Config structure:
{
    "width": 1080,        # Target width (default 1080)
    "height": 1920,       # Target height (default 1920)
    "skipIfCorrect": true # Skip if already at target resolution
}
"""

import os
import subprocess
import tempfile
from typing import Callable, Optional, Dict, Any, Tuple
from PIL import Image
import json


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
    except Exception:
        pass
    return None


def get_image_dimensions(image_path: str) -> Optional[Tuple[int, int]]:
    """Get image dimensions using PIL."""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception:
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

    # Detect file type
    ext = os.path.splitext(input_path)[1].lower()
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
        import shutil
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
                                       target_width, target_height, report_progress)
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
    report_progress: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Resize video using FFmpeg with GPU acceleration."""

    report_progress(0.15, "Building resize filter...")

    target_aspect = target_width / target_height
    original_aspect = current_width / current_height

    # Build video filter
    if abs(original_aspect - target_aspect) < 0.01:
        # Aspect ratio is close enough, just scale
        vf = f"scale_cuda={target_width}:{target_height}"
    elif original_aspect > target_aspect:
        # Video is wider, crop horizontally first then scale
        new_width = int(target_aspect * current_height)
        x = (current_width - new_width) // 2
        vf = f"crop={new_width}:{current_height}:{x}:0,scale_cuda={target_width}:{target_height}"
    else:
        # Video is taller, crop vertically first then scale
        new_height = int(current_width / target_aspect)
        y = (current_height - new_height) // 2
        vf = f"crop={current_width}:{new_height}:0:{y},scale_cuda={target_width}:{target_height}"

    report_progress(0.2, "Encoding with NVENC...")

    # FFmpeg command with GPU acceleration
    cmd = [
        'ffmpeg', '-y',
        '-hwaccel', 'cuda',
        '-hwaccel_output_format', 'cuda',
        '-i', input_path,
        '-vf', vf,
        '-c:v', 'h264_nvenc',
        '-preset', 'p4',
        '-b:v', '5000k',
        '-maxrate', '7500k',
        '-bufsize', '10000k',
        '-c:a', 'aac',
        '-b:a', '128k',
        output_path
    ]

    # Get video duration for progress
    duration = get_video_duration(input_path)

    # Run FFmpeg
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )

    # Monitor progress
    while True:
        line = process.stderr.readline()
        if not line and process.poll() is not None:
            break

        if 'time=' in line:
            try:
                time_str = line.split('time=')[1].split()[0]
                h, m, s = time_str.split(':')
                current_time = int(h) * 3600 + int(m) * 60 + float(s)
                if duration > 0:
                    progress = min(0.2 + (current_time / duration) * 0.75, 0.95)
                    report_progress(progress, f"Encoding... {int(current_time)}s / {int(duration)}s")
            except Exception:
                pass

    if process.returncode != 0:
        stderr = process.stderr.read()
        raise RuntimeError(f"FFmpeg failed: {stderr}")

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
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        return float(info.get('format', {}).get('duration', 0))
    except Exception:
        return 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor.py input_file output_file [width] [height]")
        sys.exit(1)

    test_config = {
        'width': int(sys.argv[3]) if len(sys.argv) > 3 else 1080,
        'height': int(sys.argv[4]) if len(sys.argv) > 4 else 1920,
        'skipIfCorrect': True
    }

    def progress(p, msg):
        print(f"[{int(p*100)}%] {msg}")

    result = process_resize(sys.argv[1], sys.argv[2], test_config, progress)
    print(f"Result: {result}")
