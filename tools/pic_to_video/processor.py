"""
Pic To Video Processor - Convert images to videos

Takes an image and creates a static video of configurable duration and FPS.
Uses GPU-accelerated encoding (NVENC) for fast processing.

Config structure:
{
    "duration": 10,       # Video duration in seconds (default 10)
    "fps": 24,            # Frames per second (default 24)
    "codec": "h264",      # Video codec (default h264)
    "quality": 23         # CRF quality 0-51 (lower = better, default 23)
}
"""

import os
import subprocess
from typing import Callable, Optional, Dict, Any, Tuple
from PIL import Image, ImageOps


def process_pic_to_video(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Convert an image to a static video.

    Args:
        input_path: Path to input image
        output_path: Path to output video (should be .mp4)
        config: Video configuration
        progress_callback: Optional callback(progress: 0-1, message: str)

    Returns:
        Dict with processing results
    """

    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    # Get config
    duration = config.get('duration', 10)
    fps = config.get('fps', 24)
    quality = config.get('quality', 23)

    # Validate input is an image
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff']:
        raise ValueError(f"Input must be an image, got: {ext}")

    report_progress(0.05, "Loading image...")

    # Load and prepare image
    img = Image.open(input_path)
    img = ImageOps.exif_transpose(img)

    # Convert to RGB if needed
    if img.mode in ('RGBA', 'P'):
        # Create white background for transparency
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'RGBA':
            background.paste(img, mask=img.split()[3])
        else:
            background.paste(img)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')

    width, height = img.size

    # Ensure dimensions are even (required for h264)
    if width % 2 != 0:
        width -= 1
    if height % 2 != 0:
        height -= 1

    if (width, height) != img.size:
        img = img.crop((0, 0, width, height))

    report_progress(0.1, f"Preparing {width}x{height} image...")

    # Save to temp file for FFmpeg
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        temp_path = tmp.name
        img.save(temp_path, 'JPEG', quality=95)

    try:
        report_progress(0.2, f"Encoding {duration}s video at {fps}fps...")

        # FFmpeg command - GPU accelerated
        cmd = [
            'ffmpeg', '-y',
            '-loop', '1',
            '-i', temp_path,
            '-c:v', 'h264_nvenc',
            '-preset', 'p4',
            '-t', str(duration),
            '-r', str(fps),
            '-pix_fmt', 'yuv420p',
            '-cq', str(quality),
            '-b:v', '5000k',
            '-maxrate', '7500k',
            '-bufsize', '10000k',
            output_path
        ]

        # Run FFmpeg
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

        # Monitor progress
        total_frames = duration * fps
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break

            if 'frame=' in line:
                try:
                    frame_str = line.split('frame=')[1].split()[0]
                    current_frame = int(frame_str)
                    progress = min(0.2 + (current_frame / total_frames) * 0.75, 0.95)
                    report_progress(progress, f"Encoding frame {current_frame}/{total_frames}")
                except Exception:
                    pass

        if process.returncode != 0:
            stderr = process.stderr.read()
            raise RuntimeError(f"FFmpeg failed: {stderr}")

        report_progress(1.0, "Complete")

        return {
            'duration': duration,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'total_frames': total_frames
        }

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor.py input_image output_video [duration] [fps]")
        sys.exit(1)

    test_config = {
        'duration': int(sys.argv[3]) if len(sys.argv) > 3 else 10,
        'fps': int(sys.argv[4]) if len(sys.argv) > 4 else 24,
        'quality': 23
    }

    def progress(p, msg):
        print(f"[{int(p*100)}%] {msg}")

    result = process_pic_to_video(sys.argv[1], sys.argv[2], test_config, progress)
    print(f"Result: {result}")
