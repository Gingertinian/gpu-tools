"""
Vignettes Processor - Video overlay effects

Applies vignette overlays, borders, and visual effects to videos.
Uses NVENC for hardware-accelerated encoding.

Config structure:
{
    "overlayType": "vignette" | "border" | "frame" | "blur_edges",
    "intensity": 0-100,
    "color": "#hexcolor",
    "borderWidth": pixels,
    "cornerRadius": pixels,
    "blurAmount": 0-100,
    "customOverlay": "url to overlay image"
}
"""

import os
import subprocess
import tempfile
from typing import Callable, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import json


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

        # Build FFmpeg command
        cmd = ['ffmpeg', '-y', '-hwaccel', 'cuda', '-i', input_path]

        if overlay_type != 'blur_edges' and os.path.exists(overlay_path):
            cmd.extend(['-i', overlay_path])

        filter_str = ';'.join(filters) if filters else None
        if filter_str:
            cmd.extend(['-filter_complex', filter_str])

        # NVENC encoding
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-preset', 'p4',
            '-b:v', '5000k',
            '-maxrate', '7500k',
            '-bufsize', '10000k',
            '-profile:v', 'high',
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path
        ])

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
                        progress = min(0.4 + (current_time / duration) * 0.55, 0.95)
                        report_progress(progress, f"Encoding... {int(current_time)}s / {int(duration)}s")
                except:
                    pass

        if process.returncode != 0:
            stderr = process.stderr.read()
            raise RuntimeError(f"FFmpeg failed: {stderr}")

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
        except:
            pass


def get_video_info(path: str) -> Dict[str, Any]:
    """Get video metadata using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-show_format', path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
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
        except:
            fps = 30

        return {
            'width': int(video_stream.get('width', 1920)),
            'height': int(video_stream.get('height', 1080)),
            'fps': fps,
            'duration': float(info.get('format', {}).get('duration', 0))
        }
    except:
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
    except:
        return (0, 0, 0)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor.py input_video output_video")
        sys.exit(1)

    test_config = {
        'overlayType': 'vignette',
        'intensity': 50,
        'color': '#000000'
    }

    def progress(p, msg):
        print(f"[{int(p*100)}%] {msg}")

    result = process_vignettes(sys.argv[1], sys.argv[2], test_config, progress)
    print(f"Result: {result}")
