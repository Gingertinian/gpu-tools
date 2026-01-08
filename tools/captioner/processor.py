"""
Captioner Processor - Add text captions to images and videos

Adds customizable text overlays, captions, and watermarks to media.
Supports multiple fonts, positions, styles, and animations.

Config structure:
{
    "text": "Caption text",
    "position": "top" | "center" | "bottom" | "custom",
    "customX": pixels,
    "customY": pixels,
    "fontSize": pixels,
    "fontFamily": "Arial" | "Impact" | "Roboto" | etc,
    "fontWeight": "normal" | "bold",
    "color": "#hexcolor",
    "backgroundColor": "#hexcolor" | null,
    "backgroundOpacity": 0-100,
    "strokeColor": "#hexcolor" | null,
    "strokeWidth": pixels,
    "shadow": boolean,
    "animation": "none" | "fade" | "slide" | "typewriter",
    "startTime": seconds,
    "endTime": seconds | null (full duration),
    "maxWidth": pixels | null (auto)
}
"""

import os
import subprocess
import tempfile
from typing import Callable, Optional, Dict, Any, List
from PIL import Image, ImageDraw, ImageFont
import json
import textwrap


# Font mapping (RunPod will have these installed)
FONT_MAP = {
    'Arial': '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    'Arial Bold': '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    'Impact': '/usr/share/fonts/truetype/msttcorefonts/Impact.ttf',
    'Roboto': '/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf',
    'Roboto Bold': '/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf',
    'default': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'default-bold': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
}


def process_captioner(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process image or video with text captions.

    Args:
        input_path: Path to input file
        output_path: Path to output file
        config: Caption configuration
        progress_callback: Optional callback(progress: 0-1, message: str)

    Returns:
        Dict with processing results
    """

    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    # Detect file type
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv']

    report_progress(0.05, "Analyzing file...")

    if is_video:
        return process_video_caption(input_path, output_path, config, report_progress)
    else:
        return process_image_caption(input_path, output_path, config, report_progress)


def process_image_caption(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Add caption to a single image."""

    report_progress(0.1, "Loading image...")

    img = Image.open(input_path).convert('RGBA')
    width, height = img.size

    report_progress(0.3, "Rendering caption...")

    # Create caption overlay
    overlay = create_caption_overlay(width, height, config)

    # Composite
    img = Image.alpha_composite(img, overlay)

    report_progress(0.7, "Saving output...")

    # Convert to RGB for JPEG output
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext in ['.jpg', '.jpeg']:
        img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)
    elif output_ext == '.png':
        img.save(output_path, 'PNG')
    else:
        img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)

    report_progress(1.0, "Complete")

    return {
        'text': config.get('text', ''),
        'position': config.get('position', 'bottom'),
        'resolution': f"{width}x{height}"
    }


def process_video_caption(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Add caption to video using FFmpeg drawtext filter."""

    report_progress(0.1, "Analyzing video...")

    # Get video info
    video_info = get_video_info(input_path)
    width = video_info.get('width', 1920)
    height = video_info.get('height', 1080)
    duration = video_info.get('duration', 0)

    report_progress(0.2, "Building caption filter...")

    # Extract config
    text = config.get('text', '')
    position = config.get('position', 'bottom')
    font_size = config.get('fontSize', 48)
    font_family = config.get('fontFamily', 'Arial')
    font_weight = config.get('fontWeight', 'normal')
    color = config.get('color', '#FFFFFF')
    bg_color = config.get('backgroundColor')
    bg_opacity = config.get('backgroundOpacity', 80) / 100
    stroke_color = config.get('strokeColor', '#000000')
    stroke_width = config.get('strokeWidth', 2)
    shadow = config.get('shadow', True)
    animation = config.get('animation', 'none')
    start_time = config.get('startTime', 0)
    end_time = config.get('endTime')
    max_width = config.get('maxWidth')

    # Get font path
    font_key = f"{font_family} Bold" if font_weight == 'bold' else font_family
    font_path = FONT_MAP.get(font_key, FONT_MAP.get(font_family, FONT_MAP['default']))

    if font_weight == 'bold' and font_key not in FONT_MAP:
        font_path = FONT_MAP.get('default-bold', FONT_MAP['default'])

    # Escape text for FFmpeg
    text_escaped = escape_ffmpeg_text(text)

    # Calculate position
    x, y = calculate_position(position, width, height, font_size, config)

    # Build drawtext filter
    drawtext_parts = [
        f"text='{text_escaped}'",
        f"fontfile='{font_path}'",
        f"fontsize={font_size}",
        f"fontcolor={color}",
        f"x={x}",
        f"y={y}"
    ]

    # Add stroke/border
    if stroke_color and stroke_width > 0:
        drawtext_parts.append(f"borderw={stroke_width}")
        drawtext_parts.append(f"bordercolor={stroke_color}")

    # Add shadow
    if shadow:
        drawtext_parts.append("shadowcolor=black@0.5")
        drawtext_parts.append("shadowx=2")
        drawtext_parts.append("shadowy=2")

    # Add background box
    if bg_color:
        drawtext_parts.append(f"box=1")
        alpha_hex = hex(int(bg_opacity * 255))[2:].zfill(2)
        drawtext_parts.append(f"boxcolor={bg_color}@{bg_opacity:.2f}")
        drawtext_parts.append(f"boxborderw=10")

    # Add timing (enable/disable)
    if start_time > 0 or end_time:
        enable_expr = f"between(t,{start_time},{end_time or duration})"
        drawtext_parts.append(f"enable='{enable_expr}'")

    # Add animation
    if animation == 'fade':
        # Fade in over 0.5 seconds
        drawtext_parts.append(f"alpha='if(lt(t,{start_time + 0.5}),(t-{start_time})/0.5,1)'")
    elif animation == 'slide':
        # Slide up from bottom
        drawtext_parts.append(f"y='if(lt(t,{start_time + 0.3}),h-(h-{y})*(t-{start_time})/0.3,{y})'")

    drawtext_filter = f"drawtext={':'.join(drawtext_parts)}"

    report_progress(0.3, "Encoding with NVENC...")

    # Build FFmpeg command
    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda',
        '-i', input_path,
        '-vf', drawtext_filter,
        '-c:v', 'h264_nvenc',
        '-preset', 'p4',
        '-b:v', '5000k',
        '-maxrate', '7500k',
        '-bufsize', '10000k',
        '-profile:v', 'high',
        '-c:a', 'aac',
        '-b:a', '128k',
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
                    progress = min(0.3 + (current_time / duration) * 0.65, 0.95)
                    report_progress(progress, f"Encoding... {int(current_time)}s / {int(duration)}s")
            except:
                pass

    if process.returncode != 0:
        stderr = process.stderr.read()
        raise RuntimeError(f"FFmpeg failed: {stderr}")

    report_progress(1.0, "Complete")

    return {
        'text': text,
        'position': position,
        'animation': animation,
        'resolution': f"{width}x{height}",
        'duration': duration
    }


def create_caption_overlay(width: int, height: int, config: Dict[str, Any]) -> Image.Image:
    """Create a caption overlay for images."""

    # Extract config
    text = config.get('text', '')
    position = config.get('position', 'bottom')
    font_size = config.get('fontSize', 48)
    font_family = config.get('fontFamily', 'Arial')
    font_weight = config.get('fontWeight', 'normal')
    color = config.get('color', '#FFFFFF')
    bg_color = config.get('backgroundColor')
    bg_opacity = config.get('backgroundOpacity', 80) / 100
    stroke_color = config.get('strokeColor', '#000000')
    stroke_width = config.get('strokeWidth', 2)
    shadow = config.get('shadow', True)
    max_width = config.get('maxWidth', width - 40)

    # Create transparent overlay
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Load font
    font_key = f"{font_family} Bold" if font_weight == 'bold' else font_family
    font_path = FONT_MAP.get(font_key, FONT_MAP.get(font_family, FONT_MAP['default']))

    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # Wrap text
    wrapped_text = wrap_text(text, font, max_width)

    # Calculate text bounds
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # Calculate position
    x, y = calculate_position_pixels(position, width, height, text_width, text_height, config)

    # Draw background
    if bg_color:
        padding = 10
        bg_r, bg_g, bg_b = hex_to_rgb(bg_color)
        bg_alpha = int(bg_opacity * 255)
        draw.rectangle(
            [x - padding, y - padding, x + text_width + padding, y + text_height + padding],
            fill=(bg_r, bg_g, bg_b, bg_alpha)
        )

    # Draw shadow
    if shadow:
        draw.multiline_text((x + 2, y + 2), wrapped_text, font=font, fill=(0, 0, 0, 128))

    # Draw stroke
    if stroke_color and stroke_width > 0:
        stroke_r, stroke_g, stroke_b = hex_to_rgb(stroke_color)
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx != 0 or dy != 0:
                    draw.multiline_text((x + dx, y + dy), wrapped_text, font=font, fill=(stroke_r, stroke_g, stroke_b, 255))

    # Draw text
    text_r, text_g, text_b = hex_to_rgb(color)
    draw.multiline_text((x, y), wrapped_text, font=font, fill=(text_r, text_g, text_b, 255))

    return overlay


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    """Wrap text to fit within max width."""
    lines = []
    for line in text.split('\n'):
        if font.getlength(line) <= max_width:
            lines.append(line)
        else:
            words = line.split()
            current_line = []
            for word in words:
                test_line = ' '.join(current_line + [word])
                if font.getlength(test_line) <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))

    return '\n'.join(lines)


def calculate_position(position: str, width: int, height: int, font_size: int, config: Dict[str, Any]) -> tuple:
    """Calculate FFmpeg position expressions."""
    padding = 20

    if position == 'top':
        return f'(w-text_w)/2', str(padding)
    elif position == 'center':
        return f'(w-text_w)/2', f'(h-text_h)/2'
    elif position == 'bottom':
        return f'(w-text_w)/2', f'h-text_h-{padding}'
    elif position == 'custom':
        x = config.get('customX', width // 2)
        y = config.get('customY', height - 100)
        return str(x), str(y)
    else:
        return f'(w-text_w)/2', f'h-text_h-{padding}'


def calculate_position_pixels(position: str, width: int, height: int, text_width: int, text_height: int, config: Dict[str, Any]) -> tuple:
    """Calculate pixel position for image captions."""
    padding = 20

    if position == 'top':
        return (width - text_width) // 2, padding
    elif position == 'center':
        return (width - text_width) // 2, (height - text_height) // 2
    elif position == 'bottom':
        return (width - text_width) // 2, height - text_height - padding
    elif position == 'custom':
        x = config.get('customX', (width - text_width) // 2)
        y = config.get('customY', height - text_height - padding)
        return x, y
    else:
        return (width - text_width) // 2, height - text_height - padding


def escape_ffmpeg_text(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter."""
    # Escape single quotes, colons, and backslashes
    text = text.replace('\\', '\\\\')
    text = text.replace("'", "\\'")
    text = text.replace(':', '\\:')
    text = text.replace('%', '%%')
    return text


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

        return {
            'width': int(video_stream.get('width', 1920)),
            'height': int(video_stream.get('height', 1080)),
            'duration': float(info.get('format', {}).get('duration', 0))
        }
    except:
        return {'width': 1920, 'height': 1080, 'duration': 0}


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])

    try:
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    except:
        return (255, 255, 255)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor.py input_file output_file")
        sys.exit(1)

    test_config = {
        'text': 'Sample Caption Text',
        'position': 'bottom',
        'fontSize': 48,
        'fontFamily': 'Arial',
        'color': '#FFFFFF',
        'strokeColor': '#000000',
        'strokeWidth': 2,
        'shadow': True
    }

    def progress(p, msg):
        print(f"[{int(p*100)}%] {msg}")

    result = process_captioner(sys.argv[1], sys.argv[2], test_config, progress)
    print(f"Result: {result}")
