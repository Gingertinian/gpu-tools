"""
Spoofer Processor - Full Port from Desktop App
Complete image/video transformation for pHash evasion with batch support.

Based on original Spoofer.py with all transforms:
- 9 Spatial transforms (★★★★★ Maximum pHash impact)
- 7 Tonal transforms (★★★★☆ DCT coefficients)
- 4 Visual transforms (★★☆☆☆ Visual variation)
- Full video processing with FFmpeg + NVENC
- Batch processing with ZIP output
- PARALLEL VIDEO PROCESSING: Multiple NVENC sessions for datacenter GPUs
"""

import os
import io
import random
import math
import time
import hashlib
import zipfile
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from typing import Callable, Optional, Dict, Any, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
from collections import deque
import threading
import cv2
import tempfile
from pathlib import Path

# Import gpu_utils for FFmpeg command building (optional - graceful fallback)
try:
    from tools.gpu_utils import build_ffmpeg_command, get_optimal_nvenc_params
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False

# Optional pHash support
try:
    import imagehash
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False

# Constants
PHASH_MIN_DISTANCE = 10
TARGET_RESOLUTIONS = {
    'high': (1080, 1920),
    'low': (720, 1280)
}

# Video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'}

# NVENC session limits by GPU type (per GPU)
# Consumer GPUs (GeForce): Limited to 3-5 sessions
# Datacenter GPUs (A-series, Quadro): Unlimited sessions
NVENC_SESSION_LIMITS = {
    'consumer': 3,      # RTX 3090, 4090, 4080, etc.
    'datacenter': 16,   # A5000, A6000, A100, etc. (increased for max parallel throughput)
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


def extract_videos_from_zip(zip_path: str, extract_dir: str) -> List[str]:
    """
    Extract video files from a ZIP archive.
    Returns list of paths to extracted video files.
    """
    video_paths = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            ext = os.path.splitext(name)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                # Extract to temp dir
                extracted_path = zf.extract(name, extract_dir)
                video_paths.append(extracted_path)
    return video_paths


# =============================================================================
# FRAME-BY-FRAME VIDEO PROCESSING WITH BLUR + LOGO
# =============================================================================

def process_video_with_blur_logo(
    input_path: str,
    output_path: str,
    config: dict,
    py_rng: random.Random,
    gpu_id: int = 0
) -> dict:
    """
    Process video frame-by-frame with blur background and logo overlay.
    Used when blur or logo is requested in spoofer config.

    Features:
    - Blur background zones (top/bottom) from video content
    - Logo overlay
    - All spoofer spatial/tonal transforms
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check for audio
    has_audio = _check_video_audio(str(input_path))

    print(f"[Spoofer+Blur] Input: {orig_w}x{orig_h} @ {fps:.2f} fps, {total_frames} frames")

    # Parse config
    spatial = config.get('spatial', {})
    tonal = config.get('tonal', {})
    visual = config.get('visual', {})
    video_cfg = config.get('video', {})

    # Blur config
    blur_enabled = config.get('blurBackground', True)
    blur_intensity = config.get('blurIntensity', 25)

    # Logo config
    logo_name = config.get('logoName', 'farmium_full')
    logo_size = config.get('logoSize', 15)

    # Output dimensions (9:16 by default for spoofer)
    final_w, final_h = 1080, 1920

    # Calculate layout - ALWAYS use 25% top and 25% bottom blur zones
    # This ensures blur is visible regardless of original aspect ratio
    blur_top = int(final_h * 0.25)  # 480 pixels
    blur_bottom = int(final_h * 0.25)  # 480 pixels
    content_h = final_h - blur_top - blur_bottom  # 960 pixels (50% of frame)

    # Scale content to fit in the middle 50% section
    # First try scaling by width
    scale = final_w / orig_w
    scaled_w = final_w
    scaled_h = int(orig_h * scale)
    scaled_h = scaled_h - (scaled_h % 2)  # Ensure even

    # If scaled height exceeds content area, scale by height instead
    if scaled_h > content_h:
        scale = content_h / orig_h
        scaled_h = content_h
        scaled_w = int(orig_w * scale)
        scaled_w = scaled_w - (scaled_w % 2)  # Ensure even

    # Center content horizontally and vertically within content area
    content_x = (final_w - scaled_w) // 2
    content_y = blur_top + (content_h - scaled_h) // 2

    print(f"[Spoofer+Blur] Output: {final_w}x{final_h}, blur_top={blur_top}, blur_bottom={blur_bottom}, content_h={content_h}")

    # Prepare logo
    logo_data = None
    if logo_name and logo_name != 'none':
        logo_data = _prepare_spoofer_logo(logo_name, final_w, logo_size)

    # Get spoofer transforms
    rotation_angle = 0
    if spatial.get('rotation', 0) > 0:
        rotation_angle = py_rng.uniform(-spatial['rotation'], spatial['rotation'])

    # Color adjustments - scale config values (0-100) to proper ranges
    brightness_adj = 0
    if tonal.get('brightness', 0) > 0:
        # Scale: config 5 -> ±0.05 brightness adjustment
        brightness_adj = py_rng.uniform(-tonal['brightness'], tonal['brightness']) * 0.01

    contrast_adj = 1.0
    if tonal.get('contrast', 0) > 0:
        # Scale: config 5 -> ±0.05 contrast variation (0.95 to 1.05)
        contrast_adj = 1 + py_rng.uniform(-tonal['contrast'], tonal['contrast']) * 0.01

    saturation_adj = 1.0
    if tonal.get('saturation', 0) > 0:
        # Scale: config 5 -> ±0.05 saturation variation (0.95 to 1.05)
        saturation_adj = 1 + py_rng.uniform(-tonal['saturation'], tonal['saturation']) * 0.01

    # Start FFmpeg writer
    ffmpeg_process = _start_spoofer_ffmpeg(
        str(output_path), final_w, final_h, fps,
        str(input_path) if has_audio else None,
        gpu_id
    )

    # Pre-allocate buffers
    output_frame = np.zeros((final_h, final_w, 3), dtype=np.uint8)

    # Blur params (animate every 20 frames)
    blur_update_interval = 20
    blur_params = _generate_spoofer_blur_params(py_rng)

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update blur params periodically
            if frame_idx % blur_update_interval == 0:
                blur_params = _generate_spoofer_blur_params(py_rng)

            # Apply rotation with zoom compensation
            if rotation_angle != 0:
                frame = _apply_rotation_with_zoom(frame, rotation_angle)

            # Apply color adjustments
            if brightness_adj != 0 or contrast_adj != 1.0 or saturation_adj != 1.0:
                frame = _apply_spoofer_color_adj(frame, brightness_adj, saturation_adj, contrast_adj)

            # Scale content
            content = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

            # Create blur zones from CENTER of content (always visible with 25% top/bottom)
            if blur_enabled:
                # Top blur zone - uses center of content
                blur_zone_top = _create_spoofer_blur_zone(
                    content, final_w, blur_top, 'center', blur_intensity, blur_params
                )
                output_frame[0:blur_top, :] = blur_zone_top

                # Bottom blur zone - uses center of content
                blur_zone_bottom = _create_spoofer_blur_zone(
                    content, final_w, blur_bottom, 'center', blur_intensity, blur_params
                )
                output_frame[final_h - blur_bottom:final_h, :] = blur_zone_bottom

            # Place main content in the middle section
            output_frame[content_y:content_y + scaled_h, content_x:content_x + scaled_w] = content

            # Apply logo
            if logo_data is not None:
                _apply_spoofer_logo(output_frame, logo_data, final_w, final_h)

            # Write frame
            try:
                ffmpeg_process.stdin.write(output_frame.tobytes())
            except BrokenPipeError:
                break

            frame_idx += 1

    finally:
        cap.release()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    output_size = os.path.getsize(output_path) if output_path.exists() else 0
    duration = frame_idx / fps if fps > 0 else 0

    return {
        "status": "completed",
        "outputPath": str(output_path),
        "outputSize": output_size,
        "framesProcessed": frame_idx,
        "duration": duration
    }


def _check_video_audio(video_path: str) -> bool:
    """Check if video has audio."""
    try:
        cmd = ['ffprobe', '-v', 'quiet', '-select_streams', 'a',
               '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return 'audio' in result.stdout.lower()
    except:
        return True


def _apply_rotation_with_zoom(frame: np.ndarray, angle_deg: float) -> np.ndarray:
    """Apply rotation with zoom compensation to eliminate black borders."""
    h, w = frame.shape[:2]

    # Calculate the expanded canvas size after rotation
    angle_rad = abs(angle_deg) * math.pi / 180
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    # New bounding box after rotation
    new_w = int(w * cos_a + h * sin_a) + 2
    new_h = int(w * sin_a + h * cos_a) + 2

    # Create rotation matrix for expanded canvas
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # Adjust translation for new canvas center
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Rotate into expanded canvas (no black corners)
    rotated = cv2.warpAffine(frame, M, (new_w, new_h), flags=cv2.INTER_LANCZOS4)

    # Calculate inscribed rectangle (largest rect without black borders)
    if sin_a < 0.001:  # Nearly zero rotation
        return frame

    # Scale factor for inscribed rectangle
    scale = min(
        1.0 / (cos_a + sin_a * h / w),
        1.0 / (cos_a + sin_a * w / h)
    )

    crop_w = int(w * scale)
    crop_h = int(h * scale)

    # Center crop from rotated image
    left = (new_w - crop_w) // 2
    top = (new_h - crop_h) // 2

    cropped = rotated[top:top+crop_h, left:left+crop_w]

    # Resize back to original dimensions
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)


def _apply_spoofer_color_adj(frame: np.ndarray, brightness: float, saturation: float, contrast: float) -> np.ndarray:
    """Apply color adjustments to frame.

    Args:
        frame: BGR image (uint8)
        brightness: Brightness adjustment (-1.0 to 1.0, 0 = no change)
        saturation: Saturation multiplier (1.0 = no change)
        contrast: Contrast multiplier (1.0 = no change)

    Returns:
        Adjusted BGR image (uint8)
    """
    result = frame.copy()

    # Apply brightness and saturation in HSV space
    if brightness != 0 or saturation != 1.0:
        # Use HSV_FULL for full 8-bit range (H: 0-255 instead of 0-180)
        # This prevents hue wrapping issues that can cause color inversion
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV_FULL)

        # Work with float for precision
        h, s, v = cv2.split(hsv)
        s = s.astype(np.float32)
        v = v.astype(np.float32)

        # Apply saturation adjustment
        if saturation != 1.0:
            s = np.clip(s * saturation, 0, 255)

        # Apply brightness adjustment (additive for small values, multiplicative causes issues)
        if brightness != 0:
            # For small brightness values (±0.05 range), use additive adjustment
            # brightness of 0.05 = +12.75 to V channel (out of 255)
            v = np.clip(v + brightness * 255, 0, 255)

        # Merge back and convert
        hsv = cv2.merge([h, s.astype(np.uint8), v.astype(np.uint8)])
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

    # Apply contrast adjustment in BGR space
    if contrast != 1.0:
        # convertScaleAbs: output = |alpha * input + beta|
        # For contrast, we want to scale around middle gray (128)
        result = cv2.convertScaleAbs(result, alpha=contrast, beta=128 * (1 - contrast))

    return result


def _generate_spoofer_blur_params(rng: random.Random) -> dict:
    """Generate random blur parameters."""
    return {
        'hflip': rng.random() < 0.3,
        'vflip': rng.random() < 0.2,
        'color_shift': rng.randint(-15, 15),
        'saturation': rng.uniform(0.7, 1.2),
        'brightness': rng.uniform(0.6, 0.9),
        'tilt_angle': rng.uniform(-8, 8),
        'zoom': rng.uniform(1.2, 1.5),
        'darken': rng.uniform(0.5, 0.8),
    }


def _create_spoofer_blur_zone(
    content: np.ndarray,
    target_w: int,
    target_h: int,
    position: str,
    blur_strength: int,
    params: dict
) -> np.ndarray:
    """Create blur zone from video content.

    Args:
        content: The scaled video frame
        target_w: Width of the blur zone (should match output width)
        target_h: Height of the blur zone
        position: 'top', 'bottom', or 'center' - where to extract source content from
        blur_strength: Gaussian blur kernel size
        params: Random parameters for variety (zoom, flip, darken, etc.)
    """
    if target_h <= 0:
        return np.zeros((0, target_w, 3), dtype=np.uint8)

    content_h, content_w = content.shape[:2]

    # Determine source region based on position
    # Use 65% of content height as source, taken from the specified position
    source_ratio = 0.65
    source_h = max(int(content_h * source_ratio), min(content_h, max(target_h * 2, 100)))
    source_h = max(10, min(source_h, content_h))

    if position == 'top':
        # Extract from top of content
        source = content[0:source_h, :].copy()
    elif position == 'bottom':
        # Extract from bottom of content
        start_y = max(0, content_h - source_h)
        source = content[start_y:, :].copy()
    else:  # 'center' - extract from center of content
        # Center the source region vertically
        center_y = content_h // 2
        start_y = max(0, center_y - source_h // 2)
        end_y = min(content_h, start_y + source_h)
        source = content[start_y:end_y, :].copy()

    # Zoom to fill the target dimensions
    zoom = max(target_w / source.shape[1], target_h / source.shape[0]) * params['zoom']
    zoomed_w = int(source.shape[1] * zoom)
    zoomed_h = int(source.shape[0] * zoom)

    if zoomed_w <= 0 or zoomed_h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    source = cv2.resize(source, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)

    # Center crop to target dimensions
    crop_x = max(0, (zoomed_w - target_w) // 2)
    crop_y = max(0, (zoomed_h - target_h) // 2)
    blur_section = source[crop_y:crop_y + target_h, crop_x:crop_x + target_w]

    if blur_section.shape[0] != target_h or blur_section.shape[1] != target_w:
        blur_section = cv2.resize(blur_section, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Flip for variety
    if params['hflip']:
        blur_section = cv2.flip(blur_section, 1)
    if params['vflip']:
        blur_section = cv2.flip(blur_section, 0)

    # Darken
    blur_section = (blur_section.astype(np.float32) * params['darken']).astype(np.uint8)

    # Apply blur
    if blur_strength > 0:
        k = int(blur_strength) | 1
        blur_section = cv2.GaussianBlur(blur_section, (k, k), 0)

    return blur_section


def _svg_to_cv2(svg_path: str, target_width: int):
    """
    Convert SVG to CV2 BGRA image.
    Tries multiple methods: cairosvg, svglib, or fallback to placeholder.
    """
    # Method 1: Try cairosvg (best quality, available on RunPod)
    try:
        import cairosvg
        import io
        from PIL import Image
        # Render SVG to PNG at target size
        png_data = cairosvg.svg2png(url=svg_path, output_width=target_width * 2)
        pil_img = Image.open(io.BytesIO(png_data)).convert('RGBA')
        # Convert PIL RGBA to CV2 BGRA
        img_array = np.array(pil_img)
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
    except Exception as e:
        print(f"[Spoofer] cairosvg failed: {e}")

    # Method 2: Try svglib + reportlab
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        import io
        from PIL import Image
        drawing = svg2rlg(svg_path)
        if drawing:
            scale = (target_width * 2) / drawing.width
            drawing.width *= scale
            drawing.height *= scale
            drawing.scale(scale, scale)
            png_data = io.BytesIO()
            renderPM.drawToFile(drawing, png_data, fmt='PNG')
            png_data.seek(0)
            pil_img = Image.open(png_data).convert('RGBA')
            img_array = np.array(pil_img)
            return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
    except Exception as e:
        print(f"[Spoofer] svglib failed: {e}")

    # Method 3: Fallback - create a placeholder logo
    print(f"[Spoofer] Using placeholder logo (SVG conversion not available)")
    from PIL import Image, ImageDraw, ImageFont

    logo_h = int(target_width * 0.3)
    logo = Image.new('RGBA', (target_width, logo_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo)

    try:
        font = ImageFont.truetype("arial.ttf", logo_h // 2)
    except:
        font = ImageFont.load_default()

    text = "FARMIUM"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (target_width - text_w) // 2
    y = (logo_h - text_h) // 2
    draw.text((x, y), text, fill=(255, 255, 255, 230), font=font)

    img_array = np.array(logo)
    return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)


def _prepare_spoofer_logo(logo_name: str, video_width: int, size_percent: float):
    """Prepare logo for overlay. Supports PNG and SVG."""
    try:
        script_dir = Path(__file__).parent.resolve()
        search_paths = [
            Path('/workspace/assets/logos'),
            script_dir / 'assets' / 'logos',
            script_dir.parent / 'assets' / 'logos',
            script_dir.parent.parent / 'assets' / 'logos',
        ]

        logo_source = None
        is_svg = False

        if logo_name in ['farmium_icon', 'farmium_full']:
            for search_path in search_paths:
                # Try PNG first
                png_path = search_path / f'{logo_name}.png'
                if png_path.exists():
                    logo_source = str(png_path)
                    print(f"[Spoofer] Found PNG logo: {logo_source}")
                    break
                # Try SVG fallback
                svg_path = search_path / f'{logo_name}.svg'
                if svg_path.exists():
                    logo_source = str(svg_path)
                    is_svg = True
                    print(f"[Spoofer] Found SVG logo: {logo_source}")
                    break

        if not logo_source:
            print(f"[Spoofer] Logo not found: {logo_name}")
            return None

        target_width = max(int(video_width * size_percent / 100), 80)

        # Load logo (PNG or convert SVG)
        if is_svg:
            logo_img = _svg_to_cv2(logo_source, target_width)
            if logo_img is None:
                return None
        else:
            logo_img = cv2.imread(logo_source, cv2.IMREAD_UNCHANGED)
            if logo_img is None:
                return None

        # Resize if needed
        logo_w = target_width
        logo_h = int(logo_w * logo_img.shape[0] / logo_img.shape[1])
        if logo_img.shape[1] != logo_w:
            logo_img = cv2.resize(logo_img, (logo_w, logo_h), interpolation=cv2.INTER_LANCZOS4)

        # Split alpha
        if logo_img.shape[2] == 4:
            logo_bgr = logo_img[:, :, :3]
            logo_alpha = logo_img[:, :, 3].astype(np.float32) / 255.0
        else:
            logo_bgr = logo_img
            logo_alpha = np.ones((logo_h, logo_w), dtype=np.float32)

        return {
            'image': logo_bgr,
            'alpha': logo_alpha,
            'alpha_3d': logo_alpha[:, :, np.newaxis]
        }
    except Exception as e:
        print(f"[Spoofer] Error preparing logo: {e}")
        import traceback
        traceback.print_exc()
        return None


def _apply_spoofer_logo(frame: np.ndarray, logo_data: dict, frame_w: int, frame_h: int):
    """Apply logo overlay to frame."""
    logo = logo_data['image']
    alpha_3d = logo_data['alpha_3d']
    lh, lw = logo.shape[:2]

    margin = int(frame_h * 0.05)
    x = (frame_w - lw) // 2
    y = frame_h - lh - margin

    x = max(0, min(frame_w - lw, x))
    y = max(0, min(frame_h - lh, y))

    roi = frame[y:y + lh, x:x + lw]
    blended = (alpha_3d * logo + (1 - alpha_3d) * roi).astype(np.uint8)
    frame[y:y + lh, x:x + lw] = blended


def _start_spoofer_ffmpeg(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: str = None,
    gpu_id: int = 0
) -> subprocess.Popen:
    """Start FFmpeg for frame-by-frame video writing."""
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(min(fps, 60)),
        '-thread_queue_size', '512',
        '-i', '-',
    ]

    if audio_source:
        cmd.extend(['-i', audio_source])

    # Try NVENC
    cmd.extend([
        '-c:v', 'h264_nvenc',
        '-gpu', str(gpu_id),
        '-preset', 'p4',
        '-rc', 'vbr',
        '-cq', '23',
        '-b:v', '8000k',
        '-maxrate', '12000k',
        '-bufsize', '16000k',
        '-pix_fmt', 'yuv420p',
        '-map', '0:v',
    ])

    if audio_source:
        cmd.extend(['-map', '1:a?', '-c:a', 'aac', '-b:a', '128k', '-shortest'])
    else:
        cmd.append('-an')

    cmd.extend(['-movflags', '+faststart', output_path])

    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except:
        # CPU fallback
        cmd_cpu = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24',
            '-r', str(min(fps, 60)), '-i', '-',
        ]
        if audio_source:
            cmd_cpu.extend(['-i', audio_source])
        cmd_cpu.extend(['-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-pix_fmt', 'yuv420p', '-map', '0:v'])
        if audio_source:
            cmd_cpu.extend(['-map', '1:a?', '-c:a', 'aac', '-b:a', '128k', '-shortest'])
        else:
            cmd_cpu.append('-an')
        cmd_cpu.extend(['-movflags', '+faststart', output_path])
        return subprocess.Popen(cmd_cpu, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# =============================================================================
# PARALLEL VIDEO PROCESSING
# =============================================================================

def process_single_video_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel video processing.
    Designed to be called from ThreadPoolExecutor (FFmpeg subprocess does the GPU work).
    Includes CPU fallback if NVENC fails.
    Supports multi-GPU workers via gpu_id parameter.

    Args tuple: (input_path, output_path, config, video_index, gpu_id)
    Returns: dict with status, output_path, and any errors
    """
    # Support both old (4-tuple) and new (5-tuple) format
    if len(args) == 5:
        input_path, output_path, config, video_index, gpu_id = args
    else:
        input_path, output_path, config, video_index = args
        gpu_id = 0  # Default to first GPU

    # Timing and GPU assignment logging
    start_time = time.time()
    print(f"[NVENC Worker {video_index}] Starting on GPU {gpu_id}")

    try:
        import json

        # Use nanosecond timestamp + pid + index for unique seed (prevents collisions)
        seed = int(time.time_ns()) ^ (os.getpid() << 16) ^ (video_index * 997)
        py_rng = random.Random(seed)

        # Check if blur/logo is requested - use frame-by-frame processing for these
        blur_enabled = config.get('blurBackground', False)
        logo_name = config.get('logoName', None)

        if blur_enabled or logo_name:
            print(f"[Spoofer Worker {video_index}] Using frame-by-frame processing (blur={blur_enabled}, logo={logo_name})")
            try:
                result = process_video_with_blur_logo(
                    input_path=input_path,
                    output_path=output_path,
                    config=config,
                    py_rng=py_rng,
                    gpu_id=gpu_id
                )
                elapsed = time.time() - start_time
                return {
                    'status': 'completed',
                    'index': video_index,
                    'output_path': output_path,
                    'duration': result.get('duration', 0),
                    'processing_time': elapsed
                }
            except Exception as blur_err:
                elapsed = time.time() - start_time
                print(f"[Spoofer Worker {video_index}] Blur+logo processing failed: {blur_err}")
                return {
                    'status': 'failed',
                    'index': video_index,
                    'error': f"Blur+logo processing failed: {str(blur_err)}"
                }

        # Get video info
        try:
            probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                        '-show_streams', '-show_format', input_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            video_info = json.loads(probe_result.stdout)
            video_stream = next((s for s in video_info.get('streams', [])
                               if s.get('codec_type') == 'video'), {})
            original_width = int(video_stream.get('width', 1920))
            original_height = int(video_stream.get('height', 1080))
            duration = float(video_info.get('format', {}).get('duration', 0))
        except Exception:
            original_width, original_height, duration = 1920, 1080, 0

        # Build filter chain
        filters = []
        spatial = config.get('spatial', {})
        tonal = config.get('tonal', {})
        visual = config.get('visual', {})
        video_cfg = config.get('video', {})

        # Spatial filters
        if spatial.get('crop', 0) > 0:
            crop_pct = spatial['crop'] / 100
            crop_w = int(original_width * (1 - crop_pct * 2))
            crop_h = int(original_height * (1 - crop_pct * 2))
            filters.append(f"crop={crop_w}:{crop_h}")

        if spatial.get('rotation', 0) > 0:
            angle_deg = py_rng.uniform(-spatial['rotation'], spatial['rotation'])
            angle_rad = angle_deg * math.pi / 180

            # Calculate zoom factor to eliminate black borders after rotation
            # Uses the inscribed rectangle formula based on aspect ratio
            abs_angle = abs(angle_rad)
            cos_a = math.cos(abs_angle)
            sin_a = math.sin(abs_angle)

            if sin_a > 0.001:
                # Zoom factor = inverse of inscribed rectangle scale
                # This ensures we scale up enough to crop away black corners
                zoom_w = cos_a + sin_a * (original_height / original_width)
                zoom_h = cos_a + sin_a * (original_width / original_height)
                zoom_factor = max(zoom_w, zoom_h)
                zoom_factor = max(1.0, min(zoom_factor, 2.0))

                # Scale up first, rotate, then crop to original dimensions
                # This removes black corners completely
                scaled_w = int(original_width * zoom_factor)
                scaled_h = int(original_height * zoom_factor)
                # Ensure even dimensions for video encoding
                scaled_w = scaled_w + (scaled_w % 2)
                scaled_h = scaled_h + (scaled_h % 2)

                # Pipeline: scale up → rotate → crop center
                filters.append(f"scale={scaled_w}:{scaled_h}:flags=lanczos")
                filters.append(f"rotate={angle_rad}:fillcolor=black")
                filters.append(f"crop={original_width}:{original_height}")
            else:
                # Nearly zero rotation - skip
                pass

        # Tonal filters
        eq_params = []
        if tonal.get('brightness', 0) > 0:
            b = py_rng.uniform(-tonal['brightness'], tonal['brightness']) * 0.4
            eq_params.append(f"brightness={b:.3f}")
        if tonal.get('contrast', 0) > 0:
            c = 1 + py_rng.uniform(-tonal['contrast'], tonal['contrast'])
            eq_params.append(f"contrast={c:.3f}")
        if tonal.get('saturation', 0) > 0:
            s = 1 + py_rng.uniform(-tonal['saturation'], tonal['saturation'])
            eq_params.append(f"saturation={s:.3f}")
        if tonal.get('gamma', 0) > 0:
            g = 1 + py_rng.uniform(-tonal['gamma'], tonal['gamma'])
            eq_params.append(f"gamma={g:.3f}")

        if eq_params:
            filters.append(f"eq={':'.join(eq_params)}")

        # Visual filters
        if visual.get('noise', 0) > 0:
            filters.append(f"noise=alls={visual['noise']*2}:allf=t")

        if tonal.get('vignette', 0) > 0:
            filters.append(f"vignette=PI/{3 + (1 - tonal['vignette']/100) * 4}")

        # Speed variation
        if video_cfg.get('speedVariation', 0) > 0:
            speed = 1 + py_rng.uniform(-video_cfg['speedVariation']/100, video_cfg['speedVariation']/100)
            speed = max(0.9, min(1.1, speed))
            filters.append(f"setpts={1/speed}*PTS")

        # FPS
        fps = 30
        if video_cfg.get('fpsVar', 0) > 0:
            fps = 30 + py_rng.uniform(-video_cfg['fpsVar'], video_cfg['fpsVar'])

        # Build filter string
        filter_str = ','.join(filters) if filters else None
        bitrate = int(5000 * video_cfg.get('bitrate', 90) / 100)
        keep_audio = video_cfg.get('keepAudio', True)

        # Build FFmpeg commands optimized for maximum GPU throughput
        # Key optimizations:
        # 1. Full hardware decode pipeline: -hwaccel cuda -hwaccel_device X -hwaccel_output_format cuda
        # 2. Use h264_cuvid decoder for H264 input (hardware decode)
        # 3. Fastest NVENC preset: p1 with -tune ll (low latency)
        # 4. Zero lookahead for minimum latency
        # 5. No B-frames for maximum encoding speed

        def _build_ffmpeg_cmd_max_throughput(use_nvenc: bool = True, target_gpu: int = 0) -> list:
            """Build FFmpeg command optimized for maximum GPU throughput."""
            cmd = ['ffmpeg', '-y']

            if use_nvenc:
                # FULL hardware acceleration pipeline - keeps frames in GPU memory
                cmd.extend([
                    '-hwaccel', 'cuda',
                    '-hwaccel_device', str(target_gpu),
                    '-hwaccel_output_format', 'cuda',  # Keep decoded frames in GPU memory
                    '-c:v', 'h264_cuvid',  # Hardware H264 decoder (if input is H264)
                ])

            cmd.extend(['-i', input_path])

            if filter_str:
                # For hardware pipeline, need to download to CPU for filters, then upload back
                # Use hwdownload -> filter -> hwupload for GPU memory efficiency
                # However, most spoofer filters are simple and work better on CPU
                cmd.extend(['-vf', f'hwdownload,format=nv12,{filter_str},hwupload_cuda'])
            else:
                # No filters - can stay entirely on GPU
                pass

            if use_nvenc:
                # NVENC encoding with MAXIMUM THROUGHPUT settings
                # p1 = fastest preset, ll = low latency tuning
                # No lookahead, no B-frames for minimum latency
                cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-gpu', str(target_gpu),  # Multi-GPU: select specific GPU for NVENC
                    '-preset', 'p1',          # Fastest NVENC preset (p1-p7)
                    '-tune', 'll',            # Low latency tuning (faster than hq)
                    '-rc', 'vbr',             # Variable bitrate
                    '-cq', '26',              # Slightly relaxed quality for speed
                    '-b:v', f'{bitrate}k',
                    '-maxrate', f'{int(bitrate * 2)}k',  # 2x headroom for peaks
                    '-bufsize', f'{bitrate * 2}k',
                    '-rc-lookahead', '0',     # Zero lookahead for minimum latency
                    '-bf', '0',               # No B-frames for maximum speed
                    '-spatial-aq', '0',       # Disable spatial AQ for speed
                    '-temporal-aq', '0',      # Disable temporal AQ for speed
                ])
            else:
                # CPU fallback with libx264 ultrafast
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',   # Fastest x264 preset
                    '-crf', '23',
                    '-pix_fmt', 'yuv420p',
                ])

            if keep_audio:
                cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
            else:
                cmd.append('-an')

            cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])
            return cmd

        # Also build a simpler NVENC command without hardware decode (fallback if cuvid fails)
        def _build_ffmpeg_cmd_nvenc_simple(target_gpu: int = 0) -> list:
            """Build simpler NVENC command without hardware decode (fallback)."""
            cmd = ['ffmpeg', '-y']

            # Software decode but hardware encode
            cmd.extend([
                '-hwaccel', 'cuda',
                '-hwaccel_device', str(target_gpu),
            ])

            cmd.extend(['-i', input_path])

            if filter_str:
                cmd.extend(['-vf', filter_str])

            # NVENC with fast settings
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-gpu', str(target_gpu),
                '-preset', 'p1',
                '-tune', 'll',
                '-rc', 'vbr',
                '-cq', '26',
                '-b:v', f'{bitrate}k',
                '-maxrate', f'{int(bitrate * 2)}k',
                '-bufsize', f'{bitrate * 2}k',
                '-rc-lookahead', '0',
                '-bf', '0',
            ])

            if keep_audio:
                cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
            else:
                cmd.append('-an')

            cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])
            return cmd

        def _build_ffmpeg_cmd_cpu() -> list:
            """Build CPU fallback command."""
            cmd = ['ffmpeg', '-y', '-i', input_path]

            if filter_str:
                cmd.extend(['-vf', filter_str])

            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
            ])

            if keep_audio:
                cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
            else:
                cmd.append('-an')

            cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])
            return cmd

        nvenc_cmd = _build_ffmpeg_cmd_max_throughput(use_nvenc=True, target_gpu=gpu_id)
        nvenc_simple_cmd = _build_ffmpeg_cmd_nvenc_simple(target_gpu=gpu_id)
        cpu_cmd = _build_ffmpeg_cmd_cpu()

        # Try encoding with 3-tier fallback for maximum reliability:
        # 1. Full hardware pipeline (h264_cuvid + NVENC) - fastest
        # 2. Simple NVENC (software decode + NVENC) - if cuvid fails
        # 3. CPU fallback (libx264) - if NVENC fails entirely
        print(f"[Spoofer Worker {video_index}] Using GPU {gpu_id} with full hardware pipeline")

        # NOTE: Do NOT set CUDA_VISIBLE_DEVICES - it remaps GPU indices and breaks -gpu X flag
        # The -gpu X flag in h264_nvenc is sufficient to select the specific GPU

        # Tier 1: Try full hardware pipeline (h264_cuvid decoder + NVENC encoder)
        process = subprocess.run(nvenc_cmd, capture_output=True, text=True, timeout=900)

        if process.returncode != 0:
            # Tier 2: Full hardware failed (likely input not H264), try simple NVENC
            print(f"[Spoofer Worker {video_index}] Full hardware pipeline failed, trying simple NVENC")
            process = subprocess.run(nvenc_simple_cmd, capture_output=True, text=True, timeout=900)

            if process.returncode != 0:
                # Tier 3: NVENC failed entirely, use CPU fallback
                print(f"[Spoofer Worker {video_index}] NVENC on GPU {gpu_id} failed, trying CPU fallback")
                process = subprocess.run(cpu_cmd, capture_output=True, text=True, timeout=1800)

                if process.returncode != 0:
                    elapsed = time.time() - start_time
                    print(f"[NVENC Worker {video_index}] FAILED in {elapsed:.2f}s on GPU {gpu_id}")
                    return {
                        'status': 'failed',
                        'index': video_index,
                        'error': f"All encoders failed: {process.stderr[-500:] if process.stderr else 'Unknown error'}"
                    }

        elapsed = time.time() - start_time
        print(f"[NVENC Worker {video_index}] Done in {elapsed:.2f}s on GPU {gpu_id}")

        return {
            'status': 'completed',
            'index': video_index,
            'output_path': output_path,
            'duration': duration,
            'processing_time': elapsed
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"[NVENC Worker {video_index}] TIMEOUT after {elapsed:.2f}s on GPU {gpu_id}")
        return {
            'status': 'failed',
            'index': video_index,
            'error': 'FFmpeg process timed out'
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[NVENC Worker {video_index}] ERROR after {elapsed:.2f}s on GPU {gpu_id}: {str(e)}")
        return {
            'status': 'failed',
            'index': video_index,
            'error': str(e)
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


def process_videos_parallel(
    video_paths: List[str],
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    max_parallel: int = None,
    variations: int = 1
) -> Dict[str, Any]:
    """
    Process multiple videos in parallel using multiple NVENC sessions.

    Supports VARIATIONS: Each input video can generate N unique variations,
    each with different random transforms (crop, rotation, brightness, etc.)

    Example: 20 videos × 100 variations = 2,000 unique output videos

    Uses ThreadPoolExecutor instead of ProcessPoolExecutor because:
    - FFmpeg runs as subprocess and does the actual GPU work
    - Avoids overhead of spawning Python processes
    - ThreadPoolExecutor is lighter weight for I/O-bound work

    Args:
        video_paths: List of input video paths
        output_dir: Directory for output videos
        config: Processing configuration
        progress_callback: Optional progress callback
        max_parallel: Max parallel threads (auto-detected if None)
        variations: Number of unique variations per input video (default 1)

    Returns:
        Dict with results including processed count and any errors
    """
    if not video_paths:
        return {'error': 'No videos to process'}

    # Ensure variations is at least 1
    variations = max(1, variations)

    # Auto-detect parallel limit based on GPU
    gpu_info = get_gpu_info()
    gpu_count = gpu_info['gpu_count']
    sessions_per_gpu = NVENC_SESSION_LIMITS.get(gpu_info['gpu_type'], NVENC_SESSION_LIMITS['default'])

    if max_parallel is None:
        max_parallel = gpu_info['nvenc_sessions']

    # ==========================================================================
    # MEGA-GPU WORKER SCALING (optimized for 1-20+ GPUs per worker)
    # ==========================================================================
    #
    # Architecture: Fewer workers with MORE GPUs each is BETTER because:
    #   1. Less network overhead between workers
    #   2. All GPUs share same memory/storage (faster transfers)
    #   3. Better for batch processing (single download, parallel encode)
    #   4. RunPod worker allocation is the bottleneck, not GPUs
    #
    # Scaling strategy:
    #   - Workers = NVENC_sessions × 1.2 (small buffer, semaphores handle blocking)
    #   - NO CAP - if you have 20 GPUs × 16 sessions = 320 NVENC, use 384 workers
    #   - Semaphores enforce REAL limits per GPU (no over-subscription)
    #
    # Example scaling:
    #   1 GPU:  16 sessions × 1.2 = 19 workers
    #   4 GPUs: 64 sessions × 1.2 = 77 workers
    #   9 GPUs: 144 sessions × 1.2 = 173 workers
    #   20 GPUs: 320 sessions × 1.2 = 384 workers
    #
    # The 1.2x buffer ensures:
    #   - Always work queued when NVENC session finishes
    #   - Minimal thread overhead (not 2x anymore)
    #   - Semaphores block excess threads efficiently

    # Calculate optimal worker count - NO CAP for mega-GPU workers
    base_workers = max_parallel  # = gpu_count × sessions_per_gpu

    # Small buffer (1.2x) - semaphores handle the rest
    # This is enough to keep GPUs fed without excessive thread overhead
    WORKER_BUFFER_MULTIPLIER = 1.2
    max_workers = int(base_workers * WORKER_BUFFER_MULTIPLIER)

    # Minimum workers = NVENC sessions (ensure we can saturate all GPUs)
    max_workers = max(max_workers, base_workers)

    # Calculate theoretical throughput
    # Assuming ~2 seconds per video with full GPU utilization
    theoretical_throughput = max_parallel / 2.0  # videos per second

    # Calculate total work items (videos × variations)
    total_outputs = len(video_paths) * variations

    print(f"[Parallel Video] ========== MEGA-GPU CONFIGURATION ==========")
    print(f"[Parallel Video] GPU Model: {gpu_info['gpu_name']}")
    print(f"[Parallel Video] GPU Count: {gpu_count} | Sessions/GPU: {sessions_per_gpu}")
    print(f"[Parallel Video] Total NVENC Capacity: {max_parallel} parallel encodes")
    print(f"[Parallel Video] Worker Threads: {max_workers} (1.2x buffer)")
    print(f"[Parallel Video] Theoretical Throughput: ~{theoretical_throughput:.1f} videos/sec")
    print(f"[Parallel Video] Input Videos: {len(video_paths)}")
    print(f"[Parallel Video] Variations per Video: {variations}")
    print(f"[Parallel Video] Total Output Videos: {total_outputs}")
    print(f"[Parallel Video] =============================================")

    # Initialize NVENC session tracker for load balancing across GPUs
    session_tracker = NVENCSessionTracker(gpu_count, sessions_per_gpu)

    # Prepare work items: each video × each variation = unique output
    # Each variation gets a unique work_index for different random transforms
    work_items = []
    for video_idx, video_path in enumerate(video_paths):
        basename = os.path.basename(video_path)
        name, ext = os.path.splitext(basename)

        for var_idx in range(variations):
            # Unique work index ensures different random seed per variation
            work_index = video_idx * variations + var_idx

            # Output filename includes variation number if variations > 1
            if variations > 1:
                output_filename = f"{name}_var{var_idx:04d}{ext}"
            else:
                output_filename = f"{name}_spoofed{ext}"

            output_path = os.path.join(output_dir, output_filename)
            # GPU ID will be assigned dynamically via session tracker
            work_items.append((video_path, output_path, config, work_index))

    total = len(work_items)
    completed = 0
    failed = 0
    results = []
    results_lock = threading.Lock()

    def report_progress(msg=""):
        if progress_callback:
            with results_lock:
                progress = completed / total if total > 0 else 0
            progress_callback(progress, msg)

    report_progress(f"Processing {total} videos with {max_workers} workers ({max_parallel} NVENC sessions) across {gpu_count} GPU(s)...")

    def worker_with_gpu_tracking(work_item: Tuple) -> Dict[str, Any]:
        """Wrapper that acquires/releases GPU from session tracker."""
        video_path, output_path, cfg, video_index = work_item

        # Dynamically acquire GPU with least load
        gpu_id = session_tracker.acquire_gpu()
        try:
            # Add GPU ID to work item tuple
            full_work_item = (video_path, output_path, cfg, video_index, gpu_id)
            return process_single_video_worker(full_work_item)
        finally:
            # Always release the GPU session
            session_tracker.release_gpu(gpu_id)

    # Process in parallel using ThreadPoolExecutor
    # ThreadPoolExecutor is ideal here because FFmpeg subprocess does the actual GPU work
    # Using max_workers (1.5x overprovisioned) to keep GPU queue full
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(worker_with_gpu_tracking, item): item
                        for item in work_items}

        for future in as_completed(future_to_item):
            result = future.result()

            with results_lock:
                results.append(result)

                if result['status'] == 'completed':
                    completed += 1
                else:
                    failed += 1
                    print(f"[Parallel Video] Failed video {result['index']}: {result.get('error', 'unknown')}")

            # Log GPU session stats periodically
            stats = session_tracker.get_stats()
            report_progress(f"Completed {completed}/{total} videos ({failed} failed) | GPU sessions: {stats}")

    return {
        'status': 'completed',
        'total': total,
        'completed': completed,
        'failed': failed,
        'results': results,
        'parallel_sessions': max_parallel,
        'max_workers': max_workers,
        'gpu_count': gpu_count,
        'gpu_type': gpu_info['gpu_type'],
        'input_videos': len(video_paths),
        'variations': variations
    }

# Photo defaults (from original)
PHOTO_DEFAULTS = {
    'crop': 1.5,
    'micro_resize': 1.2,
    'rotation': 0.8,
    'subpixel': 1.0,
    'warp': 0.8,
    'barrel': 0.6,
    'block_shift': 0.6,
    'scale': 98,
    'micro_rescale': 0.4,
    'brightness': 0.04,
    'gamma': 0.06,
    'contrast': 0.04,
    'vignette': 2.0,
    'freq_noise': 0,
    'invisible_watermark': 0,
    'color_space_conv': 0,
    'saturation': 0.06,
    'tint': 1.5,
    'chromatic': 0.8,
    'noise': 3.0,
    'quality': 90,
    'double_compress': 1,
    'flip': 1,
    'force_916': 1,
}


def process_single_image_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel image processing.
    Designed to be called from ThreadPoolExecutor (PIL operations are CPU-bound).

    Args tuple: (input_path, output_path, params, image_index, seed)
    Returns: dict with status, output_path, and any errors
    """
    input_path, output_path, params, image_index, seed = args

    try:
        # Load image
        img = Image.open(input_path)
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Initialize RNGs with unique seed
        py_rng = random.Random(seed)
        rng = np.random.default_rng(seed)

        # Randomize params for this copy
        varied_params = randomize_params(params, py_rng, variation=0.3)

        # Apply transforms (imported from this module - defined below)
        result_img = apply_transforms(img.copy(), varied_params, py_rng, rng)

        # Save result
        quality = int(params.get('quality', 90))
        output_ext = os.path.splitext(output_path)[1].lower()

        if output_ext in ['.jpg', '.jpeg']:
            result_img.save(output_path, 'JPEG', quality=quality, optimize=True)
        elif output_ext == '.png':
            result_img.save(output_path, 'PNG', optimize=True)
        else:
            # Default to JPEG
            output_path = os.path.splitext(output_path)[0] + '.jpg'
            result_img.save(output_path, 'JPEG', quality=quality, optimize=True)

        # Calculate pHash distance if available
        phash_distance = 0
        if PHASH_AVAILABLE:
            try:
                original_hash = imagehash.phash(img)
                result_hash = imagehash.phash(result_img)
                phash_distance = original_hash - result_hash
            except Exception:
                pass

        return {
            'status': 'completed',
            'index': image_index,
            'output_path': output_path,
            'phash_distance': phash_distance,
            'original_size': img.size,
            'output_size': result_img.size
        }

    except Exception as e:
        return {
            'status': 'failed',
            'index': image_index,
            'error': str(e)
        }


def process_images_parallel(
    image_paths: List[str],
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    max_parallel: int = None
) -> Dict[str, Any]:
    """
    Process multiple images in parallel using ThreadPoolExecutor.

    Images are CPU-bound (PIL operations), so ThreadPoolExecutor is ideal.
    Uses multiple threads to maximize CPU utilization.

    Args:
        image_paths: List of input image paths
        output_dir: Directory for output images
        config: Processing configuration
        progress_callback: Optional progress callback
        max_parallel: Max parallel threads (defaults to CPU count)

    Returns:
        Dict with results including processed count and any errors
    """
    if not image_paths:
        return {'error': 'No images to process'}

    # Default to CPU count for image processing (CPU-bound operations)
    if max_parallel is None:
        max_parallel = min(os.cpu_count() or 4, len(image_paths), 16)

    print(f"[Parallel Image] Processing {len(image_paths)} images with {max_parallel} threads")

    # Flatten config to params
    params = {}
    spatial = config.get('spatial', {})
    tonal = config.get('tonal', {})
    visual = config.get('visual', {})
    compression = config.get('compression', {})
    options = config.get('options', {})

    params.update({
        'crop': spatial.get('crop', PHOTO_DEFAULTS['crop']),
        'micro_resize': spatial.get('microResize', PHOTO_DEFAULTS['micro_resize']),
        'rotation': spatial.get('rotation', PHOTO_DEFAULTS['rotation']),
        'subpixel': spatial.get('subpixel', PHOTO_DEFAULTS['subpixel']),
        'warp': spatial.get('warp', PHOTO_DEFAULTS['warp']),
        'barrel': spatial.get('barrel', PHOTO_DEFAULTS['barrel']),
        'block_shift': spatial.get('blockShift', PHOTO_DEFAULTS['block_shift']),
        'scale': spatial.get('scale', PHOTO_DEFAULTS['scale']),
        'micro_rescale': spatial.get('microRescale', PHOTO_DEFAULTS['micro_rescale']),
        'brightness': tonal.get('brightness', PHOTO_DEFAULTS['brightness']),
        'gamma': tonal.get('gamma', PHOTO_DEFAULTS['gamma']),
        'contrast': tonal.get('contrast', PHOTO_DEFAULTS['contrast']),
        'vignette': tonal.get('vignette', PHOTO_DEFAULTS['vignette']),
        'freq_noise': tonal.get('freqNoise', PHOTO_DEFAULTS['freq_noise']),
        'invisible_watermark': tonal.get('invisibleWatermark', PHOTO_DEFAULTS['invisible_watermark']),
        'color_space_conv': tonal.get('colorSpaceConv', PHOTO_DEFAULTS['color_space_conv']),
        'saturation': tonal.get('saturation', PHOTO_DEFAULTS['saturation']),
        'tint': visual.get('tint', PHOTO_DEFAULTS['tint']),
        'chromatic': visual.get('chromatic', PHOTO_DEFAULTS['chromatic']),
        'noise': visual.get('noise', PHOTO_DEFAULTS['noise']),
        'quality': compression.get('quality', PHOTO_DEFAULTS['quality']),
        'double_compress': compression.get('doubleCompress', PHOTO_DEFAULTS['double_compress']),
        'flip': options.get('flip', PHOTO_DEFAULTS['flip']),
        'force_916': options.get('force916', PHOTO_DEFAULTS['force_916']),
    })

    # Prepare work items with unique seeds
    base_time = time.time_ns()
    work_items = []
    for i, image_path in enumerate(image_paths):
        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)
        output_ext = ext if ext.lower() in ['.jpg', '.jpeg', '.png'] else '.jpg'
        output_path = os.path.join(output_dir, f"{name}_spoofed{output_ext}")
        seed = generate_unique_seed(image_path, i, base_time + i * 1000)
        work_items.append((image_path, output_path, params, i, seed))

    total = len(work_items)
    completed = 0
    failed = 0
    results = []
    phash_distances = []
    results_lock = threading.Lock()

    def report_progress(msg=""):
        if progress_callback:
            with results_lock:
                progress = completed / total if total > 0 else 0
            progress_callback(progress, msg)

    report_progress(f"Processing {total} images with {max_parallel} threads...")

    # Process in parallel using ThreadPoolExecutor (CPU-bound PIL operations)
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_item = {executor.submit(process_single_image_worker, item): item
                        for item in work_items}

        for future in as_completed(future_to_item):
            result = future.result()

            with results_lock:
                results.append(result)

                if result['status'] == 'completed':
                    completed += 1
                    if result.get('phash_distance', 0) > 0:
                        phash_distances.append(result['phash_distance'])
                else:
                    failed += 1
                    print(f"[Parallel Image] Failed image {result['index']}: {result.get('error', 'unknown')}")

            report_progress(f"Completed {completed}/{total} images ({failed} failed)")

    # Calculate statistics
    stats = {
        'status': 'completed',
        'total': total,
        'completed': completed,
        'failed': failed,
        'results': results,
        'parallel_threads': max_parallel
    }

    if phash_distances:
        stats['phash_avg'] = sum(phash_distances) / len(phash_distances)
        stats['phash_min'] = min(phash_distances)
        stats['phash_max'] = max(phash_distances)

    return stats


def extract_images_from_zip(zip_path: str, extract_dir: str) -> List[str]:
    """
    Extract image files from a ZIP archive.
    Returns list of paths to extracted image files.
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}
    image_paths = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for name in zf.namelist():
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                # Extract to temp dir
                extracted_path = zf.extract(name, extract_dir)
                image_paths.append(extracted_path)
    return image_paths


def generate_unique_seed(img_path: str, var_idx: int, base_time: int) -> int:
    """Generate unique seed for reproducible randomization."""
    hash_input = f"{img_path}_{var_idx}_{base_time}_{os.getpid()}"
    return int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)


def randomize_params(base_params: Dict, py_rng: random.Random, variation: float = 0.3) -> Dict:
    """
    Randomize parameters within ±variation of base values.
    Each copy gets slightly different transform intensities.
    """
    result = {}
    for key, value in base_params.items():
        if isinstance(value, (int, float)) and value > 0:
            # Vary by ±variation percentage
            factor = 1.0 + py_rng.uniform(-variation, variation)
            new_value = value * factor
            # Keep same type
            result[key] = int(new_value) if isinstance(value, int) else new_value
        else:
            result[key] = value
    return result


# ═══════════════════════════════════════════════════════════════
# TIER 1: SPATIAL TRANSFORMS (★★★★★ Maximum pHash impact)
# ═══════════════════════════════════════════════════════════════

def apply_asymmetric_crop(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Asymmetric crop - different amounts from each side."""
    if strength <= 0:
        return img

    w, h = img.size
    max_crop = strength / 100.0

    # Random crop from each side
    left = int(w * py_rng.uniform(0, max_crop))
    right = int(w * py_rng.uniform(0, max_crop))
    top = int(h * py_rng.uniform(0, max_crop))
    bottom = int(h * py_rng.uniform(0, max_crop))

    # Ensure we don't crop too much
    if left + right >= w * 0.5 or top + bottom >= h * 0.5:
        return img

    return img.crop((left, top, w - right, h - bottom))


def apply_micro_resize(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Micro resize - slight scale variation."""
    if strength <= 0:
        return img

    w, h = img.size
    scale = 1.0 + py_rng.uniform(-strength/100, strength/100)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return img.resize((new_w, new_h), Image.LANCZOS)


def apply_micro_rotation(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Micro rotation with zoom compensation to eliminate black borders."""
    if strength <= 0:
        return img

    angle = py_rng.uniform(-strength, strength)
    orig_w, orig_h = img.size

    # Rotate with expand=True to avoid black corners
    rotated_img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
    rot_w, rot_h = rotated_img.size

    # Calculate the largest rectangle that fits inside the rotated image
    angle_rad = abs(angle) * math.pi / 180
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    if sin_a < 0.001:  # Nearly zero rotation
        return img

    # Scale factor for inscribed rectangle
    scale = min(
        1.0 / (cos_a + sin_a * orig_h / orig_w),
        1.0 / (cos_a + sin_a * orig_w / orig_h)
    )

    # The inscribed rectangle dimensions
    crop_w = int(orig_w * scale)
    crop_h = int(orig_h * scale)

    # Center crop from rotated image
    left = (rot_w - crop_w) // 2
    top = (rot_h - crop_h) // 2

    cropped = rotated_img.crop((left, top, left + crop_w, top + crop_h))

    # Resize back to original dimensions
    return cropped.resize((orig_w, orig_h), Image.LANCZOS)


def apply_subpixel_shift(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Subpixel shift - fractional pixel displacement."""
    if strength <= 0:
        return img

    w, h = img.size
    offset_x = py_rng.uniform(-strength, strength)
    offset_y = py_rng.uniform(-strength, strength)

    # Resize slightly larger, then crop to simulate subpixel shift
    scale = 1.02
    img_scaled = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Calculate crop position
    extra_w = img_scaled.size[0] - w
    extra_h = img_scaled.size[1] - h

    left = int(extra_w / 2 + offset_x)
    top = int(extra_h / 2 + offset_y)

    left = max(0, min(left, img_scaled.size[0] - w))
    top = max(0, min(top, img_scaled.size[1] - h))

    return img_scaled.crop((left, top, left + w, top + h))


def apply_perspective_warp(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Perspective warp - slight trapezoid distortion."""
    if strength <= 0:
        return img

    w, h = img.size
    offset = int(min(w, h) * strength / 100 * 0.1)

    if offset < 1:
        return img

    # Random corner offsets
    src_corners = [(0, 0), (w, 0), (w, h), (0, h)]
    dst_corners = [
        (py_rng.randint(0, offset), py_rng.randint(0, offset)),
        (w - py_rng.randint(0, offset), py_rng.randint(0, offset)),
        (w - py_rng.randint(0, offset), h - py_rng.randint(0, offset)),
        (py_rng.randint(0, offset), h - py_rng.randint(0, offset))
    ]

    coeffs = find_perspective_coeffs(dst_corners, src_corners)
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def apply_barrel_distortion(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Barrel/pincushion distortion."""
    if strength <= 0:
        return img

    arr = np.array(img)
    h, w = arr.shape[:2]

    # Create coordinate maps
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Normalize to -1 to 1
    x_norm = (x - w / 2) / (w / 2)
    y_norm = (y - h / 2) / (h / 2)

    # Calculate distance from center
    r = np.sqrt(x_norm ** 2 + y_norm ** 2)

    # Apply barrel distortion with random sign
    k = (strength / 100) * py_rng.choice([-1, 1]) * 0.3
    distortion = 1 + k * r ** 2

    x_dist = x_norm * distortion
    y_dist = y_norm * distortion

    # Convert back to pixel coordinates
    x_new = ((x_dist * w / 2) + w / 2).astype(np.int32)
    y_new = ((y_dist * h / 2) + h / 2).astype(np.int32)

    x_new = np.clip(x_new, 0, w - 1)
    y_new = np.clip(y_new, 0, h - 1)

    output = arr[y_new, x_new]
    return Image.fromarray(output)


def apply_block_shift(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Block shift - move small blocks of pixels."""
    if strength <= 0:
        return img

    arr = np.array(img)
    h, w = arr.shape[:2]

    block_size = max(8, int(min(w, h) * 0.05))
    max_shift = int(strength)

    if max_shift < 1:
        return img

    # Randomly shift some blocks
    num_blocks = py_rng.randint(2, 6)

    for _ in range(num_blocks):
        bx = py_rng.randint(0, max(1, w - block_size))
        by = py_rng.randint(0, max(1, h - block_size))

        shift_x = py_rng.randint(-max_shift, max_shift)
        shift_y = py_rng.randint(-max_shift, max_shift)

        # Get block
        block = arr[by:by+block_size, bx:bx+block_size].copy()

        # Calculate new position
        new_bx = max(0, min(w - block_size, bx + shift_x))
        new_by = max(0, min(h - block_size, by + shift_y))

        # Place block (simple overwrite)
        arr[new_by:new_by+block_size, new_bx:new_bx+block_size] = block

    return Image.fromarray(arr)


def apply_micro_rescale_photo(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Micro rescale - down then up to shift DCT grid."""
    if strength <= 0:
        return img

    w, h = img.size
    scale_down = 1.0 - (strength / 100) * py_rng.uniform(0.5, 1.0)
    scale_down = max(0.9, scale_down)

    # Scale down
    tmp_w = int(w * scale_down)
    tmp_h = int(h * scale_down)
    img_small = img.resize((tmp_w, tmp_h), Image.LANCZOS)

    # Scale back up
    return img_small.resize((w, h), Image.LANCZOS)


# ═══════════════════════════════════════════════════════════════
# TIER 2: TONAL TRANSFORMS (★★★★☆ DCT coefficients)
# ═══════════════════════════════════════════════════════════════

def apply_brightness_shift(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Brightness adjustment."""
    if strength <= 0:
        return img

    factor = 1.0 + py_rng.uniform(-strength, strength)
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)


def apply_gamma_shift(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Gamma correction."""
    if strength <= 0:
        return img

    gamma = 1.0 + py_rng.uniform(-strength, strength)
    gamma = max(0.5, min(2.0, gamma))

    arr = np.array(img, dtype=np.float32)
    arr = np.clip(255 * (arr / 255) ** gamma, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_contrast_shift(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Contrast adjustment."""
    if strength <= 0:
        return img

    factor = 1.0 + py_rng.uniform(-strength, strength)
    factor = max(0.5, min(1.5, factor))
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def apply_saturation_shift(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Saturation adjustment."""
    if strength <= 0:
        return img

    factor = 1.0 + py_rng.uniform(-strength, strength)
    factor = max(0.5, min(1.5, factor))
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)


def apply_vignette(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Vignette effect - darken corners."""
    if strength <= 0:
        return img

    w, h = img.size
    y, x = np.ogrid[:h, :w]
    center_x, center_y = w / 2, h / 2

    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    dist = dist / max_dist

    vignette_strength = (strength / 100) * py_rng.uniform(0.7, 1.3)
    vignette = 1 - (dist ** 2) * vignette_strength
    vignette = np.clip(vignette, 0, 1)

    arr = np.array(img, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * vignette

    return Image.fromarray(arr.astype(np.uint8))


def apply_frequency_noise(img: Image.Image, strength: float, rng: np.random.Generator) -> Image.Image:
    """Add frequency-domain noise to DCT coefficients."""
    if strength <= 0:
        return img

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Add very low-amplitude noise in frequency pattern
    noise_strength = strength * 0.5

    for c in range(3):
        # Create frequency-aware noise
        freq_noise = rng.normal(0, noise_strength, (h, w))
        arr[:, :, c] = np.clip(arr[:, :, c] + freq_noise, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def apply_invisible_watermark(img: Image.Image, strength: float, rng: np.random.Generator) -> Image.Image:
    """Add invisible watermark pattern (very subtle)."""
    if strength <= 0.5:  # Require minimum strength to avoid artifacts
        return img

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # Create subtle pattern
    pattern = rng.uniform(-strength * 0.1, strength * 0.1, (h, w))

    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + pattern, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def apply_color_space_conversion(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """RGB→YUV→RGB conversion with minimal variation."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    # RGB to YUV
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B + 128
    V = 0.615 * R - 0.51499 * G - 0.10001 * B + 128

    # Add minimal per-pixel variation
    variation_y = rng.uniform(-0.05, 0.05, (h, w))
    variation_u = rng.uniform(-0.02, 0.02, (h, w))
    variation_v = rng.uniform(-0.02, 0.02, (h, w))

    Y = np.clip(Y + variation_y, 0, 255)
    U = np.clip(U + variation_u, 0, 255)
    V = np.clip(V + variation_v, 0, 255)

    # YUV to RGB
    R_new = Y + 1.13983 * (V - 128)
    G_new = Y - 0.39465 * (U - 128) - 0.58060 * (V - 128)
    B_new = Y + 2.03211 * (U - 128)

    arr[:, :, 0] = np.clip(R_new, 0, 255)
    arr[:, :, 1] = np.clip(G_new, 0, 255)
    arr[:, :, 2] = np.clip(B_new, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


# ═══════════════════════════════════════════════════════════════
# TIER 3: VISUAL VARIATION (★★☆☆☆ Makes copies look different)
# ═══════════════════════════════════════════════════════════════

def apply_color_tint(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Add subtle color tint."""
    if strength <= 0:
        return img

    arr = np.array(img, dtype=np.float32)

    tint_r = py_rng.uniform(-strength, strength)
    tint_g = py_rng.uniform(-strength, strength)
    tint_b = py_rng.uniform(-strength, strength)

    arr[:, :, 0] = np.clip(arr[:, :, 0] + tint_r, 0, 255)
    arr[:, :, 1] = np.clip(arr[:, :, 1] + tint_g, 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] + tint_b, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def apply_chromatic_aberration(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """Chromatic aberration - RGB channel separation."""
    if strength <= 0:
        return img

    arr = np.array(img)
    h, w = arr.shape[:2]

    shift = max(1, int(strength * py_rng.uniform(0.5, 1.0)))

    new_arr = np.zeros_like(arr)

    # Shift channels in different directions
    direction = py_rng.choice(['horizontal', 'radial'])

    if direction == 'horizontal':
        new_arr[:, shift:, 0] = arr[:, :-shift, 0]  # Red right
        new_arr[:, :, 1] = arr[:, :, 1]              # Green center
        new_arr[:, :-shift, 2] = arr[:, shift:, 2]  # Blue left
    else:
        # Simple radial - just slight offset
        new_arr[:, :, 0] = arr[:, :, 0]
        new_arr[:, :, 1] = arr[:, :, 1]
        new_arr[:, shift:, 2] = arr[:, :-shift, 2]

    # Blend with original
    blend = py_rng.uniform(0.3, 0.6)
    result = (arr * (1 - blend) + new_arr * blend).astype(np.uint8)

    return Image.fromarray(result)


def apply_noise(img: Image.Image, strength: float, rng: np.random.Generator) -> Image.Image:
    """Add Gaussian noise."""
    if strength <= 0:
        return img

    arr = np.array(img, dtype=np.float32)
    noise = rng.normal(0, strength * 2, arr.shape)
    arr = np.clip(arr + noise, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def apply_double_compression(img: Image.Image, quality: int, py_rng: random.Random) -> Image.Image:
    """Double JPEG compression with slightly different quality."""
    # First compression
    buffer1 = io.BytesIO()
    q1 = max(60, quality - py_rng.randint(5, 15))
    img.save(buffer1, 'JPEG', quality=q1)
    buffer1.seek(0)
    img = Image.open(buffer1)

    # Second compression
    buffer2 = io.BytesIO()
    q2 = quality + py_rng.randint(-3, 3)
    img.save(buffer2, 'JPEG', quality=q2)
    buffer2.seek(0)

    return Image.open(buffer2).convert('RGB')


def pad_to_9_16_photo(img: Image.Image, py_rng: random.Random) -> Image.Image:
    """Force 9:16 aspect ratio with blurred background."""
    w, h = img.size
    target_ratio = 9 / 16

    if h > 0 and abs((w / h) - target_ratio) < 0.01:
        return img

    if w <= 0 or h <= 0:
        return img

    base = max(w, h)
    target_w, target_h = TARGET_RESOLUTIONS['high'] if base >= 1000 else TARGET_RESOLUTIONS['low']

    # Create blurred background
    bg = img.copy()
    scale_cover = max(target_w / w, target_h / h)
    scale_bg = scale_cover * 1.35

    bw, bh = int(w * scale_bg), int(h * scale_bg)
    bg = bg.resize((bw, bh), Image.BICUBIC)

    # Slight rotation for background
    angle_bg = py_rng.uniform(-5.0, 5.0)
    bg = bg.rotate(angle_bg, resample=Image.BICUBIC, expand=True)

    bg_w, bg_h = bg.size
    left = max(0, (bg_w - target_w) // 2)
    top = max(0, (bg_h - target_h) // 2)
    bg = bg.crop((left, top, left + target_w, top + target_h))

    if bg.size != (target_w, target_h):
        bg = bg.resize((target_w, target_h), Image.BICUBIC)

    # Blur and darken background
    blur_radius = py_rng.uniform(22.0, 40.0)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    bg = ImageEnhance.Brightness(bg).enhance(py_rng.uniform(0.55, 0.75))

    canvas = bg.convert("RGBA")

    # Foreground - the actual image
    fg = img.copy().convert("RGBA")
    angle_fg = py_rng.uniform(-1.8, 1.8)
    fg = fg.rotate(angle_fg, resample=Image.BICUBIC, expand=True)
    fw, fh = fg.size

    scale_fg = min(target_w / fw, target_h / fh) * py_rng.uniform(0.87, 0.93)
    fw2, fh2 = int(fw * scale_fg), int(fh * scale_fg)
    fg = fg.resize((fw2, fh2), Image.BICUBIC)

    # Center with slight random offset
    max_jit_x = int(target_w * 0.018)
    max_jit_y = int(target_h * 0.018)
    x = (target_w - fw2) // 2 + py_rng.randint(-max_jit_x, max_jit_x)
    y = (target_h - fh2) // 2 + py_rng.randint(-max_jit_y, max_jit_y)

    canvas.paste(fg, (x, y), fg)
    return canvas.convert("RGB")


# ═══════════════════════════════════════════════════════════════
# MAIN PROCESSING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def apply_transforms(img: Image.Image, params: Dict, py_rng: random.Random, rng: np.random.Generator) -> Image.Image:
    """Apply all image transformations based on params."""

    # TIER 1: SPATIAL (★★★★★)
    if params.get('flip', 0) and py_rng.random() > 0.5:
        img = ImageOps.mirror(img)

    if params.get('crop', 0) > 0:
        img = apply_asymmetric_crop(img, params['crop'], py_rng)

    if params.get('micro_resize', 0) > 0:
        img = apply_micro_resize(img, params['micro_resize'], py_rng)

    if params.get('rotation', 0) > 0:
        img = apply_micro_rotation(img, params['rotation'], py_rng)

    if params.get('subpixel', 0) > 0:
        img = apply_subpixel_shift(img, params['subpixel'], py_rng)

    if params.get('warp', 0) > 0:
        img = apply_perspective_warp(img, params['warp'], py_rng)

    if params.get('barrel', 0) > 0:
        img = apply_barrel_distortion(img, params['barrel'], py_rng)

    if params.get('block_shift', 0) > 0:
        img = apply_block_shift(img, params['block_shift'], py_rng)

    # TIER 2: TONAL (★★★★☆)
    if params.get('brightness', 0) > 0:
        img = apply_brightness_shift(img, params['brightness'], py_rng)

    if params.get('gamma', 0) > 0:
        img = apply_gamma_shift(img, params['gamma'], py_rng)

    if params.get('contrast', 0) > 0:
        img = apply_contrast_shift(img, params['contrast'], py_rng)

    if params.get('vignette', 0) > 0:
        img = apply_vignette(img, params['vignette'], py_rng)

    if params.get('freq_noise', 0) > 0:
        img = apply_frequency_noise(img, params['freq_noise'], rng)

    if params.get('invisible_watermark', 0) > 0.5:
        img = apply_invisible_watermark(img, params['invisible_watermark'], rng)

    if params.get('color_space_conv', 0):
        img = apply_color_space_conversion(img, rng)

    # TIER 3: VISUAL (★★☆☆☆)
    if params.get('saturation', 0) > 0:
        img = apply_saturation_shift(img, params['saturation'], py_rng)

    if params.get('tint', 0) > 0:
        img = apply_color_tint(img, params['tint'], py_rng)

    if params.get('chromatic', 0) > 0:
        img = apply_chromatic_aberration(img, params['chromatic'], py_rng)

    if params.get('noise', 0) > 0:
        img = apply_noise(img, params['noise'], rng)

    # FINAL PROCESSING
    if params.get('micro_rescale', 0) > 0:
        img = apply_micro_rescale_photo(img, params['micro_rescale'], py_rng)

    if params.get('scale', 100) < 100:
        s = params['scale'] + py_rng.uniform(-0.5, 0.5)
        w, h = img.size
        img = img.resize((int(w * (s/100)), int(h * (s/100))), Image.BICUBIC)

    if params.get('force_916', 0):
        img = pad_to_9_16_photo(img, py_rng)

    # Double compression
    q = int(params.get('quality', 88))
    if params.get('double_compress', 1):
        img = apply_double_compression(img, q, py_rng)

    return img


def process_single_copy(
    img: Image.Image,
    original_phash: Optional[Any],
    params: Dict,
    copy_idx: int,
    seed: int,
    existing_phashes: List[Any],
    phash_min: int = PHASH_MIN_DISTANCE
) -> Tuple[Image.Image, float, float, Any]:
    """
    Process a single copy with pHash verification.

    Returns:
        (processed_image, distance_from_original, min_distance_from_copies, phash)
    """
    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)

    # Randomize params for this copy
    varied_params = randomize_params(params, py_rng, variation=0.3)

    # Apply transforms
    result_img = apply_transforms(img.copy(), varied_params, py_rng, rng)

    # Calculate pHash distances
    distance = 0
    copy_distance = 999
    result_phash = None

    if PHASH_AVAILABLE:
        try:
            result_phash = imagehash.phash(result_img)

            if original_phash is not None:
                distance = original_phash - result_phash

            # Check against existing copies
            if existing_phashes:
                copy_distances = [result_phash - ph for ph in existing_phashes]
                if copy_distances:
                    copy_distance = min(copy_distances)
        except Exception as e:
            print(f"pHash error: {e}")

    return result_img, distance, copy_distance, result_phash


def process_batch_spoofer(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    copies: int,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process image and generate multiple unique copies as ZIP.

    Args:
        input_path: Path to input image
        output_path: Path for output ZIP file
        config: Transform configuration
        copies: Number of copies to generate
        progress_callback: Optional progress callback

    Returns:
        Dict with results including pHash statistics
    """
    import traceback

    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    print(f"[DEBUG] process_batch_spoofer called with:")
    print(f"[DEBUG]   input_path: {input_path}")
    print(f"[DEBUG]   output_path: {output_path}")
    print(f"[DEBUG]   copies: {copies}")
    print(f"[DEBUG]   config keys: {list(config.keys())}")

    report_progress(0.05, "Loading image...")

    # Load original image
    original_img = Image.open(input_path)
    original_img = ImageOps.exif_transpose(original_img)
    if original_img.mode != 'RGB':
        original_img = original_img.convert('RGB')

    # Calculate original pHash
    original_phash = None
    if PHASH_AVAILABLE:
        try:
            original_phash = imagehash.phash(original_img)
        except Exception:
            pass

    report_progress(0.1, f"Generating {copies} copies...")

    # Flatten config for processing
    params = {}

    # Spatial
    spatial = config.get('spatial', {})
    params['crop'] = spatial.get('crop', PHOTO_DEFAULTS['crop'])
    params['micro_resize'] = spatial.get('microResize', PHOTO_DEFAULTS['micro_resize'])
    params['rotation'] = spatial.get('rotation', PHOTO_DEFAULTS['rotation'])
    params['subpixel'] = spatial.get('subpixel', PHOTO_DEFAULTS['subpixel'])
    params['warp'] = spatial.get('warp', PHOTO_DEFAULTS['warp'])
    params['barrel'] = spatial.get('barrel', PHOTO_DEFAULTS['barrel'])
    params['block_shift'] = spatial.get('blockShift', PHOTO_DEFAULTS['block_shift'])
    params['scale'] = spatial.get('scale', PHOTO_DEFAULTS['scale'])
    params['micro_rescale'] = spatial.get('microRescale', PHOTO_DEFAULTS['micro_rescale'])

    # Tonal
    tonal = config.get('tonal', {})
    params['brightness'] = tonal.get('brightness', PHOTO_DEFAULTS['brightness'])
    params['gamma'] = tonal.get('gamma', PHOTO_DEFAULTS['gamma'])
    params['contrast'] = tonal.get('contrast', PHOTO_DEFAULTS['contrast'])
    params['vignette'] = tonal.get('vignette', PHOTO_DEFAULTS['vignette'])
    params['freq_noise'] = tonal.get('freqNoise', PHOTO_DEFAULTS['freq_noise'])
    params['invisible_watermark'] = tonal.get('invisibleWatermark', PHOTO_DEFAULTS['invisible_watermark'])
    params['color_space_conv'] = tonal.get('colorSpaceConv', PHOTO_DEFAULTS['color_space_conv'])
    params['saturation'] = tonal.get('saturation', PHOTO_DEFAULTS['saturation'])

    # Visual
    visual = config.get('visual', {})
    params['tint'] = visual.get('tint', PHOTO_DEFAULTS['tint'])
    params['chromatic'] = visual.get('chromatic', PHOTO_DEFAULTS['chromatic'])
    params['noise'] = visual.get('noise', PHOTO_DEFAULTS['noise'])

    # Compression
    compression = config.get('compression', {})
    params['quality'] = compression.get('quality', PHOTO_DEFAULTS['quality'])
    params['double_compress'] = compression.get('doubleCompress', PHOTO_DEFAULTS['double_compress'])

    # Options
    options = config.get('options', {})
    params['flip'] = options.get('flip', PHOTO_DEFAULTS['flip'])
    params['force_916'] = options.get('force916', PHOTO_DEFAULTS['force_916'])
    params['random_names'] = options.get('randomNames', 0)

    phash_min = options.get('phashMinDistance', PHASH_MIN_DISTANCE)
    verify_phash = options.get('verifyPhash', True)
    compare_copies = options.get('compareCopies', True)

    # Generate copies
    base_time = time.time_ns()
    results = []
    phash_distances = []
    copy_distances = []
    existing_phashes = []

    # Prepare ZIP in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i in range(copies):
            try:
                # Progress
                progress = 0.1 + (i / copies) * 0.8
                report_progress(progress, f"Processing copy {i+1}/{copies}...")

                # Generate unique seed
                seed = generate_unique_seed(input_path, i, base_time + i * 1000)

                # Process with retry for pHash
                max_retries = 5 if verify_phash else 1
                best_result = None
                best_distance = -1

                for retry in range(max_retries):
                    retry_seed = seed + retry * 10000

                    result_img, distance, copy_dist, result_phash = process_single_copy(
                        original_img,
                        original_phash,
                        params,
                        i,
                        retry_seed,
                        existing_phashes if compare_copies else [],
                        phash_min
                    )

                    # Check if meets threshold
                    meets_threshold = True
                    if verify_phash and distance > 0:
                        meets_threshold = distance >= phash_min
                        if compare_copies and copy_dist < 999:
                            meets_threshold = meets_threshold and copy_dist >= phash_min

                    # Track best
                    if distance > best_distance:
                        best_distance = distance
                        best_result = (result_img, distance, copy_dist, result_phash)

                    if meets_threshold:
                        break

                # Use best result
                if best_result:
                    result_img, distance, copy_dist, result_phash = best_result

                    if result_phash is not None:
                        existing_phashes.append(result_phash)

                    if distance > 0:
                        phash_distances.append(distance)
                    if copy_dist < 999:
                        copy_distances.append(copy_dist)

                # Generate filename
                py_rng = random.Random(seed)
                if params.get('random_names'):
                    filename = ''.join(py_rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=12)) + '.jpg'
                else:
                    rand_id = py_rng.randint(1000, 9999)
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    clean_name = "".join([c for c in base_name if c.isalnum() or c in (' ', '-', '_')]).strip()[:20]
                    filename = f"{clean_name}_v{i}_{rand_id}.jpg"

                # Save to ZIP
                img_buffer = io.BytesIO()
                quality = int(params.get('quality', 90))
                result_img.save(img_buffer, 'JPEG', quality=quality, optimize=True)
                img_buffer.seek(0)

                zf.writestr(filename, img_buffer.getvalue())
                results.append(filename)

            except Exception as e:
                print(f"[ERROR] Exception during copy {i}: {str(e)}")
                print(traceback.format_exc())
                raise

    # Write ZIP to output
    zip_buffer.seek(0)
    with open(output_path, 'wb') as f:
        f.write(zip_buffer.getvalue())

    report_progress(1.0, "Complete")

    # Calculate statistics
    stats = {
        'copies_generated': len(results),
        'files': results,
    }

    if phash_distances:
        stats['phash_avg'] = sum(phash_distances) / len(phash_distances)
        stats['phash_min'] = min(phash_distances)
        stats['phash_max'] = max(phash_distances)

    if copy_distances:
        stats['copy_distance_min'] = min(copy_distances)

    return stats


def process_spoofer(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process a single image or video with spoofer transforms.
    For batch processing, use process_batch_spoofer.

    NEW: If input is a ZIP with multiple videos, processes them in parallel
    using multiple NVENC sessions (optimized for datacenter GPUs like A5000/A6000).
    """
    import tempfile
    import shutil

    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    # Detect file type
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS
    is_zip = ext == '.zip'

    report_progress(0.05, "Analyzing file...")

    # NEW: Check if input is a ZIP with videos (batch video mode)
    if is_zip:
        report_progress(0.08, "Checking ZIP contents...")

        # Create temp directory for extraction
        temp_dir = tempfile.mkdtemp(prefix="spoofer_batch_")

        try:
            # Extract videos from ZIP
            video_paths = extract_videos_from_zip(input_path, temp_dir)

            if video_paths:
                # PARALLEL VIDEO PROCESSING MODE WITH VARIATIONS
                # Get variations count from config (supports both formats)
                variations = config.get('variations') or config.get('copies') or config.get('options', {}).get('copies', 1)
                variations = max(1, int(variations))

                total_outputs = len(video_paths) * variations
                report_progress(0.1, f"Found {len(video_paths)} videos × {variations} variations = {total_outputs} outputs")

                # Create output directory
                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(output_dir, exist_ok=True)

                # Process videos in parallel with variations
                result = process_videos_parallel(
                    video_paths,
                    output_dir,
                    config,
                    progress_callback=progress_callback,
                    variations=variations
                )

                if result.get('error'):
                    return result

                # Create output ZIP with processed videos
                report_progress(0.95, "Creating output ZIP...")

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
                    'input_videos': result.get('input_videos', len(video_paths)),
                    'variations': result.get('variations', variations),
                    'videos_processed': result.get('completed', 0),
                    'videos_failed': result.get('failed', 0),
                    'parallel_sessions': result.get('parallel_sessions', 1),
                    'gpu_count': result.get('gpu_count', 1),
                    'output_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
                }
            else:
                # ZIP contains images, not videos - use parallel image processing
                report_progress(0.1, "ZIP contains images, checking for batch processing...")

                # Extract images from ZIP
                image_paths = extract_images_from_zip(input_path, temp_dir)

                if image_paths:
                    # PARALLEL IMAGE PROCESSING MODE
                    report_progress(0.12, f"Found {len(image_paths)} images, starting parallel processing...")

                    # Create output directory
                    output_dir = os.path.join(temp_dir, "output")
                    os.makedirs(output_dir, exist_ok=True)

                    # Process images in parallel (CPU-bound with ThreadPoolExecutor)
                    result = process_images_parallel(
                        image_paths,
                        output_dir,
                        config,
                        progress_callback=progress_callback
                    )

                    if result.get('error'):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        return result

                    # Create output ZIP with processed images
                    report_progress(0.95, "Creating output ZIP...")

                    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
                        for r in result.get('results', []):
                            if r.get('status') == 'completed' and r.get('output_path'):
                                if os.path.exists(r['output_path']):
                                    arcname = os.path.basename(r['output_path'])
                                    zf.write(r['output_path'], arcname)

                    report_progress(1.0, "Complete")

                    shutil.rmtree(temp_dir, ignore_errors=True)

                    return {
                        'status': 'completed',
                        'mode': 'parallel_image_batch',
                        'images_processed': result.get('completed', 0),
                        'images_failed': result.get('failed', 0),
                        'parallel_threads': result.get('parallel_threads', 1),
                        'phash_avg': result.get('phash_avg'),
                        'output_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
                    }

                # No images or videos found in ZIP
                shutil.rmtree(temp_dir, ignore_errors=True)
                return {'error': 'ZIP file contains no supported images or videos'}

        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    # Check if batch mode (copies > 1)
    copies = config.get('copies') or config.get('options', {}).get('copies', 1)
    output_mode = config.get('outputMode', 'file')
    output_dir = config.get('outputDir')

    if is_video:
        if copies > 1 or output_mode == 'directory':
            # BATCH VIDEO MODE: Create multiple variations
            return process_batch_video(input_path, output_path, config, copies, progress_callback)
        else:
            return process_video(input_path, output_path, config, report_progress)
    else:
        if copies > 1:
            return process_batch_spoofer(input_path, output_path, config, copies, progress_callback)
        else:
            return process_single_image(input_path, output_path, config, report_progress)


def process_batch_video(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    copies: int,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process video in batch mode - create multiple variations.
    Uses parallel NVENC sessions for maximum throughput on datacenter GPUs.

    Args:
        input_path: Path to input video
        output_path: Output directory path (when outputMode='directory') or base path
        config: Processing configuration
        copies: Number of copies/variations to create
        progress_callback: Progress callback function
    """
    import tempfile
    import shutil

    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    copies = max(1, int(copies))
    report_progress(0.05, f"Preparing batch video processing ({copies} variations)...")

    # Determine output directory
    output_mode = config.get('outputMode', 'file')
    output_dir = config.get('outputDir')

    if output_mode == 'directory' and output_dir:
        # Pipeline mode: output to specified directory
        out_dir = output_dir
    elif os.path.isdir(output_path):
        # output_path is already a directory
        out_dir = output_path
    else:
        # Create temp directory for outputs
        out_dir = tempfile.mkdtemp(prefix="video_batch_")

    os.makedirs(out_dir, exist_ok=True)

    # Get base filename for outputs
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Create list of output paths for each variation
    video_tasks = []
    for i in range(copies):
        output_file = os.path.join(out_dir, f"{base_name}_v{i+1}.mp4")
        # Each copy gets a unique seed for different transforms
        copy_config = {
            **config,
            '_seed': int(time.time() * 1000) + i * 12345,
            '_copy_index': i
        }
        video_tasks.append({
            'input': input_path,
            'output': output_file,
            'config': copy_config
        })

    report_progress(0.1, f"Processing {copies} video variations in parallel...")

    # Use process_videos_parallel for efficient multi-NVENC processing
    # Pass the video ONCE with variations=copies to generate unique filenames
    result = process_videos_parallel(
        [input_path],  # Single input video
        out_dir,
        config,
        progress_callback=progress_callback,
        max_parallel=None,  # Auto-detect based on GPU
        variations=copies  # Create N unique variations
    )

    if result.get('error'):
        return result

    # Count successful outputs
    completed = 0
    failed = 0
    output_files = []

    for r in result.get('results', []):
        if r.get('status') == 'completed' and r.get('output_path'):
            if os.path.exists(r['output_path']):
                completed += 1
                output_files.append(r['output_path'])
        else:
            failed += 1

    # If outputMode is 'file' (not directory), create ZIP
    if output_mode != 'directory' and not os.path.isdir(output_path):
        report_progress(0.95, "Creating output ZIP...")
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zf:
            for f in output_files:
                zf.write(f, os.path.basename(f))

        # Cleanup temp directory
        if out_dir != output_path and out_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(out_dir, ignore_errors=True)

    report_progress(1.0, "Complete")

    return {
        'status': 'completed',
        'mode': 'batch_video',
        'copies_requested': copies,
        'videos_processed': completed,
        'videos_failed': failed,
        'output_files': output_files,
        'output_dir': out_dir if output_mode == 'directory' else None
    }


def process_single_image(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Process a single image (not batch mode)."""

    report_progress(0.1, "Loading image...")

    original_img = Image.open(input_path)
    original_img = ImageOps.exif_transpose(original_img)
    if original_img.mode != 'RGB':
        original_img = original_img.convert('RGB')

    original_size = original_img.size

    # Flatten config
    params = {}
    spatial = config.get('spatial', {})
    tonal = config.get('tonal', {})
    visual = config.get('visual', {})
    compression = config.get('compression', {})
    options = config.get('options', {})

    params.update({
        'crop': spatial.get('crop', 0),
        'micro_resize': spatial.get('microResize', 0),
        'rotation': spatial.get('rotation', 0),
        'subpixel': spatial.get('subpixel', 0),
        'warp': spatial.get('warp', 0),
        'barrel': spatial.get('barrel', 0),
        'block_shift': spatial.get('blockShift', 0),
        'scale': spatial.get('scale', 100),
        'micro_rescale': spatial.get('microRescale', 0),
        'brightness': tonal.get('brightness', 0),
        'gamma': tonal.get('gamma', 0),
        'contrast': tonal.get('contrast', 0),
        'vignette': tonal.get('vignette', 0),
        'freq_noise': tonal.get('freqNoise', 0),
        'invisible_watermark': tonal.get('invisibleWatermark', 0),
        'color_space_conv': tonal.get('colorSpaceConv', 0),
        'saturation': tonal.get('saturation', 0),
        'tint': visual.get('tint', 0),
        'chromatic': visual.get('chromatic', 0),
        'noise': visual.get('noise', 0),
        'quality': compression.get('quality', 92),
        'double_compress': compression.get('doubleCompress', 0),
        'flip': options.get('flip', 0),
        'force_916': options.get('force916', 0),
    })

    report_progress(0.3, "Applying transforms...")

    seed = int(time.time() * 1000) % (2**31)
    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)

    result_img = apply_transforms(original_img.copy(), params, py_rng, rng)

    report_progress(0.8, "Saving...")

    quality = int(params.get('quality', 92))
    output_ext = os.path.splitext(output_path)[1].lower()

    if output_ext in ['.jpg', '.jpeg']:
        result_img.save(output_path, 'JPEG', quality=quality, optimize=True)
    elif output_ext == '.png':
        result_img.save(output_path, 'PNG', optimize=True)
    else:
        output_path = os.path.splitext(output_path)[0] + '.jpg'
        result_img.save(output_path, 'JPEG', quality=quality, optimize=True)

    report_progress(1.0, "Complete")

    # Calculate pHash distance
    phash_distance = 0
    if PHASH_AVAILABLE:
        try:
            original_hash = imagehash.phash(original_img)
            result_hash = imagehash.phash(result_img)
            phash_distance = original_hash - result_hash
        except Exception:
            pass

    return {
        'original_size': original_size,
        'output_size': result_img.size,
        'quality': quality,
        'phash_distance': phash_distance
    }


def process_video(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Process video with FFmpeg + NVENC (fast preset)."""

    # Ensure output path has video extension (FFmpeg needs it to determine format)
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext not in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        output_path = output_path + '.mp4'
        print(f"[DEBUG] Added .mp4 extension to output: {output_path}")

    report_progress(0.1, "Analyzing video...")

    # Get video info
    try:
        import json
        probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', input_path]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)
        video_stream = next((s for s in video_info.get('streams', []) if s.get('codec_type') == 'video'), {})
        original_width = int(video_stream.get('width', 1920))
        original_height = int(video_stream.get('height', 1080))
        duration = float(video_info.get('format', {}).get('duration', 0))
    except Exception:
        original_width, original_height, duration = 1920, 1080, 0

    # Build filter chain
    filters = []
    py_rng = random.Random(int(time.time() * 1000))

    spatial = config.get('spatial', {})
    tonal = config.get('tonal', {})
    visual = config.get('visual', {})
    video_cfg = config.get('video', {})

    # Spatial filters
    if spatial.get('crop', 0) > 0:
        crop_pct = spatial['crop'] / 100
        crop_w = int(original_width * (1 - crop_pct * 2))
        crop_h = int(original_height * (1 - crop_pct * 2))
        filters.append(f"crop={crop_w}:{crop_h}")

    if spatial.get('rotation', 0) > 0:
        angle_deg = py_rng.uniform(-spatial['rotation'], spatial['rotation'])
        angle_rad = angle_deg * math.pi / 180

        # Calculate zoom factor to eliminate black borders
        abs_angle = abs(angle_rad)
        cos_a = math.cos(abs_angle)
        sin_a = math.sin(abs_angle)

        if sin_a > 0.001:
            zoom_w = cos_a + sin_a * (original_height / original_width)
            zoom_h = cos_a + sin_a * (original_width / original_height)
            zoom_factor = max(zoom_w, zoom_h)
            zoom_factor = max(1.0, min(zoom_factor, 2.0))

            scaled_w = int(original_width * zoom_factor)
            scaled_h = int(original_height * zoom_factor)
            scaled_w = scaled_w + (scaled_w % 2)
            scaled_h = scaled_h + (scaled_h % 2)

            # Pipeline: scale up → rotate → crop center
            filters.append(f"scale={scaled_w}:{scaled_h}:flags=lanczos")
            filters.append(f"rotate={angle_rad}:fillcolor=black")
            filters.append(f"crop={original_width}:{original_height}")

    # Tonal filters (eq filter)
    eq_params = []
    if tonal.get('brightness', 0) > 0:
        b = py_rng.uniform(-tonal['brightness'], tonal['brightness']) * 0.4
        eq_params.append(f"brightness={b:.3f}")
    if tonal.get('contrast', 0) > 0:
        c = 1 + py_rng.uniform(-tonal['contrast'], tonal['contrast'])
        eq_params.append(f"contrast={c:.3f}")
    if tonal.get('saturation', 0) > 0:
        s = 1 + py_rng.uniform(-tonal['saturation'], tonal['saturation'])
        eq_params.append(f"saturation={s:.3f}")
    if tonal.get('gamma', 0) > 0:
        g = 1 + py_rng.uniform(-tonal['gamma'], tonal['gamma'])
        eq_params.append(f"gamma={g:.3f}")

    if eq_params:
        filters.append(f"eq={':'.join(eq_params)}")

    # Visual filters
    if visual.get('noise', 0) > 0:
        filters.append(f"noise=alls={visual['noise']*2}:allf=t")

    if tonal.get('vignette', 0) > 0:
        filters.append(f"vignette=PI/{3 + (1 - tonal['vignette']/100) * 4}")

    # Speed variation
    if video_cfg.get('speedVariation', 0) > 0:
        speed = 1 + py_rng.uniform(-video_cfg['speedVariation']/100, video_cfg['speedVariation']/100)
        speed = max(0.9, min(1.1, speed))
        filters.append(f"setpts={1/speed}*PTS")

    # FPS variation
    fps = 30
    if video_cfg.get('fpsVar', 0) > 0:
        fps = 30 + py_rng.uniform(-video_cfg['fpsVar'], video_cfg['fpsVar'])

    # Build base FFmpeg command
    filter_str = ','.join(filters) if filters else None
    bitrate = int(5000 * video_cfg.get('bitrate', 90) / 100)
    keep_audio = video_cfg.get('keepAudio', True)

    def run_ffmpeg(cmd: list, mode_name: str) -> tuple:
        """Run FFmpeg and return (success, stderr_output, returncode)."""
        print(f"[DEBUG] FFmpeg command ({mode_name}): {' '.join(cmd)}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        stderr_lines = []

        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            stderr_lines.append(line)
            if 'time=' in line and duration > 0:
                try:
                    time_str = line.split('time=')[1].split()[0]
                    h, m, s = time_str.split(':')
                    current_time = int(h) * 3600 + int(m) * 60 + float(s)
                    progress = min(0.2 + (current_time / duration) * 0.75, 0.95)
                    report_progress(progress, f"Encoding ({mode_name})... {int(current_time)}s / {int(duration)}s")
                except:
                    pass

        stderr_output = ''.join(stderr_lines[-30:]) if stderr_lines else 'No stderr captured'
        return process.returncode == 0, stderr_output, process.returncode

    # Check if input has audio stream
    has_audio = False
    try:
        import json
        audio_probe = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-select_streams', 'a', input_path],
            capture_output=True, text=True
        )
        audio_info = json.loads(audio_probe.stdout)
        has_audio = len(audio_info.get('streams', [])) > 0
        print(f"[DEBUG] Input has audio: {has_audio}")
    except Exception as e:
        print(f"[DEBUG] Audio detection failed: {e}")
        has_audio = False

    # Get GPU ID from config (for multi-GPU batch processing)
    # Must be defined before build_ffmpeg_cmd which uses it as a closure variable
    gpu_id = config.get('_gpu_id', 0)
    if gpu_id > 0:
        print(f"[Video Processing] Using GPU {gpu_id} for NVENC encoding")

    def build_ffmpeg_cmd_full_hw(encoder_opts: list) -> list:
        """Build FFmpeg command with FULL hardware pipeline (decode + encode on GPU)."""
        cmd = ['ffmpeg', '-y']

        # FULL hardware acceleration pipeline - keeps frames in GPU memory
        cmd.extend([
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
            '-hwaccel_output_format', 'cuda',  # Keep decoded frames in GPU memory
            '-c:v', 'h264_cuvid',  # Hardware H264 decoder
        ])

        cmd.extend(['-i', input_path])

        # Apply filters with GPU memory transfer
        if filter_str:
            cmd.extend(['-vf', f'hwdownload,format=nv12,{filter_str},hwupload_cuda'])

        # Video encoder
        cmd.extend(['-c:v', 'h264_nvenc'])
        cmd.extend(encoder_opts)

        # Audio handling
        if keep_audio and has_audio:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k', '-strict', 'experimental'])
        else:
            cmd.append('-an')

        cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])
        return cmd

    def build_ffmpeg_cmd_simple_nvenc(encoder_opts: list) -> list:
        """Build FFmpeg command with software decode + NVENC encode."""
        cmd = ['ffmpeg', '-y']

        # Hardware acceleration for decoding (but not full pipeline)
        cmd.extend([
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
        ])

        cmd.extend(['-i', input_path])

        # Apply filters (CPU-based)
        if filter_str:
            cmd.extend(['-vf', filter_str])

        # Video encoder
        cmd.extend(['-c:v', 'h264_nvenc'])
        cmd.extend(encoder_opts)

        # Audio handling
        if keep_audio and has_audio:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k', '-strict', 'experimental'])
        else:
            cmd.append('-an')

        cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])
        return cmd

    def build_ffmpeg_cmd_cpu(encoder_opts: list) -> list:
        """Build FFmpeg command with CPU encoding (fallback)."""
        cmd = ['ffmpeg', '-y', '-i', input_path]

        # Apply filters (CPU-based)
        if filter_str:
            cmd.extend(['-vf', filter_str])

        # Video encoder
        cmd.extend(['-c:v', 'libx264'])
        cmd.extend(encoder_opts)

        # Audio handling
        if keep_audio and has_audio:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k', '-strict', 'experimental'])
        else:
            cmd.append('-an')

        cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])
        return cmd

    # NVENC encoder options with MAXIMUM THROUGHPUT settings
    # p1 = fastest preset, ll = low latency, no lookahead, no B-frames
    nvenc_opts = [
        '-gpu', str(gpu_id),      # Multi-GPU: select specific GPU
        '-preset', 'p1',          # Fastest NVENC preset (p1-p7)
        '-tune', 'll',            # Low latency tuning (faster than hq)
        '-rc', 'vbr',             # Variable bitrate
        '-cq', '26',              # Slightly relaxed quality for speed
        '-b:v', f'{bitrate}k',
        '-maxrate', f'{int(bitrate * 2)}k',  # 2x headroom for peaks
        '-bufsize', f'{bitrate * 2}k',
        '-rc-lookahead', '0',     # Zero lookahead for minimum latency
        '-bf', '0',               # No B-frames for maximum speed
        '-spatial-aq', '0',       # Disable spatial AQ for speed
        '-temporal-aq', '0',      # Disable temporal AQ for speed
    ]

    # libx264 (CPU) encoder options - ultrafast fallback
    libx264_opts = [
        '-preset', 'ultrafast',   # Fastest x264 preset
        '-crf', '23',
        '-b:v', f'{bitrate}k',
        '-maxrate', f'{int(bitrate * 2)}k',
        '-bufsize', f'{bitrate * 2}k',
        '-pix_fmt', 'yuv420p',
    ]

    # Tier 1: Try FULL hardware pipeline (h264_cuvid decoder + NVENC encoder)
    report_progress(0.2, "Encoding with full GPU pipeline...")
    cmd = build_ffmpeg_cmd_full_hw(nvenc_opts)
    success, stderr_output, returncode = run_ffmpeg(cmd, "NVENC-Full")

    if not success:
        # Tier 2: Full hardware failed, try simple NVENC (software decode + hardware encode)
        print(f"[DEBUG] Full GPU pipeline failed (code {returncode}), trying simple NVENC...")
        print(f"[DEBUG] Full GPU stderr: {stderr_output}")

        report_progress(0.25, "Trying simple NVENC (software decode)...")
        cmd = build_ffmpeg_cmd_simple_nvenc(nvenc_opts)
        success, stderr_output, returncode = run_ffmpeg(cmd, "NVENC-Simple")

        if not success:
            # Tier 3: NVENC failed entirely, use CPU fallback
            print(f"[DEBUG] Simple NVENC failed (code {returncode}), trying CPU fallback...")
            print(f"[DEBUG] Simple NVENC stderr: {stderr_output}")

            report_progress(0.30, "GPU failed, encoding with CPU (libx264)...")
            cmd = build_ffmpeg_cmd_cpu(libx264_opts)
            success, stderr_output, returncode = run_ffmpeg(cmd, "libx264")

            if not success:
                print(f"[DEBUG] libx264 also failed (code {returncode})")
                print(f"[DEBUG] libx264 stderr: {stderr_output}")

                # Last resort: try without audio
                if keep_audio and has_audio:
                    report_progress(0.35, "Trying without audio...")
                    # Rebuild command without audio
                    cmd = ['ffmpeg', '-y', '-i', input_path]
                    if filter_str:
                        cmd.extend(['-vf', filter_str])
                    cmd.extend([
                        '-c:v', 'libx264',
                        '-preset', 'ultrafast',
                        '-crf', '23',
                        '-pix_fmt', 'yuv420p',
                        '-an',  # No audio
                        '-r', f'{fps:.1f}',
                        '-movflags', '+faststart',
                        output_path
                    ])

                    success, stderr_output, returncode = run_ffmpeg(cmd, "libx264-no-audio")

                if not success:
                    raise RuntimeError(f"FFmpeg failed with all encoders (code {returncode}): {stderr_output}")

    report_progress(1.0, "Complete")

    return {
        'original_resolution': f"{original_width}x{original_height}",
        'duration': duration,
        'bitrate': f"{bitrate}kbps"
    }


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def find_perspective_coeffs(src_coords, dst_coords):
    """Calculate perspective transformation coefficients."""
    matrix = []
    for s, d in zip(src_coords, dst_coords):
        matrix.append([d[0], d[1], 1, 0, 0, 0, -s[0]*d[0], -s[0]*d[1]])
        matrix.append([0, 0, 0, d[0], d[1], 1, -s[1]*d[0], -s[1]*d[1]])

    A = np.array(matrix, dtype=np.float64)
    B = np.array([coord for pair in src_coords for coord in pair], dtype=np.float64)

    res = np.linalg.solve(A, B)
    return res.tolist()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor.py input_file output_file [copies]")
        sys.exit(1)

    copies = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    test_config = {
        'spatial': {'crop': 1.5, 'microResize': 1.2, 'rotation': 0.8},
        'tonal': {'brightness': 0.04, 'contrast': 0.04, 'saturation': 0.06},
        'visual': {'noise': 3},
        'compression': {'quality': 90, 'doubleCompress': 1},
        'options': {'copies': copies, 'force916': 1, 'flip': 1}
    }

    def progress(p, msg):
        print(f"[{int(p*100)}%] {msg}")

    result = process_spoofer(sys.argv[1], sys.argv[2], test_config, progress)
    print(f"Result: {result}")
