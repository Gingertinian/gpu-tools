"""
Video Processor Module

Contains all video processing functions:
- process_video_with_blur_logo: Frame-by-frame processing with blur background and logo
- process_single_video_worker: Worker function for parallel video processing
- process_videos_parallel: Process multiple videos in parallel using NVENC
- process_batch_video: Process video in batch mode with multiple variations
- process_video: Process single video with FFmpeg + NVENC

Helper functions:
- _check_video_audio: Check if video has audio stream
- _apply_rotation_with_zoom: Apply rotation with zoom compensation
- _apply_spoofer_color_adj: Apply color adjustments to frame
- _generate_spoofer_blur_params: Generate random blur parameters
- _create_spoofer_blur_zone: Create blur zone from video content
- _svg_to_cv2: Convert SVG to CV2 image
- _prepare_spoofer_logo: Prepare logo for overlay
- _apply_spoofer_logo: Apply logo overlay to frame
- _start_spoofer_ffmpeg: Start FFmpeg for frame-by-frame writing
"""

import os
import time
import math
import random
import json
import zipfile
import subprocess
import threading
import tempfile
import shutil
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .constants import NVENC_SESSION_LIMITS, VIDEO_EXTENSIONS
from .utils import apply_mode_to_config
from .gpu_session_manager import get_gpu_info, NVENCSessionTracker


# =============================================================================
# HELPER FUNCTIONS FOR VIDEO PROCESSING
# =============================================================================

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
            # For small brightness values (+/-0.05 range), use additive adjustment
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
        # Scale: config 5 -> +/-0.05 brightness adjustment
        brightness_adj = py_rng.uniform(-tonal['brightness'], tonal['brightness']) * 0.01

    contrast_adj = 1.0
    if tonal.get('contrast', 0) > 0:
        # Scale: config 5 -> +/-0.05 contrast variation (0.95 to 1.05)
        contrast_adj = 1 + py_rng.uniform(-tonal['contrast'], tonal['contrast']) * 0.01

    saturation_adj = 1.0
    if tonal.get('saturation', 0) > 0:
        # Scale: config 5 -> +/-0.05 saturation variation (0.95 to 1.05)
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

        # Apply mode multipliers to config before building filters
        config = apply_mode_to_config(config)

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

                # Pipeline: scale up -> rotate -> crop center
                filters.append(f"scale={scaled_w}:{scaled_h}:flags=lanczos")
                filters.append(f"rotate={angle_rad}:fillcolor=black")
                filters.append(f"crop={original_width}:{original_height}")

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
                cmd.extend(['-vf', f'hwdownload,format=nv12,{filter_str},hwupload_cuda'])

            if use_nvenc:
                # NVENC encoding with MAXIMUM THROUGHPUT settings
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
                    '-spatial-aq', '0',
                    '-temporal-aq', '0',
                ])
            else:
                # CPU fallback with libx264 ultrafast
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

        def _build_ffmpeg_cmd_nvenc_simple(target_gpu: int = 0) -> list:
            """Build simpler NVENC command without hardware decode (fallback)."""
            cmd = ['ffmpeg', '-y']
            cmd.extend([
                '-hwaccel', 'cuda',
                '-hwaccel_device', str(target_gpu),
            ])
            cmd.extend(['-i', input_path])

            if filter_str:
                cmd.extend(['-vf', filter_str])

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

        # Try encoding with 3-tier fallback
        print(f"[Spoofer Worker {video_index}] Using GPU {gpu_id} with full hardware pipeline")

        # Tier 1: Try full hardware pipeline
        process = subprocess.run(nvenc_cmd, capture_output=True, text=True, timeout=900)

        if process.returncode != 0:
            # Tier 2: Full hardware failed, try simple NVENC
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

    Example: 20 videos x 100 variations = 2,000 unique output videos

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

    # Calculate optimal worker count
    base_workers = max_parallel
    WORKER_BUFFER_MULTIPLIER = 1.2
    max_workers = int(base_workers * WORKER_BUFFER_MULTIPLIER)
    max_workers = max(max_workers, base_workers)

    # Calculate theoretical throughput
    theoretical_throughput = max_parallel / 2.0

    # Calculate total work items
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

    # Prepare work items
    work_items = []
    for video_idx, video_path in enumerate(video_paths):
        basename = os.path.basename(video_path)
        name, ext = os.path.splitext(basename)

        for var_idx in range(variations):
            work_index = video_idx * variations + var_idx

            if variations > 1:
                output_filename = f"{name}_var{var_idx:04d}{ext}"
            else:
                output_filename = f"{name}_spoofed{ext}"

            output_path = os.path.join(output_dir, output_filename)
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
            full_work_item = (video_path, output_path, cfg, video_index, gpu_id)
            return process_single_video_worker(full_work_item)
        finally:
            session_tracker.release_gpu(gpu_id)

    # Process in parallel using ThreadPoolExecutor
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


def process_batch_video(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    variations: int,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process video in batch mode - create multiple variations.
    Uses parallel NVENC sessions for maximum throughput on datacenter GPUs.

    Args:
        input_path: Path to input video
        output_path: Output directory path (when outputMode='directory') or base path
        config: Processing configuration
        variations: Number of variations to create
        progress_callback: Progress callback function
    """
    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    variations = max(1, int(variations))
    report_progress(0.05, f"Preparing batch video processing ({variations} variations)...")

    # Determine output directory
    output_mode = config.get('outputMode', 'file')
    output_dir = config.get('outputDir')

    if output_mode == 'directory' and output_dir:
        out_dir = output_dir
    elif os.path.isdir(output_path):
        out_dir = output_path
    else:
        out_dir = tempfile.mkdtemp(prefix="video_batch_")

    os.makedirs(out_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    report_progress(0.1, f"Processing {variations} video variations in parallel...")

    # Use process_videos_parallel for efficient multi-NVENC processing
    result = process_videos_parallel(
        [input_path],
        out_dir,
        config,
        progress_callback=progress_callback,
        max_parallel=None,
        variations=variations
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
        'variations_requested': variations,
        'videos_processed': completed,
        'videos_failed': failed,
        'output_files': output_files,
        'output_dir': out_dir if output_mode == 'directory' else None
    }


def process_video(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Process video with FFmpeg + NVENC (fast preset)."""

    # Ensure output path has video extension
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext not in ['.mp4', '.mov', '.avi', '.mkv', '.webm']:
        output_path = output_path + '.mp4'
        print(f"[DEBUG] Added .mp4 extension to output: {output_path}")

    report_progress(0.1, "Analyzing video...")

    # Get video info
    try:
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

    gpu_id = config.get('_gpu_id', 0)
    if gpu_id > 0:
        print(f"[Video Processing] Using GPU {gpu_id} for NVENC encoding")

    def build_ffmpeg_cmd_full_hw(encoder_opts: list) -> list:
        """Build FFmpeg command with FULL hardware pipeline."""
        cmd = ['ffmpeg', '-y']
        cmd.extend([
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
            '-hwaccel_output_format', 'cuda',
            '-c:v', 'h264_cuvid',
        ])
        cmd.extend(['-i', input_path])

        if filter_str:
            cmd.extend(['-vf', f'hwdownload,format=nv12,{filter_str},hwupload_cuda'])

        cmd.extend(['-c:v', 'h264_nvenc'])
        cmd.extend(encoder_opts)

        if keep_audio and has_audio:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k', '-strict', 'experimental'])
        else:
            cmd.append('-an')

        cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])
        return cmd

    def build_ffmpeg_cmd_simple_nvenc(encoder_opts: list) -> list:
        """Build FFmpeg command with software decode + NVENC encode."""
        cmd = ['ffmpeg', '-y']
        cmd.extend([
            '-hwaccel', 'cuda',
            '-hwaccel_device', str(gpu_id),
        ])
        cmd.extend(['-i', input_path])

        if filter_str:
            cmd.extend(['-vf', filter_str])

        cmd.extend(['-c:v', 'h264_nvenc'])
        cmd.extend(encoder_opts)

        if keep_audio and has_audio:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k', '-strict', 'experimental'])
        else:
            cmd.append('-an')

        cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])
        return cmd

    def build_ffmpeg_cmd_cpu(encoder_opts: list) -> list:
        """Build FFmpeg command with CPU encoding (fallback)."""
        cmd = ['ffmpeg', '-y', '-i', input_path]

        if filter_str:
            cmd.extend(['-vf', filter_str])

        cmd.extend(['-c:v', 'libx264'])
        cmd.extend(encoder_opts)

        if keep_audio and has_audio:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k', '-strict', 'experimental'])
        else:
            cmd.append('-an')

        cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])
        return cmd

    # NVENC encoder options
    nvenc_opts = [
        '-gpu', str(gpu_id),
        '-preset', 'p1',
        '-tune', 'll',
        '-rc', 'vbr',
        '-cq', '26',
        '-b:v', f'{bitrate}k',
        '-maxrate', f'{int(bitrate * 2)}k',
        '-bufsize', f'{bitrate * 2}k',
        '-rc-lookahead', '0',
        '-bf', '0',
        '-spatial-aq', '0',
        '-temporal-aq', '0',
    ]

    # libx264 (CPU) encoder options
    libx264_opts = [
        '-preset', 'ultrafast',
        '-crf', '23',
        '-b:v', f'{bitrate}k',
        '-maxrate', f'{int(bitrate * 2)}k',
        '-bufsize', f'{bitrate * 2}k',
        '-pix_fmt', 'yuv420p',
    ]

    # Tier 1: Try FULL hardware pipeline
    report_progress(0.2, "Encoding with full GPU pipeline...")
    cmd = build_ffmpeg_cmd_full_hw(nvenc_opts)
    success, stderr_output, returncode = run_ffmpeg(cmd, "NVENC-Full")

    if not success:
        # Tier 2: Try simple NVENC
        print(f"[DEBUG] Full GPU pipeline failed (code {returncode}), trying simple NVENC...")
        report_progress(0.25, "Trying simple NVENC (software decode)...")
        cmd = build_ffmpeg_cmd_simple_nvenc(nvenc_opts)
        success, stderr_output, returncode = run_ffmpeg(cmd, "NVENC-Simple")

        if not success:
            # Tier 3: CPU fallback
            print(f"[DEBUG] Simple NVENC failed (code {returncode}), trying CPU fallback...")
            report_progress(0.30, "GPU failed, encoding with CPU (libx264)...")
            cmd = build_ffmpeg_cmd_cpu(libx264_opts)
            success, stderr_output, returncode = run_ffmpeg(cmd, "libx264")

            if not success:
                # Last resort: try without audio
                if keep_audio and has_audio:
                    report_progress(0.35, "Trying without audio...")
                    cmd = ['ffmpeg', '-y', '-i', input_path]
                    if filter_str:
                        cmd.extend(['-vf', filter_str])
                    cmd.extend([
                        '-c:v', 'libx264',
                        '-preset', 'ultrafast',
                        '-crf', '23',
                        '-pix_fmt', 'yuv420p',
                        '-an',
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
