"""
Video Reframe Processor - Frame-by-Frame with Animated Blur Zones

Converts any aspect ratio video to vertical (9:16) with blur zones on TOP and BOTTOM.

Key features:
1. Video content stays CENTERED, scaled to fill the WIDTH
2. Blur zones on TOP and BOTTOM (not left/right)
3. Blur is generated FROM the video content itself
4. Blur ANIMATES - updates every ~20 frames with different random parameters
5. Optional logo overlay

For 9:16 output (1080x1920):
- Video content fills width (1080px)
- Height is scaled proportionally
- Blur fills the gaps above and below the video
"""

import os
import subprocess
import tempfile
import json
import random
import math
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List

# Import GPU utilities
try:
    from tools.gpu_utils import (
        get_gpu_count,
        assign_gpu,
        get_video_info,
        GPUManager
    )
except ImportError:
    try:
        from gpu_utils import (
            get_gpu_count,
            assign_gpu,
            get_video_info,
            GPUManager
        )
    except ImportError:
        get_gpu_count = lambda: 1
        assign_gpu = lambda x: 0
        get_video_info = None
        GPUManager = None


# =============================================================================
# BLUR EFFECTS CLASS
# =============================================================================

class BlurEffects:
    """Optimized blur and visual effects for video processing."""

    @staticmethod
    def gaussian_blur(image: np.ndarray, strength: int = 25) -> np.ndarray:
        """Apply Gaussian blur with specified strength."""
        if strength <= 0:
            return image
        k = int(strength) | 1  # Ensure odd kernel size
        return cv2.GaussianBlur(image, (k, k), 0)

    @staticmethod
    def flip_vertical(image: np.ndarray) -> np.ndarray:
        """Flip image vertically (mirror effect)."""
        return cv2.flip(image, 0)

    @staticmethod
    def flip_horizontal(image: np.ndarray) -> np.ndarray:
        """Flip image horizontally."""
        return cv2.flip(image, 1)

    @staticmethod
    def color_shift(image: np.ndarray, hue_shift: int) -> np.ndarray:
        """Shift hue values in image."""
        if hue_shift == 0:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + int(hue_shift)) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_saturation_brightness(image: np.ndarray, saturation: float = 1.0, brightness: float = 1.0) -> np.ndarray:
        """Adjust saturation and brightness."""
        if saturation == 1.0 and brightness == 1.0:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        if saturation != 1.0:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        if brightness != 1.0:
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def add_noise(image: np.ndarray, intensity: int) -> np.ndarray:
        """Add random noise to image."""
        if intensity <= 0:
            return image
        noise = np.random.randint(-intensity, intensity + 1, image.shape, dtype=np.int16)
        return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    @staticmethod
    def tilt(image: np.ndarray, angle: float, rotation_matrix: np.ndarray = None) -> np.ndarray:
        """Apply rotation/tilt to image."""
        if angle == 0:
            return image
        h, w = image.shape[:2]
        if rotation_matrix is None:
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def adjust_contrast(image: np.ndarray, contrast: float) -> np.ndarray:
        """Adjust image contrast."""
        if contrast == 1.0:
            return image
        return cv2.convertScaleAbs(image, alpha=contrast, beta=(1 - contrast) * 128)

    @staticmethod
    def darken(image: np.ndarray, factor: float = 0.7) -> np.ndarray:
        """Darken image by a factor."""
        if factor >= 1.0:
            return image
        return (image.astype(np.float32) * factor).astype(np.uint8)


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

def process_video_reframe(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    gpu_id: int = 0,
    video_index: int = 0
) -> dict:
    """
    Convert any video to vertical format with blur zones on TOP and BOTTOM.

    The video content is centered and fills the WIDTH. Blur zones fill the
    vertical gaps above and below the video content.

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        config: Configuration dict with:
            - aspectRatio: Target aspect ratio as string '9:16', '4:5', '1:1', '16:9'
            - logoName: 'farmium_icon' | 'farmium_full' | 'none' | URL
            - logoSize: Logo size as percentage (default: 15)
            - blurIntensity: Blur strength 1-100 (default: 25)
            - brightness: Brightness adjustment -50 to 50 (default: 0)
            - saturation: Saturation adjustment -100 to 100 (default: 0)
            - contrast: Contrast adjustment -50 to 50 (default: 0)
            - blurUpdateInterval: Frames between blur updates (default: 20)
        progress_callback: Optional callback(progress: float, message: str)
        gpu_id: GPU device ID (used for encoder selection)
        video_index: Index of video in batch (for random seed)

    Returns:
        dict with status, outputPath, dimensions, etc.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.05, "Analyzing video...")

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Check for audio
    has_audio = _check_audio(str(input_path))

    print(f"[VideoReframe] Input: {orig_w}x{orig_h} @ {fps:.2f} fps, {total_frames} frames, duration: {duration:.2f}s")

    # Parse config
    aspect_raw = config.get('aspectRatio', '9:16')
    if isinstance(aspect_raw, (list, tuple)) and len(aspect_raw) == 2:
        aspect_str = f"{aspect_raw[0]}:{aspect_raw[1]}"
    else:
        aspect_str = str(aspect_raw) if aspect_raw else '9:16'

    logo_name = config.get('logoName', 'farmium_full')
    logo_size = config.get('logoSize', 15)
    blur_intensity = config.get('blurIntensity', 25)
    brightness_adj = config.get('brightness', 0)
    saturation_adj = config.get('saturation', 0)
    contrast_adj = config.get('contrast', 0)
    blur_update_interval = config.get('blurUpdateInterval', 20)  # Update blur every N frames

    # Calculate output dimensions
    final_w, final_h = _get_output_dimensions(aspect_str)
    print(f"[VideoReframe] Output: {final_w}x{final_h} (aspect: {aspect_str})")

    # Calculate layout - content fills WIDTH, blur on TOP/BOTTOM
    layout = _calculate_layout(orig_w, orig_h, final_w, final_h)

    print(f"[VideoReframe] Layout: content={layout['scaled_w']}x{layout['scaled_h']}, "
          f"blur_top={layout['blur_top']}px, blur_bottom={layout['blur_bottom']}px")

    if progress_callback:
        progress_callback(0.10, "Preparing processing pipeline...")

    # Prepare logo
    logo_data = None
    if logo_name and logo_name != 'none':
        logo_data = _prepare_logo(logo_name, final_w, logo_size)
        if logo_data:
            print(f"[VideoReframe] Logo prepared: {logo_data['image'].shape[1]}x{logo_data['image'].shape[0]}")

    # Initialize blur effects
    effects = BlurEffects()

    # Random generator with seed for reproducibility
    rng = random.Random(video_index + 42)

    # Generate initial blur parameters
    blur_params = _generate_blur_params(rng)

    # Start FFmpeg process for output
    ffmpeg_process = _start_ffmpeg_writer(
        str(output_path), final_w, final_h, fps,
        str(input_path) if has_audio else None,
        gpu_id
    )

    if progress_callback:
        progress_callback(0.15, "Processing frames...")

    # Pre-allocate output frame buffer
    output_frame = np.zeros((final_h, final_w, 3), dtype=np.uint8)

    # Blur cache (only used between blur updates)
    cache_blur_top = None
    cache_blur_bottom = None

    # Extract layout values
    scaled_w = layout['scaled_w']
    scaled_h = layout['scaled_h']
    content_x = layout['content_x']
    content_y = layout['content_y']
    blur_top_h = layout['blur_top']
    blur_bottom_h = layout['blur_bottom']

    # Pre-compute rotation matrices if tilt is used
    rot_matrix_top = None
    rot_matrix_bottom = None

    frame_idx = 0
    encoder_used = 'NVENC'  # Will be updated if fallback

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Check if we need to update blur (every blur_update_interval frames)
            update_blur = (frame_idx % blur_update_interval == 0)

            if update_blur:
                # Generate new random parameters for animated blur effect
                blur_params = _generate_blur_params(rng)

                # Update rotation matrices
                if blur_params['tilt_angle'] != 0:
                    if blur_top_h > 0:
                        center_top = (final_w // 2, blur_top_h // 2)
                        rot_matrix_top = cv2.getRotationMatrix2D(center_top, blur_params['tilt_angle'], 1.0)
                    if blur_bottom_h > 0:
                        center_bottom = (final_w // 2, blur_bottom_h // 2)
                        rot_matrix_bottom = cv2.getRotationMatrix2D(center_bottom, -blur_params['tilt_angle'], 1.0)
                else:
                    rot_matrix_top = None
                    rot_matrix_bottom = None

            # Scale content to fit width
            content = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

            # Apply color adjustments to content if specified
            if brightness_adj != 0 or contrast_adj != 0 or saturation_adj != 0:
                content = _apply_color_adjustments(content, brightness_adj, saturation_adj, contrast_adj)

            # Generate blur zones FROM the video content
            if blur_top_h > 0:
                if update_blur or cache_blur_top is None:
                    cache_blur_top = _create_blur_zone(
                        effects, content, final_w, blur_top_h, 'top',
                        blur_intensity, blur_params, rot_matrix_top
                    )
                output_frame[0:blur_top_h, :] = cache_blur_top

            # Place main content in center
            output_frame[content_y:content_y + scaled_h, content_x:content_x + scaled_w] = content

            # Bottom blur zone
            if blur_bottom_h > 0:
                if update_blur or cache_blur_bottom is None:
                    cache_blur_bottom = _create_blur_zone(
                        effects, content, final_w, blur_bottom_h, 'bottom',
                        blur_intensity, blur_params, rot_matrix_bottom
                    )
                output_frame[content_y + scaled_h:, :] = cache_blur_bottom

            # Apply logo overlay
            if logo_data is not None:
                _apply_logo(output_frame, logo_data, final_w, final_h)

            # Write frame to FFmpeg
            try:
                ffmpeg_process.stdin.write(output_frame.tobytes())
            except BrokenPipeError:
                print("[VideoReframe] FFmpeg pipe closed unexpectedly")
                break

            frame_idx += 1

            # Progress update
            if progress_callback and (frame_idx % 30 == 0 or frame_idx == total_frames):
                progress = 0.15 + (frame_idx / total_frames) * 0.80
                progress_callback(min(progress, 0.95), f"Processing frame {frame_idx}/{total_frames}")

    finally:
        cap.release()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

    if progress_callback:
        progress_callback(1.0, "Complete")

    output_size = os.path.getsize(output_path) if output_path.exists() else 0

    return {
        "status": "completed",
        "outputPath": str(output_path),
        "outputSize": output_size,
        "dimensions": f"{final_w}x{final_h}",
        "encoder": encoder_used,
        "fps": fps,
        "duration": duration,
        "framesProcessed": frame_idx
    }


def _check_audio(video_path: str) -> bool:
    """Check if video has audio stream using ffprobe."""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return 'audio' in result.stdout.lower()
    except:
        return True  # Assume audio exists


def _get_output_dimensions(aspect_str: str) -> tuple:
    """Get output width/height for aspect ratio string."""
    aspect_map = {
        '9:16': (1080, 1920),
        '4:5': (1080, 1350),
        '1:1': (1080, 1080),
        '16:9': (1920, 1080),
        '4:3': (1440, 1080),
        '3:4': (1080, 1440),
    }

    if aspect_str in aspect_map:
        return aspect_map[aspect_str]

    # Parse custom aspect ratio
    try:
        if ':' in aspect_str:
            w, h = map(int, aspect_str.split(':'))
            final_w = 1080
            final_h = int(1080 * h / w)
            final_h = final_h - (final_h % 2)  # Ensure even
            return (final_w, final_h)
    except:
        pass

    return (1080, 1920)  # Default to 9:16


def _calculate_layout(orig_w: int, orig_h: int, final_w: int, final_h: int) -> dict:
    """
    Calculate layout for content and blur zones.

    KEY PRINCIPLE: Content fills the WIDTH, blur zones on TOP and BOTTOM.
    """
    # Scale content to fill the full width
    scale = final_w / orig_w
    scaled_w = final_w
    scaled_h = int(orig_h * scale)

    # Ensure even dimensions
    scaled_h = scaled_h - (scaled_h % 2)

    # If scaled height exceeds final height, scale down to fit height
    if scaled_h > final_h:
        scale = final_h / orig_h
        scaled_h = final_h
        scaled_w = int(orig_w * scale)
        scaled_w = scaled_w - (scaled_w % 2)

    # Calculate blur zones (vertical space above and below content)
    blur_space = final_h - scaled_h
    blur_top = blur_space // 2
    blur_bottom = blur_space - blur_top

    # Content position (centered)
    content_x = (final_w - scaled_w) // 2
    content_y = blur_top

    return {
        'scaled_w': scaled_w,
        'scaled_h': scaled_h,
        'content_x': content_x,
        'content_y': content_y,
        'blur_top': blur_top,
        'blur_bottom': blur_bottom,
        'needs_blur': blur_top > 0 or blur_bottom > 0
    }


def _generate_blur_params(rng: random.Random) -> dict:
    """Generate random parameters for animated blur effect."""
    return {
        'hflip': rng.random() < 0.3,  # 30% chance horizontal flip
        'vflip': rng.random() < 0.2,  # 20% chance vertical flip (for mirror effect)
        'color_shift': rng.randint(-15, 15),  # Hue shift
        'saturation': rng.uniform(0.7, 1.2),  # Saturation multiplier
        'brightness': rng.uniform(0.6, 0.9),  # Slightly darker blur
        'tilt_angle': rng.uniform(-8, 8),  # Rotation angle
        'zoom': rng.uniform(1.2, 1.5),  # Zoom factor for blur source
        'darken': rng.uniform(0.5, 0.8),  # Darken factor
    }


def _create_blur_zone(
    effects: BlurEffects,
    content: np.ndarray,
    target_w: int,
    target_h: int,
    position: str,  # 'top' or 'bottom'
    blur_strength: int,
    params: dict,
    rotation_matrix: np.ndarray = None
) -> np.ndarray:
    """
    Create a blur zone from video content.

    The blur is generated FROM the actual video content, not from a separate
    background layer. This creates a cohesive look where blur matches the video.
    """
    if target_h <= 0:
        return np.zeros((0, target_w, 3), dtype=np.uint8)

    content_h, content_w = content.shape[:2]

    # Extract source region from content (take 60-70% of content height)
    source_ratio = 0.65
    source_h = max(int(content_h * source_ratio), min(content_h, max(target_h * 2, 100)))
    source_h = max(10, min(source_h, content_h))

    if position == 'top':
        # For top blur, take from the top portion of content
        source = content[0:source_h, :].copy()
    else:
        # For bottom blur, take from the bottom portion of content
        start_y = max(0, content_h - source_h)
        source = content[start_y:, :].copy()

    # Apply zoom to fill blur zone with some overshoot
    zoom = max(target_w / source.shape[1], target_h / source.shape[0]) * params['zoom']
    zoomed_w = int(source.shape[1] * zoom)
    zoomed_h = int(source.shape[0] * zoom)

    if zoomed_w <= 0 or zoomed_h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    source = cv2.resize(source, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)

    # Crop to target size (center crop)
    crop_x = max(0, (zoomed_w - target_w) // 2)
    crop_y = max(0, (zoomed_h - target_h) // 2)
    blur_section = source[crop_y:crop_y + target_h, crop_x:crop_x + target_w]

    # Ensure exact dimensions
    if blur_section.shape[0] != target_h or blur_section.shape[1] != target_w:
        blur_section = cv2.resize(blur_section, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Apply horizontal flip for variety
    if params['hflip']:
        blur_section = effects.flip_horizontal(blur_section)

    # Apply vertical flip (mirror effect) for top zone
    if position == 'top' and params['vflip']:
        blur_section = effects.flip_vertical(blur_section)

    # Apply color adjustments
    if params['color_shift'] != 0:
        blur_section = effects.color_shift(blur_section, params['color_shift'])

    if params['saturation'] != 1.0 or params['brightness'] != 1.0:
        blur_section = effects.adjust_saturation_brightness(
            blur_section, params['saturation'], params['brightness']
        )

    # Apply rotation/tilt
    if params['tilt_angle'] != 0 and rotation_matrix is not None:
        blur_section = effects.tilt(blur_section, params['tilt_angle'], rotation_matrix)

    # Darken the blur to make content stand out
    blur_section = effects.darken(blur_section, params['darken'])

    # Apply the main blur effect
    if blur_strength > 0:
        blur_section = effects.gaussian_blur(blur_section, blur_strength)

    return blur_section


def _apply_color_adjustments(
    image: np.ndarray,
    brightness: int,
    saturation: int,
    contrast: int
) -> np.ndarray:
    """Apply brightness, saturation, and contrast adjustments."""
    result = image

    # Convert percentages to multipliers
    if brightness != 0 or saturation != 0:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)

        if saturation != 0:
            sat_mult = 1.0 + (saturation / 100.0)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mult, 0, 255)

        if brightness != 0:
            br_mult = 1.0 + (brightness / 100.0)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * br_mult, 0, 255)

        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if contrast != 0:
        ct_mult = 1.0 + (contrast / 100.0)
        result = cv2.convertScaleAbs(result, alpha=ct_mult, beta=(1 - ct_mult) * 128)

    return result


def _prepare_logo(logo_name: str, video_width: int, size_percent: float) -> Optional[dict]:
    """
    Prepare logo for overlay.
    Returns dict with 'image' (BGR), 'alpha' (float mask), and dimensions.
    """
    try:
        from PIL import Image

        # Multiple search paths for logos
        workspace = os.environ.get('WORKSPACE', '/workspace')
        script_dir = Path(__file__).parent.resolve()

        search_paths = [
            Path(workspace) / 'assets' / 'logos',
            Path('/workspace/assets/logos'),
            script_dir / 'assets' / 'logos',
            script_dir.parent / 'assets' / 'logos',
            script_dir.parent.parent / 'assets' / 'logos',
            Path('./assets/logos'),
            Path('../assets/logos'),
        ]

        logo_source = None

        if logo_name in ['farmium_icon', 'farmium_full']:
            for search_path in search_paths:
                png_path = search_path / f'{logo_name}.png'
                if png_path.exists():
                    logo_source = str(png_path)
                    break

        elif logo_name.startswith('http://') or logo_name.startswith('https://'):
            import requests
            response = requests.get(logo_name, timeout=30)
            response.raise_for_status()
            temp_logo = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_logo.write(response.content)
            temp_logo.close()
            logo_source = temp_logo.name

        elif Path(logo_name).exists():
            logo_source = logo_name

        if not logo_source:
            print(f"[VideoReframe] Logo not found: {logo_name}")
            return None

        # Load and resize logo
        pil_logo = Image.open(logo_source).convert('RGBA')

        logo_w = max(int(video_width * size_percent / 100), 80)
        logo_h = int(logo_w * pil_logo.height / pil_logo.width)

        pil_logo = pil_logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)

        # Convert to numpy
        logo_array = np.array(pil_logo)
        logo_bgr = cv2.cvtColor(logo_array[:, :, :3], cv2.COLOR_RGB2BGR)
        logo_alpha = logo_array[:, :, 3].astype(np.float32) / 255.0

        return {
            'image': logo_bgr,
            'alpha': logo_alpha,
            'alpha_3d': logo_alpha[:, :, np.newaxis]
        }

    except Exception as e:
        print(f"[VideoReframe] Error preparing logo: {e}")
        return None


def _apply_logo(frame: np.ndarray, logo_data: dict, frame_w: int, frame_h: int):
    """Apply logo overlay to frame (bottom center with 5% margin)."""
    logo = logo_data['image']
    alpha_3d = logo_data['alpha_3d']
    lh, lw = logo.shape[:2]

    # Position: bottom center, 5% from bottom
    margin = int(frame_h * 0.05)
    x = (frame_w - lw) // 2
    y = frame_h - lh - margin

    # Ensure within bounds
    x = max(0, min(frame_w - lw, x))
    y = max(0, min(frame_h - lh, y))

    # Blend logo with frame using alpha
    roi = frame[y:y + lh, x:x + lw]
    blended = (alpha_3d * logo + (1 - alpha_3d) * roi).astype(np.uint8)
    frame[y:y + lh, x:x + lw] = blended


def _start_ffmpeg_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: str = None,
    gpu_id: int = 0
) -> subprocess.Popen:
    """
    Start FFmpeg subprocess for writing video frames.
    Tries NVENC first, falls back to libx264.
    """
    import platform

    # Determine encoder based on platform
    system_os = platform.system()

    # Build command
    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(min(fps, 60)),
        '-thread_queue_size', '512',
        '-i', '-',  # Video from stdin
    ]

    # Add audio input if available
    if audio_source:
        cmd.extend(['-i', audio_source])

    # Video encoding settings - try NVENC first on Linux/Windows
    if system_os in ['Linux', 'Windows']:
        # Try NVENC
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-gpu', str(gpu_id),
            '-preset', 'p4',  # Balanced preset
            '-rc', 'vbr',
            '-cq', '23',
            '-b:v', '8000k',
            '-maxrate', '12000k',
            '-bufsize', '16000k',
        ])
    elif system_os == 'Darwin':
        # macOS - use VideoToolbox
        cmd.extend([
            '-c:v', 'h264_videotoolbox',
            '-b:v', '8000k',
            '-allow_sw', '1',
        ])
    else:
        # Fallback to CPU
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
        ])

    cmd.extend(['-pix_fmt', 'yuv420p'])

    # Map streams
    cmd.extend(['-map', '0:v'])

    if audio_source:
        cmd.extend([
            '-map', '1:a?',
            '-c:a', 'aac',
            '-b:a', '128k',
            '-shortest',
        ])
    else:
        cmd.append('-an')

    cmd.extend([
        '-movflags', '+faststart',
        output_path
    ])

    print(f"[VideoReframe] Starting FFmpeg: {' '.join(cmd[:15])}...")

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return process
    except Exception as e:
        print(f"[VideoReframe] FFmpeg start failed, trying CPU fallback: {e}")

        # Fallback to CPU encoder
        cmd_cpu = [
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(min(fps, 60)),
            '-i', '-',
        ]

        if audio_source:
            cmd_cpu.extend(['-i', audio_source])

        cmd_cpu.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-map', '0:v',
        ])

        if audio_source:
            cmd_cpu.extend(['-map', '1:a?', '-c:a', 'aac', '-b:a', '128k', '-shortest'])
        else:
            cmd_cpu.append('-an')

        cmd_cpu.extend(['-movflags', '+faststart', output_path])

        return subprocess.Popen(
            cmd_cpu,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_video_reframe_batch(
    input_paths: list,
    output_dir: str,
    config: dict,
    progress_callback: Optional[Callable] = None,
    gpu_id: int = 0
) -> list:
    """
    Process multiple videos with reframe.

    Args:
        input_paths: List of input video paths
        output_dir: Directory for output files
        config: Shared config for all videos
        progress_callback: Progress callback
        gpu_id: GPU to use

    Returns:
        List of result dicts
    """
    results = []
    total = len(input_paths)

    for i, input_path in enumerate(input_paths):
        if progress_callback:
            base_progress = i / total
            progress_callback(base_progress, f"Processing video {i+1}/{total}")

        input_name = Path(input_path).stem
        output_path = Path(output_dir) / f"{input_name}_reframed.mp4"

        try:
            def sub_progress(p, msg):
                if progress_callback:
                    overall = (i + p) / total
                    progress_callback(overall, f"[{i+1}/{total}] {msg}")

            result = process_video_reframe(
                input_path,
                str(output_path),
                config,
                sub_progress,
                gpu_id,
                video_index=i
            )
            results.append(result)

        except Exception as e:
            print(f"[VideoReframe] Error processing {input_path}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "status": "failed",
                "error": str(e),
                "inputPath": input_path
            })

    return results


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor.py <input.mp4> <output.mp4> [aspect_ratio] [logo_name]")
        print("Example: python processor.py video.mp4 out.mp4 9:16")
        print("Example with logo: python processor.py video.mp4 out.mp4 9:16 farmium_full")
        print("")
        print("Logo options: farmium_icon, farmium_full, none, or URL")
        print("")
        print("This processor creates blur zones on TOP and BOTTOM of the video,")
        print("with the video content filling the width and centered vertically.")
        print("Blur animates every 20 frames for visual interest.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    aspect = sys.argv[3] if len(sys.argv) > 3 else '9:16'
    logo = sys.argv[4] if len(sys.argv) > 4 else 'none'

    config = {
        'aspectRatio': aspect,
        'logoName': logo,
        'logoSize': 15,
        'blurIntensity': 25,
        'blurUpdateInterval': 20,  # Update blur every 20 frames (~0.66s at 30fps)
    }

    print(f"[CLI] Input: {input_file}")
    print(f"[CLI] Output: {output_file}")
    print(f"[CLI] Aspect: {aspect}")
    print(f"[CLI] Logo: {logo}")
    print(f"[CLI] Blur update interval: {config['blurUpdateInterval']} frames")

    def progress(p, msg):
        print(f"[{int(p*100):3d}%] {msg}")

    result = process_video_reframe(input_file, output_file, config, progress)
    print(f"\nResult: {result}")
