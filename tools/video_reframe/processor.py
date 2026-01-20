"""
Video/Image Reframe Processor - GPU-Accelerated with CuPy

Converts any aspect ratio video OR IMAGE to vertical (9:16) with blur zones on TOP and BOTTOM.

Key features:
1. FULL GPU PROCESSING with CuPy (when available)
2. Supports both VIDEO and IMAGE inputs
3. Image input → Image output (JPG/PNG)
4. Video input → Video output (MP4)
5. Video content stays CENTERED, scaled to fill the WIDTH
6. Blur zones on TOP and BOTTOM (not left/right)
7. Blur is generated FROM the content itself
8. Blur ANIMATES (for videos) - updates every ~20 frames
9. Optional logo overlay
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
from typing import Optional, Callable, Dict, Any, List, Union

# Import GPU operations module
try:
    from .gpu_ops import (
        GPUProcessor, GPUBlurEffects, get_gpu_processor,
        is_gpu_available, HAS_CUPY
    )
except ImportError:
    try:
        from gpu_ops import (
            GPUProcessor, GPUBlurEffects, get_gpu_processor,
            is_gpu_available, HAS_CUPY
        )
    except ImportError:
        HAS_CUPY = False
        GPUProcessor = None
        GPUBlurEffects = None
        get_gpu_processor = None
        is_gpu_available = lambda: False

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
# MEDIA TYPE DETECTION
# =============================================================================

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.wmv', '.flv'}


# =============================================================================
# CONFIG HELPERS
# =============================================================================

def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def _get_config(config: dict, key: str, default=None):
    """
    Get config value. Handles 0 and False values correctly.
    Supports both camelCase and snake_case keys for compatibility with backend conversion.

    Backend may send either:
    - camelCase: topBlurPercent (direct from frontend)
    - snake_case: top_blur_percent (converted by getProcessingConfig)
    """
    # Try camelCase first
    if key in config and config[key] is not None:
        return config[key]
    # Try snake_case version
    snake_key = _camel_to_snake(key)
    if snake_key in config and config[snake_key] is not None:
        return config[snake_key]
    return default


def _parse_aspect_ratio(config: dict) -> str:
    """Parse aspect ratio from config, supporting multiple formats."""
    aspect_raw = _get_config(config, 'aspectRatio', '9:16')
    if isinstance(aspect_raw, (list, tuple)) and len(aspect_raw) == 2:
        return f"{aspect_raw[0]}:{aspect_raw[1]}"
    return str(aspect_raw) if aspect_raw else '9:16'


def _parse_logo_position(config: dict) -> tuple:
    """
    Parse logo position from config.
    Returns (x, y) as normalized values (0-1).
    """
    # Try flat keys first (logoPositionX, logoPositionY)
    pos_x = _get_config(config, 'logoPositionX')
    pos_y = _get_config(config, 'logoPositionY')

    # If flat keys not found, try nested logoPosition object
    if pos_x is None or pos_y is None:
        logo_pos = _get_config(config, 'logoPosition', {})
        if isinstance(logo_pos, dict):
            pos_x = logo_pos.get('x', 0.5) if pos_x is None else pos_x
            pos_y = logo_pos.get('y', 0.85) if pos_y is None else pos_y

    # Default values if still not found
    pos_x = float(pos_x) if pos_x is not None else 0.5
    pos_y = float(pos_y) if pos_y is not None else 0.85

    return (pos_x, pos_y)


def is_image_file(path: str) -> bool:
    """Check if file is an image based on extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: str) -> bool:
    """Check if file is a video based on extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def get_output_extension(input_path: str, output_path: str) -> str:
    """Determine appropriate output extension."""
    input_ext = Path(input_path).suffix.lower()
    output_ext = Path(output_path).suffix.lower()

    # If input is image, output should be image
    if input_ext in IMAGE_EXTENSIONS:
        if output_ext in IMAGE_EXTENSIONS:
            return output_ext
        return '.jpg'  # Default image format

    # If input is video, output should be video
    return output_ext if output_ext in VIDEO_EXTENSIONS else '.mp4'


# =============================================================================
# CPU FALLBACK BLUR EFFECTS
# =============================================================================

class BlurEffects:
    """CPU-based blur effects (fallback when GPU not available)."""

    @staticmethod
    def gaussian_blur(image: np.ndarray, strength: int = 25) -> np.ndarray:
        if strength <= 0:
            return image
        k = int(strength) | 1
        return cv2.GaussianBlur(image, (k, k), 0)

    @staticmethod
    def flip_vertical(image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 0)

    @staticmethod
    def flip_horizontal(image: np.ndarray) -> np.ndarray:
        return cv2.flip(image, 1)

    @staticmethod
    def color_shift(image: np.ndarray, hue_shift: int) -> np.ndarray:
        if hue_shift == 0:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + int(hue_shift)) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    @staticmethod
    def adjust_saturation_brightness(image: np.ndarray, saturation: float = 1.0, brightness: float = 1.0) -> np.ndarray:
        if saturation == 1.0 and brightness == 1.0:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        if saturation != 1.0:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        if brightness != 1.0:
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def tilt(image: np.ndarray, angle: float, rotation_matrix: np.ndarray = None) -> np.ndarray:
        if angle == 0:
            return image
        h, w = image.shape[:2]
        if rotation_matrix is None:
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    @staticmethod
    def adjust_contrast(image: np.ndarray, contrast: float) -> np.ndarray:
        if contrast == 1.0:
            return image
        return cv2.convertScaleAbs(image, alpha=contrast, beta=(1 - contrast) * 128)

    @staticmethod
    def darken(image: np.ndarray, factor: float = 0.7) -> np.ndarray:
        if factor >= 1.0:
            return image
        return (image.astype(np.float32) * factor).astype(np.uint8)


# =============================================================================
# UNIFIED PROCESSOR (GPU + IMAGE SUPPORT)
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
    Convert any video OR IMAGE to vertical format with blur zones on TOP and BOTTOM.

    Automatically detects if input is image or video:
    - IMAGE input → IMAGE output (JPG/PNG)
    - VIDEO input → VIDEO output (MP4)

    Args:
        input_path: Path to input video or image file
        output_path: Path to output file
        config: Configuration dict with:
            - aspectRatio: Target aspect ratio as string '9:16', '4:5', '1:1', '16:9'
            - logoName: 'farmium_icon' | 'farmium_full' | 'none' | URL
            - logoSize: Logo size as percentage (default: 15)
            - blurIntensity: Blur strength 1-100 (default: 25)
            - forceBlur: Force blur % for same-aspect-ratio (e.g., 25 = 25% top + 25% bottom)
            - brightness: Brightness adjustment -50 to 50 (default: 0)
            - saturation: Saturation adjustment -100 to 100 (default: 0)
            - contrast: Contrast adjustment -50 to 50 (default: 0)
            - blurUpdateInterval: Frames between blur updates (default: 20)
        progress_callback: Optional callback(progress: float, message: str)
        gpu_id: GPU device ID
        video_index: Index in batch (for random seed)

    Returns:
        dict with status, outputPath, dimensions, type ('image' or 'video'), etc.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Detect input type
    if is_image_file(str(input_path)):
        return _process_image_reframe(
            input_path, output_path, config, progress_callback, gpu_id, video_index
        )
    else:
        return _process_video_reframe(
            input_path, output_path, config, progress_callback, gpu_id, video_index
        )


def _process_image_reframe(
    input_path: Path,
    output_path: Path,
    config: dict,
    progress_callback: Optional[Callable] = None,
    gpu_id: int = 0,
    video_index: int = 0
) -> dict:
    """
    Process a single IMAGE with reframe (GPU-accelerated).

    Output is an IMAGE file (JPG/PNG), not a video.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.05, "Loading image...")

    # Read image
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    orig_h, orig_w = image.shape[:2]
    print(f"[ImageReframe] Input: {orig_w}x{orig_h}")

    # Parse config using helper that supports both camelCase and snake_case
    aspect_str = _parse_aspect_ratio(config)

    logo_name = _get_config(config, 'logoName', 'farmium_full')
    logo_url = _get_config(config, 'logoUrl')  # Custom logo URL from backend
    logo_size = _get_config(config, 'logoSize', 15)
    logo_position = _parse_logo_position(config)
    blur_intensity = _get_config(config, 'blurIntensity', 25)
    brightness_adj = _get_config(config, 'brightness', 0)
    saturation_adj = _get_config(config, 'saturation', 0)
    contrast_adj = _get_config(config, 'contrast', 0)

    # Blur percentages - support individual top/bottom
    top_blur_percent = _get_config(config, 'topBlurPercent', 0)
    bottom_blur_percent = _get_config(config, 'bottomBlurPercent', 0)
    force_blur_percent = _get_config(config, 'forceBlur', 0)

    # Log parsed config
    print(f"[ImageReframe] Config: logo={logo_name}, logoUrl={logo_url}, size={logo_size}%, pos=({logo_position[0]:.2f}, {logo_position[1]:.2f})")
    print(f"[ImageReframe] Blur: intensity={blur_intensity}, top={top_blur_percent}%, bottom={bottom_blur_percent}%")

    # Calculate output dimensions
    final_w, final_h = _get_output_dimensions(aspect_str)
    print(f"[ImageReframe] Output: {final_w}x{final_h} (aspect: {aspect_str})")

    # Calculate layout with individual blur percentages
    layout = _calculate_layout(orig_w, orig_h, final_w, final_h, force_blur_percent, top_blur_percent, bottom_blur_percent)

    print(f"[ImageReframe] Layout: content={layout['scaled_w']}x{layout['scaled_h']}, "
          f"blur_top={layout['blur_top']}px, blur_bottom={layout['blur_bottom']}px")

    if progress_callback:
        progress_callback(0.15, "Initializing GPU...")

    # Initialize GPU processor
    use_gpu = is_gpu_available() and HAS_CUPY
    if use_gpu:
        gpu = get_gpu_processor(gpu_id)
        effects = GPUBlurEffects(gpu)
        print(f"[ImageReframe] Using GPU acceleration (device {gpu_id})")
    else:
        gpu = None
        effects = BlurEffects()
        print("[ImageReframe] Using CPU processing")

    if progress_callback:
        progress_callback(0.20, "Processing image...")

    # Prepare logo - use logoUrl if provided (custom user logo), otherwise use logoName
    logo_data = None
    logo_source = logo_url if logo_url else logo_name
    if logo_source and logo_source != 'none':
        logo_data = _prepare_logo(logo_source, final_w, logo_size)
        if logo_data:
            # Store logo position in logo_data for _apply_logo
            logo_data['position'] = logo_position

    # Random generator
    rng = random.Random(video_index + 42)
    blur_params = _generate_blur_params(rng)

    # Process image
    if use_gpu:
        output_frame = _process_frame_gpu(
            gpu, effects, image, layout, final_w, final_h,
            blur_intensity, blur_params,
            brightness_adj, saturation_adj, contrast_adj,
            logo_data
        )
    else:
        output_frame = _process_frame_cpu(
            effects, image, layout, final_w, final_h,
            blur_intensity, blur_params,
            brightness_adj, saturation_adj, contrast_adj,
            logo_data
        )

    if progress_callback:
        progress_callback(0.80, "Saving image...")

    # Determine output format and quality
    output_ext = output_path.suffix.lower()
    if output_ext not in IMAGE_EXTENSIONS:
        # Force image extension
        output_path = output_path.with_suffix('.jpg')
        output_ext = '.jpg'

    # Save image with appropriate quality
    if output_ext in ['.jpg', '.jpeg']:
        quality = config.get('quality', 95)
        cv2.imwrite(str(output_path), output_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif output_ext == '.png':
        compression = config.get('compression', 3)  # 0-9, lower = faster
        cv2.imwrite(str(output_path), output_frame, [cv2.IMWRITE_PNG_COMPRESSION, compression])
    elif output_ext == '.webp':
        quality = config.get('quality', 95)
        cv2.imwrite(str(output_path), output_frame, [cv2.IMWRITE_WEBP_QUALITY, quality])
    else:
        cv2.imwrite(str(output_path), output_frame)

    if progress_callback:
        progress_callback(1.0, "Complete")

    output_size = os.path.getsize(output_path) if output_path.exists() else 0

    return {
        "status": "completed",
        "type": "image",
        "outputPath": str(output_path),
        "outputSize": output_size,
        "dimensions": f"{final_w}x{final_h}",
        "inputDimensions": f"{orig_w}x{orig_h}",
        "processor": "GPU" if use_gpu else "CPU"
    }


def _process_video_reframe(
    input_path: Path,
    output_path: Path,
    config: dict,
    progress_callback: Optional[Callable] = None,
    gpu_id: int = 0,
    video_index: int = 0
) -> dict:
    """
    Process a VIDEO with reframe (GPU-accelerated).
    """
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

    # Parse config using helper that supports both camelCase and snake_case
    aspect_str = _parse_aspect_ratio(config)

    logo_name = _get_config(config, 'logoName', 'farmium_full')
    logo_url = _get_config(config, 'logoUrl')  # Custom logo URL from backend
    logo_size = _get_config(config, 'logoSize', 15)
    logo_position = _parse_logo_position(config)
    blur_intensity = _get_config(config, 'blurIntensity', 25)
    brightness_adj = _get_config(config, 'brightness', 0)
    saturation_adj = _get_config(config, 'saturation', 0)
    contrast_adj = _get_config(config, 'contrast', 0)
    blur_update_interval = _get_config(config, 'blurUpdateInterval', 20)

    # Blur percentages - support individual top/bottom
    top_blur_percent = _get_config(config, 'topBlurPercent', 0)
    bottom_blur_percent = _get_config(config, 'bottomBlurPercent', 0)
    force_blur_percent = _get_config(config, 'forceBlur', 0)

    # Log parsed config
    print(f"[VideoReframe] Config: logo={logo_name}, logoUrl={logo_url}, size={logo_size}%, pos=({logo_position[0]:.2f}, {logo_position[1]:.2f})")
    print(f"[VideoReframe] Blur: intensity={blur_intensity}, top={top_blur_percent}%, bottom={bottom_blur_percent}%")

    # Calculate output dimensions
    final_w, final_h = _get_output_dimensions(aspect_str)
    print(f"[VideoReframe] Output: {final_w}x{final_h} (aspect: {aspect_str})")

    # Calculate layout with individual blur percentages
    layout = _calculate_layout(orig_w, orig_h, final_w, final_h, force_blur_percent, top_blur_percent, bottom_blur_percent)

    print(f"[VideoReframe] Layout: content={layout['scaled_w']}x{layout['scaled_h']}, "
          f"blur_top={layout['blur_top']}px, blur_bottom={layout['blur_bottom']}px")

    if progress_callback:
        progress_callback(0.10, "Preparing processing pipeline...")

    # Initialize GPU processor
    use_gpu = is_gpu_available() and HAS_CUPY
    if use_gpu:
        gpu = get_gpu_processor(gpu_id)
        effects = GPUBlurEffects(gpu)
        print(f"[VideoReframe] Using GPU acceleration (device {gpu_id})")
    else:
        gpu = None
        effects = BlurEffects()
        print("[VideoReframe] Using CPU processing")

    # Prepare logo - use logoUrl if provided (custom user logo), otherwise use logoName
    logo_data = None
    logo_source = logo_url if logo_url else logo_name
    if logo_source and logo_source != 'none':
        logo_data = _prepare_logo(logo_source, final_w, logo_size)
        if logo_data:
            # Store logo position in logo_data for _apply_logo
            logo_data['position'] = logo_position
            print(f"[VideoReframe] Logo prepared: {logo_data['image'].shape[1]}x{logo_data['image'].shape[0]} at pos ({logo_position[0]:.2f}, {logo_position[1]:.2f})")

    # Generate stable blur parameters (no color/flip changes, smooth motion)
    rng = random.Random(video_index + 42)
    blur_params = _generate_blur_params(rng, stable=True)

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

    # Blur cache
    cache_blur_top = None
    cache_blur_bottom = None
    cache_blur_left = None
    cache_blur_right = None

    # Layout values
    scaled_w = layout['scaled_w']
    scaled_h = layout['scaled_h']
    content_x = layout['content_x']
    content_y = layout['content_y']
    blur_top_h = layout['blur_top']
    blur_bottom_h = layout['blur_bottom']
    blur_left_w = layout.get('blur_left', 0)
    blur_right_w = layout.get('blur_right', 0)

    frame_idx = 0
    encoder_used = 'NVENC' if use_gpu else 'CPU'

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update blur every 2 frames for smooth motion
            update_blur = (frame_idx % 2 == 0) or frame_idx == 0

            # Process frame
            if use_gpu:
                # Transfer frame to GPU
                gpu_frame = gpu.to_gpu(frame)

                # Resize content on GPU
                content = gpu.resize(gpu_frame, (scaled_w, scaled_h), interpolation='cubic')

                # Apply color adjustments on GPU
                if brightness_adj != 0 or saturation_adj != 0:
                    br_mult = 1.0 + (brightness_adj / 100.0)
                    sat_mult = 1.0 + (saturation_adj / 100.0)
                    content = effects.adjust_saturation_brightness(content, sat_mult, br_mult)

                if contrast_adj != 0:
                    ct_mult = 1.0 + (contrast_adj / 100.0)
                    content = effects.adjust_contrast(content, ct_mult)

                # Convert content back to CPU for blur zone creation and output
                content_cpu = gpu.to_cpu(content)

                # Generate vertical blur zones (top/bottom)
                if blur_top_h > 0:
                    if update_blur or cache_blur_top is None:
                        cache_blur_top = _create_blur_zone_gpu(
                            gpu, effects, content_cpu, final_w, blur_top_h, 'top',
                            blur_intensity, blur_params
                        )
                    output_frame[0:blur_top_h, :] = cache_blur_top

                # Place content (handle center-crop if content is wider than frame)
                if content_x < 0:
                    # Content wider than frame - center crop
                    crop_x = -content_x
                    cropped_content = content_cpu[:, crop_x:crop_x + final_w]
                    output_frame[content_y:content_y + scaled_h, 0:final_w] = cropped_content
                else:
                    # Content fits in frame
                    output_frame[content_y:content_y + scaled_h, content_x:content_x + scaled_w] = content_cpu

                # Bottom blur zone
                if blur_bottom_h > 0:
                    if update_blur or cache_blur_bottom is None:
                        cache_blur_bottom = _create_blur_zone_gpu(
                            gpu, effects, content_cpu, final_w, blur_bottom_h, 'bottom',
                            blur_intensity, blur_params
                        )
                    output_frame[content_y + scaled_h:, :] = cache_blur_bottom

                # Generate horizontal blur zones (left/right) for horizontal videos
                if blur_left_w > 0:
                    if update_blur or cache_blur_left is None:
                        cache_blur_left = _create_blur_zone_horizontal_gpu(
                            gpu, effects, content_cpu, blur_left_w, final_h, 'left',
                            blur_intensity, blur_params
                        )
                    output_frame[:, 0:blur_left_w] = cache_blur_left

                if blur_right_w > 0:
                    if update_blur or cache_blur_right is None:
                        cache_blur_right = _create_blur_zone_horizontal_gpu(
                            gpu, effects, content_cpu, blur_right_w, final_h, 'right',
                            blur_intensity, blur_params
                        )
                    output_frame[:, content_x + scaled_w:] = cache_blur_right
            else:
                # CPU processing
                content = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

                if brightness_adj != 0 or contrast_adj != 0 or saturation_adj != 0:
                    content = _apply_color_adjustments(content, brightness_adj, saturation_adj, contrast_adj)

                # Vertical blur zones (top/bottom)
                if blur_top_h > 0:
                    if update_blur or cache_blur_top is None:
                        cache_blur_top = _create_blur_zone_cpu(
                            effects, content, final_w, blur_top_h, 'top',
                            blur_intensity, blur_params
                        )
                    output_frame[0:blur_top_h, :] = cache_blur_top

                # Place content (handle center-crop if content is wider than frame)
                if content_x < 0:
                    # Content wider than frame - center crop
                    crop_x = -content_x
                    cropped_content = content[:, crop_x:crop_x + final_w]
                    output_frame[content_y:content_y + scaled_h, 0:final_w] = cropped_content
                else:
                    # Content fits in frame
                    output_frame[content_y:content_y + scaled_h, content_x:content_x + scaled_w] = content

                if blur_bottom_h > 0:
                    if update_blur or cache_blur_bottom is None:
                        cache_blur_bottom = _create_blur_zone_cpu(
                            effects, content, final_w, blur_bottom_h, 'bottom',
                            blur_intensity, blur_params
                        )
                    output_frame[content_y + scaled_h:, :] = cache_blur_bottom

                # Horizontal blur zones (left/right) for horizontal videos
                if blur_left_w > 0:
                    if update_blur or cache_blur_left is None:
                        cache_blur_left = _create_blur_zone_horizontal_cpu(
                            effects, content, blur_left_w, final_h, 'left',
                            blur_intensity, blur_params
                        )
                    output_frame[:, 0:blur_left_w] = cache_blur_left

                if blur_right_w > 0:
                    if update_blur or cache_blur_right is None:
                        cache_blur_right = _create_blur_zone_horizontal_cpu(
                            effects, content, blur_right_w, final_h, 'right',
                            blur_intensity, blur_params
                        )
                    output_frame[:, content_x + scaled_w:] = cache_blur_right

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

        # Free GPU memory
        if use_gpu:
            gpu.free_memory()

    if progress_callback:
        progress_callback(1.0, "Complete")

    output_size = os.path.getsize(output_path) if output_path.exists() else 0

    return {
        "status": "completed",
        "type": "video",
        "outputPath": str(output_path),
        "outputSize": output_size,
        "dimensions": f"{final_w}x{final_h}",
        "encoder": encoder_used,
        "processor": "GPU" if use_gpu else "CPU",
        "fps": fps,
        "duration": duration,
        "framesProcessed": frame_idx
    }


# =============================================================================
# FRAME PROCESSING HELPERS
# =============================================================================

def _process_frame_gpu(
    gpu: 'GPUProcessor',
    effects: 'GPUBlurEffects',
    frame: np.ndarray,
    layout: dict,
    final_w: int,
    final_h: int,
    blur_intensity: int,
    blur_params: dict,
    brightness_adj: int,
    saturation_adj: int,
    contrast_adj: int,
    logo_data: Optional[dict]
) -> np.ndarray:
    """Process a single frame using GPU."""
    scaled_w = layout['scaled_w']
    scaled_h = layout['scaled_h']
    content_x = layout['content_x']
    content_y = layout['content_y']
    blur_top_h = layout['blur_top']
    blur_bottom_h = layout['blur_bottom']
    blur_left_w = layout.get('blur_left', 0)
    blur_right_w = layout.get('blur_right', 0)

    # Allocate output
    output_frame = np.zeros((final_h, final_w, 3), dtype=np.uint8)

    # Transfer frame to GPU
    gpu_frame = gpu.to_gpu(frame)

    # Resize content on GPU
    content = gpu.resize(gpu_frame, (scaled_w, scaled_h), interpolation='cubic')

    # Apply color adjustments on GPU
    if brightness_adj != 0 or saturation_adj != 0:
        br_mult = 1.0 + (brightness_adj / 100.0)
        sat_mult = 1.0 + (saturation_adj / 100.0)
        content = effects.adjust_saturation_brightness(content, sat_mult, br_mult)

    if contrast_adj != 0:
        ct_mult = 1.0 + (contrast_adj / 100.0)
        content = effects.adjust_contrast(content, ct_mult)

    # Convert to CPU for final composition
    content_cpu = gpu.to_cpu(content)

    # Generate vertical blur zones (top/bottom)
    if blur_top_h > 0:
        blur_top = _create_blur_zone_gpu(
            gpu, effects, content_cpu, final_w, blur_top_h, 'top',
            blur_intensity, blur_params
        )
        output_frame[0:blur_top_h, :] = blur_top

    # Place content (handle center-crop if content is wider than frame)
    if content_x < 0:
        # Content wider than frame - center crop
        crop_x = -content_x
        cropped_content = content_cpu[:, crop_x:crop_x + final_w]
        output_frame[content_y:content_y + scaled_h, 0:final_w] = cropped_content
    else:
        # Content fits in frame
        output_frame[content_y:content_y + scaled_h, content_x:content_x + scaled_w] = content_cpu

    # Bottom blur zone
    if blur_bottom_h > 0:
        blur_bottom = _create_blur_zone_gpu(
            gpu, effects, content_cpu, final_w, blur_bottom_h, 'bottom',
            blur_intensity, blur_params
        )
        output_frame[content_y + scaled_h:, :] = blur_bottom

    # Generate horizontal blur zones (left/right) for horizontal videos
    if blur_left_w > 0:
        blur_left = _create_blur_zone_horizontal_gpu(
            gpu, effects, content_cpu, blur_left_w, final_h, 'left',
            blur_intensity, blur_params
        )
        output_frame[:, 0:blur_left_w] = blur_left

    if blur_right_w > 0:
        blur_right = _create_blur_zone_horizontal_gpu(
            gpu, effects, content_cpu, blur_right_w, final_h, 'right',
            blur_intensity, blur_params
        )
        output_frame[:, content_x + scaled_w:] = blur_right

    # Apply logo
    if logo_data is not None:
        _apply_logo(output_frame, logo_data, final_w, final_h)

    return output_frame


def _process_frame_cpu(
    effects: BlurEffects,
    frame: np.ndarray,
    layout: dict,
    final_w: int,
    final_h: int,
    blur_intensity: int,
    blur_params: dict,
    brightness_adj: int,
    saturation_adj: int,
    contrast_adj: int,
    logo_data: Optional[dict]
) -> np.ndarray:
    """Process a single frame using CPU."""
    scaled_w = layout['scaled_w']
    scaled_h = layout['scaled_h']
    content_x = layout['content_x']
    content_y = layout['content_y']
    blur_top_h = layout['blur_top']
    blur_bottom_h = layout['blur_bottom']
    blur_left_w = layout.get('blur_left', 0)
    blur_right_w = layout.get('blur_right', 0)

    # Allocate output
    output_frame = np.zeros((final_h, final_w, 3), dtype=np.uint8)

    # Resize content
    content = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)

    # Apply color adjustments
    if brightness_adj != 0 or contrast_adj != 0 or saturation_adj != 0:
        content = _apply_color_adjustments(content, brightness_adj, saturation_adj, contrast_adj)

    # Generate vertical blur zones (top/bottom)
    if blur_top_h > 0:
        blur_top = _create_blur_zone_cpu(
            effects, content, final_w, blur_top_h, 'top',
            blur_intensity, blur_params
        )
        output_frame[0:blur_top_h, :] = blur_top

    # Place content (handle center-crop if content is wider than frame)
    if content_x < 0:
        # Content wider than frame - center crop
        crop_x = -content_x
        cropped_content = content[:, crop_x:crop_x + final_w]
        output_frame[content_y:content_y + scaled_h, 0:final_w] = cropped_content
    else:
        # Content fits in frame
        output_frame[content_y:content_y + scaled_h, content_x:content_x + scaled_w] = content

    # Bottom blur zone
    if blur_bottom_h > 0:
        blur_bottom = _create_blur_zone_cpu(
            effects, content, final_w, blur_bottom_h, 'bottom',
            blur_intensity, blur_params
        )
        output_frame[content_y + scaled_h:, :] = blur_bottom

    # Generate horizontal blur zones (left/right) for horizontal videos
    if blur_left_w > 0:
        blur_left = _create_blur_zone_horizontal_cpu(
            effects, content, blur_left_w, final_h, 'left',
            blur_intensity, blur_params
        )
        output_frame[:, 0:blur_left_w] = blur_left

    if blur_right_w > 0:
        blur_right = _create_blur_zone_horizontal_cpu(
            effects, content, blur_right_w, final_h, 'right',
            blur_intensity, blur_params
        )
        output_frame[:, content_x + scaled_w:] = blur_right

    # Apply logo
    if logo_data is not None:
        _apply_logo(output_frame, logo_data, final_w, final_h)

    return output_frame


# =============================================================================
# BLUR ZONE CREATION
# =============================================================================

def _create_blur_zone_gpu(
    gpu: 'GPUProcessor',
    effects: 'GPUBlurEffects',
    content: np.ndarray,
    target_w: int,
    target_h: int,
    position: str,
    blur_strength: int,
    params: dict
) -> np.ndarray:
    """Create blur zone using GPU operations."""
    if target_h <= 0:
        return np.zeros((0, target_w, 3), dtype=np.uint8)

    content_h, content_w = content.shape[:2]

    # Extract source region
    source_ratio = 0.65
    source_h = max(int(content_h * source_ratio), min(content_h, max(target_h * 2, 100)))
    source_h = max(10, min(source_h, content_h))

    if position == 'top':
        source = content[0:source_h, :].copy()
    else:
        start_y = max(0, content_h - source_h)
        source = content[start_y:, :].copy()

    # Transfer to GPU
    gpu_source = gpu.to_gpu(source)

    # Apply zoom
    zoom = max(target_w / source.shape[1], target_h / source.shape[0]) * params['zoom']
    zoomed_w = int(source.shape[1] * zoom)
    zoomed_h = int(source.shape[0] * zoom)

    if zoomed_w <= 0 or zoomed_h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    gpu_source = gpu.resize(gpu_source, (zoomed_w, zoomed_h), interpolation='linear')

    # Crop to target size (center crop)
    crop_x = max(0, (zoomed_w - target_w) // 2)
    crop_y = max(0, (zoomed_h - target_h) // 2)

    # Need to get back to CPU for slicing, then back to GPU
    source_cpu = gpu.to_cpu(gpu_source)
    blur_section = source_cpu[crop_y:crop_y + target_h, crop_x:crop_x + target_w].copy()

    # Ensure exact dimensions
    if blur_section.shape[0] != target_h or blur_section.shape[1] != target_w:
        blur_section = cv2.resize(blur_section, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Transfer back to GPU for effects
    gpu_blur = gpu.to_gpu(blur_section)

    # Apply effects on GPU
    if params['hflip']:
        gpu_blur = effects.flip_horizontal(gpu_blur)

    if position == 'top' and params['vflip']:
        gpu_blur = effects.flip_vertical(gpu_blur)

    if params['color_shift'] != 0:
        gpu_blur = effects.color_shift(gpu_blur, params['color_shift'])

    if params['saturation'] != 1.0 or params['brightness'] != 1.0:
        gpu_blur = effects.adjust_saturation_brightness(
            gpu_blur, params['saturation'], params['brightness']
        )

    if params['tilt_angle'] != 0:
        gpu_blur = effects.tilt(gpu_blur, params['tilt_angle'])

    gpu_blur = effects.darken(gpu_blur, params['darken'])

    # Apply Gaussian blur
    if blur_strength > 0:
        gpu_blur = effects.gaussian_blur(gpu_blur, blur_strength)

    return gpu.to_cpu(gpu_blur)


def _create_blur_zone_cpu(
    effects: BlurEffects,
    content: np.ndarray,
    target_w: int,
    target_h: int,
    position: str,
    blur_strength: int,
    params: dict
) -> np.ndarray:
    """Create blur zone using CPU operations (fallback)."""
    if target_h <= 0:
        return np.zeros((0, target_w, 3), dtype=np.uint8)

    content_h, content_w = content.shape[:2]

    # Extract source region
    source_ratio = 0.65
    source_h = max(int(content_h * source_ratio), min(content_h, max(target_h * 2, 100)))
    source_h = max(10, min(source_h, content_h))

    if position == 'top':
        source = content[0:source_h, :].copy()
    else:
        start_y = max(0, content_h - source_h)
        source = content[start_y:, :].copy()

    # Apply zoom
    zoom = max(target_w / source.shape[1], target_h / source.shape[0]) * params['zoom']
    zoomed_w = int(source.shape[1] * zoom)
    zoomed_h = int(source.shape[0] * zoom)

    if zoomed_w <= 0 or zoomed_h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    source = cv2.resize(source, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)

    # Crop to target size
    crop_x = max(0, (zoomed_w - target_w) // 2)
    crop_y = max(0, (zoomed_h - target_h) // 2)
    blur_section = source[crop_y:crop_y + target_h, crop_x:crop_x + target_w]

    # Ensure exact dimensions
    if blur_section.shape[0] != target_h or blur_section.shape[1] != target_w:
        blur_section = cv2.resize(blur_section, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Apply effects
    if params['hflip']:
        blur_section = effects.flip_horizontal(blur_section)

    if position == 'top' and params['vflip']:
        blur_section = effects.flip_vertical(blur_section)

    if params['color_shift'] != 0:
        blur_section = effects.color_shift(blur_section, params['color_shift'])

    if params['saturation'] != 1.0 or params['brightness'] != 1.0:
        blur_section = effects.adjust_saturation_brightness(
            blur_section, params['saturation'], params['brightness']
        )

    if params['tilt_angle'] != 0:
        blur_section = effects.tilt(blur_section, params['tilt_angle'])

    blur_section = effects.darken(blur_section, params['darken'])

    if blur_strength > 0:
        blur_section = effects.gaussian_blur(blur_section, blur_strength)

    return blur_section


def _create_blur_zone_horizontal_gpu(
    gpu: 'GPUProcessor',
    effects: 'GPUBlurEffects',
    content: np.ndarray,
    target_w: int,
    target_h: int,
    position: str,
    blur_strength: int,
    params: dict
) -> np.ndarray:
    """Create horizontal blur zone (left/right) using GPU operations."""
    if target_w <= 0:
        return np.zeros((target_h, 0, 3), dtype=np.uint8)

    content_h, content_w = content.shape[:2]

    # Extract source region from left or right side of content
    source_ratio = 0.65
    source_w = max(int(content_w * source_ratio), min(content_w, max(target_w * 2, 100)))
    source_w = max(10, min(source_w, content_w))

    if position == 'left':
        source = content[:, 0:source_w].copy()
    else:  # right
        start_x = max(0, content_w - source_w)
        source = content[:, start_x:].copy()

    # Transfer to GPU
    gpu_source = gpu.to_gpu(source)

    # Apply zoom
    zoom = max(target_w / source.shape[1], target_h / source.shape[0]) * params['zoom']
    zoomed_w = int(source.shape[1] * zoom)
    zoomed_h = int(source.shape[0] * zoom)

    if zoomed_w <= 0 or zoomed_h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    gpu_source = gpu.resize(gpu_source, (zoomed_w, zoomed_h), interpolation='linear')

    # Crop to target size (center crop)
    crop_x = max(0, (zoomed_w - target_w) // 2)
    crop_y = max(0, (zoomed_h - target_h) // 2)

    # Need to get back to CPU for slicing, then back to GPU
    source_cpu = gpu.to_cpu(gpu_source)
    blur_section = source_cpu[crop_y:crop_y + target_h, crop_x:crop_x + target_w].copy()

    # Ensure exact dimensions
    if blur_section.shape[0] != target_h or blur_section.shape[1] != target_w:
        blur_section = cv2.resize(blur_section, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Transfer back to GPU for effects
    gpu_blur = gpu.to_gpu(blur_section)

    # Apply effects on GPU
    if position == 'left' and params['hflip']:
        gpu_blur = effects.flip_horizontal(gpu_blur)

    if params['vflip']:
        gpu_blur = effects.flip_vertical(gpu_blur)

    if params['color_shift'] != 0:
        gpu_blur = effects.color_shift(gpu_blur, params['color_shift'])

    if params['saturation'] != 1.0 or params['brightness'] != 1.0:
        gpu_blur = effects.adjust_saturation_brightness(
            gpu_blur, params['saturation'], params['brightness']
        )

    if params['tilt_angle'] != 0:
        gpu_blur = effects.tilt(gpu_blur, params['tilt_angle'])

    gpu_blur = effects.darken(gpu_blur, params['darken'])

    # Apply Gaussian blur
    if blur_strength > 0:
        gpu_blur = effects.gaussian_blur(gpu_blur, blur_strength)

    return gpu.to_cpu(gpu_blur)


def _create_blur_zone_horizontal_cpu(
    effects: BlurEffects,
    content: np.ndarray,
    target_w: int,
    target_h: int,
    position: str,
    blur_strength: int,
    params: dict
) -> np.ndarray:
    """Create horizontal blur zone (left/right) using CPU operations."""
    if target_w <= 0:
        return np.zeros((target_h, 0, 3), dtype=np.uint8)

    content_h, content_w = content.shape[:2]

    # Extract source region from left or right side of content
    source_ratio = 0.65
    source_w = max(int(content_w * source_ratio), min(content_w, max(target_w * 2, 100)))
    source_w = max(10, min(source_w, content_w))

    if position == 'left':
        source = content[:, 0:source_w].copy()
    else:  # right
        start_x = max(0, content_w - source_w)
        source = content[:, start_x:].copy()

    # Apply zoom
    zoom = max(target_w / source.shape[1], target_h / source.shape[0]) * params['zoom']
    zoomed_w = int(source.shape[1] * zoom)
    zoomed_h = int(source.shape[0] * zoom)

    if zoomed_w <= 0 or zoomed_h <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    source = cv2.resize(source, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)

    # Crop to target size
    crop_x = max(0, (zoomed_w - target_w) // 2)
    crop_y = max(0, (zoomed_h - target_h) // 2)
    blur_section = source[crop_y:crop_y + target_h, crop_x:crop_x + target_w]

    # Ensure exact dimensions
    if blur_section.shape[0] != target_h or blur_section.shape[1] != target_w:
        blur_section = cv2.resize(blur_section, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Apply effects
    if position == 'left' and params['hflip']:
        blur_section = effects.flip_horizontal(blur_section)

    if params['vflip']:
        blur_section = effects.flip_vertical(blur_section)

    if params['color_shift'] != 0:
        blur_section = effects.color_shift(blur_section, params['color_shift'])

    if params['saturation'] != 1.0 or params['brightness'] != 1.0:
        blur_section = effects.adjust_saturation_brightness(
            blur_section, params['saturation'], params['brightness']
        )

    if params['tilt_angle'] != 0:
        blur_section = effects.tilt(blur_section, params['tilt_angle'])

    blur_section = effects.darken(blur_section, params['darken'])

    if blur_strength > 0:
        blur_section = effects.gaussian_blur(blur_section, blur_strength)

    return blur_section


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
        return True


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

    try:
        if ':' in aspect_str:
            w, h = map(int, aspect_str.split(':'))
            final_w = 1080
            final_h = int(1080 * h / w)
            final_h = final_h - (final_h % 2)
            return (final_w, final_h)
    except:
        pass

    return (1080, 1920)


def _calculate_layout(
    orig_w: int,
    orig_h: int,
    final_w: int,
    final_h: int,
    force_blur_percent: float = 0,
    top_blur_percent: float = 0,
    bottom_blur_percent: float = 0
) -> dict:
    """
    Calculate layout for content and blur zones.

    Args:
        orig_w, orig_h: Original content dimensions
        final_w, final_h: Target output dimensions
        force_blur_percent: Force equal blur on top/bottom (legacy, for same-aspect videos)
        top_blur_percent: Specific top blur percentage (0-100)
        bottom_blur_percent: Specific bottom blur percentage (0-100)

    For HORIZONTAL videos going to VERTICAL output:
        - Scale to fill HEIGHT (preserve video height)
        - Add blur on LEFT and RIGHT sides

    For VERTICAL videos going to VERTICAL output:
        - Scale to fill WIDTH
        - Add blur on TOP and BOTTOM

    Priority:
        1. If top_blur_percent or bottom_blur_percent > 0: Use individual percentages
        2. Else if force_blur_percent > 0: Apply equal blur top/bottom
        3. Else: Auto-calculate based on aspect ratio difference
    """
    orig_aspect = orig_w / orig_h
    final_aspect = final_w / final_h

    # Detect if source is horizontal and target is vertical
    is_horizontal_to_vertical = orig_aspect > 1.0 and final_aspect < 1.0

    # Initialize blur values
    blur_top = 0
    blur_bottom = 0
    blur_left = 0
    blur_right = 0

    if is_horizontal_to_vertical:
        # HORIZONTAL video -> VERTICAL output
        # Strategy: Scale to fill WIDTH, keep full video visible, add blur TOP and BOTTOM
        # Video is centered with blurred zones above and below
        scale = final_w / orig_w
        scaled_w = final_w
        scaled_h = int(orig_h * scale)
        scaled_h = scaled_h - (scaled_h % 2)

        # Calculate blur zones for top and bottom
        blur_space_v = final_h - scaled_h
        blur_top = blur_space_v // 2
        blur_bottom = blur_space_v - blur_top
        print(f"[Reframe] Horizontal video: content={scaled_w}x{scaled_h}, blur top/bottom={blur_top}/{blur_bottom}px")

    else:
        # VERTICAL or SQUARE video -> any output
        # Original logic: scale to fill WIDTH, add blur top/bottom
        scale = final_w / orig_w
        scaled_w = final_w
        scaled_h = int(orig_h * scale)
        scaled_h = scaled_h - (scaled_h % 2)

        if scaled_h > final_h:
            scale = final_h / orig_h
            scaled_h = final_h
            scaled_w = int(orig_w * scale)
            scaled_w = scaled_w - (scaled_w % 2)

        # Auto-calculated blur from aspect ratio difference
        blur_space = final_h - scaled_h
        blur_top = blur_space // 2
        blur_bottom = blur_space - blur_top

    # Check if individual blur percentages are specified (override auto-calculation)
    has_individual_blur = (top_blur_percent > 0 or bottom_blur_percent > 0)

    if has_individual_blur:
        # Use individual blur percentages (from frontend)
        top_pct = min(top_blur_percent, 50)  # Cap at 50%
        bottom_pct = min(bottom_blur_percent, 50)  # Cap at 50%

        # Calculate blur heights in pixels
        blur_top = int(final_h * top_pct / 100)
        blur_bottom = int(final_h * bottom_pct / 100)
        blur_left = 0
        blur_right = 0

        # Recalculate content size to fit in remaining space
        content_space = final_h - blur_top - blur_bottom
        if content_space < scaled_h:
            # Content needs to shrink to fit
            new_scale = content_space / orig_h
            scaled_h = content_space
            scaled_h = scaled_h - (scaled_h % 2)
            scaled_w = int(orig_w * new_scale)
            scaled_w = min(scaled_w, final_w)
            scaled_w = scaled_w - (scaled_w % 2)

        print(f"[Reframe] Individual blur: top={top_pct}% ({blur_top}px), bottom={bottom_pct}% ({blur_bottom}px)")

    elif force_blur_percent > 0 and blur_top == 0 and blur_bottom == 0 and blur_left == 0 and blur_right == 0:
        # Legacy: force equal blur when content matches aspect (force_blur_percent)
        blur_percent = min(force_blur_percent, 40)
        content_percent = 1.0 - (2 * blur_percent / 100)
        scaled_h = int(final_h * content_percent)
        scaled_h = scaled_h - (scaled_h % 2)
        blur_space = final_h - scaled_h
        blur_top = blur_space // 2
        blur_bottom = blur_space - blur_top
        print(f"[Reframe] Force blur: {blur_percent}% top/bottom, content={scaled_w}x{scaled_h}")

    content_x = (final_w - scaled_w) // 2
    content_y = blur_top

    return {
        'scaled_w': scaled_w,
        'scaled_h': scaled_h,
        'content_x': content_x,
        'content_y': content_y,
        'blur_top': blur_top,
        'blur_bottom': blur_bottom,
        'blur_left': blur_left,
        'blur_right': blur_right,
        'needs_blur': blur_top > 0 or blur_bottom > 0 or blur_left > 0 or blur_right > 0
    }


def _generate_blur_params(rng: random.Random, stable: bool = True) -> dict:
    """Generate parameters for blur effect.

    Args:
        rng: Random generator for reproducibility
        stable: If True, use stable/subtle parameters. If False, use animated variation.
    """
    if stable:
        # Stable blur - no random color/orientation changes, just clean blur
        return {
            'hflip': False,
            'vflip': False,
            'color_shift': 0,
            'saturation': 1.0,
            'brightness': 0.85,  # Slightly darker
            'tilt_angle': 0,
            'zoom': 1.3,
            'darken': 0.7,
        }
    else:
        # Animated blur with subtle variation
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


def _apply_color_adjustments(
    image: np.ndarray,
    brightness: int,
    saturation: int,
    contrast: int
) -> np.ndarray:
    """Apply brightness, saturation, and contrast adjustments (CPU)."""
    result = image

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


# =============================================================================
# LOGO FUNCTIONS
# =============================================================================

def _svg_to_pil(svg_path: str, video_width: int, size_percent: float):
    """Convert SVG to PIL Image."""
    from PIL import Image

    target_width = max(int(video_width * size_percent / 100), 80)

    try:
        import cairosvg
        import io
        png_data = cairosvg.svg2png(url=svg_path, output_width=target_width * 2)
        return Image.open(io.BytesIO(png_data)).convert('RGBA')
    except Exception as e:
        print(f"[Reframe] cairosvg failed: {e}")

    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPM
        import io
        drawing = svg2rlg(svg_path)
        if drawing:
            scale = (target_width * 2) / drawing.width
            drawing.width *= scale
            drawing.height *= scale
            drawing.scale(scale, scale)
            png_data = io.BytesIO()
            renderPM.drawToFile(drawing, png_data, fmt='PNG')
            png_data.seek(0)
            return Image.open(png_data).convert('RGBA')
    except Exception as e:
        print(f"[Reframe] svglib failed: {e}")

    print(f"[Reframe] Using placeholder logo")
    from PIL import ImageDraw, ImageFont

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

    return logo


def _prepare_logo(logo_name: str, video_width: int, size_percent: float) -> Optional[dict]:
    """Prepare logo for overlay."""
    try:
        from PIL import Image

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
        is_svg = False

        if logo_name in ['farmium_icon', 'farmium_full']:
            for search_path in search_paths:
                png_path = search_path / f'{logo_name}.png'
                if png_path.exists():
                    logo_source = str(png_path)
                    print(f"[Reframe] Found PNG logo: {logo_source}")
                    break
                svg_path = search_path / f'{logo_name}.svg'
                if svg_path.exists():
                    logo_source = str(svg_path)
                    is_svg = True
                    print(f"[Reframe] Found SVG logo: {logo_source}")
                    break

        elif logo_name.startswith('http://') or logo_name.startswith('https://'):
            import requests
            response = requests.get(logo_name, timeout=30)
            response.raise_for_status()
            suffix = '.svg' if '.svg' in logo_name.lower() else '.png'
            temp_logo = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            temp_logo.write(response.content)
            temp_logo.close()
            logo_source = temp_logo.name
            is_svg = suffix == '.svg'

        elif Path(logo_name).exists():
            logo_source = logo_name
            is_svg = logo_name.lower().endswith('.svg')

        if not logo_source:
            print(f"[Reframe] Logo not found: {logo_name}")
            return None

        if is_svg:
            pil_logo = _svg_to_pil(logo_source, video_width, size_percent)
            if pil_logo is None:
                print(f"[Reframe] Failed to convert SVG: {logo_source}")
                return None
        else:
            pil_logo = Image.open(logo_source).convert('RGBA')

        logo_w = max(int(video_width * size_percent / 100), 80)
        logo_h = int(logo_w * pil_logo.height / pil_logo.width)

        pil_logo = pil_logo.resize((logo_w, logo_h), Image.Resampling.LANCZOS)

        logo_array = np.array(pil_logo)
        logo_bgr = cv2.cvtColor(logo_array[:, :, :3], cv2.COLOR_RGB2BGR)
        logo_alpha = logo_array[:, :, 3].astype(np.float32) / 255.0

        return {
            'image': logo_bgr,
            'alpha': logo_alpha,
            'alpha_3d': logo_alpha[:, :, np.newaxis]
        }

    except Exception as e:
        print(f"[Reframe] Error preparing logo: {e}")
        return None


def _apply_logo(frame: np.ndarray, logo_data: dict, frame_w: int, frame_h: int):
    """
    Apply logo overlay to frame.

    Uses position from logo_data['position'] if available (normalized 0-1 coords),
    otherwise defaults to center-bottom with 5% margin.

    Position interpretation:
        - x=0.0: Left edge, x=0.5: Center, x=1.0: Right edge
        - y=0.0: Top edge, y=0.5: Center, y=1.0: Bottom edge
        The logo is centered on the specified position point.
    """
    logo = logo_data['image']
    alpha_3d = logo_data['alpha_3d']
    lh, lw = logo.shape[:2]

    # Get position from config, default to center-bottom (0.5, 0.85)
    position = logo_data.get('position', (0.5, 0.85))
    pos_x, pos_y = position if isinstance(position, (list, tuple)) else (0.5, 0.85)

    # Convert normalized position to pixel coordinates
    # Position is the CENTER of the logo
    x = int(pos_x * frame_w - lw / 2)
    y = int(pos_y * frame_h - lh / 2)

    # Clamp to frame bounds
    x = max(0, min(frame_w - lw, x))
    y = max(0, min(frame_h - lh, y))

    roi = frame[y:y + lh, x:x + lw]
    blended = (alpha_3d * logo + (1 - alpha_3d) * roi).astype(np.uint8)
    frame[y:y + lh, x:x + lw] = blended


# =============================================================================
# FFMPEG WRITER
# =============================================================================

def _start_ffmpeg_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    audio_source: str = None,
    gpu_id: int = 0
) -> subprocess.Popen:
    """Start FFmpeg subprocess for writing video frames."""
    import platform

    system_os = platform.system()

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'warning',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}',
        '-pix_fmt', 'bgr24',
        '-r', str(min(fps, 60)),
        '-thread_queue_size', '512',
        '-i', '-',
    ]

    if audio_source:
        cmd.extend(['-i', audio_source])

    if system_os in ['Linux', 'Windows']:
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-gpu', str(gpu_id),
            '-preset', 'p4',
            '-rc', 'vbr',
            '-cq', '23',
            '-b:v', '8000k',
            '-maxrate', '12000k',
            '-bufsize', '16000k',
        ])
    elif system_os == 'Darwin':
        cmd.extend([
            '-c:v', 'h264_videotoolbox',
            '-b:v', '8000k',
            '-allow_sw', '1',
        ])
    else:
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
        ])

    cmd.extend(['-pix_fmt', 'yuv420p'])
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

    print(f"[Reframe] Starting FFmpeg: {' '.join(cmd[:15])}...")

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return process
    except Exception as e:
        print(f"[Reframe] FFmpeg start failed, trying CPU fallback: {e}")

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
    """Process multiple videos/images with reframe."""
    results = []
    total = len(input_paths)

    for i, input_path in enumerate(input_paths):
        if progress_callback:
            base_progress = i / total
            progress_callback(base_progress, f"Processing {i+1}/{total}")

        input_name = Path(input_path).stem
        input_ext = Path(input_path).suffix.lower()

        # Determine output extension based on input type
        if input_ext in IMAGE_EXTENSIONS:
            output_ext = '.jpg'
        else:
            output_ext = '.mp4'

        output_path = Path(output_dir) / f"{input_name}_reframed{output_ext}"

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
            print(f"[Reframe] Error processing {input_path}: {e}")
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
        print("Usage: python processor.py <input> <output> [aspect_ratio] [logo_name]")
        print("")
        print("Supports both VIDEO and IMAGE inputs:")
        print("  - Image input (jpg, png, webp) -> Image output")
        print("  - Video input (mp4, mov, etc.) -> Video output")
        print("")
        print("Examples:")
        print("  python processor.py video.mp4 out.mp4 9:16")
        print("  python processor.py image.jpg out.jpg 9:16 farmium_full")
        print("")
        print("Logo options: farmium_icon, farmium_full, none, or URL")
        print("")
        print(f"GPU acceleration: {'AVAILABLE' if is_gpu_available() else 'NOT AVAILABLE (using CPU)'}")
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
        'blurUpdateInterval': 20,
    }

    print(f"[CLI] Input: {input_file}")
    print(f"[CLI] Output: {output_file}")
    print(f"[CLI] Aspect: {aspect}")
    print(f"[CLI] Logo: {logo}")
    print(f"[CLI] GPU: {'ENABLED' if is_gpu_available() else 'DISABLED (CPU fallback)'}")

    def progress(p, msg):
        print(f"[{int(p*100):3d}%] {msg}")

    result = process_video_reframe(input_file, output_file, config, progress)
    print(f"\nResult: {result}")
