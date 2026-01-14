"""
Video Reframe Processor
Converts horizontal/any aspect videos to vertical (9:16) with blur areas and logo overlay

Based on video_processor_v2.py patterns with optimized GPU processing
"""

import cv2
import numpy as np
import os
import subprocess
from pathlib import Path
from PIL import Image
import tempfile

# Try importing cairosvg for SVG to PNG conversion
try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False


def process_video_reframe(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback=None
) -> dict:
    """
    Convert horizontal video to vertical (9:16) with blur areas and logo overlay

    Args:
        input_path: Path to input video file
        output_path: Path to output video file
        config: Configuration dict with:
            - logoName: 'farmium_icon' | 'farmium_full' | custom path
            - logoSize: Logo size as percentage of video width (default: 50)
            - logoPosition: {x: 0-1, y: 0-1} normalized position (default: {x: 0.5, y: 0.85})
            - aspectRatio: [width, height] (default: [9, 16])
            - blurIntensity: Gaussian blur strength (default: 25)
            - randomizeEffects: Apply random effects to blur (default: True)
            - tiltRange: Random rotation angle range in degrees (default: 2)
            - colorShiftRange: Random hue shift range (default: 10)
            - brightness: Brightness multiplier (default: 1.0)
            - saturation: Saturation multiplier (default: 1.0)
            - contrast: Contrast multiplier (default: 1.0)
        progress_callback: Optional callback(progress: float, message: str)

    Returns:
        dict with status and metadata
    """

    # Parse config with defaults
    logo_name = config.get('logoName', 'farmium_icon')
    logo_size = config.get('logoSize', 50)  # Percentage
    logo_pos = config.get('logoPosition', {'x': 0.5, 'y': 0.85})
    aspect_ratio = config.get('aspectRatio', [9, 16])
    blur_intensity = config.get('blurIntensity', 25)
    randomize_effects = config.get('randomizeEffects', True)
    tilt_range = config.get('tiltRange', 2)
    color_shift_range = config.get('colorShiftRange', 10)
    brightness = config.get('brightness', 1.0)
    saturation = config.get('saturation', 1.0)
    contrast = config.get('contrast', 1.0)

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.05, "Loading video...")

    # Open video
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate output dimensions based on aspect ratio
    aspect_w, aspect_h = aspect_ratio
    target_ratio = aspect_w / aspect_h

    # For 9:16 vertical -> 1080x1920 standard
    if aspect_w == 9 and aspect_h == 16:
        final_w = 1080
        final_h = 1920
    elif aspect_w == 16 and aspect_h == 9:
        final_w = 1920
        final_h = 1080
    elif aspect_w == 4 and aspect_h == 5:
        final_w = 1080
        final_h = 1350
    elif aspect_w == 1 and aspect_h == 1:
        final_w = 1080
        final_h = 1080
    else:
        # Fallback: use 1080 width and calculate height
        final_w = 1080
        final_h = int(1080 / target_ratio)
        final_h = final_h - (final_h % 2)  # Ensure even

    # Ensure even dimensions
    final_w = final_w - (final_w % 2)
    final_h = final_h - (final_h % 2)

    # Calculate scaling to fit original video centered
    scale_w = final_w / orig_w
    scale_h = final_h / orig_h
    content_scale = min(scale_w, scale_h)

    scaled_content_w = int(orig_w * content_scale)
    scaled_content_h = int(orig_h * content_scale)

    # Calculate blur areas
    blur_space = final_h - scaled_content_h
    blur_top_h = max(0, blur_space // 2)
    blur_bottom_h = max(0, blur_space - blur_top_h)

    # Content position (centered)
    content_x = (final_w - scaled_content_w) // 2
    content_y = blur_top_h

    if progress_callback:
        progress_callback(0.1, "Preparing logo...")

    # Prepare logo
    logo_data = None
    if logo_name:
        logo_data = _prepare_logo(logo_name, final_w, logo_size)
        if logo_data:
            lh, lw = logo_data['image'].shape[:2]
            # Calculate logo position from normalized coordinates
            logo_x = int(logo_pos.get('x', 0.5) * final_w - lw // 2)
            logo_y = int(logo_pos.get('y', 0.85) * final_h - lh // 2)
            # Clamp to frame bounds
            logo_x = max(0, min(final_w - lw, logo_x))
            logo_y = max(0, min(final_h - lh, logo_y))
            logo_data['position'] = (logo_x, logo_y)

    if progress_callback:
        progress_callback(0.15, "Starting video encoding...")

    # Prepare blur configuration
    blur_config = {
        'gaussian_blur': blur_intensity,
        'mirror': False,
        'tilt': 0,
        'color_shift': 0,
        'saturation': saturation,
        'brightness': brightness,
        'pixelate': False,
        'pixel_size': 8,
        'noise': 0,
        'contrast': contrast,
        'vignette': 0,
    }

    # Setup FFmpeg encoder
    encoder = 'libx264'
    preset = 'ultrafast'
    hw_opts = []

    # Try to use NVENC if available (GPU encoding)
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True, text=True, timeout=5
        )
        if 'h264_nvenc' in result.stdout:
            encoder = 'h264_nvenc'
            hw_opts = ['-preset', 'p4', '-tune', 'hq']
            print("🚀 Using NVENC GPU encoder")
        else:
            print("💻 Using CPU encoder (libx264)")
    except:
        print("💻 Using CPU encoder (libx264)")

    cmd = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{final_w}x{final_h}',
        '-pix_fmt', 'bgr24',
        '-r', str(fps),
        '-i', '-',
        '-i', str(input_path),
        '-map', '0:v',
        '-map', '1:a?',
        '-c:v', encoder,
    ]

    if encoder != 'h264_nvenc':
        cmd.extend(['-preset', preset])

    cmd.extend(['-pix_fmt', 'yuv420p'])
    cmd.extend(hw_opts)
    cmd.extend([
        '-c:a', 'aac', '-b:a', '128k',
        '-shortest',
        '-movflags', '+faststart',
        str(output_path)
    ])

    try:
        ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        cap.release()
        raise ValueError("FFmpeg not found. Install ffmpeg and add to PATH.")

    # Process frames
    output_frame = np.zeros((final_h, final_w, 3), dtype=np.uint8)
    cache_blur_top = None
    cache_blur_bottom = None
    use_cache = not randomize_effects  # Only cache if no randomization
    cache_valid = False

    # Random effects (generated once if randomize_effects is True)
    if randomize_effects:
        tilt_angle = np.random.uniform(-tilt_range, tilt_range)
        color_shift = np.random.randint(-color_shift_range, color_shift_range + 1)
        blur_config['tilt'] = tilt_angle
        blur_config['color_shift'] = color_shift

    frame_idx = 0
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Scale content to fit
        if frame.shape[0] != scaled_content_h or frame.shape[1] != scaled_content_w:
            content = cv2.resize(frame, (scaled_content_w, scaled_content_h),
                               interpolation=cv2.INTER_LANCZOS4)
        else:
            content = frame

        # Create blur areas from content
        blur_source = content

        if blur_top_h > 0:
            if use_cache and cache_valid and cache_blur_top is not None:
                blur_top = cache_blur_top
            else:
                blur_top = _create_blur(
                    blur_source, final_w, blur_top_h, 'top', blur_config
                )
                if use_cache:
                    cache_blur_top = blur_top.copy()
                    cache_valid = True
            np.copyto(output_frame[0:blur_top_h, :], blur_top)

        # Place content centered
        output_frame[content_y:content_y + scaled_content_h,
                    content_x:content_x + scaled_content_w] = content

        if blur_bottom_h > 0:
            if use_cache and cache_valid and cache_blur_bottom is not None:
                blur_bottom = cache_blur_bottom
            else:
                blur_bottom = _create_blur(
                    blur_source, final_w, blur_bottom_h, 'bottom', blur_config
                )
                if use_cache:
                    cache_blur_bottom = blur_bottom.copy()
            np.copyto(output_frame[content_y + scaled_content_h:, :], blur_bottom)

        # Apply logo overlay
        if logo_data is not None:
            _apply_logo(output_frame, logo_data)

        # Write frame to FFmpeg
        try:
            ffmpeg_process.stdin.write(output_frame.tobytes())
        except BrokenPipeError:
            print("❌ FFmpeg pipe closed unexpectedly")
            break

        frame_idx += 1
        processed_frames += 1

        # Progress updates
        if progress_callback and (frame_idx % 30 == 0 or frame_idx == total_frames):
            progress = 0.15 + (frame_idx / total_frames) * 0.75
            progress_callback(progress, f"Processing frame {frame_idx}/{total_frames}")

    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "status": "completed",
        "outputPath": str(output_path),
        "outputSize": os.path.getsize(output_path),
        "dimensions": f"{final_w}x{final_h}",
        "framesProcessed": processed_frames,
    }


def _prepare_logo(logo_name: str, video_width: int, size_percent: float) -> dict:
    """
    Load and prepare logo for overlay

    Args:
        logo_name: Logo identifier ('farmium_icon', 'farmium_full') or path
        video_width: Video width for scaling
        size_percent: Logo size as percentage of video width

    Returns:
        dict with 'image', 'alpha', 'position' or None if failed
    """
    try:
        # Resolve logo path
        workspace = os.environ.get('WORKSPACE', '/workspace')
        logos_dir = Path(workspace) / 'assets' / 'logos'

        # Map logo names to files
        logo_map = {
            'farmium_icon': logos_dir / 'farmium_icon.svg',
            'farmium_full': logos_dir / 'farmium_full.svg',
        }

        if logo_name in logo_map:
            logo_path = logo_map[logo_name]
        else:
            logo_path = Path(logo_name)

        if not logo_path.exists():
            print(f"⚠️ Logo not found: {logo_path}")
            return None

        # Convert SVG to PNG if needed
        if logo_path.suffix.lower() == '.svg':
            if not HAS_CAIROSVG:
                print("⚠️ cairosvg not installed, skipping SVG logo")
                return None

            # Calculate target size
            logo_w = int(video_width * size_percent / 100)

            # Convert SVG to PNG in memory
            png_data = cairosvg.svg2png(
                url=str(logo_path),
                output_width=logo_w
            )

            # Load from bytes
            from io import BytesIO
            pil_logo = Image.open(BytesIO(png_data)).convert('RGBA')
        else:
            # Load PNG/image directly
            pil_logo = Image.open(logo_path).convert('RGBA')
            logo_w = int(video_width * size_percent / 100)
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
        print(f"⚠️ Failed to load logo: {e}")
        return None


def _apply_logo(frame: np.ndarray, logo_data: dict):
    """Apply logo overlay to frame with alpha blending"""
    logo = logo_data['image']
    alpha_3d = logo_data['alpha_3d']
    x, y = logo_data['position']

    lh, lw = logo.shape[:2]

    # Extract ROI
    roi = frame[y:y + lh, x:x + lw]

    # Alpha blend
    blended = (alpha_3d * logo + (1 - alpha_3d) * roi).astype(np.uint8)
    frame[y:y + lh, x:x + lw] = blended


def _create_blur(
    content: np.ndarray,
    target_w: int,
    target_h: int,
    position: str,
    blur_config: dict
) -> np.ndarray:
    """
    Create blur section from content

    Args:
        content: Source content to blur
        target_w: Target width
        target_h: Target height
        position: 'top' or 'bottom'
        blur_config: Blur effects configuration

    Returns:
        Blurred section as numpy array
    """
    if target_h <= 0:
        return np.zeros((0, target_w, 3), dtype=np.uint8)

    content_h, content_w = content.shape[:2]

    # Extract source region (70% of content for better quality)
    source_h = max(int(content_h * 0.7), min(content_h, max(target_h * 2, 100)))
    source_h = max(10, min(source_h, content_h))

    if position == 'top':
        source = content[0:source_h, :]
    else:
        start_y = max(0, content_h - source_h)
        source = content[start_y:, :]

    # Zoom and crop
    zoom = max(target_w / source.shape[1], target_h / source.shape[0]) * 1.3
    zoomed_w = int(source.shape[1] * zoom)
    zoomed_h = int(source.shape[0] * zoom)
    zoomed = cv2.resize(source, (zoomed_w, zoomed_h), interpolation=cv2.INTER_AREA)

    crop_x = max(0, (zoomed_w - target_w) // 2)
    crop_y = max(0, (zoomed_h - target_h) // 2)
    blur_section = zoomed[crop_y:crop_y + target_h, crop_x:crop_x + target_w]

    if blur_section.shape[0] != target_h or blur_section.shape[1] != target_w:
        blur_section = cv2.resize(blur_section, (target_w, target_h))

    # Apply effects
    blur_strength = blur_config.get('gaussian_blur', 25)
    if blur_strength > 0:
        k = int(blur_strength) | 1
        blur_section = cv2.GaussianBlur(blur_section, (k, k), 0)

    # Mirror for top
    if blur_config.get('mirror', False) and position == 'top':
        blur_section = cv2.flip(blur_section, 0)

    # Color adjustments
    color_shift = blur_config.get('color_shift', 0)
    saturation = blur_config.get('saturation', 1.0)
    brightness = blur_config.get('brightness', 1.0)

    if color_shift != 0 or saturation != 1.0 or brightness != 1.0:
        hsv = cv2.cvtColor(blur_section, cv2.COLOR_BGR2HSV).astype(np.float32)
        if color_shift != 0:
            hsv[:, :, 0] = (hsv[:, :, 0] + color_shift) % 180
        if saturation != 1.0:
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        if brightness != 1.0:
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
        blur_section = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Tilt
    tilt_angle = blur_config.get('tilt', 0)
    if tilt_angle != 0:
        center = (target_w // 2, target_h // 2)
        angle = tilt_angle if position == 'top' else -tilt_angle
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        blur_section = cv2.warpAffine(blur_section, rot_matrix, (target_w, target_h),
                                     borderMode=cv2.BORDER_REFLECT)

    # Contrast
    contrast = blur_config.get('contrast', 1.0)
    if contrast != 1.0:
        blur_section = cv2.convertScaleAbs(blur_section, alpha=contrast,
                                          beta=(1 - contrast) * 128)

    # Noise
    noise = blur_config.get('noise', 0)
    if noise > 0:
        noise_array = np.random.randint(-noise, noise + 1, blur_section.shape,
                                       dtype=np.int16)
        blur_section = np.clip(blur_section.astype(np.int16) + noise_array, 0, 255).astype(np.uint8)

    return blur_section
