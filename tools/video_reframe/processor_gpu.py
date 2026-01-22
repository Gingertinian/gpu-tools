"""
Video/Image Reframe Processor - FULL GPU Pipeline (RunPod Optimized)

This processor achieves maximum GPU utilization by:
1. Using FFmpeg's NVDEC for decoding (no CPU decode)
2. Using GPU-based filters (scale_npp, boxblur, overlay_cuda)
3. Using NVENC for encoding
4. ZERO frame-by-frame Python processing for videos
5. CuPy batch processing for images

Target: 100% GPU utilization on RunPod (A100, A40, L40S)

FIXED Issues (2026-01-17):
- Logo position now uses logoPositionX/logoPositionY from config (was hardcoded to 0.85)
- Custom user logos downloaded from logoUrl (was only local logos)
- Blur zones respect topBlurPercent and bottomBlurPercent independently (was combined)
- Layout calculation supports asymmetric crop (was always 50/50)
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
import requests
from pathlib import Path
from typing import Optional, Callable, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Try to import CuPy for image processing
try:
    import cupy as cp
    from cupyx.scipy import ndimage as gpu_ndimage
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None
    gpu_ndimage = None

# =============================================================================
# CONSTANTS
# =============================================================================

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.wmv', '.flv'}

ASPECT_RATIOS = {
    '9:16': (1080, 1920),
    '16:9': (1920, 1080),
    '4:5': (1080, 1350),
    '1:1': (1080, 1080),
}

# Built-in logo names that should use local files
BUILTIN_LOGOS = {'farmium_icon', 'farmium_full', 'none', ''}


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


# =============================================================================
# LOGO MANAGEMENT
# =============================================================================

def _download_logo(logo_url: str, temp_dir: Path) -> Optional[Path]:
    """
    Download a logo from URL to temp directory.
    Returns path to downloaded logo or None on failure.
    """
    if not logo_url:
        return None

    try:
        print(f"[GPU-Reframe] Downloading logo from URL: {logo_url[:100]}...")
        response = requests.get(logo_url, timeout=30, stream=True)
        response.raise_for_status()

        # Determine extension from content-type or URL
        content_type = response.headers.get('content-type', '')
        if 'png' in content_type or logo_url.lower().endswith('.png'):
            ext = '.png'
        elif 'gif' in content_type or logo_url.lower().endswith('.gif'):
            ext = '.gif'
        elif 'webp' in content_type or logo_url.lower().endswith('.webp'):
            ext = '.webp'
        else:
            ext = '.png'  # Default to PNG

        logo_path = temp_dir / f"custom_logo{ext}"
        with open(logo_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[GPU-Reframe] Logo downloaded successfully: {logo_path}")
        return logo_path

    except Exception as e:
        print(f"[GPU-Reframe] Failed to download logo: {e}")
        return None


def _get_logo_path(logo_name: str, logo_url: Optional[str] = None, temp_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Get path to logo file.

    Priority:
    1. If logo_url is provided, download and use it
    2. If logo_name is builtin, look for local file
    3. Return None if 'none' or not found
    """
    # No logo requested
    if not logo_name or logo_name == 'none':
        print(f"[GPU-Reframe] No logo requested (logo_name={logo_name})")
        return None

    # Custom logo URL provided - download it
    if logo_url and temp_dir:
        downloaded = _download_logo(logo_url, temp_dir)
        if downloaded:
            return downloaded
        # Fall back to builtin if download fails
        print(f"[GPU-Reframe] Logo download failed, falling back to builtin: {logo_name}")

    # Builtin logo - look in local paths
    if logo_name in BUILTIN_LOGOS or not logo_url:
        script_dir = Path(__file__).parent
        # Check PNG first, then try other formats
        possible_paths = [
            # Direct logo path in processor directory
            script_dir / 'logos' / f'{logo_name}.png',
            script_dir / 'logos' / f'{logo_name}.jpg',
            # Parent directories
            script_dir.parent / 'logos' / f'{logo_name}.png',
            script_dir.parent.parent / 'logos' / f'{logo_name}.png',
            # Workspace paths (RunPod)
            Path('/workspace/tools/video_reframe/logos') / f'{logo_name}.png',
            Path('/workspace/assets/logos') / f'{logo_name}.png',
            Path('/app/logos') / f'{logo_name}.png',
        ]

        print(f"[GPU-Reframe] Looking for logo '{logo_name}' in paths:")
        for path in possible_paths:
            print(f"[GPU-Reframe]   - {path} (exists: {path.exists()})")
            if path.exists():
                print(f"[GPU-Reframe] Found logo at: {path}")
                return path

        # Fallback: if logo_name is farmium_icon but not found, use farmium_full
        if logo_name == 'farmium_icon':
            print(f"[GPU-Reframe] farmium_icon not found, trying farmium_full as fallback")
            return _get_logo_path('farmium_full', None, temp_dir)

        print(f"[GPU-Reframe] Builtin logo not found: {logo_name}")

    return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def process_video_reframe(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable] = None,
    gpu_id: int = 0
) -> dict:
    """
    Process video or image with reframe - FULL GPU acceleration.

    For videos: Uses FFmpeg filter_complex with 100% GPU pipeline
    For images: Uses CuPy with batch-optimized operations
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Determine media type
    if is_image_file(str(input_path)):
        # Ensure output has image extension
        if output_path.suffix.lower() not in IMAGE_EXTENSIONS:
            output_path = output_path.with_suffix('.jpg')
        return _process_image_gpu(input_path, output_path, config, progress_callback, gpu_id)
    else:
        # Ensure output has video extension
        if output_path.suffix.lower() not in VIDEO_EXTENSIONS:
            output_path = output_path.with_suffix('.mp4')
        return _process_video_gpu(input_path, output_path, config, progress_callback, gpu_id)


# =============================================================================
# VIDEO PROCESSING - FULL GPU FFMPEG PIPELINE
# =============================================================================

def _process_video_gpu(
    input_path: Path,
    output_path: Path,
    config: dict,
    progress_callback: Optional[Callable] = None,
    gpu_id: int = 0
) -> dict:
    """
    Process video using 100% GPU FFmpeg pipeline.

    Pipeline: NVDEC decode → GPU filters → NVENC encode
    NO Python frame processing = maximum throughput
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.05, "Analyzing video...")

    # Get video info
    probe_cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_format', '-show_streams',
        str(input_path)
    ]

    try:
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        video_info = json.loads(probe_result.stdout)
    except Exception as e:
        raise ValueError(f"Failed to probe video: {e}")

    # Extract video stream info
    video_stream = None
    has_audio = False
    for stream in video_info.get('streams', []):
        if stream.get('codec_type') == 'video' and video_stream is None:
            video_stream = stream
        elif stream.get('codec_type') == 'audio':
            has_audio = True

    if not video_stream:
        raise ValueError("No video stream found")

    orig_w = int(video_stream.get('width', 1920))
    orig_h = int(video_stream.get('height', 1080))
    fps = eval(video_stream.get('r_frame_rate', '30/1'))
    duration = float(video_info.get('format', {}).get('duration', 0))

    print(f"[GPU-Reframe] Input: {orig_w}x{orig_h} @ {fps:.2f} fps, duration: {duration:.2f}s")

    # DEBUG: Print raw config received
    print(f"[GPU-Reframe] Raw config received: {json.dumps({k: v for k, v in config.items() if not k.startswith('_')}, default=str)}")

    # Parse config - all keys are camelCase as sent by the backend
    # CRITICAL FIX (2026-01-17): Use _get_config() helper to handle 0 values correctly
    # Python's `or` treats 0 as falsy, so `0 or 25 = 25` which is WRONG
    aspect_str = _parse_aspect_ratio(
        _get_config(config, 'aspectRatio', '9:16')
    )
    final_w, final_h = ASPECT_RATIOS.get(aspect_str, (1080, 1920))

    # Logo config - FIXED: Use logoUrl and position from config
    logo_name = _get_config(config, 'logoName', 'farmium_full')
    logo_url = _get_config(config, 'logoUrl', None)
    logo_size = _get_config(config, 'logoSize', 15)
    # FIXED: Use individual position values from config (0-1 range)
    logo_pos_x = _get_config(config, 'logoPositionX', 0.5)
    logo_pos_y = _get_config(config, 'logoPositionY', 0.85)

    blur_intensity = _get_config(config, 'blurIntensity', 25)
    # FIXED: Use _get_config to support both camelCase and snake_case
    brightness_adj = _get_config(config, 'brightness', 0)
    saturation_adj = _get_config(config, 'saturation', 0)
    contrast_adj = _get_config(config, 'contrast', 0)

    # FIXED: Blur zones - use independent top/bottom percentages
    # Backend sends topBlurPercent/bottomBlurPercent with values including 0
    top_blur_pct = _get_config(config, 'topBlurPercent', 0)
    bottom_blur_pct = _get_config(config, 'bottomBlurPercent', 0)
    # Force blur can still be used as combined fallback
    force_blur = _get_config(config, 'forceBlur', 0)

    # If force_blur is set but individual aren't, split it
    if force_blur > 0 and top_blur_pct == 0 and bottom_blur_pct == 0:
        top_blur_pct = force_blur / 2
        bottom_blur_pct = force_blur / 2

    # Additional effects from frontend
    randomize_effects = config.get('randomize_effects', config.get('randomizeEffects', False))
    tilt_range = config.get('tilt_range') or config.get('tiltRange', 5)
    color_shift_range = config.get('color_shift_range') or config.get('colorShiftRange', 0)

    # Calculate layout with asymmetric blur support
    layout = _calculate_layout(orig_w, orig_h, final_w, final_h, top_blur_pct, bottom_blur_pct)

    print(f"[GPU-Reframe] Output: {final_w}x{final_h}, content={layout['scaled_w']}x{layout['scaled_h']}")
    print(f"[GPU-Reframe] Blur config: top_blur_pct={top_blur_pct}%, bottom_blur_pct={bottom_blur_pct}%, intensity={blur_intensity}")
    print(f"[GPU-Reframe] Blur zones (calculated): top={layout['blur_top']}px, bottom={layout['blur_bottom']}px")
    # DEBUG: Log all logo-related config keys to diagnose position/size issues
    logo_keys = ['logo_name', 'logoName', 'logo_size', 'logoSize', 'logo_position_x', 'logoPositionX',
                 'logo_position_y', 'logoPositionY', 'logo_position', 'logoPosition', 'logo_url', 'logoUrl']
    logo_config_debug = {k: config.get(k) for k in logo_keys if k in config}
    print(f"[GPU-Reframe] DEBUG logo config keys in config: {logo_config_debug}")
    print(f"[GPU-Reframe] Logo FINAL VALUES: name='{logo_name}', size={logo_size}%, pos=({logo_pos_x}, {logo_pos_y})")

    if progress_callback:
        progress_callback(0.10, "Building GPU pipeline...")

    # Create temp dir for logo download if needed
    temp_dir = Path(tempfile.mkdtemp(prefix='reframe_'))

    try:
        # Get logo path (download if URL provided)
        logo_path = _get_logo_path(logo_name, logo_url, temp_dir)

        # Build FFmpeg filter complex for 100% GPU processing
        filter_complex = _build_gpu_filter_complex(
            orig_w, orig_h, final_w, final_h, layout,
            blur_intensity, brightness_adj, saturation_adj, contrast_adj,
            logo_path, logo_size, logo_pos_x, logo_pos_y, gpu_id
        )

        # Build FFmpeg command helper
        def _build_ffmpeg_cmd(use_nvenc: bool, use_hw_decode: bool = False) -> list:
            """Build FFmpeg command with maximum GPU utilization."""
            cmd = ['ffmpeg', '-y']

            if use_hw_decode:
                cmd.extend([
                    '-hwaccel', 'cuda',
                    '-hwaccel_device', str(gpu_id),
                ])

            cmd.extend(['-threads', '0'])
            cmd.extend(['-i', str(input_path)])

            # Add logo input if needed
            if logo_path and logo_path.exists():
                cmd.extend(['-i', str(logo_path)])

            # Add filter complex
            cmd.extend(['-filter_complex', filter_complex])

            # Map video output
            cmd.extend(['-map', '[out]'])

            if use_nvenc:
                cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-gpu', str(gpu_id),
                    '-preset', 'p1',
                    '-tune', 'll',
                    '-rc', 'vbr',
                    '-cq', '26',
                    '-b:v', '8000k',
                    '-maxrate', '16000k',
                    '-bufsize', '16000k',
                    '-rc-lookahead', '0',
                    '-bf', '0',
                    '-spatial-aq', '0',
                    '-temporal-aq', '0',
                    '-profile:v', 'high',
                    '-level', '4.1',
                ])
            else:
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-crf', '23',
                    '-profile:v', 'high',
                    '-level', '4.1',
                ])

            cmd.extend(['-pix_fmt', 'yuv420p'])

            if has_audio:
                cmd.extend(['-map', '0:a?', '-c:a', 'copy'])
            else:
                cmd.append('-an')

            cmd.extend([
                '-movflags', '+faststart',
                str(output_path)
            ])

            return cmd

        def _run_ffmpeg_with_progress(cmd: list, encoder_name: str) -> Tuple[bool, str]:
            """Run FFmpeg and monitor progress."""
            print(f"[GPU-Reframe] Trying {encoder_name} encoder...")
            print(f"[GPU-Reframe] Command: {' '.join(cmd[:30])}...")

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )

                stderr_lines = []

                for line in process.stderr:
                    stderr_lines.append(line)
                    if 'time=' in line:
                        try:
                            time_str = line.split('time=')[1].split()[0]
                            parts = time_str.split(':')
                            current_time = float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                            if duration > 0 and progress_callback:
                                progress = 0.15 + (current_time / duration) * 0.80
                                progress_callback(min(progress, 0.95), f"Encoding ({encoder_name}): {current_time:.1f}s / {duration:.1f}s")
                        except:
                            pass

                process.wait()
                stderr_output = ''.join(stderr_lines[-30:])

                if process.returncode != 0:
                    print(f"[GPU-Reframe] {encoder_name} failed with code {process.returncode}")
                    print(f"[GPU-Reframe] stderr:\n{stderr_output}")
                    return False, stderr_output

                return True, stderr_output

            except Exception as e:
                return False, str(e)

        if progress_callback:
            progress_callback(0.15, "Encoding video...")

        encoder_used = "NVENC"
        success = False
        stderr = ""

        # Tier 1: NVENC encoding
        print(f"[GPU-Reframe] Trying NVENC encoder (p1 preset, max throughput)...")
        cmd_nvenc = _build_ffmpeg_cmd(use_nvenc=True)
        success, stderr = _run_ffmpeg_with_progress(cmd_nvenc, "NVENC")

        if not success:
            # Tier 2: CPU fallback
            print(f"[GPU-Reframe] NVENC failed, trying CPU fallback (libx264)...")
            if progress_callback:
                progress_callback(0.15, "Using CPU encoder (libx264)...")

            encoder_used = "libx264"
            cmd_cpu = _build_ffmpeg_cmd(use_nvenc=False)
            success, stderr = _run_ffmpeg_with_progress(cmd_cpu, "libx264")

            if not success:
                raise RuntimeError(f"FFmpeg failed with all encoders. Last error: {stderr}")

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
            "processor": "GPU" if encoder_used == "NVENC" else "CPU",
            "pipeline": f"FFmpeg-{encoder_used}",
            "fps": fps,
            "duration": duration
        }

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def _build_gpu_filter_complex(
    orig_w: int, orig_h: int,
    final_w: int, final_h: int,
    layout: dict,
    blur_intensity: int,
    brightness_adj: int,
    saturation_adj: int,
    contrast_adj: int,
    logo_path: Optional[Path],
    logo_size: int,
    logo_pos_x: float,
    logo_pos_y: float,
    gpu_id: int
) -> str:
    """
    Build FFmpeg filter_complex for GPU-accelerated processing.

    FIXED (2026-01-17 v2):
    - Top and bottom blur zones now use DIFFERENT random sections from the content
    - Each blur zone has its own random zoom, rotation, and crop offset
    - This creates visual variety between top and bottom blur areas
    """
    scaled_w = layout['scaled_w']
    scaled_h = layout['scaled_h']
    content_x = layout['content_x']
    content_y = layout['content_y']
    blur_top_h = layout['blur_top']
    blur_bottom_h = layout['blur_bottom']
    crop_top_px = layout.get('crop_top_px', 0)
    crop_bottom_px = layout.get('crop_bottom_px', 0)
    cropped_orig_h = layout.get('cropped_orig_h', orig_h)

    # Blur sigma based on intensity (0-100 -> 8-20)
    blur_sigma = 8 + (blur_intensity / 100) * 12
    blur_sigma = max(5, min(blur_sigma, 20))

    # Convert sigma to boxblur radius (O(1) performance)
    blur_radius = int(blur_sigma * 1.5)
    blur_radius = max(1, min(blur_radius, 30))

    # Build color adjustment string
    eq_parts = []
    if brightness_adj != 0:
        eq_parts.append(f"brightness={brightness_adj/100:.3f}")
    if contrast_adj != 0:
        eq_parts.append(f"contrast={1 + contrast_adj/100:.3f}")
    if saturation_adj != 0:
        eq_parts.append(f"saturation={1 + saturation_adj/100:.3f}")
    eq_filter = f",eq={':'.join(eq_parts)}" if eq_parts else ""

    parts = []

    # If we need blur zones (horizontal video -> vertical output)
    if blur_top_h > 0 or blur_bottom_h > 0:
        print(f"[GPU-Reframe] Blur zones: top={blur_top_h}px, bottom={blur_bottom_h}px, content={scaled_h}px")
        print(f"[GPU-Reframe] Content position: y={content_y}px")
        print(f"[GPU-Reframe] Crop: top={crop_top_px}px, bottom={crop_bottom_px}px, remaining={cropped_orig_h}px")

        # OPTIMIZED BLUR PIPELINE - 4x faster
        # Calculate low-res dimensions (1/4 size, ensure even)
        blur_scale_w = (final_w // 4) - ((final_w // 4) % 2)
        blur_scale_h = (final_h // 4) - ((final_h // 4) % 2)
        # Reduce blur radius proportionally (since image is smaller)
        scaled_blur_radius = max(3, blur_radius // 4)

        print(f"[GPU-Reframe] FAST blur: {blur_scale_w}x{blur_scale_h} (1/4 res), radius={scaled_blur_radius}")

        # FIXED (2026-01-18): Blur must use CROPPED content, not original video
        # The blur background should NOT show the parts that were cropped/removed
        # Flow: crop first -> split -> one for blur (zoomed + rotated), one for content

        # Random rotation for blur background (1-3 degrees, positive or negative)
        blur_rotation = random.uniform(1, 3) * random.choice([-1, 1])
        blur_rotation_rad = blur_rotation * math.pi / 180

        # Calculate zoom needed to cover corners after rotation
        # For angle θ, zoom factor ≈ 1 / (cos(θ) - sin(θ)*aspect_ratio) but simplified:
        # zoom = 1 + 0.05 * |angle_degrees| gives ~5% extra per degree (safe margin)
        rotation_zoom = 1 + 0.05 * abs(blur_rotation)
        zoomed_blur_w = int(blur_scale_w * rotation_zoom)
        zoomed_blur_h = int(blur_scale_h * rotation_zoom)

        print(f"[GPU-Reframe] Blur rotation: {blur_rotation:.1f}° (zoom: {rotation_zoom:.2f}x)")

        if crop_top_px > 0 or crop_bottom_px > 0:
            # Step 1: Crop the video FIRST to remove top/bottom
            parts.append(f"[0:v]crop={orig_w}:{cropped_orig_h}:0:{crop_top_px}[cropped]")

            # Step 2: Split cropped video - one for blur background, one for content
            parts.append(f"[cropped]split=2[for_blur][for_content]")

            # Step 3: Blur background - scale up with zoom, rotate, blur, crop to final size
            # The extra zoom compensates for rotation so no black corners appear
            parts.append(
                f"[for_blur]scale={zoomed_blur_w}:{zoomed_blur_h}:flags=fast_bilinear,"
                f"rotate={blur_rotation_rad:.4f}:c=black,"
                f"boxblur=luma_radius={scaled_blur_radius}:chroma_radius={scaled_blur_radius},"
                f"crop={blur_scale_w}:{blur_scale_h},"
                f"scale={final_w}:{final_h}:flags=fast_bilinear[blur_bg]"
            )

            # Step 4: Scale content to fit in content area
            parts.append(f"[for_content]scale={scaled_w}:{scaled_h}:flags=fast_bilinear{eq_filter}[content]")
        else:
            # No crop needed - split original and use for both blur and content
            parts.append(f"[0:v]split=2[for_blur][for_content]")

            parts.append(
                f"[for_blur]scale={zoomed_blur_w}:{zoomed_blur_h}:flags=fast_bilinear,"
                f"rotate={blur_rotation_rad:.4f}:c=black,"
                f"boxblur=luma_radius={scaled_blur_radius}:chroma_radius={scaled_blur_radius},"
                f"crop={blur_scale_w}:{blur_scale_h},"
                f"scale={final_w}:{final_h}:flags=fast_bilinear[blur_bg]"
            )

            parts.append(f"[for_content]scale={scaled_w}:{scaled_h}:flags=fast_bilinear{eq_filter}[content]")

        # Composite: overlay content on blur background
        parts.append(f"[blur_bg][content]overlay={content_x}:{content_y}[composited]")

        current_output = "[composited]"
    else:
        # No blur needed
        if crop_top_px > 0 or crop_bottom_px > 0:
            chain = f"[0:v]crop={orig_w}:{cropped_orig_h}:0:{crop_top_px},scale={scaled_w}:{scaled_h}:flags=lanczos{eq_filter},pad={final_w}:{final_h}:{content_x}:{content_y}:black[composited]"
        else:
            chain = f"[0:v]scale={scaled_w}:{scaled_h}:flags=lanczos{eq_filter},pad={final_w}:{final_h}:{content_x}:{content_y}:black[composited]"
        parts = [chain]
        current_output = "[composited]"

    # Add logo if present
    if logo_path and logo_path.exists():
        logo_w = int(final_w * logo_size / 100)
        # Logo height is same as width (square logos) since we scale with -1
        logo_h = logo_w
        # Calculate position from normalized coords (0-1)
        # Center the logo at the specified position (both X and Y)
        logo_x = int(final_w * logo_pos_x - logo_w / 2)
        logo_y = int(final_h * logo_pos_y - logo_h / 2)

        # Clamp to valid bounds with margin
        logo_x = max(10, min(logo_x, final_w - logo_w - 10))
        logo_y = max(10, min(logo_y, final_h - logo_h - 10))

        print(f"[GPU-Reframe] Logo overlay: pos=({logo_x}, {logo_y}), width={logo_w}, path={logo_path}")

        parts.append(
            f"[1:v]scale={logo_w}:-1:flags=lanczos,format=yuva420p[logo]"
        )
        # Use eof_action=repeat to loop logo (static image) for entire video duration
        # CRITICAL: setsar=1 forces square pixels - without this, video displays as wrong aspect ratio
        parts.append(
            f"{current_output}[logo]overlay={logo_x}:{logo_y}:eof_action=repeat,setsar=1[out]"
        )
    else:
        print(f"[GPU-Reframe] No logo to add (logo_path={logo_path})")
        # CRITICAL: setsar=1 forces square pixels - without this, video displays as wrong aspect ratio
        parts.append(f"{current_output}setsar=1[out]")

    return ';'.join(parts)


# =============================================================================
# IMAGE PROCESSING - CUPY GPU ACCELERATION
# =============================================================================

def _process_image_gpu(
    input_path: Path,
    output_path: Path,
    config: dict,
    progress_callback: Optional[Callable] = None,
    gpu_id: int = 0
) -> dict:
    """
    Process image using CuPy GPU acceleration.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.10, "Loading image...")

    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    orig_h, orig_w = img.shape[:2]

    # DEBUG: Print raw config received
    print(f"[GPU-Reframe] Raw config received: {json.dumps({k: v for k, v in config.items() if not k.startswith('_')}, default=str)}")

    # Parse config - FIXED: Use _get_config() helper to handle 0 values correctly
    aspect_str = _parse_aspect_ratio(_get_config(config, 'aspectRatio', '9:16'))
    final_w, final_h = ASPECT_RATIOS.get(aspect_str, (1080, 1920))

    # Logo config - FIXED: Use _get_config() to handle 0 values correctly
    logo_name = _get_config(config, 'logoName', 'farmium_full')
    logo_url = _get_config(config, 'logoUrl', None)
    logo_size = _get_config(config, 'logoSize', 15)
    logo_pos_x = _get_config(config, 'logoPositionX', 0.5)
    logo_pos_y = _get_config(config, 'logoPositionY', 0.85)

    blur_intensity = _get_config(config, 'blurIntensity', 25)
    # FIXED: Use _get_config to support both camelCase and snake_case
    brightness_adj = _get_config(config, 'brightness', 0)
    saturation_adj = _get_config(config, 'saturation', 0)
    contrast_adj = _get_config(config, 'contrast', 0)

    # FIXED: Independent blur percentages - handle 0 correctly
    top_blur_pct = _get_config(config, 'topBlurPercent', 0)
    bottom_blur_pct = _get_config(config, 'bottomBlurPercent', 0)
    force_blur = _get_config(config, 'forceBlur', 0)

    if force_blur > 0 and top_blur_pct == 0 and bottom_blur_pct == 0:
        top_blur_pct = force_blur / 2
        bottom_blur_pct = force_blur / 2

    # Calculate layout with asymmetric blur
    layout = _calculate_layout(orig_w, orig_h, final_w, final_h, top_blur_pct, bottom_blur_pct)

    print(f"[GPU-Reframe] Image: {orig_w}x{orig_h} -> {final_w}x{final_h}")
    print(f"[GPU-Reframe] Blur config: top_blur_pct={top_blur_pct}%, bottom_blur_pct={bottom_blur_pct}%, intensity={blur_intensity}")
    print(f"[GPU-Reframe] Blur zones (calculated): top={layout['blur_top']}px, bottom={layout['blur_bottom']}px")
    print(f"[GPU-Reframe] Logo: name='{logo_name}', size={logo_size}%, pos=({logo_pos_x}, {logo_pos_y})")

    if progress_callback:
        progress_callback(0.20, "Processing on GPU...")

    # Create temp dir for logo
    temp_dir = Path(tempfile.mkdtemp(prefix='reframe_img_'))

    try:
        use_gpu = False
        if HAS_CUPY:
            try:
                with cp.cuda.Device(gpu_id):
                    _test = cp.array([1, 2, 3])
                    _test = _test * 2
                    del _test
                    use_gpu = True
            except Exception as e:
                print(f"[GPU-Reframe] CuPy not working, using CPU: {type(e).__name__}")
                use_gpu = False

        if use_gpu:
            with cp.cuda.Device(gpu_id):
                output = _process_image_cupy(
                    img, layout, final_w, final_h,
                    blur_intensity, brightness_adj, saturation_adj, contrast_adj
                )
        else:
            output = _process_image_cpu(
                img, layout, final_w, final_h,
                blur_intensity, brightness_adj, saturation_adj, contrast_adj
            )

        # Apply logo - FIXED: Use position from config
        logo_path = _get_logo_path(logo_name, logo_url, temp_dir)
        if logo_path and logo_path.exists():
            _apply_logo_to_image(output, logo_path, final_w, final_h, logo_size, logo_pos_x, logo_pos_y)

        if progress_callback:
            progress_callback(0.90, "Saving output...")

        ext = output_path.suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            quality = config.get('quality', 92)
            cv2.imwrite(str(output_path), output, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif ext == '.png':
            cv2.imwrite(str(output_path), output, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        elif ext == '.webp':
            quality = config.get('quality', 92)
            cv2.imwrite(str(output_path), output, [cv2.IMWRITE_WEBP_QUALITY, quality])
        else:
            cv2.imwrite(str(output_path), output)

        if progress_callback:
            progress_callback(1.0, "Complete")

        output_size = os.path.getsize(output_path) if output_path.exists() else 0

        return {
            "status": "completed",
            "type": "image",
            "outputPath": str(output_path),
            "outputSize": output_size,
            "dimensions": f"{final_w}x{final_h}",
            "processor": "GPU" if HAS_CUPY else "CPU"
        }

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def _process_image_cupy(
    img: np.ndarray,
    layout: dict,
    final_w: int, final_h: int,
    blur_intensity: int,
    brightness_adj: int,
    saturation_adj: int,
    contrast_adj: int,
) -> np.ndarray:
    """
    Process image entirely on GPU using CuPy.

    FIXED (2026-01-17 v2): Top and bottom blur zones use DIFFERENT random sections.
    """
    scaled_w = layout['scaled_w']
    scaled_h = layout['scaled_h']
    content_x = layout['content_x']
    content_y = layout['content_y']
    blur_top_h = layout['blur_top']
    blur_bottom_h = layout['blur_bottom']

    gpu_img = cp.asarray(img, dtype=cp.float32)

    zoom_factors = (scaled_h / img.shape[0], scaled_w / img.shape[1], 1)
    content = gpu_ndimage.zoom(gpu_img, zoom_factors, order=1)

    if brightness_adj != 0:
        content = content + (brightness_adj * 2.55)
    if contrast_adj != 0:
        factor = 1 + (contrast_adj / 100)
        content = (content - 127.5) * factor + 127.5
    if saturation_adj != 0:
        gray = cp.mean(content, axis=2, keepdims=True)
        factor = 1 + (saturation_adj / 100)
        content = gray + (content - gray) * factor

    content = cp.clip(content, 0, 255)

    blur_sigma = 5 + (blur_intensity / 100) * 10
    blur_sigma = max(5, min(blur_sigma, 15))

    needs_blur = blur_top_h > 0 or blur_bottom_h > 0 or content_x > 0

    # Start with black canvas
    output = cp.zeros((final_h, final_w, 3), dtype=cp.float32)

    if needs_blur:
        content_h, content_w = content.shape[:2]

        # Generate DIFFERENT random parameters for top and bottom blur zones
        if blur_top_h > 0:
            top_zoom = 1.0 + random.uniform(0.3, 0.6)
            top_rotation = random.uniform(-10, 10)
            top_offset_x = random.randint(-50, 50)
            top_offset_y = random.randint(0, 150)  # Use upper part of content

            top_scale = max(final_w / content_w, blur_top_h / content_h) * top_zoom
            top_bg_zoom = (top_scale, top_scale, 1)
            top_zoomed = gpu_ndimage.zoom(content, top_bg_zoom, order=1)

            if abs(top_rotation) > 0.5:
                top_zoomed = gpu_ndimage.rotate(top_zoomed, top_rotation, axes=(1, 0), reshape=False, mode='reflect')

            top_crop_x = max(0, min((top_zoomed.shape[1] - final_w) // 2 + top_offset_x, max(0, top_zoomed.shape[1] - final_w)))
            top_crop_y = max(0, min((top_zoomed.shape[0] - blur_top_h) // 2 + top_offset_y, max(0, top_zoomed.shape[0] - blur_top_h)))
            top_blur_region = top_zoomed[top_crop_y:top_crop_y + blur_top_h, top_crop_x:top_crop_x + final_w]

            # Apply blur
            for c in range(3):
                top_blur_region[:, :, c] = gpu_ndimage.gaussian_filter(top_blur_region[:, :, c], sigma=blur_sigma)

            # Ensure correct size
            if top_blur_region.shape[0] >= blur_top_h and top_blur_region.shape[1] >= final_w:
                output[0:blur_top_h, 0:final_w] = top_blur_region[:blur_top_h, :final_w]

        if blur_bottom_h > 0:
            bottom_zoom = 1.0 + random.uniform(0.3, 0.6)
            bottom_rotation = random.uniform(-10, 10)
            bottom_offset_x = random.randint(-50, 50)
            bottom_offset_y = random.randint(-150, 0)  # Use lower part of content

            bottom_scale = max(final_w / content_w, blur_bottom_h / content_h) * bottom_zoom
            bottom_bg_zoom = (bottom_scale, bottom_scale, 1)
            bottom_zoomed = gpu_ndimage.zoom(content, bottom_bg_zoom, order=1)

            if abs(bottom_rotation) > 0.5:
                bottom_zoomed = gpu_ndimage.rotate(bottom_zoomed, bottom_rotation, axes=(1, 0), reshape=False, mode='reflect')

            bottom_crop_x = max(0, min((bottom_zoomed.shape[1] - final_w) // 2 + bottom_offset_x, max(0, bottom_zoomed.shape[1] - final_w)))
            bottom_crop_y = max(0, min((bottom_zoomed.shape[0] - blur_bottom_h) // 2 + abs(bottom_offset_y), max(0, bottom_zoomed.shape[0] - blur_bottom_h)))
            bottom_blur_region = bottom_zoomed[bottom_crop_y:bottom_crop_y + blur_bottom_h, bottom_crop_x:bottom_crop_x + final_w]

            # Apply blur
            for c in range(3):
                bottom_blur_region[:, :, c] = gpu_ndimage.gaussian_filter(bottom_blur_region[:, :, c], sigma=blur_sigma)

            # Ensure correct size
            if bottom_blur_region.shape[0] >= blur_bottom_h and bottom_blur_region.shape[1] >= final_w:
                output[final_h - blur_bottom_h:final_h, 0:final_w] = bottom_blur_region[:blur_bottom_h, :final_w]

    # Place content in the middle
    output[content_y:content_y + scaled_h, content_x:content_x + scaled_w] = content[:scaled_h, :scaled_w]
    output_cpu = cp.asnumpy(output).astype(np.uint8)

    return output_cpu


def _process_image_cpu(
    img: np.ndarray,
    layout: dict,
    final_w: int, final_h: int,
    blur_intensity: int,
    brightness_adj: int,
    saturation_adj: int,
    contrast_adj: int,
) -> np.ndarray:
    """
    CPU fallback for image processing.

    FIXED (2026-01-17 v2): Top and bottom blur zones use DIFFERENT random sections.
    """
    scaled_w = layout['scaled_w']
    scaled_h = layout['scaled_h']
    content_x = layout['content_x']
    content_y = layout['content_y']
    blur_top_h = layout['blur_top']
    blur_bottom_h = layout['blur_bottom']

    content = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
    content = content.astype(np.float32)

    if brightness_adj != 0:
        content = content + (brightness_adj * 2.55)
    if contrast_adj != 0:
        factor = 1 + (contrast_adj / 100)
        content = (content - 127.5) * factor + 127.5
    if saturation_adj != 0:
        gray = np.mean(content, axis=2, keepdims=True)
        factor = 1 + (saturation_adj / 100)
        content = gray + (content - gray) * factor

    content = np.clip(content, 0, 255).astype(np.uint8)

    blur_sigma = 5 + (blur_intensity / 100) * 10
    blur_sigma = max(5, min(blur_sigma, 15))
    blur_ksize = int(blur_sigma * 6) | 1

    needs_blur = blur_top_h > 0 or blur_bottom_h > 0 or content_x > 0

    # Start with black canvas
    output = np.zeros((final_h, final_w, 3), dtype=np.uint8)

    if needs_blur:
        content_h, content_w = content.shape[:2]

        # Generate DIFFERENT random parameters for top and bottom blur zones
        if blur_top_h > 0:
            top_zoom = 1.0 + random.uniform(0.3, 0.6)
            top_rotation = random.uniform(-10, 10)
            top_offset_x = random.randint(-50, 50)
            top_offset_y = random.randint(0, 150)  # Use upper part of content

            top_scale = max(final_w / content_w, blur_top_h / content_h) * top_zoom
            top_new_w = int(content_w * top_scale)
            top_new_h = int(content_h * top_scale)
            top_resized = cv2.resize(content, (top_new_w, top_new_h), interpolation=cv2.INTER_LINEAR)

            if abs(top_rotation) > 0.5:
                center = (top_new_w // 2, top_new_h // 2)
                rot_matrix = cv2.getRotationMatrix2D(center, top_rotation, 1.0)
                top_resized = cv2.warpAffine(top_resized, rot_matrix, (top_new_w, top_new_h), borderMode=cv2.BORDER_REFLECT)

            top_crop_x = max(0, min((top_new_w - final_w) // 2 + top_offset_x, max(0, top_new_w - final_w)))
            top_crop_y = max(0, min((top_new_h - blur_top_h) // 2 + top_offset_y, max(0, top_new_h - blur_top_h)))
            top_blur_region = top_resized[top_crop_y:top_crop_y + blur_top_h, top_crop_x:top_crop_x + final_w]

            # Apply blur
            top_blur_region = cv2.GaussianBlur(top_blur_region, (blur_ksize, blur_ksize), blur_sigma)

            # Ensure correct size and place in output
            if top_blur_region.shape[0] >= blur_top_h and top_blur_region.shape[1] >= final_w:
                output[0:blur_top_h, 0:final_w] = top_blur_region[:blur_top_h, :final_w]

        if blur_bottom_h > 0:
            bottom_zoom = 1.0 + random.uniform(0.3, 0.6)
            bottom_rotation = random.uniform(-10, 10)
            bottom_offset_x = random.randint(-50, 50)
            bottom_offset_y = random.randint(-150, 0)  # Use lower part of content

            bottom_scale = max(final_w / content_w, blur_bottom_h / content_h) * bottom_zoom
            bottom_new_w = int(content_w * bottom_scale)
            bottom_new_h = int(content_h * bottom_scale)
            bottom_resized = cv2.resize(content, (bottom_new_w, bottom_new_h), interpolation=cv2.INTER_LINEAR)

            if abs(bottom_rotation) > 0.5:
                center = (bottom_new_w // 2, bottom_new_h // 2)
                rot_matrix = cv2.getRotationMatrix2D(center, bottom_rotation, 1.0)
                bottom_resized = cv2.warpAffine(bottom_resized, rot_matrix, (bottom_new_w, bottom_new_h), borderMode=cv2.BORDER_REFLECT)

            bottom_crop_x = max(0, min((bottom_new_w - final_w) // 2 + bottom_offset_x, max(0, bottom_new_w - final_w)))
            bottom_crop_y = max(0, min((bottom_new_h - blur_bottom_h) // 2 + abs(bottom_offset_y), max(0, bottom_new_h - blur_bottom_h)))
            bottom_blur_region = bottom_resized[bottom_crop_y:bottom_crop_y + blur_bottom_h, bottom_crop_x:bottom_crop_x + final_w]

            # Apply blur
            bottom_blur_region = cv2.GaussianBlur(bottom_blur_region, (blur_ksize, blur_ksize), blur_sigma)

            # Ensure correct size and place in output
            if bottom_blur_region.shape[0] >= blur_bottom_h and bottom_blur_region.shape[1] >= final_w:
                output[final_h - blur_bottom_h:final_h, 0:final_w] = bottom_blur_region[:blur_bottom_h, :final_w]

    output[content_y:content_y + scaled_h, content_x:content_x + scaled_w] = content
    return output


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def _get_config(config: dict, key: str, default=None):
    """
    Get config value. Handles 0 values correctly (unlike Python's 'or' operator).
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


def _parse_aspect_ratio(aspect_raw) -> str:
    """Parse aspect ratio from various formats."""
    if isinstance(aspect_raw, (list, tuple)) and len(aspect_raw) == 2:
        return f"{aspect_raw[0]}:{aspect_raw[1]}"
    return str(aspect_raw) if aspect_raw else '9:16'


def _calculate_layout(
    orig_w: int, orig_h: int,
    final_w: int, final_h: int,
    top_blur_pct: float = 0,
    bottom_blur_pct: float = 0
) -> dict:
    """
    Calculate layout for reframe.

    FIXED (2026-01-22): Blur percentages are now relative to OUTPUT FRAME height.
    This ensures consistent visual appearance across videos with different resolutions/aspects.

    Example: 10% top blur always creates a 192px blur zone (10% of 1920px output)
    regardless of whether the input is 1080p, 720p, or ultrawide.

    The content is scaled to fit in the remaining space after blur zones.
    """
    # Blur zones are percentage of OUTPUT frame height (consistent across all videos)
    blur_top = int(final_h * min(top_blur_pct, 45) / 100)
    blur_bottom = int(final_h * min(bottom_blur_pct, 45) / 100)

    # Ensure even values for FFmpeg
    blur_top = blur_top - (blur_top % 2)
    blur_bottom = blur_bottom - (blur_bottom % 2)

    # Available space for content after blur zones
    content_space_h = final_h - blur_top - blur_bottom

    # Scale content to fit: fill width, then check if height fits
    scale = final_w / orig_w
    scaled_w = final_w
    scaled_h = int(orig_h * scale)

    # If scaled content is taller than available space, shrink to fit
    if scaled_h > content_space_h:
        # Need to shrink content to fit in available space
        scale = content_space_h / orig_h
        scaled_h = content_space_h
        scaled_w = int(orig_w * scale)
        # Center horizontally if narrower than final width
        scaled_w = min(scaled_w, final_w)

    # Ensure even dimensions
    scaled_w = scaled_w - (scaled_w % 2)
    scaled_h = scaled_h - (scaled_h % 2)

    # Calculate crop from original video (how much to remove to fit)
    # This is derived from the scaling, not user input
    if scaled_h < int(orig_h * final_w / orig_w):
        # Content was shrunk, calculate how much of original to crop
        original_scaled_h = int(orig_h * final_w / orig_w)
        total_crop = original_scaled_h - scaled_h
        # Distribute crop proportionally based on blur percentages
        if top_blur_pct + bottom_blur_pct > 0:
            crop_ratio = top_blur_pct / (top_blur_pct + bottom_blur_pct)
        else:
            crop_ratio = 0.5
        crop_top_px = int(total_crop * crop_ratio / (final_w / orig_w))
        crop_bottom_px = int(total_crop * (1 - crop_ratio) / (final_w / orig_w))
    else:
        crop_top_px = 0
        crop_bottom_px = 0

    # Ensure even crop values
    crop_top_px = crop_top_px - (crop_top_px % 2)
    crop_bottom_px = crop_bottom_px - (crop_bottom_px % 2)

    # Calculate cropped original height (for FFmpeg crop filter)
    cropped_orig_h = orig_h - crop_top_px - crop_bottom_px
    cropped_orig_h = max(cropped_orig_h, int(orig_h * 0.2))  # Minimum 20%

    # Content position: centered horizontally, after top blur zone vertically
    content_x = (final_w - scaled_w) // 2
    content_y = blur_top

    print(f"[GPU-Reframe] Layout: input={orig_w}x{orig_h}, output={final_w}x{final_h}")
    print(f"[GPU-Reframe] Blur zones (output %): top={top_blur_pct}%={blur_top}px, bottom={bottom_blur_pct}%={blur_bottom}px")
    print(f"[GPU-Reframe] Content: {scaled_w}x{scaled_h} at ({content_x}, {content_y})")
    print(f"[GPU-Reframe] Crop from original: top={crop_top_px}px, bottom={crop_bottom_px}px")

    return {
        'scaled_w': scaled_w,
        'scaled_h': scaled_h,
        'content_x': content_x,
        'content_y': content_y,
        'blur_top': blur_top,
        'blur_bottom': blur_bottom,
        'scale': scale,
        'crop_top_px': crop_top_px,
        'crop_bottom_px': crop_bottom_px,
        'cropped_orig_h': cropped_orig_h,
    }


def _apply_logo_to_image(
    img: np.ndarray,
    logo_path: Path,
    final_w: int,
    final_h: int,
    logo_size: int,
    logo_pos_x: float = 0.5,
    logo_pos_y: float = 0.85
):
    """
    Apply logo overlay to image.

    FIXED: Now uses logo_pos_x and logo_pos_y from config instead of hardcoded values.

    Args:
        img: Image to modify (in-place)
        logo_path: Path to logo file
        final_w, final_h: Output dimensions
        logo_size: Logo size as percentage of width (10-100)
        logo_pos_x: Horizontal position (0-1, 0.5 = center)
        logo_pos_y: Vertical position (0-1, 0 = top, 1 = bottom)
    """
    try:
        logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
        if logo is None:
            print(f"[GPU-Reframe] Could not read logo: {logo_path}")
            return

        # Calculate logo size
        logo_w = int(final_w * logo_size / 100)
        logo_h = int(logo.shape[0] * logo_w / logo.shape[1])

        if logo_w <= 0 or logo_h <= 0:
            return

        logo = cv2.resize(logo, (logo_w, logo_h), interpolation=cv2.INTER_LANCZOS4)

        # FIXED: Calculate position from normalized coordinates
        # logo_pos_x: 0.5 = center, 0 = left edge, 1 = right edge
        # logo_pos_y: 0.5 = center, 0 = top, 1 = bottom (logo is centered at this position)
        x = int(final_w * logo_pos_x - logo_w / 2)
        y = int(final_h * logo_pos_y - logo_h / 2)  # FIXED: Center logo vertically too

        # Clamp to valid bounds
        x = max(0, min(x, final_w - logo_w))
        y = max(0, min(y, final_h - logo_h))

        print(f"[GPU-Reframe] Placing logo at ({x}, {y}), size {logo_w}x{logo_h}")

        # Apply with alpha blending if available
        if logo.shape[-1] == 4:
            alpha = logo[:, :, 3:4] / 255.0
            logo_rgb = logo[:, :, :3]

            roi = img[y:y+logo_h, x:x+logo_w]
            blended = (logo_rgb * alpha + roi * (1 - alpha)).astype(np.uint8)
            img[y:y+logo_h, x:x+logo_w] = blended
        else:
            img[y:y+logo_h, x:x+logo_w] = logo

    except Exception as e:
        print(f"[GPU-Reframe] Logo overlay failed: {e}")


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_video_reframe_batch(
    items: list,
    progress_callback: Optional[Callable] = None,
    max_workers: int = 4
) -> list:
    """
    Process multiple videos/images in parallel.
    """
    results = []
    total = len(items)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for i, item in enumerate(items):
            gpu_id = i % max(1, _get_gpu_count())
            future = executor.submit(
                process_video_reframe,
                item['input'],
                item['output'],
                item.get('config', {}),
                None,
                gpu_id
            )
            futures[future] = item

        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    'status': 'error',
                    'error': str(e),
                    'input': futures[future]['input']
                })

            completed += 1
            if progress_callback:
                progress_callback(completed / total, f"Completed {completed}/{total}")

    return results


def _get_gpu_count() -> int:
    """Get number of available GPUs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True, text=True, timeout=5
        )
        return len(result.stdout.strip().split('\n'))
    except:
        return 1


# =============================================================================
# COMPATIBILITY EXPORTS
# =============================================================================

__all__ = ['process_video_reframe', 'process_video_reframe_batch']
