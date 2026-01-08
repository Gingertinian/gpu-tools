"""
Spoofer Processor - Image/Video transformation for duplicate detection evasion

Applies various transformations to media files to evade perceptual hash detection
while maintaining visual quality.

Config structure:
{
    "spatial": { "crop": %, "microResize": %, "rotation": deg, ... },
    "tonal": { "brightness": %, "gamma": val, "contrast": %, ... },
    "visual": { "noise": val, "quality": %, "tint": val, ... },
    "video": { "bitrate": %, "speedVariation": %, ... },
    "pHashTarget": int
}
"""

import os
import random
import math
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Callable, Optional, Dict, Any
import subprocess
import tempfile


def process_spoofer(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process image or video with spoofer transformations.

    Args:
        input_path: Path to input file
        output_path: Path to write output file
        config: Transformation configuration
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
        return process_video(input_path, output_path, config, report_progress)
    else:
        return process_image(input_path, output_path, config, report_progress)


def process_image(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Process a single image with transformations."""

    report_progress(0.1, "Loading image...")

    # Load image
    img = Image.open(input_path).convert('RGB')
    original_size = img.size

    # Extract config sections
    spatial = config.get('spatial', {})
    tonal = config.get('tonal', {})
    visual = config.get('visual', {})

    transforms_applied = []

    # === SPATIAL TRANSFORMS ===
    report_progress(0.2, "Applying spatial transforms...")

    # Crop
    if spatial.get('crop'):
        crop_percent = spatial['crop'] / 100
        if crop_percent > 0:
            w, h = img.size
            crop_px_w = int(w * crop_percent)
            crop_px_h = int(h * crop_percent)
            img = img.crop((crop_px_w, crop_px_h, w - crop_px_w, h - crop_px_h))
            transforms_applied.append(f"crop:{spatial['crop']}%")

    # Micro resize
    if spatial.get('microResize'):
        resize_percent = spatial['microResize'] / 100
        if resize_percent > 0:
            scale = 1 + (random.random() - 0.5) * resize_percent * 2
            w, h = img.size
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            transforms_applied.append(f"microResize:{spatial['microResize']}%")

    # Rotation
    if spatial.get('rotation'):
        rotation_deg = spatial['rotation']
        if rotation_deg > 0:
            angle = (random.random() - 0.5) * rotation_deg * 2
            img = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            transforms_applied.append(f"rotation:{rotation_deg}°")

    # Subpixel shift (via slight resize + offset)
    if spatial.get('subpixel'):
        subpixel = spatial['subpixel']
        if subpixel > 0:
            offset_x = random.uniform(-subpixel, subpixel)
            offset_y = random.uniform(-subpixel, subpixel)
            w, h = img.size
            # Create slightly larger canvas and paste with offset
            new_img = Image.new('RGB', (w, h), (0, 0, 0))
            new_img.paste(img, (int(offset_x), int(offset_y)))
            img = new_img
            transforms_applied.append(f"subpixel:{subpixel}px")

    # Barrel distortion (simplified via ImageFilter)
    if spatial.get('barrel'):
        barrel = spatial['barrel'] / 100
        if barrel > 0:
            # Apply slight lens distortion effect
            img = apply_barrel_distortion(img, barrel)
            transforms_applied.append(f"barrel:{spatial['barrel']}%")

    # Perspective warp
    if spatial.get('warp'):
        warp = spatial['warp'] / 100
        if warp > 0:
            img = apply_perspective_warp(img, warp)
            transforms_applied.append(f"warp:{spatial['warp']}%")

    report_progress(0.4, "Applying tonal transforms...")

    # === TONAL TRANSFORMS ===

    # Convert to numpy for pixel operations
    arr = np.array(img, dtype=np.float32)

    # Brightness
    if tonal.get('brightness'):
        brightness_var = tonal['brightness'] / 100
        if brightness_var > 0:
            adjustment = (random.random() - 0.5) * brightness_var * 2 * 255
            arr = np.clip(arr + adjustment, 0, 255)
            transforms_applied.append(f"brightness:{tonal['brightness']}%")

    # Gamma
    if tonal.get('gamma'):
        gamma_var = tonal['gamma']
        if gamma_var > 0:
            gamma = 1 + (random.random() - 0.5) * gamma_var * 2
            gamma = max(0.5, min(2.0, gamma))  # Clamp to reasonable range
            arr = np.clip(255 * (arr / 255) ** gamma, 0, 255)
            transforms_applied.append(f"gamma:{gamma_var}")

    # Contrast
    if tonal.get('contrast'):
        contrast_var = tonal['contrast'] / 100
        if contrast_var > 0:
            factor = 1 + (random.random() - 0.5) * contrast_var * 2
            mean = np.mean(arr)
            arr = np.clip((arr - mean) * factor + mean, 0, 255)
            transforms_applied.append(f"contrast:{tonal['contrast']}%")

    # Saturation
    if tonal.get('saturation'):
        sat_var = tonal['saturation'] / 100
        if sat_var > 0:
            factor = 1 + (random.random() - 0.5) * sat_var * 2
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
            gray = np.stack([gray, gray, gray], axis=2)
            arr = np.clip(gray + (arr - gray) * factor, 0, 255)
            transforms_applied.append(f"saturation:{tonal['saturation']}%")

    report_progress(0.6, "Applying visual transforms...")

    # === VISUAL TRANSFORMS ===

    # Noise
    if visual.get('noise'):
        noise_level = visual['noise']
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 2, arr.shape)
            arr = np.clip(arr + noise, 0, 255)
            transforms_applied.append(f"noise:{noise_level}")

    # Tint (color shift)
    if visual.get('tint'):
        tint_level = visual['tint']
        if tint_level > 0:
            tint_r = random.uniform(-tint_level, tint_level)
            tint_g = random.uniform(-tint_level, tint_level)
            tint_b = random.uniform(-tint_level, tint_level)
            arr[:, :, 0] = np.clip(arr[:, :, 0] + tint_r, 0, 255)
            arr[:, :, 1] = np.clip(arr[:, :, 1] + tint_g, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] + tint_b, 0, 255)
            transforms_applied.append(f"tint:{tint_level}")

    # Film grain
    if visual.get('grain'):
        grain_level = visual['grain'] / 100
        if grain_level > 0:
            grain = np.random.normal(0, grain_level * 30, arr.shape)
            arr = np.clip(arr + grain, 0, 255)
            transforms_applied.append(f"grain:{visual['grain']}%")

    # Convert back to PIL
    img = Image.fromarray(arr.astype(np.uint8))

    # Vignette (applied as overlay)
    if tonal.get('vignette'):
        vignette_level = tonal['vignette'] / 100
        if vignette_level > 0:
            img = apply_vignette(img, vignette_level)
            transforms_applied.append(f"vignette:{tonal['vignette']}%")

    # Chromatic aberration
    if visual.get('chromatic'):
        chromatic = visual['chromatic']
        if chromatic > 0:
            img = apply_chromatic_aberration(img, chromatic)
            transforms_applied.append(f"chromatic:{chromatic}px")

    report_progress(0.8, "Encoding output...")

    # Get output quality
    quality = visual.get('quality', 92)
    if quality < 70:
        quality = 70
    if quality > 100:
        quality = 100

    # Determine output format
    output_ext = os.path.splitext(output_path)[1].lower()

    if output_ext in ['.jpg', '.jpeg']:
        img.save(output_path, 'JPEG', quality=quality, optimize=True)
    elif output_ext == '.png':
        img.save(output_path, 'PNG', optimize=True)
    elif output_ext == '.webp':
        img.save(output_path, 'WEBP', quality=quality)
    else:
        # Default to JPEG
        output_path = os.path.splitext(output_path)[0] + '.jpg'
        img.save(output_path, 'JPEG', quality=quality, optimize=True)

    report_progress(1.0, "Complete")

    return {
        'transforms_applied': transforms_applied,
        'original_size': original_size,
        'output_size': img.size,
        'quality': quality
    }


def process_video(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None]
) -> Dict[str, Any]:
    """Process video with transformations using FFmpeg + NVENC."""

    report_progress(0.1, "Analyzing video...")

    # Extract config
    spatial = config.get('spatial', {})
    tonal = config.get('tonal', {})
    visual = config.get('visual', {})
    video_config = config.get('video', {})

    transforms_applied = []
    filters = []

    # Get video info
    probe_cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-show_format', input_path
    ]

    try:
        import json
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)

        video_stream = next(
            (s for s in video_info.get('streams', []) if s.get('codec_type') == 'video'),
            {}
        )
        original_width = int(video_stream.get('width', 1920))
        original_height = int(video_stream.get('height', 1080))
        original_fps = eval(video_stream.get('r_frame_rate', '30/1'))
        duration = float(video_info.get('format', {}).get('duration', 0))
    except Exception:
        original_width = 1920
        original_height = 1080
        original_fps = 30
        duration = 0

    report_progress(0.2, "Building transform pipeline...")

    # === SPATIAL FILTERS ===

    # Crop
    if spatial.get('crop'):
        crop_percent = spatial['crop'] / 100
        if crop_percent > 0:
            crop_w = int(original_width * (1 - crop_percent * 2))
            crop_h = int(original_height * (1 - crop_percent * 2))
            crop_x = int(original_width * crop_percent)
            crop_y = int(original_height * crop_percent)
            filters.append(f"crop={crop_w}:{crop_h}:{crop_x}:{crop_y}")
            transforms_applied.append(f"crop:{spatial['crop']}%")

    # Micro resize
    if spatial.get('microResize'):
        resize_percent = spatial['microResize'] / 100
        if resize_percent > 0:
            scale = 1 + (random.random() - 0.5) * resize_percent * 2
            new_w = int(original_width * scale)
            new_h = int(original_height * scale)
            # Ensure even dimensions for encoding
            new_w = new_w - (new_w % 2)
            new_h = new_h - (new_h % 2)
            filters.append(f"scale={new_w}:{new_h}")
            transforms_applied.append(f"microResize:{spatial['microResize']}%")

    # Rotation
    if spatial.get('rotation'):
        rotation_deg = spatial['rotation']
        if rotation_deg > 0:
            angle = (random.random() - 0.5) * rotation_deg * 2
            angle_rad = angle * math.pi / 180
            filters.append(f"rotate={angle_rad}:fillcolor=black")
            transforms_applied.append(f"rotation:{rotation_deg}°")

    report_progress(0.3, "Adding tonal filters...")

    # === TONAL FILTERS ===

    eq_params = []

    # Brightness
    if tonal.get('brightness'):
        brightness_var = tonal['brightness'] / 100
        if brightness_var > 0:
            brightness = (random.random() - 0.5) * brightness_var * 0.4
            eq_params.append(f"brightness={brightness:.4f}")
            transforms_applied.append(f"brightness:{tonal['brightness']}%")

    # Gamma
    if tonal.get('gamma'):
        gamma_var = tonal['gamma']
        if gamma_var > 0:
            gamma = 1 + (random.random() - 0.5) * gamma_var * 2
            gamma = max(0.5, min(2.0, gamma))
            eq_params.append(f"gamma={gamma:.4f}")
            transforms_applied.append(f"gamma:{gamma_var}")

    # Contrast
    if tonal.get('contrast'):
        contrast_var = tonal['contrast'] / 100
        if contrast_var > 0:
            contrast = 1 + (random.random() - 0.5) * contrast_var
            eq_params.append(f"contrast={contrast:.4f}")
            transforms_applied.append(f"contrast:{tonal['contrast']}%")

    # Saturation
    if tonal.get('saturation'):
        sat_var = tonal['saturation'] / 100
        if sat_var > 0:
            saturation = 1 + (random.random() - 0.5) * sat_var
            eq_params.append(f"saturation={saturation:.4f}")
            transforms_applied.append(f"saturation:{tonal['saturation']}%")

    if eq_params:
        filters.append(f"eq={':'.join(eq_params)}")

    report_progress(0.4, "Adding visual effects...")

    # === VISUAL FILTERS ===

    # Noise
    if visual.get('noise'):
        noise_level = visual['noise']
        if noise_level > 0:
            # FFmpeg noise strength 0-100
            filters.append(f"noise=alls={noise_level * 2}:allf=t")
            transforms_applied.append(f"noise:{noise_level}")

    # Vignette
    if tonal.get('vignette'):
        vignette_level = tonal['vignette'] / 100
        if vignette_level > 0:
            filters.append(f"vignette=PI/{3 + (1 - vignette_level) * 4}")
            transforms_applied.append(f"vignette:{tonal['vignette']}%")

    # Color tint / hue shift
    if visual.get('tint'):
        tint_level = visual['tint']
        if tint_level > 0:
            hue_shift = random.uniform(-tint_level * 3, tint_level * 3)
            filters.append(f"hue=h={hue_shift}")
            transforms_applied.append(f"tint:{tint_level}")

    report_progress(0.5, "Configuring encoder...")

    # === VIDEO-SPECIFIC ===

    # Speed variation
    speed_factor = 1.0
    if video_config.get('speedVariation'):
        speed_var = video_config['speedVariation'] / 100
        if speed_var > 0:
            speed_factor = 1 + (random.random() - 0.5) * speed_var * 2
            speed_factor = max(0.9, min(1.1, speed_factor))
            filters.append(f"setpts={1/speed_factor}*PTS")
            transforms_applied.append(f"speedVariation:{video_config['speedVariation']}%")

    # Bitrate
    bitrate_percent = video_config.get('bitrate', 90) / 100
    target_bitrate = int(5000 * bitrate_percent)  # Base 5Mbps

    # Build filter chain
    filter_str = ','.join(filters) if filters else None

    report_progress(0.6, "Encoding with NVENC...")

    # Build FFmpeg command with NVENC
    cmd = ['ffmpeg', '-y', '-hwaccel', 'cuda', '-i', input_path]

    if filter_str:
        cmd.extend(['-vf', filter_str])

    # NVENC encoding settings
    cmd.extend([
        '-c:v', 'h264_nvenc',  # NVIDIA hardware encoder
        '-preset', 'p4',       # Quality/speed balance
        '-b:v', f'{target_bitrate}k',
        '-maxrate', f'{int(target_bitrate * 1.5)}k',
        '-bufsize', f'{target_bitrate * 2}k',
        '-profile:v', 'high',
        '-rc', 'vbr',
        '-rc-lookahead', '32',
    ])

    # Audio handling
    if video_config.get('audioShift'):
        audio_shift = video_config['audioShift'] / 1000  # Convert ms to seconds
        if audio_shift != 0:
            cmd.extend(['-af', f'adelay={int(audio_shift * 1000)}|{int(audio_shift * 1000)}'])
            transforms_applied.append(f"audioShift:{video_config['audioShift']}ms")

    cmd.extend(['-c:a', 'aac', '-b:a', '128k'])

    # Speed adjustment for audio if needed
    if speed_factor != 1.0:
        cmd.extend(['-af', f'atempo={speed_factor}'])

    cmd.append(output_path)

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

        # Parse FFmpeg progress
        if 'time=' in line:
            try:
                time_str = line.split('time=')[1].split()[0]
                h, m, s = time_str.split(':')
                current_time = int(h) * 3600 + int(m) * 60 + float(s)
                if duration > 0:
                    progress = min(0.6 + (current_time / duration) * 0.35, 0.95)
                    report_progress(progress, f"Encoding... {int(current_time)}s / {int(duration)}s")
            except:
                pass

    # Check result
    if process.returncode != 0:
        stderr = process.stderr.read()
        raise RuntimeError(f"FFmpeg failed: {stderr}")

    report_progress(1.0, "Complete")

    return {
        'transforms_applied': transforms_applied,
        'original_resolution': f"{original_width}x{original_height}",
        'duration': duration,
        'bitrate': f"{target_bitrate}kbps"
    }


# === HELPER FUNCTIONS ===

def apply_vignette(img: Image.Image, strength: float) -> Image.Image:
    """Apply vignette effect to image."""
    w, h = img.size

    # Create radial gradient mask
    y, x = np.ogrid[:h, :w]
    center_x, center_y = w / 2, h / 2

    # Calculate distance from center (normalized)
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    dist = dist / max_dist

    # Create vignette mask
    vignette = 1 - (dist ** 2) * strength
    vignette = np.clip(vignette, 0, 1)

    # Apply to image
    arr = np.array(img, dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = arr[:, :, c] * vignette

    return Image.fromarray(arr.astype(np.uint8))


def apply_chromatic_aberration(img: Image.Image, strength: float) -> Image.Image:
    """Apply chromatic aberration effect."""
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Shift red channel slightly
    shift = int(strength)
    if shift > 0:
        # Shift red channel right
        new_arr = np.zeros_like(arr)
        new_arr[:, shift:, 0] = arr[:, :-shift, 0]  # Red shifted right
        new_arr[:, :, 1] = arr[:, :, 1]              # Green unchanged
        new_arr[:, :-shift, 2] = arr[:, shift:, 2]  # Blue shifted left

        # Blend with original
        blend = 0.5
        arr = (arr * (1 - blend) + new_arr * blend).astype(np.uint8)

    return Image.fromarray(arr)


def apply_barrel_distortion(img: Image.Image, strength: float) -> Image.Image:
    """Apply barrel distortion effect."""
    arr = np.array(img)
    h, w = arr.shape[:2]

    # Create coordinate maps
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)

    # Normalize to -1 to 1
    x_norm = (x - w / 2) / (w / 2)
    y_norm = (y - h / 2) / (h / 2)

    # Calculate distance from center
    r = np.sqrt(x_norm ** 2 + y_norm ** 2)

    # Apply barrel distortion
    distortion = 1 + strength * r ** 2

    # New coordinates
    x_dist = x_norm * distortion
    y_dist = y_norm * distortion

    # Convert back to pixel coordinates
    x_new = ((x_dist * w / 2) + w / 2).astype(np.int32)
    y_new = ((y_dist * h / 2) + h / 2).astype(np.int32)

    # Clip to valid range
    x_new = np.clip(x_new, 0, w - 1)
    y_new = np.clip(y_new, 0, h - 1)

    # Create output
    output = arr[y_new, x_new]

    return Image.fromarray(output)


def apply_perspective_warp(img: Image.Image, strength: float) -> Image.Image:
    """Apply perspective warp transformation."""
    w, h = img.size

    # Random corner offsets
    offset = int(min(w, h) * strength * 0.1)

    # Source corners (original)
    src_corners = [
        (0, 0),
        (w, 0),
        (w, h),
        (0, h)
    ]

    # Destination corners (warped)
    dst_corners = [
        (random.randint(0, offset), random.randint(0, offset)),
        (w - random.randint(0, offset), random.randint(0, offset)),
        (w - random.randint(0, offset), h - random.randint(0, offset)),
        (random.randint(0, offset), h - random.randint(0, offset))
    ]

    # Compute transformation coefficients
    coeffs = find_perspective_coeffs(dst_corners, src_corners)

    # Apply transformation
    return img.transform((w, h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)


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
    # Test mode
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor.py input_file output_file")
        sys.exit(1)

    test_config = {
        'spatial': {'crop': 2, 'microResize': 1, 'rotation': 0.3},
        'tonal': {'brightness': 3, 'contrast': 3, 'saturation': 5},
        'visual': {'noise': 3, 'quality': 92}
    }

    def progress(p, msg):
        print(f"[{int(p*100)}%] {msg}")

    result = process_spoofer(sys.argv[1], sys.argv[2], test_config, progress)
    print(f"Result: {result}")
