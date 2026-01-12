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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import subprocess
from collections import deque
import multiprocessing

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

# NVENC session limits by GPU type
# Consumer GPUs (GeForce): Limited to 3-5 sessions
# Datacenter GPUs (A-series, Quadro): Unlimited sessions
NVENC_SESSION_LIMITS = {
    'consumer': 3,      # RTX 3090, 4090, 4080, etc.
    'datacenter': 10,   # A5000, A6000, A100, etc. (conservative limit)
    'default': 2,       # Fallback
}


def get_gpu_info() -> Dict[str, Any]:
    """
    Detect GPU type and determine NVENC session limit.
    Returns dict with gpu_name, gpu_type ('consumer' or 'datacenter'), and nvenc_sessions.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_name = result.stdout.strip().split('\n')[0]

            # Determine GPU type
            datacenter_keywords = ['A100', 'A6000', 'A5000', 'A4000', 'A4500', 'A40', 'A30', 'A10',
                                   'V100', 'T4', 'Quadro', 'Tesla', 'H100', 'L40']
            is_datacenter = any(kw in gpu_name for kw in datacenter_keywords)

            gpu_type = 'datacenter' if is_datacenter else 'consumer'
            nvenc_sessions = NVENC_SESSION_LIMITS[gpu_type]

            return {
                'gpu_name': gpu_name,
                'gpu_type': gpu_type,
                'nvenc_sessions': nvenc_sessions
            }
    except Exception as e:
        print(f"[GPU Detection] Error: {e}")

    return {
        'gpu_name': 'Unknown',
        'gpu_type': 'default',
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


def process_single_video_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel video processing.
    Designed to be called from ProcessPoolExecutor.

    Args tuple: (input_path, output_path, config, video_index)
    Returns: dict with status, output_path, and any errors
    """
    input_path, output_path, config, video_index = args

    try:
        # Import here to avoid pickling issues
        import random
        import math
        import subprocess
        import json

        py_rng = random.Random(int(time.time() * 1000) + video_index)

        # Get video info
        try:
            probe_cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                        '-show_streams', '-show_format', input_path]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            video_info = json.loads(probe_result.stdout)
            video_stream = next((s for s in video_info.get('streams', [])
                               if s.get('codec_type') == 'video'), {})
            original_width = int(video_stream.get('width', 1920))
            original_height = int(video_stream.get('height', 1080))
            duration = float(video_info.get('format', {}).get('duration', 0))
        except Exception:
            original_width, original_height, duration = 1920, 1080, 0

        # Build filter chain (same as process_video)
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
            angle = py_rng.uniform(-spatial['rotation'], spatial['rotation']) * math.pi / 180
            filters.append(f"rotate={angle}:fillcolor=black")

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

        # Build FFmpeg command
        filter_str = ','.join(filters) if filters else None
        cmd = ['ffmpeg', '-y', '-i', input_path]

        if filter_str:
            cmd.extend(['-vf', filter_str])

        # NVENC settings
        bitrate = int(5000 * video_cfg.get('bitrate', 90) / 100)
        cmd.extend([
            '-c:v', 'h264_nvenc',
            '-preset', 'p2',
            '-rc', 'vbr',
            '-cq', '23',
            '-b:v', f'{bitrate}k',
            '-maxrate', f'{int(bitrate * 1.5)}k',
            '-bufsize', f'{bitrate * 2}k',
            '-rc-lookahead', '8',
        ])

        # Audio
        keep_audio = video_cfg.get('keepAudio', True)
        if keep_audio:
            cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
        else:
            cmd.append('-an')

        cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])

        # Run FFmpeg
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if process.returncode != 0:
            return {
                'status': 'failed',
                'index': video_index,
                'error': process.stderr[-500:] if process.stderr else 'Unknown error'
            }

        return {
            'status': 'completed',
            'index': video_index,
            'output_path': output_path,
            'duration': duration
        }

    except Exception as e:
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
    max_parallel: int = None
) -> Dict[str, Any]:
    """
    Process multiple videos in parallel using multiple NVENC sessions.

    Args:
        video_paths: List of input video paths
        output_dir: Directory for output videos
        config: Processing configuration
        progress_callback: Optional progress callback
        max_parallel: Max parallel processes (auto-detected if None)

    Returns:
        Dict with results including processed count and any errors
    """
    if not video_paths:
        return {'error': 'No videos to process'}

    # Auto-detect parallel limit based on GPU
    if max_parallel is None:
        gpu_info = get_gpu_info()
        max_parallel = gpu_info['nvenc_sessions']
        print(f"[Parallel Video] GPU: {gpu_info['gpu_name']}, Type: {gpu_info['gpu_type']}, "
              f"Max parallel: {max_parallel}")

    # Prepare work items
    work_items = []
    for i, video_path in enumerate(video_paths):
        basename = os.path.basename(video_path)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name}_spoofed{ext}")
        work_items.append((video_path, output_path, config, i))

    total = len(work_items)
    completed = 0
    failed = 0
    results = []

    def report_progress(msg=""):
        if progress_callback:
            progress = completed / total
            progress_callback(progress, msg)

    report_progress(f"Processing {total} videos with {max_parallel} parallel sessions...")

    # Process in parallel using ProcessPoolExecutor
    # Use 'spawn' context to avoid CUDA issues with fork
    with ProcessPoolExecutor(max_workers=max_parallel,
                            mp_context=multiprocessing.get_context('spawn')) as executor:
        future_to_item = {executor.submit(process_single_video_worker, item): item
                        for item in work_items}

        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)

            if result['status'] == 'completed':
                completed += 1
            else:
                failed += 1
                print(f"[Parallel Video] Failed video {result['index']}: {result.get('error', 'unknown')}")

            report_progress(f"Completed {completed}/{total} videos ({failed} failed)")

    return {
        'status': 'completed',
        'total': total,
        'completed': completed,
        'failed': failed,
        'results': results,
        'parallel_sessions': max_parallel
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
    """Micro rotation - very small rotation angles."""
    if strength <= 0:
        return img

    angle = py_rng.uniform(-strength, strength)
    return img.rotate(angle, expand=False, fillcolor=(0, 0, 0), resample=Image.BICUBIC)


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
                # PARALLEL VIDEO PROCESSING MODE
                report_progress(0.1, f"Found {len(video_paths)} videos, starting parallel processing...")

                # Create output directory
                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(output_dir, exist_ok=True)

                # Process videos in parallel
                result = process_videos_parallel(
                    video_paths,
                    output_dir,
                    config,
                    progress_callback=progress_callback
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
                    'videos_processed': result.get('completed', 0),
                    'videos_failed': result.get('failed', 0),
                    'parallel_sessions': result.get('parallel_sessions', 1),
                    'output_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
                }
            else:
                # ZIP contains images, not videos - fall through to batch image mode
                # Extract all images and process them
                report_progress(0.1, "ZIP contains images, processing as batch...")

        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            # Cleanup temp dir if we processed videos
            if 'video_paths' in dir() and video_paths:
                shutil.rmtree(temp_dir, ignore_errors=True)

    if is_video:
        return process_video(input_path, output_path, config, report_progress)
    else:
        # Check if batch mode
        # Check both config.copies (from workflows) and config.options.copies (from tool view)
        copies = config.get('copies') or config.get('options', {}).get('copies', 1)
        if copies > 1:
            return process_batch_spoofer(input_path, output_path, config, copies, progress_callback)
        else:
            return process_single_image(input_path, output_path, config, report_progress)


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
        angle = py_rng.uniform(-spatial['rotation'], spatial['rotation']) * math.pi / 180
        filters.append(f"rotate={angle}:fillcolor=black")

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

    # Build FFmpeg command: CPU decode + filters + NVENC encode
    # This is the most reliable approach - filters work on CPU frames, NVENC handles encoding
    report_progress(0.2, "Encoding with GPU (NVENC)...")

    cmd = ['ffmpeg', '-y', '-i', input_path]

    # Apply filters (CPU-based, most compatible)
    if filter_str:
        cmd.extend(['-vf', filter_str])

    # NVENC encoding (GPU-accelerated)
    cmd.extend([
        '-c:v', 'h264_nvenc',
        '-preset', 'p2',
        '-rc', 'vbr',
        '-cq', '23',
        '-b:v', f'{bitrate}k',
        '-maxrate', f'{int(bitrate * 1.5)}k',
        '-bufsize', f'{bitrate * 2}k',
        '-rc-lookahead', '8',
    ])

    # Audio
    if keep_audio:
        cmd.extend(['-c:a', 'aac', '-b:a', '128k'])
    else:
        cmd.append('-an')

    cmd.extend(['-r', f'{fps:.1f}', '-movflags', '+faststart', output_path])

    success, stderr_output, returncode = run_ffmpeg(cmd, "NVENC")

    if not success:
        raise RuntimeError(f"FFmpeg failed (code {returncode}): {stderr_output}")

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
