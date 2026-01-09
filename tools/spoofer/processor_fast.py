"""
Spoofer Processor - FAST MODE (CPU Optimized)

Key optimizations:
1. CPU Multiprocessing - Use all available cores
2. pHash disabled by default (40-50% speedup)
3. Single compression with artifacts (25-35% speedup)
4. Skip transforms with 0 values
5. Streamlined ZIP writing

Target: 200 photos in 20-30 seconds (vs 15+ minutes)
"""

import os
import io
import random
import time
import hashlib
import zipfile
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from typing import Callable, Optional, Dict, Any, List, Tuple
from multiprocessing import Pool, cpu_count
import functools

# Constants
TARGET_RESOLUTIONS = {
    'high': (1080, 1920),
    'low': (720, 1280)
}

# Fast mode defaults - skip expensive transforms
FAST_DEFAULTS = {
    'crop': 1.5,
    'micro_resize': 1.2,
    'rotation': 0.8,
    'subpixel': 0,      # Skip - slow
    'warp': 0,          # Skip - slow (linalg.solve)
    'barrel': 0,        # Skip - slow (numpy meshgrid)
    'block_shift': 0,   # Skip - slow (loops)
    'scale': 100,
    'micro_rescale': 0, # Skip - double resize
    'brightness': 0.04,
    'gamma': 0.06,
    'contrast': 0.04,
    'vignette': 0,      # Skip - slow (numpy)
    'saturation': 0.06,
    'tint': 1.5,
    'chromatic': 0,     # Skip - slow
    'noise': 3.0,
    'quality': 90,
    'double_compress': 0,  # Disable - use single
    'flip': 1,
    'force_916': 1,
}


def generate_seed(img_path: str, idx: int, base_time: int) -> int:
    """Generate unique seed for reproducible randomization."""
    hash_input = f"{img_path}_{idx}_{base_time}_{os.getpid()}"
    return int(hashlib.md5(hash_input.encode()).hexdigest()[:8], 16)


def randomize_params(base: Dict, rng: random.Random, var: float = 0.3) -> Dict:
    """Randomize parameters within ±variation."""
    result = {}
    for key, val in base.items():
        if isinstance(val, (int, float)) and val > 0:
            factor = 1.0 + rng.uniform(-var, var)
            result[key] = int(val * factor) if isinstance(val, int) else val * factor
        else:
            result[key] = val
    return result


# ═══════════════════════════════════════════════════════════════
# FAST TRANSFORMS (optimized for speed)
# ═══════════════════════════════════════════════════════════════

def fast_crop(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Fast asymmetric crop."""
    if strength <= 0:
        return img
    w, h = img.size
    m = strength / 100.0
    l, r = int(w * rng.uniform(0, m)), int(w * rng.uniform(0, m))
    t, b = int(h * rng.uniform(0, m)), int(h * rng.uniform(0, m))
    if l + r >= w * 0.4 or t + b >= h * 0.4:
        return img
    return img.crop((l, t, w - r, h - b))


def fast_resize(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Fast micro resize."""
    if strength <= 0:
        return img
    w, h = img.size
    s = 1.0 + rng.uniform(-strength/100, strength/100)
    return img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)


def fast_rotate(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Fast micro rotation."""
    if strength <= 0:
        return img
    angle = rng.uniform(-strength, strength)
    return img.rotate(angle, expand=False, fillcolor=(0, 0, 0), resample=Image.BILINEAR)


def fast_brightness(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Fast brightness adjustment."""
    if strength <= 0:
        return img
    factor = 1.0 + rng.uniform(-strength, strength)
    return ImageEnhance.Brightness(img).enhance(factor)


def fast_contrast(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Fast contrast adjustment."""
    if strength <= 0:
        return img
    factor = max(0.5, min(1.5, 1.0 + rng.uniform(-strength, strength)))
    return ImageEnhance.Contrast(img).enhance(factor)


def fast_saturation(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Fast saturation adjustment."""
    if strength <= 0:
        return img
    factor = max(0.5, min(1.5, 1.0 + rng.uniform(-strength, strength)))
    return ImageEnhance.Color(img).enhance(factor)


def fast_gamma(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Fast gamma correction using LUT."""
    if strength <= 0:
        return img
    gamma = max(0.5, min(2.0, 1.0 + rng.uniform(-strength, strength)))
    # Use lookup table for speed
    lut = [int(255 * ((i / 255) ** gamma)) for i in range(256)]
    return img.point(lut * 3)  # Apply to RGB


def fast_tint(img: Image.Image, strength: float, rng: random.Random) -> Image.Image:
    """Fast color tint using channels."""
    if strength <= 0:
        return img
    r, g, b = img.split()
    tr = int(rng.uniform(-strength, strength))
    tg = int(rng.uniform(-strength, strength))
    tb = int(rng.uniform(-strength, strength))
    r = r.point(lambda x: max(0, min(255, x + tr)))
    g = g.point(lambda x: max(0, min(255, x + tg)))
    b = b.point(lambda x: max(0, min(255, x + tb)))
    return Image.merge('RGB', (r, g, b))


def fast_noise(img: Image.Image, strength: float, rng_seed: int) -> Image.Image:
    """Fast noise addition."""
    if strength <= 0:
        return img
    np_rng = np.random.default_rng(rng_seed)
    arr = np.array(img, dtype=np.float32)
    noise = np_rng.normal(0, strength * 2, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def fast_single_compress(img: Image.Image, quality: int, rng: random.Random) -> Image.Image:
    """Single JPEG compression with slight quality variation."""
    buffer = io.BytesIO()
    q = quality + rng.randint(-5, 5)
    q = max(60, min(95, q))
    img.save(buffer, 'JPEG', quality=q)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')


def fast_pad_916(img: Image.Image, rng: random.Random) -> Image.Image:
    """Fast 9:16 padding with blurred background."""
    w, h = img.size
    target_ratio = 9 / 16

    if h > 0 and abs((w / h) - target_ratio) < 0.02:
        return img

    if w <= 0 or h <= 0:
        return img

    base = max(w, h)
    target_w, target_h = TARGET_RESOLUTIONS['high'] if base >= 1000 else TARGET_RESOLUTIONS['low']

    # Create blurred background
    bg = img.copy()
    scale = max(target_w / w, target_h / h) * 1.3
    bw, bh = int(w * scale), int(h * scale)
    bg = bg.resize((bw, bh), Image.BILINEAR)

    # Center crop for background
    bg_w, bg_h = bg.size
    left = max(0, (bg_w - target_w) // 2)
    top = max(0, (bg_h - target_h) // 2)
    bg = bg.crop((left, top, left + target_w, top + target_h))

    if bg.size != (target_w, target_h):
        bg = bg.resize((target_w, target_h), Image.BILINEAR)

    # Blur background (fast with small radius)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=25))
    bg = ImageEnhance.Brightness(bg).enhance(0.65)

    canvas = bg.convert("RGBA")

    # Foreground
    fg = img.copy().convert("RGBA")
    fw, fh = fg.size
    scale_fg = min(target_w / fw, target_h / fh) * 0.9
    fg = fg.resize((int(fw * scale_fg), int(fh * scale_fg)), Image.BILINEAR)

    # Center
    x = (target_w - fg.size[0]) // 2
    y = (target_h - fg.size[1]) // 2

    canvas.paste(fg, (x, y), fg)
    return canvas.convert("RGB")


def apply_fast_transforms(img: Image.Image, params: Dict, seed: int) -> Image.Image:
    """Apply all transforms with fast methods."""
    rng = random.Random(seed)

    # Flip
    if params.get('flip', 0) and rng.random() > 0.5:
        img = ImageOps.mirror(img)

    # Fast spatial
    img = fast_crop(img, params.get('crop', 0), rng)
    img = fast_resize(img, params.get('micro_resize', 0), rng)
    img = fast_rotate(img, params.get('rotation', 0), rng)

    # Fast tonal
    img = fast_brightness(img, params.get('brightness', 0), rng)
    img = fast_gamma(img, params.get('gamma', 0), rng)
    img = fast_contrast(img, params.get('contrast', 0), rng)
    img = fast_saturation(img, params.get('saturation', 0), rng)

    # Fast visual
    img = fast_tint(img, params.get('tint', 0), rng)
    img = fast_noise(img, params.get('noise', 0), seed)

    # 9:16 padding
    if params.get('force_916', 0):
        img = fast_pad_916(img, rng)

    # Single compression
    img = fast_single_compress(img, int(params.get('quality', 90)), rng)

    return img


# ═══════════════════════════════════════════════════════════════
# WORKER FUNCTION FOR MULTIPROCESSING
# ═══════════════════════════════════════════════════════════════

def process_single_copy_worker(args):
    """
    Worker function for multiprocessing.
    Args is a tuple: (img_bytes, params, seed, idx, base_name, quality, random_names)
    Returns: (idx, filename, jpg_bytes)
    """
    img_bytes, params, seed, idx, base_name, quality, random_names = args

    try:
        # Load image from bytes
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Randomize params for this copy
        rng = random.Random(seed)
        varied_params = randomize_params(params, rng, var=0.3)

        # Apply transforms
        result_img = apply_fast_transforms(img, varied_params, seed)

        # Generate filename
        if random_names:
            filename = ''.join(rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=12)) + '.jpg'
        else:
            rand_id = rng.randint(1000, 9999)
            clean_name = "".join([c for c in base_name if c.isalnum() or c in (' ', '-', '_')]).strip()[:20]
            filename = f"{clean_name}_v{idx}_{rand_id}.jpg"

        # Save to bytes
        buffer = io.BytesIO()
        result_img.save(buffer, 'JPEG', quality=quality, optimize=True)
        jpg_bytes = buffer.getvalue()

        return (idx, filename, jpg_bytes)

    except Exception as e:
        print(f"[Worker Error] Copy {idx}: {str(e)}")
        return (idx, None, None)


def process_batch_fast(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    copies: int,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    FAST batch processing with multiprocessing.

    Args:
        input_path: Path to input image
        output_path: Path for output ZIP
        config: Transform configuration
        copies: Number of copies
        progress_callback: Optional progress callback

    Returns:
        Dict with results
    """
    def report(p: float, msg: str = ""):
        if progress_callback:
            progress_callback(p, msg)

    start_time = time.time()
    report(0.05, "Loading image...")

    # Load and prep image
    original_img = Image.open(input_path)
    original_img = ImageOps.exif_transpose(original_img)
    if original_img.mode != 'RGB':
        original_img = original_img.convert('RGB')

    # Convert to bytes for multiprocessing
    img_buffer = io.BytesIO()
    original_img.save(img_buffer, 'PNG')
    img_bytes = img_buffer.getvalue()

    report(0.1, f"Preparing {copies} copies with {cpu_count()} CPU cores...")

    # Build params (use fast defaults for unspecified)
    params = dict(FAST_DEFAULTS)

    spatial = config.get('spatial', {})
    params['crop'] = spatial.get('crop', params['crop'])
    params['micro_resize'] = spatial.get('microResize', params['micro_resize'])
    params['rotation'] = spatial.get('rotation', params['rotation'])

    tonal = config.get('tonal', {})
    params['brightness'] = tonal.get('brightness', params['brightness'])
    params['gamma'] = tonal.get('gamma', params['gamma'])
    params['contrast'] = tonal.get('contrast', params['contrast'])
    params['saturation'] = tonal.get('saturation', params['saturation'])

    visual = config.get('visual', {})
    params['tint'] = visual.get('tint', params['tint'])
    params['noise'] = visual.get('noise', params['noise'])

    compression = config.get('compression', {})
    quality = int(compression.get('quality', 90))

    options = config.get('options', {})
    params['flip'] = options.get('flip', 1)
    params['force_916'] = options.get('force916', 1)
    random_names = options.get('randomNames', 0)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    base_time = time.time_ns()

    # Prepare worker args
    worker_args = []
    for i in range(copies):
        seed = generate_seed(input_path, i, base_time + i * 1000)
        worker_args.append((img_bytes, params, seed, i, base_name, quality, random_names))

    report(0.15, f"Processing with {cpu_count()} CPU cores...")

    # Use multiprocessing pool
    num_workers = min(cpu_count(), copies, 8)  # Max 8 workers
    results = []

    with Pool(processes=num_workers) as pool:
        # Process in chunks for progress updates
        chunk_size = max(1, copies // 10)

        for chunk_start in range(0, copies, chunk_size):
            chunk_end = min(chunk_start + chunk_size, copies)
            chunk_args = worker_args[chunk_start:chunk_end]

            chunk_results = pool.map(process_single_copy_worker, chunk_args)
            results.extend(chunk_results)

            # Progress
            progress = 0.15 + (chunk_end / copies) * 0.75
            report(progress, f"Processed {chunk_end}/{copies} copies...")

    report(0.9, "Writing ZIP file...")

    # Write results to ZIP
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        filenames = []
        for idx, filename, jpg_bytes in results:
            if filename and jpg_bytes:
                zf.writestr(filename, jpg_bytes)
                filenames.append(filename)

    elapsed = time.time() - start_time
    report(1.0, f"Complete in {elapsed:.1f}s")

    return {
        'copies_generated': len(filenames),
        'files': filenames,
        'processing_time': elapsed,
        'time_per_copy': elapsed / max(1, len(filenames)),
        'cpu_cores_used': num_workers
    }


def process_spoofer_fast(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Main entry point for fast spoofer processing.
    """
    # Check both config.copies (from workflows) and config.options.copies (from tool view)
    copies = config.get('copies') or config.get('options', {}).get('copies', 1)

    if copies > 1:
        return process_batch_fast(input_path, output_path, config, copies, progress_callback)
    else:
        # Single image - use regular fast transforms
        def report(p, m=""):
            if progress_callback:
                progress_callback(p, m)

        report(0.1, "Loading image...")
        img = Image.open(input_path)
        img = ImageOps.exif_transpose(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        report(0.3, "Applying transforms...")
        seed = int(time.time() * 1000) % (2**31)

        # Build params
        params = dict(FAST_DEFAULTS)
        spatial = config.get('spatial', {})
        tonal = config.get('tonal', {})
        visual = config.get('visual', {})
        compression = config.get('compression', {})
        options = config.get('options', {})

        params.update({
            'crop': spatial.get('crop', params['crop']),
            'micro_resize': spatial.get('microResize', params['micro_resize']),
            'rotation': spatial.get('rotation', params['rotation']),
            'brightness': tonal.get('brightness', params['brightness']),
            'gamma': tonal.get('gamma', params['gamma']),
            'contrast': tonal.get('contrast', params['contrast']),
            'saturation': tonal.get('saturation', params['saturation']),
            'tint': visual.get('tint', params['tint']),
            'noise': visual.get('noise', params['noise']),
            'quality': compression.get('quality', 90),
            'flip': options.get('flip', 1),
            'force_916': options.get('force916', 1),
        })

        result = apply_fast_transforms(img, params, seed)

        report(0.8, "Saving...")
        quality = int(params.get('quality', 90))
        result.save(output_path, 'JPEG', quality=quality, optimize=True)

        report(1.0, "Complete")
        return {'output_size': result.size, 'quality': quality}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor_fast.py input_file output_file [copies]")
        sys.exit(1)

    copies = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    test_config = {
        'spatial': {'crop': 1.5, 'microResize': 1.2, 'rotation': 0.8},
        'tonal': {'brightness': 0.04, 'contrast': 0.04, 'saturation': 0.06},
        'visual': {'noise': 3, 'tint': 1.5},
        'compression': {'quality': 90},
        'options': {'copies': copies, 'force916': 1, 'flip': 1}
    }

    def progress(p, msg):
        print(f"[{int(p*100):3d}%] {msg}")

    result = process_spoofer_fast(sys.argv[1], sys.argv[2], test_config, progress)
    print(f"\nResult: {result}")
    print(f"Speed: {result.get('time_per_copy', 0):.3f}s per copy")
