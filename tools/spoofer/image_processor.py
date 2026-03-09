"""
Image Processor Module

Contains all image processing functions:
- apply_transforms: Master orchestrator for all image transformations
- process_single_copy: Process a single image copy with pHash verification
- process_single_image: Process a single image (not batch mode)
- process_single_image_worker: Worker function for parallel image processing
- process_images_parallel: Process multiple images in parallel
- process_batch_spoofer: Process image and generate multiple variations as ZIP
"""

import os
import io
import time
import random
import zipfile
import threading
import numpy as np
from PIL import Image, ImageOps
from typing import Optional, Dict, Any, List, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .constants import PHASH_MIN_DISTANCE, PHOTO_DEFAULTS
from .utils import generate_unique_seed, randomize_params, apply_mode_to_config
from .image_transforms import (
    apply_asymmetric_crop,
    apply_micro_resize,
    apply_micro_rotation,
    apply_subpixel_shift,
    apply_perspective_warp,
    apply_barrel_distortion,
    apply_block_shift,
    apply_micro_rescale_photo,
    apply_brightness_shift,
    apply_gamma_shift,
    apply_contrast_shift,
    apply_saturation_shift,
    apply_vignette,
    apply_frequency_noise,
    apply_invisible_watermark,
    apply_color_space_conversion,
    apply_color_tint,
    apply_chromatic_aberration,
    apply_noise,
    apply_double_compression,
    pad_to_9_16_photo,
    # I1-I15: New color/visual transforms
    apply_hue_shift,
    apply_color_balance,
    apply_color_temperature,
    apply_vibrance,
    apply_curves_preset,
    apply_color_channel_mixer,
    apply_levels_adjustment,
    apply_sharpness,
    apply_gradient_overlay,
    apply_scanline_pattern,
    apply_film_grain,
    apply_color_filter_preset,
    apply_dynamic_color_shift,
    apply_edge_enhancement,
    apply_dither_pattern,
    # I21-I30: Enhancement, quality, spatial advanced
    apply_denoise,
    apply_auto_levels,
    apply_affine_shear,
    apply_elastic_deformation,
    apply_border_inject,
    apply_image_post_processing,
)

# Optional pHash support
try:
    import imagehash
    PHASH_AVAILABLE = True
except ImportError:
    PHASH_AVAILABLE = False


def apply_transforms(img: Image.Image, params: Dict, py_rng: random.Random, rng: np.random.Generator) -> Image.Image:
    """Apply all image transformations based on params."""

    # TIER 1: SPATIAL (Maximum pHash impact)
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

    # TIER 2: TONAL (DCT coefficients)
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

    # TIER 3: VISUAL (Makes copies look different)
    if params.get('saturation', 0) > 0:
        img = apply_saturation_shift(img, params['saturation'], py_rng)

    if params.get('tint', 0) > 0:
        img = apply_color_tint(img, params['tint'], py_rng)

    if params.get('chromatic', 0) > 0:
        img = apply_chromatic_aberration(img, params['chromatic'], py_rng)

    if params.get('noise', 0) > 0:
        img = apply_noise(img, params['noise'], rng)

    # NEW TIER: COLOR TRANSFORMS (I1-I8)
    if params.get('hue_shift', 0) > 0:
        img = apply_hue_shift(img, params['hue_shift'], py_rng)

    if params.get('color_balance', 0) > 0:
        img = apply_color_balance(img, params['color_balance'], py_rng)

    if params.get('color_temperature', 0) > 0:
        img = apply_color_temperature(img, params['color_temperature'], py_rng)

    if params.get('vibrance', 0) > 0:
        img = apply_vibrance(img, params['vibrance'], py_rng)

    curves = params.get('curves_preset', 'none')
    if curves and curves != 'none':
        img = apply_curves_preset(img, curves, py_rng)

    if params.get('color_channel_mixer', 0) > 0:
        img = apply_color_channel_mixer(img, params['color_channel_mixer'], py_rng)

    if params.get('levels', 0) > 0:
        img = apply_levels_adjustment(img, params['levels'], py_rng)

    if params.get('sharpness', 0) > 0:
        img = apply_sharpness(img, params['sharpness'], py_rng)

    # NEW TIER: VISUAL EFFECTS (I9-I15)
    if params.get('gradient_overlay', 0) > 0:
        img = apply_gradient_overlay(img, params['gradient_overlay'], py_rng)

    if params.get('scanline', 0) > 0:
        img = apply_scanline_pattern(img, params['scanline'], py_rng)

    if params.get('film_grain', 0) > 0:
        img = apply_film_grain(img, params['film_grain'], py_rng, rng)

    color_filter = params.get('color_filter_preset', 'none')
    if color_filter and color_filter != 'none':
        img = apply_color_filter_preset(img, color_filter, py_rng)

    if params.get('dynamic_color_shift', 0) > 0:
        img = apply_dynamic_color_shift(img, params['dynamic_color_shift'], py_rng)

    if params.get('edge_enhance', 0) > 0:
        img = apply_edge_enhancement(img, params['edge_enhance'], py_rng)

    if params.get('dither', 0) > 0:
        img = apply_dither_pattern(img, params['dither'], py_rng, rng)

    # NEW TIER: ENHANCEMENT (I22-I23)
    if params.get('denoise', 0) > 0:
        img = apply_denoise(img, params['denoise'], py_rng)

    if params.get('auto_levels', 0) > 0:
        img = apply_auto_levels(img, params['auto_levels'], py_rng)

    # NEW TIER: SPATIAL ADVANCED (I27-I28, I30)
    if params.get('affine_shear', 0) > 0:
        img = apply_affine_shear(img, params['affine_shear'], py_rng)

    if params.get('elastic_deform', 0) > 0:
        img = apply_elastic_deformation(img, params['elastic_deform'], py_rng, rng)

    if params.get('border_inject', 0) > 0:
        img = apply_border_inject(img, params['border_inject'], py_rng)

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

    # Apply mode multipliers to config
    config = apply_mode_to_config(config)

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
        'quality': compression.get('quality', PHOTO_DEFAULTS['quality']),
        'double_compress': compression.get('doubleCompress', 0),
        'flip': options.get('flip', 0),
        'force_916': options.get('force916', 0),
        # I1-I8: New color transforms
        'hue_shift': tonal.get('hueShift', 0),
        'color_balance': tonal.get('colorBalance', 0),
        'color_temperature': tonal.get('colorTemperature', 0),
        'vibrance': tonal.get('vibrance', 0),
        'curves_preset': tonal.get('curvesPreset', 'none'),
        'color_channel_mixer': tonal.get('colorChannelMixer', 0),
        'levels': tonal.get('levels', 0),
        'sharpness': visual.get('sharpness', 0),
        # I9-I15: New visual effects
        'gradient_overlay': visual.get('gradientOverlay', 0),
        'scanline': visual.get('scanline', 0),
        'film_grain': visual.get('filmGrain', 0),
        'color_filter_preset': visual.get('colorFilterPreset', 'none'),
        'dynamic_color_shift': visual.get('dynamicColorShift', 0),
        'edge_enhance': visual.get('edgeEnhance', 0),
        'dither': visual.get('dither', 0),
        # I21-I25: Enhancement
        'denoise': visual.get('denoise', 0),
        'auto_levels': tonal.get('autoLevels', 0),
        # I26-I30: Spatial advanced
        'affine_shear': spatial.get('affineShear', 0),
        'elastic_deform': spatial.get('elasticDeform', 0),
        'border_inject': spatial.get('borderInject', 0),
        # I16-I20: Anti-detection
        'trace_removal': options.get('traceRemoval', 1),
        'exif_injection': options.get('exifInjection', 0),
        'device_profile': options.get('deviceProfile', 'none'),
        'gps_injection': options.get('gpsInjection', 0),
        'timestamp_injection': options.get('timestampInjection', 0),
        '_variation_mult': config.get('_variation_mult', 0.3),
    })

    report_progress(0.3, "Applying transforms...")

    seed = int(time.time() * 1000) % (2**31)
    py_rng = random.Random(seed)
    rng = np.random.default_rng(seed)

    result_img = apply_transforms(original_img.copy(), params, py_rng, rng)

    report_progress(0.8, "Saving...")

    quality = int(params.get('quality', PHOTO_DEFAULTS['quality']))
    output_ext = os.path.splitext(output_path)[1].lower()

    if output_ext in ['.jpg', '.jpeg']:
        result_img.save(output_path, 'JPEG', quality=quality, optimize=True)
    elif output_ext == '.png':
        result_img.save(output_path, 'PNG', optimize=True)
    else:
        output_path = os.path.splitext(output_path)[0] + '.jpg'
        result_img.save(output_path, 'JPEG', quality=quality, optimize=True)

    # I16-I20: Image anti-detection post-processing
    apply_image_post_processing(result_img, output_path, params, py_rng)

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

        # Apply transforms
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

        # I16-I20: Image anti-detection post-processing
        apply_image_post_processing(result_img, output_path, params, py_rng)

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
        # I1-I8: New color transforms
        'hue_shift': tonal.get('hueShift', PHOTO_DEFAULTS['hue_shift']),
        'color_balance': tonal.get('colorBalance', PHOTO_DEFAULTS['color_balance']),
        'color_temperature': tonal.get('colorTemperature', PHOTO_DEFAULTS['color_temperature']),
        'vibrance': tonal.get('vibrance', PHOTO_DEFAULTS['vibrance']),
        'curves_preset': tonal.get('curvesPreset', PHOTO_DEFAULTS['curves_preset']),
        'color_channel_mixer': tonal.get('colorChannelMixer', PHOTO_DEFAULTS['color_channel_mixer']),
        'levels': tonal.get('levels', PHOTO_DEFAULTS['levels']),
        'sharpness': visual.get('sharpness', PHOTO_DEFAULTS['sharpness']),
        # I9-I15: New visual effects
        'gradient_overlay': visual.get('gradientOverlay', PHOTO_DEFAULTS['gradient_overlay']),
        'scanline': visual.get('scanline', PHOTO_DEFAULTS['scanline']),
        'film_grain': visual.get('filmGrain', PHOTO_DEFAULTS['film_grain']),
        'color_filter_preset': visual.get('colorFilterPreset', PHOTO_DEFAULTS['color_filter_preset']),
        'dynamic_color_shift': visual.get('dynamicColorShift', PHOTO_DEFAULTS['dynamic_color_shift']),
        'edge_enhance': visual.get('edgeEnhance', PHOTO_DEFAULTS['edge_enhance']),
        'dither': visual.get('dither', PHOTO_DEFAULTS['dither']),
        # I21-I25: Enhancement
        'denoise': visual.get('denoise', PHOTO_DEFAULTS['denoise']),
        'auto_levels': tonal.get('autoLevels', PHOTO_DEFAULTS['auto_levels']),
        # I26-I30: Spatial advanced
        'affine_shear': spatial.get('affineShear', PHOTO_DEFAULTS['affine_shear']),
        'elastic_deform': spatial.get('elasticDeform', PHOTO_DEFAULTS['elastic_deform']),
        'border_inject': spatial.get('borderInject', PHOTO_DEFAULTS['border_inject']),
        # I16-I20: Anti-detection
        'trace_removal': options.get('traceRemoval', PHOTO_DEFAULTS['trace_removal']),
        'exif_injection': options.get('exifInjection', PHOTO_DEFAULTS['exif_injection']),
        'device_profile': options.get('deviceProfile', PHOTO_DEFAULTS['device_profile']),
        'gps_injection': options.get('gpsInjection', PHOTO_DEFAULTS['gps_injection']),
        'timestamp_injection': options.get('timestampInjection', PHOTO_DEFAULTS['timestamp_injection']),
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


def process_batch_spoofer(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    variations: int,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Process image and generate multiple unique variations as ZIP.

    Args:
        input_path: Path to input image
        output_path: Path for output ZIP file
        config: Transform configuration
        variations: Number of variations to generate (processed files, not copies)
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
    print(f"[DEBUG]   variations: {variations}")
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

    report_progress(0.1, f"Generating {variations} variations...")

    # Apply mode multipliers to config (scales all transforms by mode intensity)
    config = apply_mode_to_config(config)

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

    # I1-I8: New color transforms
    params['hue_shift'] = tonal.get('hueShift', PHOTO_DEFAULTS['hue_shift'])
    params['color_balance'] = tonal.get('colorBalance', PHOTO_DEFAULTS['color_balance'])
    params['color_temperature'] = tonal.get('colorTemperature', PHOTO_DEFAULTS['color_temperature'])
    params['vibrance'] = tonal.get('vibrance', PHOTO_DEFAULTS['vibrance'])
    params['curves_preset'] = tonal.get('curvesPreset', PHOTO_DEFAULTS['curves_preset'])
    params['color_channel_mixer'] = tonal.get('colorChannelMixer', PHOTO_DEFAULTS['color_channel_mixer'])
    params['levels'] = tonal.get('levels', PHOTO_DEFAULTS['levels'])
    params['sharpness'] = visual.get('sharpness', PHOTO_DEFAULTS['sharpness'])

    # I9-I15: New visual effects
    params['gradient_overlay'] = visual.get('gradientOverlay', PHOTO_DEFAULTS['gradient_overlay'])
    params['scanline'] = visual.get('scanline', PHOTO_DEFAULTS['scanline'])
    params['film_grain'] = visual.get('filmGrain', PHOTO_DEFAULTS['film_grain'])
    params['color_filter_preset'] = visual.get('colorFilterPreset', PHOTO_DEFAULTS['color_filter_preset'])
    params['dynamic_color_shift'] = visual.get('dynamicColorShift', PHOTO_DEFAULTS['dynamic_color_shift'])
    params['edge_enhance'] = visual.get('edgeEnhance', PHOTO_DEFAULTS['edge_enhance'])
    params['dither'] = visual.get('dither', PHOTO_DEFAULTS['dither'])

    # I21-I25: Enhancement
    params['denoise'] = visual.get('denoise', PHOTO_DEFAULTS['denoise'])
    params['auto_levels'] = tonal.get('autoLevels', PHOTO_DEFAULTS['auto_levels'])

    # I26-I30: Spatial advanced
    params['affine_shear'] = spatial.get('affineShear', PHOTO_DEFAULTS['affine_shear'])
    params['elastic_deform'] = spatial.get('elasticDeform', PHOTO_DEFAULTS['elastic_deform'])
    params['border_inject'] = spatial.get('borderInject', PHOTO_DEFAULTS['border_inject'])

    # I16-I20: Anti-detection
    params['trace_removal'] = options.get('traceRemoval', PHOTO_DEFAULTS['trace_removal'])
    params['exif_injection'] = options.get('exifInjection', PHOTO_DEFAULTS['exif_injection'])
    params['device_profile'] = options.get('deviceProfile', PHOTO_DEFAULTS['device_profile'])
    params['gps_injection'] = options.get('gpsInjection', PHOTO_DEFAULTS['gps_injection'])
    params['timestamp_injection'] = options.get('timestampInjection', PHOTO_DEFAULTS['timestamp_injection'])

    # Store variation multiplier for per-copy randomization
    params['_variation_mult'] = config.get('_variation_mult', 0.3)

    phash_min = options.get('phashMinDistance', PHASH_MIN_DISTANCE)
    verify_phash = options.get('verifyPhash', True)
    compare_variations = options.get('compareVariations', options.get('compareCopies', True))

    # Generate variations
    base_time = time.time_ns()
    results = []
    phash_distances = []
    variation_distances = []
    existing_phashes = []

    # Prepare ZIP in memory
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for i in range(variations):
            try:
                # Progress
                progress = 0.1 + (i / variations) * 0.8
                report_progress(progress, f"Processing variation {i+1}/{variations}...")

                # Generate unique seed
                seed = generate_unique_seed(input_path, i, base_time + i * 1000)

                # Process with retry for pHash
                max_retries = 5 if verify_phash else 1
                best_result = None
                best_distance = -1

                for retry in range(max_retries):
                    retry_seed = seed + retry * 10000

                    result_img, distance, var_dist, result_phash = process_single_copy(
                        original_img,
                        original_phash,
                        params,
                        i,
                        retry_seed,
                        existing_phashes if compare_variations else [],
                        phash_min
                    )

                    # Check if meets threshold
                    meets_threshold = True
                    if verify_phash and distance > 0:
                        meets_threshold = distance >= phash_min
                        if compare_variations and var_dist < 999:
                            meets_threshold = meets_threshold and var_dist >= phash_min

                    # Track best
                    if distance > best_distance:
                        best_distance = distance
                        best_result = (result_img, distance, var_dist, result_phash)

                    if meets_threshold:
                        break

                # Use best result
                if best_result:
                    result_img, distance, var_dist, result_phash = best_result

                    if result_phash is not None:
                        existing_phashes.append(result_phash)

                    if distance > 0:
                        phash_distances.append(distance)
                    if var_dist < 999:
                        variation_distances.append(var_dist)

                # Generate filename
                py_rng = random.Random(seed)
                if params.get('random_names'):
                    filename = ''.join(py_rng.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=12)) + '.jpg'
                else:
                    rand_id = py_rng.randint(1000, 9999)
                    base_name = os.path.splitext(os.path.basename(input_path))[0]
                    clean_name = "".join([c for c in base_name if c.isalnum() or c in (' ', '-', '_')]).strip()[:20]
                    filename = f"{clean_name}_v{i}_{rand_id}.jpg"

                # Save to ZIP with encoding variation
                img_buffer = io.BytesIO()
                quality = int(params.get('quality', PHOTO_DEFAULTS['quality']))
                # Vary JPEG encoding per copy for unique file fingerprint
                save_rng = random.Random(seed + 7777)
                jpeg_subsampling = save_rng.choice([0, 1, 2])  # 4:4:4, 4:2:2, 4:2:0
                jpeg_progressive = save_rng.choice([True, False])
                result_img.save(
                    img_buffer, 'JPEG', quality=quality, optimize=True,
                    subsampling=jpeg_subsampling, progressive=jpeg_progressive,
                )
                img_buffer.seek(0)

                # I16-I20: Apply post-processing (EXIF injection, trace removal)
                # For ZIP mode, we apply to a temp file then read back
                post_params = params.copy()
                if post_params.get('exif_injection') or post_params.get('gps_injection') or post_params.get('timestamp_injection'):
                    import tempfile as _tmpmod
                    tmp_path = os.path.join(_tmpmod.gettempdir(), f'spoofer_post_{i}.jpg')
                    try:
                        with open(tmp_path, 'wb') as tf:
                            tf.write(img_buffer.getvalue())
                        apply_image_post_processing(result_img, tmp_path, post_params, py_rng)
                        with open(tmp_path, 'rb') as tf:
                            img_buffer = io.BytesIO(tf.read())
                    except Exception as pp_err:
                        print(f"[WARN] Post-processing failed for variation {i}: {pp_err}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

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
        'variations_generated': len(results),
        'files': results,
    }

    if phash_distances:
        stats['phash_avg'] = sum(phash_distances) / len(phash_distances)
        stats['phash_min'] = min(phash_distances)
        stats['phash_max'] = max(phash_distances)

    if variation_distances:
        stats['variation_distance_min'] = min(variation_distances)

    return stats
