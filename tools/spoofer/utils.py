"""
Spoofer Utilities Module

Contains utility functions used across the spoofer processor:
- generate_unique_seed: Generate unique seed for reproducible randomization
- extract_images_from_zip: Extract image files from ZIP archives
- extract_videos_from_zip: Extract video files from ZIP archives
- randomize_params: Randomize parameters within variation range
- get_mode_multipliers: Get mode multipliers for light/balanced/aggressive modes
- apply_mode_to_config: Apply mode multipliers to transform config
"""

import os
import time
import hashlib
import zipfile
import random
from typing import Dict, Any, List

from .constants import (
    VIDEO_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MODE_MULTIPLIERS,
)


def generate_unique_seed(img_path: str, var_idx: int, base_time: int) -> int:
    """Generate unique seed for reproducible randomization."""
    hash_input = f"{img_path}_{var_idx}_{base_time}_{os.getpid()}"
    return int(hashlib.sha256(hash_input.encode()).hexdigest()[:8], 16)


def safe_extract(zf: zipfile.ZipFile, name: str, extract_dir: str) -> str:
    """Safely extract a zip entry, preventing path traversal attacks (ZipSlip)."""
    target_path = os.path.realpath(os.path.join(extract_dir, name))
    extract_dir_real = os.path.realpath(extract_dir)
    if not target_path.startswith(extract_dir_real + os.sep) and target_path != extract_dir_real:
        raise ValueError(f"Attempted path traversal in zip entry: {name}")
    zf.extract(name, extract_dir)
    return target_path


def extract_images_from_zip(zip_path: str, extract_dir: str) -> List[str]:
    """
    Extract image files from a ZIP archive.
    Returns list of paths to extracted image files.
    """
    image_paths = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            ext = os.path.splitext(name)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                # Extract to temp dir
                extracted_path = safe_extract(zf, name, extract_dir)
                image_paths.append(extracted_path)
    return image_paths


def extract_videos_from_zip(zip_path: str, extract_dir: str) -> List[str]:
    """
    Extract video files from a ZIP archive.
    Returns list of paths to extracted video files.
    """
    video_paths = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            ext = os.path.splitext(name)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                # Extract to temp dir
                extracted_path = safe_extract(zf, name, extract_dir)
                video_paths.append(extracted_path)
    return video_paths


def randomize_params(
    base_params: Dict, py_rng: random.Random, variation: float = 0.3
) -> Dict:
    """
    Randomize parameters within +/- variation of base values.
    Each copy gets slightly different transform intensities.
    """
    result = {}
    for key, value in base_params.items():
        if isinstance(value, (int, float)) and value > 0:
            # Vary by +/- variation percentage
            factor = 1.0 + py_rng.uniform(-variation, variation)
            new_value = value * factor
            # Keep same type
            result[key] = int(new_value) if isinstance(value, int) else new_value
        else:
            result[key] = value
    return result


def get_mode_multipliers(mode: str) -> Dict[str, float]:
    """Get mode multipliers, defaulting to 'balanced' if unknown mode."""
    return MODE_MULTIPLIERS.get(mode, MODE_MULTIPLIERS["balanced"])


def apply_mode_to_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply mode multipliers to transform values in config.
    This ensures visible differences between light/balanced/aggressive modes.

    Args:
        config: Original config dict with spatial, tonal, visual sections

    Returns:
        Modified config with transform values scaled by mode multipliers
    """
    mode = config.get("mode", "balanced")
    multipliers = get_mode_multipliers(mode)

    print(f"[Spoofer] Applying mode '{mode}' multipliers: {multipliers}")

    # Deep copy to avoid modifying original
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = dict(value)
        else:
            result[key] = value

    # Apply spatial multiplier
    spatial = result.get("spatial", {})
    spatial_mult = multipliers["spatial"]
    for key in [
        "crop",
        "microResize",
        "rotation",
        "subpixel",
        "warp",
        "barrel",
        "blockShift",
        "microRescale",
    ]:
        if key in spatial and spatial[key] > 0:
            original = spatial[key]
            spatial[key] = spatial[key] * spatial_mult
            print(f"[Spoofer]   spatial.{key}: {original} -> {spatial[key]:.2f}")
    result["spatial"] = spatial

    # Apply tonal multiplier
    tonal = result.get("tonal", {})
    tonal_mult = multipliers["tonal"]
    for key in ["brightness", "gamma", "contrast", "saturation", "vignette"]:
        if key in tonal and tonal[key] > 0:
            original = tonal[key]
            tonal[key] = tonal[key] * tonal_mult
            print(f"[Spoofer]   tonal.{key}: {original} -> {tonal[key]:.3f}")
    result["tonal"] = tonal

    # Apply visual multiplier
    visual = result.get("visual", {})
    visual_mult = multipliers["visual"]
    for key in ["tint", "chromatic", "noise"]:
        if key in visual and visual[key] > 0:
            original = visual[key]
            visual[key] = visual[key] * visual_mult
            print(f"[Spoofer]   visual.{key}: {original} -> {visual[key]:.2f}")
    result["visual"] = visual

    # Store variation multiplier for randomize_params
    result["_variation_mult"] = multipliers["variation"]

    return result
