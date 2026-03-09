"""
Image Transforms Module

Contains all image transformation functions for pHash evasion:

TIER 1: SPATIAL TRANSFORMS (Maximum pHash impact)
- apply_asymmetric_crop
- apply_micro_resize
- apply_micro_rotation
- apply_subpixel_shift
- apply_perspective_warp
- apply_barrel_distortion
- apply_block_shift
- apply_micro_rescale_photo

TIER 2: TONAL TRANSFORMS (DCT coefficients)
- apply_brightness_shift
- apply_gamma_shift
- apply_contrast_shift
- apply_saturation_shift
- apply_vignette
- apply_frequency_noise
- apply_invisible_watermark
- apply_color_space_conversion

TIER 3: VISUAL VARIATION
- apply_color_tint
- apply_chromatic_aberration
- apply_noise
- apply_double_compression

HELPER FUNCTIONS:
- find_perspective_coeffs
- pad_to_9_16_photo
"""

import io
import math
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict

from .constants import TARGET_RESOLUTIONS


# =============================================================================
# TIER 1: SPATIAL TRANSFORMS (Maximum pHash impact)
# =============================================================================

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
    """Micro rotation with zoom compensation to eliminate black borders."""
    if strength <= 0:
        return img

    angle = py_rng.uniform(-strength, strength)
    orig_w, orig_h = img.size

    # Rotate with expand=True to avoid black corners
    rotated_img = img.rotate(angle, expand=True, resample=Image.BICUBIC)
    rot_w, rot_h = rotated_img.size

    # Calculate the largest rectangle that fits inside the rotated image
    angle_rad = abs(angle) * math.pi / 180
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    if sin_a < 0.001:  # Nearly zero rotation
        return img

    # Scale factor for inscribed rectangle
    scale = min(
        1.0 / (cos_a + sin_a * orig_h / orig_w),
        1.0 / (cos_a + sin_a * orig_w / orig_h)
    )

    # The inscribed rectangle dimensions
    crop_w = int(orig_w * scale)
    crop_h = int(orig_h * scale)

    # Center crop from rotated image
    left = (rot_w - crop_w) // 2
    top = (rot_h - crop_h) // 2

    cropped = rotated_img.crop((left, top, left + crop_w, top + crop_h))

    # Resize back to original dimensions
    return cropped.resize((orig_w, orig_h), Image.LANCZOS)


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


# =============================================================================
# TIER 2: TONAL TRANSFORMS (DCT coefficients)
# =============================================================================

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
    """RGB to YUV to RGB conversion with minimal variation."""
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


# =============================================================================
# TIER 3: VISUAL VARIATION
# =============================================================================

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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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


# =============================================================================
# NEW IMAGE TRANSFORMS (I1-I15, I21-I30)
# =============================================================================

def apply_hue_shift(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I1: Hue shift via HSV rotation."""
    if strength <= 0:
        return img
    shift = py_rng.uniform(-strength, strength)
    arr = np.array(img.convert('HSV'))
    arr[:, :, 0] = (arr[:, :, 0].astype(np.int16) + int(shift * 255 / 360)) % 256
    return Image.fromarray(arr, 'HSV').convert('RGB')


def apply_color_balance(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I2: Color balance adjustment per shadow/mid/highlight ranges."""
    if strength <= 0:
        return img
    arr = np.array(img, dtype=np.float32)
    for ch in range(3):
        shift = py_rng.uniform(-strength, strength) * 255
        arr[:, :, ch] = np.clip(arr[:, :, ch] + shift, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_color_temperature(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I3: Color temperature shift (warm/cool)."""
    if strength <= 0:
        return img
    temp_shift = py_rng.uniform(-strength, strength) / 500.0
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] + temp_shift * 30, 0, 255)  # Red
    arr[:, :, 2] = np.clip(arr[:, :, 2] - temp_shift * 30, 0, 255)  # Blue
    return Image.fromarray(arr.astype(np.uint8))


def apply_vibrance(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I4: Selective saturation boost (more on unsaturated areas)."""
    if strength <= 0:
        return img
    amount = py_rng.uniform(0, strength)
    hsv = np.array(img.convert('HSV'), dtype=np.float32)
    s = hsv[:, :, 1]
    # Boost low-saturation pixels more than high-saturation ones
    boost = amount * (1.0 - s / 255.0)
    hsv[:, :, 1] = np.clip(s + boost * 50, 0, 255)
    return Image.fromarray(hsv.astype(np.uint8), 'HSV').convert('RGB')


def apply_curves_preset(img: Image.Image, preset: str, py_rng: random.Random) -> Image.Image:
    """I5: Pre-defined tone curve presets (matches FFmpeg curves presets for images)."""
    if preset == 'none' or not preset:
        return img
    arr = np.array(img, dtype=np.float32)
    if preset == 'lighter':
        arr = np.clip(arr * 1.05 + 5, 0, 255)
    elif preset == 'darker':
        arr = np.clip(arr * 0.92 - 3, 0, 255)
    elif preset == 'vintage':
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.03 + 3, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.98, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.95, 0, 255)
    elif preset == 'cinema' or preset == 'cross_process':
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.97, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 1.02, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.05, 0, 255)
    elif preset == 'increase_contrast' or preset == 'strong_contrast':
        mean = arr.mean()
        factor = 1.15 if preset == 'strong_contrast' else 1.08
        arr = np.clip((arr - mean) * factor + mean, 0, 255)
    elif preset == 'medium_contrast' or preset == 'linear_contrast':
        mean = arr.mean()
        arr = np.clip((arr - mean) * 1.04 + mean, 0, 255)
    elif preset == 'negative' or preset == 'color_negative':
        arr = 255.0 - arr
        if preset == 'color_negative':
            # Slightly shift channels for color negative effect
            arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.95, 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.05, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_color_channel_mixer(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I6: Cross-channel blending matrix."""
    if strength <= 0:
        return img
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mix = py_rng.uniform(0, strength)
    arr[:, :, 0] = np.clip(r * (1 - mix) + g * mix * 0.5, 0, 255)
    arr[:, :, 1] = np.clip(g * (1 - mix) + b * mix * 0.5, 0, 255)
    arr[:, :, 2] = np.clip(b * (1 - mix) + r * mix * 0.5, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_levels_adjustment(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I7: Input/output level clipping per channel."""
    if strength <= 0:
        return img
    clip = int(py_rng.uniform(0, strength))
    if clip <= 0:
        return img
    arr = np.array(img, dtype=np.float32)
    arr = np.clip((arr - clip) * (255.0 / (255.0 - 2 * clip)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_sharpness(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I8: Unsharp mask sharpening."""
    if strength <= 0:
        return img
    amount = py_rng.uniform(0.5, 1.0 + strength)
    return img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(amount * 100), threshold=3))


def apply_gradient_overlay(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I9: Semi-transparent gradient overlay."""
    if strength <= 0:
        return img
    w, h = img.size
    arr = np.array(img, dtype=np.float32)
    # Create diagonal gradient
    gradient = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            gradient[y, x] = (x / w + y / h) / 2.0
    # Apply as subtle brightness modulation
    mod = 1.0 + (gradient - 0.5) * strength * 2
    for ch in range(3):
        arr[:, :, ch] = np.clip(arr[:, :, ch] * mod, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_scanline_pattern(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I10: Sine-wave brightness modulation per row."""
    if strength <= 0:
        return img
    arr = np.array(img, dtype=np.float32)
    h = arr.shape[0]
    for y in range(h):
        mod = 1.0 + strength * math.sin(y * math.pi)
        arr[y, :, :] = np.clip(arr[y, :, :] * mod, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_film_grain(img: Image.Image, strength: float, py_rng: random.Random, rng: np.random.Generator = None) -> Image.Image:
    """I11: Procedural film grain overlay."""
    if strength <= 0:
        return img
    arr = np.array(img, dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng(py_rng.randint(0, 2**31))
    grain = rng.normal(0, strength * 3, arr.shape).astype(np.float32)
    arr = np.clip(arr + grain, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_color_filter_preset(img: Image.Image, preset: str, py_rng: random.Random) -> Image.Image:
    """I12: Named color filter application (warm/cool/vintage/cinema)."""
    if preset == 'none' or not preset:
        return img
    arr = np.array(img, dtype=np.float32)
    if preset == 'warm':
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.04, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.96, 0, 255)
    elif preset == 'cool':
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.96, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.04, 0, 255)
    elif preset == 'vintage':
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.02 + 3, 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.98, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.94, 0, 255)
    elif preset == 'cinema':
        # Teal/orange cinematic look: boost blue shadows, warm highlights
        arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.03, 0, 255)  # Slight red boost
        arr[:, :, 1] = np.clip(arr[:, :, 1] * 0.98, 0, 255)  # Slight green reduction
        arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.05, 0, 255)  # Blue boost
        # Increase contrast slightly for cinematic feel
        mean = arr.mean()
        arr = np.clip((arr - mean) * 1.05 + mean, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_dynamic_color_shift(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I13: Per-region hue variation."""
    if strength <= 0:
        return img
    arr = np.array(img.convert('HSV'), dtype=np.float32)
    h_img, w_img = arr.shape[:2]
    block_h = max(1, h_img // 4)
    block_w = max(1, w_img // 4)
    for by in range(0, h_img, block_h):
        for bx in range(0, w_img, block_w):
            shift = py_rng.uniform(-strength, strength) * 255 / 360
            arr[by:by+block_h, bx:bx+block_w, 0] = (arr[by:by+block_h, bx:bx+block_w, 0] + shift) % 256
    return Image.fromarray(arr.astype(np.uint8), 'HSV').convert('RGB')


def apply_edge_enhancement(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I14: Subtle edge detect blend at low opacity."""
    if strength <= 0:
        return img
    edges = img.filter(ImageFilter.FIND_EDGES)
    return Image.blend(img, edges, strength)


def apply_dither_pattern(img: Image.Image, strength: float, py_rng: random.Random, rng: np.random.Generator = None) -> Image.Image:
    """I15: Ordered dither at sub-perceptual level."""
    if strength <= 0:
        return img
    arr = np.array(img, dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng(py_rng.randint(0, 2**31))
    # Add very subtle dither noise
    dither = rng.integers(-int(strength), int(strength) + 1, size=arr.shape, dtype=np.int8).astype(np.float32)
    arr = np.clip(arr + dither, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


# --- I21-I30: Enhancement, Quality, Spatial Advanced ---

def apply_denoise(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I22: Light denoising via median filter."""
    if strength <= 0:
        return img
    size = 3  # PIL median filter only supports size 3
    return img.filter(ImageFilter.MedianFilter(size=size))


def apply_auto_levels(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I23: Histogram stretching per channel."""
    if strength <= 0:
        return img
    clip_pct = strength / 100.0
    arr = np.array(img, dtype=np.float32)
    for ch in range(3):
        channel = arr[:, :, ch]
        lo = np.percentile(channel, clip_pct * 100)
        hi = np.percentile(channel, 100 - clip_pct * 100)
        if hi - lo > 1:
            arr[:, :, ch] = np.clip((channel - lo) * 255.0 / (hi - lo), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def apply_affine_shear(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I27: Slight horizontal/vertical shear."""
    if strength <= 0:
        return img
    shear = py_rng.uniform(-strength, strength) * math.pi / 180.0
    w, h = img.size
    # Affine transform coefficients for horizontal shear
    coeffs = (1, shear, -shear * h / 2, 0, 1, 0)
    return img.transform((w, h), Image.AFFINE, coeffs, resample=Image.BICUBIC)


def apply_elastic_deformation(img: Image.Image, strength: float, py_rng: random.Random, rng: np.random.Generator = None) -> Image.Image:
    """I28: Small random displacement field."""
    if strength <= 0:
        return img
    arr = np.array(img)
    h, w = arr.shape[:2]
    if rng is None:
        rng = np.random.default_rng(py_rng.randint(0, 2**31))
    # Small random displacement
    dx = rng.normal(0, strength, (h, w)).astype(np.float32)
    dy = rng.normal(0, strength, (h, w)).astype(np.float32)
    # Smooth displacement field
    from PIL import ImageFilter as IF
    dx_img = Image.fromarray((dx * 10 + 128).clip(0, 255).astype(np.uint8))
    dy_img = Image.fromarray((dy * 10 + 128).clip(0, 255).astype(np.uint8))
    dx = (np.array(dx_img.filter(IF.GaussianBlur(radius=5)), dtype=np.float32) - 128) / 10
    dy = (np.array(dy_img.filter(IF.GaussianBlur(radius=5)), dtype=np.float32) - 128) / 10

    # Apply displacement
    coords_y, coords_x = np.mgrid[0:h, 0:w]
    map_x = np.clip(coords_x + dx, 0, w - 1).astype(np.int32)
    map_y = np.clip(coords_y + dy, 0, h - 1).astype(np.int32)
    result = arr[map_y, map_x]
    return Image.fromarray(result)


def apply_border_inject(img: Image.Image, strength: float, py_rng: random.Random) -> Image.Image:
    """I30: 1-3px colored border injection."""
    if strength <= 0:
        return img
    border_px = py_rng.randint(1, max(1, int(strength)))
    color = (py_rng.randint(0, 255), py_rng.randint(0, 255), py_rng.randint(0, 255))
    w, h = img.size
    new_w = w + 2 * border_px
    new_h = h + 2 * border_px
    bordered = Image.new('RGB', (new_w, new_h), color)
    bordered.paste(img, (border_px, border_px))
    return bordered


def apply_image_post_processing(img: Image.Image, output_path: str, params: Dict, py_rng: random.Random) -> None:
    """
    I16-I20: Image metadata anti-detection post-processing.
    Applies EXIF injection, GPS injection, timestamp injection, trace removal.
    Called after image is saved.
    """
    try:
        import struct

        trace_removal = int(params.get('trace_removal', 1) or 0)
        exif_injection = int(params.get('exif_injection', 0) or 0)
        device_profile = str(params.get('device_profile', 'none') or 'none')
        gps_injection = int(params.get('gps_injection', 0) or 0)
        timestamp_injection = int(params.get('timestamp_injection', 0) or 0)

        if not any([trace_removal, exif_injection, gps_injection, timestamp_injection]):
            return

        # Try to use piexif for EXIF manipulation
        try:
            import piexif

            if trace_removal:
                # Strip all EXIF data first
                piexif.remove(output_path)

            if exif_injection or gps_injection or timestamp_injection:
                exif_dict = {'0th': {}, 'Exif': {}, 'GPS': {}, '1st': {}}

                if device_profile != 'none':
                    from .constants import DEVICE_PROFILES
                    profile = DEVICE_PROFILES.get(device_profile, {})
                    if profile:
                        exif_dict['0th'][piexif.ImageIFD.Make] = profile['make'].encode()
                        exif_dict['0th'][piexif.ImageIFD.Model] = profile['model'].encode()
                        exif_dict['0th'][piexif.ImageIFD.Software] = profile['software'].encode()

                if gps_injection:
                    lat = py_rng.uniform(25.0, 48.0)  # Continental US range
                    lon = py_rng.uniform(-125.0, -70.0)
                    # Convert to degrees/minutes/seconds
                    lat_d = int(lat)
                    lat_m = int((lat - lat_d) * 60)
                    lat_s = int(((lat - lat_d) * 60 - lat_m) * 60 * 100)
                    lon_d = int(abs(lon))
                    lon_m = int((abs(lon) - lon_d) * 60)
                    lon_s = int(((abs(lon) - lon_d) * 60 - lon_m) * 60 * 100)
                    exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = ((lat_d, 1), (lat_m, 1), (lat_s, 100))
                    exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = b'N'
                    exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = ((lon_d, 1), (lon_m, 1), (lon_s, 100))
                    exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = b'W'

                if timestamp_injection:
                    import datetime
                    days_back = py_rng.randint(1, 30)
                    dt = datetime.datetime.now() - datetime.timedelta(days=days_back)
                    ts = dt.strftime('%Y:%m:%d %H:%M:%S')
                    exif_dict['0th'][piexif.ImageIFD.DateTime] = ts.encode()
                    exif_dict['Exif'][piexif.ExifIFD.DateTimeOriginal] = ts.encode()

                exif_bytes = piexif.dump(exif_dict)
                piexif.insert(exif_bytes, output_path)

        except ImportError:
            # piexif not available - just strip metadata by re-saving
            if trace_removal:
                clean_img = Image.open(output_path)
                clean_data = clean_img.getdata()
                clean_copy = Image.new(clean_img.mode, clean_img.size)
                clean_copy.putdata(list(clean_data))
                clean_copy.save(output_path, quality=95)

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[Image Post-Processing] Error: {e}")


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
