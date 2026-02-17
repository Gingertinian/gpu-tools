"""
Spoofer Constants Module

Contains all constants used across the spoofer processor modules:
- pHash configuration
- Resolution targets
- File extensions
- NVENC session limits
- Photo defaults
- Mode multipliers
- Datacenter GPU keywords
"""

# pHash minimum distance for copy verification
PHASH_MIN_DISTANCE = 10

# Target resolutions for 9:16 aspect ratio conversion
TARGET_RESOLUTIONS = {
    'high': (1080, 1920),
    'low': (720, 1280)
}

# Supported video file extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'}

# Supported image file extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.gif'}

# NVENC session limits by GPU type (per GPU)
# Consumer GPUs (GeForce): Limited to 3-5 sessions
# Datacenter GPUs (A-series, Quadro): Unlimited sessions
NVENC_SESSION_LIMITS = {
    'consumer': 3,      # RTX 3090, 4090, 4080, etc.
    'datacenter': 16,   # A5000, A6000, A100, etc. (increased for max parallel throughput)
    'default': 2,       # Fallback
}

# Keywords to identify datacenter GPUs (for NVENC session limit detection)
DATACENTER_GPU_KEYWORDS = [
    'A100', 'A6000', 'A5000', 'A4000', 'A4500', 'A40', 'A30', 'A10',
    'V100', 'T4', 'Quadro', 'Tesla', 'H100', 'L40', 'RTX 4090', 'RTX 6000'
]

# Photo transform defaults (from original Spoofer.py)
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

# ==================== MODE PRESETS ====================
# Multipliers applied to base transform values based on mode
# Light: subtle changes (0.3-0.5x) - for content reuse
# Balanced: default (1.0x) - good balance of uniqueness and quality
# Aggressive: maximum variation (2.0-3.0x) - different hash each time

MODE_MULTIPLIERS = {
    'light': {
        'spatial': 0.3,      # Very subtle spatial changes
        'tonal': 0.4,        # Minimal color shifts
        'visual': 0.3,       # Light noise/tint
        'variation': 0.2,    # Low randomization range
    },
    'balanced': {
        'spatial': 1.0,      # Default spatial transforms
        'tonal': 1.0,        # Default color adjustments
        'visual': 1.0,       # Default noise/tint
        'variation': 0.3,    # Moderate randomization range
    },
    'aggressive': {
        'spatial': 2.5,      # Strong spatial transforms (more crop, rotation)
        'tonal': 2.0,        # Noticeable color shifts
        'visual': 2.5,       # High noise/tint
        'variation': 0.5,    # Wide randomization range
    }
}
