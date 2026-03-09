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

# Video transform defaults (restored from old Spoofer.py + new features)
VIDEO_DEFAULTS = {
    # Existing transforms
    'crop': 2.2,
    'rotation': 1.0,
    'brightness': 0.07,
    'contrast': 0.07,
    'saturation': 0.10,
    'gamma': 0.10,
    'noise': 3,
    'vignette': 2.0,
    'speedVariation': 0,
    'fpsVar': 2.0,
    'bitrate': 100,

    # V1-V15: Restored from old spoofer
    'hueShift': 3.0,          # V1: Hue shift in degrees
    'sharpness': 0.4,         # V2: Unsharp mask luma amount
    'blurSigma': 0.0,         # V3: Gaussian blur sigma
    'chromatic': 0.6,         # V4: Chromatic aberration shift
    'centerShift': 0.8,       # V5: Center shift percentage
    'zoompan': 1.0,           # V6: ZoomPan percentage
    'border': 0,              # V7: Border padding pixels
    'frameJitter': 1.0,       # V8: Frame jitter percentage
    'timeStretch': 0.8,       # V9: Time stretch percentage
    'frameDropDup': 1.5,      # V10: Frame drop/dup percentage
    'audioPitch': 0.3,        # V11: Audio pitch shift (semitones input)
    'audioTempo': 0,          # V12: Audio tempo percentage
    'flip': 1,                # V13: Horizontal flip (0/1)
    'microRescale': 0.3,      # V14: Micro rescale percentage
    'force916': 0,            # V15: 9:16 background split (0/1)

    # V16-V25: New color/visual filters
    'colorBalance': 0,        # V16: Color balance (0-0.05)
    'colorChannelMixer': 0,   # V17: Color channel mixer (0-0.05)
    'vibrance': 0,            # V18: Vibrance (0-0.3)
    'colorTemperature': 0,    # V19: Color temperature offset (0-500K)
    'colorLevels': 0,         # V20: Color levels (0-0.05)
    'curvesPreset': 'none',   # V21: Curves preset name
    'dynamicHue': 0,          # V22: Dynamic hue shift amplitude (0-5°)
    'filmGrain': 0,           # V23: Film grain amplitude (0-3)
    'scanline': 0,            # V24: Scanline effect amplitude (0-0.03)
    'gradientOverlay': 0,     # V25: Gradient overlay opacity (0-0.05)

    # V28-V29: Frame trimming
    'trimStartFrames': 0,     # V28: Trim frames from start (0-10)
    'trimEndFrames': 0,       # V29: Trim frames from end (0-10)

    # V32-V40: Audio manipulation
    'audioEq': 0,             # V32: Audio EQ gain (0-2dB)
    'audioNoise': 0,          # V33: Audio noise injection (0-0.01)
    'audioHighLowPass': 0,    # V34: Audio high/low pass (0/1)
    'audioStereoWidth': 0,    # V35: Audio stereo width (0-0.1)
    'audioEcho': 0,           # V36: Audio micro echo delay ms (0-8)
    'audioSrCycle': 0,        # V37: Audio sample rate cycle (0/1)
    'audioPan': 0,            # V38: Audio channel pan (0-0.05)
    'audioCompressor': 0,     # V39: Audio compressor (0/1)

    # V41-V47: Anti-detection
    'traceRemoval': 1,        # V41: FFmpeg trace removal (0/1)
    'metadataInjection': 0,   # V42: EXIF/metadata injection (0/1)
    'mobileEncoding': 0,      # V43: Mobile encoding simulation (0/1)
    'deviceProfile': 'none',  # V44: Device profile name
    'gpsInjection': 0,        # V45: GPS injection (0/1)
    'timestampRandomize': 0,  # V46: Timestamp randomization (0/1)

    # V48-V53: Encoding variation
    'gopVariation': 0,        # V48: GOP length variation (0/1)
    'bframeVariation': 0,     # V49: B-frame count variation (0/1)
    'refVariation': 0,        # V50: Reference frames variation (0/1)
    'crfVariation': 0,        # V51: CRF randomization range (0-3)
    'profileVariation': 0,    # V52: Profile variation (0/1)
    'deblockVariation': 0,    # V53: Deblock filter variation (0/1)
}

# Photo transform defaults (from original Spoofer.py)
# NOTE: These are BASE values for "balanced" mode. Mode multipliers scale them.
PHOTO_DEFAULTS = {
    'crop': 2.5,
    'micro_resize': 2.0,
    'rotation': 1.5,
    'subpixel': 1.2,
    'warp': 1.2,
    'barrel': 1.0,
    'block_shift': 1.0,
    'scale': 97,
    'micro_rescale': 0.6,
    'brightness': 0.08,
    'gamma': 0.10,
    'contrast': 0.07,
    'vignette': 3.0,
    'freq_noise': 0,
    'invisible_watermark': 0,
    'color_space_conv': 0,
    'saturation': 0.10,
    'tint': 2.5,
    'chromatic': 1.2,
    'noise': 4.0,
    'quality': 88,
    'double_compress': 1,
    'flip': 1,
    'force_916': 1,

    # I1-I8: New color transforms
    'hue_shift': 2.0,
    'color_balance': 0,
    'color_temperature': 0,
    'vibrance': 0.1,
    'curves_preset': 'none',
    'color_channel_mixer': 0,
    'levels': 0,
    'sharpness': 0.3,

    # I9-I15: New visual effects
    'gradient_overlay': 0,
    'scanline': 0,
    'film_grain': 1.0,
    'color_filter_preset': 'none',
    'dynamic_color_shift': 0,
    'edge_enhance': 0,
    'dither': 0,

    # Options
    'random_names': 0,

    # I16-I20: Anti-detection
    'exif_injection': 0,
    'device_profile': 'none',
    'gps_injection': 0,
    'timestamp_injection': 0,
    'trace_removal': 1,

    # I21-I23: Enhancement
    'denoise': 0,
    'auto_levels': 0,

    # I26-I30: Spatial advanced
    'affine_shear': 0,
    'elastic_deform': 0,
    'border_inject': 0,
}

# ==================== MODE PRESETS ====================
# Multipliers applied to base transform values based on mode
# Stealth: barely detectable (0.1-0.2x) - minimal fingerprint change
# Light: subtle changes (0.3-0.5x) - for content reuse
# Balanced: default (1.0x) - good balance of uniqueness and quality
# Aggressive: strong variation (2.0-2.5x) - different hash each time
# Maximum: everything cranked up (3.5-4.0x) - highest uniqueness

MODE_MULTIPLIERS = {
    'stealth': {
        'spatial': 0.15,     # Barely perceptible spatial changes
        'tonal': 0.2,        # Near-invisible color shifts
        'visual': 0.15,      # Minimal noise/tint
        'variation': 0.1,    # Very low randomization range
    },
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
    },
    'maximum': {
        'spatial': 4.0,      # Maximum spatial transforms
        'tonal': 3.0,        # Strong color shifts
        'visual': 4.0,       # Maximum noise/tint
        'variation': 0.7,    # Widest randomization range
    },
}

# ==================== DEVICE PROFILES ====================
# Used for V42-V44: Metadata injection and mobile encoding simulation
DEVICE_PROFILES = {
    'iphone_15_pro': {
        'make': 'Apple',
        'model': 'iPhone 15 Pro',
        'software': '17.4',
        'encoder': 'com.apple.avfoundation',
        'profile': 'high',
        'level': '4.1',
        'audio_sr': 44100,
        'audio_codec': 'aac',
        'gop': 60,
        'bf': 0,
        'refs': 1,
    },
    'iphone_14': {
        'make': 'Apple',
        'model': 'iPhone 14',
        'software': '16.6',
        'encoder': 'com.apple.avfoundation',
        'profile': 'high',
        'level': '4.0',
        'audio_sr': 44100,
        'audio_codec': 'aac',
        'gop': 60,
        'bf': 0,
        'refs': 1,
    },
    'samsung_s24': {
        'make': 'Samsung',
        'model': 'SM-S921B',
        'software': 'One UI 6.1',
        'encoder': 'OMX.Exynos.AVC.Encoder',
        'profile': 'high',
        'level': '4.0',
        'audio_sr': 48000,
        'audio_codec': 'aac',
        'gop': 30,
        'bf': 0,
        'refs': 1,
    },
    'pixel_8': {
        'make': 'Google',
        'model': 'Pixel 8',
        'software': 'Android 14',
        'encoder': 'c2.qti.avc.encoder',
        'profile': 'high',
        'level': '4.0',
        'audio_sr': 48000,
        'audio_codec': 'aac',
        'gop': 30,
        'bf': 0,
        'refs': 1,
    },
    'rayban_meta': {
        'make': 'Ray-Ban',
        'model': 'Meta Smart Glasses',
        'software': 'Meta View 1.0',
        'encoder': 'OMX.qcom.video.encoder.avc',
        'profile': 'main',
        'level': '3.1',
        'audio_sr': 44100,
        'audio_codec': 'aac',
        'gop': 30,
        'bf': 0,
        'refs': 1,
    },
}

# Valid curves presets for FFmpeg (V21)
CURVES_PRESETS = [
    'none', 'color_negative', 'cross_process', 'darker',
    'increase_contrast', 'lighter', 'linear_contrast',
    'medium_contrast', 'negative', 'strong_contrast', 'vintage',
]
