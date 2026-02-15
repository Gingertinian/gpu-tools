"""
Captioner Processor - Add text captions to images and videos

Adds customizable text overlays, captions, and watermarks to media.
Supports multiple fonts, positions, styles, and animations.

Config structure:
{
    "text": "Caption text",
    "position": "top" | "center" | "bottom" | "custom",
    "customX": pixels,
    "customY": pixels,
    "fontSize": pixels,
    "fontFamily": "Arial" | "Impact" | "Roboto" | etc,
    "fontWeight": "normal" | "bold",
    "color": "#hexcolor",
    "backgroundColor": "#hexcolor" | null,
    "backgroundOpacity": 0-100,
    "strokeColor": "#hexcolor" | null,
    "strokeWidth": pixels,
    "shadow": boolean,
    "animation": "none" | "fade" | "slide" | "typewriter",
    "startTime": seconds,
    "endTime": seconds | null (full duration),
    "maxWidth": pixels | null (auto)
}

PARALLEL VIDEO PROCESSING: Supports processing multiple videos in parallel
using multiple NVENC sessions (optimized for datacenter GPUs like A5000/A6000).
"""

import os
import subprocess
import tempfile
import zipfile
from typing import Callable, Optional, Dict, Any, List, Tuple
from io import BytesIO
from urllib.parse import quote as url_quote
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import json
import textwrap
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Video extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}

# NVENC session limits by GPU type
NVENC_SESSION_LIMITS = {
    "consumer": 3,
    "datacenter": 10,
    "default": 2,
}


def get_gpu_info() -> Dict[str, Any]:
    """
    Detect all GPUs, their types, and determine NVENC session limits.

    Returns:
        Dict with:
            - gpu_count: Number of GPUs detected
            - gpus: List of GPU info dicts (name, type, nvenc_sessions)
            - total_nvenc_sessions: Total NVENC sessions across all GPUs
            - gpu_name: Name of first GPU (for backwards compatibility)
            - gpu_type: Type of first GPU (for backwards compatibility)
            - nvenc_sessions: Sessions per GPU (for backwards compatibility)
    """
    datacenter_keywords = [
        "A100",
        "A6000",
        "A5000",
        "A4000",
        "A4500",
        "A40",
        "A30",
        "A10",
        "V100",
        "T4",
        "Quadro",
        "Tesla",
        "H100",
        "L40",
        "RTX 4090",
        "RTX 6000",
    ]

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            gpus = []

            for line in lines:
                if not line.strip():
                    continue
                parts = line.split(", ", 1)
                if len(parts) >= 2:
                    gpu_index = int(parts[0].strip())
                    gpu_name = parts[1].strip()
                else:
                    gpu_index = len(gpus)
                    gpu_name = line.strip()

                is_datacenter = any(kw in gpu_name for kw in datacenter_keywords)
                gpu_type = "datacenter" if is_datacenter else "consumer"

                gpus.append(
                    {
                        "index": gpu_index,
                        "name": gpu_name,
                        "type": gpu_type,
                        "nvenc_sessions": NVENC_SESSION_LIMITS[gpu_type],
                    }
                )

            if gpus:
                total_sessions = sum(g["nvenc_sessions"] for g in gpus)
                return {
                    "gpu_count": len(gpus),
                    "gpus": gpus,
                    "total_nvenc_sessions": total_sessions,
                    # Backwards compatibility (first GPU)
                    "gpu_name": gpus[0]["name"],
                    "gpu_type": gpus[0]["type"],
                    "nvenc_sessions": gpus[0]["nvenc_sessions"],
                }
    except Exception as e:
        print(f"[GPU Detection] Error: {e}")

    return {
        "gpu_count": 1,
        "gpus": [
            {
                "index": 0,
                "name": "Unknown",
                "type": "default",
                "nvenc_sessions": NVENC_SESSION_LIMITS["default"],
            }
        ],
        "total_nvenc_sessions": NVENC_SESSION_LIMITS["default"],
        "gpu_name": "Unknown",
        "gpu_type": "default",
        "nvenc_sessions": NVENC_SESSION_LIMITS["default"],
    }


def extract_videos_from_zip(zip_path: str, extract_dir: str) -> List[str]:
    """Extract video files from a ZIP archive."""
    video_paths = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            ext = os.path.splitext(name)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                extracted_path = zf.extract(name, extract_dir)
                video_paths.append(extracted_path)
    return video_paths


# MediaPipe for face detection
try:
    import mediapipe as mp

    # Verify mp.solutions exists (newer versions may have changed API)
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_detection"):
        MEDIAPIPE_AVAILABLE = True
    else:
        print(
            "[Captioner] MediaPipe installed but mp.solutions.face_detection not available"
        )
        MEDIAPIPE_AVAILABLE = False
        mp = None
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


# Font mapping (RunPod will have these installed)
# TikTok fonts are bundled in /app/fonts/
FONT_MAP = {
    "Arial": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "Arial Bold": "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "Arial.ttf": "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "Impact": "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "Impact.ttf": "/usr/share/fonts/truetype/msttcorefonts/Impact.ttf",
    "Roboto": "/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf",
    "Roboto Bold": "/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf",
    # TikTok fonts (bundled in Docker)
    "TikTokBold.otf": "/app/fonts/TikTokBold.otf",
    "LightItalic.otf": "/app/fonts/LightItalic.otf",
    "tiktok": "/app/fonts/TikTokBold.otf",
    "tiktok-bold": "/app/fonts/TikTokBold.otf",
    "tiktok-italic": "/app/fonts/LightItalic.otf",
    # Fallbacks
    "default": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "default-bold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "default-italic": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
}

LANCZOS_RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

APPLE_EMOJI_CDN = os.getenv("CAPTIONER_EMOJI_CDN", "https://emojicdn.elk.sh")
APPLE_EMOJI_STYLE = os.getenv("CAPTIONER_EMOJI_STYLE", "apple")
EMOJI_REQUEST_TIMEOUT_SECONDS = float(os.getenv("CAPTIONER_EMOJI_TIMEOUT", "4"))
_emoji_cache: Dict[Tuple[str, int], Image.Image] = {}


import re
import random


# ==================== CONFIG NORMALIZATION ====================


def snake_to_camel(s: str) -> str:
    """Convert snake_case to camelCase."""
    components = s.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize config keys from snake_case to camelCase.
    This allows the processor to work with both formats since
    the backend may convert camelCase to snake_case before sending.

    Also handles special cases like keeping certain keys intact.
    """
    if not config:
        return {}

    normalized = {}

    # Map of snake_case -> camelCase for known keys
    key_map = {
        "caption_mode": "captionMode",
        "image_index": "imageIndex",
        "avoid_faces": "avoidFaces",
        "font_size": "fontSize",
        "title_font_size": "titleFontSize",
        "stroke_width": "strokeWidth",
        "text_color": "textColor",
        "show_background": "showBackground",
        "background_color": "backgroundColor",
        "background_opacity": "backgroundOpacity",
        "stroke_color": "strokeColor",
        "text_width_ratio": "textWidthRatio",
        "block_spacing": "blockSpacing",
        "line_spacing": "lineSpacing",
        "random_tilt": "randomTilt",
        "tilt_range_min": "tiltRangeMin",
        "tilt_range_max": "tiltRangeMax",
        "position_x": "positionX",
        "position_y": "positionY",
        "custom_x": "customX",
        "custom_y": "customY",
        "center_every_enabled": "centerEveryEnabled",
        "center_every_n": "centerEveryN",
        "emoji_style": "emojiStyle",
        "font_family": "fontFamily",
        "font_weight": "fontWeight",
        "start_time": "startTime",
        "end_time": "endTime",
        "max_width": "maxWidth",
        "apply_position_for_all": "applyPositionForAll",
    }

    for key, value in config.items():
        # Check if it's a known snake_case key
        if key in key_map:
            normalized[key_map[key]] = value
        elif "_" in key:
            # Convert unknown snake_case to camelCase
            normalized[snake_to_camel(key)] = value
        else:
            # Keep camelCase and other keys as-is
            normalized[key] = value

    return normalized


# ==================== TEXT UTILITIES ====================


def replace_quotes(text: str) -> str:
    """Replace straight quotes with curly quotes for better typography."""
    if not text:
        return text
    text = str(text)
    in_double = False
    in_single = False
    res = ""
    for char in text:
        if char == '"':
            res += '"' if not in_double else '"'
            in_double = not in_double
        elif char == "'":
            res += """ if not in_single else """
            in_single = not in_single
        else:
            res += char
    return res


def safe_debug_repr(value: Any) -> str:
    """Return an ASCII-safe representation for debug logs."""
    try:
        return ascii(value)
    except Exception:
        return "<unprintable>"


def is_regional_indicator(codepoint: int) -> bool:
    return 0x1F1E6 <= codepoint <= 0x1F1FF


def is_skin_tone_modifier(codepoint: int) -> bool:
    return 0x1F3FB <= codepoint <= 0x1F3FF


def is_emoji_codepoint(codepoint: int) -> bool:
    emoji_ranges = [
        (0x1F300, 0x1FAFF),
        (0x2600, 0x27BF),
        (0xFE00, 0xFE0F),
    ]
    for start, end in emoji_ranges:
        if start <= codepoint <= end:
            return True
    return False


def consume_emoji_cluster(text: str, start_index: int) -> Tuple[Optional[str], int]:
    if start_index >= len(text):
        return None, start_index

    first_cp = ord(text[start_index])

    if is_regional_indicator(first_cp):
        if start_index + 1 < len(text) and is_regional_indicator(
            ord(text[start_index + 1])
        ):
            return text[start_index : start_index + 2], start_index + 2
        return text[start_index], start_index + 1

    if not is_emoji_codepoint(first_cp):
        return None, start_index

    cluster_chars = [text[start_index]]
    i = start_index + 1

    while i < len(text):
        cp = ord(text[i])

        if cp in (0xFE0F, 0x20E3) or is_skin_tone_modifier(cp):
            cluster_chars.append(text[i])
            i += 1
            continue

        if cp == 0x200D:
            cluster_chars.append(text[i])
            i += 1
            if i < len(text):
                cluster_chars.append(text[i])
                i += 1
            continue

        break

    return "".join(cluster_chars), i


def tokenize_caption_text(text: str) -> List[Tuple[str, str]]:
    tokens: List[Tuple[str, str]] = []
    buffer = ""
    i = 0

    while i < len(text):
        emoji_cluster, next_i = consume_emoji_cluster(text, i)
        if emoji_cluster:
            if buffer:
                tokens.append(("text", buffer))
                buffer = ""
            tokens.append(("emoji", emoji_cluster))
            i = next_i
            continue

        buffer += text[i]
        i += 1

    if buffer:
        tokens.append(("text", buffer))

    return tokens


def get_font_line_height(font: Any) -> int:
    try:
        ascent, descent = font.getmetrics()
        return max(1, int(ascent + descent))
    except Exception:
        bbox = font.getbbox("Ag")
        return max(1, int(bbox[3] - bbox[1]))


def estimate_emoji_size(font: Any) -> int:
    font_size = int(getattr(font, "size", 48) or 48)
    return max(12, int(font_size * 1.05))


def measure_text_width(text: str, font: Any, emoji_size: Optional[int] = None) -> float:
    if not text:
        return 0.0

    effective_emoji_size = emoji_size or estimate_emoji_size(font)
    emoji_gap = max(1, int(effective_emoji_size * 0.08))
    width = 0.0

    for token_type, token_value in tokenize_caption_text(text):
        if token_type == "text":
            width += float(font.getlength(token_value))
        else:
            width += float(effective_emoji_size + emoji_gap)

    return width


def wrap_text_mixed(text: str, font: Any, max_width: int) -> str:
    if not text:
        return ""

    wrapped_lines: List[str] = []
    emoji_size = estimate_emoji_size(font)

    for paragraph in str(text).split("\n"):
        if paragraph == "":
            wrapped_lines.append("")
            continue

        words = paragraph.split(" ")
        current_line = words[0]

        for word in words[1:]:
            candidate = f"{current_line} {word}"
            if measure_text_width(candidate, font, emoji_size) <= max_width:
                current_line = candidate
            else:
                wrapped_lines.append(current_line)
                current_line = word

        wrapped_lines.append(current_line)

    return "\n".join(wrapped_lines)


def get_ios_emoji_image(emoji_text: str, size: int) -> Optional[Image.Image]:
    cache_key = (emoji_text, size)
    cached = _emoji_cache.get(cache_key)
    if cached is not None:
        return cached.copy()

    encoded = url_quote(emoji_text, safe="")
    url = f"{APPLE_EMOJI_CDN}/{encoded}?style={APPLE_EMOJI_STYLE}"

    try:
        response = requests.get(url, timeout=EMOJI_REQUEST_TIMEOUT_SECONDS)
        if response.status_code != 200 or not response.content:
            return None

        image = Image.open(BytesIO(response.content)).convert("RGBA")
        if image.size != (size, size):
            image = image.resize((size, size), LANCZOS_RESAMPLE)

        _emoji_cache[cache_key] = image
        return image.copy()
    except Exception:
        return None


def _draw_caption_tokens(
    base_image: Image.Image,
    draw: ImageDraw.ImageDraw,
    tokens: List[Tuple[str, str]],
    x: float,
    y: float,
    font: ImageFont.ImageFont,
    fill: Tuple[int, int, int, int],
    emoji_size: int,
    emoji_style: str,
    draw_emojis: bool,
) -> None:
    cursor_x = float(x)
    line_height = get_font_line_height(font)
    emoji_gap = max(1, int(emoji_size * 0.08))

    for token_type, token_value in tokens:
        if token_type == "text":
            draw.text((cursor_x, y), token_value, font=font, fill=fill)
            cursor_x += float(font.getlength(token_value))
            continue

        if draw_emojis and emoji_style == "ios":
            emoji_image = get_ios_emoji_image(token_value, emoji_size)
            if emoji_image is not None:
                emoji_y = int(round(y + max(0, (line_height - emoji_size) / 2)))
                base_image.paste(
                    emoji_image, (int(round(cursor_x)), emoji_y), emoji_image
                )
            else:
                draw.text((cursor_x, y), token_value, font=font, fill=fill)

        cursor_x += float(emoji_size + emoji_gap)


def draw_mixed_text_line(
    base_image: Image.Image,
    draw: ImageDraw.ImageDraw,
    text: str,
    x: float,
    y: float,
    font: ImageFont.ImageFont,
    text_color: Tuple[int, int, int, int],
    emoji_style: str,
    stroke_color: Optional[Tuple[int, int, int, int]] = None,
    stroke_width: int = 0,
    shadow: bool = False,
) -> None:
    tokens = tokenize_caption_text(text)
    if not tokens:
        return

    emoji_size = estimate_emoji_size(font)

    if shadow:
        _draw_caption_tokens(
            base_image=base_image,
            draw=draw,
            tokens=tokens,
            x=x + 2,
            y=y + 2,
            font=font,
            fill=(0, 0, 0, 128),
            emoji_size=emoji_size,
            emoji_style=emoji_style,
            draw_emojis=False,
        )

    if stroke_color and stroke_width > 0:
        for dx in range(-stroke_width, stroke_width + 1):
            for dy in range(-stroke_width, stroke_width + 1):
                if dx == 0 and dy == 0:
                    continue
                _draw_caption_tokens(
                    base_image=base_image,
                    draw=draw,
                    tokens=tokens,
                    x=x + dx,
                    y=y + dy,
                    font=font,
                    fill=stroke_color,
                    emoji_size=emoji_size,
                    emoji_style=emoji_style,
                    draw_emojis=False,
                )

    _draw_caption_tokens(
        base_image=base_image,
        draw=draw,
        tokens=tokens,
        x=x,
        y=y,
        font=font,
        fill=text_color,
        emoji_size=emoji_size,
        emoji_style=emoji_style,
        draw_emojis=True,
    )


# ==================== FACE DETECTION ====================

# Global face detector (lazy loaded)
_face_detector = None


def get_face_detector():
    """Lazy load MediaPipe face detector."""
    global _face_detector
    if _face_detector is None and MEDIAPIPE_AVAILABLE and mp is not None:
        mp_solutions = getattr(mp, "solutions", None)
        mp_face_detection = (
            getattr(mp_solutions, "face_detection", None) if mp_solutions else None
        )
        if mp_face_detection is None:
            return _face_detector
        _face_detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 0=short range (2m), 1=full range (5m)
            min_detection_confidence=0.5,
        )
    return _face_detector


def detect_faces(image: Image.Image) -> List[Dict[str, Any]]:
    """
    Detect faces in an image using MediaPipe.

    Returns list of face bounding boxes:
    [
        {
            'x': normalized x (0-1),
            'y': normalized y (0-1),
            'width': normalized width (0-1),
            'height': normalized height (0-1),
            'confidence': 0-1
        }
    ]
    """
    if not MEDIAPIPE_AVAILABLE:
        return []

    detector = get_face_detector()
    if detector is None:
        return []

    # Convert PIL to RGB numpy array
    img_rgb = image.convert("RGB")
    img_np = np.array(img_rgb)

    # Process with MediaPipe
    results = detector.process(img_np)

    faces = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            faces.append(
                {
                    "x": bbox.xmin,
                    "y": bbox.ymin,
                    "width": bbox.width,
                    "height": bbox.height,
                    "confidence": detection.score[0] if detection.score else 0.5,
                }
            )

    return faces


def get_face_safe_position(
    faces: List[Dict[str, Any]],
    caption_height: int,
    caption_width: int,
    image_width: int,
    image_height: int,
    preferred_position: str,
    preferred_alignment: str,
    config: Dict[str, Any],
) -> Tuple[str, float, str]:
    """
    Calculate a caption position and alignment that avoids faces.
    Similar to original Captioner app logic.

    Args:
        faces: List of detected faces (normalized coords)
        caption_height: Height of caption block in pixels
        caption_width: Width of caption block in pixels
        image_width: Image width
        image_height: Image height
        preferred_position: User's preferred position ('top', 'center', 'bottom', 'custom')
        preferred_alignment: User's preferred alignment ('left', 'center', 'right')
        config: Full config dict

    Returns:
        Tuple of (final_position, y_ratio, alignment) where y_ratio is 0-1
    """
    if not faces:
        # No faces, use preferred position and alignment
        return (
            preferred_position,
            get_y_ratio_for_position(preferred_position, config),
            preferred_alignment,
        )

    # Safety margin around faces (30% of face size, similar to original)
    FACE_MARGIN = 0.30

    # Calculate caption areas for each position
    caption_ratio = caption_height / image_height
    caption_width_ratio = caption_width / image_width

    # Define Y zones (positions as ratios, with safe margins from edges)
    y_zones = {
        "top": 0.15,  # 15% from top (safe margin)
        "center": 0.50,  # True center
        "bottom": 0.82,  # 18% from bottom (safe margin)
    }

    # Define X positions for each alignment
    def get_x_bounds(alignment: str) -> Tuple[float, float]:
        """Get left and right of caption as ratios (0-1) for alignment."""
        if alignment == "left":
            return 0.05, 0.05 + caption_width_ratio
        elif alignment == "right":
            return 0.95 - caption_width_ratio, 0.95
        else:  # center
            half_width = caption_width_ratio / 2
            return 0.5 - half_width, 0.5 + half_width

    def get_y_bounds(y_center_ratio: float) -> Tuple[float, float]:
        """Get top and bottom of caption as ratios (0-1)."""
        half_height = caption_ratio / 2
        return max(0, y_center_ratio - half_height), min(
            1, y_center_ratio + half_height
        )

    def overlaps_face(y_ratio: float, alignment: str) -> bool:
        """Check if caption at this position/alignment overlaps any face."""
        caption_top, caption_bottom = get_y_bounds(y_ratio)
        caption_left, caption_right = get_x_bounds(alignment)

        for face in faces:
            # Expand face rect with safety margin
            face_left = face["x"] - face["width"] * FACE_MARGIN
            face_right = face["x"] + face["width"] * (1 + FACE_MARGIN)
            face_top = face["y"] - face["height"] * FACE_MARGIN
            face_bottom = face["y"] + face["height"] * (1 + FACE_MARGIN)

            # Check 2D overlap
            horizontal_overlap = caption_left < face_right and caption_right > face_left
            vertical_overlap = caption_top < face_bottom and caption_bottom > face_top

            if horizontal_overlap and vertical_overlap:
                return True

        return False

    # Try preferred position and alignment first (default to center)
    y_ratio = y_zones.get(preferred_position, 0.50)
    if preferred_position == "custom":
        y_ratio = config.get("customY", config.get("positionY", 0.5))

    if not overlaps_face(y_ratio, preferred_alignment):
        return preferred_position, y_ratio, preferred_alignment

    # Try different alignments at preferred Y
    for alt_align in ["center", "left", "right"]:
        if not overlaps_face(y_ratio, alt_align):
            return preferred_position, y_ratio, alt_align

    # Try different Y positions with all alignments (prioritize center first)
    for y_pos in ["center", "bottom", "top"]:
        y = y_zones[y_pos]
        for alignment in ["center", "left", "right"]:
            if not overlaps_face(y, alignment):
                return y_pos, y, alignment

    # Scan for any safe spot (avoiding extreme edges for visibility)
    # Use safer Y values: keep minimum 8% from top/bottom edges
    for y in [0.80, 0.75, 0.70, 0.65, 0.35, 0.30, 0.25, 0.20]:
        for alignment in ["center", "left", "right"]:
            if not overlaps_face(y, alignment):
                return "custom", y, alignment

    # Last resort: try near edges but not at absolute edge
    for y in [0.88, 0.12]:
        for alignment in ["center", "left", "right"]:
            if not overlaps_face(y, alignment):
                return "custom", y, alignment

    # Fallback: put at safe bottom position, center
    return "bottom", 0.88, "center"


def get_y_ratio_for_position(position: str, config: Dict[str, Any]) -> float:
    """Get the Y ratio (0-1) for a named position. Default is center."""
    if position == "top":
        return 0.15  # 15% from top (safe margin)
    elif position == "center":
        return 0.50
    elif position == "bottom":
        return 0.82  # 18% from bottom (safe margin)
    elif position == "custom":
        return config.get("customY", config.get("positionY", 0.5))
    else:
        return 0.50  # Default to center


# ==================== CAPTION FORMAT PARSING ====================


def parse_caption_format(text: str, allow_title: bool = True) -> Dict[str, Any]:
    """
    Parse special caption format:
    - ##Title## at the start becomes italic title with larger font (if enabled)
    - &&& becomes newline
    - Straight quotes converted to curly quotes

    Example: "##BIG TITLE##This is body&&&with line break"
    Returns: {'title': 'BIG TITLE', 'body': 'This is body\nwith line break'}
    """
    result = {"title": None, "body": ""}

    if not text:
        return result

    # Replace straight quotes with curly quotes
    text = replace_quotes(text)

    # Extract ##Title## if present at start (only when explicitly enabled)
    if allow_title:
        title_match = re.match(r"^##(.+?)##\s*", text)
        if title_match:
            result["title"] = title_match.group(1)
            text = text[title_match.end() :]

    # Replace &&& with newlines
    result["body"] = text.replace("&&&", "\n")

    return result


def get_caption_for_index(config: Dict[str, Any], index: int) -> str:
    """
    Get caption text for a specific image index.
    In batch mode, cycles through captions array.
    In single mode, returns the text.
    """
    caption_mode = config.get("captionMode", "single")

    if caption_mode == "batch":
        captions = config.get("captions", [])
        if captions:
            return captions[index % len(captions)]
        return config.get("text", "")
    else:
        return config.get("text", "")


def process_single_video_caption_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel video captioning.
    Designed to be called from ThreadPoolExecutor.

    Args (tuple):
        input_path: Path to input video
        output_path: Path to output video
        config: Caption configuration dict
        video_index: Index of video in batch
        gpu_id: GPU device ID for NVENC encoding (0, 1, 2, etc.)
    """
    input_path, output_path, config, video_index, gpu_id = args

    try:
        worker_config = dict(config)
        worker_config["gpuId"] = gpu_id

        result = process_video_caption(
            input_path=input_path,
            output_path=output_path,
            config=worker_config,
            report_progress=lambda _progress, _message: None,
            image_index=video_index,
        )

        return {
            "status": "completed",
            "index": video_index,
            "output_path": output_path,
            "duration": result.get("duration", 0),
            "gpu_id": gpu_id,
        }

    except Exception as e:
        return {
            "status": "failed",
            "index": video_index,
            "gpu_id": gpu_id,
            "error": str(e),
        }


def process_videos_parallel_caption(
    video_paths: List[str],
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    max_parallel: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Process multiple videos with captions in parallel using multiple NVENC sessions.

    Uses ThreadPoolExecutor (not ProcessPoolExecutor) because FFmpeg subprocesses
    already handle the GPU work - no need for separate Python processes.

    Implements round-robin GPU assignment to distribute work across all available GPUs.
    """
    if not video_paths:
        return {"error": "No videos to process"}

    # Detect all GPUs and calculate total parallel capacity
    gpu_info = get_gpu_info()
    gpu_count = gpu_info.get("gpu_count", 1)
    gpus = gpu_info.get("gpus", [{"index": 0, "nvenc_sessions": 2}])
    total_nvenc_sessions = gpu_info.get("total_nvenc_sessions", 2)

    if max_parallel is None:
        max_parallel = total_nvenc_sessions

    # Log GPU configuration
    gpu_names = [g.get("name", "Unknown") for g in gpus]
    print(f"[Parallel Caption] Detected {gpu_count} GPU(s): {', '.join(gpu_names)}")
    print(
        f"[Parallel Caption] Total NVENC sessions available: {total_nvenc_sessions}, using: {max_parallel}"
    )

    # Build work items with round-robin GPU assignment
    # Each video gets assigned to a GPU in round-robin fashion
    work_items = []
    for i, video_path in enumerate(video_paths):
        basename = os.path.basename(video_path)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name}_captioned{ext}")

        # Round-robin GPU assignment based on video index
        gpu_id = gpus[i % gpu_count]["index"]
        work_items.append((video_path, output_path, config, i, gpu_id))

    total = len(work_items)
    completed = 0
    failed = 0
    results = []

    def report_progress(msg=""):
        if progress_callback:
            progress_callback(completed / total if total > 0 else 0, msg)

    report_progress(
        f"Processing {total} videos across {gpu_count} GPU(s) with {max_parallel} parallel sessions..."
    )

    # Use ThreadPoolExecutor - FFmpeg handles GPU work in subprocesses
    # This is more efficient than ProcessPoolExecutor for I/O-bound subprocess calls
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_item = {
            executor.submit(process_single_video_caption_worker, item): item
            for item in work_items
        }

        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)
            if result["status"] == "completed":
                completed += 1
            else:
                failed += 1
            gpu_used = result.get("gpu_id", 0)
            report_progress(
                f"Completed {completed}/{total} videos ({failed} failed) [Last: GPU {gpu_used}]"
            )

    return {
        "status": "completed",
        "total": total,
        "completed": completed,
        "failed": failed,
        "results": results,
        "parallel_sessions": max_parallel,
        "gpu_count": gpu_count,
        "gpus_used": gpu_names,
    }


def process_captioner(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Process image or video with text captions.

    NEW: If input is a ZIP with multiple videos, processes them in parallel
    using multiple NVENC sessions (optimized for datacenter GPUs like A5000/A6000).

    Args:
        input_path: Path to input file
        output_path: Path to output file
        config: Caption configuration
        progress_callback: Optional callback(progress: 0-1, message: str)

    Returns:
        Dict with processing results
    """
    import shutil

    # Debug: Print raw config BEFORE normalization
    print(f"[Captioner] RAW config keys: {list(config.keys())}")
    print(f"[Captioner] RAW text: {safe_debug_repr(config.get('text', 'NOT FOUND'))}")
    print(
        f"[Captioner] RAW captions: {safe_debug_repr(config.get('captions', 'NOT FOUND'))}"
    )
    print(
        f"[Captioner] RAW caption_mode: {safe_debug_repr(config.get('caption_mode', 'NOT FOUND'))}"
    )
    print(
        f"[Captioner] RAW captionMode: {safe_debug_repr(config.get('captionMode', 'NOT FOUND'))}"
    )

    # Normalize config keys (snake_case -> camelCase)
    config = normalize_config(config)

    # Debug: Print config AFTER normalization
    print(f"[Captioner] NORMALIZED config keys: {list(config.keys())}")
    print(
        f"[Captioner] NORMALIZED text: {safe_debug_repr(config.get('text', 'NOT FOUND'))}"
    )
    print(
        f"[Captioner] NORMALIZED captions: {safe_debug_repr(config.get('captions', 'NOT FOUND'))}"
    )
    print(
        f"[Captioner] NORMALIZED captionMode: {safe_debug_repr(config.get('captionMode', 'NOT FOUND'))}"
    )

    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    # Detect file type
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS
    is_zip = ext == ".zip"

    # Get image index for batch mode (passed from workflow execution)
    image_index = config.get("imageIndex", 0)

    # Debug logging for batch mode troubleshooting
    caption_mode = config.get("captionMode", "single")
    captions = config.get("captions", [])
    text = config.get("text", "")
    print(
        f"[Captioner] Mode: {caption_mode}, imageIndex: {image_index}, captions count: {len(captions)}, text: {safe_debug_repr(text[:50] if text else '')}"
    )

    # Debug: Show actual caption that will be used
    actual_caption = get_caption_for_index(config, image_index)
    print(
        f"[Captioner] ACTUAL caption to render: {safe_debug_repr(actual_caption[:100] if actual_caption else 'EMPTY')}"
    )

    if caption_mode == "batch" and captions:
        print(
            f"[Captioner] Using caption {image_index % len(captions)}: {safe_debug_repr(captions[image_index % len(captions)][:50])}..."
        )

    report_progress(0.05, "Analyzing file...")

    # NEW: Check if input is a ZIP with videos (batch video mode)
    if is_zip:
        report_progress(0.08, "Checking ZIP contents...")
        temp_dir = tempfile.mkdtemp(prefix="captioner_batch_")
        video_paths: List[str] = []

        try:
            video_paths = extract_videos_from_zip(input_path, temp_dir)

            if video_paths:
                report_progress(
                    0.1,
                    f"Found {len(video_paths)} videos, starting parallel processing...",
                )

                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(output_dir, exist_ok=True)

                result = process_videos_parallel_caption(
                    video_paths, output_dir, config, progress_callback=progress_callback
                )

                if result.get("error"):
                    return result

                report_progress(0.95, "Creating output ZIP...")

                with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as zf:
                    for r in result.get("results", []):
                        if r.get("status") == "completed" and r.get("output_path"):
                            if os.path.exists(r["output_path"]):
                                zf.write(
                                    r["output_path"], os.path.basename(r["output_path"])
                                )

                report_progress(1.0, "Complete")

                return {
                    "status": "completed",
                    "mode": "parallel_video_batch",
                    "videos_processed": result.get("completed", 0),
                    "videos_failed": result.get("failed", 0),
                    "parallel_sessions": result.get("parallel_sessions", 1),
                    "output_size": os.path.getsize(output_path)
                    if os.path.exists(output_path)
                    else 0,
                }

        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            if video_paths:
                shutil.rmtree(temp_dir, ignore_errors=True)

    if is_video:
        return process_video_caption(
            input_path, output_path, config, report_progress, image_index
        )
    else:
        return process_image_caption(
            input_path, output_path, config, report_progress, image_index
        )


def process_image_caption(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None],
    image_index: int = 0,
) -> Dict[str, Any]:
    """
    Add caption to a single image.
    Supports ##Title## format and batch mode with cycling captions.
    Supports face detection to avoid placing captions over faces.

    Args:
        input_path: Path to input image
        output_path: Path to output image
        config: Caption configuration
        report_progress: Progress callback
        image_index: Index for batch mode caption selection (cycles through captions array)
    """

    report_progress(0.1, "Loading image...")

    img = Image.open(input_path).convert("RGBA")
    width, height = img.size

    # Get the actual text that will be rendered (for logging)
    caption_text = get_caption_for_index(config, image_index)
    parsed = parse_caption_format(
        caption_text, allow_title=bool(config.get("enableTitleSyntax", False))
    )

    # Face detection (if enabled)
    faces = []
    avoid_faces = config.get("avoidFaces", False)
    if avoid_faces and MEDIAPIPE_AVAILABLE:
        report_progress(0.2, "Detecting faces...")
        faces = detect_faces(img)
        if faces:
            report_progress(0.25, f"Found {len(faces)} face(s)")

    report_progress(0.3, "Rendering caption...")

    # Create caption overlay with batch index support and face avoidance
    overlay = create_caption_overlay(width, height, config, image_index, faces)

    # Composite
    img = Image.alpha_composite(img, overlay)

    report_progress(0.7, "Saving output...")

    # Convert to RGB for JPEG output
    output_ext = os.path.splitext(output_path)[1].lower()
    if output_ext in [".jpg", ".jpeg"]:
        img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=95)
    elif output_ext == ".png":
        img.save(output_path, "PNG")
    else:
        img = img.convert("RGB")
        img.save(output_path, "JPEG", quality=95)

    report_progress(1.0, "Complete")

    # Determine final position (may have been adjusted for faces)
    final_position = config.get("position", "bottom")
    position_adjusted = False
    if faces and avoid_faces:
        position_adjusted = True  # Actual adjustment happens in create_caption_overlay

    return {
        "text": caption_text,
        "title": parsed["title"],
        "body": parsed["body"],
        "position": final_position,
        "resolution": f"{width}x{height}",
        "captionMode": config.get("captionMode", "single"),
        "imageIndex": image_index,
        "facesDetected": len(faces),
        "positionAdjusted": position_adjusted,
    }


def process_video_caption(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None],
    image_index: int = 0,
) -> Dict[str, Any]:
    """
    Add caption to video using FFmpeg drawtext filter.

    Supports GPU selection via config['gpuId'] parameter for multi-GPU systems.
    Supports batch mode: in batch mode, cycles through captions array using image_index.
    """

    report_progress(0.1, "Analyzing video...")

    video_info = get_video_info(input_path)
    width = int(video_info.get("width", 1920))
    height = int(video_info.get("height", 1080))
    duration = float(video_info.get("duration", 0))

    report_progress(0.2, "Rendering caption overlay...")
    overlay_image = create_caption_overlay(width, height, config, image_index, [])

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as overlay_file:
        overlay_path = overlay_file.name

    overlay_image.save(overlay_path, "PNG")

    text = get_caption_for_index(config, image_index)
    position = config.get("position", "bottom")
    animation = str(config.get("animation", "none")).lower()
    start_time = float(config.get("startTime", 0) or 0)
    raw_end_time = config.get("endTime")
    end_time = float(raw_end_time) if raw_end_time not in (None, "") else None
    gpu_id = int(config.get("gpuId", config.get("gpu_id", 0)) or 0)

    filter_parts = ["[1:v]format=rgba[caption_base]"]
    overlay_stream = "[caption_base]"

    if animation == "fade":
        fade_duration = 0.35
        filter_parts.append(
            f"{overlay_stream}fade=t=in:st={start_time:.3f}:d={fade_duration:.3f}:alpha=1[caption_animated]"
        )
        overlay_stream = "[caption_animated]"

    overlay_filter = f"[0:v]{overlay_stream}overlay=0:0:shortest=1"
    if start_time > 0 or end_time is not None:
        final_end = end_time if end_time is not None else duration
        final_end = max(final_end, start_time + 0.01)
        overlay_filter += f":enable='between(t,{start_time:.3f},{final_end:.3f})'"
    overlay_filter += "[vout]"
    filter_parts.append(overlay_filter)

    filter_complex = ";".join(filter_parts)

    report_progress(0.3, f"Encoding with NVENC on GPU {gpu_id}...")

    def build_ffmpeg_cmd(use_nvenc: bool = True) -> list:
        """Build FFmpeg command with GPU or CPU encoding."""
        cmd = ["ffmpeg", "-y"]
        if use_nvenc:
            cmd.extend(["-hwaccel", "cuda", "-hwaccel_device", str(gpu_id)])

        cmd.extend(
            [
                "-i",
                input_path,
                "-loop",
                "1",
                "-i",
                overlay_path,
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "0:a?",
            ]
        )

        if use_nvenc:
            cmd.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-gpu",
                    str(gpu_id),
                    "-preset",
                    "p4",
                    "-b:v",
                    "5000k",
                    "-maxrate",
                    "7500k",
                    "-bufsize",
                    "10000k",
                    "-profile:v",
                    "high",
                ]
            )
        else:
            cmd.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-profile:v",
                    "high",
                ]
            )

        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-shortest", output_path])
        return cmd

    def run_ffmpeg_with_progress(cmd: list, mode_name: str) -> tuple:
        """Run FFmpeg and capture stderr properly. Returns (success, stderr_output)."""
        stderr_lines = []

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

        if process.stderr is None:
            process.kill()
            return False, "Failed to capture FFmpeg stderr output"

        # Read stderr line by line for progress
        try:
            while True:
                line = process.stderr.readline()
                if not line:
                    if process.poll() is not None:
                        break
                    continue

                stderr_lines.append(line)
                # Keep last 50 lines to avoid memory issues
                if len(stderr_lines) > 50:
                    stderr_lines.pop(0)

                if "time=" in line:
                    try:
                        time_str = line.split("time=")[1].split()[0]
                        h, m, s = time_str.split(":")
                        current_time = int(h) * 3600 + int(m) * 60 + float(s)
                        if duration > 0:
                            progress = min(0.3 + (current_time / duration) * 0.65, 0.95)
                            report_progress(
                                progress,
                                f"Encoding ({mode_name})... {int(current_time)}s / {int(duration)}s",
                            )
                    except:
                        pass

            # Wait for process to complete with timeout
            try:
                process.wait(timeout=600)  # 10 minute timeout
            except subprocess.TimeoutExpired:
                process.kill()
                return False, "FFmpeg process timed out after 600 seconds"

        except Exception as e:
            process.kill()
            return False, f"Error during encoding: {str(e)}"

        stderr_output = "".join(stderr_lines)
        return process.returncode == 0, stderr_output

    try:
        cmd = build_ffmpeg_cmd(use_nvenc=True)
        success, stderr = run_ffmpeg_with_progress(cmd, "NVENC")

        if not success:
            print(
                f"[Captioner] NVENC failed, trying CPU fallback. Error: {stderr[-500:]}"
            )
            report_progress(0.3, "NVENC failed, trying CPU encoding...")
            cmd = build_ffmpeg_cmd(use_nvenc=False)
            success, stderr = run_ffmpeg_with_progress(cmd, "CPU")

        if not success:
            raise RuntimeError(f"FFmpeg failed with all encoders: {stderr[-1000:]}")

        report_progress(1.0, "Complete")

        return {
            "text": text,
            "position": position,
            "animation": animation,
            "resolution": f"{width}x{height}",
            "duration": duration,
            "gpuId": gpu_id,
        }
    finally:
        try:
            os.remove(overlay_path)
        except Exception:
            pass


def create_caption_overlay(
    width: int,
    height: int,
    config: Dict[str, Any],
    image_index: int = 0,
    faces: Optional[List[Dict[str, Any]]] = None,
) -> Image.Image:
    """
    Create a caption overlay for images.
    Supports ##Title## format (italic title) and &&& (line breaks).
    Supports batch mode with cycling captions.
    Supports face detection to avoid placing captions over faces.
    Scaling formula: scaled_size = config_size * (image_width / 1080)

    Args:
        width: Image width
        height: Image height
        config: Caption configuration dict
        image_index: Index for batch mode caption selection
        faces: List of detected faces (normalized coordinates) for smart positioning
    """
    if faces is None:
        faces = []

    show_background = config.get("showBackground", False)
    enable_title_syntax = bool(config.get("enableTitleSyntax", False))

    raw_text = get_caption_for_index(config, image_index)
    parsed = parse_caption_format(raw_text, allow_title=enable_title_syntax)
    title_text = parsed["title"]
    body_text = parsed["body"]

    base_width = 1080
    scale_factor = width / base_width

    center_every_enabled = config.get("centerEveryEnabled", False)
    center_every_n = config.get("centerEveryN", 7)
    force_center = center_every_enabled and (image_index + 1) % center_every_n == 0

    position = "center" if force_center else config.get("position", "center")
    alignment = config.get("alignment", "center")
    text_width_ratio = float(config.get("textWidthRatio", 0.85) or 0.85)
    if show_background:
        text_width_ratio = max(0.75, min(0.95, text_width_ratio))
    max_width = int(width * text_width_ratio)

    emoji_style = str(config.get("emojiStyle", "ios")).lower()

    base_font_size = config.get("fontSize", 47)
    base_title_size = config.get("titleFontSize", 75)
    base_stroke_width = config.get("strokeWidth", 0 if show_background else 3.5)

    font_size = max(12, int(base_font_size * scale_factor))
    title_font_size = max(14, int(base_title_size * scale_factor))
    stroke_width = max(0, int(base_stroke_width * scale_factor))

    if show_background:
        text_color = "#000000"
        bg_color = "#FFFFFF"
        bg_opacity = 1.0
        stroke_width = 0
        shadow = False
    else:
        text_color = config.get("textColor", config.get("color", "#FFFFFF"))
        bg_color = None
        bg_opacity_raw = config.get("backgroundOpacity", 80)
        if isinstance(bg_opacity_raw, (int, float)) and bg_opacity_raw <= 1:
            bg_opacity = float(bg_opacity_raw)
        else:
            bg_opacity = float(bg_opacity_raw) / 100
        shadow = config.get("shadow", False)

    stroke_color = config.get("strokeColor", "#000000")
    block_spacing = max(4, int(config.get("blockSpacing", 40) * scale_factor))
    if show_background:
        line_spacing = max(0, int(font_size * 0.08))
    else:
        line_spacing = max(1, int(config.get("lineSpacing", 20) * scale_factor * 0.5))

    random_tilt = config.get("randomTilt", False)
    tilt_angle = 0
    if random_tilt:
        tilt_min = config.get("tiltRangeMin", -5)
        tilt_max = config.get("tiltRangeMax", 5)
        tilt_angle = random.uniform(tilt_min, tilt_max)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_name = config.get("font", config.get("fontFamily", "TikTokBold.otf"))
    font_path = FONT_MAP.get(font_name, FONT_MAP["default-bold"])
    italic_font_name = font_name.replace("Bold", "Italic").replace("bold", "italic")
    if "TikTok" in font_name or "tiktok" in font_name:
        italic_font_path = FONT_MAP.get(
            "LightItalic.otf", FONT_MAP.get("tiktok-italic", FONT_MAP["default-italic"])
        )
    else:
        italic_font_path = FONT_MAP.get(italic_font_name, FONT_MAP["default-italic"])

    try:
        body_font = ImageFont.truetype(font_path, font_size)
    except Exception:
        body_font = ImageFont.load_default()

    try:
        title_font = ImageFont.truetype(italic_font_path, title_font_size)
    except Exception:
        title_font = body_font

    content_blocks: List[Dict[str, Any]] = []
    total_height = 0

    if title_text:
        wrapped_title = wrap_text_mixed(title_text, title_font, max_width)
        title_lines = wrapped_title.split("\n") if wrapped_title else [""]
        title_line_height = max(
            get_font_line_height(title_font), estimate_emoji_size(title_font)
        )
        title_width = max(
            measure_text_width(line if line else " ", title_font)
            for line in title_lines
        )
        title_height = (
            len(title_lines) * title_line_height
            + max(0, len(title_lines) - 1) * line_spacing
        )
        content_blocks.append(
            {
                "type": "title",
                "lines": title_lines,
                "font": title_font,
                "width": int(title_width),
                "height": int(title_height),
                "lineHeight": int(title_line_height),
            }
        )
        total_height += int(title_height)

    if body_text:
        wrapped_body = wrap_text_mixed(body_text, body_font, max_width)
        body_lines = wrapped_body.split("\n") if wrapped_body else [""]
        body_line_height = max(
            get_font_line_height(body_font), estimate_emoji_size(body_font)
        )
        body_width = max(
            measure_text_width(line if line else " ", body_font) for line in body_lines
        )
        body_height = (
            len(body_lines) * body_line_height
            + max(0, len(body_lines) - 1) * line_spacing
        )
        content_blocks.append(
            {
                "type": "body",
                "lines": body_lines,
                "font": body_font,
                "width": int(body_width),
                "height": int(body_height),
                "lineHeight": int(body_line_height),
            }
        )
        total_height += int(body_height)

    if not content_blocks:
        return overlay

    if len(content_blocks) > 1:
        total_height += block_spacing

    max_content_width = max(block["width"] for block in content_blocks)

    avoid_faces = config.get("avoidFaces", False)
    position_x = float(config.get("positionX", config.get("customX", 0.5)))
    position_y = float(config.get("positionY", config.get("customY", 0.82)))

    if avoid_faces and faces:
        adjusted_position, adjusted_y, adjusted_alignment = get_face_safe_position(
            faces=faces,
            caption_height=int(total_height),
            caption_width=int(max_content_width),
            image_width=width,
            image_height=height,
            preferred_position=position,
            preferred_alignment=alignment,
            config=config,
        )
        position = adjusted_position
        position_y = adjusted_y
        alignment = adjusted_alignment

    if position == "top":
        base_y = int(height * 0.15) - total_height // 2
    elif position == "center":
        base_y = (height - total_height) // 2
    elif position == "bottom":
        base_y = int(height * 0.82) - total_height // 2
    elif position == "custom":
        base_y = int(position_y * height) - total_height // 2
    else:
        base_y = (height - total_height) // 2

    if position == "custom":
        anchor_x = int(position_x * width)
        if alignment == "left":
            base_x = anchor_x
        elif alignment == "right":
            base_x = anchor_x - max_content_width
        else:
            base_x = anchor_x - max_content_width // 2
    else:
        if alignment == "left":
            base_x = 20
        elif alignment == "right":
            base_x = width - max_content_width - 20
        else:
            base_x = (width - max_content_width) // 2

    if bg_color:
        bg_pad_x = max(8, int(font_size * 0.32))
        bg_pad_y = max(3, int(font_size * 0.09))
        bubble_height = max(1, get_font_line_height(body_font) + (bg_pad_y * 2))
        bg_radius = max(4, int(bubble_height * 0.14))
        bg_join = 0
    else:
        bg_pad_x = 0
        bg_pad_y = 0
        bg_radius = 0
        bg_join = 0
    effective_width = max_content_width + (bg_pad_x * 2)

    margin_x = max(20, int(width * 0.05))
    margin_y = max(30, int(height * 0.05))
    base_x = max(margin_x, min(base_x, width - effective_width - margin_x))
    base_y = max(margin_y, min(base_y, height - total_height - margin_y))

    text_r, text_g, text_b = hex_to_rgb(text_color)
    stroke_rgba: Optional[Tuple[int, int, int, int]] = None
    if stroke_color and stroke_width > 0:
        stroke_r, stroke_g, stroke_b = hex_to_rgb(stroke_color)
        stroke_rgba = (stroke_r, stroke_g, stroke_b, 255)

    if bg_color:
        bg_r, bg_g, bg_b = hex_to_rgb(bg_color)
        bg_alpha = int(max(0.0, min(1.0, bg_opacity)) * 255)
    else:
        bg_r = bg_g = bg_b = bg_alpha = 0

    line_layouts: List[Dict[str, Any]] = []
    current_y = base_y

    for block_index, block in enumerate(content_blocks):
        block_font = block["font"]
        line_height = int(block["lineHeight"])
        line_count = len(block["lines"])

        for line_index, line in enumerate(block["lines"]):
            line_text = line if line else " "
            line_width = int(measure_text_width(line_text, block_font))

            if alignment == "left":
                line_x = base_x
            elif alignment == "right":
                line_x = base_x + max_content_width - line_width
            else:
                line_x = base_x + (max_content_width - line_width) // 2

            line_layouts.append(
                {
                    "text": line_text,
                    "x": line_x,
                    "y": current_y,
                    "font": block_font,
                    "lineWidth": line_width,
                    "lineHeight": line_height,
                    "blockIndex": block_index,
                    "lineIndex": line_index,
                    "lineCount": line_count,
                }
            )

            current_y += line_height
            if line_index < line_count - 1:
                current_y += line_spacing

        if block_index < len(content_blocks) - 1:
            current_y += block_spacing

    # Draw all background bubbles first so multi-line joins never cover previous text.
    if bg_color:
        aa_factor = 2 if font_size >= 28 else 1
        bg_mask = Image.new("L", (width * aa_factor, height * aa_factor), 0)
        bg_draw = ImageDraw.Draw(bg_mask)

        def aa(value: int) -> int:
            return int(round(value * aa_factor))

        for line in line_layouts:
            bg_top = line["y"] - bg_pad_y
            bg_bottom = line["y"] + line["lineHeight"] + bg_pad_y

            if bg_join > 0:
                if line["lineIndex"] > 0:
                    bg_top -= bg_join
                if line["lineIndex"] < line["lineCount"] - 1:
                    bg_bottom += bg_join

            bubble_left = line["x"] - bg_pad_x
            bubble_right = line["x"] + line["lineWidth"] + bg_pad_x

            bg_draw.rounded_rectangle(
                [
                    aa(bubble_left),
                    aa(bg_top),
                    aa(bubble_right),
                    aa(bg_bottom),
                ],
                radius=max(1, aa(bg_radius)),
                fill=255,
            )

        # Vector-like contour smoothing on high-res mask to avoid jagged notches.
        if aa_factor > 1:
            contour_blur = max(0.6, aa_factor * 0.55)
            bg_mask = bg_mask.filter(ImageFilter.GaussianBlur(contour_blur))
            threshold_table = [0] * 140 + [255] * (256 - 140)
            bg_mask = bg_mask.point(threshold_table)

        if aa_factor > 1:
            bg_mask = bg_mask.resize((width, height), LANCZOS_RESAMPLE)

        overlay.paste((bg_r, bg_g, bg_b, bg_alpha), (0, 0), bg_mask)

    # Draw text after backgrounds to prevent clipping between joined lines.
    for line in line_layouts:
        draw_mixed_text_line(
            base_image=overlay,
            draw=draw,
            text=line["text"],
            x=line["x"],
            y=line["y"],
            font=line["font"],
            text_color=(text_r, text_g, text_b, 255),
            emoji_style=emoji_style,
            stroke_color=stroke_rgba,
            stroke_width=stroke_width,
            shadow=shadow,
        )

    # Apply random tilt if enabled
    if tilt_angle != 0:
        # Rotate the overlay around the center of the caption
        # First, find the caption bounding box
        caption_center_x = base_x + max_content_width // 2
        caption_center_y = base_y + total_height // 2

        # Create a larger canvas to accommodate rotation
        diagonal = int((width**2 + height**2) ** 0.5)
        rotated = Image.new("RGBA", (diagonal, diagonal), (0, 0, 0, 0))

        # Calculate offset to center the overlay in the rotated canvas
        offset_x = (diagonal - width) // 2
        offset_y = (diagonal - height) // 2

        rotated.paste(overlay, (offset_x, offset_y))

        # Rotate around center
        rotated = rotated.rotate(
            tilt_angle,
            center=(diagonal // 2, diagonal // 2),
            expand=False,
            fillcolor=(0, 0, 0, 0),
        )

        # Crop back to original size
        crop_x = (diagonal - width) // 2
        crop_y = (diagonal - height) // 2
        overlay = rotated.crop((crop_x, crop_y, crop_x + width, crop_y + height))

    return overlay


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    """Wrap text to fit within max width."""
    lines = []
    for line in text.split("\n"):
        if font.getlength(line) <= max_width:
            lines.append(line)
        else:
            words = line.split()
            current_line = []
            for word in words:
                test_line = " ".join(current_line + [word])
                if font.getlength(test_line) <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(" ".join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(" ".join(current_line))

    return "\n".join(lines)


def calculate_position(
    position: str, width: int, height: int, font_size: int, config: Dict[str, Any]
) -> tuple:
    """Calculate FFmpeg position expressions."""
    if position == "top":
        return "(w-text_w)/2", f"{int(height * 0.15)}-text_h/2"
    elif position == "center":
        return "(w-text_w)/2", "(h-text_h)/2"
    elif position == "bottom":
        return "(w-text_w)/2", f"{int(height * 0.82)}-text_h/2"
    elif position == "custom":
        pos_x = float(config.get("positionX", config.get("customX", 0.5)) or 0.5)
        pos_y = float(config.get("positionY", config.get("customY", 0.5)) or 0.5)
        x = int(pos_x * width)
        y = int(pos_y * height)
        return f"{x}-text_w/2", f"{y}-text_h/2"
    else:
        return "(w-text_w)/2", f"{int(height * 0.82)}-text_h/2"


def calculate_position_pixels(
    position: str,
    width: int,
    height: int,
    text_width: int,
    text_height: int,
    config: Dict[str, Any],
) -> tuple:
    """Calculate pixel position for image captions."""
    padding = 20

    if position == "top":
        return (width - text_width) // 2, padding
    elif position == "center":
        return (width - text_width) // 2, (height - text_height) // 2
    elif position == "bottom":
        return (width - text_width) // 2, height - text_height - padding
    elif position == "custom":
        pos_x = float(config.get("positionX", config.get("customX", 0.5)) or 0.5)
        pos_y = float(config.get("positionY", config.get("customY", 0.5)) or 0.5)
        x = int(pos_x * width) - text_width // 2
        y = int(pos_y * height) - text_height // 2
        return x, y
    else:
        return (width - text_width) // 2, height - text_height - padding


def escape_ffmpeg_text(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter."""
    # Escape single quotes, colons, and backslashes
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace("%", "%%")
    return text


def get_video_info(path: str) -> Dict[str, Any]:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)

        video_stream = next(
            (s for s in info.get("streams", []) if s.get("codec_type") == "video"), {}
        )

        return {
            "width": int(video_stream.get("width", 1920)),
            "height": int(video_stream.get("height", 1080)),
            "duration": float(info.get("format", {}).get("duration", 0)),
        }
    except:
        return {"width": 1920, "height": 1080, "duration": 0}


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c * 2 for c in hex_color])

    try:
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    except:
        return (255, 255, 255)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor.py input_file output_file")
        sys.exit(1)

    test_config = {
        "text": "Sample Caption Text",
        "position": "center",  # Default position is now center
        "fontSize": 48,
        "fontFamily": "Arial",
        "color": "#FFFFFF",
        "strokeColor": "#000000",
        "strokeWidth": 2,
        "shadow": True,
    }

    def progress(p, msg):
        print(f"[{int(p * 100)}%] {msg}")

    result = process_captioner(sys.argv[1], sys.argv[2], test_config, progress)
    print(f"Result: {result}")
