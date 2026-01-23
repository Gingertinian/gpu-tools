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
from PIL import Image, ImageDraw, ImageFont
import json
import textwrap
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Video extensions
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'}

# NVENC session limits by GPU type
NVENC_SESSION_LIMITS = {
    'consumer': 3,
    'datacenter': 10,
    'default': 2,
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
    datacenter_keywords = ['A100', 'A6000', 'A5000', 'A4000', 'A4500', 'A40', 'A30', 'A10',
                           'V100', 'T4', 'Quadro', 'Tesla', 'H100', 'L40', 'RTX 4090', 'RTX 6000']

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpus = []

            for line in lines:
                if not line.strip():
                    continue
                parts = line.split(', ', 1)
                if len(parts) >= 2:
                    gpu_index = int(parts[0].strip())
                    gpu_name = parts[1].strip()
                else:
                    gpu_index = len(gpus)
                    gpu_name = line.strip()

                is_datacenter = any(kw in gpu_name for kw in datacenter_keywords)
                gpu_type = 'datacenter' if is_datacenter else 'consumer'

                gpus.append({
                    'index': gpu_index,
                    'name': gpu_name,
                    'type': gpu_type,
                    'nvenc_sessions': NVENC_SESSION_LIMITS[gpu_type]
                })

            if gpus:
                total_sessions = sum(g['nvenc_sessions'] for g in gpus)
                return {
                    'gpu_count': len(gpus),
                    'gpus': gpus,
                    'total_nvenc_sessions': total_sessions,
                    # Backwards compatibility (first GPU)
                    'gpu_name': gpus[0]['name'],
                    'gpu_type': gpus[0]['type'],
                    'nvenc_sessions': gpus[0]['nvenc_sessions']
                }
    except Exception as e:
        print(f"[GPU Detection] Error: {e}")

    return {
        'gpu_count': 1,
        'gpus': [{'index': 0, 'name': 'Unknown', 'type': 'default', 'nvenc_sessions': NVENC_SESSION_LIMITS['default']}],
        'total_nvenc_sessions': NVENC_SESSION_LIMITS['default'],
        'gpu_name': 'Unknown',
        'gpu_type': 'default',
        'nvenc_sessions': NVENC_SESSION_LIMITS['default']
    }


def extract_videos_from_zip(zip_path: str, extract_dir: str) -> List[str]:
    """Extract video files from a ZIP archive."""
    video_paths = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
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
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_detection'):
        MEDIAPIPE_AVAILABLE = True
    else:
        print("[Captioner] MediaPipe installed but mp.solutions.face_detection not available")
        MEDIAPIPE_AVAILABLE = False
        mp = None
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    mp = None


# Font mapping (RunPod will have these installed)
# TikTok fonts are bundled in /app/fonts/
FONT_MAP = {
    'Arial': '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    'Arial Bold': '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    'Arial.ttf': '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
    'Impact': '/usr/share/fonts/truetype/msttcorefonts/Impact.ttf',
    'Impact.ttf': '/usr/share/fonts/truetype/msttcorefonts/Impact.ttf',
    'Roboto': '/usr/share/fonts/truetype/roboto/Roboto-Regular.ttf',
    'Roboto Bold': '/usr/share/fonts/truetype/roboto/Roboto-Bold.ttf',
    # TikTok fonts (bundled in Docker)
    'TikTokBold.otf': '/app/fonts/TikTokBold.otf',
    'LightItalic.otf': '/app/fonts/LightItalic.otf',
    'tiktok': '/app/fonts/TikTokBold.otf',
    'tiktok-bold': '/app/fonts/TikTokBold.otf',
    'tiktok-italic': '/app/fonts/LightItalic.otf',
    # Fallbacks
    'default': '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    'default-bold': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    'default-italic': '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf'
}


import re
import random


# ==================== CONFIG NORMALIZATION ====================

def snake_to_camel(s: str) -> str:
    """Convert snake_case to camelCase."""
    components = s.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])


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
        'caption_mode': 'captionMode',
        'image_index': 'imageIndex',
        'avoid_faces': 'avoidFaces',
        'font_size': 'fontSize',
        'title_font_size': 'titleFontSize',
        'stroke_width': 'strokeWidth',
        'text_color': 'textColor',
        'show_background': 'showBackground',
        'background_color': 'backgroundColor',
        'background_opacity': 'backgroundOpacity',
        'stroke_color': 'strokeColor',
        'text_width_ratio': 'textWidthRatio',
        'block_spacing': 'blockSpacing',
        'line_spacing': 'lineSpacing',
        'random_tilt': 'randomTilt',
        'tilt_range_min': 'tiltRangeMin',
        'tilt_range_max': 'tiltRangeMax',
        'position_x': 'positionX',
        'position_y': 'positionY',
        'custom_x': 'customX',
        'custom_y': 'customY',
        'center_every_enabled': 'centerEveryEnabled',
        'center_every_n': 'centerEveryN',
        'font_family': 'fontFamily',
        'font_weight': 'fontWeight',
        'start_time': 'startTime',
        'end_time': 'endTime',
        'max_width': 'maxWidth',
        'apply_position_for_all': 'applyPositionForAll',
    }

    for key, value in config.items():
        # Check if it's a known snake_case key
        if key in key_map:
            normalized[key_map[key]] = value
        elif '_' in key:
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
    res = ''
    for char in text:
        if char == '"':
            res += '"' if not in_double else '"'
            in_double = not in_double
        elif char == "'":
            res += ''' if not in_single else '''
            in_single = not in_single
        else:
            res += char
    return res


# ==================== FACE DETECTION ====================

# Global face detector (lazy loaded)
_face_detector = None


def get_face_detector():
    """Lazy load MediaPipe face detector."""
    global _face_detector
    if _face_detector is None and MEDIAPIPE_AVAILABLE:
        mp_face_detection = mp.solutions.face_detection
        _face_detector = mp_face_detection.FaceDetection(
            model_selection=1,  # 0=short range (2m), 1=full range (5m)
            min_detection_confidence=0.5
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
    img_rgb = image.convert('RGB')
    img_np = np.array(img_rgb)

    # Process with MediaPipe
    results = detector.process(img_np)

    faces = []
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            faces.append({
                'x': bbox.xmin,
                'y': bbox.ymin,
                'width': bbox.width,
                'height': bbox.height,
                'confidence': detection.score[0] if detection.score else 0.5
            })

    return faces


def get_face_safe_position(
    faces: List[Dict[str, Any]],
    caption_height: int,
    caption_width: int,
    image_width: int,
    image_height: int,
    preferred_position: str,
    preferred_alignment: str,
    config: Dict[str, Any]
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
        return preferred_position, get_y_ratio_for_position(preferred_position, config), preferred_alignment

    # Safety margin around faces (30% of face size, similar to original)
    FACE_MARGIN = 0.30

    # Calculate caption areas for each position
    caption_ratio = caption_height / image_height
    caption_width_ratio = caption_width / image_width

    # Define Y zones (positions as ratios, with safe margins from edges)
    y_zones = {
        'top': 0.15,     # 15% from top (safe margin)
        'center': 0.50,  # True center
        'bottom': 0.82,  # 18% from bottom (safe margin)
    }

    # Define X positions for each alignment
    def get_x_bounds(alignment: str) -> Tuple[float, float]:
        """Get left and right of caption as ratios (0-1) for alignment."""
        if alignment == 'left':
            return 0.05, 0.05 + caption_width_ratio
        elif alignment == 'right':
            return 0.95 - caption_width_ratio, 0.95
        else:  # center
            half_width = caption_width_ratio / 2
            return 0.5 - half_width, 0.5 + half_width

    def get_y_bounds(y_center_ratio: float) -> Tuple[float, float]:
        """Get top and bottom of caption as ratios (0-1)."""
        half_height = caption_ratio / 2
        return max(0, y_center_ratio - half_height), min(1, y_center_ratio + half_height)

    def overlaps_face(y_ratio: float, alignment: str) -> bool:
        """Check if caption at this position/alignment overlaps any face."""
        caption_top, caption_bottom = get_y_bounds(y_ratio)
        caption_left, caption_right = get_x_bounds(alignment)

        for face in faces:
            # Expand face rect with safety margin
            face_left = face['x'] - face['width'] * FACE_MARGIN
            face_right = face['x'] + face['width'] * (1 + FACE_MARGIN)
            face_top = face['y'] - face['height'] * FACE_MARGIN
            face_bottom = face['y'] + face['height'] * (1 + FACE_MARGIN)

            # Check 2D overlap
            horizontal_overlap = caption_left < face_right and caption_right > face_left
            vertical_overlap = caption_top < face_bottom and caption_bottom > face_top

            if horizontal_overlap and vertical_overlap:
                return True

        return False

    # Try preferred position and alignment first (default to center)
    y_ratio = y_zones.get(preferred_position, 0.50)
    if preferred_position == 'custom':
        y_ratio = config.get('customY', config.get('positionY', 0.5))

    if not overlaps_face(y_ratio, preferred_alignment):
        return preferred_position, y_ratio, preferred_alignment

    # Try different alignments at preferred Y
    for alt_align in ['center', 'left', 'right']:
        if not overlaps_face(y_ratio, alt_align):
            return preferred_position, y_ratio, alt_align

    # Try different Y positions with all alignments (prioritize center first)
    for y_pos in ['center', 'bottom', 'top']:
        y = y_zones[y_pos]
        for alignment in ['center', 'left', 'right']:
            if not overlaps_face(y, alignment):
                return y_pos, y, alignment

    # Scan for any safe spot (avoiding extreme edges for visibility)
    # Use safer Y values: keep minimum 8% from top/bottom edges
    for y in [0.80, 0.75, 0.70, 0.65, 0.35, 0.30, 0.25, 0.20]:
        for alignment in ['center', 'left', 'right']:
            if not overlaps_face(y, alignment):
                return 'custom', y, alignment

    # Last resort: try near edges but not at absolute edge
    for y in [0.88, 0.12]:
        for alignment in ['center', 'left', 'right']:
            if not overlaps_face(y, alignment):
                return 'custom', y, alignment

    # Fallback: put at safe bottom position, center
    return 'bottom', 0.88, 'center'


def get_y_ratio_for_position(position: str, config: Dict[str, Any]) -> float:
    """Get the Y ratio (0-1) for a named position. Default is center."""
    if position == 'top':
        return 0.15  # 15% from top (safe margin)
    elif position == 'center':
        return 0.50
    elif position == 'bottom':
        return 0.82  # 18% from bottom (safe margin)
    elif position == 'custom':
        return config.get('customY', config.get('positionY', 0.5))
    else:
        return 0.50  # Default to center


# ==================== CAPTION FORMAT PARSING ====================


def parse_caption_format(text: str) -> Dict[str, Any]:
    """
    Parse special caption format:
    - ##Title## at the start becomes italic title with larger font
    - &&& becomes newline
    - Straight quotes converted to curly quotes

    Example: "##BIG TITLE##This is body&&&with line break"
    Returns: {'title': 'BIG TITLE', 'body': 'This is body\nwith line break'}
    """
    result = {'title': None, 'body': ''}

    if not text:
        return result

    # Replace straight quotes with curly quotes
    text = replace_quotes(text)

    # Extract ##Title## if present at start
    title_match = re.match(r'^##(.+?)##\s*', text)
    if title_match:
        result['title'] = title_match.group(1)
        text = text[title_match.end():]

    # Replace &&& with newlines
    result['body'] = text.replace('&&&', '\n')

    return result


def get_caption_for_index(config: Dict[str, Any], index: int) -> str:
    """
    Get caption text for a specific image index.
    In batch mode, cycles through captions array.
    In single mode, returns the text.
    """
    caption_mode = config.get('captionMode', 'single')

    if caption_mode == 'batch':
        captions = config.get('captions', [])
        if captions:
            return captions[index % len(captions)]
        return config.get('text', '')
    else:
        return config.get('text', '')


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
    import subprocess
    import json

    try:
        # Get video info
        video_info_result = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-show_format', input_path],
            capture_output=True, text=True
        )
        video_info = json.loads(video_info_result.stdout) if video_info_result.returncode == 0 else {}
        video_stream = next((s for s in video_info.get('streams', []) if s.get('codec_type') == 'video'), {})
        width = int(video_stream.get('width', 1920))
        height = int(video_stream.get('height', 1080))
        duration = float(video_info.get('format', {}).get('duration', 0))

        # Get caption text - support batch mode with cycling
        text = get_caption_for_index(config, video_index)

        # Config values
        position = config.get('position', 'bottom')
        font_size = config.get('fontSize', 48)
        color = config.get('textColor', config.get('color', '#FFFFFF'))
        stroke_color = config.get('strokeColor', '#000000')
        stroke_width = config.get('strokeWidth', 2)
        shadow = config.get('shadow', False)

        # Get font path - use TikTok font by default
        font_name = config.get('font', 'TikTokBold.otf')
        font_path = FONT_MAP.get(font_name, FONT_MAP.get('default', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))

        # Position calculations - support custom position with normalized coords
        if position == 'top':
            y_expr = f"{int(height * 0.15)}"
        elif position == 'center':
            y_expr = '(h-text_h)/2'
        elif position == 'bottom':
            y_expr = f"{int(height * 0.85)}-text_h"
        elif position == 'custom':
            # Custom position uses normalized coords (0-1)
            pos_x = config.get('positionX', 0.5)
            pos_y = config.get('positionY', 0.5)
            y_expr = f"{int(height * pos_y)}-text_h/2"
        else:
            y_expr = f"{int(height * 0.85)}-text_h"

        # Escape text for FFmpeg (handle special characters)
        text_escaped = text.replace('\\', '\\\\').replace("'", "\\'").replace(':', '\\:').replace('%', '%%')
        # Handle ##Title## format - remove for video (simpler rendering)
        text_escaped = text_escaped.replace('##', '').replace('&&&', ' ')

        # Build drawtext filter with fontfile
        drawtext_parts = [
            f"text='{text_escaped}'",
            f"fontfile='{font_path}'",
            f"fontsize={font_size}",
            f"fontcolor={color}",
            f"x=(w-text_w)/2",
            f"y={y_expr}"
        ]

        # Add stroke/border
        if stroke_color and stroke_width > 0:
            drawtext_parts.append(f"borderw={stroke_width}")
            drawtext_parts.append(f"bordercolor={stroke_color}")

        # Add shadow
        if shadow:
            drawtext_parts.append("shadowcolor=black@0.5")
            drawtext_parts.append("shadowx=2")
            drawtext_parts.append("shadowy=2")

        drawtext = f"drawtext={':'.join(drawtext_parts)}"

        # FFmpeg command with GPU selection
        # -gpu flag specifies which GPU to use for h264_nvenc encoding
        cmd = [
            'ffmpeg', '-y',
            '-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id),
            '-i', input_path,
            '-vf', drawtext,
            '-c:v', 'h264_nvenc', '-gpu', str(gpu_id), '-preset', 'p2',
            '-b:v', '5000k', '-maxrate', '7500k',
            '-c:a', 'aac', '-b:a', '128k',
            output_path
        ]

        print(f"[Video Caption Worker] GPU {gpu_id}, video {video_index}: '{text[:50]}...' with font {font_path}")
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if process.returncode != 0:
            # Try fallback to CPU encoding if NVENC fails
            print(f"[Video Caption Worker] NVENC failed, trying CPU fallback: {process.stderr[-200:]}")
            cmd_cpu = [
                'ffmpeg', '-y',
                '-i', input_path,
                '-vf', drawtext,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',
                output_path
            ]
            process = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=600)
            if process.returncode != 0:
                return {'status': 'failed', 'index': video_index, 'gpu_id': gpu_id, 'error': process.stderr[-500:]}

        return {'status': 'completed', 'index': video_index, 'output_path': output_path, 'duration': duration, 'gpu_id': gpu_id}

    except Exception as e:
        return {'status': 'failed', 'index': video_index, 'gpu_id': gpu_id, 'error': str(e)}


def process_videos_parallel_caption(
    video_paths: List[str],
    output_dir: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None,
    max_parallel: int = None
) -> Dict[str, Any]:
    """
    Process multiple videos with captions in parallel using multiple NVENC sessions.

    Uses ThreadPoolExecutor (not ProcessPoolExecutor) because FFmpeg subprocesses
    already handle the GPU work - no need for separate Python processes.

    Implements round-robin GPU assignment to distribute work across all available GPUs.
    """
    if not video_paths:
        return {'error': 'No videos to process'}

    # Detect all GPUs and calculate total parallel capacity
    gpu_info = get_gpu_info()
    gpu_count = gpu_info.get('gpu_count', 1)
    gpus = gpu_info.get('gpus', [{'index': 0, 'nvenc_sessions': 2}])
    total_nvenc_sessions = gpu_info.get('total_nvenc_sessions', 2)

    if max_parallel is None:
        max_parallel = total_nvenc_sessions

    # Log GPU configuration
    gpu_names = [g.get('name', 'Unknown') for g in gpus]
    print(f"[Parallel Caption] Detected {gpu_count} GPU(s): {', '.join(gpu_names)}")
    print(f"[Parallel Caption] Total NVENC sessions available: {total_nvenc_sessions}, using: {max_parallel}")

    # Build work items with round-robin GPU assignment
    # Each video gets assigned to a GPU in round-robin fashion
    work_items = []
    for i, video_path in enumerate(video_paths):
        basename = os.path.basename(video_path)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name}_captioned{ext}")

        # Round-robin GPU assignment based on video index
        gpu_id = gpus[i % gpu_count]['index']
        work_items.append((video_path, output_path, config, i, gpu_id))

    total = len(work_items)
    completed = 0
    failed = 0
    results = []

    def report_progress(msg=""):
        if progress_callback:
            progress_callback(completed / total if total > 0 else 0, msg)

    report_progress(f"Processing {total} videos across {gpu_count} GPU(s) with {max_parallel} parallel sessions...")

    # Use ThreadPoolExecutor - FFmpeg handles GPU work in subprocesses
    # This is more efficient than ProcessPoolExecutor for I/O-bound subprocess calls
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_item = {executor.submit(process_single_video_caption_worker, item): item for item in work_items}

        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)
            if result['status'] == 'completed':
                completed += 1
            else:
                failed += 1
            gpu_used = result.get('gpu_id', 0)
            report_progress(f"Completed {completed}/{total} videos ({failed} failed) [Last: GPU {gpu_used}]")

    return {
        'status': 'completed',
        'total': total,
        'completed': completed,
        'failed': failed,
        'results': results,
        'parallel_sessions': max_parallel,
        'gpu_count': gpu_count,
        'gpus_used': gpu_names
    }


def process_captioner(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    progress_callback: Optional[Callable[[float, str], None]] = None
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
    print(f"[Captioner] RAW text: {repr(config.get('text', 'NOT FOUND'))}")
    print(f"[Captioner] RAW captions: {repr(config.get('captions', 'NOT FOUND'))}")
    print(f"[Captioner] RAW caption_mode: {repr(config.get('caption_mode', 'NOT FOUND'))}")
    print(f"[Captioner] RAW captionMode: {repr(config.get('captionMode', 'NOT FOUND'))}")

    # Normalize config keys (snake_case -> camelCase)
    config = normalize_config(config)

    # Debug: Print config AFTER normalization
    print(f"[Captioner] NORMALIZED config keys: {list(config.keys())}")
    print(f"[Captioner] NORMALIZED text: {repr(config.get('text', 'NOT FOUND'))}")
    print(f"[Captioner] NORMALIZED captions: {repr(config.get('captions', 'NOT FOUND'))}")
    print(f"[Captioner] NORMALIZED captionMode: {repr(config.get('captionMode', 'NOT FOUND'))}")

    def report_progress(progress: float, message: str = ""):
        if progress_callback:
            progress_callback(progress, message)

    # Detect file type
    ext = os.path.splitext(input_path)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS
    is_zip = ext == '.zip'

    # Get image index for batch mode (passed from workflow execution)
    image_index = config.get('imageIndex', 0)

    # Debug logging for batch mode troubleshooting
    caption_mode = config.get('captionMode', 'single')
    captions = config.get('captions', [])
    text = config.get('text', '')
    print(f"[Captioner] Mode: {caption_mode}, imageIndex: {image_index}, captions count: {len(captions)}, text: {repr(text[:50] if text else '')}")

    # Debug: Show actual caption that will be used
    actual_caption = get_caption_for_index(config, image_index)
    print(f"[Captioner] ACTUAL caption to render: {repr(actual_caption[:100] if actual_caption else 'EMPTY')}")

    if caption_mode == 'batch' and captions:
        print(f"[Captioner] Using caption {image_index % len(captions)}: {captions[image_index % len(captions)][:50]}...")

    report_progress(0.05, "Analyzing file...")

    # NEW: Check if input is a ZIP with videos (batch video mode)
    if is_zip:
        report_progress(0.08, "Checking ZIP contents...")
        temp_dir = tempfile.mkdtemp(prefix="captioner_batch_")

        try:
            video_paths = extract_videos_from_zip(input_path, temp_dir)

            if video_paths:
                report_progress(0.1, f"Found {len(video_paths)} videos, starting parallel processing...")

                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(output_dir, exist_ok=True)

                result = process_videos_parallel_caption(
                    video_paths, output_dir, config, progress_callback=progress_callback
                )

                if result.get('error'):
                    return result

                report_progress(0.95, "Creating output ZIP...")

                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as zf:
                    for r in result.get('results', []):
                        if r.get('status') == 'completed' and r.get('output_path'):
                            if os.path.exists(r['output_path']):
                                zf.write(r['output_path'], os.path.basename(r['output_path']))

                report_progress(1.0, "Complete")

                return {
                    'status': 'completed',
                    'mode': 'parallel_video_batch',
                    'videos_processed': result.get('completed', 0),
                    'videos_failed': result.get('failed', 0),
                    'parallel_sessions': result.get('parallel_sessions', 1),
                    'output_size': os.path.getsize(output_path) if os.path.exists(output_path) else 0
                }

        except Exception as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            if 'video_paths' in dir() and video_paths:
                shutil.rmtree(temp_dir, ignore_errors=True)

    if is_video:
        return process_video_caption(input_path, output_path, config, report_progress, image_index)
    else:
        return process_image_caption(input_path, output_path, config, report_progress, image_index)


def process_image_caption(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None],
    image_index: int = 0
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

    img = Image.open(input_path).convert('RGBA')
    width, height = img.size

    # Get the actual text that will be rendered (for logging)
    caption_text = get_caption_for_index(config, image_index)
    parsed = parse_caption_format(caption_text)

    # Face detection (if enabled)
    faces = []
    avoid_faces = config.get('avoidFaces', False)
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
    if output_ext in ['.jpg', '.jpeg']:
        img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)
    elif output_ext == '.png':
        img.save(output_path, 'PNG')
    else:
        img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=95)

    report_progress(1.0, "Complete")

    # Determine final position (may have been adjusted for faces)
    final_position = config.get('position', 'bottom')
    position_adjusted = False
    if faces and avoid_faces:
        position_adjusted = True  # Actual adjustment happens in create_caption_overlay

    return {
        'text': caption_text,
        'title': parsed['title'],
        'body': parsed['body'],
        'position': final_position,
        'resolution': f"{width}x{height}",
        'captionMode': config.get('captionMode', 'single'),
        'imageIndex': image_index,
        'facesDetected': len(faces),
        'positionAdjusted': position_adjusted
    }


def process_video_caption(
    input_path: str,
    output_path: str,
    config: Dict[str, Any],
    report_progress: Callable[[float, str], None],
    image_index: int = 0
) -> Dict[str, Any]:
    """
    Add caption to video using FFmpeg drawtext filter.

    Supports GPU selection via config['gpuId'] parameter for multi-GPU systems.
    Supports batch mode: in batch mode, cycles through captions array using image_index.
    """

    report_progress(0.1, "Analyzing video...")

    # Get video info
    video_info = get_video_info(input_path)
    width = video_info.get('width', 1920)
    height = video_info.get('height', 1080)
    duration = video_info.get('duration', 0)

    report_progress(0.2, "Building caption filter...")

    # Extract config - support both frontend keys (font, textColor) and legacy keys (fontFamily, color)
    text = get_caption_for_index(config, image_index)  # Support batch mode
    position = config.get('position', 'bottom')
    font_size = config.get('fontSize', 48)
    font_name = config.get('font', config.get('fontFamily', 'TikTokBold.otf'))  # frontend sends 'font'
    font_weight = config.get('fontWeight', 'normal')
    color = config.get('textColor', config.get('color', '#FFFFFF'))  # frontend sends 'textColor'
    bg_color = config.get('backgroundColor')
    bg_opacity = config.get('backgroundOpacity', 80) / 100
    stroke_color = config.get('strokeColor', '#000000')
    stroke_width = config.get('strokeWidth', 2)
    shadow = config.get('shadow', True)
    animation = config.get('animation', 'none')
    start_time = config.get('startTime', 0)
    end_time = config.get('endTime')
    max_width = config.get('maxWidth')

    # GPU selection for multi-GPU systems (default to GPU 0)
    gpu_id = config.get('gpuId', config.get('gpu_id', 0))

    # Get font path - check FONT_MAP with font name directly (e.g., 'TikTokBold.otf')
    font_path = FONT_MAP.get(font_name, FONT_MAP.get('default', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))

    # Fallback: try with bold suffix if font_weight is bold
    if font_weight == 'bold' and font_name not in FONT_MAP:
        font_key = f"{font_name} Bold"
        font_path = FONT_MAP.get(font_key, FONT_MAP.get('default-bold', font_path))

    # Escape text for FFmpeg
    text_escaped = escape_ffmpeg_text(text)

    # Calculate position
    x, y = calculate_position(position, width, height, font_size, config)

    # Build drawtext filter
    drawtext_parts = [
        f"text='{text_escaped}'",
        f"fontfile='{font_path}'",
        f"fontsize={font_size}",
        f"fontcolor={color}",
        f"x={x}",
        f"y={y}"
    ]

    # Add stroke/border
    if stroke_color and stroke_width > 0:
        drawtext_parts.append(f"borderw={stroke_width}")
        drawtext_parts.append(f"bordercolor={stroke_color}")

    # Add shadow
    if shadow:
        drawtext_parts.append("shadowcolor=black@0.5")
        drawtext_parts.append("shadowx=2")
        drawtext_parts.append("shadowy=2")

    # Add background box
    if bg_color:
        drawtext_parts.append(f"box=1")
        alpha_hex = hex(int(bg_opacity * 255))[2:].zfill(2)
        drawtext_parts.append(f"boxcolor={bg_color}@{bg_opacity:.2f}")
        drawtext_parts.append(f"boxborderw=10")

    # Add timing (enable/disable)
    if start_time > 0 or end_time:
        enable_expr = f"between(t,{start_time},{end_time or duration})"
        drawtext_parts.append(f"enable='{enable_expr}'")

    # Add animation
    if animation == 'fade':
        # Fade in over 0.5 seconds
        drawtext_parts.append(f"alpha='if(lt(t,{start_time + 0.5}),(t-{start_time})/0.5,1)'")
    elif animation == 'slide':
        # Slide up from bottom
        drawtext_parts.append(f"y='if(lt(t,{start_time + 0.3}),h-(h-{y})*(t-{start_time})/0.3,{y})'")

    drawtext_filter = f"drawtext={':'.join(drawtext_parts)}"

    report_progress(0.3, f"Encoding with NVENC on GPU {gpu_id}...")

    def build_ffmpeg_cmd(use_nvenc: bool = True) -> list:
        """Build FFmpeg command with GPU or CPU encoding."""
        cmd = ['ffmpeg', '-y']
        if use_nvenc:
            # Specify GPU for hardware acceleration
            cmd.extend(['-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id)])
        cmd.extend([
            '-i', input_path,
            '-vf', drawtext_filter,
        ])

        if use_nvenc:
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-gpu', str(gpu_id),  # Specify GPU for NVENC encoder
                '-preset', 'p4',
                '-b:v', '5000k',
                '-maxrate', '7500k',
                '-bufsize', '10000k',
                '-profile:v', 'high',
            ])
        else:
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-profile:v', 'high',
            ])

        cmd.extend([
            '-c:a', 'aac',
            '-b:a', '128k',
            output_path
        ])
        return cmd

    def run_ffmpeg_with_progress(cmd: list, mode_name: str) -> tuple:
        """Run FFmpeg and capture stderr properly. Returns (success, stderr_output)."""
        stderr_lines = []

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )

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

                if 'time=' in line:
                    try:
                        time_str = line.split('time=')[1].split()[0]
                        h, m, s = time_str.split(':')
                        current_time = int(h) * 3600 + int(m) * 60 + float(s)
                        if duration > 0:
                            progress = min(0.3 + (current_time / duration) * 0.65, 0.95)
                            report_progress(progress, f"Encoding ({mode_name})... {int(current_time)}s / {int(duration)}s")
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

        stderr_output = ''.join(stderr_lines)
        return process.returncode == 0, stderr_output

    # Try NVENC first
    cmd = build_ffmpeg_cmd(use_nvenc=True)
    success, stderr = run_ffmpeg_with_progress(cmd, "NVENC")

    # If NVENC failed, try CPU fallback
    if not success:
        print(f"[Captioner] NVENC failed, trying CPU fallback. Error: {stderr[-500:]}")
        report_progress(0.3, "NVENC failed, trying CPU encoding...")
        cmd = build_ffmpeg_cmd(use_nvenc=False)
        success, stderr = run_ffmpeg_with_progress(cmd, "CPU")

    if not success:
        raise RuntimeError(f"FFmpeg failed with all encoders: {stderr[-1000:]}")

    report_progress(1.0, "Complete")

    return {
        'text': text,
        'position': position,
        'animation': animation,
        'resolution': f"{width}x{height}",
        'duration': duration,
        'gpuId': gpu_id
    }


def create_caption_overlay(
    width: int,
    height: int,
    config: Dict[str, Any],
    image_index: int = 0,
    faces: List[Dict[str, Any]] = None
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

    # Get text based on mode (single or batch)
    raw_text = get_caption_for_index(config, image_index)

    # Parse ##Title## and &&& formatting
    parsed = parse_caption_format(raw_text)
    title_text = parsed['title']
    body_text = parsed['body']

    # Scaling factor (base resolution 1080px)
    BASE_WIDTH = 1080
    scale_factor = width / BASE_WIDTH

    # Check if this image should be force-centered (center every N feature)
    center_every_enabled = config.get('centerEveryEnabled', False)
    center_every_n = config.get('centerEveryN', 7)
    force_center = center_every_enabled and (image_index + 1) % center_every_n == 0

    # Extract config with scaling (default position is now CENTER)
    position = 'center' if force_center else config.get('position', 'center')

    # Scale font sizes
    base_font_size = config.get('fontSize', 47)
    base_title_size = config.get('titleFontSize', 75)
    base_stroke_width = config.get('strokeWidth', 3.5)

    font_size = max(12, int(base_font_size * scale_factor))
    title_font_size = max(14, int(base_title_size * scale_factor))
    stroke_width = max(1, int(base_stroke_width * scale_factor))

    font_name = config.get('font', config.get('fontFamily', 'TikTokBold.otf'))
    text_color = config.get('textColor', config.get('color', '#FFFFFF'))
    bg_color = config.get('backgroundColor') if config.get('showBackground', False) else None
    bg_opacity = config.get('backgroundOpacity', 80) / 100
    stroke_color = config.get('strokeColor', '#000000')
    shadow = config.get('shadow', False)
    alignment = config.get('alignment', 'center')
    text_width_ratio = config.get('textWidthRatio', 0.85)
    block_spacing = max(5, int(config.get('blockSpacing', 40) * scale_factor))
    line_spacing = max(2, int(config.get('lineSpacing', 20) * scale_factor))

    # Random tilt (default -5 to +5 degrees)
    random_tilt = config.get('randomTilt', False)
    tilt_angle = 0
    if random_tilt:
        tilt_min = config.get('tiltRangeMin', -5)
        tilt_max = config.get('tiltRangeMax', 5)
        tilt_angle = random.uniform(tilt_min, tilt_max)

    max_width = int(width * text_width_ratio)

    # Create transparent overlay
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Load fonts
    font_path = FONT_MAP.get(font_name, FONT_MAP['default-bold'])
    italic_font_name = font_name.replace('Bold', 'Italic').replace('bold', 'italic')
    if 'TikTok' in font_name or 'tiktok' in font_name:
        italic_font_path = FONT_MAP.get('LightItalic.otf', FONT_MAP.get('tiktok-italic', FONT_MAP['default-italic']))
    else:
        italic_font_path = FONT_MAP.get(italic_font_name, FONT_MAP['default-italic'])

    try:
        body_font = ImageFont.truetype(font_path, font_size)
    except:
        body_font = ImageFont.load_default()

    try:
        title_font = ImageFont.truetype(italic_font_path, title_font_size)
    except:
        title_font = body_font

    # Calculate total content height and widths
    total_height = 0
    content_blocks = []

    if title_text:
        wrapped_title = wrap_text(title_text, title_font, max_width)
        title_bbox = draw.multiline_textbbox((0, 0), wrapped_title, font=title_font, spacing=line_spacing)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
        content_blocks.append({
            'type': 'title',
            'text': wrapped_title,
            'font': title_font,
            'width': title_w,
            'height': title_h
        })
        total_height += title_h + block_spacing

    if body_text:
        wrapped_body = wrap_text(body_text, body_font, max_width)
        body_bbox = draw.multiline_textbbox((0, 0), wrapped_body, font=body_font, spacing=line_spacing)
        body_w = body_bbox[2] - body_bbox[0]
        body_h = body_bbox[3] - body_bbox[1]
        content_blocks.append({
            'type': 'body',
            'text': wrapped_body,
            'font': body_font,
            'width': body_w,
            'height': body_h
        })
        total_height += body_h

    if not content_blocks:
        return overlay

    # Calculate max content width
    max_content_width = max(block['width'] for block in content_blocks)

    # Face avoidance: adjust position AND alignment if faces are detected
    avoid_faces = config.get('avoidFaces', False)
    position_x = config.get('positionX', config.get('customX', 0.5))
    position_y = config.get('positionY', config.get('customY', 0.85))

    if avoid_faces and faces:
        # Calculate safe position and alignment that avoids faces
        adjusted_position, adjusted_y, adjusted_alignment = get_face_safe_position(
            faces=faces,
            caption_height=total_height,
            caption_width=max_content_width,
            image_width=width,
            image_height=height,
            preferred_position=position,
            preferred_alignment=alignment,
            config=config
        )
        # Update position and alignment if adjusted
        position = adjusted_position
        position_y = adjusted_y
        alignment = adjusted_alignment

    # Calculate Y position based on (possibly adjusted) position
    # Using safe margins: 15% from top, 18% from bottom
    if position == 'top':
        base_y = int(height * 0.15) - total_height // 2
    elif position == 'center':
        base_y = (height - total_height) // 2
    elif position == 'bottom':
        base_y = int(height * 0.82) - total_height // 2
    elif position == 'custom':
        base_y = int(position_y * height) - total_height // 2
    else:
        # Default to center
        base_y = (height - total_height) // 2

    # Calculate X based on alignment
    if alignment == 'left':
        base_x = 20
    elif alignment == 'right':
        base_x = width - max_content_width - 20
    else:  # center
        if position == 'custom':
            base_x = int(position_x * width) - max_content_width // 2
        else:
            base_x = (width - max_content_width) // 2

    # Clamp positions to be within safe margins (5% of image dimensions)
    # This prevents captions from hiding in the edges
    margin_x = max(20, int(width * 0.05))
    margin_y = max(30, int(height * 0.05))
    base_x = max(margin_x, min(base_x, width - max_content_width - margin_x))
    base_y = max(margin_y, min(base_y, height - total_height - margin_y))

    # Draw background if enabled
    if bg_color:
        padding = 15
        bg_r, bg_g, bg_b = hex_to_rgb(bg_color)
        bg_alpha = int(bg_opacity * 255)
        draw.rounded_rectangle(
            [base_x - padding, base_y - padding,
             base_x + max_content_width + padding, base_y + total_height + padding],
            radius=8,
            fill=(bg_r, bg_g, bg_b, bg_alpha)
        )

    # Draw each content block
    current_y = base_y
    text_r, text_g, text_b = hex_to_rgb(text_color)

    for block in content_blocks:
        text = block['text']
        font = block['font']
        block_width = block['width']
        block_height = block['height']

        # Calculate X for this block based on alignment
        if alignment == 'left':
            x = base_x
        elif alignment == 'right':
            x = base_x + max_content_width - block_width
        else:  # center
            x = base_x + (max_content_width - block_width) // 2

        # Draw shadow
        if shadow:
            draw.multiline_text(
                (x + 2, current_y + 2),
                text,
                font=font,
                fill=(0, 0, 0, 128),
                spacing=line_spacing,
                align=alignment
            )

        # Draw stroke
        if stroke_color and stroke_width > 0:
            stroke_r, stroke_g, stroke_b = hex_to_rgb(stroke_color)
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        draw.multiline_text(
                            (x + dx, current_y + dy),
                            text,
                            font=font,
                            fill=(stroke_r, stroke_g, stroke_b, 255),
                            spacing=line_spacing,
                            align=alignment
                        )

        # Draw text
        draw.multiline_text(
            (x, current_y),
            text,
            font=font,
            fill=(text_r, text_g, text_b, 255),
            spacing=line_spacing,
            align=alignment
        )

        current_y += block_height
        if block['type'] == 'title':
            current_y += block_spacing

    # Apply random tilt if enabled
    if tilt_angle != 0:
        # Rotate the overlay around the center of the caption
        # First, find the caption bounding box
        caption_center_x = base_x + max_content_width // 2
        caption_center_y = base_y + total_height // 2

        # Create a larger canvas to accommodate rotation
        diagonal = int((width ** 2 + height ** 2) ** 0.5)
        rotated = Image.new('RGBA', (diagonal, diagonal), (0, 0, 0, 0))

        # Calculate offset to center the overlay in the rotated canvas
        offset_x = (diagonal - width) // 2
        offset_y = (diagonal - height) // 2

        rotated.paste(overlay, (offset_x, offset_y))

        # Rotate around center
        rotated = rotated.rotate(tilt_angle, center=(diagonal // 2, diagonal // 2), expand=False, fillcolor=(0, 0, 0, 0))

        # Crop back to original size
        crop_x = (diagonal - width) // 2
        crop_y = (diagonal - height) // 2
        overlay = rotated.crop((crop_x, crop_y, crop_x + width, crop_y + height))

    return overlay


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    """Wrap text to fit within max width."""
    lines = []
    for line in text.split('\n'):
        if font.getlength(line) <= max_width:
            lines.append(line)
        else:
            words = line.split()
            current_line = []
            for word in words:
                test_line = ' '.join(current_line + [word])
                if font.getlength(test_line) <= max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))

    return '\n'.join(lines)


def calculate_position(position: str, width: int, height: int, font_size: int, config: Dict[str, Any]) -> tuple:
    """Calculate FFmpeg position expressions."""
    padding = 20

    if position == 'top':
        return f'(w-text_w)/2', str(padding)
    elif position == 'center':
        return f'(w-text_w)/2', f'(h-text_h)/2'
    elif position == 'bottom':
        return f'(w-text_w)/2', f'h-text_h-{padding}'
    elif position == 'custom':
        x = config.get('customX', width // 2)
        y = config.get('customY', height - 100)
        return str(x), str(y)
    else:
        return f'(w-text_w)/2', f'h-text_h-{padding}'


def calculate_position_pixels(position: str, width: int, height: int, text_width: int, text_height: int, config: Dict[str, Any]) -> tuple:
    """Calculate pixel position for image captions."""
    padding = 20

    if position == 'top':
        return (width - text_width) // 2, padding
    elif position == 'center':
        return (width - text_width) // 2, (height - text_height) // 2
    elif position == 'bottom':
        return (width - text_width) // 2, height - text_height - padding
    elif position == 'custom':
        x = config.get('customX', (width - text_width) // 2)
        y = config.get('customY', height - text_height - padding)
        return x, y
    else:
        return (width - text_width) // 2, height - text_height - padding


def escape_ffmpeg_text(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter."""
    # Escape single quotes, colons, and backslashes
    text = text.replace('\\', '\\\\')
    text = text.replace("'", "\\'")
    text = text.replace(':', '\\:')
    text = text.replace('%', '%%')
    return text


def get_video_info(path: str) -> Dict[str, Any]:
    """Get video metadata using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_streams', '-show_format', path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)

        video_stream = next(
            (s for s in info.get('streams', []) if s.get('codec_type') == 'video'),
            {}
        )

        return {
            'width': int(video_stream.get('width', 1920)),
            'height': int(video_stream.get('height', 1080)),
            'duration': float(info.get('format', {}).get('duration', 0))
        }
    except:
        return {'width': 1920, 'height': 1080, 'duration': 0}


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c * 2 for c in hex_color])

    try:
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    except:
        return (255, 255, 255)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python processor.py input_file output_file")
        sys.exit(1)

    test_config = {
        'text': 'Sample Caption Text',
        'position': 'center',  # Default position is now center
        'fontSize': 48,
        'fontFamily': 'Arial',
        'color': '#FFFFFF',
        'strokeColor': '#000000',
        'strokeWidth': 2,
        'shadow': True
    }

    def progress(p, msg):
        print(f"[{int(p*100)}%] {msg}")

    result = process_captioner(sys.argv[1], sys.argv[2], test_config, progress)
    print(f"Result: {result}")
