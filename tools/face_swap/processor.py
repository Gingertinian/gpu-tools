"""
Face Swap Processor

Uses InsightFace for face detection and inswapper model for face swapping.
Supports images and videos.

Config options:
- source_face_url: str (URL of source face image - the face to use)
- target_face_index: int = 0 (which face in target to swap, -1 for all)
- enhance_face: bool = True (apply GFPGAN enhancement after swap)
- similarity_threshold: float = 0.6 (minimum face similarity for detection)
"""

import logging
import os
import gc
import tempfile
import shutil
import subprocess
from typing import Callable, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def download_models():
    """Ensure required models are downloaded"""
    import insightface
    from insightface.app import FaceAnalysis

    # This will download the buffalo_l model if not present
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_face_swapper():
    """Load the inswapper model"""
    import insightface

    model_path = os.path.join(
        os.path.dirname(insightface.__file__),
        'models',
        'inswapper_128.onnx'
    )

    # Alternative paths
    alt_paths = [
        '/workspace/models/inswapper_128.onnx',
        '/opt/models/inswapper_128.onnx',
        os.path.expanduser('~/.insightface/models/inswapper_128.onnx'),
    ]

    for path in [model_path] + alt_paths:
        if os.path.exists(path):
            return insightface.model_zoo.get_model(path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    raise FileNotFoundError(
        "inswapper_128.onnx not found. Download from: "
        "https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx"
    )


def get_face_enhancer():
    """Load GFPGAN face enhancer"""
    try:
        from gfpgan import GFPGANer

        return GFPGANer(
            model_path='weights/GFPGANv1.4.pth',
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None,
        )
    except Exception:
        return None


def download_source_face(url: str, temp_dir: str, max_retries: int = 3) -> str:
    """Download source face image from URL with retry logic."""
    import requests
    import time as _time
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Determine extension from content-type or URL
            ext = '.jpg'
            content_type = response.headers.get('content-type', '')
            if 'png' in content_type:
                ext = '.png'
            elif 'webp' in content_type:
                ext = '.webp'

            path = os.path.join(temp_dir, f'source_face{ext}')
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return path
        except (requests.RequestException, IOError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                _logger.warning(f"[FaceSwap] Download attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                _time.sleep(wait_time)

    raise RuntimeError(f"Face source download failed after {max_retries} attempts: {last_error}")


def process_face_swap(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Swap faces in image using InsightFace.

    Args:
        input_path: Path to target image (image to modify)
        output_path: Path for output image
        config: Configuration including source_face_url
        progress_callback: Optional callback for progress updates

    Returns:
        dict with processing results
    """
    source_face_url = config.get("source_face_url")
    if not source_face_url:
        raise ValueError("source_face_url is required in config")

    target_face_index = config.get("target_face_index", 0)
    enhance_face = config.get("enhance_face", True)
    similarity_threshold = config.get("similarity_threshold", 0.6)

    temp_dir = tempfile.mkdtemp(prefix="face_swap_")

    try:
        if progress_callback:
            progress_callback(0.1, "Loading models...")

        # Load models
        face_analyzer = download_models()
        face_swapper = get_face_swapper()
        face_enhancer = get_face_enhancer() if enhance_face else None

        if progress_callback:
            progress_callback(0.2, "Downloading source face...")

        # Download source face
        source_path = download_source_face(source_face_url, temp_dir)

        if progress_callback:
            progress_callback(0.3, "Analyzing faces...")

        # Load images
        source_img = cv2.imread(source_path)
        target_img = cv2.imread(input_path)

        if source_img is None:
            raise ValueError(f"Failed to load source image: {source_path}")
        if target_img is None:
            raise ValueError(f"Failed to load target image: {input_path}")

        # Detect faces
        source_faces = face_analyzer.get(source_img)
        target_faces = face_analyzer.get(target_img)

        if len(source_faces) == 0:
            raise ValueError("No face detected in source image")
        if len(target_faces) == 0:
            raise ValueError("No face detected in target image")

        # Use first face from source
        source_face = source_faces[0]

        if progress_callback:
            progress_callback(0.5, "Swapping faces...")

        # Swap face(s)
        result = target_img.copy()

        if target_face_index == -1:
            # Swap all faces
            for face in target_faces:
                result = face_swapper.get(result, face, source_face, paste_back=True)
            faces_swapped = len(target_faces)
        else:
            # Swap specific face
            if target_face_index >= len(target_faces):
                target_face_index = 0
            result = face_swapper.get(result, target_faces[target_face_index], source_face, paste_back=True)
            faces_swapped = 1

        if progress_callback:
            progress_callback(0.7, "Enhancing result...")

        # Enhance faces if enabled
        if face_enhancer is not None:
            try:
                _, _, result = face_enhancer.enhance(result, has_aligned=False, only_center_face=False, paste_back=True)
            except Exception:
                pass  # Continue without enhancement if it fails

        if progress_callback:
            progress_callback(0.9, "Saving result...")

        # Save output
        cv2.imwrite(output_path, result)

        h, w = result.shape[:2]

        if progress_callback:
            progress_callback(1.0, "Complete")

        return {
            "faces_swapped": faces_swapped,
            "faces_detected_in_target": len(target_faces),
            "output_size": f"{w}x{h}",
            "enhanced": face_enhancer is not None,
        }

    finally:
        # Clean up GPU memory between jobs
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_face_swap_video(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Swap faces in video frame by frame.
    """
    source_face_url = config.get("source_face_url")
    if not source_face_url:
        raise ValueError("source_face_url is required in config")

    target_face_index = config.get("target_face_index", 0)
    enhance_face = config.get("enhance_face", False)  # Default off for video (speed)

    temp_dir = tempfile.mkdtemp(prefix="face_swap_video_")

    try:
        if progress_callback:
            progress_callback(0.05, "Loading models...")

        # Load models
        face_analyzer = download_models()
        face_swapper = get_face_swapper()
        face_enhancer = get_face_enhancer() if enhance_face else None

        # Download source face
        source_path = download_source_face(source_face_url, temp_dir)
        source_img = cv2.imread(source_path)
        source_faces = face_analyzer.get(source_img)

        if len(source_faces) == 0:
            raise ValueError("No face detected in source image")

        source_face = source_faces[0]

        if progress_callback:
            progress_callback(0.1, "Analyzing video...")

        # Open video
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup output
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir)

        if progress_callback:
            progress_callback(0.15, f"Processing {frame_count} frames...")

        # Process frames
        frame_idx = 0
        faces_swapped_total = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and swap faces
            target_faces = face_analyzer.get(frame)

            if len(target_faces) > 0:
                result = frame.copy()

                if target_face_index == -1:
                    for face in target_faces:
                        result = face_swapper.get(result, face, source_face, paste_back=True)
                    faces_swapped_total += len(target_faces)
                else:
                    idx = min(target_face_index, len(target_faces) - 1)
                    result = face_swapper.get(result, target_faces[idx], source_face, paste_back=True)
                    faces_swapped_total += 1

                # Optional enhancement
                if face_enhancer is not None:
                    try:
                        _, _, result = face_enhancer.enhance(result, has_aligned=False, only_center_face=False, paste_back=True)
                    except Exception:
                        pass
            else:
                result = frame

            # Save frame
            cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"), result)

            frame_idx += 1

            if progress_callback and frame_idx % 30 == 0:
                progress = 0.15 + (frame_idx / frame_count) * 0.7
                progress_callback(progress, f"Processing frame {frame_idx}/{frame_count}")

        cap.release()

        if progress_callback:
            progress_callback(0.9, "Encoding video...")

        # Encode output video with NVENC
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(frames_dir, "frame_%06d.png"),
                "-i", input_path,  # For audio
                "-map", "0:v",
                "-map", "1:a?",
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", "20",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                output_path
            ], capture_output=True, check=True, timeout=300)
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg video encoding timed out after 300 seconds")
            raise RuntimeError("FFmpeg video encoding timed out after 300 seconds")

        if progress_callback:
            progress_callback(1.0, "Complete")

        return {
            "frames_processed": frame_idx,
            "faces_swapped_total": faces_swapped_total,
            "fps": fps,
            "output_size": f"{width}x{height}",
        }

    finally:
        # Clean up GPU memory between jobs
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
        shutil.rmtree(temp_dir, ignore_errors=True)
