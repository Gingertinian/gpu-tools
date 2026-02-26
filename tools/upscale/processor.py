"""
Upscale Processor

Uses Real-ESRGAN for high-quality AI upscaling.
Supports both images and videos (frame-by-frame for videos).

Config options:
- scale: int = 4 (upscale factor: 2, 3, or 4)
- model: str = "realesrgan-x4plus" | "realesrgan-x4plus-anime" | "realesr-general-x4v3"
- denoise_strength: float = 0.5 (0-1, only for realesr-general-x4v3)
- face_enhance: bool = False (use GFPGAN for face enhancement)
- tile: int = 0 (tile size for processing large images, 0 = auto)
- fp32: bool = False (use fp32 precision instead of fp16)
"""

import logging
import os
import gc
import subprocess
import tempfile
import shutil
from typing import Callable, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def get_realesrgan_path():
    """Get path to Real-ESRGAN executable"""
    # Check common locations
    locations = [
        "/workspace/Real-ESRGAN/realesrgan-ncnn-vulkan",
        "/opt/realesrgan/realesrgan-ncnn-vulkan",
        "/usr/local/bin/realesrgan-ncnn-vulkan",
        shutil.which("realesrgan-ncnn-vulkan"),
    ]

    for loc in locations:
        if loc and os.path.exists(loc):
            return loc

    # Try Python package
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        return "python"  # Use Python implementation
    except ImportError:
        pass

    return None


def process_upscale_python(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Upscale using Python Real-ESRGAN implementation (GPU accelerated)"""
    import cv2
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    scale = config.get("scale", 4)
    model_name = config.get("model", "realesrgan-x4plus")
    denoise_strength = config.get("denoise_strength", 0.5)
    face_enhance = config.get("face_enhance", False)
    tile = config.get("tile", 0)
    use_fp32 = config.get("fp32", False)

    if progress_callback:
        progress_callback(0.1, "Loading model...")

    # Model configurations
    model_configs = {
        "realesrgan-x4plus": {
            "model_path": "weights/RealESRGAN_x4plus.pth",
            "netscale": 4,
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        },
        "realesrgan-x4plus-anime": {
            "model_path": "weights/RealESRGAN_x4plus_anime_6B.pth",
            "netscale": 4,
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
        },
        "realesr-general-x4v3": {
            "model_path": "weights/realesr-general-x4v3.pth",
            "netscale": 4,
            "model": RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        },
    }

    mc = model_configs.get(model_name, model_configs["realesrgan-x4plus"])

    # Initialize upsampler
    upsampler = RealESRGANer(
        scale=mc["netscale"],
        model_path=mc["model_path"],
        dni_weight=None if model_name != "realesr-general-x4v3" else denoise_strength,
        model=mc["model"],
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=not use_fp32 and torch.cuda.is_available(),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )

    # Face enhancer
    face_enhancer = None
    if face_enhance:
        try:
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='weights/GFPGANv1.3.pth',
                upscale=scale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )
        except ImportError:
            pass

    if progress_callback:
        progress_callback(0.2, "Processing image...")

    # Read image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {input_path}")

    h, w = img.shape[:2]

    if progress_callback:
        progress_callback(0.4, "Upscaling...")

    # Process
    if face_enhancer is not None:
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    else:
        output, _ = upsampler.enhance(img, outscale=scale)

    if progress_callback:
        progress_callback(0.8, "Saving result...")

    # Save output
    cv2.imwrite(output_path, output)

    new_h, new_w = output.shape[:2]

    # Clean up GPU memory between jobs
    del upsampler
    if face_enhancer is not None:
        del face_enhancer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "original_size": f"{w}x{h}",
        "output_size": f"{new_w}x{new_h}",
        "scale": scale,
        "model": model_name,
        "face_enhance": face_enhance,
    }


def process_upscale_cli(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Upscale using Real-ESRGAN CLI (NCNN Vulkan - faster)"""
    from PIL import Image

    scale = config.get("scale", 4)
    model_name = config.get("model", "realesrgan-x4plus")
    tile = config.get("tile", 0)

    realesrgan_path = get_realesrgan_path()
    if not realesrgan_path or realesrgan_path == "python":
        raise RuntimeError("Real-ESRGAN CLI not found")

    if progress_callback:
        progress_callback(0.1, "Starting upscale...")

    # Get original dimensions
    img = Image.open(input_path)
    orig_w, orig_h = img.size
    img.close()

    # Model name mapping for CLI
    model_map = {
        "realesrgan-x4plus": "realesrgan-x4plus",
        "realesrgan-x4plus-anime": "realesrgan-x4plus-anime",
        "realesr-general-x4v3": "realesr-animevideov3",
    }

    cli_model = model_map.get(model_name, "realesrgan-x4plus")

    # Build command
    cmd = [
        realesrgan_path,
        "-i", input_path,
        "-o", output_path,
        "-n", cli_model,
        "-s", str(scale),
    ]

    if tile > 0:
        cmd.extend(["-t", str(tile)])

    if progress_callback:
        progress_callback(0.3, "Processing...")

    # Run
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        raise RuntimeError(f"Real-ESRGAN failed: {result.stderr}")

    if progress_callback:
        progress_callback(0.9, "Finalizing...")

    # Get output dimensions
    out_img = Image.open(output_path)
    new_w, new_h = out_img.size
    out_img.close()

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "original_size": f"{orig_w}x{orig_h}",
        "output_size": f"{new_w}x{new_h}",
        "scale": scale,
        "model": model_name,
        "backend": "ncnn-vulkan",
    }


def process_upscale(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Upscale image using Real-ESRGAN.
    Automatically selects best available backend.
    """
    backend = get_realesrgan_path()

    if backend == "python":
        return process_upscale_python(input_path, output_path, config, progress_callback)
    elif backend:
        return process_upscale_cli(input_path, output_path, config, progress_callback)
    else:
        raise RuntimeError(
            "Real-ESRGAN not available. Install with: pip install realesrgan basicsr"
        )


def process_upscale_video(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Upscale video frame by frame.
    Uses FFmpeg for extraction/encoding and Real-ESRGAN for upscaling.
    """
    import cv2

    scale = config.get("scale", 4)

    if progress_callback:
        progress_callback(0.05, "Analyzing video...")

    # Get video info
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Create temp directory for frames
    temp_dir = tempfile.mkdtemp(prefix="upscale_")
    frames_dir = os.path.join(temp_dir, "frames")
    upscaled_dir = os.path.join(temp_dir, "upscaled")
    os.makedirs(frames_dir)
    os.makedirs(upscaled_dir)

    try:
        if progress_callback:
            progress_callback(0.1, "Extracting frames...")

        # Extract frames with FFmpeg
        try:
            subprocess.run([
                "ffmpeg", "-i", input_path,
                "-qscale:v", "2",
                os.path.join(frames_dir, "frame_%06d.png")
            ], capture_output=True, check=True, timeout=300)
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg frame extraction timed out after 300 seconds")
            raise RuntimeError("FFmpeg frame extraction timed out after 300 seconds")

        # Get list of frames
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        total_frames = len(frames)

        if progress_callback:
            progress_callback(0.2, f"Upscaling {total_frames} frames...")

        # Upscale each frame
        for i, frame_name in enumerate(frames):
            frame_path = os.path.join(frames_dir, frame_name)
            out_path = os.path.join(upscaled_dir, frame_name)

            process_upscale(frame_path, out_path, config, None)

            if progress_callback and i % 10 == 0:
                progress = 0.2 + (i / total_frames) * 0.6
                progress_callback(progress, f"Upscaling frame {i+1}/{total_frames}")

        if progress_callback:
            progress_callback(0.85, "Encoding video...")

        # Encode upscaled frames back to video
        new_width = width * scale
        new_height = height * scale

        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(upscaled_dir, "frame_%06d.png"),
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", "20",
                "-pix_fmt", "yuv420p",
                output_path
            ], capture_output=True, check=True, timeout=300)
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg video encoding timed out after 300 seconds")
            raise RuntimeError("FFmpeg video encoding timed out after 300 seconds")

        if progress_callback:
            progress_callback(1.0, "Complete")

        return {
            "original_size": f"{width}x{height}",
            "output_size": f"{new_width}x{new_height}",
            "scale": scale,
            "frames_processed": total_frames,
            "fps": fps,
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
        # Cleanup temp files
        shutil.rmtree(temp_dir, ignore_errors=True)
