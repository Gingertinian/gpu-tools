"""
AI Video Generation Processor

Generates video from image using AI models.
Supports multiple backends:
- Stable Video Diffusion (local, requires ~24GB VRAM)
- Replicate API (cloud, pay per use)

Config options:
- backend: str = "replicate" | "svd" (stable video diffusion)
- motion_bucket_id: int = 127 (1-255, higher = more motion)
- fps: int = 7 (frames per second in output)
- num_frames: int = 25 (number of frames to generate)
- decode_chunk_size: int = 8 (for memory optimization)
- noise_aug_strength: float = 0.02 (noise augmentation)
- seed: int = None (random seed for reproducibility)
- replicate_model: str = "stability-ai/stable-video-diffusion"
"""

import os
import gc
import tempfile
import shutil
import subprocess
from typing import Callable, Optional
from PIL import Image


def process_video_gen_replicate(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Generate video using Replicate API"""
    import replicate

    model = config.get("replicate_model", "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438")
    motion_bucket_id = config.get("motion_bucket_id", 127)
    fps = config.get("fps", 7)
    num_frames = config.get("num_frames", 25)
    seed = config.get("seed")

    if progress_callback:
        progress_callback(0.1, "Uploading image...")

    # Read and prepare input image (use with statement to avoid file handle leak)
    with Image.open(input_path) as img:
        # SVD expects 1024x576 or similar 16:9 aspect
        target_w, target_h = 1024, 576
        img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Save temp resized image
        temp_dir = tempfile.mkdtemp()
        temp_input = os.path.join(temp_dir, "input.png")
        img_resized.save(temp_input)

    try:
        if progress_callback:
            progress_callback(0.2, "Starting generation...")

        # Build input (use with statement to properly close file handle)
        input_file = open(temp_input, "rb")
        try:
            replicate_input = {
                "input_image": input_file,
                "motion_bucket_id": motion_bucket_id,
                "fps": fps,
                "num_frames": num_frames,
            }

            if seed is not None:
                replicate_input["seed"] = seed

            # Run prediction
            output = replicate.run(model, input=replicate_input)
        finally:
            input_file.close()

        if progress_callback:
            progress_callback(0.8, "Downloading result...")

        # Download result with retry logic
        import requests
        import time as _time
        import logging as _logging
        _logger = _logging.getLogger(__name__)

        _max_retries = 3
        _last_err = None
        for _attempt in range(_max_retries):
            try:
                response = requests.get(output, stream=True, timeout=120)
                response.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                _last_err = None
                break
            except (requests.RequestException, IOError) as e:
                _last_err = e
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                if _attempt < _max_retries - 1:
                    _wait = 2 ** _attempt
                    _logger.warning(f"[VideoGen] Download attempt {_attempt + 1} failed: {e}. Retrying in {_wait}s...")
                    _time.sleep(_wait)
        if _last_err is not None:
            raise RuntimeError(f"VideoGen download failed after {_max_retries} attempts: {_last_err}")

        if progress_callback:
            progress_callback(1.0, "Complete")

        return {
            "backend": "replicate",
            "model": model,
            "fps": fps,
            "num_frames": num_frames,
            "motion_bucket_id": motion_bucket_id,
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_video_gen_svd(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """Generate video using local Stable Video Diffusion"""
    import torch
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import export_to_video

    motion_bucket_id = config.get("motion_bucket_id", 127)
    fps = config.get("fps", 7)
    num_frames = config.get("num_frames", 25)
    decode_chunk_size = config.get("decode_chunk_size", 8)
    noise_aug_strength = config.get("noise_aug_strength", 0.02)
    seed = config.get("seed")

    if progress_callback:
        progress_callback(0.1, "Loading model...")

    # Load pipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.enable_model_cpu_offload()

    if progress_callback:
        progress_callback(0.2, "Preparing image...")

    # Load and prepare image
    image = Image.open(input_path)
    image = image.resize((1024, 576))

    # Set seed
    generator = None
    if seed is not None:
        generator = torch.manual_seed(seed)

    if progress_callback:
        progress_callback(0.3, "Generating video...")

    # Generate
    frames = pipe(
        image,
        decode_chunk_size=decode_chunk_size,
        generator=generator,
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=noise_aug_strength,
        num_frames=num_frames,
    ).frames[0]

    if progress_callback:
        progress_callback(0.9, "Exporting video...")

    # Export to video file
    export_to_video(frames, output_path, fps=fps)

    # Clean up GPU memory between jobs
    del pipe
    del frames
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "backend": "stable-video-diffusion",
        "fps": fps,
        "num_frames": num_frames,
        "motion_bucket_id": motion_bucket_id,
    }


def process_video_gen(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Generate AI video from image.

    Args:
        input_path: Path to input image
        output_path: Path for output video
        config: Configuration options
        progress_callback: Optional callback for progress updates

    Returns:
        dict with generation results
    """
    backend = config.get("backend", "replicate")

    # Check for API keys
    replicate_token = os.environ.get("REPLICATE_API_TOKEN")

    if backend == "replicate":
        if not replicate_token:
            raise ValueError(
                "REPLICATE_API_TOKEN environment variable is required for Replicate backend. "
                "Get your token at https://replicate.com/account/api-tokens"
            )
        return process_video_gen_replicate(input_path, output_path, config, progress_callback)

    elif backend == "svd":
        # Check VRAM
        try:
            import torch
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram < 20:
                    raise RuntimeError(
                        f"SVD requires ~24GB VRAM, only {vram:.1f}GB available. "
                        "Use backend='replicate' instead."
                    )
        except ImportError:
            pass

        return process_video_gen_svd(input_path, output_path, config, progress_callback)

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'replicate' or 'svd'.")


def process_video_gen_runway(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Generate video using Runway ML API (Gen-3 Alpha).
    Requires RUNWAY_API_TOKEN environment variable.
    """
    import requests
    import time

    api_token = os.environ.get("RUNWAY_API_TOKEN")
    if not api_token:
        raise ValueError("RUNWAY_API_TOKEN environment variable required")

    prompt = config.get("prompt", "")
    duration = config.get("duration", 5)  # 5 or 10 seconds

    if progress_callback:
        progress_callback(0.1, "Starting Runway generation...")

    # Read and encode image
    import base64
    with open(input_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    # Start generation task
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        "https://api.runwayml.com/v1/image_to_video",
        headers=headers,
        json={
            "promptImage": f"data:image/png;base64,{image_data}",
            "promptText": prompt,
            "duration": duration,
            "watermark": False,
        },
        timeout=30,
    )
    response.raise_for_status()
    task_id = response.json()["id"]

    if progress_callback:
        progress_callback(0.2, "Waiting for generation...")

    # Poll for completion
    while True:
        status_response = requests.get(
            f"https://api.runwayml.com/v1/tasks/{task_id}",
            headers=headers,
            timeout=30,
        )
        status_response.raise_for_status()
        status = status_response.json()

        if status["status"] == "SUCCEEDED":
            video_url = status["output"][0]
            break
        elif status["status"] == "FAILED":
            raise RuntimeError(f"Runway generation failed: {status.get('error', 'Unknown error')}")

        time.sleep(5)

        if progress_callback:
            progress = min(0.8, 0.2 + (status.get("progress", 0) * 0.6))
            progress_callback(progress, "Generating...")

    if progress_callback:
        progress_callback(0.9, "Downloading video...")

    # Download result with retry logic
    _max_retries = 3
    _last_err = None
    for _attempt in range(_max_retries):
        try:
            video_response = requests.get(video_url, stream=True, timeout=120)
            video_response.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in video_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            _last_err = None
            break
        except (requests.RequestException, IOError) as e:
            _last_err = e
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            if _attempt < _max_retries - 1:
                time.sleep(2 ** _attempt)
    if _last_err is not None:
        raise RuntimeError(f"Runway video download failed after {_max_retries} attempts: {_last_err}")

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "backend": "runway",
        "duration": duration,
        "prompt": prompt,
    }
