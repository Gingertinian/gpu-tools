"""
RunPod Serverless Handler for Farmium GPU Tools

This handler processes GPU jobs for:
- Vignettes: Video overlay effects
- Spoofer: Image/video transformations for duplicate detection evasion
- Captioner: Add text captions to images/videos
- Resize: Resize/crop images/videos to target resolution
- Pic to Video: Convert images to static videos
- BG Remove: Remove background from images
- Upscale: AI upscaling with Real-ESRGAN
- Face Swap: Swap faces using InsightFace
- Video Gen: AI video generation from images

Usage:
    Deploy to RunPod Serverless with network volume containing tools/ directory
"""

import runpod
import os
import sys
import requests
import tempfile
import shutil
from pathlib import Path

# Add tools directory to path
WORKSPACE = os.environ.get('WORKSPACE', '/workspace')
sys.path.insert(0, os.path.join(WORKSPACE, 'tools'))

# ==================== Import Tool Processors ====================

# Vignettes
try:
    from vignettes.processor import process_vignettes
except ImportError:
    process_vignettes = None

# Spoofer
try:
    from spoofer.processor import process_spoofer
except ImportError:
    process_spoofer = None

try:
    from spoofer.processor_fast import process_spoofer_fast
except ImportError:
    process_spoofer_fast = None

# Captioner
try:
    from captioner.processor import process_captioner
except ImportError:
    process_captioner = None

# Resize
try:
    from resize.processor import process_resize
except ImportError:
    process_resize = None

# Pic to Video
try:
    from pic_to_video.processor import process_pic_to_video
except ImportError:
    process_pic_to_video = None

# BG Remove
try:
    from bg_remove.processor import process_bg_remove
except ImportError:
    process_bg_remove = None

# Upscale
try:
    from upscale.processor import process_upscale, process_upscale_video
except ImportError:
    process_upscale = None
    process_upscale_video = None

# Face Swap
try:
    from face_swap.processor import process_face_swap, process_face_swap_video
except ImportError:
    process_face_swap = None
    process_face_swap_video = None

# Video Gen
try:
    from video_gen.processor import process_video_gen
except ImportError:
    process_video_gen = None


# ==================== Helper Functions ====================

def download_file(url: str, path: str) -> None:
    """Download file from presigned URL"""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def upload_file(path: str, url: str) -> None:
    """Upload file to presigned URL"""
    ext = os.path.splitext(path)[1].lower()
    content_types = {
        '.zip': 'application/zip',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm',
    }
    content_type = content_types.get(ext, 'application/octet-stream')

    with open(path, 'rb') as f:
        response = requests.put(
            url,
            data=f,
            headers={'Content-Type': content_type},
            timeout=600
        )
        response.raise_for_status()


def create_progress_callback(job):
    """Create a progress callback that sends updates to RunPod"""
    def callback(progress: float, message: str = None):
        scaled_progress = int(10 + progress * 80)
        update = {"progress": scaled_progress}
        if message:
            update["status"] = message
        runpod.serverless.progress_update(job, update)
    return callback


def get_file_extension(url: str, config: dict) -> str:
    """Extract file extension from URL or config"""
    ext = config.get("inputExtension")
    if not ext:
        url_path = url.split('?')[0]
        ext = os.path.splitext(url_path)[1].lower() or ".jpg"
    return ext


def is_image(ext: str) -> bool:
    """Check if extension is an image format"""
    return ext.lower() in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']


def is_video(ext: str) -> bool:
    """Check if extension is a video format"""
    return ext.lower() in ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v']


# ==================== Main Handler ====================

def handler(job):
    """
    Main handler for GPU processing jobs

    Input format:
    {
        "tool": "vignettes" | "spoofer" | "captioner" | "resize" | "pic_to_video" |
                "bg_remove" | "upscale" | "face_swap" | "video_gen",
        "inputUrl": "presigned download URL",
        "outputUrl": "presigned upload URL",
        "config": { tool-specific configuration }
    }
    """
    job_input = job["input"]
    job_id = job["id"]

    tool = job_input.get("tool", "").lower()
    input_url = job_input.get("inputUrl")
    output_url = job_input.get("outputUrl")
    config = job_input.get("config", {})

    # Validate inputs
    if not tool:
        return {"error": "Missing 'tool' parameter"}
    if not input_url:
        return {"error": "Missing 'inputUrl' parameter"}
    if not output_url:
        return {"error": "Missing 'outputUrl' parameter"}

    # Create temp directory for this job
    temp_dir = tempfile.mkdtemp(prefix=f"farmium_{job_id}_")

    try:
        # Determine file extensions
        input_ext = get_file_extension(input_url, config)
        output_ext = config.get("outputExtension", input_ext)

        # Tool-specific output extension handling
        if tool == "spoofer":
            copies = config.get("copies") or config.get("options", {}).get("copies", 1)
            if copies > 1:
                output_ext = ".zip"
        elif tool == "bg_remove":
            output_ext = ".png"  # Always PNG for transparency
        elif tool == "pic_to_video":
            output_ext = ".mp4"
        elif tool == "video_gen":
            output_ext = ".mp4"

        input_path = os.path.join(temp_dir, f"input{input_ext}")
        output_path = os.path.join(temp_dir, f"output{output_ext}")

        # Download input file
        runpod.serverless.progress_update(job, {
            "progress": 5,
            "status": "downloading"
        })
        download_file(input_url, input_path)

        # Process based on tool type
        runpod.serverless.progress_update(job, {
            "progress": 10,
            "status": "processing"
        })

        progress_callback = create_progress_callback(job)
        result = {}

        # ==================== VIGNETTES ====================
        if tool == "vignettes":
            if process_vignettes is None:
                return {"error": "Vignettes processor not available"}
            result = process_vignettes(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== SPOOFER ====================
        elif tool == "spoofer":
            input_is_image = is_image(input_ext)

            try:
                if input_is_image and process_spoofer_fast is not None:
                    runpod.serverless.progress_update(job, {
                        "progress": 12,
                        "status": "processing (CPU fast mode)"
                    })
                    result = process_spoofer_fast(
                        input_path, output_path, config,
                        progress_callback=progress_callback
                    )
                elif process_spoofer is not None:
                    runpod.serverless.progress_update(job, {
                        "progress": 12,
                        "status": "processing (GPU mode)"
                    })
                    result = process_spoofer(
                        input_path, output_path, config,
                        progress_callback=progress_callback
                    )
                else:
                    return {"error": "Spoofer processor not available"}
            except Exception as e:
                import traceback
                return {
                    "error": f"Spoofer processing error: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "tool": tool,
                    "mode": "cpu_fast" if input_is_image else "gpu"
                }

        # ==================== CAPTIONER ====================
        elif tool == "captioner":
            if process_captioner is None:
                return {"error": "Captioner processor not available"}
            result = process_captioner(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== RESIZE ====================
        elif tool == "resize":
            if process_resize is None:
                return {"error": "Resize processor not available"}
            result = process_resize(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== PIC TO VIDEO ====================
        elif tool == "pic_to_video":
            if process_pic_to_video is None:
                return {"error": "Pic to Video processor not available"}
            output_path = os.path.join(temp_dir, "output.mp4")
            result = process_pic_to_video(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== BG REMOVE ====================
        elif tool == "bg_remove":
            if process_bg_remove is None:
                return {"error": "BG Remove processor not available. Install: pip install rembg"}
            output_path = os.path.join(temp_dir, "output.png")
            result = process_bg_remove(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== UPSCALE ====================
        elif tool == "upscale":
            if process_upscale is None:
                return {"error": "Upscale processor not available. Install: pip install realesrgan basicsr"}

            if is_video(input_ext):
                if process_upscale_video is None:
                    return {"error": "Video upscale not available"}
                result = process_upscale_video(
                    input_path, output_path, config,
                    progress_callback=progress_callback
                )
            else:
                result = process_upscale(
                    input_path, output_path, config,
                    progress_callback=progress_callback
                )

        # ==================== FACE SWAP ====================
        elif tool == "face_swap":
            if process_face_swap is None:
                return {"error": "Face Swap processor not available. Install: pip install insightface onnxruntime-gpu"}

            if is_video(input_ext):
                if process_face_swap_video is None:
                    return {"error": "Video face swap not available"}
                result = process_face_swap_video(
                    input_path, output_path, config,
                    progress_callback=progress_callback
                )
            else:
                result = process_face_swap(
                    input_path, output_path, config,
                    progress_callback=progress_callback
                )

        # ==================== VIDEO GEN ====================
        elif tool == "video_gen":
            if process_video_gen is None:
                return {"error": "Video Gen processor not available. Install: pip install replicate diffusers"}
            output_path = os.path.join(temp_dir, "output.mp4")
            result = process_video_gen(
                input_path, output_path, config,
                progress_callback=progress_callback
            )

        # ==================== UNKNOWN TOOL ====================
        else:
            return {"error": f"Unknown tool: {tool}"}

        # Check if output was created
        if not os.path.exists(output_path):
            return {"error": "Processing failed - no output file created"}

        # Upload result
        runpod.serverless.progress_update(job, {
            "progress": 95,
            "status": "uploading"
        })
        upload_file(output_path, output_url)

        # Success
        return {
            "status": "completed",
            "tool": tool,
            "outputSize": os.path.getsize(output_path),
            **result
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "tool": tool
        }

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# Start the serverless handler
runpod.serverless.start({"handler": handler})
