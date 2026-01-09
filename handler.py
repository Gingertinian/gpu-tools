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
import zipfile
from pathlib import Path

# ==================== Constants ====================

# Already-compressed file formats (ZIP_STORED = no compression, saves CPU)
COMPRESSED_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.webp', '.gif',  # Images (already compressed)
    '.mp4', '.mov', '.webm', '.mkv', '.avi', '.m4v',  # Videos (already compressed)
    '.mp3', '.aac', '.ogg', '.flac',  # Audio (already compressed)
    '.zip', '.gz', '.bz2', '.xz', '.7z',  # Archives
}

# Larger chunk size for faster downloads (1MB instead of 8KB)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB

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

def get_zip_compression(filepath: str) -> int:
    """
    Get optimal ZIP compression method for a file.
    Returns ZIP_STORED for already-compressed formats (saves CPU).
    Returns ZIP_DEFLATED for uncompressed formats.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in COMPRESSED_EXTENSIONS:
        return zipfile.ZIP_STORED  # No compression - file already compressed
    return zipfile.ZIP_DEFLATED  # Compress uncompressed files


def download_file(url: str, path: str) -> None:
    """Download file from presigned URL with optimized chunk size"""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
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


# ==================== Pipeline Processor ====================

def process_pipeline(job, temp_dir: str, input_path: str, output_url: str, pipeline: list, progress_callback) -> dict:
    """
    Process a pipeline of tools sequentially without intermediate uploads.

    Each tool's output becomes the next tool's input.
    Only the final output is uploaded to R2.

    Args:
        job: RunPod job object for progress updates
        temp_dir: Temporary directory for processing
        input_path: Path to downloaded input file
        output_url: Presigned URL for final upload
        pipeline: List of {"tool": "...", "config": {...}} dicts
        progress_callback: Callback for progress updates

    Returns:
        dict with status and results
    """
    import zipfile
    from glob import glob

    if not pipeline or len(pipeline) == 0:
        return {"error": "Pipeline is empty"}

    # Track current files being processed
    # Start with the single input file
    current_files = [input_path]
    total_steps = len(pipeline)

    for step_idx, step in enumerate(pipeline):
        tool = step.get("tool", "").lower()
        config = step.get("config", {})

        step_progress_base = int((step_idx / total_steps) * 80) + 10

        runpod.serverless.progress_update(job, {
            "progress": step_progress_base,
            "status": f"Step {step_idx + 1}/{total_steps}: {tool}"
        })

        # Create output directory for this step
        step_output_dir = os.path.join(temp_dir, f"step_{step_idx}_{tool}")
        os.makedirs(step_output_dir, exist_ok=True)

        next_files = []

        # Process each current file through this tool
        for file_idx, current_file in enumerate(current_files):
            input_ext = os.path.splitext(current_file)[1].lower()

            # Determine output extension based on tool
            if tool == "bg_remove":
                output_ext = ".png"
            elif tool in ["pic_to_video", "video_gen"]:
                output_ext = ".mp4"
            else:
                output_ext = input_ext if input_ext else ".jpg"

            # For spoofer with copies > 1, output is a directory of files
            copies = config.get("copies") or config.get("options", {}).get("copies", 1)
            is_batch_spoofer = tool == "spoofer" and copies > 1

            if is_batch_spoofer:
                # Spoofer batch mode: output to a subdirectory
                spoofer_output_dir = os.path.join(step_output_dir, f"batch_{file_idx}")
                os.makedirs(spoofer_output_dir, exist_ok=True)
                output_path = spoofer_output_dir  # Pass directory, not file

                # Modify config to output individual files, not ZIP
                batch_config = {**config, "outputMode": "directory", "outputDir": spoofer_output_dir}
            else:
                output_path = os.path.join(step_output_dir, f"output_{file_idx}{output_ext}")
                batch_config = config

            # Create step-specific progress callback
            def step_callback(progress: float, message: str = None):
                file_progress = (file_idx + progress) / len(current_files)
                overall = step_progress_base + int(file_progress * (80 / total_steps))
                update = {"progress": min(overall, 90)}
                if message:
                    update["status"] = f"Step {step_idx + 1}: {tool} - {message}"
                runpod.serverless.progress_update(job, update)

            # Process based on tool type
            try:
                if tool == "spoofer":
                    input_is_image = is_image(input_ext)
                    if input_is_image and process_spoofer_fast is not None:
                        result = process_spoofer_fast(current_file, output_path, batch_config, progress_callback=step_callback)
                    elif process_spoofer is not None:
                        result = process_spoofer(current_file, output_path, batch_config, progress_callback=step_callback)
                    else:
                        return {"error": "Spoofer processor not available"}

                    # Collect output files
                    if is_batch_spoofer:
                        # Collect all files from the batch output directory
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                            next_files.extend(glob(os.path.join(spoofer_output_dir, ext)))
                    else:
                        if os.path.exists(output_path):
                            next_files.append(output_path)

                elif tool == "captioner":
                    if process_captioner is None:
                        return {"error": "Captioner processor not available"}

                    # For captioner, pass the image index for batch caption matching
                    captioner_config = {**batch_config, "imageIndex": file_idx}
                    result = process_captioner(current_file, output_path, captioner_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "resize":
                    if process_resize is None:
                        return {"error": "Resize processor not available"}
                    result = process_resize(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "vignettes":
                    if process_vignettes is None:
                        return {"error": "Vignettes processor not available"}
                    result = process_vignettes(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "bg_remove":
                    if process_bg_remove is None:
                        return {"error": "BG Remove processor not available"}
                    result = process_bg_remove(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "upscale":
                    if process_upscale is None:
                        return {"error": "Upscale processor not available"}
                    if is_video(input_ext) and process_upscale_video:
                        result = process_upscale_video(current_file, output_path, batch_config, progress_callback=step_callback)
                    else:
                        result = process_upscale(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "face_swap":
                    if process_face_swap is None:
                        return {"error": "Face Swap processor not available"}
                    if is_video(input_ext) and process_face_swap_video:
                        result = process_face_swap_video(current_file, output_path, batch_config, progress_callback=step_callback)
                    else:
                        result = process_face_swap(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "pic_to_video":
                    if process_pic_to_video is None:
                        return {"error": "Pic to Video processor not available"}
                    result = process_pic_to_video(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                elif tool == "video_gen":
                    if process_video_gen is None:
                        return {"error": "Video Gen processor not available"}
                    result = process_video_gen(current_file, output_path, batch_config, progress_callback=step_callback)
                    if os.path.exists(output_path):
                        next_files.append(output_path)

                else:
                    return {"error": f"Unknown tool in pipeline: {tool}"}

            except Exception as e:
                import traceback
                return {
                    "error": f"Pipeline step {step_idx + 1} ({tool}) failed: {str(e)}",
                    "traceback": traceback.format_exc(),
                    "step": step_idx,
                    "tool": tool
                }

        # Update current files for next step
        if not next_files:
            return {"error": f"Pipeline step {step_idx + 1} ({tool}) produced no output files"}

        current_files = next_files

    # Create final ZIP with all output files
    runpod.serverless.progress_update(job, {
        "progress": 92,
        "status": "Creating ZIP"
    })

    final_zip_path = os.path.join(temp_dir, "pipeline_output.zip")
    with zipfile.ZipFile(final_zip_path, 'w') as zf:
        for idx, file_path in enumerate(current_files):
            ext = os.path.splitext(file_path)[1]
            arcname = f"output_{idx + 1:04d}{ext}"
            # Use smart compression: ZIP_STORED for already-compressed files
            compression = get_zip_compression(file_path)
            zf.write(file_path, arcname, compress_type=compression)

    # Upload final ZIP
    runpod.serverless.progress_update(job, {
        "progress": 95,
        "status": "Uploading"
    })
    upload_file(final_zip_path, output_url)

    return {
        "status": "completed",
        "mode": "pipeline",
        "steps": len(pipeline),
        "outputFiles": len(current_files),
        "outputSize": os.path.getsize(final_zip_path)
    }


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

    # ==================== PIPELINE MODE ====================
    # Process entire workflow in one job without intermediate uploads
    if tool == "pipeline":
        pipeline = job_input.get("pipeline", [])
        if not pipeline:
            return {"error": "Missing 'pipeline' parameter for pipeline mode"}

        temp_dir = tempfile.mkdtemp(prefix=f"farmium_{job_id}_")
        try:
            # Download input
            runpod.serverless.progress_update(job, {
                "progress": 5,
                "status": "downloading"
            })
            input_ext = get_file_extension(input_url, config)
            input_path = os.path.join(temp_dir, f"input{input_ext}")
            download_file(input_url, input_path)

            # Process pipeline
            progress_callback = create_progress_callback(job)
            result = process_pipeline(job, temp_dir, input_path, output_url, pipeline, progress_callback)

            return result

        except Exception as e:
            import traceback
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "tool": "pipeline"
            }
        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

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
