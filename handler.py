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
import subprocess
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


def upload_file(path: str, url: str, max_retries: int = 3) -> dict:
    """
    Upload file to presigned URL with retry logic and verification.
    Returns dict with upload status and details.
    Raises exception on failure after all retries.
    """
    ext = os.path.splitext(path)[1].lower()
    content_types = {
        '.zip': 'application/zip',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm',
        '.mkv': 'video/x-matroska',
        '.avi': 'video/x-msvideo',
        '.m4v': 'video/x-m4v',
    }
    content_type = content_types.get(ext, 'application/octet-stream')

    # Verify file exists and has content
    if not os.path.exists(path):
        raise FileNotFoundError(f"Output file not found: {path}")

    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValueError(f"Output file is empty: {path}")

    last_error = None
    for attempt in range(max_retries):
        try:
            with open(path, 'rb') as f:
                response = requests.put(
                    url,
                    data=f,
                    headers={'Content-Type': content_type},
                    timeout=600
                )
                response.raise_for_status()

            # Upload succeeded
            return {
                "uploaded": True,
                "size": file_size,
                "contentType": content_type,
                "attempts": attempt + 1
            }
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff: 1s, 2s, 4s
                wait_time = (2 ** attempt)
                import time
                time.sleep(wait_time)

    # All retries failed
    raise RuntimeError(f"Upload failed after {max_retries} attempts: {last_error}")


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


# ==================== Batch Mode Processor ====================

def get_gpu_info() -> dict:
    """Detect GPU type, count GPUs, and determine NVENC session limit."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            gpu_lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            gpu_count = len(gpu_lines)
            gpu_name = gpu_lines[0] if gpu_lines else 'Unknown'

            datacenter_keywords = ['A100', 'A6000', 'A5000', 'A4000', 'A4500', 'A40', 'A30', 'A10',
                                   'V100', 'T4', 'Quadro', 'Tesla', 'H100', 'L40', 'L4', 'RTX 6000', 'RTX 4090']
            is_datacenter = any(kw in gpu_name for kw in datacenter_keywords)
            gpu_type = 'datacenter' if is_datacenter else 'consumer'
            base_sessions = 12 if is_datacenter else 3
            nvenc_sessions = base_sessions * gpu_count

            print(f"[GPU Detection] Found {gpu_count} GPU(s): {gpu_name}")
            print(f"[GPU Detection] Type: {gpu_type}, Sessions/GPU: {base_sessions}, Total: {nvenc_sessions}")

            return {
                'gpu_name': gpu_name,
                'gpu_type': gpu_type,
                'gpu_count': gpu_count,
                'nvenc_sessions': nvenc_sessions
            }
    except Exception as e:
        print(f"[GPU Detection] Error: {e}")
    return {'gpu_name': 'Unknown', 'gpu_type': 'default', 'gpu_count': 1, 'nvenc_sessions': 2}


def download_files_parallel(urls: list, paths: list, max_workers: int = 10) -> list:
    """Download multiple files in parallel. Returns list of results."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(urls)

    def download_one(idx: int, url: str, path: str):
        try:
            download_file(url, path)
            return {"index": idx, "success": True, "path": path}
        except Exception as e:
            return {"index": idx, "success": False, "error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_one, i, u, p) for i, (u, p) in enumerate(zip(urls, paths))]
        for future in as_completed(futures):
            result = future.result()
            results[result["index"]] = result

    return results


def process_single_file_for_batch(args: tuple) -> dict:
    """
    Worker function for batch processing.
    Processes a single file with the specified tool.
    Supports multi-GPU via gpu_id parameter.
    """
    # Support both old (5-tuple) and new (6-tuple) format with gpu_id
    if len(args) == 6:
        input_path, output_path, tool, config, file_index, gpu_id = args
    else:
        input_path, output_path, tool, config, file_index = args
        gpu_id = 0

    try:
        input_ext = os.path.splitext(input_path)[1].lower()

        # Process based on tool type
        if tool == "spoofer":
            if process_spoofer is not None:
                # Pass gpu_id in config for multi-GPU support
                spoofer_config = {**config, "_gpu_id": gpu_id}
                print(f"[Batch Worker {file_index}] Processing on GPU {gpu_id}")
                result = process_spoofer(input_path, output_path, spoofer_config)
            else:
                return {"index": file_index, "status": "failed", "error": "Spoofer not available"}

        elif tool == "captioner":
            if process_captioner is not None:
                # Pass file index for batch caption matching
                captioner_config = {**config, "imageIndex": file_index}
                result = process_captioner(input_path, output_path, captioner_config)
            else:
                return {"index": file_index, "status": "failed", "error": "Captioner not available"}

        elif tool == "vignettes":
            if process_vignettes is not None:
                result = process_vignettes(input_path, output_path, config)
            else:
                return {"index": file_index, "status": "failed", "error": "Vignettes not available"}

        elif tool == "resize":
            if process_resize is not None:
                result = process_resize(input_path, output_path, config)
            else:
                return {"index": file_index, "status": "failed", "error": "Resize not available"}

        else:
            return {"index": file_index, "status": "failed", "error": f"Unknown tool: {tool}"}

        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return {"index": file_index, "status": "completed", "output_path": output_path, "result": result}
        else:
            return {"index": file_index, "status": "failed", "error": "Output file not created"}

    except Exception as e:
        return {"index": file_index, "status": "failed", "error": str(e)}


def process_batch_mode(job, job_input: dict) -> dict:
    """
    Process multiple files in parallel using multiple NVENC sessions.

    Input format:
    {
        "tool": "batch",
        "processor": "spoofer" | "captioner" | "vignettes" | "resize",
        "inputUrls": ["url1", "url2", ...],
        "outputUrls": ["url1", "url2", ...],
        "config": { tool-specific configuration }
    }

    Returns:
    {
        "status": "completed",
        "mode": "batch_parallel",
        "total": N,
        "completed": X,
        "failed": Y,
        "results": [...]
    }
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    processor = job_input.get("processor", "").lower()
    input_urls = job_input.get("inputUrls", [])
    output_urls = job_input.get("outputUrls", [])
    config = job_input.get("batchConfig", job_input.get("config", {}))
    job_id = job["id"]

    # Validate
    if not processor:
        return {"error": "Missing 'processor' parameter for batch mode"}
    if not input_urls or not output_urls:
        return {"error": "Missing 'inputUrls' or 'outputUrls' for batch mode"}
    if len(input_urls) != len(output_urls):
        return {"error": "inputUrls and outputUrls must have same length"}

    total_files = len(input_urls)

    # Detect GPU capabilities
    gpu_info = get_gpu_info()
    gpu_count = gpu_info.get('gpu_count', 1)
    max_parallel = gpu_info['nvenc_sessions']
    print(f"[Batch Mode] GPU: {gpu_info['gpu_name']}, Type: {gpu_info['gpu_type']}, GPUs: {gpu_count}, Max parallel: {max_parallel}")

    runpod.serverless.progress_update(job, {
        "progress": 5,
        "status": f"Batch mode: {total_files} files, {max_parallel} parallel"
    })

    temp_dir = tempfile.mkdtemp(prefix=f"farmium_batch_{job_id}_")

    try:
        # 1. Download all input files in parallel
        runpod.serverless.progress_update(job, {
            "progress": 10,
            "status": f"Downloading {total_files} files..."
        })

        input_paths = []
        output_paths = []
        for i, url in enumerate(input_urls):
            ext = get_file_extension(url, config)
            input_paths.append(os.path.join(temp_dir, f"input_{i}{ext}"))

            # Determine output extension
            if processor in ["spoofer", "captioner", "vignettes", "resize"]:
                output_ext = ext if ext in ['.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'] else '.jpg'
            else:
                output_ext = ext
            output_paths.append(os.path.join(temp_dir, f"output_{i}{output_ext}"))

        download_results = download_files_parallel(input_urls, input_paths, max_workers=10)
        failed_downloads = [r for r in download_results if not r.get("success")]
        if failed_downloads:
            print(f"[Batch Mode] {len(failed_downloads)} downloads failed")

        # 2. Process all files in parallel
        runpod.serverless.progress_update(job, {
            "progress": 30,
            "status": f"Processing {total_files} files with {max_parallel} sessions across {gpu_count} GPU(s)..."
        })

        # Prepare work items with GPU assignment (round-robin across GPUs)
        work_items = []
        for i in range(total_files):
            if download_results[i].get("success"):
                gpu_id = i % gpu_count  # Distribute across GPUs
                work_items.append((input_paths[i], output_paths[i], processor, config, i, gpu_id))

        completed = 0
        failed = 0
        results = [None] * total_files

        # Mark failed downloads
        for r in download_results:
            if not r.get("success"):
                results[r["index"]] = {"index": r["index"], "status": "failed", "error": f"Download failed: {r.get('error')}"}
                failed += 1

        # Detect if processing videos (use ThreadPoolExecutor for videos since FFmpeg is subprocess)
        VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv', '.m4v'}
        has_videos = any(os.path.splitext(input_paths[i])[1].lower() in VIDEO_EXTENSIONS
                        for i in range(total_files) if download_results[i].get("success"))

        # Process in parallel
        if work_items:
            if has_videos:
                # Use ThreadPoolExecutor for videos (FFmpeg is the subprocess that does the work)
                print(f"[Batch Mode] Using ThreadPoolExecutor for {len(work_items)} videos")
                from concurrent.futures import ThreadPoolExecutor
                ExecutorClass = ThreadPoolExecutor
                executor_kwargs = {"max_workers": min(max_parallel, len(work_items))}
            else:
                # Use ProcessPoolExecutor for images (CPU-bound PIL operations)
                print(f"[Batch Mode] Using ProcessPoolExecutor for {len(work_items)} images")
                ctx = multiprocessing.get_context('spawn')
                ExecutorClass = ProcessPoolExecutor
                executor_kwargs = {"max_workers": max_parallel, "mp_context": ctx}

            with ExecutorClass(**executor_kwargs) as executor:
                future_to_idx = {executor.submit(process_single_file_for_batch, item): item[4] for item in work_items}

                for future in as_completed(future_to_idx):
                    try:
                        result = future.result(timeout=600)  # 10 min timeout per file
                    except Exception as e:
                        # Handle timeout or other errors
                        idx = future_to_idx[future]
                        result = {"index": idx, "status": "failed", "error": str(e)}

                    idx = result["index"]
                    results[idx] = result

                    if result["status"] == "completed":
                        completed += 1
                    else:
                        failed += 1

                    # Update progress
                    progress = 30 + int(((completed + failed) / total_files) * 50)
                    runpod.serverless.progress_update(job, {
                        "progress": progress,
                        "status": f"Processed {completed + failed}/{total_files} files ({failed} failed)"
                    })

        # 3. Upload all output files in parallel
        runpod.serverless.progress_update(job, {
            "progress": 85,
            "status": f"Uploading {completed} processed files..."
        })

        files_to_upload = []
        urls_to_upload = []
        for i, result in enumerate(results):
            if result and result.get("status") == "completed" and result.get("output_path"):
                if os.path.exists(result["output_path"]):
                    files_to_upload.append(result["output_path"])
                    urls_to_upload.append(output_urls[i])

        if files_to_upload:
            upload_results = upload_files_parallel(files_to_upload, urls_to_upload, max_workers=10)
            upload_failed = [r for r in upload_results if not r.get("success")]
            if upload_failed:
                print(f"[Batch Mode] {len(upload_failed)} uploads failed")
                failed += len(upload_failed)
                completed -= len(upload_failed)

        runpod.serverless.progress_update(job, {
            "progress": 100,
            "status": "Completed"
        })

        return {
            "status": "completed",
            "mode": "batch_parallel",
            "gpu": gpu_info['gpu_name'],
            "gpu_type": gpu_info['gpu_type'],
            "parallel_sessions": max_parallel,
            "total": total_files,
            "completed": completed,
            "failed": failed,
            "results": results
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "mode": "batch_parallel"
        }

    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


# ==================== Pipeline Processor ====================

def upload_files_parallel(files: list, urls: list, max_workers: int = 10) -> list:
    """
    Upload multiple files in parallel to their respective presigned URLs.
    Returns list of {"index": i, "success": bool, "size": int, "error": str?}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(files)

    def upload_one(idx: int, file_path: str, url: str):
        try:
            result = upload_file(file_path, url)
            return {"index": idx, "success": True, "size": result.get("size", 0)}
        except Exception as e:
            return {"index": idx, "success": False, "error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, (f, u) in enumerate(zip(files, urls)):
            futures.append(executor.submit(upload_one, i, f, u))

        for future in as_completed(futures):
            result = future.result()
            results[result["index"]] = result

    return results


def process_pipeline(job, temp_dir: str, input_path: str, output_url: str, pipeline: list, progress_callback, output_urls: list = None) -> dict:
    """
    Process a pipeline of tools sequentially without intermediate uploads.

    Each tool's output becomes the next tool's input.

    If output_urls is provided (array of presigned URLs), uploads each file directly
    to its corresponding URL - much faster than ZIP + re-upload flow.

    If output_url is provided (single URL), creates ZIP and uploads (legacy mode).

    Args:
        job: RunPod job object for progress updates
        temp_dir: Temporary directory for processing
        input_path: Path to downloaded input file
        output_url: Presigned URL for final ZIP upload (legacy mode)
        pipeline: List of {"tool": "...", "config": {...}} dicts
        progress_callback: Callback for progress updates
        output_urls: Array of presigned URLs for direct individual uploads (optional)

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
                        # Collect all files from the batch output directory (images and videos)
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.mp4', '*.mov', '*.avi', '*.webm']:
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

    # ==================== DIRECT UPLOAD MODE ====================
    # If output_urls array is provided, upload each file directly (FAST)
    if output_urls and len(output_urls) >= len(current_files):
        runpod.serverless.progress_update(job, {
            "progress": 92,
            "status": "Uploading files directly"
        })

        # Upload files in parallel (10 concurrent uploads)
        upload_results = upload_files_parallel(
            current_files[:len(output_urls)],
            output_urls[:len(current_files)],
            max_workers=10
        )

        # Check for failures
        failed = [r for r in upload_results if not r.get("success")]
        if failed:
            return {
                "error": f"Failed to upload {len(failed)} files: {failed[0].get('error', 'unknown')}",
                "mode": "pipeline_direct",
                "steps": len(pipeline),
                "outputFiles": len(current_files),
                "uploadedFiles": len(current_files) - len(failed),
                "failedUploads": len(failed)
            }

        total_size = sum(r.get("size", 0) for r in upload_results)

        runpod.serverless.progress_update(job, {
            "progress": 100,
            "status": "Completed"
        })

        return {
            "status": "completed",
            "mode": "pipeline_direct",
            "steps": len(pipeline),
            "outputFiles": len(current_files),
            "uploadedFiles": len(current_files),
            "totalSize": total_size
        }

    # ==================== LEGACY ZIP MODE ====================
    # Fallback: Create ZIP and upload (for backwards compatibility)
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

    # Upload final ZIP with error handling
    runpod.serverless.progress_update(job, {
        "progress": 95,
        "status": "Uploading"
    })

    try:
        upload_result = upload_file(final_zip_path, output_url)
    except Exception as e:
        return {
            "error": f"Failed to upload pipeline output: {str(e)}",
            "mode": "pipeline",
            "steps": len(pipeline),
            "outputFiles": len(current_files)
        }

    return {
        "status": "completed",
        "mode": "pipeline",
        "steps": len(pipeline),
        "outputFiles": len(current_files),
        "outputSize": os.path.getsize(final_zip_path),
        "uploadAttempts": upload_result.get("attempts", 1)
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

    # ==================== BATCH MODE ====================
    # Process multiple files in parallel using multiple NVENC sessions
    # Optimized for datacenter GPUs (A5000, A6000) with unlimited NVENC sessions
    if tool == "batch":
        return process_batch_mode(job, job_input)

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

        # Direct upload mode: array of presigned URLs for individual files (FAST)
        output_urls = job_input.get("outputUrls", [])

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
            result = process_pipeline(
                job, temp_dir, input_path, output_url, pipeline, progress_callback,
                output_urls=output_urls if output_urls else None
            )

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

        output_size = os.path.getsize(output_path)
        if output_size == 0:
            return {"error": "Processing failed - output file is empty"}

        # Upload result with error handling
        runpod.serverless.progress_update(job, {
            "progress": 95,
            "status": "uploading"
        })

        try:
            upload_result = upload_file(output_path, output_url)
        except Exception as e:
            return {
                "error": f"Failed to upload output: {str(e)}",
                "tool": tool,
                "outputSize": output_size
            }

        # Success
        return {
            "status": "completed",
            "tool": tool,
            "outputSize": output_size,
            "uploadAttempts": upload_result.get("attempts", 1),
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
