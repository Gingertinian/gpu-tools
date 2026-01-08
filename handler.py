"""
RunPod Serverless Handler for Farmium GPU Tools

This handler processes GPU jobs for:
- Vignettes: Video overlay effects
- Spoofer: Image/video transformations for duplicate detection evasion
- Captioner: Add text captions to images/videos

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

# Import tool processors (will be available after setup)
try:
    from vignettes.processor import process_vignettes
except ImportError:
    process_vignettes = None

try:
    from spoofer.processor import process_spoofer
except ImportError:
    process_spoofer = None

try:
    from captioner.processor import process_captioner
except ImportError:
    process_captioner = None


def download_file(url: str, path: str) -> None:
    """Download file from presigned URL"""
    response = requests.get(url, stream=True, timeout=300)
    response.raise_for_status()

    with open(path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def upload_file(path: str, url: str) -> None:
    """Upload file to presigned URL"""
    # Determine content type based on extension
    ext = os.path.splitext(path)[1].lower()
    content_types = {
        '.zip': 'application/zip',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
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
        # Map progress 0-1 to 10-90 (leaving room for download/upload)
        scaled_progress = int(10 + progress * 80)
        update = {"progress": scaled_progress}
        if message:
            update["status"] = message
        runpod.serverless.progress_update(job, update)
    return callback


def handler(job):
    """
    Main handler for GPU processing jobs

    Input format:
    {
        "tool": "vignettes" | "spoofer" | "captioner",
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
        # Determine file extension from URL or config
        input_ext = config.get("inputExtension", ".mp4")
        output_ext = config.get("outputExtension", ".mp4")

        # For spoofer batch mode (copies > 1), output is always ZIP
        if tool == "spoofer":
            copies = config.get("options", {}).get("copies", 1)
            if copies > 1:
                output_ext = ".zip"

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

        if tool == "vignettes":
            if process_vignettes is None:
                return {"error": "Vignettes processor not available"}
            result = process_vignettes(
                input_path,
                output_path,
                config,
                progress_callback=progress_callback
            )

        elif tool == "spoofer":
            if process_spoofer is None:
                return {"error": "Spoofer processor not available"}
            try:
                result = process_spoofer(
                    input_path,
                    output_path,
                    config,
                    progress_callback=progress_callback
                )
            except Exception as spoofer_error:
                import traceback
                return {
                    "error": f"Spoofer processing error: {str(spoofer_error)}",
                    "traceback": traceback.format_exc(),
                    "tool": tool
                }

        elif tool == "captioner":
            if process_captioner is None:
                return {"error": "Captioner processor not available"}
            result = process_captioner(
                input_path,
                output_path,
                config,
                progress_callback=progress_callback
            )

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
        return {
            "error": str(e),
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
