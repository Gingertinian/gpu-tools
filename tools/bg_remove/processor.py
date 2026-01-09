"""
Background Removal Processor

Uses rembg library with U2-Net model for high-quality background removal.
Supports images only (videos would need frame-by-frame processing).

Config options:
- model: str = "u2net" | "u2netp" | "u2net_human_seg" | "silueta"
- alpha_matting: bool = False (enables edge refinement)
- alpha_matting_foreground_threshold: int = 240
- alpha_matting_background_threshold: int = 10
- bgcolor: list = None (replacement background color [R,G,B,A])
"""

import os
from typing import Callable, Optional
from PIL import Image
import io


def process_bg_remove(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Remove background from image using rembg.

    Args:
        input_path: Path to input image
        output_path: Path for output image (PNG for transparency)
        config: Processing configuration
        progress_callback: Optional callback for progress updates

    Returns:
        dict with processing results
    """
    from rembg import remove, new_session

    if progress_callback:
        progress_callback(0.1, "Loading model...")

    # Get config options
    model_name = config.get("model", "u2net")
    alpha_matting = config.get("alpha_matting", False)
    fg_threshold = config.get("alpha_matting_foreground_threshold", 240)
    bg_threshold = config.get("alpha_matting_background_threshold", 10)
    bgcolor = config.get("bgcolor", None)  # [R, G, B, A] or None for transparent

    # Create session with specified model
    session = new_session(model_name)

    if progress_callback:
        progress_callback(0.3, "Loading image...")

    # Load input image
    with open(input_path, "rb") as f:
        input_data = f.read()

    if progress_callback:
        progress_callback(0.4, "Removing background...")

    # Process with rembg
    output_data = remove(
        input_data,
        session=session,
        alpha_matting=alpha_matting,
        alpha_matting_foreground_threshold=fg_threshold,
        alpha_matting_background_threshold=bg_threshold,
        bgcolor=tuple(bgcolor) if bgcolor else None,
    )

    if progress_callback:
        progress_callback(0.8, "Saving result...")

    # Ensure output is PNG for transparency support
    output_path_png = output_path
    if not output_path.lower().endswith('.png'):
        output_path_png = os.path.splitext(output_path)[0] + '.png'

    # Save output
    with open(output_path_png, "wb") as f:
        f.write(output_data)

    # Get output dimensions
    img = Image.open(io.BytesIO(output_data))
    width, height = img.size

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "model": model_name,
        "alpha_matting": alpha_matting,
        "output_width": width,
        "output_height": height,
        "has_transparency": True,
    }


def process_bg_remove_batch(
    input_paths: list,
    output_dir: str,
    config: dict,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Process multiple images with background removal.
    More efficient as it reuses the model session.
    """
    from rembg import remove, new_session
    import zipfile

    model_name = config.get("model", "u2net")
    alpha_matting = config.get("alpha_matting", False)
    fg_threshold = config.get("alpha_matting_foreground_threshold", 240)
    bg_threshold = config.get("alpha_matting_background_threshold", 10)
    bgcolor = config.get("bgcolor", None)

    if progress_callback:
        progress_callback(0.05, "Loading model...")

    # Create session once for all images
    session = new_session(model_name)

    results = []
    total = len(input_paths)

    for i, input_path in enumerate(input_paths):
        if progress_callback:
            progress = 0.1 + (i / total) * 0.8
            progress_callback(progress, f"Processing {i+1}/{total}...")

        with open(input_path, "rb") as f:
            input_data = f.read()

        output_data = remove(
            input_data,
            session=session,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=fg_threshold,
            alpha_matting_background_threshold=bg_threshold,
            bgcolor=tuple(bgcolor) if bgcolor else None,
        )

        # Save to output directory
        basename = os.path.basename(input_path)
        name, _ = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name}.png")

        with open(output_path, "wb") as f:
            f.write(output_data)

        results.append(output_path)

    if progress_callback:
        progress_callback(0.95, "Creating ZIP...")

    # Create ZIP of all outputs
    zip_path = os.path.join(output_dir, "bg_removed.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for path in results:
            zf.write(path, os.path.basename(path))

    if progress_callback:
        progress_callback(1.0, "Complete")

    return {
        "processed_count": len(results),
        "model": model_name,
        "zip_path": zip_path,
    }
