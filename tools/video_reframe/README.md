# Video Reframe Tool

Converts horizontal/any aspect videos to vertical (9:16) with blur areas and logo overlay.

## Setup

### Logo Files

Copy the logo SVG files to `gpu-tools/assets/logos/`:

```bash
# From: C:\Users\erudito\Downloads\Telegram Desktop\
# Copy "Asset 2.svg" to "gpu-tools/assets/logos/farmium_icon.svg"
# Copy "Asset 2555.svg" to "gpu-tools/assets/logos/farmium_full.svg"
```

**Manual steps required:**
1. Navigate to `C:\Users\erudito\Downloads\Telegram Desktop\`
2. Copy `Asset 2.svg` → `C:\Users\erudito\Documents\Proyect Farmium\gpu-tools\assets\logos\farmium_icon.svg`
3. Copy `Asset 2555.svg` → `C:\Users\erudito\Documents\Proyect Farmium\gpu-tools\assets\logos\farmium_full.svg`

### Dependencies

The tool requires:
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- cairosvg (for SVG to PNG conversion)
- FFmpeg (system binary)

Install cairosvg if not already installed:
```bash
pip install cairosvg
```

## Configuration

```python
config = {
    'logoName': 'farmium_icon',  # or 'farmium_full' or custom path
    'logoSize': 50,  # percentage of video width
    'logoPosition': {'x': 0.5, 'y': 0.85},  # normalized 0-1
    'aspectRatio': [9, 16],  # target aspect ratio
    'blurIntensity': 25,  # gaussian blur strength
    'randomizeEffects': True,  # apply random tilt/color shift
    'tiltRange': 2,  # max rotation angle (degrees)
    'colorShiftRange': 10,  # max hue shift
    'brightness': 1.0,  # brightness multiplier
    'saturation': 1.0,  # saturation multiplier
    'contrast': 1.0,  # contrast multiplier
}
```

## Processing Logic

1. **Input**: Any aspect ratio video
2. **Output**: 9:16 vertical video (1080x1920 default)
3. **Content**: Original video scaled to fit width, centered vertically
4. **Blur Areas**:
   - Top and bottom areas filled with blurred/zoomed version of content
   - Extracted from 70% of content area for better quality
   - Effects: gaussian blur, color shift, tilt, brightness/saturation adjustments
5. **Logo**: SVG converted to PNG and overlaid at specified position with alpha blending

## Encoder

- **GPU (NVENC)**: If available, uses `h264_nvenc` for fast encoding
- **CPU Fallback**: Uses `libx264` with `ultrafast` preset

## Example Usage

```python
from video_reframe.processor import process_video_reframe

result = process_video_reframe(
    input_path="/path/to/horizontal_video.mp4",
    output_path="/path/to/vertical_output.mp4",
    config={
        'logoName': 'farmium_icon',
        'logoSize': 50,
        'logoPosition': {'x': 0.5, 'y': 0.85},
        'aspectRatio': [9, 16],
        'blurIntensity': 25,
        'randomizeEffects': True,
    },
    progress_callback=lambda p, msg: print(f"{int(p*100)}%: {msg}")
)

print(result)
# {
#   "status": "completed",
#   "outputPath": "/path/to/vertical_output.mp4",
#   "outputSize": 15728640,
#   "dimensions": "1080x1920",
#   "framesProcessed": 300
# }
```

## Notes

- Based on `video_processor_v2.py` patterns from Radioactive Test
- Uses FFmpeg piping for efficient frame processing
- Caches blur frames when `randomizeEffects=False` for faster processing
- Logo alpha blending preserves transparency
- Supports custom logo paths in addition to built-in logos
