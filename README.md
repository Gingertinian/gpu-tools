# RunPod GPU Processing Setup

This directory contains the RunPod serverless handler for Farmium GPU tools.

## Setup Instructions

### 1. Create RunPod Account
1. Go to https://runpod.io
2. Create account and add credits ($25 minimum to start)

### 2. Create Network Volume
1. Go to Storage > Network Volumes
2. Create new volume:
   - Name: `farmium-tools`
   - Size: 100GB
   - Datacenter: Choose one with RTX 4090 availability
3. Note the volume ID

### 3. Upload Tools to Volume
1. Start a temporary GPU pod with the volume attached
2. SSH into the pod
3. Create tools directory structure:
```bash
mkdir -p /workspace/tools/{vignettes,spoofer,captioner}
```
4. Copy tool code from `C:\Users\erudito\Documents\Apps\`:
   - Vignettes → `/workspace/tools/vignettes/`
   - Spoofer → `/workspace/tools/spoofer/`
   - Captioner → `/workspace/tools/captioner/`
5. Create `processor.py` wrapper for each tool (see examples below)

### 4. Build and Push Docker Image
```bash
# Build locally
docker build -t your-dockerhub/farmium-runpod:latest .

# Push to DockerHub
docker push your-dockerhub/farmium-runpod:latest
```

### 5. Create Serverless Endpoint
1. Go to Serverless > Endpoints
2. Create new endpoint:
   - Name: `farmium-gpu`
   - Docker Image: `your-dockerhub/farmium-runpod:latest`
   - GPU: RTX 4090 (or L40 for video generation)
   - Volume: Attach `farmium-tools` at `/workspace`
   - Min Workers: 0
   - Max Workers: 3
   - Idle Timeout: 5 seconds
   - FlashBoot: Enabled
3. Copy the Endpoint ID

### 6. Configure Backend
Add to `.env`:
```env
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
```

## Tool Processor Wrapper Examples

Each tool needs a `processor.py` that exposes a standard interface:

### vignettes/processor.py
```python
def process_vignettes(input_path, output_path, config, progress_callback=None):
    """
    Process video with vignette effects

    Args:
        input_path: Path to input video
        output_path: Path for output video
        config: Dict with vignette_strength, overlay_type, etc.
        progress_callback: Optional callback(progress: float, message: str)

    Returns:
        Dict with processing metadata
    """
    # Import and call your existing vignette processing code
    from app_vignette import process_video_ffmpeg

    result = process_video_ffmpeg(
        input_path,
        output_path,
        vignette_strength=config.get('vignetteStrength', 2),
        overlay_type=config.get('overlayType', 'pinwheel'),
        # ... other params
    )

    return {"processed": True}
```

### spoofer/processor.py
```python
def process_spoofer(input_path, output_path, config, progress_callback=None):
    """
    Apply transformations to evade duplicate detection
    """
    # Your spoofer logic here
    return {"transformed": True, "hashDistance": 42}
```

### captioner/processor.py
```python
def process_captioner(input_path, output_path, config, progress_callback=None):
    """
    Add captions to image or video
    """
    # Your captioner logic here
    return {"captioned": True}
```

## Testing Locally

```bash
# Test with runpodctl
pip install runpod

# Create test input
cat > test_input.json << EOF
{
    "input": {
        "tool": "vignettes",
        "inputUrl": "https://example.com/test.mp4",
        "outputUrl": "https://example.com/upload",
        "config": {
            "vignetteStrength": 2
        }
    }
}
EOF

# Run locally (requires GPU)
python handler.py --test_input test_input.json
```

## Cost Estimates

| GPU | Price/Hour | Typical Job | Cost/Job |
|-----|-----------|-------------|----------|
| RTX 4090 | $0.69 | 3 min | $0.035 |
| L40 | $0.99 | 5 min | $0.083 |

Network Volume: $0.07/GB/month = $7/month for 100GB
