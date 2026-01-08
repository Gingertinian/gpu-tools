# GPU Tools - RunPod Serverless

GPU processing tools for image/video transformations.

## Tools

- **Spoofer**: Image/video transformations with CPU fast mode for images
- **Vignettes**: Video overlay effects
- **Captioner**: Add text captions

## Quick Start

1. Create endpoint at https://runpod.io/console/serverless
2. Use image: `ghcr.io/gingertinian/gpu-tools:latest`
3. Select GPU (RTX 3090 recommended)
4. Set env vars: `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`

## API

```json
{
  "input": {
    "tool": "spoofer",
    "inputUrl": "presigned-download-url",
    "outputUrl": "presigned-upload-url",
    "config": {}
  }
}
```
