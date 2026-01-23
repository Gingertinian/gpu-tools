# Changelog

All notable changes to GPU Tools (RunPod Handlers) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Spoofer refactor with modular architecture (in progress)

---

## [1.1.0] - 2026-01-23

### Added
- **VIDEO_REFRAME Tool**: Convert horizontal videos to vertical (9:16) with blur background
- Random blur zones with different zoom/crop/hue for top and bottom
- PNG logo support for VIDEO_REFRAME
- Mode presets for Spoofer (light/balanced/aggressive)
- Automatic RunPod worker restart after deploy

### Changed
- VIDEO_REFRAME optimized: 15x faster blur using 1/4 resolution
- VIDEO_REFRAME uses boxblur O(1) instead of gblur O(n²)
- Batch mode uses 20 parallel workers (up from 5)
- Spoofer batch mode renamed "copies" to "variations"

### Fixed
- Video reframe logo positioning (use round() instead of int())
- Video reframe blur zones consistency across resolutions
- Horizontal video support in video_reframe
- Logo Y position centering for video and image
- Brightness/saturation/contrast config parsing
- CamelCase and snake_case config key support
- SAR, blur source, rotation with zoom issues
- CPU fallback for VIDEO_REFRAME
- Batch mode DOWNLOAD_WORKERS/UPLOAD_WORKERS undefined

---

## [1.0.0] - 2026-01-10

### Added
- **Spoofer Tool**: Hash evasion transforms for images and videos
  - Batch processing with multiprocessing
  - Intelligent CPU/GPU routing
  - Spatial transforms (crop, rotation, resize)
  - Tonal adjustments (brightness, contrast, saturation)
  - Visual effects (noise, tint)
- **Captioner Tool**: Add stylized captions to images/videos
  - TikTok fonts (Bold, Light Italic)
  - Face detection with MediaPipe to avoid faces
  - Special formatting (##Title##, &&& line breaks)
  - Batch mode with caption cycling
  - Random tilt and center-every-N features
- **Vignettes Tool**: Animated overlays on videos
- RunPod Serverless handler architecture
- Cloudflare R2 integration for file storage
- GitHub Actions auto-deploy on push to master

---

[Unreleased]: https://github.com/Gingertinian/gpu-tools/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/Gingertinian/gpu-tools/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Gingertinian/gpu-tools/releases/tag/v1.0.0
