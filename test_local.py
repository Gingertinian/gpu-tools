#!/usr/bin/env python3
"""
Local test script for GPU tool processors.

Usage:
    python test_local.py spoofer input.jpg output.jpg
    python test_local.py vignettes input.mp4 output.mp4
    python test_local.py captioner input.mp4 output.mp4
"""

import sys
import os

# Add tools to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))


def test_spoofer(input_path: str, output_path: str):
    """Test spoofer processor."""
    from spoofer.processor import process_spoofer

    config = {
        'spatial': {
            'crop': 2,
            'microResize': 1,
            'rotation': 0.3
        },
        'tonal': {
            'brightness': 3,
            'contrast': 3,
            'saturation': 5
        },
        'visual': {
            'noise': 3,
            'quality': 92
        }
    }

    def progress(p, msg):
        print(f"[{int(p * 100):3d}%] {msg}")

    result = process_spoofer(input_path, output_path, config, progress)
    print(f"\nResult: {result}")


def test_vignettes(input_path: str, output_path: str):
    """Test vignettes processor."""
    from vignettes.processor import process_vignettes

    config = {
        'overlayType': 'vignette',
        'intensity': 50,
        'color': '#000000'
    }

    def progress(p, msg):
        print(f"[{int(p * 100):3d}%] {msg}")

    result = process_vignettes(input_path, output_path, config, progress)
    print(f"\nResult: {result}")


def test_captioner(input_path: str, output_path: str):
    """Test captioner processor."""
    from captioner.processor import process_captioner

    config = {
        'text': 'Hello World!',
        'position': 'bottom',
        'fontSize': 48,
        'fontFamily': 'Arial',
        'color': '#FFFFFF',
        'strokeColor': '#000000',
        'strokeWidth': 2,
        'shadow': True
    }

    def progress(p, msg):
        print(f"[{int(p * 100):3d}%] {msg}")

    result = process_captioner(input_path, output_path, config, progress)
    print(f"\nResult: {result}")


def test_video_reframe(input_path: str, output_path: str):
    """Test video_reframe processor."""
    from video_reframe.processor_gpu import process_video_reframe

    config = {
        'aspectRatio': '9:16',
        'blurIntensity': 25,
        'brightness': 0,
        'saturation': 0,
        'contrast': 0,
        'forceBlur': 0,
        'logoName': 'none',
        'logoSize': 15,
    }

    def progress(p, msg):
        print(f"[{int(p * 100):3d}%] {msg}")

    result = process_video_reframe(input_path, output_path, config, progress)
    print(f"\nResult: {result}")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        print("\nAvailable tools: spoofer, vignettes, captioner, video_reframe")
        sys.exit(1)

    tool = sys.argv[1].lower()
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Testing {tool}...")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print()

    if tool == 'spoofer':
        test_spoofer(input_path, output_path)
    elif tool == 'vignettes':
        test_vignettes(input_path, output_path)
    elif tool == 'captioner':
        test_captioner(input_path, output_path)
    elif tool == 'video_reframe':
        test_video_reframe(input_path, output_path)
    else:
        print(f"Unknown tool: {tool}")
        print("Available: spoofer, vignettes, captioner, video_reframe")
        sys.exit(1)

    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        print(f"\n[OK] Output created: {output_path} ({size:,} bytes)")
    else:
        print(f"\n[FAIL] Output not created!")
        sys.exit(1)


if __name__ == '__main__':
    main()
