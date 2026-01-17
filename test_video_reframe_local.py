"""
Local test for video_reframe processor.
Tests both image and video processing without RunPod.

Run: python test_video_reframe_local.py
"""

import os
import sys
import time
import tempfile
import shutil
import requests

# Add tools directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

from video_reframe.processor_gpu import process_video_reframe, is_image_file, is_video_file

# Test images
TEST_IMAGE_URL = "https://picsum.photos/800/600"
TEST_IMAGE_LOCAL = r'C:\Users\erudito\Pictures\Screenshots\Screenshot 2026-01-02 144608.png'

OUTPUT_DIR = r'C:\Users\erudito\Downloads\reframe_tests'

def download_test_image(temp_dir):
    """Download a test image from the internet"""
    print("Downloading test image...")
    response = requests.get(TEST_IMAGE_URL, timeout=30)
    response.raise_for_status()

    # Save as jpg
    img_path = os.path.join(temp_dir, "test_input.jpg")
    with open(img_path, 'wb') as f:
        f.write(response.content)

    print(f"Downloaded test image: {img_path} ({len(response.content)} bytes)")
    return img_path


def test_image_reframe():
    """Test reframe with image input"""
    print("\n" + "=" * 60)
    print("  TEST 1: Image Reframe")
    print("=" * 60)

    # Create temp directories
    temp_dir = tempfile.mkdtemp(prefix='reframe_test_')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Use local image if exists, otherwise download
        if os.path.exists(TEST_IMAGE_LOCAL):
            input_path = TEST_IMAGE_LOCAL
            print(f"Using local image: {input_path}")
        else:
            input_path = download_test_image(temp_dir)

        # Output path - handler sets this to .mp4 even for images
        output_path = os.path.join(temp_dir, "output.mp4")

        # Test config with various settings
        config = {
            'aspectRatio': '9:16',
            'logoName': 'farmium_full',
            'logoSize': 20,
            'logoPositionX': 0.3,  # Left side
            'logoPositionY': 0.15,  # Near top
            'topBlurPercent': 30,
            'bottomBlurPercent': 15,
            'blurIntensity': 25,
            'brightness': 5,
            'saturation': 10,
            'contrast': 0,
        }

        print(f"\nConfig:")
        for k, v in config.items():
            print(f"  {k}: {v}")

        print(f"\nProcessing...")
        start_time = time.time()

        def progress_callback(progress, message=None):
            msg = f" - {message}" if message else ""
            print(f"  Progress: {progress*100:.1f}%{msg}")

        result = process_video_reframe(input_path, output_path, config, progress_callback)

        elapsed = time.time() - start_time
        print(f"\nResult:")
        for k, v in result.items():
            print(f"  {k}: {v}")

        # Check actual output path from result
        actual_output = result.get('outputPath', output_path)

        print(f"\nExpected output (handler): {output_path}")
        print(f"Actual output (processor): {actual_output}")

        if os.path.exists(actual_output):
            size = os.path.getsize(actual_output)
            print(f"\n[SUCCESS] Output file exists: {actual_output}")
            print(f"  Size: {size:,} bytes")

            # Copy to output directory for inspection
            final_path = os.path.join(OUTPUT_DIR, f"reframe_image_test_{int(time.time())}.jpg")
            shutil.copy(actual_output, final_path)
            print(f"  Saved to: {final_path}")
        else:
            print(f"\n[FAILED] Output file NOT found: {actual_output}")
            if os.path.exists(output_path):
                print(f"  But original path exists: {output_path}")

            # List all files in temp dir
            print(f"\n  Files in temp dir:")
            for f in os.listdir(temp_dir):
                fpath = os.path.join(temp_dir, f)
                print(f"    {f} ({os.path.getsize(fpath):,} bytes)")

        print(f"\nElapsed time: {elapsed:.2f}s")

        return os.path.exists(actual_output)

    finally:
        # Cleanup temp dir
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def test_handler_simulation():
    """Simulate what handler.py does to verify the fix"""
    print("\n" + "=" * 60)
    print("  TEST 2: Handler Simulation")
    print("=" * 60)

    temp_dir = tempfile.mkdtemp(prefix='handler_sim_')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Download test image
        input_path = download_test_image(temp_dir)

        # Handler sets output_path to .mp4 even for images
        output_path = os.path.join(temp_dir, "output.mp4")

        config = {
            'aspectRatio': '9:16',
            'logoName': 'farmium_full',
            'logoSize': 25,
            'logoPositionX': 0.5,  # Center
            'logoPositionY': 0.85,  # Bottom
            'topBlurPercent': 25,
            'bottomBlurPercent': 25,
            'blurIntensity': 20,
        }

        print(f"Input: {input_path}")
        print(f"Handler output_path: {output_path}")

        # Call processor (like handler does)
        result = process_video_reframe(input_path, output_path, config)

        # Handler fix: use actual outputPath from result if available
        if result and result.get('outputPath'):
            output_path = result['outputPath']
            print(f"Fixed output_path: {output_path}")

        # Check if output exists (what handler does)
        if os.path.exists(output_path):
            output_size = os.path.getsize(output_path)
            print(f"\n[SUCCESS] Handler simulation passed!")
            print(f"  Output: {output_path}")
            print(f"  Size: {output_size:,} bytes")

            # Save for inspection
            ext = os.path.splitext(output_path)[1]
            final_path = os.path.join(OUTPUT_DIR, f"handler_sim_test_{int(time.time())}{ext}")
            shutil.copy(output_path, final_path)
            print(f"  Saved to: {final_path}")

            return True
        else:
            print(f"\n[FAILED] Handler simulation failed!")
            print(f"  Output not found: {output_path}")
            return False

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def test_logo_positions():
    """Test various logo positions"""
    print("\n" + "=" * 60)
    print("  TEST 3: Logo Position Tests")
    print("=" * 60)

    positions = [
        (0.1, 0.1, "top-left"),
        (0.9, 0.1, "top-right"),
        (0.5, 0.5, "center"),
        (0.1, 0.9, "bottom-left"),
        (0.9, 0.9, "bottom-right"),
    ]

    temp_dir = tempfile.mkdtemp(prefix='logo_pos_test_')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        input_path = download_test_image(temp_dir)

        for x, y, name in positions:
            print(f"\n  Testing position: {name} ({x}, {y})")

            output_path = os.path.join(temp_dir, f"output_{name}.mp4")

            config = {
                'aspectRatio': '9:16',
                'logoName': 'farmium_full',
                'logoSize': 15,
                'logoPositionX': x,
                'logoPositionY': y,
                'topBlurPercent': 20,
                'bottomBlurPercent': 20,
                'blurIntensity': 20,
            }

            result = process_video_reframe(input_path, output_path, config)
            actual_output = result.get('outputPath', output_path)

            if os.path.exists(actual_output):
                size = os.path.getsize(actual_output)
                final_path = os.path.join(OUTPUT_DIR, f"logo_pos_{name}_{int(time.time())}.jpg")
                shutil.copy(actual_output, final_path)
                print(f"    [OK] Saved to: {final_path} ({size:,} bytes)")
            else:
                print(f"    [FAILED] No output")
                return False

        print(f"\n[SUCCESS] All logo positions tested!")
        return True

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def test_blur_zones():
    """Test asymmetric blur zones"""
    print("\n" + "=" * 60)
    print("  TEST 4: Asymmetric Blur Zones")
    print("=" * 60)

    blur_configs = [
        (0, 0, "no-blur"),
        (50, 0, "top-only"),
        (0, 50, "bottom-only"),
        (30, 10, "asymmetric-top"),
        (10, 30, "asymmetric-bottom"),
    ]

    temp_dir = tempfile.mkdtemp(prefix='blur_test_')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        input_path = download_test_image(temp_dir)

        for top, bottom, name in blur_configs:
            print(f"\n  Testing blur: {name} (top={top}%, bottom={bottom}%)")

            output_path = os.path.join(temp_dir, f"output_{name}.mp4")

            config = {
                'aspectRatio': '9:16',
                'logoName': 'none',  # No logo for blur tests
                'topBlurPercent': top,
                'bottomBlurPercent': bottom,
                'blurIntensity': 25,
            }

            result = process_video_reframe(input_path, output_path, config)
            actual_output = result.get('outputPath', output_path)

            if os.path.exists(actual_output):
                size = os.path.getsize(actual_output)
                final_path = os.path.join(OUTPUT_DIR, f"blur_{name}_{int(time.time())}.jpg")
                shutil.copy(actual_output, final_path)
                print(f"    [OK] Saved to: {final_path} ({size:,} bytes)")
            else:
                print(f"    [FAILED] No output")
                return False

        print(f"\n[SUCCESS] All blur configurations tested!")
        return True

    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


def main():
    print("=" * 60)
    print("  VIDEO REFRAME LOCAL TESTS")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")

    results = []

    # Test 1: Basic image reframe
    results.append(("Image Reframe", test_image_reframe()))

    # Test 2: Handler simulation
    results.append(("Handler Simulation", test_handler_simulation()))

    # Test 3: Logo positions
    results.append(("Logo Positions", test_logo_positions()))

    # Test 4: Blur zones
    results.append(("Blur Zones", test_blur_zones()))

    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\nAll tests passed! Ready to deploy.")
    else:
        print("\nSome tests failed. Please fix before deploying.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
