"""
Local test for fast spoofer processor.
Test without RunPod to verify speed improvements.

Run: python test_fast_local.py
"""

import os
import sys
import time

# Add tools directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))

from spoofer.processor_fast import process_spoofer_fast

# Test configuration
INPUT_IMAGE = r'C:\Users\erudito\Pictures\Screenshots\Screenshot 2026-01-02 144608.png'
OUTPUT_DIR = r'C:\Users\erudito\Downloads'
VARIATIONS = 1000  # Test with 1000 variations

def main():
    print("=" * 60)
    print("  FAST SPOOFER LOCAL TEST")
    print("=" * 60)

    if not os.path.exists(INPUT_IMAGE):
        print(f"Error: Input image not found: {INPUT_IMAGE}")
        sys.exit(1)

    output_path = os.path.join(OUTPUT_DIR, f"fast_test_{VARIATIONS}variations_{int(time.time())}.zip")

    config = {
        'spatial': {
            'crop': 1.5,
            'microResize': 1.2,
            'rotation': 0.8,
        },
        'tonal': {
            'brightness': 0.04,
            'gamma': 0.06,
            'contrast': 0.04,
            'saturation': 0.06,
        },
        'visual': {
            'tint': 1.5,
            'noise': 3.0,
        },
        'compression': {
            'quality': 90,
        },
        'options': {
            'variations': VARIATIONS,  # Also accepts 'copies' for backward compatibility
            'force916': 1,
            'flip': 1,
        }
    }

    def progress(p, msg):
        bar_len = 30
        filled = int(bar_len * p)
        bar = '=' * filled + ' ' * (bar_len - filled)
        print(f"\r[{bar}] {int(p*100):3d}% - {msg}                    ", end='', flush=True)

    print(f"\nInput: {INPUT_IMAGE}")
    print(f"Output: {output_path}")
    print(f"Variations: {VARIATIONS}")
    print()

    start_time = time.time()

    result = process_spoofer_fast(INPUT_IMAGE, output_path, config, progress)

    elapsed = time.time() - start_time
    print("\n")
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Variations generated: {result.get('variations_generated', 0)}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Time per variation: {result.get('time_per_variation', elapsed/VARIATIONS):.3f}s")
    print(f"  CPU cores used: {result.get('cpu_cores_used', 'N/A')}")
    print(f"  Output: {output_path}")

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"  ZIP size: {size_mb:.2f} MB")

    print("=" * 60)

    # Compare with expected
    print("\nPerformance comparison:")
    print(f"  - Target: <1 minute for 200 variations")
    print(f"  - Actual: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if elapsed < 60:
        print("  - STATUS: EXCELLENT - Target met!")
    elif elapsed < 120:
        print("  - STATUS: GOOD - Acceptable performance")
    else:
        print("  - STATUS: NEEDS OPTIMIZATION")

if __name__ == "__main__":
    main()
