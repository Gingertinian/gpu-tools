"""
GPU Operations Module - CuPy-based image processing

All operations run entirely on GPU using CuPy.
Minimizes CPU-GPU transfers by keeping data in GPU memory.

Key features:
1. CuPy arrays for all operations (GPU memory)
2. Fused operations to reduce memory bandwidth
3. Automatic CPU fallback when GPU not available
4. Optimized for video frame processing (reuse allocations)
"""

import numpy as np
from typing import Optional, Tuple, Union

# Try to import cupy and verify it actually works
HAS_CUPY = False
cp = np  # Default to numpy
ndimage = None

try:
    import cupy as _cp
    import cupyx.scipy.ndimage as _ndimage

    # Test if CuPy actually works (CUDA toolkit must be installed)
    _test = _cp.array([1, 2, 3])
    _test = _test * 2  # This will fail if NVRTC is missing
    del _test

    # If we get here, CuPy works
    cp = _cp
    ndimage = _ndimage
    HAS_CUPY = True
    print("[GPU] CuPy available and working - GPU acceleration enabled")
except ImportError:
    print("[GPU] CuPy not installed - using CPU fallback")
except Exception as e:
    # CuPy installed but CUDA toolkit incomplete (common on Windows)
    print(f"[GPU] CuPy installed but CUDA not working: {type(e).__name__}")
    print("[GPU] Using CPU fallback (install CUDA Toolkit for GPU support)")

# Array type for type hints
ArrayType = Union['cp.ndarray', np.ndarray]


class GPUProcessor:
    """
    GPU-accelerated image processing using CuPy.

    All operations stay on GPU until explicitly transferred back.
    Call to_cpu() only when needed for output.
    """

    def __init__(self, device_id: int = 0):
        """Initialize GPU processor on specified device."""
        self.device_id = device_id
        self.has_gpu = HAS_CUPY

        if self.has_gpu:
            try:
                cp.cuda.Device(device_id).use()
                # Pre-allocate memory pools for efficiency
                self.mempool = cp.get_default_memory_pool()
                self.pinned_mempool = cp.get_default_pinned_memory_pool()
                print(f"[GPU] Initialized on device {device_id}")
            except Exception as e:
                print(f"[GPU] Failed to initialize device {device_id}: {e}")
                self.has_gpu = False

    def to_gpu(self, arr: np.ndarray) -> ArrayType:
        """Transfer numpy array to GPU."""
        if self.has_gpu:
            return cp.asarray(arr)
        return arr

    def to_cpu(self, arr: ArrayType) -> np.ndarray:
        """Transfer GPU array back to CPU."""
        if self.has_gpu and hasattr(arr, 'get'):
            return cp.asnumpy(arr)
        return arr if isinstance(arr, np.ndarray) else np.asarray(arr)

    def zeros(self, shape: Tuple, dtype=np.uint8) -> ArrayType:
        """Create zero array on GPU."""
        if self.has_gpu:
            return cp.zeros(shape, dtype=dtype)
        return np.zeros(shape, dtype=dtype)

    def empty(self, shape: Tuple, dtype=np.uint8) -> ArrayType:
        """Create uninitialized array on GPU."""
        if self.has_gpu:
            return cp.empty(shape, dtype=dtype)
        return np.empty(shape, dtype=dtype)

    # =========================================================================
    # RESIZE OPERATIONS
    # =========================================================================

    def resize(self, image: ArrayType, new_size: Tuple[int, int],
               interpolation: str = 'linear') -> ArrayType:
        """
        Resize image on GPU.

        Args:
            image: Input image (H, W, C) on GPU
            new_size: Target size as (width, height)
            interpolation: 'nearest', 'linear', or 'cubic'

        Returns:
            Resized image on GPU
        """
        target_w, target_h = new_size
        h, w = image.shape[:2]

        if h == target_h and w == target_w:
            return image

        if self.has_gpu and ndimage is not None:
            # Calculate zoom factors
            zoom_h = target_h / h
            zoom_w = target_w / w

            # Order: 0=nearest, 1=linear, 3=cubic
            order_map = {'nearest': 0, 'linear': 1, 'cubic': 3}
            order = order_map.get(interpolation, 1)

            # Handle channels
            if len(image.shape) == 3:
                zoom_factors = (zoom_h, zoom_w, 1)  # Don't zoom channels
            else:
                zoom_factors = (zoom_h, zoom_w)

            return ndimage.zoom(image.astype(cp.float32), zoom_factors, order=order).astype(cp.uint8)
        else:
            # CPU fallback using cv2
            import cv2
            cpu_img = self.to_cpu(image)
            interp_map = {'nearest': cv2.INTER_NEAREST, 'linear': cv2.INTER_LINEAR, 'cubic': cv2.INTER_CUBIC}
            resized = cv2.resize(cpu_img, (target_w, target_h), interpolation=interp_map.get(interpolation, cv2.INTER_LINEAR))
            return self.to_gpu(resized)

    def resize_lanczos(self, image: ArrayType, new_size: Tuple[int, int]) -> ArrayType:
        """
        High-quality resize using Lanczos interpolation.
        Falls back to cubic on GPU (Lanczos not directly available in cupy).
        """
        return self.resize(image, new_size, interpolation='cubic')

    # =========================================================================
    # BLUR OPERATIONS
    # =========================================================================

    def gaussian_blur(self, image: ArrayType, sigma: float = 5.0) -> ArrayType:
        """
        Apply Gaussian blur on GPU.

        Args:
            image: Input image (H, W, C)
            sigma: Blur strength (higher = more blur)

        Returns:
            Blurred image
        """
        if sigma <= 0:
            return image

        if self.has_gpu and ndimage is not None:
            # Apply blur per channel for color images
            if len(image.shape) == 3:
                result = cp.empty_like(image)
                for c in range(image.shape[2]):
                    result[:, :, c] = ndimage.gaussian_filter(
                        image[:, :, c].astype(cp.float32), sigma=sigma
                    ).astype(cp.uint8)
                return result
            else:
                return ndimage.gaussian_filter(image.astype(cp.float32), sigma=sigma).astype(cp.uint8)
        else:
            # CPU fallback
            import cv2
            cpu_img = self.to_cpu(image)
            k = int(sigma * 4) | 1  # Kernel size ~4 sigma, ensure odd
            blurred = cv2.GaussianBlur(cpu_img, (k, k), sigma)
            return self.to_gpu(blurred)

    # =========================================================================
    # FLIP OPERATIONS
    # =========================================================================

    def flip_horizontal(self, image: ArrayType) -> ArrayType:
        """Flip image horizontally (left-right)."""
        if self.has_gpu:
            return cp.ascontiguousarray(image[:, ::-1])
        return np.ascontiguousarray(image[:, ::-1])

    def flip_vertical(self, image: ArrayType) -> ArrayType:
        """Flip image vertically (up-down)."""
        if self.has_gpu:
            return cp.ascontiguousarray(image[::-1, :])
        return np.ascontiguousarray(image[::-1, :])

    # =========================================================================
    # COLOR OPERATIONS
    # =========================================================================

    def bgr_to_hsv(self, image: ArrayType) -> ArrayType:
        """
        Convert BGR to HSV on GPU.

        HSV ranges: H=[0,180], S=[0,255], V=[0,255]
        """
        if self.has_gpu:
            img = image.astype(cp.float32) / 255.0
            b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]

            cmax = cp.maximum(cp.maximum(r, g), b)
            cmin = cp.minimum(cp.minimum(r, g), b)
            delta = cmax - cmin

            # Hue calculation
            h = cp.zeros_like(cmax)

            # When max == r
            mask_r = (cmax == r) & (delta > 0)
            h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)

            # When max == g
            mask_g = (cmax == g) & (delta > 0)
            h[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)

            # When max == b
            mask_b = (cmax == b) & (delta > 0)
            h[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)

            # Saturation
            s = cp.where(cmax > 0, delta / cmax, 0)

            # Value
            v = cmax

            # Scale to OpenCV ranges: H=[0,180], S=[0,255], V=[0,255]
            h = (h / 2.0).astype(cp.uint8)  # [0,360] -> [0,180]
            s = (s * 255).astype(cp.uint8)
            v = (v * 255).astype(cp.uint8)

            return cp.stack([h, s, v], axis=-1)
        else:
            import cv2
            cpu_img = self.to_cpu(image)
            return self.to_gpu(cv2.cvtColor(cpu_img, cv2.COLOR_BGR2HSV))

    def hsv_to_bgr(self, hsv: ArrayType) -> ArrayType:
        """Convert HSV back to BGR on GPU."""
        if self.has_gpu:
            h = hsv[:, :, 0].astype(cp.float32) * 2.0  # [0,180] -> [0,360]
            s = hsv[:, :, 1].astype(cp.float32) / 255.0
            v = hsv[:, :, 2].astype(cp.float32) / 255.0

            c = v * s
            x = c * (1 - cp.abs((h / 60.0) % 2 - 1))
            m = v - c

            h_sector = (h / 60.0).astype(cp.int32) % 6

            r = cp.zeros_like(h)
            g = cp.zeros_like(h)
            b = cp.zeros_like(h)

            # Sector 0: R=C, G=X, B=0
            mask = (h_sector == 0)
            r[mask], g[mask], b[mask] = c[mask], x[mask], 0

            # Sector 1: R=X, G=C, B=0
            mask = (h_sector == 1)
            r[mask], g[mask], b[mask] = x[mask], c[mask], 0

            # Sector 2: R=0, G=C, B=X
            mask = (h_sector == 2)
            r[mask], g[mask], b[mask] = 0, c[mask], x[mask]

            # Sector 3: R=0, G=X, B=C
            mask = (h_sector == 3)
            r[mask], g[mask], b[mask] = 0, x[mask], c[mask]

            # Sector 4: R=X, G=0, B=C
            mask = (h_sector == 4)
            r[mask], g[mask], b[mask] = x[mask], 0, c[mask]

            # Sector 5: R=C, G=0, B=X
            mask = (h_sector == 5)
            r[mask], g[mask], b[mask] = c[mask], 0, x[mask]

            r = ((r + m) * 255).astype(cp.uint8)
            g = ((g + m) * 255).astype(cp.uint8)
            b = ((b + m) * 255).astype(cp.uint8)

            return cp.stack([b, g, r], axis=-1)
        else:
            import cv2
            cpu_hsv = self.to_cpu(hsv)
            return self.to_gpu(cv2.cvtColor(cpu_hsv, cv2.COLOR_HSV2BGR))

    def adjust_hsv(self, image: ArrayType, hue_shift: int = 0,
                   saturation: float = 1.0, brightness: float = 1.0) -> ArrayType:
        """
        Adjust HSV values in a single fused operation (more efficient).

        Args:
            image: BGR image
            hue_shift: Hue offset (-180 to 180)
            saturation: Saturation multiplier (0.0 to 2.0)
            brightness: Brightness/Value multiplier (0.0 to 2.0)
        """
        if hue_shift == 0 and saturation == 1.0 and brightness == 1.0:
            return image

        xp = cp if self.has_gpu else np

        hsv = self.bgr_to_hsv(image).astype(xp.float32)

        if hue_shift != 0:
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180

        if saturation != 1.0:
            hsv[:, :, 1] = xp.clip(hsv[:, :, 1] * saturation, 0, 255)

        if brightness != 1.0:
            hsv[:, :, 2] = xp.clip(hsv[:, :, 2] * brightness, 0, 255)

        return self.hsv_to_bgr(hsv.astype(xp.uint8))

    def adjust_brightness(self, image: ArrayType, factor: float) -> ArrayType:
        """Simple brightness adjustment (multiply all channels)."""
        if factor == 1.0:
            return image

        xp = cp if self.has_gpu else np
        return xp.clip(image.astype(xp.float32) * factor, 0, 255).astype(xp.uint8)

    def adjust_contrast(self, image: ArrayType, factor: float) -> ArrayType:
        """Adjust contrast around midpoint (128)."""
        if factor == 1.0:
            return image

        xp = cp if self.has_gpu else np
        result = 128 + (image.astype(xp.float32) - 128) * factor
        return xp.clip(result, 0, 255).astype(xp.uint8)

    def darken(self, image: ArrayType, factor: float) -> ArrayType:
        """Darken image by factor (0.0 to 1.0)."""
        return self.adjust_brightness(image, factor)

    # =========================================================================
    # TRANSFORM OPERATIONS
    # =========================================================================

    def rotate(self, image: ArrayType, angle: float,
               border_mode: str = 'reflect') -> ArrayType:
        """
        Rotate image by angle degrees (counter-clockwise).

        Args:
            image: Input image
            angle: Rotation angle in degrees
            border_mode: 'reflect', 'constant', 'nearest', 'wrap'
        """
        if angle == 0:
            return image

        if self.has_gpu and ndimage is not None:
            mode_map = {
                'reflect': 'reflect',
                'constant': 'constant',
                'nearest': 'nearest',
                'wrap': 'wrap'
            }
            mode = mode_map.get(border_mode, 'reflect')

            # Rotate each channel
            if len(image.shape) == 3:
                result = cp.empty_like(image)
                for c in range(image.shape[2]):
                    result[:, :, c] = ndimage.rotate(
                        image[:, :, c].astype(cp.float32),
                        angle,
                        reshape=False,
                        mode=mode
                    ).astype(cp.uint8)
                return result
            else:
                return ndimage.rotate(
                    image.astype(cp.float32), angle, reshape=False, mode=mode
                ).astype(cp.uint8)
        else:
            # CPU fallback
            import cv2
            cpu_img = self.to_cpu(image)
            h, w = cpu_img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            border_map = {
                'reflect': cv2.BORDER_REFLECT,
                'constant': cv2.BORDER_CONSTANT,
                'nearest': cv2.BORDER_REPLICATE,
                'wrap': cv2.BORDER_WRAP
            }
            border = border_map.get(border_mode, cv2.BORDER_REFLECT)
            rotated = cv2.warpAffine(cpu_img, matrix, (w, h), borderMode=border)
            return self.to_gpu(rotated)

    # =========================================================================
    # COMPOSITE OPERATIONS
    # =========================================================================

    def alpha_blend(self, foreground: ArrayType, background: ArrayType,
                    alpha: ArrayType) -> ArrayType:
        """
        Alpha blend foreground over background on GPU.

        Args:
            foreground: Foreground image (H, W, C)
            background: Background image (H, W, C)
            alpha: Alpha mask (H, W) or (H, W, 1), values 0-1
        """
        xp = cp if self.has_gpu else np

        # Ensure alpha is 3D for broadcasting
        if alpha.ndim == 2:
            alpha = alpha[:, :, xp.newaxis]

        fg = foreground.astype(xp.float32)
        bg = background.astype(xp.float32)
        alpha = alpha.astype(xp.float32)

        result = fg * alpha + bg * (1 - alpha)
        return result.astype(xp.uint8)

    def composite_frame(self, output: ArrayType,
                        blur_top: Optional[ArrayType],
                        content: ArrayType,
                        blur_bottom: Optional[ArrayType],
                        layout: dict) -> ArrayType:
        """
        Composite full frame from components on GPU.
        Single operation to combine blur zones and content.

        Args:
            output: Pre-allocated output buffer (H, W, C)
            blur_top: Top blur zone or None
            content: Main content
            blur_bottom: Bottom blur zone or None
            layout: Dict with positions/sizes
        """
        blur_top_h = layout['blur_top']
        content_y = layout['content_y']
        content_x = layout['content_x']
        scaled_h = layout['scaled_h']
        scaled_w = layout['scaled_w']

        # Place blur top
        if blur_top is not None and blur_top_h > 0:
            output[0:blur_top_h, :] = blur_top

        # Place content
        output[content_y:content_y + scaled_h, content_x:content_x + scaled_w] = content

        # Place blur bottom
        if blur_bottom is not None and layout['blur_bottom'] > 0:
            output[content_y + scaled_h:, :] = blur_bottom

        return output

    # =========================================================================
    # UTILITY
    # =========================================================================

    def crop(self, image: ArrayType, x: int, y: int, w: int, h: int) -> ArrayType:
        """Crop region from image."""
        return image[y:y+h, x:x+w].copy()

    def copy_region(self, src: ArrayType, dst: ArrayType,
                    src_rect: Tuple, dst_pos: Tuple) -> ArrayType:
        """Copy region from src to dst."""
        sx, sy, sw, sh = src_rect
        dx, dy = dst_pos
        dst[dy:dy+sh, dx:dx+sw] = src[sy:sy+sh, sx:sx+sw]
        return dst

    def ensure_contiguous(self, arr: ArrayType) -> ArrayType:
        """Ensure array is contiguous in memory."""
        xp = cp if self.has_gpu else np
        return xp.ascontiguousarray(arr)

    def free_memory(self):
        """Free GPU memory pools."""
        if self.has_gpu:
            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global processor instance (lazy initialization)
_gpu_processor: Optional[GPUProcessor] = None


def get_gpu_processor(device_id: int = 0) -> GPUProcessor:
    """Get or create GPU processor instance."""
    global _gpu_processor
    if _gpu_processor is None or _gpu_processor.device_id != device_id:
        _gpu_processor = GPUProcessor(device_id)
    return _gpu_processor


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return HAS_CUPY


# =============================================================================
# BLUR EFFECTS (GPU VERSION)
# =============================================================================

class GPUBlurEffects:
    """
    GPU-accelerated blur effects for video processing.
    All operations stay on GPU.
    """

    def __init__(self, processor: GPUProcessor):
        self.gpu = processor

    def gaussian_blur(self, image: ArrayType, strength: int = 25) -> ArrayType:
        """Apply Gaussian blur with specified strength (kernel size)."""
        if strength <= 0:
            return image
        # Convert kernel size to sigma (approximate)
        sigma = strength / 4.0
        return self.gpu.gaussian_blur(image, sigma)

    def flip_vertical(self, image: ArrayType) -> ArrayType:
        return self.gpu.flip_vertical(image)

    def flip_horizontal(self, image: ArrayType) -> ArrayType:
        return self.gpu.flip_horizontal(image)

    def color_shift(self, image: ArrayType, hue_shift: int) -> ArrayType:
        return self.gpu.adjust_hsv(image, hue_shift=hue_shift)

    def adjust_saturation_brightness(self, image: ArrayType,
                                      saturation: float = 1.0,
                                      brightness: float = 1.0) -> ArrayType:
        return self.gpu.adjust_hsv(image, saturation=saturation, brightness=brightness)

    def tilt(self, image: ArrayType, angle: float,
             rotation_matrix: np.ndarray = None) -> ArrayType:
        """Apply rotation/tilt to image."""
        return self.gpu.rotate(image, angle, border_mode='reflect')

    def adjust_contrast(self, image: ArrayType, contrast: float) -> ArrayType:
        return self.gpu.adjust_contrast(image, contrast)

    def darken(self, image: ArrayType, factor: float = 0.7) -> ArrayType:
        return self.gpu.darken(image, factor)


# =============================================================================
# BATCH FRAME PROCESSING
# =============================================================================

class GPUFrameBatch:
    """
    Batch frame processing on GPU.
    Processes multiple frames simultaneously for better throughput.
    """

    def __init__(self, processor: GPUProcessor, batch_size: int = 8):
        self.gpu = processor
        self.batch_size = batch_size
        self.frame_buffer = []

    def add_frame(self, frame: np.ndarray):
        """Add frame to batch."""
        self.frame_buffer.append(frame)

    def process_batch(self, process_fn) -> list:
        """
        Process all frames in batch.

        Args:
            process_fn: Function that takes GPU frame and returns processed GPU frame

        Returns:
            List of processed frames (CPU numpy arrays)
        """
        if not self.frame_buffer:
            return []

        results = []

        # Transfer batch to GPU
        for frame in self.frame_buffer:
            gpu_frame = self.gpu.to_gpu(frame)
            processed = process_fn(gpu_frame)
            results.append(self.gpu.to_cpu(processed))

        self.frame_buffer.clear()
        return results

    def is_full(self) -> bool:
        return len(self.frame_buffer) >= self.batch_size

    def has_frames(self) -> bool:
        return len(self.frame_buffer) > 0
