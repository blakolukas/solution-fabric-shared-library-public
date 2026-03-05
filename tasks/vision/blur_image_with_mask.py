import cv2
import numpy as np

from core.task import task

# Blur optimization constants
BLUR_OPTIMIZATION_THRESHOLD = 30  # Use downscale optimization for blur_radius > this value
DOWNSCALE_FACTOR = 2  # Factor to downscale image before blurring


@task(
    outputs=["blurred_image_array"],
    output_types={"blurred_image_array": "image"},
    display_name="Blur Image with Mask",
    description="Apply Gaussian blur to image regions based on a mask",
    category="vision",
    parameters={
        "image_array": {
            "type": "image",
            "required": True,
            "description": "Input image as numpy array (H, W, C)",
        },
        "mask_array": {
            "type": "image",
            "required": True,
            "description": "Binary mask - 0 for blur regions, non-zero for keep regions",
        },
        "blur_radius": {
            "type": "int",
            "required": False,
            "default": 15,
            "min": 1,
            "max": 101,
            "description": "Radius for Gaussian blur kernel",
        },
    },
)
def blur_image_with_mask(image_array, mask_array, blur_radius: int = 15):
    """
    Apply Gaussian blur to an image based on a mask.

    Blurs regions where the mask is zero, and keeps original image where mask is non-zero.

    Args:
        image_array: Input image as numpy array (H, W, C)
        mask_array: Binary mask as numpy array (H, W) - 0 for blur regions, non-zero for keep regions
        blur_radius: Radius for Gaussian blur kernel (default: 15)

    Returns:
        Blurred image as numpy array
    """
    if not isinstance(image_array, np.ndarray):
        raise ValueError("'image_array' must be a numpy array.")

    if not isinstance(mask_array, np.ndarray):
        raise ValueError("'mask_array' must be a numpy array.")

    # Ensure mask is 2D (remove channel dimension if present)
    if len(mask_array.shape) == 3:
        mask_array = mask_array.squeeze()

    # Ensure mask is uint8 and same spatial size as image
    if mask_array.shape != image_array.shape[:2]:
        mask_array = cv2.resize(mask_array, (image_array.shape[1], image_array.shape[0]))

    # Convert mask to uint8 if it isn't already
    if mask_array.dtype != np.uint8:
        # Handle both float (0.0-1.0) and integer (0-255) ranges
        if mask_array.max() <= 1.0:
            mask_array = (mask_array * 255).astype(np.uint8)
        else:
            mask_array = mask_array.astype(np.uint8)

    # Ensure an odd, positive kernel size
    ksize = max(1, int(blur_radius))
    if ksize % 2 == 0:
        ksize += 1

    # Create binary mask (threshold at 127 to handle any values)
    _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

    # Optimize blur for large radii using downscale-blur-upscale technique
    # This is significantly faster for blur_radius > 30 (3.5x faster for radius=90)
    if blur_radius > BLUR_OPTIMIZATION_THRESHOLD:
        # Downscale image for faster blur
        h, w = image_array.shape[:2]
        small_w, small_h = w // DOWNSCALE_FACTOR, h // DOWNSCALE_FACTOR
        # cv2.resize expects (width, height)
        small_image = cv2.resize(image_array, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply blur at lower resolution
        # Keep kernel size proportional to achieve similar visual blur effect
        small_ksize = max(3, ksize // DOWNSCALE_FACTOR)
        if small_ksize % 2 == 0:
            small_ksize += 1
        blurred_small = cv2.GaussianBlur(small_image, (small_ksize, small_ksize), 0)
        
        # Upscale back to exact original resolution - cv2.resize expects (width, height)
        blurred_image = cv2.resize(blurred_small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        # Use standard blur for small radii
        blurred_image = cv2.GaussianBlur(image_array, (ksize, ksize), 0)

    # Optimized blending using vectorized NumPy operations for better performance
    # Convert binary mask to float for smooth blending
    mask_float = binary_mask.astype(np.float32) / 255.0
    
    # Expand mask to match image channels
    if len(image_array.shape) == 3:
        mask_float = np.expand_dims(mask_float, axis=2)
    
    # Blend images using vectorized operations (faster than bitwise operations)
    final_image = (image_array * mask_float + blurred_image * (1.0 - mask_float)).astype(image_array.dtype)

    return final_image
