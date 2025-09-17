"""
Image processor compatibility wrapper for Qwen2.5-VL spatial merge requirements
Ensures all processed images have token counts compatible with spatial_merge_unit=4
"""

import copy
import math
import logging
from typing import Tuple, Dict, Any
from PIL import Image
import torch

logger = logging.getLogger(__name__)


def smart_resize_compatible(height: int, width: int, factor: int = 28,
                          min_pixels: int = 56*56, max_pixels: int = 14*14*4*1280,
                          merge_size: int = 2, spatial_merge_unit: int = 4) -> Tuple[int, int]:
    """
    Enhanced smart_resize that ensures spatial merge compatibility

    Args:
        height, width: Original image dimensions
        factor: Patch size factor (typically 28 for Qwen2.5-VL)
        min_pixels, max_pixels: Size constraints
        merge_size: Vision encoder merge size (2 for Qwen2.5-VL)
        spatial_merge_unit: Spatial merge unit (4 for Qwen2.5-VL)

    Returns:
        (height, width) that ensures (h*w // merge_size**2) % spatial_merge_unit == 0
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")

    # Initial resize using standard logic
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # Apply size constraints
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = int(math.floor(height / beta / factor) * factor)
        w_bar = int(math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = int(math.ceil(height * beta / factor) * factor)
        w_bar = int(math.ceil(width * beta / factor) * factor)

    # Check spatial merge compatibility
    total_patches = (h_bar // factor) * (w_bar // factor)
    tokens_after_merge = total_patches // (merge_size ** 2)

    if tokens_after_merge % spatial_merge_unit == 0:
        # Already compatible
        logger.debug(f"Image {width}x{height} → {w_bar}x{h_bar}: {tokens_after_merge} tokens (compatible)")
        return h_bar, w_bar

    # Find compatible dimensions with minimal adjustment
    best_h, best_w = h_bar, w_bar
    min_adjustment = float('inf')

    # Try small adjustments around the current dimensions
    for dh in range(-2*factor, 3*factor, factor):  # Try ±2 factors
        for dw in range(-2*factor, 3*factor, factor):
            test_h = h_bar + dh
            test_w = w_bar + dw

            # Ensure dimensions are valid
            if test_h < factor or test_w < factor:
                continue
            if test_h * test_w < min_pixels or test_h * test_w > max_pixels:
                continue
            if max(test_h, test_w) / min(test_h, test_w) > 200:
                continue

            # Check compatibility
            test_patches = (test_h // factor) * (test_w // factor)
            test_tokens = test_patches // (merge_size ** 2)

            if test_tokens % spatial_merge_unit == 0:
                # Calculate adjustment cost (prefer minimal change)
                adjustment = abs(test_h - h_bar) + abs(test_w - w_bar)
                if adjustment < min_adjustment:
                    min_adjustment = adjustment
                    best_h, best_w = test_h, test_w

    # Log the adjustment made
    if best_h != h_bar or best_w != w_bar:
        original_tokens = total_patches // (merge_size ** 2)
        new_patches = (best_h // factor) * (best_w // factor)
        new_tokens = new_patches // (merge_size ** 2)
        logger.info(f"Adjusted {width}x{height} → {best_w}x{best_h}: {original_tokens}→{new_tokens} tokens (made compatible)")

    return best_h, best_w


class CompatibleImageProcessor:
    """
    Wrapper around Qwen2.5-VL image processor that ensures spatial merge compatibility
    """

    def __init__(self, original_processor, enable_compatibility: bool = True):
        """
        Args:
            original_processor: The original Qwen2VLImageProcessor
            enable_compatibility: Whether to enforce spatial merge compatibility
        """
        self.original_processor = original_processor
        self.enable_compatibility = enable_compatibility
        self.merge_size = getattr(original_processor, 'merge_size', 2)
        self.spatial_merge_unit = 4  # Hardcoded in Qwen2.5-VL architecture

        # Track statistics
        self.stats = {
            'total_processed': 0,
            'adjusted_images': 0,
            'adjustment_details': []
        }

    def __call__(self, images, **kwargs):
        """Process images with compatibility ensuring"""

        if not self.enable_compatibility:
            # Use original processor directly
            return self.original_processor(images=images, **kwargs)

        # Handle single image vs batch
        if isinstance(images, Image.Image):
            images = [images]
            single_image = True
        else:
            single_image = False

        processed_images = []
        all_pixel_values = []
        all_grid_thw = []

        for image in images:
            self.stats['total_processed'] += 1

            # Get original size
            original_width, original_height = image.size

            # Calculate compatible dimensions
            try:
                compatible_height, compatible_width = smart_resize_compatible(
                    original_height, original_width,
                    factor=getattr(self.original_processor, 'patch_size', 28),
                    min_pixels=getattr(self.original_processor, 'min_pixels', 56*56),
                    max_pixels=getattr(self.original_processor, 'max_pixels', 14*14*4*1280),
                    merge_size=self.merge_size,
                    spatial_merge_unit=self.spatial_merge_unit
                )

                # Resize image to compatible dimensions if needed
                if compatible_height != original_height or compatible_width != original_width:
                    image = image.resize((compatible_width, compatible_height), Image.LANCZOS)
                    self.stats['adjusted_images'] += 1
                    self.stats['adjustment_details'].append({
                        'original_size': (original_width, original_height),
                        'adjusted_size': (compatible_width, compatible_height)
                    })

                # Process with original processor
                result = self.original_processor(images=image, **kwargs)

                # Validate result
                pixel_values = result["pixel_values"]
                grid_thw = result["image_grid_thw"][0]

                if isinstance(pixel_values, list):
                    pixel_values = pixel_values[0]

                # Verify compatibility
                t, h, w = grid_thw.tolist()
                total_patches = t * h * w
                tokens_after_merge = total_patches // (self.merge_size ** 2)

                if tokens_after_merge % self.spatial_merge_unit != 0:
                    logger.warning(f"Compatibility check failed: {tokens_after_merge} tokens not divisible by {self.spatial_merge_unit}")

                all_pixel_values.append(pixel_values)
                all_grid_thw.append(grid_thw.unsqueeze(0))

            except Exception as e:
                logger.error(f"Failed to process image with compatibility wrapper: {e}")
                # Fallback to original processor
                fallback_result = self.original_processor(images=image, **kwargs)
                all_pixel_values.append(fallback_result["pixel_values"][0] if isinstance(fallback_result["pixel_values"], list) else fallback_result["pixel_values"])
                all_grid_thw.append(fallback_result["image_grid_thw"][0].unsqueeze(0))

        # Combine results
        combined_result = {
            "pixel_values": all_pixel_values if not single_image else all_pixel_values[0],
            "image_grid_thw": torch.cat(all_grid_thw, dim=0) if not single_image else all_grid_thw[0]
        }

        return combined_result

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        adjustment_rate = (self.stats['adjusted_images'] / max(1, self.stats['total_processed'])) * 100
        return {
            **self.stats,
            'adjustment_rate_percent': adjustment_rate
        }

    def reset_stats(self):
        """Reset processing statistics"""
        self.stats = {
            'total_processed': 0,
            'adjusted_images': 0,
            'adjustment_details': []
        }

    def __getattr__(self, name):
        """Delegate attribute access to original processor"""
        return getattr(self.original_processor, name)


def wrap_image_processor(processor, enable_compatibility: bool = True):
    """
    Convenience function to wrap an image processor with compatibility ensuring

    Args:
        processor: Original Qwen2VLImageProcessor
        enable_compatibility: Whether to enable compatibility mode

    Returns:
        CompatibleImageProcessor instance
    """
    return CompatibleImageProcessor(processor, enable_compatibility)