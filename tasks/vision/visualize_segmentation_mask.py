import numpy as np

from core.task import task


@task(
    outputs=["colored_mask", "class_summary"],
    output_types={"colored_mask": "image", "class_summary": "str"},
    parameters={
        "mask_array": {
            "type": "ndarray",
            "required": True,
            "description": "Segmentation mask with class IDs",
        },
        "class_names": {
            "type": "list",
            "required": False,
            "description": "Optional list of class names corresponding to class IDs",
        },
    },
)
def visualize_segmentation_mask(mask_array, class_names=None):
    """
    Visualize a semantic segmentation mask by colorizing class IDs.

    Converts a segmentation mask with integer class IDs into a colorized RGB image
    using a dynamically generated colormap. Also returns a summary of detected classes.

    Args:
        mask_array: Segmentation mask as numpy array (H, W) with class IDs
        class_names: Optional list of class names for labeling (e.g., ["background", "person", ...])

    Returns:
        Tuple of:
        - colored_mask: RGB image (H, W, 3) with colors for each class
        - class_summary: String describing which classes were detected and their pixel counts
    """
    import imgviz
    
    if not isinstance(mask_array, np.ndarray):
        raise ValueError("'mask_array' must be a numpy array.")

    # Ensure mask is 2D
    if len(mask_array.shape) == 3:
        mask_array = mask_array.squeeze()

    # Get mask as integers
    mask_int = mask_array.astype(np.int32)

    # Find unique classes and their counts
    unique_classes, counts = np.unique(mask_int, return_counts=True)
    total_pixels = mask_int.size

    # Build class summary
    class_info = []
    for cls_id, count in zip(unique_classes, counts):
        percentage = (count / total_pixels) * 100
        if class_names and 0 <= cls_id < len(class_names):
            class_name = class_names[cls_id]
        else:
            class_name = f"class_{cls_id}"
        class_info.append(f"Class {cls_id} ({class_name}): {percentage:.1f}%")

    class_summary = "\n".join(class_info)

    # Generate colormap at runtime based on the maximum class ID found
    max_class_id = int(mask_int.max())
    n_labels = max_class_id + 1
    colormap = imgviz.label_colormap(n_labels)

    # Create colored mask
    h, w = mask_int.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Apply colormap
    for cls_id in unique_classes:
        if cls_id < len(colormap):
            color = colormap[cls_id]
        else:
            # Fallback to white for unexpected classes
            color = np.array([255, 255, 255], dtype=np.uint8)

        colored_mask[mask_int == cls_id] = color

    return colored_mask, class_summary

