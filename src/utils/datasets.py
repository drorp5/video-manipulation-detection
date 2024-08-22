from typing import Dict, Tuple

from utils.detection_utils import Rectangle


def get_largest_bounding_box(annotation: dict, target_object: str) -> Rectangle:
    """
    Find the largest bounding box for a target object in the given annotation.

    Args:
        annotation (Dict): A dictionary containing object annotations.
        target_object (str): The label of the target object to find.

    Returns:
        Rectangle: The largest bounding box for the target object.
    """
    min_area = 0
    largest_gt_bounding_box_annotation = None
    for object in annotation["objects"]:
        if object["label"] == target_object:
            gt_bounding_box_annotation = object["bbox"]
            area = (
                gt_bounding_box_annotation["xmax"] - gt_bounding_box_annotation["xmin"]
            ) * (
                gt_bounding_box_annotation["ymax"] - gt_bounding_box_annotation["ymin"]
            )
            if area > min_area:
                min_area = area
                largest_gt_bounding_box_annotation = gt_bounding_box_annotation
    return Rectangle(
        (
            largest_gt_bounding_box_annotation["xmin"],
            largest_gt_bounding_box_annotation["ymin"],
        ),
        (
            largest_gt_bounding_box_annotation["xmax"],
            largest_gt_bounding_box_annotation["ymax"],
        ),
    )


def resize_bounding_box(
    bounding_box: Rectangle, old_shape: Tuple[int, int], new_shape: Tuple[int, int]
) -> None:
    """
    Resize a bounding box based on the change in image dimensions.

    Args:
        bounding_box (Rectangle): The bounding box to resize.
        old_shape (Tuple[int, int]): The original image shape (width, height).
        new_shape (Tuple[int, int]): The new image shape (width, height).

    Returns:
        None: The bounding box is modified in-place.
    """
    width_resizing_factor = new_shape[0] / old_shape[0]
    height_resizing_factor = new_shape[1] / old_shape[1]
    bounding_box.resize(width_resizing_factor, height_resizing_factor)
    bounding_box.enforce_super_pixel_size(2)  # Bayer
