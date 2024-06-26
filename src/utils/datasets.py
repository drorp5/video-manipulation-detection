from typing import Dict, Tuple

from utils.detection_utils import Rectangle


def get_largest_bounding_box(annotation: Dict, target_object: str) -> Rectangle:
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
    width_resizing_factor = new_shape[0] / old_shape[0]
    height_resizing_factor = new_shape[1] / old_shape[1]
    bounding_box.resize(width_resizing_factor, height_resizing_factor)
    bounding_box.enforce_super_pixel_size(2)  # Bayer
