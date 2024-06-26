from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Iterator, Optional
import math
import cv2
import numpy as np

# Define type aliases
Point = Tuple[int, int]


class Rectangle:
    def __init__(self, upper_left_corner: Point, lower_right_corner: Point) -> None:
        self.xmin, self.ymin = upper_left_corner
        self.xmax, self.ymax = lower_right_corner

    @property
    def num_rows(self) -> int:
        return self.ymax - self.ymin

    @property
    def num_columns(self) -> int:
        return self.xmax - self.xmin

    def resize(self, width_factor: float = 1, height_factor: float = 1) -> None:
        self.xmin = int(self.xmin * width_factor)
        self.ymin = int(self.ymin * height_factor)
        self.xmax = int(self.xmax * width_factor)
        self.ymax = int(self.ymax * height_factor)

    def enforce_super_pixel_size(self, pixel_size: int) -> None:
        self.xmin = math.floor(self.xmin / pixel_size) * pixel_size
        self.xmax = math.ceil(self.xmax / pixel_size) * pixel_size
        self.ymin = math.floor(self.ymin / pixel_size) * pixel_size
        self.ymax = math.ceil(self.ymax / pixel_size) * pixel_size

    def __str__(self) -> str:
        return f"({self.xmin}, {self.ymin}), ({self.xmax}, {self.ymax})"

    def to_points(self) -> Tuple[Point, Point]:
        return (self.xmin, self.ymin), (self.xmax, self.ymax)


@dataclass
class DetectedObject:
    bounding_box: Iterable[int]  # x,y,w,h
    confidence: Optional[float] = None

    def __getitem__(self, index: int) -> int:
        return self.bounding_box[index]

    def get_upper_left_corner(self) -> Point:
        return self.bounding_box[0], self.bounding_box[1]

    def get_lower_right_corner(self) -> Point:
        return self.bounding_box[0] + self.width, self.bounding_box[1] + self.height

    @property
    def width(self) -> int:
        return self.bounding_box[2]

    @property
    def height(self) -> int:
        return self.bounding_box[3]

    def offset_by(self, dx: int, dy: int) -> DetectedObject:
        return DetectedObject(
            bounding_box=(
                self.bounding_box[0] + dx,
                self.bounding_box[1] + dy,
                self.width,
                self.height,
            ),
            confidence=self.confidence,
        )


def calculate_iou(rect1: Rectangle, rect2: Rectangle) -> float:
    """
    Calculate the Intersection over Union (IoU) of two rectangles.

    Parameters:
    rect1, rect2: Each is a Rectangle defined by two Points (top-left and bottom-right),
                  e.g., ((x1, y1), (x2, y2))

    Returns:
    float: IoU value
    """

    # Unpack the input rectangles
    (x1_min, y1_min), (x1_max, y1_max) = rect1.to_points()
    (x2_min, y2_min), (x2_max, y2_max) = rect2.to_points()

    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    # Calculate the area of the intersection rectangle
    if x_inter_min < x_inter_max and y_inter_min < y_inter_max:
        inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    else:
        inter_area = 0

    # Calculate the area of both input rectangles
    rect1_area = (x1_max - x1_min) * (y1_max - y1_min)
    rect2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate the union area
    union_area = rect1_area + rect2_area - inter_area

    # Compute the Intersection over Union (IoU)
    iou = inter_area / union_area if union_area != 0 else 0

    return iou


def sliding_window(
    image: np.ndarray,
    window_size: Tuple[int, int],
    step_size: Optional[Tuple[int, int]] = None,
) -> Iterator[Tuple[int, int, np.ndarray]]:
    """
    Slide a window across the image.

    Parameters:
    - image: The input image.
    - window_size: The size of the window (width, height).
    - step_size: The number of pixels to move the window by in (x, y) directions. If None, slidiing witout overlap

    Yields:
    - (x, y, window): The top-left corner (x, y) and the window image.
    """

    window_width, window_height = window_size
    if step_size is None:
        step_size = window_size
    step_x, step_y = step_size

    for y in range(0, image.shape[0], step_y):
        for x in range(0, image.shape[1], step_x):
            # Calculate the end coordinates of the window
            end_x = min(x + window_width, image.shape[1])
            end_y = min(y + window_height, image.shape[0])
            # Calculate the start coordinates to ensure the window is the correct size
            start_x = end_x - window_width
            start_y = end_y - window_height
            # Adjust start coordinates if they are negative
            if start_x < 0:
                start_x = 0
            if start_y < 0:
                start_y = 0
            yield (start_x, start_y, image[start_y:end_y, start_x:end_x])


def non_maximal_supression(
    detections: List[DetectedObject], confidence_th: float = 0, nms_th: float = 0.4
) -> List[DetectedObject]:

    result = []
    # values with confidence None are added without supression
    boxes = []
    confidences = []
    for detection in detections:
        if detection.confidence is None:
            result.append(detection)
        else:
            boxes.append(detection.bounding_box)
            confidences.append(detection.confidence)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_th, nms_th)
    for index in indices:
        result.append(
            DetectedObject(bounding_box=boxes[index], confidence=confidences[index])
        )
    return result
