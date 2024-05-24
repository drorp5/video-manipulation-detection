from typing import Tuple

# Define type aliases
Point = Tuple[int, int]
Rectangle = Tuple[Point, Point]


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
    (x1_min, y1_min), (x1_max, y1_max) = rect1
    (x2_min, y2_min), (x2_max, y2_max) = rect2

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
