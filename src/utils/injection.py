from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from utils.detection_utils import Rectangle
from gige.utils import bgr_img_to_packets_payload


def get_stripe(img: np.ndarray, rect: Rectangle) -> np.ndarray:
    """
    Extract a stripe from an image based on the given rectangle.

    Args:
        img (np.ndarray): The input image.
        rect (Rectangle): The rectangle defining the stripe area.

    Returns:
        np.ndarray: The extracted stripe.
    """
    return img[rect.ymin : rect.ymax, :, :]


def insert_stripe_to_img(
    img: np.ndarray, stripe: np.ndarray, target_row: int = 0
) -> np.ndarray:
    """
    Insert a stripe into an image at the specified row.

    Args:
        img (np.ndarray): The input image.
        stripe (np.ndarray): The stripe to insert.
        target_row (int): The row at which to insert the stripe.

    Returns:
        np.ndarray: The image with the inserted stripe.
    """
    assert img.shape[1] == stripe.shape[1], "NUM COLUMNS MISMATCH"
    assert img.shape[2] == stripe.shape[2], "NUM COLORS MISMATCH"
    dst = img.copy()
    num_rows_in_stripe = stripe.shape[0]
    num_rows_in_img = img.shape[0]
    target_row = min(target_row, num_rows_in_img - num_rows_in_stripe)
    dst[target_row : target_row + num_rows_in_stripe, :, :] = stripe
    return dst


def get_masked_stripe(img: np.ndarray, rect: Rectangle) -> np.ndarray:
    """
    Get a masked stripe from an image based on the given rectangle.

    Args:
        img (np.ndarray): The input image.
        rect (Rectangle): The rectangle defining the stripe area.

    Returns:
        np.ndarray: The masked stripe.
    """
    stripe = get_stripe(img, rect)
    background_img = np.zeros_like(img)
    masked_stripe = insert_stripe_to_img(background_img, stripe, rect.ymin)
    return masked_stripe


def get_stripe_gvsp_payload_bytes(
    img_bgr: np.ndarray,
    bounding_box: Rectangle,
    max_payload_bytes: int,
    target_row: int = 0,
) -> Tuple[List[int], List[bytes]]:
    """
    Get GVSP payload bytes for a stripe in the image.

    Args:
        img_bgr (np.ndarray): The input BGR image.
        bounding_box (Rectangle): The bounding box defining the stripe area.
        max_payload_bytes (int): Maximum payload size in bytes.
        target_row (int): The target row for insertion.

    Returns:
        Tuple[List[int], List[bytes]]: GVSP packet IDs and payload bytes.
    """
    stripe = get_stripe(img_bgr, bounding_box)
    background_img = np.zeros_like(img_bgr)
    injection_img = insert_stripe_to_img(background_img, stripe, target_row)
    all_gvsp_payload = bgr_img_to_packets_payload(injection_img, max_payload_bytes)

    gvsp_payload_stripe = []
    gvsp_packets_ids = []

    for ind, pkt_payload in enumerate(all_gvsp_payload):
        if (pkt_payload != 0).any():
            gvsp_payload_stripe.append(pkt_payload)
            gvsp_packets_ids.append(ind + 1)

    return gvsp_packets_ids, gvsp_payload_stripe


@dataclass
class RectangularBoundaries:
    """
    Data class to store the boundaries of a rectangular region in an image.
    """

    top_row: int
    bottom_row: int
    left_col: int
    right_col: int
