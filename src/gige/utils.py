from typing import List, Tuple
import numpy as np


from gige.gige_constants import BYTES_PER_PIXEL
from utils.image_processing import bgr_to_bayer_rg


IMG_SHAPE = Tuple[int, int]


def img_to_packets_payload(img: np.ndarray, max_payload_bytes: int) -> List[bytes]:
    dst_pixels = img.flatten()
    num_packets = int(np.ceil(len(dst_pixels) / BYTES_PER_PIXEL / max_payload_bytes))
    payload_pixels = [
        dst_pixels[pkt_ind * max_payload_bytes : (pkt_ind + 1) * max_payload_bytes]
        for pkt_ind in range(num_packets - 1)
    ]
    payload_pixels.append(dst_pixels[(num_packets - 1) * max_payload_bytes :])
    return payload_pixels


def bgr_img_to_packets_payload(img: np.ndarray, max_payload_bytes: int) -> List[bytes]:
    img_bayer = bgr_to_bayer_rg(img)
    return img_to_packets_payload(img_bayer, max_payload_bytes)


def packet_id_to_payload_indices(
    packet_id: int,
    payload_size_bytes: int,
    max_payload_size_bytes: int,
    shape: IMG_SHAPE,
) -> Tuple[np.ndarray, np.ndarray]:
    pixels_per_packet = int(max_payload_size_bytes / BYTES_PER_PIXEL)
    payload_size_pixels = int(payload_size_bytes / BYTES_PER_PIXEL)
    start_index_ravelled = (packet_id - 1) * pixels_per_packet
    return np.unravel_index(
        np.arange(payload_size_pixels) + start_index_ravelled, shape
    )
