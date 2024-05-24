import numpy as np
import cv2


def bgr_to_bayer_rg(img_bgr: np.ndarray) -> np.ndarray:
    # note: in open cv I need to read the image as bayer BG to convert it correctly
    (B, G, R) = cv2.split(img_bgr)

    dst_img_bayer = np.empty(B.shape, np.uint8)
    # strided slicing for this pattern:
    #   R G
    #   G B
    dst_img_bayer[0::2, 0::2] = R[0::2, 0::2]  # top left
    dst_img_bayer[0::2, 1::2] = G[0::2, 1::2]  # top right
    dst_img_bayer[1::2, 0::2] = G[1::2, 0::2]  # bottom left
    dst_img_bayer[1::2, 1::2] = B[1::2, 1::2]  # bottom right
    return dst_img_bayer
