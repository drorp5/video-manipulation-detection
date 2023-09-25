import cv2
from vimba import Frame, PixelFormat
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('src')
from config import CV2_CONVERSIONS

def gvsp_frame_to_rgb(frame: Frame, cv2_transformation_code: int =  CV2_CONVERSIONS[PixelFormat.BayerRG8]):
    """Extract RGB image from gvsp frame object"""
    img = frame.as_opencv_image()
    rgb_img = cv2.cvtColor(img, cv2_transformation_code)
    return rgb_img

def plot_rgb(rgb_img: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(rgb_img)
    plt.tight_layout()
    plt.show()