from injectors.injector import Injector


import cv2
import numpy as np


from typing import Optional, Tuple


class FullFrameInjector(Injector):
    """
    Manipulation method that replaces the entire frame with a new one.
    """

    def __init__(
        self, fake_img: np.ndarray, dst_shape: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize the FullFrameInjector.

        Args:
            fake_img (np.ndarray): The fake image to inject.
            dst_shape (Optional[Tuple[int, int]]): The desired shape of the output frame.
        """
        self.fake_img = fake_img
        self.dst_shape = dst_shape
        if dst_shape is not None:
            self.fake_img = cv2.resize(self.fake_img, dst_shape)

    def inject(self, frame_1: np.ndarray, frame_2: np.ndarray) -> np.ndarray:
        """
        Replace the entire frame with the fake image.

        Args:
            frame_1 (np.ndarray): The first frame (unused in this injector).
            frame_2 (np.ndarray): The second frame (unused in this injector).

        Returns:
            np.ndarray: The fake frame resized to match the input frame dimensions.
        """
        assert frame_1.shape == frame_2.shape, "SHAPE ERROR: frames must be same size"
        return cv2.resize(self.fake_img, (frame_1.shape[1], frame_1.shape[0]))

    @property
    def name(self):
        return "full_frame_injector"
