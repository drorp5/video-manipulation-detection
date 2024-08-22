from injectors.injector import Injector


import cv2
import numpy as np


from typing import Optional, Tuple


class StripeInjector(Injector):
    """
    Manipulation method that replaces a stripe in the upper part of the frame.
    """

    def __init__(
        self,
        fake_img: np.ndarray,
        first_row: int,
        last_row: int,
        dst_shape: Optional[Tuple[int, int]] = None,
    ):
        """
        Initialize the StripeInjector.

        Args:
            fake_img (np.ndarray): The fake image to inject.
            first_row (int): The first row of the stripe.
            last_row (int): The last row of the stripe.
            dst_shape (Optional[Tuple[int, int]]): The desired shape of the output frame.
        """
        self.fake_img = fake_img
        self.first_row = first_row
        self.last_row = last_row
        if dst_shape is not None:
            self.fake_img, self.first_row, self.last_row = self._resize(
                self.fake_img, dst_width=dst_shape[0], dst_height=dst_shape[1]
            )

    def _resize(
        self, img: np.ndarray, dst_width: int, dst_height: int
    ) -> Tuple[np.ndarray, int, int]:
        """
        Resize the image and adjust the stripe boundaries.

        Args:
            img (np.ndarray): The image to resize.
            dst_width (int): The desired width.
            dst_height (int): The desired height.

        Returns:
            Tuple[np.ndarray, int, int]: The resized image and new stripe boundaries.
        """
        if img.shape[1] == dst_width and img.shape[1] == dst_height:
            return img, self.first_row, self.last_row
        resized = cv2.resize(img, (dst_width, dst_height))
        resize_factor = resized.shape[0] / img.shape[0]
        first_row = int(np.floor(self.first_row * resize_factor))
        last_row = int(np.ceil(self.last_row * resize_factor))
        return resized, first_row, last_row

    def inject(self, frame_1: np.ndarray, frame_2: np.ndarray) -> np.ndarray:
        """
        Inject a stripe of the fake image into the frame.

        Args:
            frame_1 (np.ndarray): The first frame (unused in this injector).
            frame_2 (np.ndarray): The second frame to manipulate.

        Returns:
            np.ndarray: The manipulated frame with the injected stripe.
        """
        assert frame_1.shape == frame_2.shape, "SHAPE ERROR: frames must be same size"
        if self.fake_img.shape != frame_1.shape:
            fake_img, first_row, last_row = self._resize(
                self.fake_img, dst_width=frame_1.shape[1], dst_height=frame_1.shape[0]
            )
        else:
            fake_img = self.fake_img
            first_row = self.first_row
            last_row = self.last_row
        fake_frame = frame_2.copy()
        fake_frame[: last_row - first_row + 1, :, :] = fake_img[
            first_row : last_row + 1, :, :
        ]
        return fake_frame

    @property
    def name(self):
        return "stripe_injector"
