from typing import Optional
from injectors.patch_injector import PatchInjector


import cv2
import numpy as np


class SignPatchInjector(PatchInjector):
    """
    Injector that inserts a sign patch in the upper part of the second frame.
    """

    def __init__(
        self,
        sign_img: np.ndarray,
        side_length: int,
        first_row: int = 0,
        last_row: Optional[int] = None,
        first_col: int = 0,
        last_col: Optional[int] = None,
    ):
        """
        Initialize the SignPatchInjector.

        Args:
            sign_img (np.ndarray): The sign image to inject.
            side_length (int): The side length of the octagon.
            first_row (int): The first row of the patch.
            last_row (Optional[int]): The last row of the patch.
            first_col (int): The first column of the patch.
            last_col (Optional[int]): The last column of the patch.
        """
        assert sign_img.shape[0] == sign_img.shape[1], "sign image must be square"
        super().__init__(
            patch_img=sign_img,
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )
        self.side_length = side_length
        self.set_mask()

    def set_mask(self) -> None:
        """
        Create an octagon mask for the sign patch.
        """
        mask = np.zeros_like(self.patch_img)
        octagon_color = 255
        angle_offset = np.pi / 8
        vertices = []
        for i in range(8):
            angle = i * np.pi / 4 + angle_offset
            x = int(self.center[0] + self.side_length * np.cos(angle))
            y = int(self.center[1] + self.side_length * np.sin(angle))
            vertices.append((x, y))
        cv2.fillPoly(
            mask, [np.array(vertices)], (octagon_color, octagon_color, octagon_color)
        )
        self._mask = mask

    @property
    def name(self):
        return "sign_patch_injector"
