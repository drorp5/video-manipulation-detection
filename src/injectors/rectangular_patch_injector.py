from typing import Optional
from injectors.patch_injector import PatchInjector


import numpy as np


class RectangularPatchInjector(PatchInjector):
    """
    Injector that inserts a rectangular patch in the upper part of the second frame.
    """

    def __init__(
        self,
        patch_img: np.ndarray,
        first_row: int = 0,
        last_row: Optional[int] = None,
        first_col: int = 0,
        last_col: Optional[int] = None,
    ):
        """
        Initialize the RectangularPatchInjector.

        Args:
            patch_img (np.ndarray): The patch image to inject.
            first_row (int): The first row of the patch.
            last_row (Optional[int]): The last row of the patch.
            first_col (int): The first column of the patch.
            last_col (Optional[int]): The last column of the patch.
        """
        super().__init__(
            patch_img=patch_img,
            first_row=first_row,
            last_row=last_row,
            first_col=first_col,
            last_col=last_col,
        )
        self.set_mask()

    def set_mask(self) -> None:
        """
        Create a rectangular mask for the patch.
        """
        self._mask = np.ones_like(self.patch_img) * 255

    @property
    def name(self):
        return "rectangular_patch_injector"
