from typing import Optional
from injectors.injector import Injector
from utils.injection import RectangularBoundaries


import numpy as np


from abc import abstractmethod


class PatchInjector(Injector):
    """
    Base class for injectors that insert a patch into a frame.
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
        Initialize the PatchInjector.

        Args:
            patch_img (np.ndarray): The patch image to inject.
            first_row (int): The first row of the patch.
            last_row (Optional[int]): The last row of the patch.
            first_col (int): The first column of the patch.
            last_col (Optional[int]): The last column of the patch.
        """
        if last_col is None:
            last_col = patch_img.shape[1] - 1

        self.patch_img = patch_img[:, first_col : last_col + 1]

        self.first_row = first_row
        if last_row is None:
            self.last_row = self.patch_img.shape[0] - 1
        else:
            self.last_row = last_row

        self.num_rows = self.last_row - self.first_row + 1
        self.num_cols = self.patch_img.shape[1]
        self.center = (self.patch_img.shape[0] // 2, self.patch_img.shape[1] // 2)

    @abstractmethod
    def set_mask(self) -> None:
        self._mask = None

    @property
    def mask(self) -> np.ndarray:
        """
        Set the mask for the patch. To be implemented by subclasses.
        """
        return self._mask

    def _inject_patch(
        self, frame: np.ndarray, boundaries: RectangularBoundaries
    ) -> np.ndarray:
        """
        Inject the patch into a frame.

        Args:
            frame (np.ndarray): The frame to inject the patch into.
            boundaries (RectangularBoundaries): The boundaries of the injection area.

        Returns:
            np.ndarray: The frame with the injected patch.
        """
        dst_frame = frame.copy()
        frame_cropped = frame[
            boundaries.top_row : boundaries.bottom_row,
            boundaries.left_col : boundaries.right_col,
            :,
        ]
        dst_frame[
            boundaries.top_row : boundaries.bottom_row,
            boundaries.left_col : boundaries.right_col,
            :,
        ] = np.where(
            ~self.mask[self.first_row : self.last_row + 1, :, :],
            frame_cropped,
            self.patch_img[self.first_row : self.last_row + 1, :, :],
        )
        return dst_frame

    def _get_injection_boundaries(
        self, frame: np.ndarray, destination_center=None
    ) -> RectangularBoundaries:
        """
        Calculate the boundaries for patch injection.

        Args:
            frame (np.ndarray): The frame to inject into.
            destination_center (Optional[Tuple[int, int]]): The center point for injection.

        Returns:
            RectangularBoundaries: The calculated boundaries for injection.
        """
        if destination_center is None:
            destination_center = (
                self.num_rows // 2,
                frame.shape[1] - int(np.ceil(self.patch_img.shape[1] / 2)),
            )
        top_row = destination_center[0] - self.num_rows // 2
        bottom_row = destination_center[0] + self.num_rows // 2 + self.num_rows % 2
        left_col = destination_center[1] - self.num_cols // 2
        right_col = destination_center[1] + self.num_cols // 2 + self.num_cols % 2
        return RectangularBoundaries(
            top_row=top_row,
            bottom_row=bottom_row,
            left_col=left_col,
            right_col=right_col,
        )

    def inject(self, frame_1: np.ndarray, frame_2: np.ndarray) -> np.ndarray:
        """
        Inject the patch into the frame.

        Args:
            frame_1 (np.ndarray): The first frame.
            frame_2 (np.ndarray): The second frame.

        Returns:
            np.ndarray: The manipulated frame with the injected patch.
        """
        assert frame_1.shape == frame_2.shape, "SHAPE ERROR: frames must be same size"
        boundaries = self._get_injection_boundaries(frame_1)
        frame_1_injected = self._inject_patch(frame=frame_1, boundaries=boundaries)
        fake_frame = frame_2.copy()
        fake_frame[: self.num_rows, :, :] = frame_1_injected[
            boundaries.top_row : boundaries.bottom_row, :, :
        ]
        return fake_frame
