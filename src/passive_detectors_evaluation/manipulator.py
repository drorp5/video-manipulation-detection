from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import cv2
from typing import Optional, Tuple
from dataclasses import dataclass


class Injector(ABC):
    """
    Abstract base class for frame injectors.
    """

    @abstractmethod
    def inject(self, frame_1: np.ndarray, frame_2: np.ndarray) -> np.ndarray:
        """
        Inject content into a frame.

        Args:
            frame_1 (np.ndarray): The first frame.
            frame_2 (np.ndarray): The second frame.

        Returns:
            np.ndarray: The manipulated frame.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the injector.

        Returns:
            str: The name of the injector.
        """
        pass


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


@dataclass
class RectangularBoundaries:
    """
    Data class to store the boundaries of a rectangular region in an image.
    """

    top_row: int
    bottom_row: int
    left_col: int
    right_col: int


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
