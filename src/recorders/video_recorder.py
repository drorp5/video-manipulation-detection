from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


from recorders.recorder import Recorder

class VideoReocrder(Recorder):
    def __init__(
        self, video_path: Path, video_shape: Tuple[int, int], fps: Optional[float] = 20, save_images: bool = False 
    ) -> None:
        self.video_path = video_path
        self.video_shape = video_shape
        self.fps = fps
        self.save_images = save_images

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            video_path.as_posix(), fourcc, fps, video_shape
        )

    @property
    def width(self) -> int:
        return self.video_shape[0]

    @property
    def height(self) -> int:
        return self.video_shape[1]

    import numpy as np

    def pad_image(self, img: np.ndarray) -> np.ndarray:
        """
        Pads an image (2D or 3D) with black pixels to match the video width and height.

        Args:
            img: A numpy array representing the image (2D or 3D).

        Returns:
            A numpy array representing the padded image.
        """

        # Check if image is 2D or 3D (grayscale or color)
        if len(img.shape) == 2:
            # Grayscale image
            pad_value = [0]  # Black for grayscale
        else:
            # Color image
            pad_value = [0, 0, 0]  # Black for each color channel (RGB)

        # Calculate padding required in height and width
        pad_height = self.height - img.shape[0]
        pad_width = self.width - img.shape[1]

        # Check if padding is needed
        if pad_height <= 0 or pad_width <= 0:
            return img  # No padding required, return original image

        # Create padding arrays with black pixel values
        padding_bottom = np.full((pad_height, img.shape[1]), pad_value, dtype=img.dtype)
        padding_right = np.full((img.shape[0], pad_width), pad_value, dtype=img.dtype)

        # Concatenate image with padding arrays
        padded_img = np.concatenate((img, padding_bottom), axis=0)
        padded_img = np.concatenate((padded_img, padding_right), axis=1)

        return padded_img

    def write(self, img: np.ndarray, id: Optional[int] = None) -> None:
        padded_img = self.pad_image(img)
        self.video_writer.write(padded_img)
        if self.save_images:
            dst_path = self.video_path.parent / f"{id}.png"
            cv2.imwrite(dst_path.as_posix(), img)

    def release(self) -> None:
        self.video_writer.release()

    def info(self) -> str:
        return f"Video saved in {self.video_path.absolute().as_posix()}"