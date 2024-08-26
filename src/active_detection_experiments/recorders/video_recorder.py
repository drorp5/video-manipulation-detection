from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


from recorders.recorder import Recorder


class VideoReocrder(Recorder):
    """
    A recorder class for saving frames as a video file.
    """

    def __init__(
        self,
        video_path: Path,
        video_shape: Tuple[int, int],
        fps: Optional[float] = 20,
        save_images: bool = False,
    ) -> None:
        """
        Initialize the VideoRecorder.

        Args:
            video_path (Path): The path where the video will be saved.
            video_shape (Tuple[int, int]): The shape (width, height) of the video.
            fps (Optional[float]): Frames per second for the video. Defaults to 20.
            save_images (bool): Whether to save individual frames as images. Defaults to False.
        """
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
        """
        Write a frame to the video file and optionally save as an image.

        Args:
            img (np.ndarray): The image array to record.
            id (Optional[int]): An optional identifier for the frame.
        """
        padded_img = self.pad_image(img)
        self.video_writer.write(padded_img)
        if self.save_images:
            dst_path = self.video_path.parent / f"{id}.png"
            cv2.imwrite(dst_path.as_posix(), img)

    def release(self) -> None:
        """
        Release the video writer resources.
        """
        self.video_writer.release()

    def info(self) -> str:
        """
        Provide information about where the video is saved.

        Returns:
            str: A string indicating the path where the video is saved.
        """
        return f"Video saved in {self.video_path.absolute().as_posix()}"
