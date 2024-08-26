from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
from numpy import ndarray

from recorders.recorder import Recorder


class FramesRecorder(Recorder):
    """
    A recorder class for saving individual frames as image files.
    """

    def __init__(self, dst_dir: Path) -> None:
        """
        Initialize the FramesRecorder.

        Args:
            dst_dir (Path): The destination directory to save the frames.
        """
        super().__init__()
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True)
        self.dst_dir = dst_dir

    def write(self, img: ndarray, id: Optional[int] = None) -> None:
        """
        Write a frame to disk as an image file.

        Args:
            img (ndarray): The image array to save.
            id (Optional[int]): An optional identifier for the frame. If not provided,
                                a timestamp will be used.
        """
        if id is not None:
            dst_path = self.dst_dir / f"{id}.png"
        else:
            now = datetime.now()
            start_time_string = now.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
            dst_path = self.dst_dir / f"{start_time_string}.png"
        cv2.imwrite(dst_path.as_posix(), img)

    def info(self) -> str:
        """
        Provide information about where the frames are saved.

        Returns:
            str: A string indicating the directory where frames are saved.
        """
        return f"Frames saved in {self.dst_dir.absolute().as_posix()}"

    def release(self) -> None:
        """
        Release any resources used by the recorder.
        """
        return super().release()
