from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Recorder(ABC):
    """
    Abstract base class for recorders.

    This class defines the interface for different types of recorders
    (e.g., video recorders, frame recorders).
    """

    @abstractmethod
    def write(self, img: np.ndarray, id: Optional[int] = None) -> None:
        """
        Write an image to the recorder.

        Args:
            img (np.ndarray): The image array to record.
            id (Optional[int]): An optional identifier for the image.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """
        Release any resources used by the recorder.
        """
        pass

    @abstractmethod
    def info(self) -> str:
        """
        Provide information about the recorder.

        Returns:
            str: A string with information about the recorder.
        """
        pass
