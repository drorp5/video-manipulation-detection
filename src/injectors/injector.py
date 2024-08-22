import numpy as np


from abc import ABC, abstractmethod


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
