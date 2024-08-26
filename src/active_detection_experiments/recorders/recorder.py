from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class Recorder(ABC):
    @abstractmethod
    def write(self, img: np.ndarray, id: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def info(self) -> str:
        pass
