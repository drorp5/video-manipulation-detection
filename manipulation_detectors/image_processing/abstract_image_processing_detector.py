from ..abstract_detector import *
import cv2
import numpy as np
from typing import Tuple

class ImageProcessingDetector(ManipulationDetector):
    "Abstract class for detection based on image processing techniques"

    @abstractmethod
    def pre_process(self, rgb_img: np.ndarray) -> None:
        pass
    
    def detect(self, rgb_img: np.ndarray) -> Tuple[bool, str]:
        self.pre_process(rgb_img)
        detection_result = self.validate()
        self.post_process()
        is_real_img = detection_result.passed
        message = detection_result.message.value
        return is_real_img, message
