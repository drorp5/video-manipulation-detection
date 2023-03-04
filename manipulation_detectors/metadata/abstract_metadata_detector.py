from ..abstract_detector import *
from vimba import Frame, PixelFormat
from typing import Tuple

class MetadataDetector(ManipulationDetector):
    def __init__(self):
        self.current_metadata = None
        self.prev_metadata = None
    
    @abstractmethod
    def pre_process(self, frame: Frame) -> None:
        pass

    def post_process(self) -> None:
        self.prev_metadata = self.current_metadata
        self.current_metadata = None

    def detect(self, frame: Frame) -> Tuple[bool, str]:
        self.pre_process(frame)
        detection_result = self.validate()
        self.post_process()
        is_real_img = detection_result.passed
        message = detection_result.message.value
        return is_real_img, message
