from abc import ABC, abstractmethod
from typing import Optional
from tenacity import retry
from vimba import Camera, Frame, FrameStatus
import numpy as np
import cv2
import threading


from gige.gvsp_frame import convert_image_to_rgb


class GigeHandler(ABC):
    def __init__(self) -> None:
        self.shutdown_event = threading.Event()

    @abstractmethod
    def __call__(self, cam: Camera, frame: Frame) -> None:
        raise NotImplemented

    def get_rgb_image(self, cam: Camera, frame: Frame) -> Optional[np.ndarray]:
        with cam:
            if frame.get_status() == FrameStatus.Complete:
                img = frame.as_opencv_image()
                pixel_format = frame.get_pixel_format()
                img = convert_image_to_rgb(img, pixel_format)
                return img
        return None
