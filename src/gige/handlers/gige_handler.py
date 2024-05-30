from abc import ABC, abstractmethod
from typing import Optional
from vimba import Camera, Frame, FrameStatus
import numpy as np
import threading
import logging


from gige.gvsp_frame import convert_image_to_rgb


class GigeHandler(ABC):
    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.shutdown_event = threading.Event()
        self.logger = logger

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

    def log(self, msg, log_level=logging.INFO):
        if self.logger is None:
            print(msg)
            return
        if log_level == logging.DEBUG:
            self.logger.debug(msg)
        elif log_level == logging.INFO:
            self.logger.info(msg)
        elif log_level == logging.WARNING:
            self.logger.warning(msg)
        elif log_level == logging.ERROR:
            self.logger.error(msg)
        elif log_level == logging.CRITICAL:
            self.logger.critical(msg)
        else:
            raise ValueError(f"Invalid log level: {log_level}")
