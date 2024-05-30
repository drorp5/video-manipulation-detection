from logging import Logger
from typing import Optional
import cv2
import numpy as np
from vimba import Camera, Frame

from gige.handlers import ViewerHandler
from sign_detectors.stop_sign_detectors import StopSignDetector, draw_bounding_boxes


StopSignDetector


ENTER_KEY_CODE = 13


class SignDetectorHandler(ViewerHandler):
    def __init__(
        self,
        logger: Optional[Logger] = None,
        downfactor: int = 4,
        detector: Optional[StopSignDetector] = None,
    ) -> None:
        super().__init__(logger=logger, downfactor=downfactor)
        self.detector = detector

    def detect_objects_in_image(self, img: np.ndarray) -> np.ndarray:
        detections = self.detector.detect(img)
        img = draw_bounding_boxes(img, detections)
        return img

    def plot(self, img: np.ndarray, cam: Camera) -> None:
        img = self.resize_image(img)
        if self.detector is not None:
            img = self.detect_objects_in_image(img)
        window_name = f"Stream from '{cam.get_name()}'. Press <Enter> to stop stream."
        cv2.imshow(window_name, img)
