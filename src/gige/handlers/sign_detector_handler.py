from logging import Logger
from typing import List, Optional, Tuple
import cv2
import numpy as np
from vimba import Camera, Frame

from gige.handlers.viewer_handler import ViewerHandler
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
        ViewerHandler.__init__(self, logger=logger, downfactor=downfactor)
        self.detector = detector

    def resize_for_detection(self, img: np.ndarray) -> np.ndarray:
        return self.resize_image(img)

    def detect_objects_in_image(self, img: np.ndarray) -> Optional[List[np.ndarray]]:
        if self.detector is not None:
            return self.detector.detect(img)

    def plot_detected(
        self,
        img: np.ndarray,
        cam: Camera,
        detections: Optional[List[np.ndarray]] = None,
    ) -> np.ndarray:
        if detections is not None:
            img = draw_bounding_boxes(img, detections)
        window_name = f"{self.start_time}: Press <Enter> to stop stream."
        cv2.imshow(window_name, img)
        return img

    def plot(self, img: np.ndarray, cam: Camera) -> np.ndarray:
        img = self.resize_for_detection(img)
        detections = self.detect_objects_in_image(img)
        self._plotted_img = self.plot_detected(img, cam, detections)
        return self._plotted_img
