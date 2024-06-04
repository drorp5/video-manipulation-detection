from logging import Logger
from typing import Optional
import cv2
import numpy as np

from gige.handlers.gige_handler import GigeHandler
from vimba import Camera, Frame


ENTER_KEY_CODE = 13


class ViewerHandler(GigeHandler):
    def __init__(self, logger: Optional[Logger] = None, downfactor: int = 4) -> None:
        super().__init__(logger)
        self.downfactor = downfactor

    def resize_image(self, img: np.ndarray) -> np.ndarray:
        height = int(img.shape[0] / self.downfactor)
        width = int(img.shape[1] / self.downfactor)
        dim = (width, height)
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def plot(self, img: np.ndarray, cam: Camera) -> None:
        resized_img = self.resize_image(img)
        window_name = f"Stream from '{cam.get_name()}'. Press <Enter> to stop stream."
        cv2.imshow(window_name, resized_img)

    def is_stop_key_selected(self, delay_ms: int = 1) -> bool:
        key = cv2.waitKey(delay_ms)
        if key == ENTER_KEY_CODE:
            self.shutdown_event.set()
            return True
        return False

    def __call__(self, cam: Camera, frame: Frame) -> None:
        if self.is_stop_key_selected():
            return

        with cam:
            img = self.get_rgb_image(cam, frame)
            if img is not None:
                self.plot(img, cam)

            cam.queue_frame(frame)

    def cleanup(self, cam: Camera) -> None:
        cv2.destroyAllWindows()
