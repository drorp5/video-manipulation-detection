from logging import Logger
import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from vimba import Camera, Frame

from gige.handlers.gige_handler import GigeHandler
from recorders import Recorder

ENTER_KEY_CODE = 13


class RecorderHandler(GigeHandler):
    def __init__(
        self,
        recorder: Recorder,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__(logger=logger)
        self.recorder = recorder

    def __call__(self, cam: Camera, frame: Frame) -> None:
        if self.is_stop_key_selected():
            return

        with cam:
            img = self.get_rgb_image(cam, frame)
            if img is not None:
                self.recorder.write(img)

            cam.queue_frame(frame)

    def cleanup(self, cam: Camera) -> None:
        self.recorder.release()
        self.log(self.recorder.info(), log_level=logging.DEBUG)
