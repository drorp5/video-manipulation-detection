from logging import Logger
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from vimba import Camera, Frame

from gige.handlers.gige_handler import GigeHandler
from utils.video_recorder import VideoReocrder

ENTER_KEY_CODE = 13


class VideoRecorderHandler(GigeHandler):
    def __init__(
        self,
        video_recorder: VideoReocrder,
        logger: Optional[Logger] = None,
    ) -> None:
        super().__init__(logger=logger)
        self.video_recoder = video_recorder

    def __call__(self, cam: Camera, frame: Frame) -> None:
        if self.is_stop_key_selected():
            return

        with cam:
            img = self.get_rgb_image(cam, frame)
            if img is not None:
                self.video_recoder.write(img)

            cam.queue_frame(frame)

    def cleanup(self, cam: Camera) -> None:
        self.video_recoder.release()
        self.log(
            f"Video saved in {self.video_recoder.video_path.absolute().as_posix()}"
        )
