import logging
from typing import List, Optional
from vimba import Camera, Frame, FrameStatus

from active_manipulation_detectors.side_channel.bits_encoder import (
    IntBitsEncoderDecoder,
)
from active_manipulation_detectors.side_channel.data_generator import (
    RandomBitsGenerator,
)
from active_manipulation_detectors.side_channel.validation import DataValidator
from gige.handlers import ViewerHandler, SignDetectorHandler
from sign_detectors.stop_sign_detectors import StopSignDetector


MAX_HEIGHT = 1216
MAX_WIDTH = 1936


class VaryingShapeHandler(SignDetectorHandler):
    def __init__(
        self,
        random_bits_generator: RandomBitsGenerator,
        data_validator: DataValidator,
        num_levels: int,
        increment: int = 2,
        logger: Optional[logging.Logger] = None,
        downfactor: int = 4,
        sign_detector: Optional[StopSignDetector] = None,
        view: bool = True,
    ) -> None:
        super().__init__(logger=logger, downfactor=downfactor, detector=sign_detector)
        self.height_values = [MAX_HEIGHT - increment * ind for ind in range(num_levels)]
        self.width_values = [MAX_WIDTH - increment * ind for ind in range(num_levels)]
        self.encoder_decoder = IntBitsEncoderDecoder(self.width_values)
        self.random_bits_generator = random_bits_generator
        self.random_bits_generator.num_bits_per_iteration = (
            self.encoder_decoder.bits_per_symbol
        )
        self.data_validator = data_validator
        self.shape_changed = False
        self.view = view

    def __call__(self, cam: Camera, frame: Frame):
        if self.is_stop_key_selected():
            return

        with cam:
            img = self.get_rgb_image(cam, frame)
            if img is not None:
                frame_id = frame.get_id()
                # read data of current image
                if self.shape_changed:
                    # height = frame.get_height()
                    width = frame.get_width()
                    received_symbol = self.encoder_decoder.encode(width)
                    validation_result = self.data_validator.validate(received_symbol)
                    self.log(f"Frame # {frame_id}: {width} -> {validation_result}")

                if self.detector is not None or self.view:
                    img = self.resize_for_detection(img)
                    if self.detector is not None:
                        detections = self.detect_objects_in_image(img)
                        if len(detections) > 0:
                            self.log(f"DETECTIONS: {detections.__str__()}")
                    if self.view:
                        self.plot_detected(img, cam, detections)

                # change shape for next frame
                symbol = next(self.random_bits_generator)
                new_width = self.encoder_decoder.decode(symbol=symbol)
                self.log(
                    f"Frame # {frame_id}: Setting next frame width = {new_width}",
                    log_level=logging.DEBUG,
                )
                cam.Width.set(new_width)
                self.data_validator.add_trasnmitted_data(symbol)
                self.shape_changed = True

            cam.queue_frame(frame)

    def cleanup(self, cam: Camera) -> None:
        super().cleanup(cam)
        with cam:
            cam.Width.set(MAX_WIDTH)
