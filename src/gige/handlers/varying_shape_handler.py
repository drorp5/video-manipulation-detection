import logging
import time
from typing import List, Optional
from vimba import Camera, Frame

from active_manipulation_detectors.side_channel.bits_encoder import (
    IntBitsEncoderDecoder,
)
from active_manipulation_detectors.side_channel.data_generator import (
    RandomBitsGenerator,
)
from active_manipulation_detectors.side_channel.validation import DataValidator
from gige.handlers.sign_detector_handler import SignDetectorHandler
from gige.handlers.recorder_handler import RecorderHandler
from active_detection_experiments.recorders import Recorder
from utils.view import add_text_box
from sign_detectors.stop_sign_detectors import StopSignDetector, draw_detections
from gige.gige_constants import MAX_HEIGHT, MAX_WIDTH


class VaryingShapeHandler(SignDetectorHandler, RecorderHandler):
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
        recorder: Optional[Recorder] = None,
    ) -> None:
        SignDetectorHandler.__init__(
            self, logger=logger, downfactor=downfactor, detector=sign_detector
        )
        self.record = recorder is not None
        if self.record:
            RecorderHandler.__init__(self, recorder=recorder, logger=logger)

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
                timestamp = frame.get_timestamp()
                log_dict = {"frame_id": frame_id, "timestamp": timestamp}
                # read data of current image
                if self.shape_changed:
                    # height = frame.get_height()
                    width = frame.get_width()
                    validation_result = self.data_validator.validate(width)
                    log_dict["width"] = width
                    log_dict["validation_result"] = validation_result.name

                if self.detector is not None or self.view or self.record:
                    img_for_detection = self.resize_image(img)
                    if self.detector is not None:
                        detections = self.detect_objects_in_image(img_for_detection)
                        if len(detections) > 0:
                            log_dict["detections"] = detections
                    else:
                        detections = []
                    if self.view:
                        # img = add_text_box(img, f"{frame_id}")
                        plotted_img = self.plot_detected(
                            img_for_detection, cam, detections
                        )
                    else:
                        plotted_img = draw_detections(img_for_detection, detections)
                    if self.record:
                        # plotted_img = add_text_box(plotted_img, f"{frame_id}")
                        self.recorder.write(img_for_detection, id=frame_id)

                # change shape for next frame
                symbol = next(self.random_bits_generator)
                new_width = self.encoder_decoder.decode(symbol=symbol)
                log_dict["next_width"] = new_width
                cam.Width.set(new_width)
                self.data_validator.add_trasnmitted_data(new_width)
                self.shape_changed = True

            self.log(log_dict)
            cam.queue_frame(frame)

    def cleanup(self, cam: Camera) -> None:
        SignDetectorHandler.cleanup(self, cam)
        if self.record:
            RecorderHandler.cleanup(self, cam)
        with cam:
            time.sleep(1)
            cam.Width.set(MAX_WIDTH)
            self.log("Cleanup: Setting width to maximal value", log_level=logging.DEBUG)
