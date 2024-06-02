import logging
import threading
from typing import Optional
from vimba import *


from car import Car
from active_manipulation_detectors.asynchronous_grab_opencv_active_detection import (
    get_camera,
    setup_camera,
)
from gige.handlers import GigeHandler
from gige.handlers.varying_shape_handler import (
    VaryingShapeHandler,
)
from active_manipulation_detectors.side_channel.data_generator import (
    RandomBitsGenerator,
)
from active_manipulation_detectors.side_channel.validation import DataValidator
from sign_detectors.stop_sign_detectors import get_detector


class ShapeVaryingLogicCar(Car):
    def __init__(
        self,
        config: dict,
        random_bits_generator: RandomBitsGenerator,
        data_validator: DataValidator,
        logger: Optional[logging.Logger] = None,
        camera_started_event: Optional[threading.Event] = None,
        camera_stopped_event: Optional[threading.Event] = None) -> None:
        super().__init__(logger=logger, camera_started_event=camera_started_event, camera_stopped_event=camera_stopped_event)
        self.config = config
        self.random_bits_generator = random_bits_generator
        self.data_validator = data_validator

    def get_handler(self) -> GigeHandler:
        sign_detector = get_detector(self.config["actions"]["detector"])
        handler = VaryingShapeHandler(
                logger=self.logger,
                random_bits_generator=self.random_bits_generator,
                data_validator=self.data_validator,
                num_levels=self.config["variation"]["num_widths"],
                increment=2,  # bayer
                sign_detector=sign_detector,
                view=self.config["actions"]["viewer"]
            )
        return handler
        
    def _run(self) -> None:
        handler = self.get_handler()
        with Vimba.get_instance():
            with get_camera() as cam:
                setup_camera(cam, fps_val=self.config["camera"]["fps"])
                self.log(
                    f'Finished setting up camera with frame rate {self.config["camera"]["fps"]}',
                    logging.DEBUG)
            
                # Start Streaming with a custom frames buffer
                self.log(
                    f"Starting camera with varying widths: {handler.width_values}",
                    logging.DEBUG,
                )
                try:
                    cam.start_streaming(
                        handler=handler,
                        buffer_count=self.config["camera"]["streaming_buffer"],
                    )
                    if self.camera_started_event is not None:
                        self.camera_started_event.set()
                    handler.shutdown_event.wait(self.config["duration"])
                    self.log(
                        f"Shutting down camera",
                        logging.DEBUG,
                    )
                except Exception as e:
                    self.log(e, logging.ERROR)
                finally:
                    handler.cleanup(cam)
                    cam.stop_streaming()
                    if self.camera_stopped_event is not None:
                        self.camera_stopped_event.set()

