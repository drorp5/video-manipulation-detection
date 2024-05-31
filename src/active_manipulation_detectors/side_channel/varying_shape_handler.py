from typing import List, Optional
from vimba import Camera, Frame, FrameStatus

from active_manipulation_detectors.side_channel.bits_encoder import IntBitsEncoderDecoder
from active_manipulation_detectors.side_channel.data_generator import RandomBitsGenerator
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
        sign_detector: Optional[StopSignDetector] = None
    ) -> None:
        super().__init__(detector=sign_detector)
        self.height_values = [MAX_HEIGHT - increment * ind for ind in range(num_levels)]
        self.width_values = [MAX_WIDTH - increment * ind for ind in range(num_levels)]
        self.encoder_decoder = IntBitsEncoderDecoder(self.width_values)
        self.random_bits_generator = random_bits_generator
        self.random_bits_generator.num_bits_per_iteration = (self.encoder_decoder.bits_per_symbol)
        self.data_validator = data_validator
        self.shape_changed = False

    def __call__(self, cam: Camera, frame: Frame):
        if self.is_stop_key_selected():
            return

        with cam:
            img = self.get_rgb_image(cam, frame)
            if img is not None:
                # read data of current image
                if self.shape_changed:
                    # height = frame.get_height()
                    width = frame.get_width()
                    received_symbol = self.encoder_decoder.encode(width)
                    validation_result = self.data_validator.validate(received_symbol)
                    print(f'Frame #{frame.get_id()}: {width} -> {validation_result}\n')

                try:
                    self.plot(img, cam)
                except Exception as e:
                    print(e)
                    pass
            
                # change height for next frame
                symbol = next(self.random_bits_generator) #TODO check if returns bitarray
                new_width = self.encoder_decoder.decode(symbol=symbol)
                cam.Width.set(new_width)
                self.data_validator.add_trasnmitted_data(symbol)
                self.shape_changed = True

            cam.queue_frame(frame)
    
    def cleanup(self, cam: Camera) -> None:
        with cam:
            cam.Width.set(MAX_WIDTH)
