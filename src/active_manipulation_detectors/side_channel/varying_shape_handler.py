from typing import List
from vimba import Camera, Frame, FrameStatus

from active_manipulation_detectors.side_channel.bits_encoder import IntBitsEncoderDecoder
from active_manipulation_detectors.side_channel.data_generator import RandomBitsGenerator
from active_manipulation_detectors.side_channel.validation import DataValidator
from gige.handlers import ViewerHandler


TOTAL_ROWS = 1216


class VaryingShapeHandler(ViewerHandler):
    def __init__(
        self,
        random_bits_generator: RandomBitsGenerator,
        data_validator: DataValidator,
        num_levels: int,
        increment: int = 1,
    ) -> None:
        super().__init__()
        self.rows_values = [TOTAL_ROWS - increment * ind for ind in range(num_levels)]
        self.encoder_decoder = IntBitsEncoderDecoder(self.rows_values)
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
                height = img.shape[0]
                received_symbol = self.encoder_decoder.encode(height)
                if self.shape_changed:
                    validation_result = self.data_validator.validate (received_symbol)
                    print(validation_result)


                self.plot(img, cam)

            
                # change height for next frame
                symbol = next(self.random_bits_generator) #TODO check if returns bitarray
                num_rows = self.encoder_decoder.decode(symbol=symbol)
                cam.Height.set(num_rows)
                self.shape_changed = True


            cam.queue_frame(frame)
