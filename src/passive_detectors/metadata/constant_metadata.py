from dataclasses import dataclass
from vimba import PixelFormat, Frame

@dataclass
class FrameConstantMetadata:
    width : int
    height : int
    pixel_format : PixelFormat
    

def extract_constant_metadata(frame: Frame) -> FrameConstantMetadata:
    width = frame.get_width()
    height = frame.get_height()
    pixel_format = frame.get_pixel_format()
    return FrameConstantMetadata(width, height, pixel_format)
