from dataclasses import dataclass
from vimba import PixelFormat, Frame

def nanoseconds_to_seconds(nano_sec: float) -> float:
    return nano_sec / 1e9

@dataclass
class FrameVaryingMetadata:
    frame_id : int
    timestamp_seconds : float

def extract_varying_metadata(frame: Frame) -> FrameVaryingMetadata:
    frame_id = frame.get_id()
    timestamp_seconds = nanoseconds_to_seconds(frame.get_timestamp())
    return FrameVaryingMetadata(frame_id=frame_id, timestamp_seconds=timestamp_seconds)
