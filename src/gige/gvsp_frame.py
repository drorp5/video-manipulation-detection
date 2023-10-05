import cv2
from vimba import Frame, PixelFormat
from typing import Tuple
import numpy as np
from scapy.all import PacketList
from vimba import PixelFormat
import sys
sys.path.append('src')
from manipultation_utils import Gvsp, GvspLeader, GvspTrailer #TODO: change location of modules
from dataclasses import dataclass
from src.gige.constansts import Layers, CV2_CONVERSIONS, INT_TO_PIXEL_FORMAT

class MissingLeaderError(Exception):
    pass

@dataclass
class DefaultLeader:
    SizeY = 1216
    SizeX = 1936
    PixelFormat = 0x1080009
    Timestamp = None

class MockFrame:
    """This class mocks GVSP frame for testing.
     Wraps image data and some metadata."""

    def __init__(self, gvsp_frame_packets: PacketList):
        # check that starts with leader and ends with trailer
        if gvsp_frame_packets[0].haslayer(Layers.GVSP_LEADER.value):
            # raise MissingLeaderError
            self.leader = gvsp_frame_packets[0]
            payload_first_ind = 1
        else:
            self.leader = DefaultLeader()
            payload_first_ind = 0

        if gvsp_frame_packets[-1].haslayer(Layers.GVSP_TRAILER.value):
            payload_last_ind = len(gvsp_frame_packets) - 1 
        else:
            payload_last_ind = len(gvsp_frame_packets)
        
        self.first_packet = gvsp_frame_packets[0]

        raw_pixels, success_status = self.payload_packets_to_raw_image(gvsp_frame_packets[payload_first_ind:payload_last_ind])
        self._success_status = success_status
        self._img = self.adjust_raw_image(raw_pixels) #TODO: account for pixel format
    
    def packet_id_to_payload_indices(self, packet_id: int, payload_size_bytes: int, max_payload_size_bytes: int) -> Tuple[np.ndarray, np.ndarray]:
        bytes_per_pixel = 1
        pixels_per_packet = int(max_payload_size_bytes / bytes_per_pixel)
        payload_size_pixels = int(payload_size_bytes / bytes_per_pixel)
        start_index_ravelled =  (packet_id - 1) * pixels_per_packet
        return np.unravel_index(np.arange(payload_size_pixels) + start_index_ravelled, self.shape)
    
    def payload_packets_to_raw_image(self, payload_packets: PacketList) -> Tuple[np.ndarray, bool]:
        raw_pixels = np.zeros(self.shape, dtype=np.uint8)
        assigned_pixels = np.zeros(self.shape, dtype=bool)
        
        max_payload_size_bytes = np.max([len(bytes(pkt[Layers.GVSP.value].payload)) for pkt in payload_packets])
        for packet in payload_packets:
            current_id = packet.PacketID
            pkt_bytes = bytes(packet[Layers.GVSP.value].payload)
            rows_indices, cols_indices = self.packet_id_to_payload_indices(packet_id=current_id, payload_size_bytes=len(pkt_bytes),
                                                                        max_payload_size_bytes=max_payload_size_bytes)
            raw_pixels[rows_indices, cols_indices] = np.frombuffer(pkt_bytes, dtype=np.uint8)
            assigned_pixels[rows_indices, cols_indices] = True
        return raw_pixels, np.all(assigned_pixels)
    
    def bggr_to_rggb(self, bggr_pixels: np.ndarray) -> np.ndarray:
        rggb_pixels = np.empty((self.height, self.width), np.uint8)
        rggb_pixels = np.copy(bggr_pixels)
        # strided slicing for this pattern:
        #   R G
        #   G B
        rggb_pixels[0::2, 0::2] = bggr_pixels[1::2, 1::2] # top left
        rggb_pixels[1::2, 1::2] = bggr_pixels[0::2, 0::2] # bottom right
        return rggb_pixels

    def adjust_raw_image(self, raw_image: np.ndarray) -> np.ndarray:
        return self.bggr_to_rggb(raw_image)
    
    @property
    def width(self):
        if self.leader:
            return self.leader.SizeX
        return None

    def get_width(self):
        return self.width

    @property
    def height(self):
        if self.leader:
            return self.leader.SizeY
        return None

    def get_height(self):
        return self.height
    
    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def pixel_format(self):
        if self.leader:
            return INT_TO_PIXEL_FORMAT[self.leader.PixelFormat]
        return None

    def get_pixel_format(self):
        return self.pixel_format

    @property
    def id(self):
        return self.first_packet.BlockID
        
    def get_id(self):
        return self.id

    @property
    def timestamp(self):
        if self.leader:
            return self.leader.Timestamp
        return None

    def get_timestamp(self):
        return self.timestamp

    @property
    def img(self):
        return self._img

    def as_opencv_image(self):
        return self.img
    
    @property
    def success_status(self):
        return self._success_status


def gvsp_frame_to_rgb(frame: Frame, cv2_transformation_code: int =  CV2_CONVERSIONS[PixelFormat.BayerRG8]) -> np.array:
    """Extract RGB image from gvsp frame object"""
    img = frame.as_opencv_image()
    rgb_img = cv2.cvtColor(img, cv2_transformation_code)
    return rgb_img
