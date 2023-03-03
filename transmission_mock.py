from typing import Tuple
from scapy.all import rdpcap, PacketList, PcapReader
from manipultation_utils import Gvsp, GvspLeader, GvspTrailer
from vimba import PixelFormat
import numpy as np
import cv2
import matplotlib.pyplot as plt
from icecream import ic

# TODO: get names fron config/constant/other file
GVSP_LAYER = 'Gvsp' 
GVSP_LEADER_LAYER = "GVSP_LEADER"
GVSP_TRAILER_LAYER = "GVSP_TRAILER"

INT_TO_PIXEL_FORMAT = {0x1080009: PixelFormat.BayerRG8}
    

class MissingLeaderError(Exception):
    pass

class MockFrame:
    """This class mocks GVSP frame for testing.
     Wraps image data and some metadata."""

    def __init__(self, gvsp_frame_packets: PacketList):
        # check that starts with leader and ends with trailer
        if not gvsp_frame_packets[0].haslayer(GVSP_LEADER_LAYER):
            raise MissingLeaderError
        self.leader = gvsp_frame_packets[0]

        raw_pixels, success_status = self.payload_packets_to_raw_image(gvsp_frame_packets[1:-1])        
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
        
        max_payload_size_bytes = np.max([len(bytes(pkt[GVSP_LAYER].payload)) for pkt in payload_packets])
        for packet in payload_packets:
            current_id = packet.PacketID
            pkt_bytes = bytes(packet[GVSP_LAYER].payload)
            rows_indices, cols_indices = self.packet_id_to_payload_indices(packet_id=current_id,payload_size_bytes=len(pkt_bytes),
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
        return self.leader.SizeX

    def get_width(self):
        return self.width

    @property
    def height(self):
        return self.leader.SizeY

    def get_height(self):
        return self.height
    
    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def pixel_format(self):
        return self.INT_TO_PIXEL_FORMAT[self.leader.PixelFormat]

    def get_pixel_format(self):
        return self.pixel_format

    @property
    def id(self):
        return self.leader.BlockID
    
    def get_id(self):
        return self.id

    @property
    def timestamp(self):
        return self.leader.Timestamp

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


class MockGvspTransmission():
    def __init__(self, gvsp_pcap_path: str):
        self.pcap_reader = PcapReader(gvsp_pcap_path)
        self.iteration_stopped = False
        
    def _next(self) -> MockFrame or None:
        frame_id = None
        while(frame_id is None):
            try:
                pkt = next(self.pcap_reader)
                if pkt.haslayer(GVSP_LEADER_LAYER):
                    frame_id = pkt.BlockID
            except StopIteration:
                self.iteration_stopped = True
                return None
            
        frame_packets = []
        is_gvsp_packet = pkt.haslayer(GVSP_LAYER)
        while(pkt.BlockID == frame_id or not is_gvsp_packet):
            if is_gvsp_packet:
                frame_packets.append(pkt)
                if pkt.haslayer(GVSP_TRAILER_LAYER):
                    return MockFrame(PacketList(frame_packets))
            try:
                pkt = next(self.pcap_reader)
                is_gvsp_packet = pkt.haslayer(GVSP_LAYER)
            except StopIteration:
                self.iteration_stopped = True
                break
        return None
        
    @property
    def frames(self):
        frame = self._next()
        while not self.iteration_stopped:
            yield frame
            try:
                frame = self._next()
                if not frame.success_status:
                    frame = None
            except MissingLeaderError as e:
                frame = None
                        
if __name__ == "__main__":
    from manipulation_detectors import gvsp_frame_to_rgb
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\single_frame_gvsp.pcapng"
    gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\live_stream_defaults_part.pcapng"

    gvsp_transmission = MockGvspTransmission(gvsp_pcap_path=gvsp_pcap_path)
    num_frames = 0
    for frame in gvsp_transmission.frames:
        if frame is not None:
            rgb_img = gvsp_frame_to_rgb(frame)
            # plt.imshow(rgb_img)
            # plt.show()
            num_frames += 1
            ic(frame.id)