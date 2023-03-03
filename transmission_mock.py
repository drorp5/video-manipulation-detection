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
    


class MockFrame:
    """This class mocks GVSP frame for testing.
     Wraps image data and some metadata."""

    def __init__(self, gvsp_frame_packets: PacketList):
        # check that starts with leader and ends with trailer
        assert gvsp_frame_packets[0].haslayer(GVSP_LEADER_LAYER)
        leader = gvsp_frame_packets[0]
        self._id = leader.BlockID
        self._timestamp = leader.Timestamp
        self._width = leader.SizeX
        self._height = leader.SizeY
        self._pixel_format = INT_TO_PIXEL_FORMAT[leader.PixelFormat]
        
        assert gvsp_frame_packets[1].PacketID == 1
        payload_length = len(bytes(gvsp_frame_packets[1][GVSP_LAYER].payload))

        pixels_bytes = []
        next_frame_id = 1
        for packet in gvsp_frame_packets[1:-1]:
            current_id = packet.PacketID
            if current_id != next_frame_id:
                missing_frames = current_id - next_frame_id
                filling_bytes = bytes([0] * payload_length * missing_frames)
                pixels_bytes += filling_bytes
                next_frame_id = current_id
            pkt_bytes = bytes(packet[GVSP_LAYER].payload)
            pixels_bytes += pkt_bytes
            next_frame_id += 1
            
        assert len(pixels_bytes) == self._height * self._width
        pixels = np.array(pixels_bytes, dtype=np.uint8)
        bggr_pixels = pixels.reshape((self._height, self._width))
        self._img = np.empty((self._height, self._width), np.uint8)
        # strided slicing for this pattern:
        #   R G
        #   G B
        self._img[0::2, 0::2] = bggr_pixels[1::2, 1::2] # top left
        self._img[0::2, 1::2] = bggr_pixels[0::2, 1::2] # top right
        self._img[1::2, 0::2] = bggr_pixels[1::2, 0::2] # bottom left
        self._img[1::2, 1::2] = bggr_pixels[0::2, 0::2] # bottom right

    @property
    def width(self):
        return self._width

    def get_width(self):
        return self.width

    @property
    def height(self):
        return self._height

    def get_height(self):
        return self.height

    @property
    def pixel_format(self):
        return self._pixel_format

    def get_pixel_format(self):
        return self.pixel_format

    @property
    def id(self):
        return self._id
    
    def get_id(self):
        return self.id

    @property
    def timestamp(self):
        return self._timestamp

    def get_timestamp(self):
        return self.timestamp

    @property
    def img(self):
        return self._img

    def as_opencv_image(self):
        return self.img


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
            except Exception as e:
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