from scapy.all import PacketList, PcapReader
from .gvsp_frame import MockFrame, MissingLeaderError
from .constansts import *
from manipultation_utils import Gvsp, GvspLeader, GvspTrailer #TODO: change location of modules


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
        while(not is_gvsp_packet or pkt.BlockID == frame_id):
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
            except MissingLeaderError as e:
                frame = None
