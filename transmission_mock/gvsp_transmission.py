from scapy.all import PacketList, PcapReader
from .gvsp_frame import MockFrame, MissingLeaderError
from .constansts import *
from manipultation_utils import Gvsp, GvspLeader, GvspTrailer #TODO: change location of modules
from pathlib import Path


class MockGvspTransmission():
    def __init__(self, gvsp_pcap_path: Path):
        self.pcap_reader = PcapReader(gvsp_pcap_path.as_posix())
        self.iteration_stopped = False
        self.last_packet = None
    
    def _get_next_packet(self):
        if self.last_packet:
            return self.last_packet
        return next(self.pcap_reader)
        
    def _next(self) -> MockFrame or None:
        frame_id = None
        while(frame_id is None):
            try:
                pkt = self._get_next_packet()
                if pkt.haslayer(Gvsp):
                    frame_id = pkt.BlockID
                    if frame_id == 0:
                        frame_id = None
                    
            except StopIteration:
                self.iteration_stopped = True
                return None
            
        frame_packets = []
        while(not pkt.haslayer(GVSP_LAYER) or pkt.BlockID == frame_id):
            if pkt.haslayer(GVSP_LAYER):
                frame_packets.append(pkt)
            try:
                pkt = next(self.pcap_reader)
                if not pkt.haslayer(GVSP_LAYER):
                    continue
            except StopIteration:
                self.iteration_stopped = True
                break
        self.last_packet = pkt
        return MockFrame(PacketList(frame_packets))
        
    @property
    def frames(self):
        frame = self._next()
        while not self.iteration_stopped:
            yield frame
            try:
                frame = self._next()
            except MissingLeaderError as e:
                frame = None
