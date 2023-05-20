from typing import List
from scapy.all import PacketList, PcapReader, wrpcap
from .gvsp_frame import MockFrame, MissingLeaderError
from .constansts import *
from manipultation_utils import Gvsp, GvspLeader, GvspTrailer #TODO: change location of modules
from pathlib import Path
import numpy as np


class MockGvspTransmission():
    def __init__(self, gvsp_pcap_path: Path, start_block_id: int=0, end_block_id: int=np.inf):
        self.pcap_reader = PcapReader(gvsp_pcap_path.as_posix())
        self.iteration_stopped = False
        self.start_block_id = start_block_id
        self.end_block_id = end_block_id
        
    def _next(self) -> MockFrame or None:
        frame_id = None
        while(frame_id is None):
            try:
                pkt = next(self.pcap_reader)
                if pkt.haslayer(GVSP_LEADER_LAYER):
                    if pkt.BlockID >= self.start_block_id:
                        frame_id = pkt.BlockID
                    if pkt.BlockID >= self.end_block_id:
                        return None
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

class SplitGvspTransmission():
    def __init__(self, gvsp_pcap_path: Path, scenario_block_start_end_id: List[tuple], dst_dir: Path):
        self.gvsp_pcap_path = gvsp_pcap_path
        self.pcap_reader = PcapReader(gvsp_pcap_path.as_posix())
        self.iteration_stopped = False
        self.start_end_block_indices = scenario_block_start_end_id
        self.current_block_index = 0
        self.start_block_id, self.end_block_id =  scenario_block_start_end_id[self.current_block_index]
        self.current_scenario_packets = []
        self.dst_dir = dst_dir
        self.finished = False
        
    def write_pcap(self):
        output_path = self.dst_dir / f'{self.gvsp_pcap_path.stem}_{self.start_block_id, self.end_block_id}.pcap'
        wrpcap(output_path.as_posix(), self.current_scenario_packets)
        self.current_scenario_packets = []
        self.current_block_index += 1
        self.start_block_id, self.end_block_id =  self.scenario_block_start_end_id[self.current_block_index]
        if self.current_block_index == len(self.scenario_block_start_end_id):
            self.finished = True
        print(f'{output_path.as_posix()} saved')

    def split(self) -> None:
        pkt = next(self.pcap_reader)
        while not self.finished:
            if pkt.haslayer(Gvsp):
                if pkt.BlockID >= self.start_block_id:
                    if pkt.BlockID > self.end_block_id:
                        self.write_pcap()
                    self.current_scenario_packets.append(pkt)
            try:
                pkt = next(self.pcap_reader)
            except StopIteration:
                self.write_pcap()
                self.iteration_stopped = True
                break