import cv2
import numpy as np
from scapy.all import PacketList, PcapReader, RawPcapNgReader, Ether, Packet
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Optional, Tuple

from gige.gige_constants import Layers
from gige.gvsp_frame import MockFrame
from gige.gvsp_transmission import GvspPcapParser


class AttackedGvspPcapParser(GvspPcapParser):
    def __init__(
        self,
        pcap_path: Path,
        max_frames: Optional[int] = None,
    ):
        super().__init__(pcap_path, max_frames=max_frames)
        self.max_frames = max_frames

    def _next(self) -> Optional[MockFrame]:
        # Find leader packet
        frame_id = None
        while frame_id == None:
            try:
                pkt = self._get_next_packet()
            except StopIteration:
                self.iteration_stopped = True
                return None
            if pkt.haslayer(Layers.GVSP_LEADER.value):
                frame_id = pkt.BlockID if pkt.BlockID != 0 else None

        # Construct frame packets
        frame_packets = [pkt]
        try:
            pkt = next(self.pcap_reader)
        except StopIteration:
            self.iteration_stopped = True
            return None
        while not pkt.haslayer(Layers.GVSP_LEADER.value):
            if pkt.haslayer(Layers.GVSP.value):
                if pkt.BlockID == frame_id:
                    frame_packets.append(pkt)
            try:
                pkt = next(self.pcap_reader)
            except StopIteration:
                self.iteration_stopped = True
                break
        self.last_packet = pkt
        return MockFrame(PacketList(frame_packets))
