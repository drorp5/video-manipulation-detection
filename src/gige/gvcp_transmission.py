from scapy.all import RawPcapNgReader, Ether
from pathlib import Path
from typing import Dict, Optional
from .pcap import PcapParser
from .gige_constants import GigERegisters, GvcpCommands
from .constansts import Layers

class GvcpPcapParser(PcapParser):
    def __init__(self, pcap_path: Path):
        super().__init__(pcap_path)
    
    def find_writereg_ack_packet(self, reg:GigERegisters, packet_offset: int) -> Optional[int]:
        write_reg_pkt = None
        write_reg_pkt_offset = None
        write_reg_ack_pkt = None
        write_reg_ack_pkt_offset = None

        raw_reader = RawPcapNgReader(self.pcap_path.as_posix())
        pkts_counter = 0
        for packet_data, _ in raw_reader:
            if pkts_counter >= packet_offset:
                # check if gvcp write or ack
                pkt = Ether(packet_data)
                if pkt.haslayer(Layers.GVCP.value):
                    if pkt[Layers.GVCP.value].Command == GvcpCommands.WRITEREG_CMD and pkt[Layers.GVCP.value].RegisterAddress == reg:
                        write_reg_pkt = pkt
                        write_reg_pkt_offset = pkts_counter
                    elif pkt[Layers.GVCP.value].Command == GvcpCommands.WRITEREG_ACK and write_reg_pkt and pkt[Layers.GVCP.value].RequestID == write_reg_pkt.RequestID:
                        #TODO: check if ack is SUCCESS
                        write_reg_ack_pkt = pkt
                        write_reg_ack_pkt_offset = pkts_counter
                        break
            pkts_counter += 1
        raw_reader.close()
        if write_reg_ack_pkt_offset:
            return write_reg_ack_pkt_offset
