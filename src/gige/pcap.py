from pathlib import Path
from scapy.all import RawPcapNgReader

class PcapParser():
    def __init__(self, pcap_path: Path):
        assert pcap_path.exists(), 'pcap not found'
        self.pcap_path = pcap_path
        self.name = pcap_path.stem
        self.base_dir = pcap_path.parent
        
    @property
    def length(self):
        pkts_counter = 0
        raw_reader = RawPcapNgReader(self.pcap_path.as_posix())
        for _ in raw_reader:
            pkts_counter += 1
        raw_reader.close()
        return pkts_counter
