import cv2
import numpy as np
from scapy.all import PacketList, PcapReader, RawPcapNgReader, Ether
from tqdm import tqdm
from .gvsp_frame import MockFrame, MissingLeaderError, gvsp_frame_to_rgb
from .constansts import Layers
from manipultation_utils import Gvsp, GvspLeader, GvspTrailer #TODO: change location of modules
from pathlib import Path
from typing import Dict, Optional, Tuple
from .pcap import PcapParser
from .constansts import Layers


class GvspPcapParser(PcapParser):
    def __init__(self, pcap_path: Path, max_frames:Optional[int]=None, completed_only:bool=True):
        super().__init__(pcap_path)
        self.max_frames = max_frames
        self.completed_only=completed_only
        self.iteration_stopped = False
        self.last_packet = None
    
    def _get_next_packet(self):
        if self.last_packet:
            return self.last_packet
        return next(self.pcap_reader)
        
    def _next(self) -> Optional[MockFrame]:
        frame_id = None
        while(frame_id is None):
            try:
                pkt = self._get_next_packet()
                if pkt.haslayer(Layers.GVSP.value):
                    frame_id = pkt.BlockID
                    if frame_id == 0:
                        frame_id = None
                    
            except StopIteration:
                self.iteration_stopped = True
                return None
            
        frame_packets = []
        while(not pkt.haslayer(Layers.GVSP.value) or pkt.BlockID == frame_id):
            if pkt.haslayer(Layers.GVSP.value):
                frame_packets.append(pkt)
            try:
                pkt = next(self.pcap_reader)
                if not pkt.haslayer(Layers.GVSP.value):
                    continue
            except StopIteration:
                self.iteration_stopped = True
                break
        self.last_packet = pkt
        return MockFrame(PacketList(frame_packets))
        
    @property
    def frames(self):
        self.pcap_reader = PcapReader(self.pcap_path.as_posix())
        frame = self._next()
        while not self.iteration_stopped:
            yield frame
            try:
                frame = self._next()
            except MissingLeaderError as e:
                frame = None
    
    @property
    def images(self):
        self.pcap_reader = PcapReader(self.pcap_path.as_posix())
        frame = None
        while not self.iteration_stopped:
            while not self.iteration_stopped and frame is None:
                frame = self._next()
                if self.completed_only and not frame.success_status:
                    frame = None
            if frame is not None:
                img =  cv2.cvtColor(gvsp_frame_to_rgb(frame), cv2.COLOR_RGB2BGR)
                frame_id = frame.get_id()
                frame = None
                yield img, frame_id

    def save_images(self, dst_dir: Path):
        dst_dir_path = Path(dst_dir)
        if not dst_dir_path.exists():
            dst_dir_path.mkdir(exist_ok=True)
        frames_counter = 0
        for img, frame_id in tqdm(self.images):
            output_path = dst_dir_path / f'frame_{frame_id}.jpg'
            cv2.imwrite(output_path.as_posix(), img)
            frames_counter += 1
            if self.max_frames is not None and frames_counter >= self.max_frames:
                break

    def save_intensities(self, dst_path: Optional[Path]=None) -> Dict:
        if dst_path is None:
            dst_path = self.base_dir / f'{self.name}_intensities.txt'
        intensities = {}
        frames_counter = 0
        for img, frame_id in tqdm(self.images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
            intensities[frame_id] = np.mean(gray)
            with open(dst_path, 'a') as f:
                f.write(f'frame {frame_id}: {intensities[frame_id]}\n')
            frames_counter += 1
            if self.max_frames is not None and frames_counter >= self.max_frames:
                break
        return intensities

    def save_images_and_intensities(self, dst_dir: Path) -> Dict:
        dst_dir_path = Path(dst_dir)
        if not dst_dir_path.exists():
            dst_dir_path.mkdir(exist_ok=True)
        intensities = {}
        dst_path = dst_dir / f'{self.name}_intensities.txt'
        frames_counter = 0
        for img, frame_id in tqdm(self.images):
            output_path = dst_dir_path / f'frame_{frame_id}.jpg'
            cv2.imwrite(output_path.as_posix(), img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
            intensities[frame_id] = np.mean(gray)
            with open(dst_path, 'a') as f:
                f.write(f'frame {frame_id}: {intensities[frame_id]}\n')
            frames_counter += 1
            if self.max_frames is not None and frames_counter >= self.max_frames:
                break

    def save_video(self, dst_dir: Path, fps=30):
        filename = f'{self.name}.mp4'
        if not dst_dir.exists():
            dst_dir.mkdir(exist_ok=True)
        dst_path = dst_dir / filename
        height = None
        width = None
        prev_id = None
        prev_img = None
        frames_counter = 0
        first_img = True
        for img, frame_id in self.images:
            if first_img:
                height, width = img.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(dst_path.as_posix(), fourcc, fps, (width, height))
                prev_id = frame_id-1
                first_img = False
            for _ in range(prev_id+1, frame_id):
                video_writer.write(prev_img)
            video_writer.write(img)
            prev_img = img
            prev_id = frame_id
            frames_counter += 1
            if self.max_frames is not None and frames_counter >= self.max_frames:
                break
        video_writer.release()
            
    def get_next_frame_id_and_packet_offset(self, packet_offset: int = 0) -> Optional[Tuple[int, int]]:
        raw_reader = RawPcapNgReader(self.pcap_path.as_posix())
        pkts_counter = 0
        gvsp_frame_found = False
        for packet_data, _ in raw_reader:
            if pkts_counter >= packet_offset:
                # check if gvsp leader packet
                pkt = Ether(packet_data)
                if pkt.haslayer(Layers.GVSP_LEADER.value):
                    gvsp_frame_found = True
                    break 
            pkts_counter += 1
        raw_reader.close()
        if gvsp_frame_found:
            return pkt[Layers.GVSP.value].BlockID, pkts_counter
        
    def find_frame_id(self, target_frame_id: int) -> Optional[Tuple[int, int]]:
        # implement binary search to find nearest frame to target
        start_marker = 0
        end_marker = self.length
        nearest_frame_id = np.inf
        nearest_packet_offset = None
        frames_err = 10
        
        while start_marker < end_marker:
            try:
                frame_id, packet_offset = self.get_next_frame_id_and_packet_offset(packet_offset=(start_marker + end_marker) // 2)
            except:
                # next frame not found
                end_marker = np.floor((start_marker + end_marker) / 2)
                continue
            if frame_id > target_frame_id:
                # current offset too big
                end_marker = np.floor((start_marker + end_marker) / 2)
            elif frame_id < target_frame_id:
                # current offset too little
                start_marker = np.ceil((start_marker + end_marker) / 2)
            else:
                # frame found
                return frame_id, packet_offset
            if abs(frame_id - target_frame_id) < abs(nearest_frame_id - target_frame_id):
                nearest_frame_id = frame_id
                nearest_packet_offset = packet_offset
            if abs(nearest_frame_id - target_frame_id) < frames_err:
                break
        
        est_packets_per_frame = 270
        if nearest_frame_id > target_frame_id:
            nearest_packet_offset -= est_packets_per_frame * (nearest_frame_id - target_frame_id)
            return self.get_next_frame_id_and_packet_offset(packet_offset=nearest_packet_offset)
        return nearest_frame_id, nearest_packet_offset