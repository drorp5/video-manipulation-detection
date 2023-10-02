import cv2
import numpy as np
from scapy.all import PacketList, PcapReader
from tqdm import tqdm
from .gvsp_frame import MockFrame, MissingLeaderError, gvsp_frame_to_rgb
from .constansts import *
from manipultation_utils import Gvsp, GvspLeader, GvspTrailer #TODO: change location of modules
from pathlib import Path
from typing import Dict, Optional


class GvspPcapExtractor():
    def __init__(self, gvsp_pcap_path: Path, max_frames:Optional[int]=None, completed_only:bool=True):
        assert gvsp_pcap_path.exists(), 'pcap not found'
        self.pcap_path = gvsp_pcap_path
        self.name = gvsp_pcap_path.stem
        self.base_dir = gvsp_pcap_path.parent
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
            