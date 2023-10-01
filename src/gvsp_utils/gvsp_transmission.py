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
    def __init__(self, gvsp_pcap_path: Path):
        assert gvsp_pcap_path.exists(), 'pcap not found'
        self.name = gvsp_pcap_path.stem
        self.pcap_reader = PcapReader(gvsp_pcap_path.as_posix())
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
        frame = self._next()
        while not self.iteration_stopped:
            yield frame
            try:
                frame = self._next()
            except MissingLeaderError as e:
                frame = None
    
    @property
    def images(self, completed_only=True):
        frame = None
        while not self.iteration_stopped:
            while not self.iteration_stopped and frame is None:
                try:
                    frame = self._next()
                except MissingLeaderError as e:
                    frame = None
                if completed_only and not frame.success_status:
                    frame = None
            img =  cv2.cvtColor(gvsp_frame_to_rgb(frame), cv2.COLOR_RGB2BGR)
            frame_id = frame.get_id()
            yield img, frame_id

    def save_images(self, dst_dir: Path, completed_only=True, max_frames=None):
        dst_dir_path = Path(dst_dir)
        if not dst_dir_path.exists():
            dst_dir_path.mkdir(exist_ok=True)
        frames_counter = 0
        for img, frame_id in tqdm(self.frames(completed_only)):
            output_path = dst_dir_path / f'frame_{frame_id}.jpg'
            cv2.imwrite(output_path.as_posix(), img)
            frames_counter += 1
            if frames_counter >= max_frames:
                break

    def save_intensities(self, dst_path: Path, max_frames=None) -> Dict:
        intensities = {}
        frames_counter = 0
        for img, frame_id in tqdm(self.frames(completed_only=True)):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
            intensities[frame_id] = np.mean(gray)
            with open(dst_path, 'a') as f:
                f.write(f'frame {frame_id}: {intensities[frame_id]}\n')
            frames_counter += 1
            if frames_counter >= max_frames:
                break
        return intensities

    def save_images_and_intensities(self, dst_dir: Path, max_frames=None) -> Dict:
        dst_dir_path = Path(dst_dir)
        if not dst_dir_path.exists():
            dst_dir_path.mkdir(exist_ok=True)
        intensities = {}
        dst_path = dst_dir / f'{self.name}_intensities.txt'
        frames_counter = 0
        for img, frame_id in tqdm(self.frames(completed_only=True)):
            output_path = dst_dir_path / f'frame_{frame_id}.jpg'
            cv2.imwrite(output_path.as_posix(), img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
            intensities[frame_id] = np.mean(gray)
            with open(dst_path, 'a') as f:
                f.write(f'frame {frame_id}: {intensities[frame_id]}\n')
            frames_counter += 1
            if frames_counter >= max_frames:
                break

    def save_video(self, dst_dir: str, completed_only=True, max_frames=None, fps=30):
        filename = f'{self.name}.mp4'
        dst_dir_path = Path(dst_dir)
        if not dst_dir_path.exists():
            dst_dir_path.mkdir(exist_ok=True)
        dst_path = dst_dir_path / filename
        height = None
        width = None
        prev_id = None
        prev_img = None
        frames_counter = 0
        first_img = True
        for img, frame_id in self.images(completed_only):
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
            if frames_counter >= max_frames:
                break
        video_writer.release()
            