from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, List, Optional
from pathlib import Path
import cv2

def to_batches(l: List, batch_size: int) -> List:
    """split list to batches"""
    return [l[offset:offset+batch_size] for offset in range(0,len(l),batch_size)]

def read_frame(frame_path: Path) -> np.ndarray:
    """read RGB image from path"""
    frame = cv2.imread(frame_path.as_posix())
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame
   
   
class EvaluationDataset(ABC):
    def __init__(self, name: str): 
        self._index = 0
        self._name = name
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._index < len(self):
            val = self[self._index]
            self._index += 1
            return val
        else:
            raise StopIteration
            
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    def name(self) -> str:
        return self._name

class FramesDirectoryDataset(EvaluationDataset):
    def __init__(self, frames_dir: Path, name: str, min_frame_id: int = 0):
        super().__init__(name=name)
        self.frames_dir = frames_dir
        self.min_frame_id = min_frame_id
        self.all_frames_path = self.get_all_frames_path()
        self.pairs_path = to_batches(self.all_frames_path, batch_size=2)
    
    def get_all_frames_path(self) -> List[Path]:
        """return path of all jpg images in directory ordered by ids"""
        all_frames_path = []
        for frame_path in self.frames_dir.glob('*.jpg'):
            id = int(frame_path.stem.split('_')[1])
            if id < self.min_frame_id:
                pass
            else:
                all_frames_path.append(frame_path)
        return sorted(all_frames_path, key=lambda x: int(x.stem.split("_")[1]))
        
    def __len__(self):
        return len(self.pairs_path)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return read_frame(self.pairs_path[idx][0]), read_frame(self.pairs_path[idx][1])
    

class VideoDataset(EvaluationDataset):
    def __init__(self,video_path:Path, num_frames: int, dst_shape: Optional[Tuple[int, int]]=None, flip: bool=False, name: Optional[str]=None):
        if name is None: 
            name = video_path.stem
        super().__init__(name=name)
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path.as_posix())
        self.num_frames = num_frames
        self.frames_offset = self.get_linear_spaced_offsets()
        self.dst_shape = dst_shape
        self.flip = flip
    
    def get_frame(self, index: int) -> np.ndarray:
        """get single frame of video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if self.flip:
            frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.dst_shape is not None:
            frame = cv2.resize(frame, self.dst_shape)
        return frame
    
    def get_linear_spaced_offsets(self):
        """get frames of video in equal steps"""
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_offset = np.linspace(0, total_frames-3, self.num_frames).astype(int)
        return frames_offset
    
    def release(self) -> None:
        self.cap.release()
        
    def __len__(self) -> int:
        return len(self.frames_offset)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_frame(self.frames_offset[idx]), self.get_frame(self.frames_offset[idx]+1)
    

class VideosDirectoryDataset(EvaluationDataset):
    def __init__(self,videos_dir:Path, name:str, num_frames: int, dst_shape: Optional[Tuple[int, int]]=None, flip: bool=False):
        super().__init__(name=name)
        self.videos_dir = videos_dir
        self.videos_path = list(videos_dir.glob('*'))
        self.num_frames = num_frames
        self.dst_shape = dst_shape
        self.flip = flip
        self._video_index = 0
        self._set_cur_dataset()
        
    def _set_cur_dataset(self):
        if self._video_index < len(self.videos_path):
            self.video_dataset = VideoDataset(video_path=self.videos_path[self._video_index], num_frames=self.num_frames, dst_shape=self.dst_shape, flip=self.flip)
        else:
            self.video_dataset = None

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.video_dataset.__getitem__(idx)

    def __len__(self) -> int:
        return len(self.videos_path)

    def __iter__(self):
        return self
    
    def __next__(self):
        if self._video_index < len(self):
            try:
                val = self.video_dataset.__next__()
                return val
            except StopIteration:
                self.video_dataset.release()
                self._video_index += 1
                self._set_cur_dataset()
                return self.__next__()
            except Exception as e:
                print(f'Error loading video from {self.videos_path[self._video_index]}: {e}')
                self.video_dataset.release()
                self._video_index += 1
                self._set_cur_dataset()
                return self.__next__()
        else:
            raise StopIteration