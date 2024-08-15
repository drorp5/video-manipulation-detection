from datetime import datetime
from pathlib import Path
from typing import Optional
import cv2
from numpy import ndarray

from recorders.recorder import Recorder


class FramesRecorder(Recorder):
    def __init__(self, dst_dir: Path) -> None:
        super().__init__()
        if not dst_dir.exists():
            dst_dir.mkdir(parents=True)
        self.dst_dir = dst_dir

    def write(self, img: ndarray, id: Optional[int] = None) -> None:
        if id is not None:
            dst_path = self.dst_dir / f"{id}.png"
        else:
            now = datetime.now()
            start_time_string = now.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
            dst_path = self.dst_dir / f"{start_time_string}.png"
        cv2.imwrite(dst_path.as_posix(), img)
    
    def info(self) -> str:
        return f"Frames saved in {self.dst_dir.absolute().as_posix()}"
    
    def release(self) -> None:
        return super().release()