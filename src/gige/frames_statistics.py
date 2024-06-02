from dataclasses import dataclass


@dataclass
class FramesStatistics:
    total_frames: int
    completed_frames: int
    partial_frames: int

    def pl(self) -> float:
        return 1 - self.completed_frames / self.total_frames
