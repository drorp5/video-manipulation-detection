from dataclasses import dataclass


@dataclass
class FramesStatistics:
    total_frames: int
    completed_frames: int
    partial_frames: int

    def pl(self) -> float:
        return 1 - self.completed_frames / self.total_frames

    def __str__(self) -> str:
        return f"Total Frames = {self.total_frames}\nCompleted Frames = {self.completed_frames}\nPartial Frames = {self.partial_frames}\nPL = {self.pl()}"
