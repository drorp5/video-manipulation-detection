from typing import Tuple
from pathlib import Path
import re


def get_logged_frames_info(log_path: Path) -> Tuple[int, int, int]:
    if not log_path.exists():
        raise FileNotFoundError

    unique_frames = set()
    frame_id_regex = r"Frame # (\d+)"
    with open(log_path.as_posix(), "r") as log_file:
        for line in log_file:
            matches = re.findall(frame_id_regex, line)
            for match in matches:
                unique_frames.add(int(match))
    if len(unique_frames) == 0:
        return 0, 0, 0
    return min(unique_frames), max(unique_frames), len(unique_frames)


def get_number_of_detections_frames(log_path: Path) -> int:
    if not log_path.exists():
        raise FileNotFoundError
    detections = 0
    with open(log_path.as_posix(), "r") as log_file:
        for line in log_file:
            if "DETECTIONS" in line:
                detections += 1
    return detections


def summarize_log_file(log_path: Path) -> str:
    try:
        first_frame, last_frame, num_frames = get_logged_frames_info(log_path)
        detections_frames = get_number_of_detections_frames(log_path)
    except FileNotFoundError:
        return "Log Not Found"

    summary = f"First frame ID = {first_frame}\n"
    summary = f"Last frame ID = {last_frame}\n"
    summary += f"# Logged Frames = {num_frames}\n"
    summary += f"Total {100*num_frames/(last_frame-first_frame+1):.1f}% logged\n"
    summary += f"Detections Frames = {detections_frames}"
    return summary
