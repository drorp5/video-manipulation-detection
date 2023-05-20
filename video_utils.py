import numpy as np
import cv2
import matplotlib.pyplot as plt
from transmission_mock.gvsp_transmission import MockGvspTransmission
from manipulation_detectors.utils import gvsp_frame_to_rgb
from pathlib import Path
from tqdm import tqdm

def gvsp_pcap_to_video(pcap_path: str, dst_dir: str):
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError
    filename = f'{pcap_path.stem}.mp4'
    dst_dir_path = Path(dst_dir)
    if not dst_dir_path.exists():
        dst_dir_path.mkdir(exist_ok=True)
    dst_path = dst_dir_path / filename

    gvsp_transmission = MockGvspTransmission(pcap_path.as_posix())

    frame_validator = lambda frame: frame is not None and frame.success_status
    frame_pre_processing = lambda frame: cv2.cvtColor(gvsp_frame_to_rgb(frame), cv2.COLOR_RGB2BGR)
    fps = 30
    create_video_from_frames_iterator(gvsp_transmission.frames, dst_path, fps, frame_validator, frame_pre_processing)


def create_video_from_frames_iterator(iterator, output_path: Path, fps=30, frame_validator=lambda frame: frame is not None,
                                                                    frame_pre_processing=lambda frame: frame):
    first_image = None
    for image in tqdm(iterator):
        if frame_validator(image):
            first_image = frame_pre_processing(image)
            break
    if first_image is None:
        raise ValueError("Iterator does not contain any valid images")

    height, width, _ = first_image.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path.as_posix(), fourcc, fps, (width, height))

    video_writer.write(first_image)

    previous_image = first_image
    for image in tqdm(iterator):
        if frame_validator(image):
            image = frame_pre_processing(image)
            video_writer.write(image)
            previous_image = image
        else:
            video_writer.write(previous_image)

    video_writer.release()


if __name__ ==  "__main__":
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\single_frame_gvsp.pcapng"
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\live_stream_defaults_part.pcapng"
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\driving_in_uni_1-001.pcapng"
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\short_driving_in_parking-002.pcapng"
    gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\driving_in_uni_2.pcapng"



    gvsp_pcap_to_video(gvsp_pcap_path, r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\OUTPUT")