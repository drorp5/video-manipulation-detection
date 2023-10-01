import numpy as np
import cv2
import matplotlib.pyplot as plt
from gvsp_utils.gvsp_transmission import GvspPcapExtractor
from manipulation_detectors.utils import gvsp_frame_to_rgb
from pathlib import Path
from tqdm import tqdm


def gvsp_pcap_to_raw_images(pcap_path: str, dst_dir: str,  max_frames=None, intensities_only=False):
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError
    dst_dir_path = Path(dst_dir)
    if not dst_dir_path.exists():
        dst_dir_path.mkdir(exist_ok=True)
    if max_frames is None:
        max_frames = 10000000 #inf

    intensities_path = dst_dir_path / 'averaged_intensities.txt'
    intensities = {}
    gvsp_transmission = GvspPcapExtractor(pcap_path)
    for frame in tqdm(gvsp_transmission.frames, total=max_frames):
        if frame is not None and frame.success_status:
            img = cv2.cvtColor(gvsp_frame_to_rgb(frame), cv2.COLOR_RGB2BGR)
            frame_id = frame.get_id()
            output_path = dst_dir_path / f'frame_{frame_id}.jpg'
            if not intensities_only:
                cv2.imwrite(output_path.as_posix(), img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
            intensities[frame_id] = np.mean(gray)
            with open(intensities_path, 'a') as f:
                f.write(f'frame {frame_id}: {intensities[frame_id]}\n')
    
def gvsp_pcap_to_video(pcap_path: str, dst_dir: str,  max_frames=None):
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError
    filename = f'new_{pcap_path.stem}.mp4'
    dst_dir_path = Path(dst_dir)
    if not dst_dir_path.exists():
        dst_dir_path.mkdir(exist_ok=True)
    dst_path = dst_dir_path / filename

    gvsp_transmission = GvspPcapExtractor(pcap_path)

    frame_validator = lambda frame: frame is not None and frame.success_status
    frame_pre_processing = lambda frame: cv2.cvtColor(gvsp_frame_to_rgb(frame), cv2.COLOR_RGB2BGR)
    fps = 30
    create_video_from_frames_iterator(gvsp_transmission.frames, dst_path, fps, frame_validator, frame_pre_processing,  max_frames=max_frames)

def create_video_from_frames_iterator(iterator, output_path: Path, fps=30, frame_validator=lambda frame: frame is not None,
                                                                    frame_pre_processing=lambda frame: frame, 
                                                                    max_frames=None):
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
    if max_frames == None:
        max_frames = 10000000 #inf

    frames_counter = 0
    for image in tqdm(iterator, total=max_frames):
        if frame_validator(image):
            image = frame_pre_processing(image)
            video_writer.write(image)
            previous_image = image
        else:
            video_writer.write(previous_image)
        frames_counter +=1 
        if frames_counter == max_frames:
            break

    video_writer.release()


if __name__ ==  "__main__":
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\single_frame_gvsp.pcapng"
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\live_stream_defaults_part.pcapng"
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\driving_in_uni_1-001.pcapng"
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\short_driving_in_parking-002.pcapng"
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\driving_in_uni_2.pcapng"
    gvsp_pcap_path = r"C:\Users\user\Desktop\Dror\video-manipulation-detection\OUTPUT\recording_14_25_57.pcap"

    # gvsp_pcap_to_video(gvsp_pcap_path, r"C:\Users\user\Desktop\Dror\video-manipulation-detection\OUTPUT",  max_frames=10000)
    gvsp_pcap_to_raw_images(gvsp_pcap_path, r'OUTPUT/recording_14_25_57_images')