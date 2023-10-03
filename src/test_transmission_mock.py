import cv2
from matplotlib import pyplot as plt
from icecream import ic
from gige.gvsp_transmission import GvspPcapExtractor
from manipulation_detectors.utils import gvsp_frame_to_rgb
from pathlib import Path

if __name__ == "__main__":
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\single_frame_gvsp.pcapng"
    gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\live_stream_defaults_part.pcapng"
    # gvsp_pcap_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\driving_in_uni_1-001.pcapng"

    gvsp_transmission = GvspPcapExtractor(gvsp_pcap_path=Path(gvsp_pcap_path))
    
    num_frames = 0
    for frame in gvsp_transmission.frames:
        if frame is not None and frame.success_status:
            rgb_img = gvsp_frame_to_rgb(frame)
            window_name = 'fig'
            cv2.imshow(window_name, cv2.cvtColor(rgb_img,cv2.COLOR_RGB2BGR))
            num_frames += 1
            ic(frame.id)
            plt.show()
            plt.pause(2)
            