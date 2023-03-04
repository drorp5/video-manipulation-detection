import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from icecream import ic
from transmission_mock.gvsp_transmission import MockGvspTransmission
from manipulation_detectors.abstract_detector import *
from manipulation_detectors.metadata import *
from manipulation_detectors.image_processing import *
from manipulation_detectors.combined import CombinedDetector
from manipulation_detectors.utils import *
from sign_detectors.stop_sign_detectors import HaarDetector


if __name__ == "__main__":
    gvsp_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\live_stream_defaults_start.pcapng"
    # gvsp_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\faking_matlab_rec_3.pcapng"
    # gvsp_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\short_driving_in_parking-002.pcapng"
    
    
    gvsp_path = Path(gvsp_path)
    
    gvsp_transmission = MockGvspTransmission(gvsp_pcap_path=gvsp_path)
    
    constant_metadata_detector = ConstantMetadataDetector()
    frame_id_detector = FrameIDDetector()
    timestamp_detector = TimestampDetector(0.1 * 3)

    histogram_detector = HueSaturationHistogramDetector(0.4)
    mse_detector = MSEImageDetector(0.01)
    
    combined_detector = CombinedDetector([constant_metadata_detector, frame_id_detector, timestamp_detector], [mse_detector, histogram_detector])
    
    stop_sign_detector = HaarDetector()

    scores = []
    stop_sign_detection = []
    for frame in gvsp_transmission.frames:
        if frame is not None and frame.success_status:
            passed, detection_results, status = combined_detector.detect_experiments(frame)
            scores.append(detection_results)
            ic(passed)
            if not(passed):
                ic(status)
            # ic(combined_detector.detect_experiments(frame))

            detections = stop_sign_detector.detect(gvsp_frame_to_rgb(frame))
            stop_sign_detection.append(len(detections) > 0)
            
    scores_df = pd.DataFrame(scores)
    fig, axs = plt.subplots(6,1)
    axs[0].plot(stop_sign_detection)
    axs[0].set_ylabel('stop sign')
    ax_id = 1
    for name, res in scores_df.iteritems():
        ax = axs[ax_id]
        res.plot(ax=axs[ax_id])
        ax.grid(True)
        ax.set_ylabel(name)
        ax_id +=1
    plt.show()
    print(passed)