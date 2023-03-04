from pathlib import Path
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
from sign_detectors.stop_sign_detectors import StopSignDetector, HaarDetector


def detect_in_gvsp_transmission(gvsp_transmission: MockGvspTransmission,
                                combined_detector: CombinedDetector,
                                stop_sign_detector: StopSignDetector) -> pd.DataFrame:
    scores = []
    for frame in gvsp_transmission.frames:
        if frame is not None and frame.success_status:
            stop_sign_detections = stop_sign_detector.detect(gvsp_frame_to_rgb(frame))
            detection_scores = {'stop_sign' :  int(len(stop_sign_detections) > 0)}

            manipulation_detection_results = combined_detector.detect_experiments(frame)
            failed_status = [res.message for res in manipulation_detection_results.values() if not res.passed]
            passed = len(failed_status) == 0
            ic(passed)
            if not(passed):
                ic(failed_status)
            manipulation_detection_scores = zip(manipulation_detection_results.keys(), [res.score for res in manipulation_detection_results.values()])
            detection_scores.update(manipulation_detection_scores)
            scores.append(detection_scores)
            
    scores_df = pd.DataFrame(scores)
    return scores_df

def plot_results(results_df: pd.DataFrame):
    fig, axs = plt.subplots(6,1)
    ax_id = 0
    for name, res in results_df.iteritems():
        ax = axs[ax_id]
        res.plot(ax=axs[ax_id])
        ax.grid(True)
        ax.set_ylabel(name)
        ax_id +=1
    plt.show()

if __name__ == "__main__":
    dst_dir = Path('OUTPUT')
    
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

    results_df = detect_in_gvsp_transmission(gvsp_transmission=gvsp_transmission, combined_detector=combined_detector, stop_sign_detector=stop_sign_detector)
    results_df.to_pickle(dst_dir/f'{gvsp_path.stem}.pkl')
    plot_results(results_df)


