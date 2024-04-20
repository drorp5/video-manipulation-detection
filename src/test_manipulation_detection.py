import sys
sys.path.append('src')
from gige.gvsp_transmission import GvspPcapParser
from gige.gvsp_frame import gvsp_frame_to_rgb
from manipulation_detectors.metadata import ConstantMetadataDetector, FrameIDDetector, TimestampDetector, TimestampRateDetector
from manipulation_detectors.image_processing import OpticalFlowDetector, MSEImageDetector, HueSaturationHistogramDetector
from manipulation_detectors.combined import CombinedDetector
from sign_detectors.stop_sign_detectors import StopSignDetector, HaarDetector, MobileNetDetector, draw_bounding_boxes, get_detector
from pathlib import Path
from typing import Tuple
import pandas as pd
from icecream import ic
import traceback
import cv2


def detect_in_gvsp_transmission(gvsp_transmission: GvspPcapParser,
                                combined_detector: CombinedDetector,
                                vehicle_detector: StopSignDetector,
                                print_every: int = 1,
                                output_video_path: Path = None) -> pd.DataFrame:
    
    scores = []
    process_time = []
    fake = []
    frames = []
    num_frames = 0
    video_writer = None

    try:
        for frame in gvsp_transmission.frames:
            if frame is not None and frame.success_status:
                vehicle_detections = vehicle_detector.detect(gvsp_frame_to_rgb(frame))
                detection_scores = {'vehicle' :  int(len(vehicle_detections) > 0)}
                
                manipulation_detection_results = combined_detector.detect_experiments(frame)
                failed_status = [res.message for res in manipulation_detection_results.values() if not res.passed]
                passed = len(failed_status) == 0
                manipulation_detection_scores = zip(manipulation_detection_results.keys(), [res.score for res in manipulation_detection_results.values()])
                detection_scores.update(manipulation_detection_scores)
                
                fake_results = {'combined': not passed}
                detectors_fake_flag = zip(manipulation_detection_results.keys(), [not res.passed for res in manipulation_detection_results.values()])
                fake_results.update(detectors_fake_flag)
                
                frames.append(frame.id)
                scores.append(detection_scores)
                fake.append(fake_results)
                num_frames += 1
                
                process_time.append(dict(zip(manipulation_detection_results.keys(), [res.process_time_sec for res in manipulation_detection_results.values()])))
                
                if num_frames % print_every == 0:
                    print(f'frame {frame.id} : {passed}')
                    if not(passed):
                        ic(failed_status)

                if output_video_path is not None:
                    frame_img = gvsp_frame_to_rgb(frame)
    
                    if video_writer is None:
                        height, width, _ = frame_img.shape
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        fps = 20
                        video_writer = cv2.VideoWriter(output_video_path.as_posix(), fourcc, fps, (width, height))
                    if len(vehicle_detections) > 0:
                        frame_img = draw_bounding_boxes(img=frame_img, bounding_boxes=vehicle_detections)
                    try:
                        optical_flow_img = combined_detector.image_processing_detectors[-1].draw_optical_flow_tracks()
                        frame_img = cv2.add(frame_img, optical_flow_img)
                    except:
                        pass
                    video_writer.write(cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR))        
    except Exception as e:
        traceback.print_exc() 
    finally:
        if video_writer is not None:
            video_writer.release()
        scores_df = pd.DataFrame(scores, index=frames)
        process_time_df = pd.DataFrame(process_time, index=frames)
        fake_df =  pd.DataFrame(fake, index=frames)
        results_df = pd.concat([scores_df, fake_df.add_prefix('fake_'), process_time_df.add_prefix('time_')], axis=1)        
    return results_df

if __name__ == "__main__":
    dst_dir = Path('OUTPUT')
    
    # gvsp_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\live_stream_defaults_start.pcapng"
    gvsp_path = r"D:\Thesis\video-manipulation-detection\INPUT\old_recordings\faking_matlab_rec_3.pcapng"
    # gvsp_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\short_driving_in_parking-002.pcapng"
    # gvsp_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\driving_in_uni_2.pcapng"
    # gvsp_path = r"C:\Users\drorp\Desktop\University\Thesis\video-manipulation-detection\INPUT\driving_in_uni_1-001.pcapng"
    # gvsp_path = r"D:\Thesis\video-manipulation-detection\INPUT\old_recordings\drive_around_uni.pcapng"
    
    gvsp_path = Path(gvsp_path)
    
    gvsp_transmission = GvspPcapParser(pcap_path=gvsp_path)
    
    constant_metadata_detector = ConstantMetadataDetector()
    frame_id_detector = FrameIDDetector()
    timestamp_detector = TimestampDetector(max_th=0.1 * 3)
    timestamp_rate_detector = TimestampRateDetector(max_th=0.1)

    histogram_detector = HueSaturationHistogramDetector(max_th=0.4)
    mse_detector = MSEImageDetector(min_th=0.01, max_th=1000)
    optical_flow_detector = OpticalFlowDetector(max_th=20)
    combined_detector = CombinedDetector([constant_metadata_detector, frame_id_detector, timestamp_detector, timestamp_rate_detector],
     [mse_detector, histogram_detector, optical_flow_detector])
    # combined_detector = CombinedDetector([], [optical_flow_detector])
    
    # vehicle_detector = get_detector('Haar')
    vehicle_detector = get_detector('MobileNet')
    # vehicle_detector = get_detector('Yolo')

    # output_video_path = dst_dir/f'{gvsp_path.stem}_{vehicle_detector.name}.mp4'
    output_video_path = None
    results_df = detect_in_gvsp_transmission(gvsp_transmission=gvsp_transmission,
                                            combined_detector=combined_detector,
                                            vehicle_detector=vehicle_detector,
                                            print_every=5,
                                            output_video_path=output_video_path)
    results_df.to_pickle(dst_dir/f'temp_{gvsp_path.stem}_{vehicle_detector.name}.pkl')
