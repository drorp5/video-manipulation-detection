from pathlib import Path
from typing import List, Tuple
import random
import numpy as np
import cv2
import pandas as pd
import multiprocessing
from tqdm import tqdm
from functools import partial
from manipulation_detectors.image_processing import OpticalFlowDetector, MSEImageDetector, HueSaturationHistogramDetector
from sign_detectors.stop_sign_detectors import draw_bounding_boxes, get_detector

REAL_LABEL = 0
FAKE_LABEL = 1

def get_frame_of_video(cap, index: int) -> np.ndarray:
    """get single frame of video

    Args:
        cap (_type_): video capture (video opened with opencv)
        index (int): frame index

    Returns:
        np.ndarray: frame
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame1 = cap.read()
    return frame

def get_frames_couple_of_video(cap, index: int) -> Tuple[np.ndarray,np.ndarray]:
    """get two consecutive frames of video

    Args:
        cap (_type_): video capture (video opened with opencv)
        index (int): index of first frames

    Returns:
        two frames
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    _, frame_1 = cap.read()
    frame_1 = cv2.flip(frame_1, 0)
    frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2RGB)
    _, frame_2 = cap.read()
    frame_2 = cv2.flip(frame_2, 0)
    frame_2 = cv2.cvtColor(frame_2, cv2.COLOR_BGR2RGB)
    return frame_1, frame_2

def calc_score(detector, frame_1, frame_2):
    """calc anomaly detector score""" 
    detector.pre_process(rgb_img=frame_1)
    detector.post_process()
    detector.pre_process(rgb_img=frame_2)
    res = detector.validate()
    return res.score

def score_detectors(detectors: List, pair_path, fake_frame):
    """run detection twice with and without fake frame and return the results"""
    frame_1_path, frame_2_path = pair_path
    frame_1 = get_frame(frame_1_path)
    frame_2 = get_frame(frame_2_path)

    res = {}
    for detector in detectors:
        res[detector.name] = {'real': calc_score(detector=detector, frame_1=frame_1, frame_2=frame_2),
                            'fake': calc_score(detector=detector, frame_1=frame_1, frame_2=fake_frame),
        }
    return res
            
def score_detectors_partial_image(frames: Tuple[np.ndarray, np.ndarray], detectors: List, stop_sign_img: np.ndarray, num_lines: int):
    """score detector by reading frames directory. fake frame is the second frame with the top lines of the sign image"""
    frame_1, frame_2 = frames    
    fake_frame = frame_2.copy()
    fake_frame[:num_lines, :, :] = stop_sign_img[:num_lines, :, :]

    res = {}
    for detector in detectors:
        res[detector.name] = {'real': calc_score(detector=detector, frame_1=frame_1, frame_2=frame_2),
                            'fake': calc_score(detector=detector, frame_1=frame_1, frame_2=fake_frame),
        }
    return res 

def get_frames_of_video(video_path:Path, num_frames:int): #-> Tuple[List[int], List[Tuple[np.ndarray, np.ndarray]]
    """get frames of video in equal steps"

    Args:
        video_path (Path): path to video streams
        num_frames (int): number of frames pairs to get of the stream.

    Returns: frames indices and two sequential frames of video in eqaul steps 
    """
    cap = cv2.VideoCapture(video_path.as_posix())
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_offset = np.linspace(0, total_frames-3, num_frames).astype(int)
    frames = [get_frames_couple_of_video(cap, frame_offset) for frame_offset in frames_offset]
    cap.release()
    return frames_offset, frames

def score_detectors_using_video(video_path:Path, num_frames:int, scoring_function, detectors, fake_img, num_lines):
    try:
        frames_offset, frames = get_frames_of_video(video_path, num_frames)
        video_res = []
        for frame_offset, frames_pair in zip(frames_offset, frames): 
            res = scoring_function(frames_pair, detectors, fake_img, num_lines)
            for detector in detectors:
                res[detector.name].update({'frame_id': frame_offset, 'video': video_path.stem})
            video_res.append(res)
        return video_res
    except Exception:
        print(f'video {video_path.name} failed')
        return None

def fake_frame(video_path:Path, fake_img:np.ndarray, num_lines:int):
    _, frames = get_frames_of_video(video_path, 1)
    fake_frame = frames[0][1]
    fake_frame[:num_lines, :, :] = fake_img[:num_lines, :, :]
    return fake_frame
    
def fake_frame_optical_flow(video_path:Path, fake_img:np.ndarray, num_lines:int):
    _, frames = get_frames_of_video(video_path, 1)
    fake_frame = frames[0][1]
    # fake_frame[:num_lines, :, :] = fake_img[:num_lines, :, :]
    optical_flow_detector = OpticalFlowDetector(0)
    optical_flow_detector.pre_process(rgb_img=frames[0][0])
    optical_flow_detector.post_process()
    optical_flow_detector.pre_process(rgb_img=fake_frame)
    res = optical_flow_detector.validate()
    optical_flow_img = optical_flow_detector.draw_optical_flow_tracks()
    fake_frame = cv2.add(fake_frame, optical_flow_img)
    return fake_frame


def get_all_frames_path(frames_dir: Path, min_frame_id: int = 1800) -> List[Path]:
    all_frames_path = []
    for p in frames_dir.glob('*.jpg'):
        id = int(p.stem.split('_')[1])
        if id < min_frame_id:
            pass
        else:
            all_frames_path.append(p)
    return all_frames_path

def fake_frame_optical_flow2(frames_path, fake_img:np.ndarray, num_lines:int, dst_dir: Path):
    frame_path1, frame_path2 = frames_path
    frame1 = cv2.imread(frame_path1.as_posix())
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2 = cv2.imread(frame_path2.as_posix())
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    fake_frame = frame2
    fake_frame[:num_lines, :, :] = fake_img[:num_lines, :, :]
    optical_flow_detector = OpticalFlowDetector(0)
    optical_flow_detector.pre_process(rgb_img=frame1)
    optical_flow_detector.post_process()
    optical_flow_detector.pre_process(rgb_img=fake_frame)
    res = optical_flow_detector.validate()
    optical_flow_img = optical_flow_detector.draw_optical_flow_tracks()
    # fake_frame = cv2.add(fake_frame, optical_flow_img)
    
    vehicle_detector = get_detector('MobileNet')
    vehicle_detections = vehicle_detector.detect(fake_frame)
    fake_frame = cv2.cvtColor(fake_frame, cv2.COLOR_RGB2BGR)
    if len(vehicle_detections) > 0:
        fake_frame = draw_bounding_boxes(img=fake_frame, bounding_boxes=vehicle_detections)
    dst_path = dst_dir / f'{frame_path1.stem}.png'
    cv2.imwrite(dst_path.as_posix(), fake_frame)
        
    return len(vehicle_detections) 

if __name__ == '__main__':
    videos_dir = Path(r'D:\Thesis\video-manipulation-detection\Datasets\BDD100K\bdd100k_videos_test_00\bdd100k\videos\test')
    videos_path_list = list(videos_dir.glob('*'))
    stop_sign_road = cv2.imread(r'INPUT/stop_sign_road_2.jpg')
    stop_sign_road = cv2.cvtColor(stop_sign_road, cv2.COLOR_BGR2RGB)
    # stop_sign_road = cv2.resize(stop_sign_road, (1280, 720))
    # num_lines = 286
    stop_sign_road = cv2.resize(stop_sign_road, (1936, 1216))
    

    num_lines = 460
    
    # vehicle_detector = get_detector('MobileNet')
    dst_dir = Path('OUTPUT/drive_around_uni_frames_fake_partial')
    dst_dir.mkdir(exist_ok=True)
    
    all_frames_path = get_all_frames_path(Path('OUTPUT/drive_around_uni_frames'), min_frame_id=8300)
    all_frames_path = sorted(all_frames_path, key=lambda x: int(x.stem.split("_")[1]))
    frames_pairs_path = [tuple(all_frames_path[offset:offset+2]) for offset in range(0,len(all_frames_path),2)]
    frames_pairs_path = frames_pairs_path[:10]

    
    helper = partial(fake_frame_optical_flow2, fake_img=stop_sign_road, num_lines=num_lines, dst_dir=dst_dir)
    total_detections = 0
    with multiprocessing.Pool(6) as pool:
        for res in tqdm(pool.imap(helper, frames_pairs_path), total=len(frames_pairs_path)):
            total_detections += res

    print(total_detections)
    
    # optical_flow_detector = OpticalFlowDetector(0)
    # hist_detector = HueSaturationHistogramDetector(0)

    # num_frames = 100
    # # res = score_detectors_using_video(videos_path_list[0], num_frames, score_detectors_partial_image, [optical_flow_detector, hist_detector], stop_sign_road, num_lines)
    
    
    # helper = partial(score_detectors_using_video, num_frames=num_frames, scoring_function=score_detectors_partial_image, detectors=[optical_flow_detector, hist_detector], fake_img=stop_sign_road, num_lines=num_lines)
    # all_res = []
    # with multiprocessing.Pool(6) as pool:
    #     for res in tqdm(pool.imap(helper, videos_path_list), total=len(videos_path_list)):
    #         if res is not None:
    #             all_res.extend(res)

    # histogram_res =  {'score': [], 'label': [], 'video': [], 'frame': []}
    # flow_res = {'score': [], 'label': [], 'video': [], 'frame': []}

    # for res in all_res:
    #     histogram_res['score'].append(res['Histogram']['real'])
    #     histogram_res['label'].append(REAL_LABEL)
    #     histogram_res['score'].append(res['Histogram']['fake'])
    #     histogram_res['label'].append(FAKE_LABEL)
    #     histogram_res['video'].append(res['Histogram']['video'])
    #     histogram_res['video'].append(res['Histogram']['video'])
    #     histogram_res['frame'].append(res['Histogram']['frame_id'])
    #     histogram_res['frame'].append(res['Histogram']['frame_id'])
        
    #     flow_res['score'].append(res['OpticalFlow']['real'])
    #     flow_res['label'].append(REAL_LABEL)
    #     flow_res['score'].append(res['OpticalFlow']['fake'])
    #     flow_res['label'].append(FAKE_LABEL)
    #     flow_res['video'].append(res['OpticalFlow']['video'])
    #     flow_res['video'].append(res['OpticalFlow']['video'])
    #     flow_res['frame'].append(res['OpticalFlow']['frame_id'])
    #     flow_res['frame'].append(res['OpticalFlow']['frame_id'])
        
        
    # histogram_res = pd.DataFrame(histogram_res)
    # flow_res = pd.DataFrame(flow_res)

    # histogram_res.to_csv('OUTPUT/bdd100k_test_videos_100_each_histogram_detector_partial_sign.csv')
    # flow_res.to_csv('OUTPUT/bdd100k_test_videos_100_each_flow_detector_scores_10000_samples_partial_sign.csv')