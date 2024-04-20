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

REAL_LABEL = 0
FAKE_LABEL = 1

def get_all_frames_path(frames_dir: Path, min_frame_id: int) -> List[Path]:
    all_frames_path = []
    min_frame_id = 1800
    for p in frames_dir.glob('*.jpg'):
        id = int(p.stem.split('_')[1])
        if id < min_frame_id:
            pass
        else:
            all_frames_path.append(p)
    all_frames_path = sorted(all_frames_path, key=lambda x: int(x.stem.split("_")[1]))
    return all_frames_path

def to_batches(l: List, batch_size: int) -> List:
    return [l[offset:offset+batch_size] for offset in range(0,len(l),batch_size)]

def get_frame(frame_path: Path) -> np.ndarray:
    frame = cv2.imread(frame_path.as_posix())
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

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
            
def create_octagon_mask(image_size, center, side_length):
    mask = np.zeros(image_size, dtype=np.uint8)
    octagon_color = 255
    angle_offset = np.pi / 8
    vertices = []
    for i in range(8):
        angle = i * np.pi / 4 + angle_offset
        x = int(center[0] + side_length * np.cos(angle))
        y = int(center[1] + side_length * np.sin(angle))
        vertices.append((x, y))
    cv2.fillPoly(mask, [np.array(vertices)], (octagon_color,octagon_color,octagon_color))
    return mask

def place_stop_sign(img, destination_center, stop_sign, octagon_mask):
    dst_img = img.copy()

    height, width, _ = stop_sign.shape
    
    top_ind = destination_center[0] - height//2
    bottom_ind = destination_center[0] + height//2
    if height % 2 == 1:
        bottom_ind += 1

    left_ind = destination_center[1] - width//2
    right_ind = destination_center[1] + width//2
    if width % 2 == 1:
        right_ind += 1
    
    img_crop = img[top_ind:bottom_ind, left_ind:right_ind, :]
    dst_img[top_ind:bottom_ind, left_ind:right_ind, :] = np.where(~octagon_mask, img_crop, stop_sign)
    return dst_img

def score_detectors_sign_impainted(pair_path, detectors: List, stop_sign: np.ndarray, octagon_mask:np.ndarray):
    """score detector by reading frames directory. fake frame is the first frame with stop sign impainted on it"""
    frame_1_path, frame_2_path = pair_path
    frame_1 = get_frame(frame_1_path)
    frame_2 = get_frame(frame_2_path)
    
    destination_center = (350,1500)
    fake_frame = place_stop_sign(frame_1, destination_center, stop_sign, octagon_mask) 

    res = {}
    for detector in detectors:
        res[detector.name] = {'real': calc_score(detector=detector, frame_1=frame_1, frame_2=frame_2),
                            'fake': calc_score(detector=detector, frame_1=frame_1, frame_2=fake_frame),
        }
    return res 

def score_detectors_partial_image(pair_path, detectors: List, stop_sign_img: np.ndarray, num_lines: int):
    """score detector by reading frames directory. fake frame is the second frame with the top lines of the sign image"""
    frame_1_path, frame_2_path = pair_path
    frame_1 = get_frame(frame_1_path)
    frame_2 = get_frame(frame_2_path)
    
    fake_frame = frame_2.copy()
    fake_frame[:num_lines, :, :] = stop_sign_img[:num_lines, :, :]

    res = {}
    for detector in detectors:
        res[detector.name] = {'real': calc_score(detector=detector, frame_1=frame_1, frame_2=frame_2),
                            'fake': calc_score(detector=detector, frame_1=frame_1, frame_2=fake_frame),
        }
    return res 


if __name__ == '__main__':
    frames_dir = Path(r'OUTPUT/drive_around_uni_frames/')
    all_frames_path = get_all_frames_path(frames_dir, min_frame_id=1800)
    frames_pairs_path = to_batches(all_frames_path, batch_size=2)
    frames_pairs_path = frames_pairs_path[:10000]

    stop_sign = cv2.imread(r'INPUT/stop_sign.png')
    original_dim = stop_sign.shape[0]
    stop_sign = cv2.resize(stop_sign, (150,150))
    stop_sign = cv2.cvtColor(stop_sign, cv2.COLOR_BGR2RGB)
    resize_ratio = stop_sign.shape[0] / original_dim

    stop_sign_center = (stop_sign.shape[1] // 2, stop_sign.shape[0] // 2)
    side_length = 122.11674741819813 * resize_ratio
    octagon_mask = create_octagon_mask(stop_sign.shape, stop_sign_center, side_length)


    stop_sign_road = cv2.imread(r'INPUT/stop_sign_road_2.jpg')
    stop_sign_road = cv2.resize(stop_sign_road, (1936, 1216))
    stop_sign_road = cv2.cvtColor(stop_sign_road, cv2.COLOR_BGR2RGB)
    num_lines = 460
    

    optical_flow_detector = OpticalFlowDetector(0)
    hist_detector = HueSaturationHistogramDetector(0)

    # helper = partial(score_detectors_sign_impainted, detectors=[optical_flow_detector, hist_detector], stop_sign=stop_sign, octagon_mask=octagon_mask)
    helper = partial(score_detectors_partial_image, detectors=[optical_flow_detector, hist_detector], stop_sign_img=stop_sign_road, num_lines=num_lines)
    all_res = []
    with multiprocessing.Pool(6) as pool:
        for res in tqdm(pool.imap(helper, frames_pairs_path), total=len(frames_pairs_path)):
            all_res.append(res)

    histogram_res =  {'score': [], 'label': []}
    flow_res = {'score': [], 'label': []}

    for res in all_res:
        histogram_res['score'].append(res['Histogram']['real'])
        histogram_res['label'].append(REAL_LABEL)
        histogram_res['score'].append(res['Histogram']['fake'])
        histogram_res['label'].append(FAKE_LABEL)
        
        flow_res['score'].append(res['OpticalFlow']['real'])
        flow_res['label'].append(REAL_LABEL)
        flow_res['score'].append(res['OpticalFlow']['fake'])
        flow_res['label'].append(FAKE_LABEL)
        
    histogram_res = pd.DataFrame(histogram_res)
    flow_res = pd.DataFrame(flow_res)

    histogram_res.to_csv('OUTPUT/drive_around_uni_histogram_detector_scores_10000_samples_partial_sign.csv')
    flow_res.to_csv('OUTPUT/drive_around_uni_optical_flow_detector_scores_10000_samples_partial_sign.csv')