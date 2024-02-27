import sys
sys.path.append('./src')
from pathlib import Path
import cv2
from tqdm import tqdm
import multiprocessing
from functools import partial
import pandas as pd
from manipulation_detectors.image_processing import OpticalFlowDetector, MSEImageDetector, HueSaturationHistogramDetector
from detectors_evaluation import FullFrameInjector, StripeInjector, SignPatchInjector, Evaluator, EvaluationResult, EvaluationDataset, FramesDirectoryDataset, Label, evaluate_pair, VideoDataset, VideosDirectoryDataset

def run(evaluator: Evaluator, dataset: EvaluationDataset, dst_dir_path: Path):
    res_by_detector = {detector.name: {'score': [], 'label': []} for detector in evaluator.detectors}

    helper = partial(evaluate_pair, evaluator)
    with multiprocessing.Pool(6) as pool:
        for res in tqdm(pool.imap(helper, dataset), total=len(dataset)):
            for detector_res in res:
                res_by_detector[detector_res.detector]['label'].append(Label.REAL.value)
                res_by_detector[detector_res.detector]['score'].append(detector_res.real)
                res_by_detector[detector_res.detector]['label'].append(Label.FAKE.value)
                res_by_detector[detector_res.detector]['score'].append(detector_res.fake)
    
        # all_res = []
        # for (frame_1, frame_2) in tqdm(dataset):
        #     res = evaluator.evaluate(frame_1, frame_2)
        #     all_res.append(res)
        
        # save results
        for detector in evaluator.detectors:    
            df = pd.DataFrame(res_by_detector[detector.name])
            dst_path = dst_dir_path / f'{dataset.name}_{evaluator.injector.name}_{detector.name}.csv'
            df.to_csv(dst_path)

if __name__ == '__main__':
    base_dir = Path('OUTPUT')
    data_source = 'BDD'
    only_save_example = False
    injector_type = 'sign_patch'
    dst_shape = (1936, 1216)
    
    # set detectors
    hist_detector = HueSaturationHistogramDetector(0)
    optical_flow_detector = OpticalFlowDetector(0)
    detectors = [hist_detector, optical_flow_detector]
    
    for data_source in ['experiment', 'BDD']:
        for injector_type in ['full_frame', 'stripe', 'sign_patch']:
            print(f'running {injector_type} on {data_source}')            
            # set dataset
            if data_source == 'experiment':
                frames_dir = Path(r'OUTPUT/drive_around_uni_frames/')
                dataset_name = 'drive_around_uni'
                min_frame_id = 1800
                # min_frame_id = 19500
                dataset = FramesDirectoryDataset(frames_dir=frames_dir, name=dataset_name, min_frame_id=min_frame_id)
            elif data_source == 'BDD':
                videos_path=Path(r'D:\Thesis\video-manipulation-detection\Datasets\BDD100K\bdd100k_videos_test_00\bdd100k\videos\test')
                num_frames=100
                flip=True
                dst_shape=(1936, 1216)
                dataset = VideosDirectoryDataset(videos_dir=videos_path, name='BDD', num_frames=num_frames, flip=flip, dst_shape=dst_shape)
            else:
                raise NotImplementedError
                
            # set manipulation injectors
            if injector_type == 'full_frame':
                stop_sign_road = cv2.imread(r'INPUT/stop_sign_road_2.jpg')
                stop_sign_road = cv2.cvtColor(stop_sign_road, cv2.COLOR_BGR2RGB)
                injector = FullFrameInjector(fake_img=stop_sign_road, dst_shape=dst_shape)
            elif injector_type == 'stripe':
                stop_sign_road = cv2.imread(r'INPUT/stop_sign_road_2.jpg')
                stop_sign_road = cv2.resize(stop_sign_road, dst_shape)
                stop_sign_road = cv2.cvtColor(stop_sign_road, cv2.COLOR_BGR2RGB)
                injector = StripeInjector(fake_img=stop_sign_road, first_row=354, last_row=489)
            elif injector_type == 'sign_patch':
                stop_sign = cv2.imread(r'INPUT/stop_sign_road_2_resized_cropped.jpg')
                sign_img = cv2.cvtColor(stop_sign, cv2.COLOR_BGR2RGB)
                side_length = 108
                first_row = 4
                last_row = 138
                injector = SignPatchInjector(sign_img=sign_img, side_length=side_length, first_row=first_row, last_row=last_row)
            else:
                raise NotImplementedError
            
            # run
            if only_save_example:
                # save example
                dst_path = Path(f'./OUTPUT/fake_{data_source}_{injector.name}.jpg')
                fake_frame = injector.inject(frame_1=dataset[0][0], frame_2=dataset[0][1])
                fake_frame_bgr = cv2.cvtColor(fake_frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(dst_path.as_posix(), fake_frame_bgr)
                print(f'saved in {dst_path}')
            else:
                evaluator = Evaluator(detectors=detectors, injector=injector)
                run(evaluator=evaluator, dataset=dataset, dst_dir_path=base_dir)