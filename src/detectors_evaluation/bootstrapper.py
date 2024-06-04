import cv2
from typing import List
from pathlib import Path
from detectors_evaluation.manipulator import (
    FullFrameInjector,
    StripeInjector,
    SignPatchInjector,
    RectangularPatchInjector,
    Injector,
)
from detectors_evaluation.datasets import (
    EvaluationDataset,
    FramesDirectoryDataset,
    VideoDataset,
    VideosDirectoryDataset,
)

from manipulation_detectors.image_processing import (
    OpticalFlowDetector,
    HueSaturationHistogramDetector,
    ImageProcessingDetector,
)
from gige import MAX_HEIGHT, MAX_WIDTH

DST_SHAPE = (MAX_WIDTH, MAX_HEIGHT)


def get_stop_sign_injector(injector_type: str) -> Injector:
    if injector_type == "full_frame":
        stop_sign_road = cv2.imread(r"INPUT/stop_sign_road_2.jpg")
        stop_sign_road = cv2.cvtColor(stop_sign_road, cv2.COLOR_BGR2RGB)
        return FullFrameInjector(fake_img=stop_sign_road, dst_shape=DST_SHAPE)
    elif injector_type == "stripe":
        stop_sign_road = cv2.imread(r"INPUT/stop_sign_road_2.jpg")
        stop_sign_road = cv2.resize(stop_sign_road, DST_SHAPE)
        stop_sign_road = cv2.cvtColor(stop_sign_road, cv2.COLOR_BGR2RGB)
        return StripeInjector(fake_img=stop_sign_road, first_row=354, last_row=489)
    elif injector_type == "patch":
        stop_sign = cv2.imread(r"INPUT/stop_sign_road_2_resized_cropped.jpg")
        sign_img = cv2.cvtColor(stop_sign, cv2.COLOR_BGR2RGB)
        side_length = 108
        first_row = 4
        last_row = 138
        return SignPatchInjector(
            sign_img=sign_img,
            side_length=side_length,
            first_row=first_row,
            last_row=last_row,
        )
    else:
        raise ValueError


def get_red_light_injector(injector_type: str) -> Injector:
    red_light_path = r"INPUT/red_light.jpg"
    red_light_img = cv2.imread(red_light_path)
    red_light_img = cv2.cvtColor(red_light_img, cv2.COLOR_BGR2RGB)

    if injector_type == "full_frame":
        return FullFrameInjector(fake_img=red_light_img, dst_shape=DST_SHAPE)
    elif injector_type == "stripe":
        return StripeInjector(fake_img=red_light_img, first_row=133, last_row=197)
    elif injector_type == "patch":
        return RectangularPatchInjector(
            patch_img=red_light_img,
            first_row=133,
            last_row=197,
            first_col=810,
            last_col=875,
        )
    else:
        raise ValueError


def get_image_processing_detectors() -> List[ImageProcessingDetector]:
    hist_detector = HueSaturationHistogramDetector(0)
    optical_flow_detector = OpticalFlowDetector(0)
    return [hist_detector, optical_flow_detector]


def get_dataset(data_source: str) -> EvaluationDataset:
    if data_source == "experiment":
        frames_dir = Path(r"OUTPUT/drive_around_uni_frames/")
        dataset_name = "drive_around_uni"
        min_frame_id = 1800
        return FramesDirectoryDataset(
            frames_dir=frames_dir, name=dataset_name, min_frame_id=min_frame_id
        )
    elif data_source == "BDD":
        videos_path = Path(
            r"D:\Thesis\video-manipulation-detection\Datasets\BDD100K\bdd100k_videos_test_00\bdd100k\videos\test"
        )
        num_frames = 100
        flip = True
        return VideosDirectoryDataset(
            videos_dir=videos_path,
            name="BDD",
            num_frames=num_frames,
            flip=flip,
            dst_shape=DST_SHAPE,
        )
    else:
        raise ValueError
