from typing import List, Tuple, Optional
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import json
import random
import numpy as np
import cv2
from IPython.display import display, Image
from pprint import pprint
import multiprocessing
from dataclasses import dataclass


from passive_detectors_evaluation.bootstrapper import DST_SHAPE
from sign_detectors import get_detector, draw_detections
from utils.datasets import get_largest_bounding_box, resize_bounding_box
from utils.detection_utils import Rectangle, calculate_iou, DetectedObject


dataset_directory = Path("../datasets/mtsd_v2_fully_annotated")
images_directory = dataset_directory / "images"
annotations_directory = dataset_directory / "annotations"
stop_sign_object = "regulatory--stop--g1"


detector = get_detector("MobileNet")
detector.confidence_th = 0
detector.nms_th = 0.01


iou_th = 0.3


def load_annotation_from_path(annotation_path: Path) -> dict:
    with open(annotation_path.as_posix(), "r") as fid:
        anno = json.load(fid)
    anno["image_key"] = annotation_path.stem
    return anno


def get_all_annotations(max_files: Optional[int] = None) -> List[dict]:
    all_files = list(annotations_directory.glob("*.json"))
    if max_files is not None:
        random.shuffle(all_files)
        all_files = all_files[:max_files]

    return [load_annotation_from_path(json_path) for json_path in tqdm(all_files)]


def split_annotations(
    annotations: List[dict], target_object: str
) -> Tuple[List[dict], List[dict]]:
    # filter only annotation of the target object
    target_annotations_files = []
    non_target_annotations = []
    for annotation in tqdm(annotations):
        if not (images_directory / f"{annotation['image_key']}.jpg").exists():
            continue
        if any(
            [
                labelled_object["label"] == target_object
                for labelled_object in annotation["objects"]
            ]
        ):
            target_annotations_files.append(annotation)
        else:
            non_target_annotations.append(annotation)

    # set only taret objects if few in same image
    target_annotations = []
    for annotation in tqdm(target_annotations_files):
        new_annotation = annotation.copy()
        new_annotation["objects"] = []
        for labelled_object in annotation["objects"]:
            if labelled_object["label"] == target_object:
                new_annotation["objects"].append(labelled_object)

        target_annotations.append(new_annotation)
    return target_annotations, non_target_annotations


@dataclass
class ImageShape:
    width: int
    height: int


class DatasetImage:
    def __init__(self, img_key: str) -> None:
        # read
        img_path = images_directory / f"{img_key}.jpg"
        img_bgr = cv2.imread(img_path.as_posix())
        height, width, _ = img_bgr.shape
        self._original_shape = ImageShape(width=width, height=height)
        # resize
        self.img = cv2.resize(img_bgr, DST_SHAPE)

    @property
    def shape(self) -> ImageShape:
        return ImageShape(width=self.img.shape[1], height=self.img.shape[0])

    @property
    def original_shape(self) -> ImageShape:
        return self._original_shape

    @property
    def resize_width_factor(self) -> float:
        return self.shape.width / self.original_shape.width

    @property
    def resize_height_factor(self) -> float:
        return self.shape.height / self.original_shape.height


def process_positive_annotation(annotation: dict) -> float:
    image_key = annotation["image_key"]
    img = DatasetImage(image_key)
    gt_bounding_box = get_largest_bounding_box(
        annotation=annotation, target_object=stop_sign_object
    )
    gt_bounding_box.resize(
        height_factor=img.resize_height_factor, width_factor=img.resize_width_factor
    )
    detections = detector.detect(img.img)
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)

    if len(detections) > 0:
        pred_detection = detections[0]
        pred_bounding_box = Rectangle(
            detections[0].get_upper_left_corner(),
            detections[0].get_lower_right_corner(),
        )
        iou = calculate_iou(gt_bounding_box, pred_bounding_box)
        if iou > iou_th:
            pred_score = pred_detection.confidence
        else:
            pred_score = 0
    else:
        pred_score = 0

    return pred_score


def process_negative_annotation(annotation: dict) -> float:
    image_key = annotation["image_key"]
    img = DatasetImage(image_key)
    detections = detector.detect(img.img)
    detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
    if len(detections) > 0:
        pred_detection = detections[0]
        pred_score = pred_detection.confidence
    else:
        pred_score = 0

    return pred_score


if __name__ == "__main__":
    annotations = get_all_annotations(max_files=None)
    target_annotations, other_annotations = split_annotations(
        annotations, target_object=stop_sign_object
    )

    num_non_target = 5000
    sampled_other_annotations = random.sample(other_annotations, num_non_target)
    scores = []
    labels = []

    with multiprocessing.Pool(4) as pool:
        for res in tqdm(
            pool.imap(process_positive_annotation, target_annotations),
            total=len(target_annotations),
        ):
            scores.append(res)
            labels.append(True)

    with multiprocessing.Pool(4) as pool:
        for res in tqdm(
            pool.imap(process_negative_annotation, sampled_other_annotations),
            total=len(sampled_other_annotations),
        ):
            scores.append(res)
            labels.append(False)

    df = pd.DataFrame({"scores": scores, "labels": labels})
    df.to_csv("../OUTPUT/sign_detector_calibration_overlapped.csv", index=False)
