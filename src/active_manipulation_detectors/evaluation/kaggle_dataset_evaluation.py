from pathlib import Path
from tqdm import tqdm
from icecream import ic
from pprint import pprint
from typing import List, Dict
import cv2

from active_manipulation_detectors.evaluation.labels_parsing_utils import (
    parse_voc_xml,
)
from detectors_evaluation.bootstrapper import DST_SHAPE
from sign_detectors import get_detector, draw_bounding_boxes


def get_target_annotations(annotations_directory: Path, target_object) -> List[Dict]:
    annotations = [
        parse_voc_xml(xml_path) for xml_path in annotations_directory.glob("*.xml")
    ]

    # filter only annotation of the target object
    target_annotations_files = list(
        filter(
            lambda annotation: any(
                [
                    labelled_object["name"] == target_object
                    for labelled_object in annotation["objects"]
                ]
            ),
            annotations,
        )
    )

    target_annotations = []
    for annotation in target_annotations_files:
        new_annotation = annotation.copy()
        new_annotation["objects"] = []
        for labelled_object in annotation["objects"]:
            if labelled_object["name"] == target_object:
                new_annotation["objects"].append(labelled_object)

        target_annotations.append(new_annotation)
    return target_annotations


dataset_directory = Path()
images_directory = dataset_directory / "images"
annotations_directory = dataset_directory / "annotations"

target_annotations = get_target_annotations(annotations_directory, target_object="stop")

# iterate on images in dataset and check sign detector
sign_detector = get_detector("MobileNet")

# without resampling
for annotation in tqdm(target_annotations):
    img_path = images_directory / annotation["filename"]
    img_bgr = cv2.imread(img_path.as_posix())
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, DST_SHAPE)
    detections = sign_detector.detect(img_rgb)
    img_with_detections = draw_bounding_boxes(img_rgb, detections)
    dst_path = (
        dataset_directory
        / "../road_sign_detection_kaggle_MobileNet_Detections_resized"
        / f'detected_{annotation["filename"]}'
    )
    cv2.imwrite(
        dst_path.as_posix(), cv2.cvtColor(img_with_detections, cv2.COLOR_RGB2BGR)
    )
