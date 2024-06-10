from pathlib import Path
from re import M
from typing import Dict, List, Tuple
from tqdm import tqdm
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import random


from detectors_evaluation.bootstrapper import DST_SHAPE
from sign_detectors import StopSignDetector, get_detector, draw_bounding_boxes
from gige.utils import payload_gvsp_bytes_to_raw_image
from utils.image_processing import bggr_to_rggb
from utils.detection_utils import Rectangle, calculate_iou
from utils.injection import get_stripe_gvsp_payload_bytes
from gige.utils import bgr_img_to_packets_payload
from detectors_evaluation.manipulator import RectangularPatchInjector


dataset_directory = Path("../datasets/mtsd_v2_fully_annotated")
original_images_directory = dataset_directory / "images"
stop_sign_images_directory = dataset_directory / "MobileNet_detections"
annotations_directory = dataset_directory / "annotations"

detector = get_detector("MobileNet")

injections_images_directory = dataset_directory / "MobileNet_detections_injections"
if not injections_images_directory.exists():
    injections_images_directory.mkdir(parents=True)


def draw_background_image(background_images_path: List[Path]) -> np.ndarray:
    random_background_iamge_path = random.choice(background_images_path)
    background_img_bgr = cv2.imread(random_background_iamge_path.as_posix())
    background_img_bgr = cv2.resize(background_img_bgr, DST_SHAPE)
    return background_img_bgr


def get_largest_bounding_box(annotation: Dict, target_object: str) -> Rectangle:
    min_area = 0
    largest_gt_bounding_box_annotation = None
    for object in annotation["objects"]:
        if object["label"] == target_object:
            gt_bounding_box_annotation = object["bbox"]
            area = (
                gt_bounding_box_annotation["xmax"] - gt_bounding_box_annotation["xmin"]
            ) * (
                gt_bounding_box_annotation["ymax"] - gt_bounding_box_annotation["ymin"]
            )
            if area > min_area:
                min_area = area
                largest_gt_bounding_box_annotation = gt_bounding_box_annotation
    return Rectangle(
        (
            largest_gt_bounding_box_annotation["xmin"],
            largest_gt_bounding_box_annotation["ymin"],
        ),
        (
            largest_gt_bounding_box_annotation["xmax"],
            largest_gt_bounding_box_annotation["ymax"],
        ),
    )


def resize_bounding_box(
    bounding_box: Rectangle, old_shape: Tuple[int, int], new_shape: Tuple[int, int]
) -> None:
    width_resizing_factor = new_shape[0] / old_shape[0]
    height_resizing_factor = new_shape[1] / old_shape[1]
    bounding_box.resize(width_resizing_factor, height_resizing_factor)
    bounding_box.enforce_super_pixel_size(2)  # Bayer


if __name__ == "__main__":
    num_symbols = 8
    max_payload_bytes = 8963
    target_object = "regulatory--stop--g1"
    black_background = False
    debug = False

    # path for possible background image
    all_images_path = list(original_images_directory.glob("*jpg"))
    stop_sign_keys = [p.stem for p in list(stop_sign_images_directory.glob("*.jpg"))]
    background_images_path = list(
        filter(lambda p: p.stem not in stop_sign_keys, all_images_path)
    )

    # iterate on images
    max_width = DST_SHAPE[0]
    widths = [max_width - 2 * i for i in range(0, num_symbols)]
    total_images_tested = 0
    successful_injection_counter = {width: 0 for width in widths}

    for img_path in tqdm(list(stop_sign_images_directory.glob("*jpg"))):
        # draw background image
        background_img_bgr = draw_background_image(background_images_path)

        # read image
        img_key = img_path.stem
        img_bgr = cv2.imread(img_path.as_posix())

        # read annotation
        annotation_path = annotations_directory / f"{img_key}.json"
        with open(annotation_path.as_posix(), "r") as fid:
            annotation = json.load(fid)
        annotation["image_key"] = annotation_path.stem

        # set largest bounding box
        gt_bounding_box = get_largest_bounding_box(
            annotation=annotation, target_object=target_object
        )

        # resize bounding box
        resize_bounding_box(
            gt_bounding_box,
            old_shape=(annotation["width"], annotation["height"]),
            new_shape=(img_bgr.shape[1], img_bgr.shape[0]),
        )
        gt_stripe = Rectangle(
            (0, gt_bounding_box.ymin), (img_bgr.shape[1], gt_bounding_box.ymax)
        )

        injector = RectangularPatchInjector(
            patch_img=img_bgr,
            first_col=gt_bounding_box.xmin,
            last_col=gt_bounding_box.xmax,
            first_row=gt_bounding_box.ymin,
            last_row=gt_bounding_box.ymax,
        )
        background_with_injection = injector.inject(
            background_img_bgr, background_img_bgr
        )
        gt_stripe = Rectangle(
            (0, 0), (img_bgr.shape[1], gt_bounding_box.ymax - gt_bounding_box.ymin)
        )

        # get sripe injection payload bytes
        stripe_packets_ids, gvsp_payload_stripe = get_stripe_gvsp_payload_bytes(
            img_bgr=background_with_injection,
            bounding_box=gt_stripe,
            max_payload_bytes=max_payload_bytes,
            target_row=gt_stripe.ymin,
        )

        # combine to image in desired width
        for width in widths:
            # simulate injection
            if black_background:
                transmitted_payload_packets = gvsp_payload_stripe
                transmitted_packets_ids = stripe_packets_ids
            else:
                transmitted_payload_packets = bgr_img_to_packets_payload(
                    img=background_img_bgr[:, :width, :],
                    max_payload_bytes=max_payload_bytes,
                )
                transmitted_packets_ids = list(
                    range(1, len(transmitted_payload_packets) + 1)
                )
                transmitted_payload_packets.extend(gvsp_payload_stripe)
                transmitted_packets_ids.extend(stripe_packets_ids)

            raw_image, assigned_pixels = payload_gvsp_bytes_to_raw_image(
                payload_packets=transmitted_payload_packets,
                shape=(DST_SHAPE[1], width),
                packets_ids=transmitted_packets_ids,
            )
            raw_image = bggr_to_rggb(raw_image)
            assigned_pixels = bggr_to_rggb(assigned_pixels)
            injected_img = cv2.cvtColor(raw_image, cv2.COLOR_BayerRG2RGB)

            # detect in injected
            detections = detector.detect(injected_img)

            # filter only detection which intersect the stripe
            valid_detections = []
            for detection in detections:
                x, y, w, h = detection
                pred_detection = Rectangle((x, y), (x + w, y + h))
                iou = calculate_iou(pred_detection, gt_stripe)
                if iou > 0:
                    valid_detections.append(detection)

            if debug:
                img_with_detections = draw_bounding_boxes(
                    injected_img, valid_detections
                )
                plt.figure()
                plt.imshow(img_with_detections)
                plt.title("Injected")
                plt.show(block=False)

                original_detections = detector.detect(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                )
                original_img_with_detections = draw_bounding_boxes(
                    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), original_detections
                )

                plt.figure()
                plt.imshow(original_img_with_detections)
                plt.title("Original")
                plt.show(block=False)

            if width == max_width:
                if len(valid_detections) == 0:
                    break
                total_images_tested += 1

            # save image
            dst_path = injections_images_directory / f"{img_key}_width_{int(width)}.jpg"
            cv2.imwrite(
                dst_path.as_posix(), cv2.cvtColor(injected_img, cv2.COLOR_RGB2BGR)
            )

            if len(valid_detections) > 0:
                img_with_detections = draw_bounding_boxes(
                    injected_img, valid_detections
                )
                dst_path = (
                    injections_images_directory
                    / f"{img_key}_width_{int(width)}_detected.jpg"
                )
                cv2.imwrite(
                    dst_path.as_posix(),
                    cv2.cvtColor(img_with_detections, cv2.COLOR_RGB2BGR),
                )

                if width != max_width:
                    print(dst_path)
                    successful_injection_counter[width] = (
                        successful_injection_counter[width] + 1
                    )

    print(f"Total images = {total_images_tested}")
    print("successfull injections per width:")
    print(successful_injection_counter)
