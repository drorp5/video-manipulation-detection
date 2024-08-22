import copy
import multiprocessing
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import random


from passive_detectors_evaluation.manipulator import RectangularPatchInjector
from sign_detectors import StopSignDetector, get_detector, draw_detections
from gige.utils import payload_gvsp_bytes_to_raw_image
from gige.utils import bgr_img_to_packets_payload
from utils.image_processing import bggr_to_rggb
from utils.detection_utils import Rectangle, calculate_iou
from utils.injection import get_stripe_gvsp_payload_bytes
from utils.datasets import get_largest_bounding_box, resize_bounding_box
from driving_experiments.evaluation.experiments_summary import (
    get_normal_completed_pcap_frames_ids,
    extract_metadata_key_of_config_path,
)
from driving_experiments.evaluation.experiments_summary import parse_log_file


# MTSD dataset
dataset_directory = Path("../datasets/mtsd_v2_fully_annotated")
original_images_directory = dataset_directory / "images"
stop_sign_images_directory = dataset_directory / "MobileNet_detections"
annotations_directory = dataset_directory / "annotations"
target_object = "regulatory--stop--g1"
stop_sign_images_path = list(stop_sign_images_directory.glob("*"))


# Sign Detector
detector = get_detector("MobileNet")
detector.confidence_th = 0


# Output directory
injections_images_directory = Path(
    r"D:\Thesis\video-manipulation-detection\driving_experiments_injections\patch_roc"
)
if not injections_images_directory.exists():
    injections_images_directory.mkdir(parents=True)

# GVSP
max_payload_bytes = 8963

# experiments directory
experiments_directory = Path(
    r"D:\Thesis\video-manipulation-detection\driving_experiments"
)


def evaluate_frames_pair(frame_path_pair: Tuple[Path, Path]) -> dict:
    frame_1_path, frame_2_path = frame_path_pair
    # read frames
    frame_1_bgr = cv2.imread(frame_1_path.as_posix())
    frame_1_width = frame_1_bgr.shape[1]
    frame_2_bgr = cv2.imread(frame_2_path.as_posix())
    frame_2_width = frame_2_bgr.shape[1]

    # draw image of stop sign of dataset
    stop_sign_img_path = random.choice(stop_sign_images_path)
    img_key = stop_sign_img_path.stem
    stop_sign_img_bgr = cv2.imread(stop_sign_img_path.as_posix())

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
        new_shape=(stop_sign_img_bgr.shape[1], stop_sign_img_bgr.shape[0]),
    )

    # inject patch to first frame
    patch_injector = RectangularPatchInjector(
        patch_img=stop_sign_img_bgr,
        first_row=gt_bounding_box.ymin,
        last_row=gt_bounding_box.ymax,
        first_col=gt_bounding_box.xmin,
        last_col=gt_bounding_box.xmax,
    )
    frame_1_bgr_injected = patch_injector.inject(frame_1_bgr, frame_1_bgr)
    gt_bounding_box = Rectangle(
        upper_left_corner=(
            frame_1_bgr_injected.shape[1] - gt_bounding_box.num_columns,
            0,
        ),
        lower_right_corner=(frame_1_bgr_injected.shape[1], gt_bounding_box.num_rows),
    )

    # get sripe injection payload bytes
    target_row = 0
    stripe_packets_ids, gvsp_payload_stripe = get_stripe_gvsp_payload_bytes(
        img_bgr=frame_1_bgr_injected,
        bounding_box=gt_bounding_box,
        max_payload_bytes=max_payload_bytes,
        target_row=target_row,
    )
    rows_per_packet = max_payload_bytes / frame_2_bgr.shape[1]
    affected_rows = int(np.ceil(len(stripe_packets_ids) * rows_per_packet))
    gt_stripe = Rectangle(
        (0, target_row),
        (frame_2_bgr.shape[1], target_row + affected_rows),
    )

    # simulate injection
    transmitted_payload_packets = bgr_img_to_packets_payload(
        img=frame_2_bgr,
        max_payload_bytes=max_payload_bytes,
    )
    transmitted_packets_ids = list(range(1, len(transmitted_payload_packets) + 1))
    transmitted_payload_packets.extend(gvsp_payload_stripe)
    transmitted_packets_ids.extend(stripe_packets_ids)

    raw_image, assigned_pixels = payload_gvsp_bytes_to_raw_image(
        payload_packets=transmitted_payload_packets,
        shape=frame_2_bgr.shape[:2],
        packets_ids=transmitted_packets_ids,
    )
    raw_image = bggr_to_rggb(raw_image)
    assigned_pixels = bggr_to_rggb(assigned_pixels)
    injected_img = cv2.cvtColor(raw_image, cv2.COLOR_BayerRG2RGB)
    injected_img_bgr = cv2.cvtColor(injected_img, cv2.COLOR_RGB2BGR)

    # save image
    experiment_id = frame_2_path.parent.parent.stem.split("_")[-1]
    dst_name = f"experiment_{experiment_id}_{frame_2_path.stem}_injected_{img_key}_widths_{frame_1_width}_{frame_2_width}"
    dst_path_original = injections_images_directory / f"{dst_name}.jpg"
    cv2.imwrite(dst_path_original.as_posix(), injected_img_bgr)

    # detect in injected
    config_path = frame_2_path.parent.parent / f"config_{experiment_id}.yaml"
    num_widths, time_of_day, road_type, _ = extract_metadata_key_of_config_path(
        config_path
    )
    detections = detector.detect(injected_img_bgr)
    # filter only detection which intersect the stripe
    valid_detections = []
    valid_ious = []
    for detection in detections:
        pred_detection = Rectangle(
            detection.get_upper_left_corner(), detection.get_lower_right_corner()
        )
        gt_stripe_detection = Rectangle(
            (pred_detection.xmin, gt_stripe.ymin), (pred_detection.xmax, gt_stripe.ymax)
        )
        iou = calculate_iou(pred_detection, gt_stripe_detection)
        if iou > 0:
            valid_detections.append(detection)
            valid_ious.append(iou)

    max_confidence = 0
    selected_iou = 0
    if len(valid_detections) > 0:
        selected_detection = None
        for detection, iou in zip(valid_detections, valid_ious):
            if detection.confidence > max_confidence:
                max_confidence = detection.confidence
                selected_detection = detection
                selected_iou = iou

        valid_detections = [selected_detection]
        valid_ious = [selected_iou]

        img_with_detections = draw_detections(
            injected_img_bgr, valid_detections, with_confidence=True
        )
        dst_path = injections_images_directory / f"{dst_name}_detected.jpg"
        cv2.imwrite(dst_path.as_posix(), img_with_detections)
    else:
        max_confidence = 0
        selected_iou = 0

    return {
        "experiemnt": experiment_id,
        "num_widths": num_widths,
        "time_of_day": time_of_day,
        "road_type": road_type,
        "first_frame": frame_1_path.stem,
        "second_frame": frame_2_path.stem,
        "injected_img_key": img_key,
        "frame_1_width": frame_1_width,
        "frame_2_width": frame_2_width,
        "saved_path": dst_path_original,
        "detection_confidence": max_confidence,
        "detection_iou": selected_iou,
    }


if __name__ == "__main__":
    debug = False
    total_injection_counter = {}
    successful_injection_counter = {}

    # iterate on normal, not attacked frmaes
    block_id_diff_th = 2
    frames_path_pairs = []
    for experiment_dir in experiments_directory.glob("2024*"):
        experiment_id = experiment_dir.stem.split("_")[-1]
        log_path = experiment_dir / f"log_{experiment_id}.log"
        if not log_path.exists():
            continue
        frames_df = parse_log_file(file_path=log_path, during_attack=False)
        normal_frames_id = frames_df["frame_id"].to_list()

        completed_frames_dir = experiment_dir / "pcap_completed_frames"
        completed_frames_ids = []
        for frame_path in completed_frames_dir.glob("*"):
            frame_id = int(frame_path.stem.split("_")[1])
            block_id = int(frame_path.stem.split("_")[3])
            completed_frames_ids.append((frame_id, block_id))
        completed_frames_ids = sorted(completed_frames_ids)

        for ii in range(1, len(completed_frames_ids)):
            current_frame_id, current_block_id = completed_frames_ids[ii]
            prev_frame_id, prev_block_id = completed_frames_ids[ii - 1]

            if (
                current_frame_id not in normal_frames_id
                or prev_frame_id not in normal_frames_id
            ):
                continue

            if current_block_id - prev_block_id <= block_id_diff_th:
                prev_path = (
                    completed_frames_dir
                    / f"frame_{prev_frame_id}_BlockID_{prev_block_id}.png"
                )
                current_path = (
                    completed_frames_dir
                    / f"frame_{current_frame_id}_BlockID_{current_block_id}.png"
                )

                frames_path_pairs.append((prev_path, current_path))

    # for frames_path_pair in frames_path_pairs:
    #     res = evaluate_frames_pair(frames_path_pair)

    all_res = []
    with multiprocessing.Pool(6) as pool:
        for res in tqdm(
            pool.imap(evaluate_frames_pair, frames_path_pairs),
            total=len(frames_path_pairs),
        ):
            all_res.append(res)

    results_df = pd.DataFrame(all_res)
    csv_path = injections_images_directory / "results.csv"
    results_df.to_csv(csv_path.as_posix())
