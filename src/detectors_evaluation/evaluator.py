from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from passive_detectors.image_processing.abstract_image_processing_detector import (
    ImageProcessingDetector,
)
from detectors_evaluation.manipulator import Injector


class Label(Enum):
    REAL = 0
    FAKE = 1


@dataclass
class EvaluationResult:
    """
    Data class to store the evaluation results for a detector.
    """

    detector: str
    real: float
    fake: float

    def to_dict(self) -> Dict[str : Dict[str, float]]:
        return {self.detector: {Label.REAL.name: self.real, Label.FAKE.name: self.fake}}


class Evaluator:
    """
    Class for evaluating passive image manipulation detection.
    """

    def __init__(self, detectors: List[ImageProcessingDetector], injector: Injector):
        """
        Initialize the Evaluator.

        Args:
            detectors (List[ImageProcessingDetector]): List of detectors to use for evaluation.
            injector (Injector): The injector used to create fake frames.
        """
        self.detectors = detectors
        self.injector = injector

    def evaluate(
        self, frame_1: np.ndarray, frame_2: np.ndarray
    ) -> List[EvaluationResult]:
        """
        Evaluate the detection performance on a pair of frames.

        Args:
            frame_1 (np.ndarray): The first frame.
            frame_2 (np.ndarray): The second frame.

        Returns:
            List[EvaluationResult]: A list of evaluation results for each detector.
        """
        fake_frame = self.injector.inject(frame_1, frame_2)
        results = []
        for detector in self.detectors:
            real_score = detector.calc_score(frame_1, frame_2)
            fake_score = detector.calc_score(frame_1, fake_frame)
            results.append(
                EvaluationResult(
                    detector=detector.name, real=real_score, fake=fake_score
                )
            )
        return results


def evaluate_pair(
    evaluator: Evaluator, x: Tuple[np.ndarray, np.ndarray]
) -> List[EvaluationResult]:
    """
    Evaluate a pair of frames using the given evaluator.

    Args:
        evaluator (Evaluator): The evaluator to use.
        x (Tuple[np.ndarray, np.ndarray]): A tuple containing two frames.

    Returns:
        List[EvaluationResult]: A list of evaluation results.
    """
    return evaluator.evaluate(x[0], x[1])
