from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Callable, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from manipulation_detectors.image_processing.abstract_image_processing_detector import ImageProcessingDetector
from detectors_evaluation.manipulator import Injector

class Label(Enum):
    REAL = 0
    FAKE = 1
    
@dataclass
class EvaluationResult:
    detector: str
    real: float
    fake: float

    def to_dict(self) -> Dict[str: Dict[str, float]]:
        return {self.detector: {Label.REAL.name: self.real, Label.FAKE.name: self.fake}}
        
        
class Evaluator():
    def __init__(self, detectors: List[ImageProcessingDetector], injector: Injector):
        self.detectors = detectors
        self.injector = injector
        
    def evaluate(self, frame_1: np.ndarray, frame_2:np.ndarray) -> List[EvaluationResult]:
        fake_frame = self.injector.inject(frame_1, frame_2)
        results = []
        for detector in self.detectors:
            real_score = detector.calc_score(frame_1, frame_2)
            fake_score = detector.calc_score(frame_1, fake_frame)
            results.append(EvaluationResult(detector=detector.name, real=real_score, fake=fake_score))
        return results
    
def evaluate_pair(evaluator:Evaluator, x:Tuple[np.ndarray, np.ndarray]) -> List[EvaluationResult]:
    return evaluator.evaluate(x[0], x[1])
