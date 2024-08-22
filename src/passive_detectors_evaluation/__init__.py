from passive_detectors_evaluation.manipulator import (
    Injector,
    FullFrameInjector,
    StripeInjector,
    SignPatchInjector,
    RectangularPatchInjector,
)
from passive_detectors_evaluation.evaluator import (
    Evaluator,
    EvaluationResult,
    Label,
    evaluate_pair,
)
from passive_detectors_evaluation.datasets import (
    EvaluationDataset,
    FramesDirectoryDataset,
    VideoDataset,
    VideosDirectoryDataset,
)
