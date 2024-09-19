# Video Manipulation Detection

## Project Overview

This project implements a comprehensive video manipulation detection system for GigE Vision cameras, focusing on automotive and security applications. It includes tools for simulating attacks, implementing defenses, and detecting manipulations using both passive and active methods.

## Key Components

1. **Attack Simulation**:
   - Simulates various types of attacks on GigE Vision streams.
   - Supports frame injection, stripe injection, and other manipulation techniques.

2. **Defense Mechanisms**:
   - Implements defensive strategies against GigE Vision attacks.
   - Includes both preventive measures and real-time defense techniques.

3. **Passive Detection**:
   - Various detectors for identifying manipulations in image streams without modifying the transmission.
   - Includes metadata-based and image processing-based detection methods.

4. **Active Detection**:
   - Implements active detection mechanisms, including the width varying detector.
   - Provides analysis tools for evaluating active detection performance.
   - Implements experiments for active detection of manipulations in GigE Vision streams.
   - Includes car simulation, attacker simulation, and recording capabilities.

5. **Sign Detectors**:
   - Implements stop sign detection using various methods (Haar Cascade, YOLO, MobileNet).

6. **Evaluation Framework**:
   - Tools for evaluating the performance of both passive and active detectors.
   - Supports different datasets and injector types.

7. **GigE Vision Utilities**:
   - Parsers and handlers for GigE Vision protocol.
   - Includes GVSP and GVCP packet handling.

## Installation

## Usage

### Simulating Attacks
While runninng an active GigE Vision stream use
```console
    python attack_tool/attack_tool_main.py -m full_frame_injection -p /path/to/image.jpg -d 10 --fps 30 --setup WINDOWS_VIMBA
```

for a full frame injection attack and

```console
    python attack_tool/attack_tool_main.py -m stripe_injection -p /path/to/image.jpg -d 10 --fps 30 --setup WINDOWS_VIMBA
```

for stripe injection attack (the numbr of injected rows is an adjustable parameter in the script).

The network parameters are configured in the src\attack_tool\gige_attack_config.py file.

Another option via a CLI:

```python
from attack_tool import GigEVisionAttackTool

attack_tool = GigEVisionAttackTool(interface="eth0", cp_ip="192.168.1.1", camera_ip="192.168.1.2", cp_mac="00:00:00:00:00:00", camera_mac="00:00:00:00:00:01", max_payload_bytes=8963,img_width=1936, img_height=1216)

# Frame injection
attack_tool.fake_still_image("fake_image.png", duration=5, fps=30)

# Single Stripe injection
attack_tool.inject_stripe("fake_image.png", first_row=100, num_rows=50, future_id_diff=1, count=1)

# Consecutive Stripe injection
attack_tool.inject_stripe("fake_image.png", first_row=100, num_rows=50, fps=30, injection_duration=5, future_id_diff=1, count=1)
```


### Running Passive Detection

```python
from passive_detectors import CombinedDetector
from passive_detectors.image_processing import HistogramDetector, OpticalFlowDetector

detector = CombinedDetector([HistogramDetector(), OpticalFlowDetector()])
result = detector.detect(frame1, frame2)
```

### Running Active Detection Experiments

You can run active detection experiments either through the command line interface (CLI) or using the graphical user interface (GUI).

#### Using the CLI:

```python
from active_detection_experiments import run_experiment

config_path = "path/to/config.yaml"
run_experiment_using_config_path(config_path)
```

#### Using the GUI:

To launch the GUI for configuring and running experiments 

```console
python active_detection_experiments/gui.py
```

The GUI provides an intuitive interface for setting up experiment parameters, running experiments, and viewing results.

### Evaluating Detection Methods

```python
from passive_detectors_evaluation import run_evaluation

evaluator = Evaluator([HistogramDetector(), OpticalFlowDetector()], StripInjector())
dataset = VideoDataset("path/to/video.mp4")
run_evaluation(evaluator, dataset, Path("results/"))
```

## Project Structure

```
video-manipulation-detection/
├── INPUT/
├── OUTPUT/
├── src/
│   ├── active_manipulation_detectors/
│   ├── active_detection_experiments/
│   ├── attack_tool/
│   ├── gige/
│   ├── injectors/
│   ├── passive_detectors/
│   ├── passive_detectors_evaluation/
│   ├── sign_detectors/
│   └── utils/
└── README.md
```

- `INPUT/`: Directory for input data, such as test videos or configuration files
- `OUTPUT/`: Directory for output data, including results and logs
- `src/`: Source code for all project components
  - `active_manipulation_detectors/`: Implementation of active detection mechanisms (e.g., width varying detector) and analysis tools
  - `active_detection_experiments/`: Scripts and modules for running and managing active detection experiments
  - `attack_tool/`: Tools for simulating various attacks on GigE Vision streams
  - `gige/`: Utilities for handling GigE Vision protocol, including GVSP and GVCP packet processing
  - `injectors/`: Implementation of different injection methods for simulating attacks
  - `passive_detectors/`: Collection of passive detection algorithms for identifying manipulations
  - `passive_detectors_evaluation/`: Tools and scripts for evaluating the performance of passive detectors
  - `sign_detectors/`: Implementation of different stop sign detection methods (Haar, YOLO, MobileNet)
  - `utils/`: Common utility functions and helper modules used across the project

## Contributing

## License

This project uses the Vimba toolbox, which is licensed under the BSD 2-Clause License:

### BSD 2-Clause License

Copyright (c) 2019, Allied Vision Technologies GmbH
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Acknowledgments

- The Haar detector for stop sign detection using OpenCV is based on the work from the Stop-Sign-detection_OpenCV repository by maurehur: https://github.com/maurehur/Stop-Sign-detection_OpenCV

- The MobileNet detector implementation is based on the object_detection_COCO repository by zafarRehan: https://github.com/zafarRehan/object_detection_COCO

- The YOLO detector implementation is inspired by the tutorial "Object Detection with Yolo Python and OpenCV- Yolo 2 | CloudxLab Blog": https://cloudxlab.com/blog/object-detection-yolo-and-python-pydarknet/