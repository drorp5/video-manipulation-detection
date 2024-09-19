"""
Configuration file for the GigE Vision Attack Tool.

This file contains the necessary parameters and configurations for different
attack scenarios and setups.
"""

from enum import Enum


class Setup(Enum):
    WINDOWS_VIMBA = 1
    WINDOWS_MATLAB_REC = 2
    WINDOWS_MATLAB_PREVIEW = 3
    LINUX_ROS = 4


# Default setup
DEFAULT_SETUP = Setup.WINDOWS_VIMBA

# MAC addresses
CAMERA_MAC = "00:0f:31:03:67:c4"
CP_MAC = "00:18:7d:c8:e6:31"

# Configuration for different setups
SETUP_CONFIG = {
    Setup.WINDOWS_VIMBA: {
        "interface": "Ethernet 6",
        "cp_ip": "192.168.1.100",
        "camera_ip": "192.168.10.150",
        "img_width": 1936,
        "img_height": 1216,
        "max_payload_bytes": 8963,
        "sendp_ampiric_frame_time": 0.13,
    },
    Setup.WINDOWS_MATLAB_REC: {
        "interface": "Ethernet 6",
        "cp_ip": "192.168.1.100",
        "camera_ip": "192.168.0.1",
        "img_width": 1920,
        "img_height": 1080,
        "max_payload_bytes": 8950,
        "sendp_ampiric_frame_time": 0.13,
    },
    Setup.WINDOWS_MATLAB_PREVIEW: {
        "interface": "Ethernet 6",
        "cp_ip": "192.168.1.100",
        "camera_ip": "192.168.0.1",
        "img_width": 1936,
        "img_height": 1216,
        "max_payload_bytes": 8950,
        "sendp_ampiric_frame_time": 0.13,
    },
    Setup.LINUX_ROS: {
        "interface": "enp1s0",
        "cp_ip": "192.168.1.100",
        "camera_ip": "192.168.0.1",
        "img_width": 1936,
        "img_height": 1216,
        "max_payload_bytes": 8950,
        "sendp_ampiric_frame_time": 0.13,
    },
}


def get_config(setup: Setup = DEFAULT_SETUP) -> dict:
    """
    Get the configuration for the specified setup.

    Args:
        setup (Setup): The setup to get the configuration for.

    Returns:
        dict: A dictionary containing the configuration parameters.
    """
    config = {"cp_mac": CP_MAC, "camera_mac": CAMERA_MAC}
    config.update(SETUP_CONFIG[setup])
    return config
