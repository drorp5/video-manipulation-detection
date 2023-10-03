#TODO: check if better in somewhere else
import cv2
from vimba import PixelFormat
from enum import Enum
class Layers(Enum):
    IP = "IP"
    UDP = "UDP"
    GVSP = "GVSP" 
    GVSP_LEADER = "GVSP_LEADER"
    GVSP_TRAILER = "GVSP_TRAILER"
    GVCP = "GVCP_CMD"


INT_TO_PIXEL_FORMAT = {0x1080009: PixelFormat.BayerRG8}
CV2_CONVERSIONS = {PixelFormat.BayerRG8: cv2.COLOR_BayerRG2RGB}