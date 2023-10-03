import cv2
from vimba import PixelFormat

#TODO: check if better in somewhere else

GVSP_LAYER = "Gvsp" 
GVSP_LEADER_LAYER = "GVSP_LEADER"
GVSP_TRAILER_LAYER = "GVSP_TRAILER"

INT_TO_PIXEL_FORMAT = {0x1080009: PixelFormat.BayerRG8}
CV2_CONVERSIONS = {PixelFormat.BayerRG8: cv2.COLOR_BayerRG2RGB}