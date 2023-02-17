from vimba import *
import cv2

CV2_CONVERSIONS = {PixelFormat.BayerRG8: cv2.COLOR_BayerRG2RGB}
MAX_PIXEL_VAL = 255
MAX_SATURATION = 255
MAX_HUE = 179