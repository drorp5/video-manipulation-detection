from enum import IntEnum

class GigERegisters(IntEnum):
    ACQUISITION = 0x000130f4
    CCP = 0x0a00
    EXPOSURE_VALUE  = 0x00014110