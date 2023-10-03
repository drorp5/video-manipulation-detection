from enum import IntEnum

class GigERegisters(IntEnum):
    ACQUISITION = 0x000130f4
    CCP = 0x0a00
    EXPOSURE_VALUE  = 0x00014110

class GvcpCommands(IntEnum):
    READREG_CMD = 0x0080
    READREG_ACK = 0x0081
    WRITEREG_CMD = 0x0082
    WRITEREG_ACK = 0x0083    