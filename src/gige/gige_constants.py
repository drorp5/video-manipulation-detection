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


class Ports(IntEnum):
    GVSP_SRC = 10010
    GVCP_DST = 3956


class GvspFormat(IntEnum):
    LEADER = 0x01
    TRAILER = 0x02
    PAYLOAD = 0x03



BYTES_PER_PIXEL = 1    