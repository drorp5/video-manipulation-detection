from enum import Enum, IntEnum


class GigERegisters(IntEnum):
    ACQUISITION = 0x000130F4
    CCP = 0x0A00
    EXPOSURE_VALUE = 0x00014110
    WIDTH = 0x12124
    HEIGHT = 0x12128
    SCPC = 0xD04  # packet size


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
MAX_HEIGHT = 1216
MAX_WIDTH = 1936


class Layers(Enum):
    IP = "IP"
    UDP = "UDP"
    GVSP = "GVSP"
    GVSP_LEADER = "GVSP_LEADER"
    GVSP_TRAILER = "GVSP_TRAILER"
    GVCP = "GVCP_CMD"
