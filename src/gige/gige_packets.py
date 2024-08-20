from gige.constansts import BYTE
from gige.gige_constants import GvcpCommands, GvspFormat, Layers


from scapy.fields import (
    IntField,
    Packet,
    ShortField,
    XBitField,
    XByteEnumField,
    XShortEnumField,
)
from scapy.packet import Packet


class GvcpCmd(Packet):
    # TODO split to sub commands according to the command value
    name = Layers.GVCP.value
    fields_desc = [
        XBitField("MessageKeyCode", 0x42, BYTE),
        XBitField("Flags", 0x01, BYTE),
        XShortEnumField(
            "Command", None, {v.value: k for k, v in GvcpCommands._member_map_.items()}
        ),
        ShortField("PayloadLength", 0x0008),
        ShortField("RequestID", 1),
        XBitField("RegisterAddress", 0x000130F4, 4 * BYTE),
        IntField("value", None),
    ]


class GvspLeader(Packet):
    name = Layers.GVSP_LEADER.value
    fields_desc = [
        ShortField("FieldInfo", 0),
        ShortField("PayloadType", 0x0001),
        XBitField("Timestamp", 1, 8 * BYTE),
        XBitField("PixelFormat", 0x01080009, 4 * BYTE),
        IntField("SizeX", img_width),
        IntField("SizeY", img_height),
        IntField("OffsetX", 0),
        IntField("OffsetY", 0),
        ShortField("PaddingX", 0),
        ShortField("PaddingY", 0),
    ]


class GvspTrailer(Packet):
    name = Layers.GVSP_TRAILER.value
    fields_desc = [
        ShortField("FieldInfo", 0),
        ShortField("PayloadType", 0x0001),
        IntField("SizeY", img_height),
    ]


class Gvsp(Packet):
    name = Layers.GVSP.value
    fields_desc = [
        XBitField("Status", 0x0000, 2 * BYTE),
        ShortField("BlockID", 0),
        XByteEnumField(
            "Format", None, {v.value: k for k, v in GvspFormat._member_map_.items()}
        ),
        XBitField("PacketID", 0x000000, 3 * BYTE),
    ]
