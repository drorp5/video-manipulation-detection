"""
This module defines the packet structures used in GigE Vision communication.

It includes definitions for both Control Protocol (GVCP) and Stream Protocol (GVSP) packets.
These packet definitions are based on the GigE Vision standard and are used for 
communication between a control point (usually a computer) and a GigE Vision camera.

Classes:
    GvcpCmd: Represents a GVCP command packet.
    GvspLeader: Represents the leader packet of a GVSP stream.
    GvspTrailer: Represents the trailer packet of a GVSP stream.
    Gvsp: Represents the base GVSP packet structure.

The module also sets up the appropriate layer bindings for Scapy to correctly 
parse and build these packets in network traffic.

Note: This module requires the gige_constants module for various enumeration values.
"""

from gige.gige_constants import (
    GvcpCommands,
    GvspFormat,
    Layers,
    GigERegisters,
    MAX_HEIGHT,
    MAX_WIDTH,
    Ports,
    BYTE,
)

from scapy.fields import (
    IntField,
    ShortField,
    XBitField,
    XByteEnumField,
    XShortEnumField,
    X3BytesField,
    XIntField,
    LongField,
)
from scapy.packet import Packet, bind_layers
from scapy.layers.inet import UDP


class GvcpCmd(Packet):
    """
    GigE Vision Control Protocol (GVCP) Command Packet.

    This packet represents a GVCP command sent from the control point to the device.
    It is used for various control operations such as reading and writing registers.

    Attributes:
        name (str): The name of the packet layer.
        fields_desc (list): List of fields that describe the packet structure.
    """

    name = Layers.GVCP.value
    fields_desc = [
        XBitField("MessageKeyCode", 0x42, BYTE),
        XBitField("Flags", 0x01, BYTE),
        XShortEnumField(
            "Command", None, {v.value: k for k, v in GvcpCommands._member_map_.items()}
        ),
        ShortField("PayloadLength", 0x0008),
        ShortField("RequestID", 1),
        XBitField("RegisterAddress", GigERegisters.ACQUISITION.value, 4 * BYTE),
        IntField("value", None),
    ]


class GvspLeader(Packet):
    """
    GigE Vision Stream Protocol (GVSP) Leader Packet.

    This packet is the first packet of a new image or part in the GVSP stream.
    It contains metadata about the following image data.

    Attributes:
        name (str): The name of the packet layer.
        fields_desc (list): List of fields that describe the packet structure.
    """

    name = Layers.GVSP_LEADER.value
    fields_desc = [
        ShortField("FieldInfo", 0),
        ShortField("PayloadType", 0x0001),
        LongField("Timestamp", 1),
        XIntField("PixelFormat", 0x01080009),
        IntField("SizeX", MAX_WIDTH),
        IntField("SizeY", MAX_HEIGHT),
        IntField("OffsetX", 0),
        IntField("OffsetY", 0),
        ShortField("PaddingX", 0),
        ShortField("PaddingY", 0),
    ]


class GvspTrailer(Packet):
    """
    GigE Vision Stream Protocol (GVSP) Trailer Packet.

    This packet is the last packet of an image or part in the GVSP stream.
    It may contain additional metadata about the transmitted image.

    Attributes:
        name (str): The name of the packet layer.
        fields_desc (list): List of fields that describe the packet structure.
    """

    name = Layers.GVSP_TRAILER.value
    fields_desc = [
        ShortField("FieldInfo", 0),
        ShortField("PayloadType", 0x0001),
        IntField("SizeY", MAX_HEIGHT),
    ]


class Gvsp(Packet):
    """
    GigE Vision Stream Protocol (GVSP) Packet.

    This is the base packet for all GVSP communication. It includes common fields
    used in all GVSP packets, such as packet type, block ID, and format.

    Attributes:
        name (str): The name of the packet layer.
        fields_desc (list): List of fields that describe the packet structure.
    """

    name = Layers.GVSP.value
    fields_desc = [
        XBitField("Status", 0x0000, 2 * BYTE),
        ShortField("BlockID", 0),
        XByteEnumField(
            "Format", None, {v.value: k for k, v in GvspFormat._member_map_.items()}
        ),
        X3BytesField("PacketID", 0x000000),
    ]


# Bind layers
bind_layers(UDP, GvcpCmd, dport=Ports.GVCP_DST)
bind_layers(UDP, Gvsp, sport=Ports.GVSP_SRC)
bind_layers(Gvsp, GvspLeader, Format=GvspFormat.LEADER)
bind_layers(Gvsp, GvspTrailer, Format=GvspFormat.TRAILER)
