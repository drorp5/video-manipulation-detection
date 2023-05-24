"""BSD 2-Clause License

Copyright (c) 2019, Allied Vision Technologies GmbH
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from pathlib import Path
import threading
import sys
import cv2
from typing import Dict, Optional
from vimba import *
from sign_detectors import stop_sign_detectors as detectors 
import time
import argparse
from config import CV2_CONVERSIONS
import json
from datetime import datetime
import subprocess

def print_preamble():
    print('///////////////////////////////////////////////////////')
    print('/// Vimba API Asynchronous Grab with OpenCV Example ///')
    print('///////////////////////////////////////////////////////\n')


def print_usage():
    print('Usage:')
    print('    python asynchronous_grab_opencv.py [camera_id]')
    print('    python asynchronous_grab_opencv.py [/h] [-h]')
    print()
    print('Parameters:')
    print('    camera_id   ID of the camera to use (using first camera if not specified)')
    print()


def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')

    if usage:
        print_usage()

    sys.exit(return_code)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--camera_id", help="camera ID for direct access")
    parser.add_argument("-d", "--detector", choices=detectors.get_detectors_dict(), help="detection method")

    # parser.add_argument("-h", "--help")
    
    args = parser.parse_args()
    """
    for arg in args:
        if arg in ('/h', '-h'):
            print_usage()
            sys.exit(0)

    if argc > 1:
        abort(reason="Invalid number of arguments. Abort.", return_code=2, usage=True)

    return None if argc == 0 else args[0]
    """
    return args

def get_camera(camera_id: Optional[str]) -> Camera:
    with Vimba.get_instance() as vimba:
        if camera_id:
            try:
                return vimba.get_camera_by_id(camera_id)

            except VimbaCameraError:
                abort('Failed to access Camera \'{}\'. Abort.'.format(camera_id))

        else:
            cams = vimba.get_all_cameras()
            if not cams:
                abort('No Cameras accessible. Abort.')

            return cams[0]


def setup_camera(cam: Camera):
    with cam:
        # Enable auto exposure time setting if camera supports it
        try:
            cam.ExposureAuto.set('Continuous')

        except (AttributeError, VimbaFeatureError):
            pass

        # Enable white balancing if camera supports it
        try:
            cam.BalanceWhiteAuto.set('Continuous')

        except (AttributeError, VimbaFeatureError):
            pass

        # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
        try:
            cam.GVSPAdjustPacketSize.run()

            while not cam.GVSPAdjustPacketSize.is_done():
                pass

        except (AttributeError, VimbaFeatureError):
            pass

        # Query available, open_cv compatible pixel formats
        # prefer color formats over monochrome formats
        cv_fmts = intersect_pixel_formats(cam.get_pixel_formats(), OPENCV_PIXEL_FORMATS)
        color_fmts = intersect_pixel_formats(cv_fmts, COLOR_PIXEL_FORMATS)

        if color_fmts:
            seleted_frame_format = color_fmts[0]
            cam.set_pixel_format(seleted_frame_format)            
        else:
            mono_fmts = intersect_pixel_formats(cv_fmts, MONO_PIXEL_FORMATS)
            if mono_fmts:
                seleted_frame_format = mono_fmts[0]
                cam.set_pixel_format(seleted_frame_format)

            else:
                abort('Camera does not support a OpenCV compatible format natively. Abort.')


class Handler:
    def __init__(self, detector_name: str, output_parameters_path:Path):
        self.shutdown_event = threading.Event()
        self.detector = detectors.get_detector(detector_name)
        self.downfactor = 4
        self.output_parameters_path = output_parameters_path


    def __call__(self, cam: Camera, frame: Frame):
        ENTER_KEY_CODE = 13

        key = cv2.waitKey(1)
        if key == ENTER_KEY_CODE:
            self.shutdown_event.set()
            return

        elif frame.get_status() == FrameStatus.Complete:
            print('{} acquired {}'.format(cam, frame), flush=True)

            msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
            img = frame.as_opencv_image()

            # get values of exposure and gain
            try:
                exposure_us = cam.get_feature_by_name('ExposureTimeAbs').get()
                gain_db = cam.get_feature_by_name('Gain').get()
                with open(self.output_parameters_path.absolute().as_posix(), 'a') as file:
                    file.write(f"""{{
        frame_id: {frame.get_id()},
        exposure_us: {exposure_us},
        gain_db: {gain_db}
        }},
        """)
            except:
                print('WARNING: cant query parameters')
                 
                
            conversion_started = time.time()
            pixel_format = frame.get_pixel_format()
            if pixel_format in CV2_CONVERSIONS.keys():
                img = cv2.cvtColor(img, CV2_CONVERSIONS[pixel_format])
            conversion_finished = time.time()
            conversion_time = conversion_finished - conversion_started

            resizing_started = time.time()
            
            width = int(img.shape[1] / self.downfactor)
            height = int(img.shape[0] / self.downfactor)
            dim = (width, height)
            processed_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)                
            resizing_finished = time.time()
            resizing_time = resizing_finished - resizing_started
            
            detection_started = time.time()
            if self.detector is not None:
                detections = self.detector.detect(processed_img)
                processed_img = detectors.draw_bounding_boxes(processed_img, detections)
            detection_finished = time.time()
            detection_time = detection_finished - detection_started

            window_name = msg.format(cam.get_name())           
            cv2.imshow(window_name, processed_img)
            
            print(f'Conversion time = {conversion_time}')
            print(f'resizing time = {resizing_time}')
            print(f'detection time = {detection_time}')
            print(f'total processing time = {conversion_time + resizing_time + detection_time}')
        cam.queue_frame(frame)


def main():
    print_preamble()
    args = parse_args()
    cam_id = args.camera_id
    detector = args.detector

    # Get the current time
    current_time = datetime.now()
    # Format the current time as a string
    time_string = current_time.strftime("%H_%M_%S")

    output_parameters_path = Path(rf'./OUTPUT/adaptive_parameters_{time_string}.json')
    with open(output_parameters_path.absolute().as_posix(), 'w') as file:
        file.write(f'{{\n\trecording time = {time_string},\n')
        

    # Command to start tshark with pcap writer and filter for GVSP or GVCP packets
    pcap_file = Path(rf'./OUTPUT/recording_{time_string}.pcap')
    tshark_command = ["tshark", "-i", "Ethernet 6", "-w", pcap_file.absolute().as_posix(), "((src host 192.168.10.150) and (dst host 192.168.1.100)) or ((dst host 192.168.10.150) and (src host 192.168.1.100))"]
    
    # Start the subprocess
    process = subprocess.Popen(tshark_command)

    with Vimba.get_instance():
        with get_camera(cam_id) as cam:

            # Start Streaming, wait for five seconds, stop streaming
            setup_camera(cam)
            handler = Handler(detector, output_parameters_path)

            try:
                # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                cam.start_streaming(handler=handler, buffer_count=10)
                handler.shutdown_event.wait()

            finally:
                cam.stop_streaming()
    
    with open(output_parameters_path.absolute().as_posix(), 'a') as file:
        file.write(f'\n}}')
    
    process.terminate()



if __name__ == '__main__':
    main()
