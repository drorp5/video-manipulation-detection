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
import re 

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
    parser.add_argument("-p", "--pcap", type=bool, help="whether to save pcap dump", default=False)
    parser.add_argument("-ad", "--adaptive", type=bool, help="whether to save adaptive parameters", default=False)
    parser.add_argument("-save", "--save_frames", type=bool, help="whether to save collected frames", default=False)

    # parser.add_argument("-h", "--help")
    
    args = parser.parse_args()
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
    def __init__(self, detector_name: str, output_parameters_path:Path=None, svaed_frames_dir:Path=None):
        self.shutdown_event = threading.Event()
        self.detector = detectors.get_detector(detector_name)
        self.downfactor = 4
        self.output_parameters_path = output_parameters_path
        self.svaed_frames_dir = svaed_frames_dir


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
            if self.output_parameters_path:
                try:
                    frame_data = {}
                    frame_data["exposure_us"] = cam.get_feature_by_name('ExposureTimeAbs').get()
                    frame_data["gain_db"] = cam.get_feature_by_name('Gain').get()
                    json_data = {f'frame_{frame.get_id()}': frame_data}
                    with open(self.output_parameters_path.absolute().as_posix(), 'a+') as file:
                        file.write(',\n')
                        file.write(json.dumps(json_data, indent=2))
                except:
                    print('WARNING: cant query parameters')
                    
                
            conversion_started = time.time()
            pixel_format = frame.get_pixel_format()

            if pixel_format in CV2_CONVERSIONS.keys():
                img = cv2.cvtColor(img, CV2_CONVERSIONS[pixel_format])
            conversion_finished = time.time()
            conversion_time = conversion_finished - conversion_started


            if self.svaed_frames_dir is not None:
                output_img_path = self.svaed_frames_dir / f'frame_{frame.get_id()}.png'
                cv2.imwrite(output_img_path.as_posix(), img)
            
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
    save_pcap = args.pcap
    save_adaptive = args.adaptive
    save_frames = args.save_frames

    # Get the current time
    current_time = datetime.now()
    # Format the current time as a string
    time_string = current_time.strftime("%H_%M_%S")


    output_parameters_path = None
    if save_adaptive:
        output_parameters_path = Path(rf'./OUTPUT/adaptive_parameters_{time_string}.json')
        with open(output_parameters_path.absolute().as_posix(), 'w') as file:
            file.write(json.dumps({'recording time': time_string}, indent=2))

    if save_pcap:
        # Command to start tshark with pcap writer and filter for GVSP or GVCP packets
        pcap_file = Path(rf'./OUTPUT/recording_{time_string}.pcap')
        tshark_command = ["tshark", "-i", "Ethernet 6", "-w", pcap_file.absolute().as_posix(), "((src host 192.168.10.150) and (dst host 192.168.1.100)) or ((dst host 192.168.10.150) and (src host 192.168.1.100))"]
        
        # Start the subprocess
        process = subprocess.Popen(tshark_command)

    svaed_frames_dir = None
    if save_frames:
        svaed_frames_dir = Path(rf'./OUTPUT/recording_{time_string}_images')
        svaed_frames_dir.mkdir()
    with Vimba.get_instance():
        with get_camera(cam_id) as cam:
            
            cam.get_feature_by_name('DSPSubregionTop').set(0)
            cam.get_feature_by_name('DSPSubregionBottom').set(1216)
            cam.get_feature_by_name('DSPSubregionLeft').set(0)
            cam.get_feature_by_name('DSPSubregionRight').set(1936)

            dsp_subregion = {'top': 400,
                            'bottom': 410,
                            'left': 950,
                            'right':960}
            
            if save_adaptive:
                output_parameters_path = Path(rf'./OUTPUT/adaptive_parameters_{time_string}.json')
                with open(output_parameters_path.absolute().as_posix(), 'w') as file:
                    file.write(json.dumps(dsp_subregion, indent=2))

            cam.get_feature_by_name('DSPSubregionTop').set(dsp_subregion['top'])
            cam.get_feature_by_name('DSPSubregionBottom').set(dsp_subregion['bottom'])
            cam.get_feature_by_name('DSPSubregionLeft').set(dsp_subregion['left'])
            cam.get_feature_by_name('DSPSubregionRight').set(dsp_subregion['right'])
            

            # Start Streaming, wait for five seconds, stop streaming
            setup_camera(cam)
            handler = Handler(detector, output_parameters_path, svaed_frames_dir)

            try:
                # Start Streaming with a custom a buffer of 10 Frames (defaults to 5)
                cam.start_streaming(handler=handler, buffer_count=10)
                handler.shutdown_event.wait()

            finally:
                cam.stop_streaming()
    if save_pcap:
        process.terminate()
    
    if save_adaptive:
        # fix json file
        with open(output_parameters_path.absolute().as_posix(), 'r') as file:
            json_data = file.read()
        fixed_json = re.sub(r'\n},?\n{', ',',json_data)
        with open(output_parameters_path.absolute().as_posix(), 'w') as file:
            file.write(fixed_json)
        
if __name__ == '__main__':
    main()
