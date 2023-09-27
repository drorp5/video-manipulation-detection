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
from typing import Dict, Optional, Tuple
from vimba import *
from sign_detectors import stop_sign_detectors as detectors 
import time
import argparse
from config import CV2_CONVERSIONS
import json
from datetime import datetime
import subprocess
import re 
import numpy as np
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        debug = kwargs.get('debug', False)
        if debug:
            start_time = time.perf_counter()
        result = func(*args, **kwargs)
        if debug:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            return result, total_time
        return result, 0
    return timeit_wrapper

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

def setup_camera(cam: Camera, exposure_val: Optional[int], fps_val: Optional[int]):
    with cam:
        # Enable auto exposure time setting if camera supports it
        if exposure_val is None:
            try:
                cam.ExposureAuto.set('Continuous')

            except (AttributeError, VimbaFeatureError):
                pass
        else:
            try:
                cam.ExposureAuto.set('Off')
                cam.ExposureTimeAbs.set(exposure_val)
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

        # Set constant frame rate if specified
        if fps_val is not None:
            try:
                cam.TriggerMode.set("Off")
                cam.AcquisitionFrameRateAbs.set(fps_val)
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

def setup_camera_dsp_subregion(cam: Camera, top:int=0, bottom:int=1216, left:int=0, right:int=1936):
    # set dsp region to full scale before changing
    full_scale_top = 0
    full_scale_bottom = 1216
    full_scale_left = 0
    full_scale_right = 1936

    cam.get_feature_by_name('DSPSubregionTop').set(full_scale_top)
    cam.get_feature_by_name('DSPSubregionBottom').set(full_scale_bottom)
    cam.get_feature_by_name('DSPSubregionLeft').set(full_scale_left)
    cam.get_feature_by_name('DSPSubregionRight').set(full_scale_right)

    if top != full_scale_top:
        cam.get_feature_by_name('DSPSubregionTop').set(top)
    if bottom != full_scale_bottom:
        cam.get_feature_by_name('DSPSubregionBottom').set(bottom)
    if left != full_scale_left:
        cam.get_feature_by_name('DSPSubregionLeft').set(left)
    if right != full_scale_right:
        cam.get_feature_by_name('DSPSubregionRight').set(right)

def change_exposure_time(cam: Camera, exposure_diff: int):
    with cam:
       prev = cam.ExposureTimeAbs.get()
       cam.ExposureTimeAbs.set(prev + exposure_diff) 

class Handler:
    def __init__(self, plot:bool, detector_name: str, output_parameters_path:Path=None, saved_frames_dir:Path=None, debug:bool=True):
        self.shutdown_event = threading.Event()
        self.detector = detectors.get_detector(detector_name)
        self.downfactor = 4
        self.output_parameters_path = output_parameters_path
        self.saved_frames_dir = saved_frames_dir
        self.debug = debug
        self.plot = plot

    @timeit
    def convert_image(self, img: np.array, pixel_format: PixelFormat, debug=False) -> np.array:
        if pixel_format in CV2_CONVERSIONS.keys():
            img = cv2.cvtColor(img, CV2_CONVERSIONS[pixel_format])
        return img

    @timeit
    def resize_image(self, img: np.array, debug=False) -> np.array:
        height = int(img.shape[0] / self.downfactor)
        width = int(img.shape[1] / self.downfactor)
        dim = (width, height)    
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    @timeit
    def detect_objects_in_image(self, img: np.array, debug=False):
        if self.detector is not None:
            detections = self.detector.detect(img)
            img = detectors.draw_bounding_boxes(img, detections)
        return img

    def __call__(self, cam: Camera, frame: Frame):
        ENTER_KEY_CODE = 13
        if self.plot:
            key = cv2.waitKey(1)
            if key == ENTER_KEY_CODE:
                self.shutdown_event.set()
                return

        if frame.get_status() == FrameStatus.Complete:
            if self.debug:
                print('{} acquired {}'.format(cam, frame), flush=True)

            msg = 'Stream from \'{}\'. Press <Enter> to stop stream.'
            img = frame.as_opencv_image()

            # convert image to BGR format 
            pixel_format = frame.get_pixel_format()
            img, conversion_time = self.convert_image(img, pixel_format, debug=self.debug)
            
            # get values of exposure and gain
            if self.output_parameters_path:
                try:
                    frame_data = {}
                    frame_data["exposure_us"] = cam.get_feature_by_name('ExposureTimeAbs').get()
                    # frame_data["gain_db"] = cam.get_feature_by_name('Gain').get()
                    frame_data["live_intensity"] =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float).mean()
                    json_data = {f'frame_{frame.get_id()}': frame_data}
                    with open(self.output_parameters_path.absolute().as_posix(), 'a+') as file:
                        file.write(',\n')
                        file.write(json.dumps(json_data, indent=2))
                except:
                    print('WARNING: cant query parameters')
                
            if self.saved_frames_dir is not None:
                output_img_path = self.saved_frames_dir / f'frame_{frame.get_id()}.png'
                cv2.imwrite(output_img_path.as_posix(), img)
            
            if self.plot:
                processed_img, resizing_time = self.resize_image(img, debug=self.debug)
                processed_img, detection_time = self.detect_objects_in_image(processed_img, debug=self.debug)
                window_name = msg.format(cam.get_name())           
                cv2.imshow(window_name, processed_img)
            
            if self.debug:
                print(f'conversion time = {conversion_time}')
                print(f'resizing time = {resizing_time}')
                print(f'detection time = {detection_time}')
                print(f'total processing time = {conversion_time + resizing_time + detection_time}')
        cam.queue_frame(frame)

def parse_args() -> Dict:
    # Get the current time and formant it as a string
    current_time = datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", help="camera ID for direct access")
    parser.add_argument("--detector", choices=detectors.get_detectors_dict(), help="detection method")
    parser.add_argument("--pcap", help="save pcap dump", action="store_true")
    parser.add_argument("--pcap_name", help="otuput pcap file name", type=str, default=f'recording_{time_string}')
    parser.add_argument("--adaptive", help="query and save adaptive parameters", action="store_true")
    parser.add_argument("--adaptive_name", help="adaptive parameters file name", type=str, default=f'adaptive_parameters_{time_string}')
    parser.add_argument("--save_frames", help="save collected frames", action="store_true")
    parser.add_argument("--frames_dir", help="output frames directory", type=str, default= f'recording_{time_string}_images')
    parser.add_argument("--debug", help="debug mode", action="store_true")
    parser.add_argument("--output_dir", help="base output directory", type=Path, default=Path('./OUTPUT'))
    parser.add_argument("--plot", help="plot live stream of camera", action="store_true")
    parser.add_argument("--duration", help="duration of streaming [seconds]. -1 infinite", type=float, default=-1)
    parser.add_argument("--buffer_count", help="streaming buffer", type=int, default=10)
    parser.add_argument("--exposure", help="exposure time value in manual mode [microseconds]", type=int, default=-1)
    parser.add_argument("--exposure_diff", help="exposure time change during stream [microseconds]", type=int, default=0)
    parser.add_argument("--exposure_change_timing", help="duration till exposure time change during stream [seconds]", type=float, default=0)
    parser.add_argument("--fps", help="frame rate [frames per second]", type=int, default=-1)
    
    args = parser.parse_args()
    args.time_string = time_string
    if args.duration < 0:
        args.duration = None
    if args.exposure < 0:
        args.exposure = None
    if args.fps < 0:
        args.fps = None
    return args

def main_script():
    print_preamble()
    args = parse_args()
    return start_async_grab(args)

def assert_args(args):
    assert args.plot or args.duration is not None, 'if the stream is not plotted, duration must be specified'
    assert not(args.exposure_diff!=0 and args.exposure_change_timing==0), 'exposure diff value must be specified with timing'
    assert args.duration is None or (args.duration > args.exposure_change_timing) 
    
def start_async_grab(args):
    assert_args(args)
    if not args.output_dir.exists():
        args.output_dir.mkdir(parents=True)
    if args.save_frames:
        saved_frames_dir = args.output_dir / args.frames_dir
        saved_frames_dir.mkdir()
    else:
        saved_frames_dir = None
    
    with Vimba.get_instance():
        with get_camera(args.camera_id) as cam:
            #  set dsp region
            dsp_subregion = {'top': 0,
                            'bottom': 1216,
                            'left': 0,
                            'right': 1936}
            setup_camera_dsp_subregion(cam, dsp_subregion['top'], dsp_subregion['bottom'], dsp_subregion['left'], dsp_subregion['right'])

            if args.adaptive:
                adaptive_metadata = {}
                adaptive_metadata['time'] = args.time_string
                adaptive_metadata["dsp_subregion"] = dsp_subregion
                adaptive_metadata["buffer_count"] = args.buffer_count
                if args.fps is not None:
                    adaptive_metadata["fps"] = args.fps
                if args.exposure is not None:
                    adaptive_metadata["initial_exposure"] = args.exposure
                if args.exposure_diff != 0:
                    adaptive_metadata["exposure_diff"] = args.exposure_diff
                if args.exposure_change_timing != 0:
                    adaptive_metadata["exposure_change_timing"] = args.exposure_change_timing
                output_parameters_path = args.output_dir / f'{args.adaptive_name}.json'
                with open(output_parameters_path.absolute().as_posix(), 'w') as file:
                    file.write(json.dumps(adaptive_metadata, indent=2))
            else:
                output_parameters_path = None
            
            setup_camera(cam, exposure_val=args.exposure, fps_val=args.fps)
            handler = Handler(args.plot, args.detector, output_parameters_path, saved_frames_dir,args.debug)

            if args.pcap:
                # Command to start tshark with pcap writer and filter for GVSP or GVCP packets
                pcap_path = args.output_dir / f'{args.pcap_name}.pcap'
                cp_ip = '192.168.1.100'
                camera_ip = '192.168.10.150'
                gvsp_gvcp_filter = f"((src host {camera_ip}) and (dst host {cp_ip})) or ((dst host {camera_ip}) and (src host {cp_ip}))"
                tshark_command = ["tshark", "-i", "Ethernet 6", "-w", pcap_path.absolute().as_posix(), gvsp_gvcp_filter]
                # Start the subprocess
                process = subprocess.Popen(tshark_command)

            try:
                # Start Streaming with a custom frames buffer
                cam.start_streaming(handler=handler, buffer_count=args.buffer_count)
                if args.exposure_change_timing > 0:
                    exposure_change_timer = threading.Timer(args.exposure_change_timing, change_exposure_time, args=(cam, args.exposure_diff))
                    exposure_change_timer.start()
                handler.shutdown_event.wait(args.duration)
                if args.exposure_change_timing > 0:
                    exposure_change_timer.join()

            finally:
                cam.stop_streaming()
                if args.pcap:
                    process.terminate()
                if args.adaptive:
                    # fix json file
                    with open(output_parameters_path.absolute().as_posix(), 'r') as file:
                        json_data = file.read()
                    fixed_json = re.sub(r'\n},?\n{', ',',json_data)
                    with open(output_parameters_path.absolute().as_posix(), 'w') as file:
                        file.write(fixed_json)
        
if __name__ == '__main__':
    main_script()
