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
import sys
from typing import Optional
from vimba import *
import argparse
from argparse import Namespace
from icecream import ic
import traceback
import numpy as np

from gige.handlers import ViewerHandler
from gige.handlers.varying_shape_handler import VaryingShapeHandler
from active_manipulation_detectors.side_channel.data_generator import RandomBitsGeneratorRC4, SequentialBitsGenerator
from active_manipulation_detectors.side_channel.validation import DataValidatorKSymbols, DataValidatorKSymbolsDelayed
from sign_detectors.stop_sign_detectors import get_detector, get_detectors_dict


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

def get_camera(camera_id: Optional[str] = None) -> Camera:
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

def setup_camera(cam: Camera, fps_val: Optional[int] = None):
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

        # Set constant frame rate if specified
        if fps_val is not None:
            try:
                cam.TriggerMode.set("Off")
                cam.AcquisitionFrameRateAbs.set(fps_val)
            except (AttributeError, VimbaFeatureError) as e:
                traceback.print_exc()
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

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", help="camera ID for direct access")
    parser.add_argument("--duration", help="duration of streaming [seconds]. -1 infinite", type=float, default=None)
    parser.add_argument("--buffer_count", help="streaming buffer", type=int, default=1)
    parser.add_argument("--fps", help="frame rate [frames per second]", type=int, default=1)
    parser.add_argument("--detector", choices=get_detectors_dict(), help="detection method")
    
    args = parser.parse_args()
    return args

def main_script():
    print_preamble()
    args = parse_args()
    # args.duration = 20
    args.buffer_count = 1
    start_async_grab(args)
    
def start_async_grab(args):
    with Vimba.get_instance():
        with get_camera(args.camera_id) as cam:
            setup_camera(cam, fps_val=args.fps)
                        
            num_symbols = 8
            bits_per_symbol = int(np.ceil(np.log2(num_symbols)))
            random_bits_generator = RandomBitsGeneratorRC4(key=b'key', num_bits_per_iteration=bits_per_symbol)
            # random_bits_generator = SequentialBitsGenerator(key=b'key', num_bits_per_iteration=bits_per_symbol)
            data_validator = DataValidatorKSymbolsDelayed(bits_in_symbol=bits_per_symbol, symbols_for_detection=2, max_delay=1)
            sign_detector = get_detector(args.detector)

            handler = VaryingShapeHandler(random_bits_generator=random_bits_generator, 
                                        data_validator=data_validator,
                                        num_levels=num_symbols,
                                        increment=2,
                                        sign_detector=sign_detector)
            # Start Streaming with a custom frames buffer
            try:
                cam.start_streaming(handler=handler, buffer_count=args.buffer_count)
                handler.shutdown_event.wait(args.duration)
            except Exception as e:
                print(e)
            finally:
                handler.cleanup(cam)
                cam.stop_streaming()
              
if __name__ == '__main__':
    main_script()
