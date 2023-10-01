import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional
import pandas as pd
from datetime import datetime
import sys
sys.path.append('./src')
import asynchronous_grab_opencv
import adaptive_parameters.utils
from video_utils import gvsp_pcap_to_raw_images
import numpy as np
from playsound import playsound
import time
import os

os.chdir(r'C:\Users\user\Desktop\Dror\video-manipulation-detection')

SUCCESS_SOUND_PATH = Path('./INPUT/success.mp3')
FAILURE_SOUND_PATH = Path('./INPUT/failure.mp3')
LOG_PATH = Path(r'./OUTPUT/recordings_bank/experiments_log.txt')

# Get the current time and formant it as a string
def run_experiment(exposure=-1, exposure_diff=0):
    current_time = datetime.now()
    time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    args = asynchronous_grab_opencv.get_default_args()
    args.output_dir = Path('./OUTPUT/recordings_bank')
    args.buffer_count = 1
    args.exposure = exposure
    args.exposure_diff = exposure_diff
    args.exposure_change_timing = 5

    prefix = ''
    postfix = ''
    if args.exposure > 0 and args.exposure_diff != 0:
        postfix = f'_driving_exp_{args.exposure}_diff_{args.exposure_diff}'

    args.fps = 20
    args.pcap = True
    args.pcap_name = f'{prefix}recording_{time_string}{postfix}'
    args.adaptive = True
    args.adaptive_name = f'{prefix}adaptive_parameters_{time_string}{postfix}'
    args.save_frames = False
    args.plot = False
    args.duration = 10

    asynchronous_grab_opencv.start_async_grab(args)

    pcap_path = args.output_dir /  f'{args.pcap_name}.pcap'
    adaptive_parameters_path = args.output_dir /  f'{args.adaptive_name}.json'
    return pcap_path, adaptive_parameters_path

def get_exposure_dataframe(adaptive_parameters_path: Path) -> Optional[pd.DataFrame]:
    df = adaptive_parameters.utils.read_adaptive_data(adaptive_parameters_path)
    return df
    
def get_intensity_dataframe(pcap_path: Path, intenities_dst_dir:Optional[Path]=None) -> Optional[pd.DataFrame]:
        if intenities_dst_dir is None:
             intenities_dst_dir = pcap_path.parent / pcap_path.stem
        gvsp_pcap_to_raw_images(pcap_path=pcap_path.as_posix(), dst_dir=intenities_dst_dir.as_posix(), intensities_only=True)
        frames_intensity_id, frames_intensity = adaptive_parameters.utils.read_intensity_data(intenities_dst_dir)
        intensity_df = pd.DataFrame({'intensity': frames_intensity}, index=frames_intensity_id)
        return intensity_df

def get_merged_dataframe(df1: Optional[pd.DataFrame], df2: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df1 is not None and df2 is not None:
        return pd.merge(df1, df2, how="outer", left_index=True, right_index=True)
    if df1 is not None and df2 is None:
        return df1
    if df2 is not None and df1 is None:
        return df2

def summerize_results(pcap_path:Optional[Path]=None, adaptive_parameters_path:Optional[Path]=None):
    if adaptive_parameters_path is not None:
        adaptive_df = adaptive_parameters.utils.read_adaptive_data(adaptive_parameters_path)
    else:
        adaptive_df = None
    if pcap_path is not None:
        intensity_df = get_intensity_dataframe(pcap_path)
    else:
        intensity_df = None
    df = get_merged_dataframe(adaptive_df, intensity_df)
    
    total_num_frames = df.index[-1] - df.index[0] + 1
    output_text = f'Total {total_num_frames} frames collected\n'
    for column_name, column_values in df.iteritems():
        num_missing_frames = total_num_frames - len(column_values[~column_values.isna()])
        output_text += f'Missing {num_missing_frames} {column_name} values\n'

    target_frame = 101
    before_target = False
    after_target= False
    for frmae_id in range(target_frame-3,target_frame+5):
        try:
            output_text += f'frame # {frmae_id}: exposure = {df.loc[frmae_id]["exposure_us"]}, intensity = {df.loc[frmae_id]["live_intensity"]}\n'
            if frmae_id <= target_frame:
                before_target = True
            if frmae_id > target_frame:
                after_target = True
        except:
            continue
    
    enough_intensities_recorded = before_target and after_target
    output_text += f'Enough intensities recorded = {enough_intensities_recorded}\n'
    return output_text, enough_intensities_recorded
    
def wrapper(exposure: int, exposure_diff: int):
    if not LOG_PATH.parent.exists():
        LOG_PATH.parent.mkdir(parents=True)

    pcap_path, adaptive_parameters_path = run_experiment(exposure, exposure_diff)
    output_text = f'-----------------------------------\n'
    output_text += f'Experiment:\nexposure = {exposure}, exposure_diff = {exposure_diff}\n'
    summary_text, success_flag  = summerize_results(adaptive_parameters_path=adaptive_parameters_path) #pcap_path=pcap_path
    output_text += summary_text
    output_text += f'-----------------------------------\n'

    with open(LOG_PATH.as_posix(), 'a') as f:
        f.write(output_text)

    print(output_text)

    if success_flag:
        playsound(SUCCESS_SOUND_PATH.as_posix())
    else:
        playsound(FAILURE_SOUND_PATH.as_posix())
        os.remove(pcap_path.as_posix())
        os.remove(adaptive_parameters_path.as_posix())
    return success_flag
    
def parse_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exposure", help="initial exposure value", type=int)
    parser.add_argument("--diff", help="exposure diff value", type=int)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    return wrapper(exposure=args.exposure, exposure_diff=args.diff)
    
if __name__=='__main__':
    suceess_flag = main()
    if suceess_flag:
        sys.exit(0)
    else:
        sys.exit(1)
# if __name__ == '__main__':
#     # exposure = 400
#     # exposure_diff = 500

#     if True:
#         time.sleep(15)
    
#     repetitions = 3
#     max_times = 5

#     # exposures = np.arange(500, 5000, 200).astype(int)
#     # exposures_diff = np.arange(-300, 300, 30).astype(int)
    
#     # exposures = np.array([500, 1000]).astype(int)
#     # exposures_diff = np.array([-30, 30]).astype(int)
    
#     exposures = np.arange(900, 1400, 200).astype(int)
#     exposures_diff = np.arange(-200, 200, 45).astype(int)
        
#     for _ in range(repetitions):
#         for exposure in exposures:
#             for exposure_diff in exposures_diff:
#                 if exposure_diff == 0:
#                     continue
#                 success_flag = False
#                 attempts = 0
#                 while attempts < max_times and not success_flag:
#                     try:
#                         wrapper(exposure.item(), exposure_diff.item())
#                         time.sleep(3)

#                     except KeyboardInterrupt:
#                         raise
#                     except:
#                         pass
#                     finally:
#                         attempts += 1
                        