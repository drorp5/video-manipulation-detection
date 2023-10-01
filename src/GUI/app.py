from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Optional
import pandas as pd
from datetime import datetime
import sys
sys.path.append('./src')
import asynchronous_grab_opencv
import adaptive_parameters.utils
from video_utils import gvsp_pcap_to_raw_images

# Create the main application window
root = tk.Tk()
root.title("Experiment GUI")

# Get the current time and formant it as a string
current_time = datetime.now()
time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")

# Initialize BooleanVar variables
plot_var = tk.BooleanVar()
save_pcap_var = tk.BooleanVar()
save_adaptive_var = tk.BooleanVar()
save_frames_var = tk.BooleanVar()

# Initialize file name input variables
output_dir_var = tk.StringVar()
pcap_filename_var = tk.StringVar()
adaptive_filename_var = tk.StringVar()
frames_filename_var = tk.StringVar()

# Initialize numeric variables
duration_var = tk.DoubleVar(value=10) #-1
buffer_count_var = tk.IntVar(value=1) #10
initial_exposure_var = tk.IntVar(value=-1)
exposure_diff_var = tk.IntVar(value=0)
exposure_change_timing_var = tk.DoubleVar(value=5)
fps_var = tk.IntVar(value=20)

def toggle_entry(entry: tk.Entry, default: str):
    if entry["state"]=="disabled":
        entry["state"] = "normal"
        entry.insert(0, default)
    elif entry["state"] == "normal":
        entry.delete(0, tk.END)
        entry["state"] = "disabled"
        
# Define the function that performs the experiment
def run_experiment():
    # get default arguments
    args = asynchronous_grab_opencv.get_default_args()
    args.output_dir = Path(output_dir_var.get())
    args.buffer_count = buffer_count_var.get()
    args.exposure = initial_exposure_var.get()
    args.exposure_diff = exposure_diff_var.get()
    args.exposure_change_timing = exposure_change_timing_var.get()

    prefix = ''
    postfix = ''
    if args.exposure > 0 and args.exposure_diff > 0:
        postfix = f'_static_exp_{args.exposure}_diff_{args.exposure_diff}'

    args.fps = fps_var.get()
    args.pcap = save_pcap_var.get()
    if args.pcap:
        adaptive_entry.insert(0, prefix)
        pcap_entry.insert(tk.END, postfix)
        args.pcap_name = pcap_filename_var.get()
    args.adaptive = save_adaptive_var.get()
    if args.adaptive:
        adaptive_entry.insert(0, prefix)
        adaptive_entry.insert(tk.END, postfix)
        args.adaptive_name = adaptive_filename_var.get()
    args.save_frames = save_frames_var.get()
    if args.save_frames:
        adaptive_entry.insert(0, prefix)
        frames_dir_entry.insert(tk.END, postfix)
        args.frames_dir = frames_filename_var.get()
    args.plot = plot_var.get()
    if duration_var.get() > 0:
        args.duration = duration_var.get()
    asynchronous_grab_opencv.start_async_grab(args)
    update_display()
    
# Function to update the text widget with the DataFrame
def get_exposure_dataframe() -> Optional[pd.DataFrame]:
    if save_adaptive_var.get():
        adaptive_parameters_path = Path(output_dir_var.get()) /  f'{adaptive_filename_var.get()}.json'
        df = adaptive_parameters.utils.read_adaptive_data(adaptive_parameters_path)
        return df
    
def get_intensity_dataframe() -> Optional[pd.DataFrame]:
    if save_pcap_var.get():
        pcap_path = Path(output_dir_var.get()) / f'{pcap_filename_var.get()}.pcap'
        intenities_dst_dir = Path(output_dir_var.get()) / pcap_filename_var.get()
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

def update_display():
    # run_experiment()
    exposure_df = get_exposure_dataframe()
    intensity_df = get_intensity_dataframe()
    df = get_merged_dataframe(exposure_df, intensity_df)
    if df is None:
        return

    for widget in root.winfo_children():
        if isinstance(widget, (ttk.Treeview, ttk.Scrollbar, tk.Text)):
            widget.destroy()
    
    tree = ttk.Treeview(root, columns=['Index'] + list(df.columns), show='headings')
    tree.heading('Index', text='Frame ID')
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)  # Adjust the column width as needed
    for i, row in df.iterrows():
        tree.insert("", "end", values=[i] + list(row))
    scrollbar = ttk.Scrollbar(root, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar.set)
    tree.pack()
    
    # Create a text widget to display the results
    text_widget = tk.Text(root, height=6, width=40)
    total_num_frames = df.index[-1] - df.index[0] + 1
    output_text = f'Total {total_num_frames} frames collected\n'
    for column_name, column_values in df.iteritems():
        num_missing_frames = total_num_frames - len(column_values[~column_values.isna()])
        output_text += f'Missing {num_missing_frames} {column_name} values\n'
    text_widget.insert(tk.END, output_text)
    text_widget.pack()

# Create checkboxes with BooleanVar variables
save_pcap_checkbox = tk.Checkbutton(root, text="Show Live Stream", variable=plot_var)
save_pcap_checkbox.pack()

duration_label = tk.Label(root, text="Streaming Duration Seconds:")
duration_entry = tk.Entry(root, textvariable=duration_var)
duration_label.pack()
duration_entry.pack()

output_dir_label = tk.Label(root, text="Output Base Dir:")
output_dir_entry = tk.Entry(root, textvariable=output_dir_var)
output_dir_entry.insert(0,r'OUTPUT/recordings_bank')
output_dir_label.pack()
output_dir_entry.pack()

buffer_count_label = tk.Label(root, text="Streaming Buffer Count:")
buffer_count_entry = tk.Entry(root, textvariable=buffer_count_var)
buffer_count_label.pack()
buffer_count_entry.pack()

fps_label = tk.Label(root, text="Frames Per Second:")
fps_entry = tk.Entry(root, textvariable=fps_var)
fps_label.pack()
fps_entry.pack()

initial_exposure_label = tk.Label(root, text="Initial Exposure Time microseconds (-1 auto):")
initial_exposure_entry = tk.Entry(root, textvariable=initial_exposure_var)
initial_exposure_label.pack()
initial_exposure_entry.pack()

exposure_diff_label = tk.Label(root, text="Exposure Time Diff microseconds:")
exposure_diff_entry = tk.Entry(root, textvariable=exposure_diff_var)
exposure_diff_label.pack()
exposure_diff_entry.pack()

exposure_change_timing_var_label = tk.Label(root, text="Exposure Change Timing seconds:")
exposure_change_timing_var_entry = tk.Entry(root, textvariable=exposure_change_timing_var)
exposure_change_timing_var_label.pack()
exposure_change_timing_var_entry.pack()

pcap_entry_label = tk.Label(root, text="PCAP File Name:")
pcap_entry = tk.Entry(root, textvariable=pcap_filename_var, width=50, state='disabled' )
save_pcap_checkbox = tk.Checkbutton(root, text="Save PCAP", variable=save_pcap_var,
                                     command=lambda: toggle_entry(pcap_entry, f'recording_{time_string}'),
                                     )
save_pcap_checkbox.pack()
pcap_entry_label.pack()
pcap_entry.pack()

adaptive_entry_label = tk.Label(root, text="Adaptive Parameters File Name:")
adaptive_entry = tk.Entry(root, textvariable=adaptive_filename_var, width=50, state='disabled' )
save_adaptive_checkbox = tk.Checkbutton(root, text="Save Adaptive", variable=save_adaptive_var,
                                         command=lambda: toggle_entry(adaptive_entry, f'adaptive_parameters_{time_string}'))
save_adaptive_checkbox.pack()
adaptive_entry_label.pack()
adaptive_entry.pack()

frames_dir_entry_label = tk.Label(root, text="Frames File Name:")
frames_dir_entry = tk.Entry(root, textvariable=frames_filename_var, width=50, state='disabled' )
save_frames_checkbox = tk.Checkbutton(root, text="Save Frames", variable=save_frames_var,
                                       command=lambda: toggle_entry(frames_dir_entry, f'recording_{time_string}_images'))
save_frames_checkbox.pack()
frames_dir_entry_label.pack()
frames_dir_entry.pack()

# Create a "Run" button
run_button = tk.Button(root, text="Run Experiment", command=run_experiment)
run_button.pack(pady=10)

# Create a text widget to display the results
# text_widget = tk.Text(root, height=10, width=50)
# text_widget.pack()

# Start the GUI main loop
root.mainloop()
