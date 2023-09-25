import tkinter as tk
import pandas as pd
import subprocess
from datetime import datetime
from ..asynchronous_grab_opencv import initialize_async_grab, parse_args

# Create the main application window
root = tk.Tk()
root.title("Experiment GUI")

# Get the current time and formant it as a string
current_time = datetime.now()
time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")

# Initialize BooleanVar variables
save_pcap_var = tk.BooleanVar()
save_adaptive_var = tk.BooleanVar()
save_frames_var = tk.BooleanVar()

# Initialize file name input variables
pcap_filename_var = tk.StringVar()
adaptive_filename_var = tk.StringVar()
frames_filename_var = tk.StringVar()

# Function to create or destroy file name input field based on the checkbox state
def update_filename_entry(checkbox_var, filename_var, label, default_text):
    if checkbox_var.get():
        entry = tk.Entry(root, textvariable=filename_var, width=30)
        entry.insert(0, default_text)
        entry_label = tk.Label(root, text=label)
        entry_label.pack()
        entry.pack()
    else:
        filename_var.set("")  # Clear the filename variable
        for widget in root.winfo_children():
            if isinstance(widget, (tk.Entry, tk.Label)):
                widget.pack_forget()

def update_pcap_entry(checkbox_var, filename_var, label, default_text=f'recording_{time_string}.pcap'):
    return update_filename_entry(checkbox_var, filename_var, label, default_text)

def update_adaptive_entry(checkbox_var, filename_var, label, default_text=f'adaptive_parameters_{time_string}.pcap'):
    return update_filename_entry(checkbox_var, filename_var, label, default_text)

def update_frames_entry(checkbox_var, filename_var, label, default_text=f'recording_{time_string}_images'):
    return update_filename_entry(checkbox_var, filename_var, label, default_text)


# Define the function that performs the experiment
def run_experiment():
    # get default arguments
    args = parse_args()
    args.pcap = save_pcap_var.get()
    if args.pcap:
        args.pcap_name = pcap_filename_var.get()
    args.adaptive = save_adaptive_var.get()
    if args.adaptive:
        args.adaptive_name = adaptive_filename_var.get()
    args.save_frames = save_frames_var.get()
    if args.save_frames:
        args.frames_dir = frames_filename_var.get()
    initialize_async_grab(args)
    
# Function to update the text widget with the DataFrame
def update_display():
    run_experiment()

# Create checkboxes with BooleanVar variables
save_pcap_checkbox = tk.Checkbutton(root, text="Save PCAP", variable=save_pcap_var,
                                     command=lambda: update_pcap_entry(save_pcap_var, pcap_filename_var, "PCAP File Name:"))
save_pcap_checkbox.pack()

save_adaptive_checkbox = tk.Checkbutton(root, text="Save Adaptive", variable=save_adaptive_var,
                                         command=lambda: update_adaptive_entry(save_adaptive_var, adaptive_filename_var, "Adaptive File Name:"))
save_adaptive_checkbox.pack()

save_frames_checkbox = tk.Checkbutton(root, text="Save Frames", variable=save_frames_var,
                                       command=lambda: update_frames_entry(save_frames_var, frames_filename_var, "Frames File Name:"))
save_frames_checkbox.pack()

# Create a "Run" button
run_button = tk.Button(root, text="Run Experiment", command=update_display)
run_button.pack(pady=10)

# Create a text widget to display the results
text_widget = tk.Text(root, height=10, width=50)
text_widget.pack()

# Start the GUI main loop
root.mainloop()
