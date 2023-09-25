import tkinter as tk
from tkinter import ttk
import pandas as pd
import subprocess

# Create the main application window
root = tk.Tk()
root.title("Experiment GUI")

# Initialize BooleanVar variables
save_pcap_var = tk.BooleanVar()
save_adaptive_var = tk.BooleanVar()
save_frames_var = tk.BooleanVar()

# Initialize file name input variables
pcap_filename_var = tk.StringVar()
adaptive_filename_var = tk.StringVar()
frames_filename_var = tk.StringVar()

# Function to create or destroy file name input field based on the checkbox state
def update_filename_entry(checkbox_var, filename_var, label):
    if checkbox_var.get():
        entry = ttk.Entry(root, textvariable=filename_var)
        entry_label = ttk.Label(root, text=label)
        entry_label.pack()
        entry.pack()
    else:
        filename_var.set("")  # Clear the filename variable
        for widget in root.winfo_children():
            if isinstance(widget, (ttk.Entry, ttk.Label)):
                widget.pack_forget()

# Callback function to update the displayed filenames
def update_displayed_filenames():
    # Get the values of BooleanVar variables
    save_pcap = save_pcap_var.get()
    save_adaptive = save_adaptive_var.get()
    save_frames = save_frames_var.get()
    
    # Get the specified filenames
    pcap_filename = pcap_filename_var.get()
    adaptive_filename = adaptive_filename_var.get()
    frames_filename = frames_filename_var.get()
    
    # Update displayed filenames
    pcap_filename_label.config(text=f"PCAP File Name: {pcap_filename if save_pcap else ''}")
    adaptive_filename_label.config(text=f"Adaptive File Name: {adaptive_filename if save_adaptive else ''}")
    frames_filename_label.config(text=f"Frames File Name: {frames_filename if save_frames else ''}")

# Define the function that performs the experiment
def run_experiment():
    # Get the values of BooleanVar variables
    save_pcap = save_pcap_var.get()
    save_adaptive = save_adaptive_var.get()
    save_frames = save_frames_var.get()
    
    # Get the specified filenames
    pcap_filename = pcap_filename_var.get()
    adaptive_filename = adaptive_filename_var.get()
    frames_filename = frames_filename_var.get()
    
    # Construct the command using string formatting
    command = (
        f"python script.py "
        f"--pcap {save_pcap} --pcap_filename {pcap_filename} "
        f"--adaptive {save_adaptive} --adaptive_filename {adaptive_filename} "
        f"--save_frames {save_frames} --frames_filename {frames_filename}"
    )
    
    try:
        # Execute the command and capture the output
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        output = result.stdout
        error = result.stderr
    except subprocess.CalledProcessError as e:
        # Handle any errors from the external script
        output = f"Error: {e}\n"
        error = e.stderr
    
    # Display the output in the text widget
    text_widget.delete(1.0, tk.END)  # Clear previous content
    text_widget.insert(tk.END, f"Output:\n{output}\nError:\n{error}")

# Create checkboxes with BooleanVar variables
save_pcap_checkbox = ttk.Checkbutton(root, text="Save PCAP", variable=save_pcap_var,
                                     command=update_displayed_filenames)
save_pcap_checkbox.pack()
update_filename_entry(save_pcap_var, pcap_filename_var, "PCAP File Name:")
pcap_filename_label = ttk.Label(root, text="")

save_adaptive_checkbox = ttk.Checkbutton(root, text="Save Adaptive", variable=save_adaptive_var,
                                         command=update_displayed_filenames)
save_adaptive_checkbox.pack()
update_filename_entry(save_adaptive_var, adaptive_filename_var, "Adaptive File Name:")
adaptive_filename_label = ttk.Label(root, text="")

save_frames_checkbox = ttk.Checkbutton(root, text="Save Frames", variable=save_frames_var,
                                       command=update_displayed_filenames)
save_frames_checkbox.pack()
update_filename_entry(save_frames_var, frames_filename_var, "Frames File Name:")
frames_filename_label = ttk.Label(root, text="")

# Create a "Run" button
run_button = ttk.Button(root, text="Run Experiment", command=update_displayed_filenames)
run_button.pack(pady=10)

# Create a text widget to display the results
text_widget = tk.Text(root, height=10, width=50)
text_widget.pack()

# Start the GUI main loop
root.mainloop()
