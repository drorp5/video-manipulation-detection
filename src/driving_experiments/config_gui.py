import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml

from driving_experiments.run_experiment import run_experiment
from attacker import Attackers
from driving_experiments.experiment import Experiment
from gige import FramesStatistics, GvspPcapParser
from gige.pcap import PcapParser

# Define float keys globally
INT_KEYS = {
    "num_widths",
    "num_symbols",
    "max_delay",
    "first_row",
    "num_rows",
    "future_id_diff",
    "count",
}

FLOAT_KEYS = {
    "duration",
    "pre_attack_duration_in_seconds",
    "attack_duration_in_seconds",
    "ampiric_frame_time_in_seconds",
}


class ConfigGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Configuration GUI")

        # Set the default window size
        self.root.geometry("1200x600")

        # Path to the YAML configuration file
        config_path = "driving_experiments/experiment_config.yaml"

        # Load the YAML file
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.entries = {}

        # Create a canvas to hold the frame
        self.canvas = tk.Canvas(root)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add a scrollbar to the canvas
        self.scrollbar = ttk.Scrollbar(
            root, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Create a frame inside the canvas
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Create GUI based on the YAML structure
        self.create_widgets(self.config, "", self.scrollable_frame)

        # Save button
        save_button = ttk.Button(
            root, text="Save Configuration", command=self.save_config
        )
        save_button.pack(pady=5)

        # Run Experiment button
        run_button = ttk.Button(
            root, text="Run Experiment", command=self.run_experiment
        )
        run_button.pack(pady=5)

        # Validate experiment button
        validate_button = ttk.Button(
            root, text="Validate Experiment", command=self.validate_experiment
        )
        validate_button.pack(pady=5)

        # Configure canvas scrolling
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.scrollable_frame.bind("<Configure>", self.on_frame_configure)

        self._experiment = None

    def create_widgets(self, config, parent_key, parent_frame):
        for key, value in config.items():
            current_key = f"{parent_key}.{key}" if parent_key else key

            if isinstance(value, dict):
                if key == "car":
                    car_frame = ttk.LabelFrame(parent_frame, text="Car")
                    car_frame.pack(
                        side="left", padx=10, pady=5, fill="both", expand=True
                    )
                    self.create_widgets(value, current_key, car_frame)
                elif key == "attacker":
                    attacker_frame = ttk.LabelFrame(
                        parent_frame, text="Attacker", width=300
                    )
                    attacker_frame.pack(
                        side="left", padx=10, pady=5, fill="both", expand=True
                    )
                    self.create_widgets(value, current_key, attacker_frame)
                else:
                    self.create_widgets(value, current_key, parent_frame)
            else:
                frame = ttk.Frame(parent_frame)
                frame.pack(fill="x", padx=10, pady=5)

                label = ttk.Label(frame, text=key.capitalize())
                label.pack(side="top", anchor="w")

                if key == "log_type":
                    self.entries[current_key] = {}
                    options_frame = ttk.Frame(frame)
                    options_frame.pack(side="top", fill="x", expand=True, anchor="w")
                    for option in ["console", "file"]:
                        var = tk.BooleanVar(value=True)  # Default is checked
                        checkbox = ttk.Checkbutton(
                            options_frame, text=option, variable=var
                        )
                        checkbox.pack(side="top", anchor="w")
                        self.entries[current_key][option] = var
                elif key in [
                    "fake_path",
                    "images_dir",
                    "annotations_dir",
                    "results_directory",
                ]:
                    var = tk.StringVar(value=value if value is not None else "")
                    entry = ttk.Entry(frame, textvariable=var)
                    entry.pack(side="left", fill="x", expand=True)
                    self.entries[current_key] = var
                    button = ttk.Button(
                        frame,
                        text="Browse",
                        command=lambda var=var, key=key: self.browse(var, key),
                    )
                    button.pack(side="left")
                elif key == "detector":
                    options = ["null", "Haar", "Yolo", "MobileNet"]
                    var = tk.StringVar(value=value)  # Default to "null"
                    combobox = ttk.Combobox(frame, textvariable=var, values=options)
                    combobox.pack(side="top", fill="x", expand=True, anchor="w")
                    self.entries[current_key] = var
                elif key == "attack_type":
                    options = list(Attackers.keys())
                    var = tk.StringVar(value=value)
                    combobox = ttk.Combobox(frame, textvariable=var, values=options)
                    combobox.pack(side="top", fill="x", expand=True, anchor="w")
                    self.entries[current_key] = var
                elif key in ["record_pcap", "viewer"]:
                    var = tk.BooleanVar(
                        value=(value is True)
                    )  # Default to True if True in config, otherwise False
                    checkbox = ttk.Checkbutton(frame, variable=var)
                    checkbox.pack(side="top", anchor="w")
                    self.entries[current_key] = var
                else:
                    var = tk.StringVar(value=str(value) if value is not None else "")
                    if current_key in {
                        "experiment.duration",
                        "attacker.timing.pre_attack_duration_in_seconds",
                    }:
                        var.trace("w", self.sync_duration)
                    entry = ttk.Entry(frame, textvariable=var)
                    entry.pack(side="left", fill="x", expand=True)
                    self.entries[current_key] = var

                    if current_key == "car.duration":
                        entry.config(state="disabled")

    def sync_duration(self, *args):
        experiment_duration_key = "experiment.duration"
        car_duration_key = "car.duration"
        if experiment_duration_key in self.entries:
            experiment_duration = self.entries[experiment_duration_key].get()
            self.entries[car_duration_key].set(experiment_duration)

    def browse(self, var, key):
        if key == "fake_path":
            file_path = filedialog.askopenfilename()
            if file_path:
                var.set(file_path)
        else:  # results_directory, images_dir, and annotations_dir
            directory_path = filedialog.askdirectory()
            if directory_path:
                var.set(directory_path)

    def validate_duration(self):
        experiment_duration = self.entries.get(
            "experiment.duration", tk.StringVar()
        ).get()
        car_duration = self.entries.get("car.duration", tk.StringVar()).get()

        if not experiment_duration or not car_duration:
            messagebox.showerror(
                "Validation Error",
                "Both Experiment Duration and Car Duration must be filled.",
            )
            return False
        return True

    def save_config(self):
        if not self.validate_duration():
            return
        new_config = self.get_entries(self.config, "")
        with open("config_updated.yaml", "w") as file:
            yaml.safe_dump(new_config, file)

    def run_experiment(self):
        if not self.validate_duration():
            return
        new_config = self.get_entries(self.config, "")
        self._experiment = run_experiment(new_config)

    def validate_experiment(self):
        parser = GvspPcapParser(self._experiment.pcap_path)
        frames_statistics = parser.get_frames_statistics()
        messagebox.showinfo(
            "Experiment Recording Statistics",
            str(frames_statistics),
        )

    def get_entries(self, config, parent_key):
        new_config = {}
        for key, value in config.items():
            current_key = f"{parent_key}.{key}" if parent_key else key

            if isinstance(value, dict):
                new_config[key] = self.get_entries(value, current_key)
            else:
                if key == "log_type":
                    selected_options = [
                        option
                        for option, var in self.entries[current_key].items()
                        if var.get()
                    ]
                    new_config[key] = selected_options
                elif key in ["record_pcap", "viewer"]:
                    new_config[key] = self.entries[current_key].get()
                elif key == "detector":
                    new_config[key] = (
                        self.entries[current_key].get()
                        if self.entries[current_key].get() != "null"
                        else None
                    )
                else:
                    entry_value = self.entries[current_key].get()
                    if entry_value == "":
                        new_config[key] = None
                    elif key in INT_KEYS:
                        new_config[key] = int(entry_value)
                    elif key in FLOAT_KEYS:
                        new_config[key] = float(entry_value)
                    else:
                        try:
                            new_config[key] = int(entry_value)
                        except ValueError:
                            try:
                                new_config[key] = float(entry_value)
                            except ValueError:
                                new_config[key] = (
                                    entry_value if entry_value != "None" else None
                                )
        return new_config

    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_frame_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


if __name__ == "__main__":
    root = tk.Tk()
    app = ConfigGUI(root)
    root.mainloop()
