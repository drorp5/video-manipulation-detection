from pathlib import Path
from uuid import uuid4
import time
import threading
import subprocess
from datetime import datetime
import yaml
import logging
import sys

from attacker import GigEAttacker
from car import Car


def run_thread(func):
    thread = threading.Thread(target=func)
    thread.start()
    return thread


class Experiment:
    def __init__(
        self,
        config: dict,
        logger: logging.Logger,
        car: Car,
        attacker: GigEAttacker,
    ) -> None:
        self.logger = logger
        self.attacker = attacker
        self.car = car
        self.config = config

        self.id = str(uuid4())
        now = datetime.now()
        self.start_time_string = now.strftime("%Y_%m_%d_%H_%M_%S")

        # initialize results directory
        self.base_results_dir = (
            Path(self.config["experiment"]["results_directory"])
            / f"{self.start_time_string}_{self.id}"
        )
        self.base_results_dir.mkdir(parents=True)

        # save cpnfiguration file
        yaml_file_path = self.base_results_dir / f"config_{self.id}.yaml"
        with yaml_file_path.open("w") as file:
            yaml.dump(self.config, file, default_flow_style=False)

        # add logger handlers
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        log_level = self.logger.getEffectiveLevel()
        if "console" in self.config["experiment"]["log_type"]:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if "file" in self.config["experiment"]["log_type"]:
            log_path = self.base_results_dir / f"log_{self.id}.log"
            file_handler = logging.FileHandler(log_path.as_posix())
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _start_pcap_recording(self) -> None:
        self.pcap_path = self.base_results_dir / f"{self.id}.pcap"
        cp_ip = self.attacker.cp_ip
        camera_ip = self.attacker.camera_ip
        gvsp_gvcp_filter = f"((src host {camera_ip}) and (dst host {cp_ip})) or ((dst host {camera_ip}) and (src host {cp_ip}))"
        tshark_command = [
            "tshark",
            "-i",
            self.attacker.interface,
            "-w",
            self.pcap_path.absolute().as_posix(),
            "-f",
            gvsp_gvcp_filter,
            "-B",
            5,
        ]

        # Start the subprocess
        self.logger.info("Pcap Recording Started")
        process = subprocess.Popen(
            tshark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            # Wait for the subprocess to finish or for the shutdown event to be set
            while process.poll() is None:
                if self.pcap_shutdown_event.wait(
                    timeout=1
                ):  # Wait for 1 second or event set
                    process.terminate()  # Terminate the process if shutdown event is set
                    process.wait()  # Wait for the process to terminate
                    break
        finally:
            # Ensure the process is terminated if the function exits for any reason
            if process.poll() is None:
                process.terminate()
                process.wait()
        self.logger.info("Pcap Recording Stopped")

    def start_pcap_recording_thread(self):
        self.pcap_shutdown_event = self.car.external_event
        thread = threading.Thread(target=self._start_pcap_recording)
        thread.start()
        return thread

    def run(self):
        if self.config["experiment"]["record_pcap"]:
            tshark_thread = self.start_pcap_recording_thread()
        car_thread = run_thread(self.car.run)
        attacker_thread = run_thread(self.attacker.run)

        threading.Timer(
            self.config["experiment"]["duration"], self.car.shutdown_event.set
        ).start()
        threading.Timer(
            self.config["experiment"]["duration"], self.attacker.shutdown_event.set
        ).start()

        # Join threads
        attacker_thread.join()
        car_thread.join()
        if self.config["experiment"]["record_pcap"]:
            tshark_thread.join()
