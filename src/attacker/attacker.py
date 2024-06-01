from time import sleep, time
from abc import ABC, abstractmethod
import threading
import logging
from typing import Optional

from manipultation_utils import GigELink


class GigEAttacker(ABC):
    def __init__(
        self,
        config: dict,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.config = config
        self.shutdown_event = threading.Event()
        self.logger = logger

    def set_gige_link(self) -> None:
        self.gige_link = GigELink(
            interface=self.config["gige"]["interface"],
            cp_ip=self.cp_ip,
            camera_ip=self.camera_ip,
            img_width=self.config["gige"]["gvsp"]["width"],
            img_height=self.config["gige"]["gvsp"]["height"],
            max_payload_bytes=self.config["gige"]["gvsp"]["max_payload_bytes"],
        )
        self.log('Sniffing link parameters')
        self.gige_link.sniff_link_parameters()
        self.log(self.gige_link.get_summary(), log_level=logging.DEBUG)

    def run_pre_attack_stage(self) -> None:
        waiting_time = self.config["timing"]["pre_attack_duration_in_seconds"]
        self.log('Attacking Pre stage - waiting for {waiting_time} seconds', log_level=logging.DEBUG)
        sleep(waiting_time)

    @abstractmethod
    def attack(self) -> None:
        raise NotImplementedError

    def run_attack_stage(self) -> None:
        self.log('Starting attack stage') 
        start_time = time()
        self.set_gige_link()
        while time() - start_time < self.config["timing"]["attack_duration_in_seconds"]:
            self.log('Attacking', log_level=logging.DEBUG)
            self.attack()

    def run_post_attack_stage(self) -> None:
        while self.shutdown_event.wait(1):
            pass

    def run(self):
        self.run_pre_attack_stage()
        self.run_attack_stage()
        self.run_post_attack_stage()

    @property
    def cp_ip(self) -> str:
        return self.config["gige"]["cp"]["ip"]

    @property
    def camera_ip(self) -> str:
        return self.config["gige"]["camera"]["ip"]

    @property
    def interface(self) -> str:
        return self.config["gige"]["interface"]
    
    def log(self, msg, log_level=logging.INFO):
        if self.logger is None:
            print(msg)
            return
        if log_level == logging.DEBUG:
            self.logger.debug(msg)
        elif log_level == logging.INFO:
            self.logger.info(msg)
        elif log_level == logging.WARNING:
            self.logger.warning(msg)
        elif log_level == logging.ERROR:
            self.logger.error(msg)
        elif log_level == logging.CRITICAL:
            self.logger.critical(msg)
        else:
            raise ValueError(f"Invalid log level: {log_level}")

