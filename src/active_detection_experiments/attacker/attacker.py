from time import sleep, time
from abc import ABC, abstractmethod
import threading
import logging
from typing import Optional

from attack_tool.gige_attack_tool import GigEVisionAttackTool


class GigEAttacker(ABC):
    def __init__(
        self,
        config: dict,
        logger: Optional[logging.Logger] = None,
        initialization_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.shutdown_event = threading.Event()
        self.logger = logger
        self.initialization_event = initialization_event

    def set_gige_link(self) -> None:
        self.gige_link = GigEVisionAttackTool(
            interface=self.config["gige"]["interface"],
            cp_ip=self.cp_ip,
            camera_ip=self.camera_ip,
            cp_mac=self.cp_mac,
            camera_mac=self.camera_mac,
            img_width=self.config["gige"]["gvsp"]["width"],
            img_height=self.config["gige"]["gvsp"]["height"],
            max_payload_bytes=self.config["gige"]["gvsp"]["max_payload_bytes"],
            logger=self.logger,
        )
        self.log("Sniffing link parameters", log_level=logging.DEBUG)
        self.gige_link.sniff_link_parameters()

    def run_pre_attack_stage(self) -> None:
        if self.initialization_event is not None:
            self.initialization_event.wait()
        waiting_time = self.config["timing"]["pre_attack_duration_in_seconds"]
        self.log(
            f"Attacking Pre stage - waiting for {waiting_time} seconds",
            log_level=logging.DEBUG,
        )
        sleep(waiting_time)

    @abstractmethod
    def attack(self) -> None:
        raise NotImplementedError

    def run_attack_stage(self) -> None:
        self.log("Starting attack stage", log_level=logging.DEBUG)
        start_time = time()
        self.set_gige_link()
        while time() - start_time < self.config["timing"]["attack_duration_in_seconds"]:
            self.log("Attacking", log_level=logging.DEBUG)
            self.attack()

    def run(self):
        self.run_pre_attack_stage()
        self.run_attack_stage()

    @property
    def cp_ip(self) -> str:
        return self.config["gige"]["cp"]["ip"]

    @property
    def camera_ip(self) -> str:
        return self.config["gige"]["camera"]["ip"]

    @property
    def cp_mac(self) -> str:
        return self.config["gige"]["cp"]["mac"]

    @property
    def camera_mac(self) -> str:
        return self.config["gige"]["camera"]["mac"]

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
