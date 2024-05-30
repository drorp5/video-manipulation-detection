from time import sleep, time
from abc import ABC, abstractmethod

from manipultation_utils import GigELink


class GigEAttacker(ABC):
    def __init__(
        self,
        config: dict,
    ) -> None:
        self.config = config

    def set_gige_link(self) -> None:
        self.gige_link = GigELink(
            interface=self.config["gige"]["interface"],
            cp_ip=self.cp_ip,
            camera_ip=self.camera_ip,
            img_width=self.config["gige"]["gvsp"]["width"],
            img_height=self.config["gige"]["gvsp"]["height"],
            max_payload_bytes=self.config["gige"]["gvsp"]["max_payload_bytes"],
        )
        self.gige_link.sniff_link_parameters()

    def run_pre_attack_stage(self) -> None:
        sleep(self.config["timing"]["pre_attack_duration_in_seconds"])

    @abstractmethod
    def attack(self) -> None:
        raise NotImplementedError

    def run_attack_stage(self) -> None:
        start_time = time()
        self.set_gige_link()
        while time() - start_time < self.config["timing"]["attack_duration_in_seconds"]:
            self.attack()

    def run_post_attack_stage(self) -> None:
        sleep(self.post_attack_duration_in_seconds)

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
