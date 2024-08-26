import logging

from attacker import GigEAttacker


class GigEAttackerStripeInjection(GigEAttacker):
    """
    GigE Vision attacker that performs stripe injection attacks.

    This class implements a specific type of attack where stripes of fake data
    are injected into consecutive frames of the GigE Vision stream.
    """

    def attack(self) -> None:
        """
        Execute the stripe injection attack.

        This method injects stripes of fake data into consecutive frames
        based on the configuration parameters.
        """
        self.gige_link.inject_stripe_consecutive_frames(
            img_path=self.config["injection"]["fake_path"],
            first_row=self.config["injection"]["stripe"]["first_row"],
            num_rows=self.config["injection"]["stripe"]["num_rows"],
            future_id_diff=self.config["injection"]["stripe"]["future_id_diff"],
            count=self.config["injection"]["stripe"]["count"],
            injection_duration=self.config["timing"]["attack_duration_in_seconds"],
            fps=self.config["timing"]["fps"],
        )

    def run_attack_stage(self) -> None:
        """
        Run the attack stage for stripe injection.

        This method sets up the GigE link and executes the stripe injection attack.
        """
        self.set_gige_link()
        self.log("Starting Stripe Injection Attack", log_level=logging.DEBUG)
        self.attack()
