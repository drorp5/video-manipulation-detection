from attacker.attacker import GigEAttacker
from attacker.attacker_frame_injection import GigEAttackerFrameInjection
from attacker.attacker_stripe_injection import GigEAttackerStripeInjection


Attackers = {'FullFrameInjection': GigEAttackerFrameInjection,
             'StripeInjection': GigEAttackerStripeInjection}