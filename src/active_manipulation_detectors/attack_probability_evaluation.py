import math
from math import comb, ceil, floor


def expectation_of_waiting_time_in_seconds(
    num_symbols: int, success_attack_duration_in_seconds: float, frame_rate_in_hz: float
) -> float:
    p = 1 / num_symbols
    run_length = int(ceil(success_attack_duration_in_seconds * frame_rate_in_hz))
    waiting_length_expectation = sum([1 / p**k for k in range(1, run_length + 1)])
    waiting_time_expectation = waiting_length_expectation / frame_rate_in_hz
    return waiting_time_expectation


def probability_of_attack(
    num_symbols: int,
    success_attack_duration_in_seconds: float,
    total_time_in_seconds: float,
    frame_rate_in_hz: float,
) -> float:
    # reference Introduction to mathematical probability, pages 77-79
    p = 1 / num_symbols
    q = 1 - p
    r = int(ceil(success_attack_duration_in_seconds * frame_rate_in_hz))
    n = int(ceil(total_time_in_seconds * frame_rate_in_hz))
    beta = lambda n, r: sum(
        [
            (-1) ** l * comb(n - l * r, l) * (q * p**r) ** l
            for l in range(int(floor(n / (r + 1))) + 1)
        ]
    )
    z = beta(n, r) - p**r * beta(n - r, r)
    y = 1 - z
    return y
