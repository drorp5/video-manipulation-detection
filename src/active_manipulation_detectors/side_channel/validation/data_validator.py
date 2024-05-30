from abc import ABC, abstractmethod

from active_manipulation_detectors.validation_status import ValidationStatus
from active_manipulation_detectors.side_channel.queue import FixedSizeNaryQueue


class DataValidator(ABC):
    def __init__(self, queue_size: int, num_symbols: int) -> None:
        self.transmitted_data = FixedSizeNaryQueue(
            base=num_symbols, max_size=queue_size
        )
        self.received_data = FixedSizeNaryQueue(base=num_symbols, max_size=queue_size)

    def add_received_data(self, received_symbol: int) -> None:
        self.received_data.enqueue(received_symbol)

    def add_trasnmitted_data(self, transmitted_symbol: int) -> None:
        self.transmitted_data.enqueue(transmitted_symbol)

    @abstractmethod
    def _validate(self) -> ValidationStatus:
        raise NotImplementedError

    def validate(self, received_symbol: int) -> ValidationStatus:
        self.add_received_data(received_symbol)
        result = self._validate()
        return result
