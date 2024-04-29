from abc import ABC, abstractmethod
from bitarray import bitarray

from active_manipulation_detectors import ValidationStatus


class DataValidator(ABC):
    def __init__(self) -> None:
        self.transmitted_data = bitarray()
        self.received_data = bitarray()

    def add_received_data(self, received_bits: bitarray) -> None:
        self.received_data.extend(received_bits)

    def add_trasnmitted_data(self, transmitted_bits: bitarray):
        self.transmitted_data.extend(transmitted_bits)

    @abstractmethod
    def _validate(self) -> ValidationStatus:
        raise NotImplementedError

    @abstractmethod
    def clean(self, result: ValidationStatus) -> None:
        raise NotImplementedError

    def validate(self, received_bits: bitarray) -> ValidationStatus:
        self.add_received_data(received_bits)
        result = self._validate()
        self.clean(result)
        return result
