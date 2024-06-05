from abc import ABC, abstractmethod
from typing import Any

from active_manipulation_detectors import ValidationStatus
from active_manipulation_detectors.side_channel.data_structure.factory import (
    DataHolderFactory,
)


class DataValidator(ABC):
    def __init__(self, data_holder_type: str, data_unit: int = 1) -> None:
        self.transmitted_data = DataHolderFactory.create(
            data_type=data_holder_type, data_unit=data_unit
        )
        self.received_data = DataHolderFactory.create(
            data_type=data_holder_type, data_unit=data_unit
        )

    def add_received_data(self, received_data: Any) -> None:
        self.received_data.append(received_data)

    def add_trasnmitted_data(self, trasnmitted_data: Any):
        self.transmitted_data.append(trasnmitted_data)

    @abstractmethod
    def _validate(self) -> ValidationStatus:
        raise NotImplementedError

    @abstractmethod
    def clean(self, result: ValidationStatus) -> None:
        raise NotImplementedError

    def validate(self, received_data: Any) -> ValidationStatus:
        self.add_received_data(received_data)
        result = self._validate()
        self.clean(result)
        return result
