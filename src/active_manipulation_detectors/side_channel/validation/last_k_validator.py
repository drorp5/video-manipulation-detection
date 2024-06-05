from active_manipulation_detectors import ValidationStatus
from active_manipulation_detectors.side_channel.validation import DataValidator


class DataValidatorKSymbols(DataValidator):
    def __init__(
        self, symbols_for_detection: int, data_holder_type: str, data_unit: int = 1
    ) -> None:
        super().__init__(data_holder_type, data_unit)
        self.symbols_for_detection = symbols_for_detection

    def _validate(self) -> ValidationStatus:
        if len(self.received_data) < self.symbols_for_detection:
            return ValidationStatus.Incomplete

        if self.received_data == self.transmitted_data[: self.symbols_for_detection]:
            return ValidationStatus.Valid

        return ValidationStatus.Invalid

    def clean(self, result: ValidationStatus) -> None:
        if result == ValidationStatus.Incomplete:
            return
        self.transmitted_data = self.transmitted_data[1:]
        self.received_data = self.received_data[1:]
