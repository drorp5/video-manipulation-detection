from active_manipulation_detectors.validation_status import ValidationStatus
from active_manipulation_detectors.side_channel.validation import DataValidator


class DataValidatorKSymbols(DataValidator):
    def __init__(self, num_symbols: int, symbols_for_detection: int) -> None:
        super().__init__(num_symbols=num_symbols, queue_size=symbols_for_detection)
        self.symbols_for_detection = symbols_for_detection

    def _validate(self) -> ValidationStatus:
        if not self.received_data.is_full():
            return ValidationStatus.Incomplete
        if self.received_data == self.transmitted_data:
            return ValidationStatus.Valid
        return ValidationStatus.Invalid
