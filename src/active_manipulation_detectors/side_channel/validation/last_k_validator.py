from bitarray import bitarray

from active_manipulation_detectors import ValidationStatus
from active_manipulation_detectors.side_channel.validation import DataValidator


class DataValidatorKSymbols(DataValidator):
    def __init__(self, bits_in_symbol: int, symbols_for_detection: int) -> None:
        super().__init__()
        self.bits_in_symbol= bits_in_symbol
        self.symbols_for_detection = symbols_for_detection
        self.bits_for_detection = symbols_for_detection * bits_in_symbol

    def _validate(self) -> ValidationStatus:
        if len(self.received_data) < self.bits_for_detection:
            return ValidationStatus.Incomplete
        
        if self.received_data == self.transmitted_data[-self.bits_for_detection:]
            return ValidationStatus.Valid
        
        return ValidationStatus.Invalid

    def clean(self, result: ValidationStatus) -> None:
        if result == ValidationStatus.Incomplete:
            return
        self.transmitted_data = bitarray()
        self.received_data = bitarray()
