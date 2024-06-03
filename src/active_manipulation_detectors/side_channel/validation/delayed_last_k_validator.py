from active_manipulation_detectors import ValidationStatus
from active_manipulation_detectors.side_channel.validation import DataValidatorKSymbols


class DataValidatorKSymbolsDelayed(DataValidatorKSymbols):
    def __init__(
        self, bits_in_symbol: int, symbols_for_detection: int, max_delay: int = 0
    ) -> None:
        super().__init__(
            bits_in_symbol=bits_in_symbol, symbols_for_detection=symbols_for_detection
        )
        self.max_delay = max_delay
        self.search_window = (max_delay + symbols_for_detection) * bits_in_symbol
        self.detected_delay = None

    def _validate(self) -> ValidationStatus:
        if len(self.received_data) < self.search_window:
            return ValidationStatus.Incomplete

        target = self.transmitted_data[: self.bits_for_detection]
        for delay in range(self.max_delay + 1):
            delayed_received_pattern = self.received_data[
                delay * self.bits_in_symbol : delay * self.bits_in_symbol
                + self.bits_for_detection
            ]
            if delayed_received_pattern == target:
                self.detected_delay = delay
                return ValidationStatus.Valid
        self.detected_delay = None
        return ValidationStatus.Invalid

    def clean(self, result: ValidationStatus) -> None:
        if result == ValidationStatus.Incomplete:
            return
        self.transmitted_data = self.transmitted_data[self.bits_in_symbol :]

        if self.detected_delay is None:
            self.received_data = self.received_data[self.bits_in_symbol :]
        else:
            self.received_data = self.received_data[
                self.detected_delay * self.bits_in_symbol :
            ]
