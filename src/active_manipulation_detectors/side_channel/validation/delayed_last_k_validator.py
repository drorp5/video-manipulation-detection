from active_manipulation_detectors import ValidationStatus
from active_manipulation_detectors.side_channel.validation import DataValidatorKSymbols


class DataValidatorKSymbolsDelayed(DataValidatorKSymbols):
    def __init__(
        self,
        symbols_for_detection: int,
        data_holder_type: str,
        data_unit: int = 1,
        max_delay: int = 0,
    ) -> None:
        super().__init__(symbols_for_detection, data_holder_type, data_unit)
        self.max_delay = max_delay
        self.search_window = max_delay + self.symbols_for_detection
        self.detected_delay = None

    def _validate(self) -> ValidationStatus:
        if len(self.transmitted_data) < self.search_window:
            return ValidationStatus.Incomplete

        for delay in range(self.max_delay, -1, -1):
            offset = len(self.transmitted_data) - self.symbols_for_detection - delay
            delayed_trasnmitted_pattern = self.transmitted_data[
                offset : offset + self.symbols_for_detection
            ]

            if delayed_trasnmitted_pattern == self.received_data:
                self.detected_delay = delay
                return ValidationStatus.Valid
        self.detected_delay = None
        return ValidationStatus.Invalid

    def clean(self, result: ValidationStatus) -> None:
        if len(self.received_data) == self.symbols_for_detection:
            self.received_data = self.received_data[1:]
        if result == ValidationStatus.Incomplete:
            return
        if self.detected_delay is None:
            self.transmitted_data = self.transmitted_data[1:]
        else:
            offset = (
                len(self.transmitted_data)
                - self.symbols_for_detection
                - self.detected_delay
            )
            self.transmitted_data = self.transmitted_data[offset + 1 :]
