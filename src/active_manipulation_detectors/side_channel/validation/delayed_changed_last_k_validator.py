from active_manipulation_detectors import ValidationStatus
from active_manipulation_detectors.side_channel.validation import (
    DataValidatorKSymbolsDelayed,
)


class DataValidatorKSymbolsDelayedChanged(DataValidatorKSymbolsDelayed):
    def _validate(self) -> ValidationStatus:
        if len(self.transmitted_data) < self.search_window:
            return ValidationStatus.Incomplete

        for delayed_transmitted_pattern in self.transmitted_data.get_combinations(
            self.symbols_for_detection
        ):
            if delayed_transmitted_pattern == self.received_data:
                return ValidationStatus.Valid
        self.detected_delay = None
        return ValidationStatus.Invalid

    def clean(self, result: ValidationStatus) -> None:
        if len(self.received_data) == self.symbols_for_detection:
            self.received_data = self.received_data[1:]
        if result == ValidationStatus.Incomplete:
            return
        self.transmitted_data = self.transmitted_data[1:]
